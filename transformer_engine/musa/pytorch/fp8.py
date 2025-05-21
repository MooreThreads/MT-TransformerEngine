from pydantic.dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import os

import torch, torch_musa

import transformer_engine_torch as tex
from transformer_engine.common.recipe import (
    Format,
    Recipe,
)
from transformer_engine.pytorch.fp8 import (
    RecipeState,
    get_fp8_te_dtype,
    FP8GlobalStateManager,
    DelayedScaling
)
from .tensor.mtfp8_tensor import (
    MTFP8Quantizer,
    MTFP8Tensor
)
from transformer_engine.pytorch.utils import get_device_compute_capability

from .utils import add_attr, wrap_attr, replace_attr


@dataclass()
class MTFP8BlockScaling(Recipe):
    margin: int = 0
    fp8_format: Format = Format.HYBRID
    fp8_dpa: bool = False
    fp8_mha: bool = False

    tile_size: int = 128

    def __post_init__(self) -> None:
        assert self.fp8_format != Format.E5M2, "Pure E5M2 training is not supported."
        assert self.tile_size == 128, "Only supports 128 tile_size yet."

    def __repr__(self) -> str:
        return (
            f"margin={self.margin}, "
            f"format={str(self.fp8_format).split('.')[1]}, "
            f"tile_size={self.tile_size}, "
            f"fp8_dpa={self.fp8_dpa}, "
            f"fp8_mha={self.fp8_mha}"
        )


def musa_recipe_mtfp8(self):
    return isinstance(self, MTFP8BlockScaling)


def common_recipe___init___workaround():
    from transformer_engine.common import recipe
    add_attr(recipe, "MTFP8BlockScaling", MTFP8BlockScaling)
    add_attr(recipe.Recipe, "mtfp8", musa_recipe_mtfp8)
    replace_attr(recipe, "MXFP8BlockScaling", MTFP8BlockScaling)
common_recipe___init___workaround()

def replace_mtfp8_tensor():
    from transformer_engine.pytorch.tensor import mxfp8_tensor
    replace_attr(mxfp8_tensor, "MXFP8Tensor", MTFP8Tensor)
replace_mtfp8_tensor()

class MTFP8BlockScalingRecipeState(RecipeState):
    recipe: MTFP8BlockScaling
    mode: str
    dtype: tex.DType

    def __init__(
        self,
        recipe: MTFP8BlockScaling,
        *,
        mode: str,
        num_quantizers: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        self.recipe = recipe
        self.mode = mode
        self.num_quantizers = num_quantizers
        self.dtype = get_fp8_te_dtype(recipe, mode == "forward")

        activation_blocks = {
            "block_m": 1,
            "block_n": self.recipe.tile_size,
        }
        weight_blocks = {
            "block_m": self.recipe.tile_size,
            "block_n": self.recipe.tile_size,
        }

        if mode == "forward":
            assert num_quantizers % 3 == 0
            n_gemms = self.num_quantizers // 3
            self.blocks = [activation_blocks] * n_gemms
            self.blocks += [weight_blocks] * n_gemms
            self.blocks += [activation_blocks] * n_gemms
        else:
            assert num_quantizers % 2 == 0
            n_gemms = self.num_quantizers // 2
            self.blocks = [activation_blocks] * n_gemms
            self.blocks += [activation_blocks] * n_gemms

        if device is None:
            device = torch.device("musa")

    def make_quantizers(self) -> list:
        return [MTFP8Quantizer(
            self.dtype,
            **(self.blocks[i % self.num_quantizers]),
        ) for i in range(self.num_quantizers)]


def musa_recipe_state_create(
    recipe: Recipe,
    *,
    mode: str,
    num_quantizers: int = 1,
    device: Optional[torch.device] = None,
) -> RecipeState:
    if recipe.mtfp8():
        return MTFP8BlockScalingRecipeState(
            recipe,
            mode=mode,
            num_quantizers=num_quantizers,
            device=device,
        )
    return RecipeState._orig_create(
        recipe,
        mode=mode,
        num_quantizers=num_quantizers,
        device=device,
    )


def musa_check_fp8_support() -> Tuple[bool, str]:
    if get_device_compute_capability() >= (3, 1):
        return True, ""
    return False, "Device compute capability 3.1 or higher required for FP8 execution."


@classmethod
def musa_add_fp8_tensors_to_global_buffer(
    cls,
    fp8_meta: Dict[str, Any],
) -> None:
    if fp8_meta["recipe"].mtfp8():
        return
    cls._orig_add_fp8_tensors_to_global_buffer(fp8_meta)


@classmethod
def musa_copy_forward_fp8_meta_tensors_for_recompute(cls, fp8_meta: Dict[str, Any]) -> None:
    if fp8_meta["recipe"].mtfp8():
        return
    cls._orig_copy_forward_fp8_meta_tensors_for_recompute(fp8_meta)


@classmethod
def musa_get_old_fp8_meta_tensors_for_recompute(cls, fp8_meta: Dict[str, Any], quantizers=None) -> None:
    if fp8_meta["recipe"].mtfp8():
        return
    # [Previous Version HACK - Preserved for historical context]
    #HACK(huang.huang): not call _orig_get_old_fp8_meta_tensors_for_recompute directly while needs
    #to modify the ori implement of get_old_fp8_meta_tensors_for_recompute;
    #add .clone() when save meta into updated*, otherwise updated tensor will change along with meta and cause precision issue
    #
    # [New Optimization HACK - Pointer Swap for D2D Overhead]
    #Replace clone()/copy() with pointer swapping to avoid D2D transfers (~100μs each).
    #
    # [New Fix HACK - change scale in quantizer instead of fp8_meta]
    #Set the stash scale to the quantizer, as the scale used in the cast is actually the scale saved in quantizer, not fp8_meta
    #On the other hand, since scale in quantizer and fp8_meta are not same ptr after pointer swapping, it's not necessary to save fp8_meta.
    #Since we only update scale with amax once forward_step or backward_step finished, it's okay to temporarily decouple fp8_meta and quantizer    
    if not int(os.getenv("USE_RECOMPUTE_VARIANCE", 0)):
        cls._orig_get_old_fp8_meta_tensors_for_recompute(fp8_meta)
    else:
        # below is revised vesrion of ori get_old_fp8_meta_tensors_for_recompute

        # Retrieve stashed amaxes and scales from phase 1 pre forward.
        buffer_position_key = "global_fp8_buffer_pos_fwd_recompute"
        stashed_fp8_meta = cls.fp8_tensors_recompute_buffer[fp8_meta[buffer_position_key]].popleft()

        # Replace amaxes and scales with stashed values for phase 2 forward
        for i, quantizer in enumerate(quantizers["scaling_fwd"]):
            quantizer.amax_history = stashed_fp8_meta[0][0][i]
            quantizer.scale = stashed_fp8_meta[1][i]
    #HACK(huang.huang)

def musa_restore_fp8_meta_tensors(fp8_meta: Dict[str, Any], quantizers=None) -> None:
    if fp8_meta["recipe"].mtfp8():
        return
    #HACK(huang.huang): Replace clone()/copy() with pointer swapping to avoid D2D transfers (~100μs each),
    # worked with musa_get_old_fp8_meta_tensors_for_recompute
    # [New Fix HACK - change scale in quantizer instead of fp8_meta]
    # restore scale in quantizer from fp8_meta 
    if not int(os.getenv("USE_RECOMPUTE_VARIANCE", 0)):
        FP8GlobalStateManager._orig_restore_fp8_meta_tensors(fp8_meta)
    else:
        # below is revised vesrion of ori restore_fp8_meta_tensors
        for i, quantizer in enumerate(quantizers["scaling_fwd"]):
            quantizer.amax_history = fp8_meta["scaling_fwd"].amax_history[0][i]
            quantizer.scale = fp8_meta["scaling_fwd"].scale[i]
    ##HACK(huang.huang)

def musa_get_default_fp8_recipe() -> Recipe:
    """FP8 recipe with default args."""
    if os.getenv("FP8_PER_BLOCK", False):
        return MTFP8BlockScaling()
    return DelayedScaling()

#HACK(huang.huang): add flag `skip` to change the behavior of reduce which is not necessary in recompute
# TE will reduce amx history once exit a recompute context in forward and backward, we move them to the end of forward and backward,
# the corresponding call is in megatron/core/pipeline_parallel/schedules.py
@classmethod
def musa_reduce_and_update_fp8_tensors(
    cls,
    forward: bool = True,
    skip: bool = True,
) -> None:
    if skip:
        return
    cls._orig_reduce_and_update_fp8_tensors(forward)
#HACK(huang.huang)

def pytorch_fp8_workaround():
    from transformer_engine.pytorch import fp8
    add_attr(fp8, "MTFP8BlockScalingRecipeState", MTFP8BlockScalingRecipeState)
    wrap_attr(fp8.RecipeState, "create", musa_recipe_state_create)
    replace_attr(fp8, "check_fp8_support", musa_check_fp8_support)
    wrap_attr(
        fp8.FP8GlobalStateManager,
        "add_fp8_tensors_to_global_buffer",
        musa_add_fp8_tensors_to_global_buffer,
    )
    wrap_attr(
        fp8.FP8GlobalStateManager,
        "copy_forward_fp8_meta_tensors_for_recompute",
        musa_copy_forward_fp8_meta_tensors_for_recompute,
    )
    wrap_attr(
        fp8.FP8GlobalStateManager,
        "get_old_fp8_meta_tensors_for_recompute",
        musa_get_old_fp8_meta_tensors_for_recompute,
    )
    wrap_attr(
        fp8.FP8GlobalStateManager,
        "restore_fp8_meta_tensors",
        musa_restore_fp8_meta_tensors,
    )
    if int(os.getenv("USE_RECOMPUTE_VARIANCE", 0)):
        wrap_attr(
            fp8.FP8GlobalStateManager,
            "reduce_and_update_fp8_tensors",
            musa_reduce_and_update_fp8_tensors,
        )
    replace_attr(fp8, "get_default_fp8_recipe", musa_get_default_fp8_recipe)


pytorch_fp8_workaround()
