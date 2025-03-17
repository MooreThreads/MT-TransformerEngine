from pydantic.dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

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
)
from .tensor.mtfp8_tensor import (
    MTFP8Quantizer,
)
from transformer_engine.pytorch.utils import get_device_compute_capability

from .utils import add_attr, wrap_attr, replace_attr


@dataclass()
class MTFP8BlockScaling(Recipe):
    margin: int = 0
    fp8_format: Format = Format.HYBRID
    fp8_dpa: bool = False
    fp8_mha: bool = False

    activation_block_m: int = 1
    activation_block_n: int = 128
    weight_block_m: int = 128
    weight_block_n: int = 128

    def __post_init__(self) -> None:
        assert self.fp8_format != Format.E5M2, "Pure E5M2 training is not supported."

    def __repr__(self) -> str:
        return (
            f"margin={self.margin}, "
            f"format={str(self.fp8_format).split('.')[1]}, "
            f"activation_block_m={self.activation_block_m}, "
            f"activation_block_n={self.activation_block_n}, "
            f"weight_block_m={self.weight_block_m}, "
            f"weight_block_n={self.weight_block_n}, "
            f"fp8_dpa={self.fp8_dpa}, "
            f"fp8_mha={self.fp8_mha}"
        )


def musa_recipe_mtfp8(self):
    return isinstance(self, MTFP8BlockScaling)


def common_recipe___init___workaround():
    from transformer_engine.common import recipe
    add_attr(recipe, "MTFP8BlockScaling", MTFP8BlockScaling)
    add_attr(recipe.Recipe, "mtfp8", musa_recipe_mtfp8)
common_recipe___init___workaround()


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
            "block_m": self.recipe.activation_block_m,
            "block_n": self.recipe.activation_block_n,
        }
        weight_blocks = {
            "block_m": self.recipe.weight_block_m,
            "block_n": self.recipe.weight_block_n,
        }

        if mode == "forward":
            assert num_quantizers % 3 == 0
            self.blocks = [activation_blocks, weight_blocks, activation_blocks]
        else:
            assert num_quantizers % 2 == 0
            self.blocks = [activation_blocks] * 2

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
def musa_get_old_fp8_meta_tensors_for_recompute(cls, fp8_meta: Dict[str, Any]) -> None:
    if fp8_meta["recipe"].mtfp8():
        return
    cls._orig_get_old_fp8_meta_tensors_for_recompute(fp8_meta)


def musa_restore_fp8_meta_tensors(fp8_meta: Dict[str, Any]) -> None:
    if fp8_meta["recipe"].mtfp8():
        return
    FP8GlobalStateManager._orig_restore_fp8_meta_tensors(fp8_meta)


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
pytorch_fp8_workaround()
