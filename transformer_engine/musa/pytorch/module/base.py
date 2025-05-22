import os
from typing import Any, Dict, Generator
from contextlib import contextmanager
import torch
from transformer_engine.pytorch.fp8 import Recipe, FP8GlobalStateManager
from ..fp8 import MTFP8BlockScalingRecipeState
from transformer_engine.pytorch.distributed import (
    is_fp8_activation_recompute_enabled,
    in_fp8_activation_recompute_phase,
)
from ..utils import wrap_attr, replace_attr


def musa_set_meta_tensor(self, fwd: bool, recipe: Recipe) -> None:
    fp8_meta_tensor_key = "scaling_fwd" if fwd else "scaling_bwd"

    if self.fp8_meta_tensors_initialized:
        recipe_state = self.fp8_meta[fp8_meta_tensor_key]
        if recipe.mtfp8() and isinstance(recipe_state, MTFP8BlockScalingRecipeState):
            return

    self._orig_set_meta_tensor(fwd, recipe)

#HACK(huang.huang): support Pointer Swap for D2D Overhead
#just change the args parse to get_old_fp8_meta_tensors_for_recompute and restore_fp8_meta_tensors
@contextmanager
def TransformerEngineBaseModule_prepare_forward(
    self,
    inp: torch.Tensor,
    num_gemms: int = 1,
    allow_non_contiguous: bool = False,
) -> Generator[torch.Tensor, None, None]:
    """Checks and prep for FWD.
    The context manager is needed because there isn't a way for a module to know
    if it's the last FP8 module in the forward autocast. It is useful
    to setup the forward aggregated amax reduction for every module
    just in case. The autocast exit will pick up the most recent one.
    """
    if not int(os.getenv("USE_RECOMPUTE_VARIANCE", 0)):
        with self._orig_prepare_forward(inp, num_gemms, allow_non_contiguous) as processed_inp:
            yield processed_inp
    else:
        # Activation recomputation is used and this is the second forward phase.
        if self.fp8 and in_fp8_activation_recompute_phase():
            FP8GlobalStateManager.get_old_fp8_meta_tensors_for_recompute(self.fp8_meta, self.quantizers)
        else:
            assert inp.is_cuda, "TransformerEngine needs CUDA."

            if self.tp_size > 1:
                assert self.tp_group_initialized, "TP group not initialized."

            self.set_activation_dtype(inp)
            self.init_fp8_metadata(num_gemms=num_gemms)

            if self.fp8 and self.sequence_parallel and self.fp8_meta["recipe"].delayed():
                assert self.fp8_meta["recipe"].reduce_amax, (
                    "Amax reduction across tensor parallel group is "
                    "necessary when using sequence parallelism with FP8."
                )

            if self.fp8 and not FP8GlobalStateManager.fp8_graph_capturing():
                FP8GlobalStateManager.add_fp8_tensors_to_global_buffer(self.fp8_meta)

            # Activation recomputation is used and this is the first forward phase.
            if self.fp8 and self.training and is_fp8_activation_recompute_enabled():
                FP8GlobalStateManager.copy_forward_fp8_meta_tensors_for_recompute(self.fp8_meta)

        with torch.cuda.nvtx.range(self.__class__.__name__ + " forward"):
            if not allow_non_contiguous and not inp.is_contiguous():
                inp = inp.contiguous()
            yield inp

        if self.fp8 and in_fp8_activation_recompute_phase():
            FP8GlobalStateManager.restore_fp8_meta_tensors(self.fp8_meta, self.quantizers)
##HACK(huang.huang)

def pytorch_module_base_workaround():
    from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
    from transformer_engine.pytorch import fp8
    wrap_attr(TransformerEngineBaseModule, "set_meta_tensor", musa_set_meta_tensor)
    replace_attr(TransformerEngineBaseModule, "prepare_forward", TransformerEngineBaseModule_prepare_forward)
pytorch_module_base_workaround()
