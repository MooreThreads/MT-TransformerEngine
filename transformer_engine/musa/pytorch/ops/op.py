from typing import Optional

import torch

from transformer_engine.pytorch.fp8 import (
    FP8GlobalStateManager,
    Recipe,
    DelayedScalingRecipeState,
    MXFP8BlockScalingRecipeState,
)
from ..fp8 import MTFP8BlockScalingRecipeState

from ..utils import replace_attr

def musa__update_quantization_recipe_state(
    self,
    *,
    recipe: Optional[Recipe] = None,
) -> None:
    # Quantization recipe
    if recipe is None:
        recipe = FP8GlobalStateManager.get_fp8_recipe()

    # Reset quantization state if needed
    if self._fp8_metas is None or self._quantizers is None:
        self._reset_quantization_recipe_state(recipe=recipe)
        return
    for mode in ("forward", "backward"):
        fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(
            forward=(mode == "forward"),
        )
        if self._fp8_metas[mode] is None or fp8_meta_key not in self._fp8_metas[mode]:
            continue
        recipe_state = self._fp8_metas[mode][fp8_meta_key]
        need_to_reset_recipe_state = (
            recipe.delayed() and not isinstance(recipe_state, DelayedScalingRecipeState)
        ) or (recipe.mxfp8() and not isinstance(recipe_state, MXFP8BlockScalingRecipeState)
        ) or (recipe.mtfp8() and not isinstance(recipe_state, MTFP8BlockScalingRecipeState))
        if need_to_reset_recipe_state:
            self._reset_quantization_recipe_state(recipe=recipe)
            return

    # Quantization recipe state for forward and backward pass
    for mode in ("forward", "backward"):
        num_quantizers = self.num_quantizers(mode)
        if num_quantizers == 0:
            continue

        # Update FP8 metadata
        fp8_meta = self._fp8_metas[mode]
        fp8_meta["recipe"] = recipe
        fp8_meta["fp8_group"] = FP8GlobalStateManager.get_fp8_group()

        # Get recipe state
        fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(
            forward=(mode == "forward"),
        )
        recipe_state = fp8_meta[fp8_meta_key]

        # Reallocate amax history if needed
        if recipe.mxfp8() or recipe.mtfp8():
            continue

        current_length = recipe_state.amax_history.size(0)
        target_length = recipe.amax_history_len
        if current_length != target_length:
            with torch.no_grad():
                if target_length < current_length:
                    recipe_state.amax_history = recipe_state.amax_history[
                        :target_length
                    ].clone()
                else:
                    recipe_state.amax_history = torch.nn.functional.pad(
                        recipe_state.amax_history,
                        pad=(0, 0, 0, target_length - current_length),
                    )
            self._quantizers[mode] = recipe_state.make_quantizers()


def pytorch_ops_op_workaround():
    from transformer_engine.pytorch.ops.op import BasicOperation
    replace_attr(
        BasicOperation,
        "_update_quantization_recipe_state",
        musa__update_quantization_recipe_state,
    )
pytorch_ops_op_workaround()
