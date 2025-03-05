from transformer_engine.pytorch.fp8 import Recipe
from ..fp8 import MTFP8BlockScalingRecipeState

from ..utils import wrap_attr


def musa_set_meta_tensor(self, fwd: bool, recipe: Recipe) -> None:
    fp8_meta_tensor_key = "scaling_fwd" if fwd else "scaling_bwd"

    if self.fp8_meta_tensors_initialized:
        recipe_state = self.fp8_meta[fp8_meta_tensor_key]
        if recipe.mtfp8() and isinstance(recipe_state, MTFP8BlockScalingRecipeState):
            return

    self._orig_set_meta_tensor(fwd, recipe)


def pytorch_module_base_workaround():
    from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
    wrap_attr(TransformerEngineBaseModule, "set_meta_tensor", musa_set_meta_tensor)
pytorch_module_base_workaround()
