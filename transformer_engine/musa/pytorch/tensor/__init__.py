from ..utils import add_attr


def pytorch_mtfp8_tensor_workaround():
    from transformer_engine.pytorch import tensor
    from . import mtfp8_tensor, mtfp8_tensor_base
    add_attr(tensor, "mtfp8_tensor", mtfp8_tensor)
    add_attr(tensor._internal, "mtfp8_tensor_base", mtfp8_tensor_base)
pytorch_mtfp8_tensor_workaround()
