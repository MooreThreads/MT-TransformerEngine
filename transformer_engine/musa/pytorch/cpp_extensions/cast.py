import torch

from transformer_engine.pytorch.tensor.float8_tensor import (
    Float8Tensor,
)


def cast_to_fp8(src: torch.Tensor, out: Float8Tensor):
    assert isinstance(src, torch.Tensor)
    assert isinstance(out, Float8Tensor), "Only supports Float8Tensor now."
    out.quantize_(src, noop_flag=None)


def weak_support_fp8_cast():
    import transformer_engine.pytorch.cpp_extensions as m
    setattr(m, "cast_to_fp8", cast_to_fp8)


weak_support_fp8_cast()
