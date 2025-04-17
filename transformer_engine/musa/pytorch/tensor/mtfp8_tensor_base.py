from __future__ import annotations
from typing import Optional, Dict, Any, Tuple

import torch

import transformer_engine_torch as tex

from transformer_engine.pytorch.constants import (
    TE_DType as torch_to_transformer_engine_dtype,
)
from transformer_engine.pytorch.tensor import Quantizer


class _FromMTFP8Func(torch.autograd.Function):
    @staticmethod
    def forward(
        _ctx: Optional[torch.autograd.function.FunctionCtx],  # unused
        tensor: MTFP8TensorBase,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        dtype = torch_to_transformer_engine_dtype[dtype]

        if tensor._rowwise_data is not None:
            return tex.dequantize(tensor, dtype)
        raise NotImplementedError("Casting back from the transpose not implemented yet!")

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        return grad, None


class MTFP8TensorBase:
    _rowwise_data: Optional[torch.Tensor]
    _columnwise_data: Optional[torch.Tensor]
    _quantizer: Optional[Quantizer]
    _fp8_dtype: tex.DType
    _rowwise_scale_inv: Optional[torch.Tensor]
    _columnwise_scale_inv: Optional[torch.Tensor]

    def __new__(
        cls,
        *args,
        rowwise_data: torch.Tensor,
        rowwise_scale_inv: torch.Tensor,
        columnwise_data: Optional[torch.Tensor],
        columnwise_scale_inv: Optional[torch.Tensor],
        fp8_dtype: tex.DType,
        quantizer: Optional[Quantizer] = None,
        **kwargs,
    ):
        instance = super().__new__(cls, *args, **kwargs)
        instance._rowwise_data = rowwise_data
        instance._columnwise_data = columnwise_data
        instance._rowwise_scale_inv = rowwise_scale_inv
        instance._columnwise_scale_inv = columnwise_scale_inv
        instance._fp8_dtype = fp8_dtype
        instance._quantizer = quantizer

        return instance

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "rowwise_data": self._rowwise_data,
            "rowwise_scale_inv": self._rowwise_scale_inv,
            "columnwise_data": self._columnwise_data,
            "columnwise_scale_inv": self._columnwise_scale_inv,
            "fp8_dtype": self._fp8_dtype,
            "quantizer": self._quantizer,
        }

    def prepare_for_saving(self) -> Tuple[list[Optional[torch.Tensor]], MTFP8TensorBase]:
        tensors = [self._rowwise_data, self._columnwise_data]
        return tensors, self

    def restore_from_saved(
        self, tensors: list[Optional[torch.Tensor]]
    ) -> list[Optional[torch.Tensor]]:
        self._rowwise_data = tensors[0]
        self._columnwise_data = tensors[1]
        return tensors[2:]

    def get_data_tensors(self):
        return self._rowwise_data, self._columnwise_data

    def dequantize(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return _FromMTFP8Func.forward(None, self, dtype)

    def size(self, *args, **kwargs):
        if self._rowwise_data is not None:
            return self._rowwise_data.size(*args, **kwargs)
        return self._columnwise_data.size(*args, **kwargs)

    def __repr__(self):
        data = self.dequantize()
        if self._rowwise_data is not None:
            descriptor = "rowwise"
            sinv = self._rowwise_scale_inv
        else:
            descriptor = "columnwise"
            sinv = self._columnwise_scale_inv

        return (
            "MTFP8TensorBase("
            f"fp8_dtype={self._fp8_dtype}, "
            f"{descriptor}_scaled_data={data}, "
            f"{descriptor}_scale_inv={sinv}"
            ")"
        )
