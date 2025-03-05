"""Tensor class with FP8 data and FP32 scales"""
from __future__ import annotations
import math
from typing import List, Optional, Iterable, Tuple

import torch, torch_musa
import transformer_engine_torch as tex

from transformer_engine.pytorch.tensor.quantized_tensor import (
    _IdentityFunc,
)
from transformer_engine.pytorch.tensor import (
    Quantizer,
    QuantizedTensor,
)
from transformer_engine.pytorch.utils import (
    devices_match,
    round_up_to_nearest_multiple,
)

from .mtfp8_tensor_base import (
    MTFP8TensorBase,
    _FromMTFP8Func,
)

aten = torch.ops.aten

class MTFP8Quantizer(Quantizer):
    dtype: tex.DType
    block_m: int
    block_n: int

    def __init__(
        self,
        fp8_dtype: tex.DType,
        block_m: int,
        block_n: int,
        *,
        rowwise: bool = True,
        columnwise: bool = True,
    ) -> None:
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.dtype = fp8_dtype
        self.block_m = block_m
        self.block_n = block_n

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: QuantizedTensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> QuantizedTensor:

        assert isinstance(dst, MTFP8Tensor), f"Cannot store quantized MTFP8 in {type(dst)} type."

        if not devices_match(src.device, dst.device):
            src = src.to(device=dst.device)
        if not src.is_contiguous():
            src = src.contiguous()

        tex.quantize(src, self, dst, noop_flag)

        dst._fp8_dtype = self.dtype

        return dst

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> MTFP8Tensor:

        if device is None:
            device = torch.device("musa")

        assert (
            shape[-1] % self.block_n == 0
            and math.prod(shape[:-1]) % self.block_m == 0
        ), (
            f"Incorrect shape {shape} for MTFP8. Tensor dims must be divisible by"
            f" [{self.block_m}, {self.block_n}]"
        )

        data = torch.empty(shape, dtype=torch.uint8, device=device)
        scale_inv = torch.zeros(
            round_up_to_nearest_multiple(math.prod(shape[:-1]), self.block_m),
            round_up_to_nearest_multiple(shape[-1], self.block_n),
            dtype=torch.uint8,
            device=device,
        )

        columnwise_scale_inv = None
        if self.columnwise_usage:
            columnwise_scale_inv = torch.zeros(
                scale_inv.shape[::-1],
                dtype=torch.uint8,
                device=device,
            )

        return MTFP8Tensor(
            shape=shape,
            dtype=dtype,
            rowwise_data=data,
            rowwise_scale_inv=scale_inv,
            columnwise_scale_inv=columnwise_scale_inv,
            fp8_dtype=self.dtype,
            quantizer=self,
            requires_grad=requires_grad,
        )

    def calibrate(self, tensor: torch.Tensor) -> None:
        pass


class MTFP8Tensor(MTFP8TensorBase, QuantizedTensor):
    def __repr__(self, *, tensor_contents=None):
        return f"MTFP8Tensor(fp8_dtype={self._fp8_dtype}, data={self.dequantize(dtype=self.dtype)})"

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if dtype is None:
            dtype = self.dtype

        if torch.is_grad_enabled():
            return _FromMTFP8Func.apply(self, dtype)
        return _FromMTFP8Func.forward(None, self, dtype)

    def _get_quantizer(self) -> Quantizer:
        if self._quantizer is not None:
            return self._quantizer

        assert self._rowwise_data is not None
        assert self._rowwise_scale_inv is not None
        rowwise_data_shape = self._rowwise_data.shape
        rowwise_scale_inv_shape = self._rowwise_scale_inv.shape
        assert len(rowwise_data_shape) == 2
        assert len(rowwise_scale_inv_shape) == 2
        assert rowwise_data_shape[0] % rowwise_scale_inv_shape[0] == 0
        assert rowwise_data_shape[1] % rowwise_scale_inv_shape[1] == 0

        return MTFP8Quantizer(
            fp8_dtype=self._fp8_dtype,
            block_m=(rowwise_data_shape[0] // rowwise_scale_inv_shape[0]),
            block_n=(rowwise_data_shape[1] // rowwise_scale_inv_shape[1]),
        )

    def quantize_(
        self,
        tensor: torch.Tensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> MTFP8Tensor:
        if isinstance(tensor, QuantizedTensor):
            return self.quantize_(tensor.dequantize())
        self._get_quantizer().update_quantized(tensor, self, noop_flag=noop_flag)
        return self

    def detach(self) -> MTFP8Tensor:
        return MTFP8Tensor.make_like(self)

    def update_usage(self, rowwise_usage=True, columnwise_usage=True):
        assert rowwise_usage or columnwise_usage, "Could not disable all usages of the tensor."

        if columnwise_usage and rowwise_usage:
            assert (
                self._rowwise_data is not None
                and self._rowwise_scale_inv is not None
                and self._columnwise_scale_inv is not None
            ), "Cannot update to rowwise and columnwise usage."
            return

        if rowwise_usage:
            assert (
                self._rowwise_data is not None and self._rowwise_scale_inv is not None
            ), "Cannot update to rowwise usage."
            self._columnwise_scale_inv = None
            return

        assert (
            self._rowwise_data is not None and self._columnwise_scale_inv is not None
        ), "Cannot update to columnwise usage."
        self._rowwise_scale_inv = None
        return

    def clone(self) -> MTFP8Tensor:
        assert self._rowwise_data is not None
        rowwise_data = self._rowwise_data.detach().clone()
        return _IdentityFunc.apply(
            self,
            {
                "rowwise_data": rowwise_data,
            },
        )

    def view(self, *shape: Tuple[int]) -> MTFP8Tensor:
        return _ViewFunc.apply(self, shape)

    def reshape(self, *shape: Tuple[int]) -> MTFP8Tensor:
        return _ReshapeFunc.apply(self, shape)

    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> MTFP8Tensor:
        if self._rowwise_data is not None and self._rowwise_data.is_contiguous(
            memory_format=memory_format
        ):
            return self
        raise ValueError("MTFP8Tensor does not support different memory formats!")

    def clear(self):
        self._rowwise_data = torch.Tensor() if self._rowwise_data is not None else None

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):

        if func == aten.view.default:
            tensor = args[0]
            data = tensor._rowwise_data
            out_data = data.__torch_dispatch__(
                func,
                types,
                [data] + list(args[1:]),
                kwargs,
            )
            out_shape = out_data.size()
            return MTFP8Tensor(
                shape=out_shape,
                dtype=tensor.dtype,
                rowwise_data=out_data,
                rowwise_scale_inv=tensor._rowwise_scale_inv,
                columnwise_scale_inv=tensor._columnwise_scale_inv,
                fp8_dtype=tensor._fp8_dtype,
                quantizer=tensor._quantizer,
                requires_grad=False,
            )

        return super().__torch_dispatch__(func, types, args, kwargs)

    @classmethod
    def _make_in_reduce_ex(
        cls,
        rowwise_data: torch.Tensor,
        rowwise_scale_inv: torch.Tensor,
        columnwise_scale_inv: torch.Tensor,
        fp8_dtype: tex.DType,
        dtype: torch.dtype,
    ) -> MTFP8Tensor:
        return MTFP8Tensor(
            dtype=dtype,
            rowwise_data=rowwise_data,
            rowwise_scale_inv=rowwise_scale_inv,
            columnwise_scale_inv=columnwise_scale_inv,
            fp8_dtype=fp8_dtype,
        )

    def __reduce_ex__(self, protocol: int) -> tuple:
        return (
            MTFP8Tensor._make_in_reduce_ex,
            (
                self._rowwise_data,
                self._rowwise_scale_inv,
                self._columnwise_scale_inv,
                self._fp8_dtype,
                self.dtype,
            ),
        )

    def _get_data(self) -> MTFP8Tensor:
        return super().data

    @torch.no_grad()
    def _set_data(self, tensor: torch.Tensor) -> None:
        new_device = tensor.device if tensor.is_musa else self.device

        if isinstance(tensor, MTFP8Tensor):
            if (
                self.size() != tensor.size()
                or self.stride() != tensor.stride()
                or self.storage_offset() != tensor.storage_offset()
                or self.dtype != tensor.dtype
                or self.layout != tensor.layout
                or not devices_match(self.device, new_device)
            ):
                dummy_tensor = torch.Tensor._make_wrapper_subclass(
                    MTFP8Tensor,
                    tensor.size(),
                    strides=tensor.stride(),
                    storage_offset=tensor.storage_offset(),
                    dtype=tensor.dtype,
                    layout=tensor.layout,
                    requires_grad=tensor.requires_grad,
                    device=new_device,
                )
                super(MTFP8Tensor, type(self)).data.__set__(self, dummy_tensor)
            self._rowwise_data = tensor._rowwise_data
            self._quantizer = tensor._quantizer
            self._fp8_dtype = tensor._fp8_dtype
            self._rowwise_scale_inv = tensor._rowwise_scale_inv
            self._columnwise_scale_inv = tensor._columnwise_scale_inv
            return

        assert self._quantizer is not None, "Can't quantize without a quantizer"
        self.data = self._quantizer.quantize(tensor)
        if self.requires_grad != tensor.requires_grad:
            self.requires_grad_(requires_grad=tensor.requires_grad)

    data = property(_get_data, _set_data)


class _ViewFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tensor: MTFP8Tensor,
        shape: Optional[list[int]] = None,
    ) -> MTFP8Tensor:
        ctx.shape = tensor.shape
        if shape is None:
            return tensor

        if not isinstance(shape, Iterable):
            shape = [shape]
        elif len(shape) == 1 and isinstance(shape[0], Iterable):
            shape = shape[0]
        if -1 in shape:
            shape = list(shape)
            d_inferred = -math.prod(ctx.shape) // math.prod(shape)
            for i, d in enumerate(shape):
                if d == -1:
                    shape[i] = d_inferred
                    break
        if shape[-1] != ctx.shape[-1]:
            raise RuntimeError(
                "MTFP8Tensor does not support reshaping inner dimension "
                f"(attempted to reshape dims={tuple(tensor.shape)} to {tuple(shape)})"
            )

        new_rowwise_data = None
        if tensor._rowwise_data is not None:
            new_rowwise_data = tensor._rowwise_data.view(*shape)

        return MTFP8Tensor(
            shape,
            tensor.dtype,
            rowwise_data=new_rowwise_data,
            rowwise_scale_inv=tensor._rowwise_scale_inv,
            columnwise_scale_inv=tensor._columnwise_scale_inv,
            fp8_dtype=tensor._fp8_dtype,
            quantizer=tensor._quantizer,
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        if isinstance(grad, MTFP8Tensor):
            new_data = (
                grad._rowwise_data.view(*ctx.shape) if grad._rowwise_data is not None else None
            )
            dgrad = MTFP8Tensor(
                ctx.shape,
                grad.dtype,
                rowwise_data=new_data,
                rowwise_scale_inv=grad._rowwise_scale_inv,
                columnwise_scale_inv=grad._columnwise_scale_inv,
                fp8_dtype=grad._fp8_dtype,
                quantizer=grad._quantizer,
            )
            return dgrad, None
        return grad.view(ctx.shape), None


class _ReshapeFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tensor: MTFP8Tensor,
        shape: Optional[list[int]] = None,
    ) -> MTFP8Tensor:
        ctx.shape = tensor.shape
        if shape is None:
            return tensor

        if not isinstance(shape, Iterable):
            shape = [shape]
        elif len(shape) == 1 and isinstance(shape[0], Iterable):
            shape = shape[0]
        if -1 in shape:
            shape = list(shape)
            d_inferred = -math.prod(ctx.shape) // math.prod(shape)
            for i, d in enumerate(shape):
                if d == -1:
                    shape[i] = d_inferred
                    break
        if shape[-1] != ctx.shape[-1]:
            raise RuntimeError(
                "MTFP8Tensor does not support reshaping inner dimension "
                f"(attempted to reshape dims={tuple(tensor.shape)} to {tuple(shape)})"
            )

        new_rowwise_data = None
        if tensor._rowwise_data is not None:
            new_rowwise_data = tensor._rowwise_data.reshape(*shape)

        return MTFP8Tensor(
            shape,
            tensor.dtype,
            rowwise_data=new_rowwise_data,
            rowwise_scale_inv=tensor._rowwise_scale_inv,
            columnwise_scale_inv=tensor._columnwise_scale_inv,
            fp8_dtype=tensor._fp8_dtype,
            quantizer=tensor._quantizer,
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        if isinstance(grad, MTFP8Tensor):
            new_rowwise_data = None
            if grad._rowwise_data is not None:
                new_rowwise_data = grad._rowwise_data.view(*ctx.shape)
            dgrad = MTFP8Tensor(
                ctx.shape,
                grad.dtype,
                rowwise_data=new_rowwise_data,
                rowwise_scale_inv=grad._rowwise_scale_inv,
                columnwise_scale_inv=grad._columnwise_scale_inv,
                fp8_dtype=grad._fp8_dtype,
                quantizer=grad._quantizer,
            )
            return dgrad, None
        return grad.view(ctx.shape), None
