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
        assert self.block_m == 1 or (self.block_m == self.block_n)

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
    
    def get_scale_shape(self, shape: Iterable[int], columnwise: bool) -> Tuple[int, int]:
        """Calculate the shape of the scaling tensor for blockwise quantization.

        This method determines the shape of the scaling tensor needed for blockwise quantization,
        taking into account the input tensor shape and whether columnwise scaling is used.
        The scales are padded to multiples of 4 on the inner dimension for compatibility with GEMM.

        Parameters

        """
        M, K = 1, 1
        for i in range(len(shape) - 1):
            M *= shape[i]
        if len(shape) > 0:
            K = shape[-1]

        def ceil_div(a, b):
            return (a + b - 1) // b
        
        outer = ceil_div(M, self.block_m)
        inner = ceil_div(K, self.block_n)

        return (outer, inner)

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

        def ceil_div(a, b):
            return (a + b - 1) // b

        data = torch.empty(shape, dtype=torch.uint8, device=device)
        scale_inv = torch.empty(
            ceil_div(math.prod(shape[:-1]), self.block_m),
            ceil_div(shape[-1], self.block_n),
            dtype=torch.float,
            device=device,
        )

        columnwise_data = None
        columnwise_scale_inv = None
        if self.columnwise_usage and self.block_m != self.block_n:
            columnwise_data = torch.empty_like(data)
            columnwise_scale_inv = torch.empty(
                ceil_div(math.prod(shape[:-1]), self.block_n),
                ceil_div(shape[-1], self.block_m),
                dtype=torch.float,
                device=device,
            )

        return MTFP8Tensor(
            shape=shape,
            dtype=dtype,
            rowwise_data=data,
            rowwise_scale_inv=scale_inv,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale_inv,
            fp8_dtype=self.dtype,
            quantizer=self,
            requires_grad=requires_grad,
        )

    def calibrate(self, tensor: torch.Tensor) -> None:
        pass

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"rowwise_usage={self.rowwise_usage}, "
            f"columnwise_usage={self.columnwise_usage}, "
            f"internal={self.internal}, "
            f"block_m={self.block_m}, "
            f"block_n={self.block_n}, "
            f"dtype={self.dtype}, "
            ")"
        )


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
        assert self._quantizer is not None
        return self._quantizer
        # if self._quantizer is not None:
        #     return self._quantizer

        # rowwise_data_shape = self._rowwise_data.shape
        # rowwise_scale_inv_shape = self._rowwise_scale_inv.shape
        # assert len(rowwise_data_shape) == 2
        # assert len(rowwise_scale_inv_shape) == 2

        # m, n = rowwise_data_shape[0], rowwise_data_shape[1]
        # sinv_m, sinv_n = rowwise_scale_inv_shape[0], rowwise_scale_inv_shape[1]

        # def next_power_of_2(x):
        #     assert x >= 1
        #     return 2 ** math.ceil(math.log2(x))

        # if m == 1 or m == sinv_m:
        #     block_m = 1
        # else:
        #     block_m = next_power_of_2(m // sinv_m)
        # block_n = next_power_of_2(n // sinv_n)

        # return MTFP8Quantizer(
        #     fp8_dtype=self._fp8_dtype,
        #     block_m=block_m,
        #     block_n=block_n,
        # )

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
                and self._columnwise_data is not None
                and self._columnwise_scale_inv is not None
            ), "Cannot update to rowwise and columnwise usage."
            return

        if rowwise_usage:
            assert (
                self._rowwise_data is not None and self._rowwise_scale_inv is not None
            ), "Cannot update to rowwise usage."
            self._columnwise_data = None
            self._columnwise_scale_inv = None
            return

        assert (
            self._columnwise_data is not None and self._columnwise_scale_inv is not None
        ), "Cannot update to columnwise usage."
        self._rowwise_data = None
        self._rowwise_scale_inv = None
        return

    def clone(self) -> MTFP8Tensor:
        rowwise_data = None
        if self._rowwise_data is not None:
            rowwise_data = self._rowwise_data.detach().clone()
        columnwise_data = None
        if self._columnwise_data is not None:
            columnwise_data = self._columnwise_data.detach().clone()
        return _IdentityFunc.apply(
            self,
            {
                "rowwise_data": rowwise_data,
                "columnwise_data": columnwise_data,
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
        if (
            self._rowwise_data is not None
            and self._rowwise_data.is_contiguous(memory_format=memory_format)
            and (
                (self._columnwise_data is None)
                or (self._columnwise_data.is_contiguous(memory_format=memory_format))
            )
        ):
            return self
        raise ValueError("MTFP8Tensor does not support different memory formats!")

    def clear(self):
        self._rowwise_data = torch.Tensor() if self._rowwise_data is not None else None
        self._columnwise_data = torch.Tensor() if self._columnwise_data is not None else None

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):

        if func == aten.view.default:
            tensor = args[0]
            data = tensor._rowwise_data
            orig_size = data.size()
            out_data = data.__torch_dispatch__(
                func,
                types,
                [data] + list(args[1:]),
                kwargs,
            )
            if orig_size != out_data.size():
                raise NotImplementedError(
                    "Changing shape with view not implemented "
                    " (scales and columnwise data untouched)."
                )
            return MTFP8Tensor.make_like(tensor)

        return super().__torch_dispatch__(func, types, args, kwargs)

    @classmethod
    def _make_in_reduce_ex(
        cls,
        shape: torch.Size,
        rowwise_data: torch.Tensor,
        rowwise_scale_inv: torch.Tensor,
        columnwise_data: torch.Tensor,
        columnwise_scale_inv: torch.Tensor,
        fp8_dtype: tex.DType,
        dtype: torch.dtype,
        quantizer: Quantizer,
    ) -> MTFP8Tensor:
        return MTFP8Tensor(
            shape=shape,
            dtype=dtype,
            rowwise_data=rowwise_data,
            rowwise_scale_inv=rowwise_scale_inv,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale_inv,
            fp8_dtype=fp8_dtype,
            quantizer=quantizer,
        )

    def __reduce_ex__(self, protocol: int) -> tuple:
        return (
            MTFP8Tensor._make_in_reduce_ex,
            (
                self.shape,
                self._rowwise_data,
                self._rowwise_scale_inv,
                self._columnwise_data,
                self._columnwise_scale_inv,
                self._fp8_dtype,
                self.dtype,
                self._quantizer,
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
            self._columnwise_data = tensor._columnwise_data
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
            d_inferred = -math.prod(tensor.shape) // math.prod(shape)
            for i, d in enumerate(shape):
                if d == -1:
                    shape[i] = d_inferred
                    break

        new_rowwise_data = None
        new_columnwise_data = None
        if tensor._rowwise_data is not None:
            new_rowwise_data = tensor._rowwise_data.view(*shape)
        if tensor._columnwise_data is not None:
            new_columnwise_data = tensor._columnwise_data.view(*shape)
        return MTFP8Tensor(
            shape,
            tensor.dtype,
            rowwise_data=new_rowwise_data,
            rowwise_scale_inv=tensor._rowwise_scale_inv,
            columnwise_data=new_columnwise_data,
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
            new_columnwise_data = (
                grad._columnwise_data.view(*ctx.shape) if grad._columnwise_data is not None else None
            )
            dgrad = MTFP8Tensor(
                ctx.shape,
                grad.dtype,
                rowwise_data=new_data,
                rowwise_scale_inv=grad._rowwise_scale_inv,
                columnwise_data=new_columnwise_data,
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
            d_inferred = -math.prod(tensor.shape) // math.prod(shape)
            for i, d in enumerate(shape):
                if d == -1:
                    shape[i] = d_inferred
                    break

        new_rowwise_data = None
        new_columnwise_data = None
        if tensor._rowwise_data is not None:
            new_rowwise_data = tensor._rowwise_data.reshape(*shape)
        if tensor._columnwise_data is not None:
            new_columnwise_data = tensor._columnwise_data.reshape(*shape)

        return MTFP8Tensor(
            shape,
            tensor.dtype,
            rowwise_data=new_rowwise_data,
            rowwise_scale_inv=tensor._rowwise_scale_inv,
            columnwise_data=new_columnwise_data,
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
            new_columnwise_data = (
                grad._columnwise_data.view(*ctx.shape) if grad._columnwise_data is not None else None
            )
            dgrad = MTFP8Tensor(
                ctx.shape,
                grad.dtype,
                rowwise_data=new_data,
                rowwise_scale_inv=grad._rowwise_scale_inv,
                columnwise_data=new_columnwise_data,
                columnwise_scale_inv=grad._columnwise_scale_inv,
                fp8_dtype=grad._fp8_dtype,
                quantizer=grad._quantizer,
            )
            return dgrad, None
        return grad.view(ctx.shape), None
