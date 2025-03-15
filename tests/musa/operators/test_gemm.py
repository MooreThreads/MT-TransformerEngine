import pytest
import torch, torch_musa

from transformer_engine.pytorch.cpp_extensions.gemm import (
    general_gemm,
)
from transformer_engine.pytorch.tensor.float8_tensor import (
    Float8Tensor,
)
from transformer_engine.musa.pytorch.tensor.mtfp8_tensor import (
    MTFP8Tensor,
)

from test_cast import (
    dev,
    te_dtype_from_th_dtype,
    _gen_mtfp8_groupwise_cast_transpose_golden,
    composite_blockwise_cast,
    composite_blockwise_uncast,
    composite_groupwise_uncast,
)


def get_non_fp8_tol(dtype):
    assert dtype in [torch.bfloat16, torch.float16, torch.float]
    if dtype == torch.bfloat16:
        return dict(atol=1e-2, rtol=1e-2)
    if dtype == torch.float16:
        return dict(atol=1e-3, rtol=1e-3)
    return dict(atol=1e-5, rtol=1.3e-6)


def get_fp8_tol(dtypes):
    if isinstance(dtypes, torch.dtype):
        dtypes = [dtypes]
    if torch.float8_e5m2 in dtypes:
        return dict(atol=1e-2, rtol=0.25)
    return dict(atol=1e-2, rtol=0.125) 


def get_test_workspace():
    return torch.empty(16, dtype=torch.uint8, device=dev)


common_m = [4096, 180]
common_k = [2048, 4096]
common_n = [2560, 8192, 14336]
common_layout = ["TN", "NN", "NT"]


def layout_matmul(weight, input, layout):
    assert layout in common_layout
    if layout == "TN":
        return torch.matmul(input, weight.t())
    if layout == "NN":
        return torch.matmul(input, weight)
    return torch.matmul(input.t(), weight)


@pytest.mark.parametrize("dtype", [
    torch.bfloat16,
])
@pytest.mark.parametrize("M", common_m)
@pytest.mark.parametrize("K", common_k)
@pytest.mark.parametrize("N", common_n)
@pytest.mark.parametrize("layout", common_layout)
def test_non_fp8_gemm(dtype, M, K, N, layout):
    transa = layout[0] == "T"
    transb = layout[1] == "T"

    weight_shape = [N, K] if transa else [K, N]
    weight = torch.rand(weight_shape, dtype=dtype, device=dev)

    input_shape = [K, M] if transb else [M, K]
    input = torch.rand(input_shape, dtype=dtype, device=dev)

    out_gold = layout_matmul(weight, input, layout)

    out_te = torch.empty(M, N, dtype=dtype, device=dev)
    workspace = get_test_workspace()
    general_gemm(
        weight,
        input,
        workspace,
        out_dtype=dtype,
        out=out_te,
        layout=layout,
    )
    torch.testing.assert_close(out_te, out_gold, **get_non_fp8_tol(dtype))


@pytest.mark.parametrize("dtypes", [
    [torch.float8_e4m3fn, torch.float8_e4m3fn, torch.bfloat16],
    [torch.float8_e5m2, torch.float8_e4m3fn, torch.bfloat16],
    [torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16],
    [torch.float8_e4m3fn, torch.float8_e5m2, torch.float],
])
@pytest.mark.parametrize("scales", [
    [3.0, 0.5],
    [2.0, 3.0],
])
@pytest.mark.parametrize("M", common_m)
@pytest.mark.parametrize("K", common_k)
@pytest.mark.parametrize("N", common_n)
@pytest.mark.parametrize("layout", common_layout)
def test_f8_f8_f16_per_tensor_gemm(dtypes, scales, M, K, N, layout):
    w_t, i_t, o_t = dtypes
    w_scale, i_scale = scales
    transa = layout[0] == "T"
    transb = layout[1] == "T"

    weight_shape = [N, K] if transa else [K, N]
    weight = torch.rand(weight_shape, device=dev).to(w_t)
    weight_gold = weight.float() * w_scale

    input_shape = [K, M] if transb else [M, K]
    input = torch.rand(input_shape, device=dev).to(i_t)
    input_gold = input.float() * i_scale

    out_gold = layout_matmul(weight_gold, input_gold, layout).to(o_t)

    weight_te = Float8Tensor(
        shape=weight_shape,
        dtype=torch.float,
        data=weight.view(torch.uint8),
        fp8_scale_inv = torch.tensor([w_scale], dtype=torch.float32, device=dev),
        fp8_dtype=te_dtype_from_th_dtype(w_t),
        data_transpose=None,
        quantizer=None,
    )

    input_te = Float8Tensor(
        shape=input_shape,
        dtype=torch.float,
        data=input.view(torch.uint8),
        fp8_scale_inv = torch.tensor(i_scale, dtype=torch.float32, device=dev),
        fp8_dtype=te_dtype_from_th_dtype(i_t),
        data_transpose=None,
        quantizer=None,
    )

    out_te = torch.empty(M, N, dtype=o_t, device=dev)
    workspace = get_test_workspace()
    general_gemm(
        weight_te,
        input_te,
        workspace,
        out_dtype=o_t,
        out=out_te,
        layout=layout,
    )

    torch.testing.assert_close(out_te, out_gold, **get_fp8_tol(dtypes))


@pytest.mark.parametrize("dtypes", [
    [torch.float8_e4m3fn, torch.float8_e4m3fn, torch.bfloat16],
    [torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16],
    [torch.float8_e4m3fn, torch.float8_e5m2, torch.float],
])
@pytest.mark.parametrize("M", common_m)
@pytest.mark.parametrize("K", common_k)
@pytest.mark.parametrize("N", common_n)
@pytest.mark.parametrize("layout", common_layout)
@pytest.mark.parametrize("tile_size", [128])
def test_f8_f8_f16_mtfp8_tile_gemm(dtypes, M, K, N, layout, tile_size):
    w_t, i_t, o_t = dtypes
    src_dtype = torch.bfloat16

    # layout|input|weight|stage
    #-------|-----|------|-------
    #  TN   |group|block |fprop = A @ W.T
    #  NN   |group|block |dgrad = dG @ W
    #  NT   |group|group |wgrad = dG.T @ A

    if layout == "TN":
        weight_shape = [N, K]  # weight
        input_shape = [M, K]   # input
        out_shape = [M, N]
    elif layout == "NN":
        weight_shape = [N, K]  # weight
        input_shape = [M, N]   # grad_out
        out_shape = [M, K]
    else:
        weight_shape = [M, K]  # input
        input_shape = [M, N]   # grad_out
        out_shape = [N, K]

    weight_use_block = (layout != "NT")
    weight = torch.rand(weight_shape, device=dev, dtype=src_dtype)
    wrow_t, wrow_sinv, wcol_t, wcol_sinv = None, None, None, None
    weight_gold = None
    if weight_use_block:
        wrow_t, wrow_sinv = composite_blockwise_cast(weight, tile_size, w_t)
        weight_gold = composite_blockwise_uncast(wrow_t.float(), wrow_sinv, tile_size, src_dtype)
    else:
        wrow_t, wrow_sinv, wcol_t, wcol_sinv = \
            _gen_mtfp8_groupwise_cast_transpose_golden(weight, tile_size, w_t)
        weight_gold = composite_groupwise_uncast(wrow_t.float(), wrow_sinv, tile_size, src_dtype)

    weight_te = MTFP8Tensor(
        shape=weight_shape,
        dtype=src_dtype,
        rowwise_data=wrow_t,
        rowwise_scale_inv=wrow_sinv,
        columnwise_data=wcol_t,
        columnwise_scale_inv=wcol_sinv,
        fp8_dtype=te_dtype_from_th_dtype(w_t),
        quantizer=None,
    )

    input = torch.rand(input_shape, device=dev, dtype=src_dtype)
    irow_t, irow_sinv, icol_t, icol_sinv = \
        _gen_mtfp8_groupwise_cast_transpose_golden(input, tile_size, i_t)
    input_gold = composite_groupwise_uncast(irow_t.float(), irow_sinv, tile_size, src_dtype)

    input_te = MTFP8Tensor(
        shape=input_shape,
        dtype=src_dtype,
        rowwise_data=irow_t,
        rowwise_scale_inv=irow_sinv,
        columnwise_data=icol_t,
        columnwise_scale_inv=icol_sinv,
        fp8_dtype=te_dtype_from_th_dtype(i_t),
        quantizer=None,
    )

    out_gold = layout_matmul(weight_gold, input_gold, layout).to(o_t)
    assert out_gold.shape == torch.Size(out_shape)

    out_te = torch.empty(out_shape, dtype=o_t, device=dev)
    workspace = get_test_workspace()
    general_gemm(
        weight_te,
        input_te,
        workspace,
        out_dtype=o_t,
        out=out_te,
        layout=layout,
    )

    torch.testing.assert_close(out_te, out_gold, **get_fp8_tol(dtypes))
