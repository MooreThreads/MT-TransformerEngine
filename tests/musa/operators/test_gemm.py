import pytest
import torch, torch_musa

from transformer_engine.pytorch.cpp_extensions.gemm import (
    general_gemm,
)
from transformer_engine.pytorch.tensor.float8_tensor import (
    Float8Tensor,
)

from test_cast import (
    dev,
    te_dtype_from_th_dtype,
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


common_m = [4096]
common_k = [2048, 4096]
common_n = [2560, 8192, 14336, 25024]
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
