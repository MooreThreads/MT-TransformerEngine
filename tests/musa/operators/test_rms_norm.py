import pytest
import torch, torch_musa
from torch.nn import init

import transformer_engine as te
import transformer_engine_torch as tex

from test_cast import (
    dev,
    te_dtype_from_th_dtype,
)

def get_non_fp8_tol(dtype):
    assert dtype in [torch.bfloat16, torch.float16, torch.float]
    if dtype == torch.bfloat16:
        return dict(atol=1e-3, rtol=1e-2)
    if dtype == torch.float16:
        return dict(atol=1e-3, rtol=1e-2)
    return dict(atol=1e-6, rtol=1e-6)


op_fwd_te = tex.rmsnorm_fwd
op_bwd_te = tex.rmsnorm_bwd

op_fwd_th = torch.ops.aten._fused_rmsnorm_forward
op_bwd_th = torch.ops.aten._fused_rmsnorm_backward

common_shape = [
    [71, 229],
    [29, 541],
    [768, 6144],
    [2048, 12288],
]


@pytest.mark.parametrize("shape", common_shape)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_non_fp8_rms_norm_fwd(shape, dtype):
    normalized_shape = shape[-1:]
    eps = 1e-5

    input = torch.randn(shape, dtype=dtype, device=dev)
    weight = torch.randn(normalized_shape, dtype=dtype, device=dev)
    init.trunc_normal_(weight)
    
    out_gold, invvar_gold = op_fwd_th(input, normalized_shape, eps, weight)

    out_te = torch.zeros_like(out_gold)
    out_te, _, invvar_te = op_fwd_te(
        input,
        weight,
        eps,
        ln_out=out_te,
        quantizer=None,
        otype=te_dtype_from_th_dtype(dtype),
        sm_margin=0,
        zero_centered_gamma=False,
    )
    torch.testing.assert_close(out_te, out_gold, **get_non_fp8_tol(dtype))
    torch.testing.assert_close(invvar_te, invvar_gold, **get_non_fp8_tol(dtype))


@pytest.mark.parametrize("shape", common_shape)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_non_fp8_rms_norm_bwd(shape, dtype):
    normalized_shape = shape[-1:]
    invvar_shape = shape[:-1]
    eps = 1e-5
    
    grad_out = torch.randn(shape, dtype=dtype, device=dev)
    invvar = torch.randn(invvar_shape, dtype=dtype, device=dev)

    input = torch.randn(shape, dtype=dtype, device=dev)
    weight = torch.randn(normalized_shape, dtype=dtype, device=dev)
    init.trunc_normal_(weight)

    dx_gold, dgamma_gold = op_bwd_th(
        grad_out,
        invvar,
        input,
        normalized_shape,
        eps,
        weight,
    )
    
    dx_te, dgamma_te = op_bwd_te(
        grad_out,
        input,
        invvar,
        weight,
        0,
        False,
    )
    torch.testing.assert_close(dx_te, dx_gold, **get_non_fp8_tol(dtype))
    torch.testing.assert_close(dgamma_te, dgamma_gold, **get_non_fp8_tol(dtype))
