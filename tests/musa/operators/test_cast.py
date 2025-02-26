import torch, torch_musa
import pytest

import transformer_engine as te
import transformer_engine_torch as tex

from transformer_engine.common.recipe import (
    DelayedScaling,
)
from transformer_engine.pytorch.fp8 import (
    DelayedScalingRecipeState,
)


dev = "musa"


def to_cpu_cast(t, dst_dtype):
    cpu_t = t if t.is_cpu else t.cpu()
    return cpu_t.to(dst_dtype)


def abs_max_per_tensor(t, dtype):
    type_t = t if t.dtype == dtype else t.to(dtype)
    return type_t.abs().max()


def mode_from_th_dtype(th_dtype):
    assert th_dtype in [torch.float8_e5m2, torch.float8_e4m3fn]
    if th_dtype == torch.float8_e5m2:
        return "backward"
    return "forward"


def te_dtype_from_th_dtype(th_dtype):
    if th_dtype == torch.bfloat16:
        return tex.DType.kBFloat16
    if th_dtype == torch.float16:
        return tex.DType.kFloat16
    if th_dtype == torch.float:
        return tex.DType.kFloat32
    if th_dtype == torch.float8_e5m2:
        return tex.DType.kFloat8E5M2
    return tex.DType.kFloat8E4M3


def create_per_tensor_recipe_state(scale, mode):
    n_gemms = 1
    state = DelayedScalingRecipeState(
        DelayedScaling(),
        mode=mode,
        num_quantizers=n_gemms,
        device=torch.device(dev),
    )
    state.scale = torch.tensor(
        [scale] * n_gemms, dtype=torch.float32, device=dev)
    return state


@pytest.mark.parametrize("shape", [
    [2048, 12288],
    [768, 1024],
    [256, 65536],
    [65536, 128],
    [256, 256],
    [120, 2080],
    [8, 8],
    [1223, 1583],
    [1, 541],
    [1987, 1],
    [256, 128],
])
@pytest.mark.parametrize("src_dtype", [
    torch.float16,
    torch.bfloat16,
])
@pytest.mark.parametrize("dst_dtype", [
    torch.float8_e5m2,
    torch.float8_e4m3fn,
])
@pytest.mark.parametrize("scale", [0.5, 1.0, 1.5, 2.0])
def test_float_cast_to_fp8_per_tensor(shape, src_dtype, dst_dtype, scale):
    rs = create_per_tensor_recipe_state(scale, mode_from_th_dtype(dst_dtype))
    quantizer = rs.make_quantizers()[0]

    musa_src = torch.randn(shape, dtype = src_dtype, device = dev)
    cpu_src = to_cpu_cast(musa_src, torch.float)

    cpu_amax = abs_max_per_tensor(cpu_src, torch.float).reshape(-1)
    cpu_gold = cpu_src * scale
    cpu_gold = cpu_gold.to(dst_dtype).float()

    musa_dst = quantizer(musa_src)

    assert musa_dst._transpose is None
    assert musa_dst._transpose_invalid

    assert musa_dst.dtype == src_dtype
    assert musa_dst._fp8_dtype == te_dtype_from_th_dtype(dst_dtype)

    assert musa_dst._data.dtype == torch.uint8
    assert torch.allclose(
        to_cpu_cast(musa_dst._scale_inv, torch.float32),
        to_cpu_cast(quantizer.scale.reciprocal(), torch.float32),
    )

    cpu_dst = to_cpu_cast(musa_dst._data.view(dst_dtype), torch.float)

    assert torch.equal(cpu_amax, to_cpu_cast(quantizer.amax, torch.float))
    assert torch.equal(cpu_gold, cpu_dst)

    # out version
    musa_dst._scale_inv.zero_()
    musa_dst._data.zero_()
    quantizer.amax.zero_()
    quantizer.update_quantized(musa_src, musa_dst)
    cpu_dst = to_cpu_cast(musa_dst._data.view(dst_dtype), torch.float)

    assert torch.allclose(
        to_cpu_cast(musa_dst._scale_inv, torch.float32),
        to_cpu_cast(quantizer.scale.reciprocal(), torch.float32),
    )
    assert torch.equal(cpu_amax, to_cpu_cast(quantizer.amax, torch.float))
    assert torch.equal(cpu_gold, cpu_dst)
