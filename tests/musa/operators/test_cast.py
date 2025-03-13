import torch, torch_musa
import pytest

import transformer_engine as te
import transformer_engine_torch as tex

from transformer_engine.pytorch.cpp_extensions import (
    cast_to_fp8,
)
from transformer_engine.common.recipe import (
    DelayedScaling,
    MTFP8BlockScaling,
)
from transformer_engine.pytorch.fp8 import (
    DelayedScalingRecipeState,
    MTFP8BlockScalingRecipeState,
)
from transformer_engine.pytorch.tensor.float8_tensor import (
    Float8Tensor,
)


dev = "musa"


def ceil_div(a, b):
    return (a + b - 1) // b


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


def fp8_max(th_dtype):
    assert th_dtype in [torch.float8_e5m2, torch.float8_e4m3fn]
    if th_dtype == torch.float8_e5m2:
        return 57344.0
    return 448.0


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


def test_legacy_cast_to_fp8_per_tensor():
    shape = (128, 256)
    fake_dtype = torch.bfloat16
    th_dtype = torch.float8_e4m3fn
    te_dtype = te_dtype_from_th_dtype(th_dtype)
    scale = 0.5

    inp_cpu = torch.rand(shape, dtype=fake_dtype)
    res_cpu = to_cpu_cast(inp_cpu, torch.float)
    res_cpu = res_cpu / scale
    res_cpu = to_cpu_cast(res_cpu.to(th_dtype), torch.float)

    inp_musa = inp_cpu.to(dev).zero_().to(th_dtype)
    te_tensor = Float8Tensor(
        shape=shape,
        dtype=fake_dtype,
        data=inp_musa.view(torch.uint8),
        fp8_scale_inv = torch.tensor([scale], dtype=torch.float32, device=dev),
        fp8_dtype=te_dtype,
        data_transpose=None,
        quantizer=None,
    )
    cast_to_fp8(inp_cpu.to(dev), te_tensor)
    res_musa = to_cpu_cast(te_tensor._data.view(th_dtype), torch.float)

    assert torch.equal(res_cpu, res_musa)


def create_mtfp8_groupwise_recipe_state(mode, group_size):
    if mode == "forward":
        n_gemms = 3
    else:
        n_gemms = 2
    state = MTFP8BlockScalingRecipeState(
        MTFP8BlockScaling(
            activation_block_m=1,
            activation_block_n=group_size,
            weight_block_m=group_size,
            weight_block_n=group_size,
        ),
        mode=mode,
        num_quantizers=n_gemms,
        device=torch.device(dev),
    )
    return state


def composite_groupwise_cast(src, group_size, dst_dtype):
    fp_max = fp8_max(dst_dtype)
    cols = src.size(-1)
    temp = src.reshape(-1, cols).float()

    temp = temp.reshape(-1, group_size)
    amax = torch.abs(temp).max(-1, keepdim=True)[0]
    scale = fp_max / amax

    dst = (temp * scale).to(dst_dtype).reshape(-1, cols)
    sinv = (amax / fp_max).reshape(-1, cols // group_size)
    return dst, sinv


@pytest.mark.parametrize("shape", [
    # align
    [[1024, 1024], 128],
    [[1024, 4096], 128],
    [[4096, 4096], 128],
    [[16384, 4096], 128],
    [[16384, 16384], 128],
    [[16384, 65536], 128],
    [[65536, 65536], 128],
    # not align
    [[768, 640], 128],
    [[256, 65664], 128],
    [[2048, 2176], 128],
    [[80, 1024], 128],
    [[180, 4096], 128],
    [[1000, 4096], 128],
    [[1581, 16384], 128],
])
@pytest.mark.parametrize("src_dtype", [
    torch.bfloat16,
    torch.float,
])
@pytest.mark.parametrize("dst_dtype", [
    torch.float8_e4m3fn,
    torch.float8_e5m2,
])
def test_mtfp8_groupwise_cast_to_fp8(shape, src_dtype, dst_dtype):
    shape, group_size = shape
    rs = create_mtfp8_groupwise_recipe_state(mode_from_th_dtype(dst_dtype), group_size)
    quantizer = rs.make_quantizers()[0]
    quantizer.columnwise_usage = False

    musa_src = torch.randn(shape, dtype = src_dtype, device = dev)

    gold_t, gold_sinv = composite_groupwise_cast(musa_src, group_size, dst_dtype)
    gold_t = gold_t.float()

    musa_dst = quantizer(musa_src)
    dst_sinv = musa_dst._rowwise_scale_inv
    dst_t = musa_dst._rowwise_data.view(dst_dtype).float()

    assert torch.equal(gold_sinv, dst_sinv)
    assert torch.equal(gold_t, dst_t)

    musa_dst._rowwise_data.zero_()
    musa_dst._rowwise_scale_inv.zero_()
    quantizer.update_quantized(musa_src, musa_dst)
    dst_sinv = musa_dst._rowwise_scale_inv
    dst_t = musa_dst._rowwise_data.view(dst_dtype).float()

    assert torch.equal(gold_sinv, dst_sinv)
    assert torch.equal(gold_t, dst_t)


@pytest.mark.parametrize("shape", [
    [[768, 1024], 128],
])
@pytest.mark.parametrize("src_dtype", [
    torch.bfloat16,
])
@pytest.mark.parametrize("dst_dtype", [
    torch.float8_e4m3fn,
])
def test_mtfp8_groupwise_cast_transpose(shape, src_dtype, dst_dtype):
    shape, group_size = shape
    rs = create_mtfp8_groupwise_recipe_state(mode_from_th_dtype(dst_dtype), group_size)
    quantizer = rs.make_quantizers()[0]
    musa_src = torch.randn(shape, dtype = src_dtype, device = dev)
    musa_dst = quantizer(musa_src)


def composite_blockwise_cast(x, group_size, dst_dtype):
    assert x.dim() == 2
    m, n = x.shape
    fpmax = fp8_max(dst_dtype)
    x_padded = torch.zeros(
        (
            ceil_div(m, group_size) * group_size,
            ceil_div(n, group_size) * group_size,
        ),
        dtype=x.dtype,
        device=x.device,
    )
    x_padded[:m, :n] = x

    x_view = x_padded.view(-1, group_size, x_padded.size(1) // group_size, group_size)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True)
    x_scaled = (x_view * (fpmax / x_amax)).to(dst_dtype)
    
    data = x_scaled.view_as(x_padded)[:m, :n].contiguous()
    sinv = (x_amax / fpmax).view(x_view.size(0), x_view.size(2))
    return data, sinv


@pytest.mark.parametrize("shape", [
    # align
    [[1024, 1024], 128],
    [[1024, 4096], 128],
    [[4096, 4096], 128],
    [[16384, 4096], 128],
    [[16384, 16384], 128],
    [[16384, 65536], 128],
    [[65536, 65536], 128],
    # not align
    [[80, 1024], 128],
    [[180, 4096], 128],
    [[1000, 4096], 128],
    [[1581, 16384], 128],
])
@pytest.mark.parametrize("src_dtype", [
    torch.bfloat16,
    torch.float,
])
def test_mtfp8_blockwise_cast_to_fp8(shape, src_dtype):
    shape, group_size = shape
    dst_dtype = torch.float8_e4m3fn

    mode = "forward"
    rs = create_mtfp8_groupwise_recipe_state(mode, group_size)
    quantizer = rs.make_quantizers()[1]
    quantizer.columnwise_usage = False

    musa_src = torch.randn(shape, dtype = src_dtype, device = dev)

    gold_t, gold_sinv = composite_blockwise_cast(musa_src, group_size, dst_dtype)
    gold_t = gold_t.float()

    musa_dst = quantizer(musa_src)
    dst_sinv = musa_dst._rowwise_scale_inv
    dst_t = musa_dst._rowwise_data.view(dst_dtype).float()

    assert torch.equal(gold_sinv, dst_sinv)
    assert torch.equal(gold_t, dst_t)

    musa_dst._rowwise_data.zero_()
    musa_dst._rowwise_scale_inv.zero_()
    quantizer.update_quantized(musa_src, musa_dst)
    dst_sinv = musa_dst._rowwise_scale_inv
    dst_t = musa_dst._rowwise_data.view(dst_dtype).float()

    assert torch.equal(gold_sinv, dst_sinv)
    assert torch.equal(gold_t, dst_t)
