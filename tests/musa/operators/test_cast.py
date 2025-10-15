import torch, torch_musa
import pytest
import numpy as np

import math

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
        MTFP8BlockScaling(tile_size=group_size),
        mode=mode,
        num_quantizers=n_gemms,
        device=torch.device(dev),
    )
    return state


def composite_groupwise_cast(_src, group_size, dst_dtype):
    assert _src.dim() == 2
    m, n = _src.shape
    pad_n = ceil_div(n, group_size) * group_size
    src = torch.zeros(m, pad_n, dtype=_src.dtype, device=_src.device)
    src[:, :n] = _src

    fp_max = fp8_max(dst_dtype)
    cols = src.size(-1)
    temp = src.reshape(-1, cols).float()

    temp = temp.reshape(-1, group_size)
    amax = torch.abs(temp).max(-1, keepdim=True)[0]
    amax = amax.clamp(1e-4)
    scale = fp_max / amax

    dst = (temp * scale).to(dst_dtype).reshape(-1, cols)
    sinv = (amax / fp_max).reshape(-1, cols // group_size)
    return dst[:, :n].contiguous(), sinv


def composite_groupwise_uncast(_x, sinv, group_size, src_dtype):
    assert _x.dim() == 2
    m, n = _x.shape
    pad_n = ceil_div(n, group_size) * group_size
    x = torch.zeros(m, pad_n, dtype=_x.dtype, device=_x.device)
    x[:, :n] = _x

    orig_shape = x.shape
    res = ((x.reshape(-1, group_size)) * (sinv.reshape(-1, 1))).to(src_dtype)
    return res.reshape(orig_shape)[:, :n].contiguous()


def composite_groupwise_uncast_for_cast_transpose(x, sinv, group_size, src_dtype):
    orig_shape = x.shape
    padded_num = sinv.size(-1) * group_size - orig_shape[-1]
    if padded_num:
        new_shape = list(orig_shape); new_shape[-1] = sinv.size(-1) * group_size
        padded_tensor_shape = list(orig_shape); padded_tensor_shape[-1] = padded_num
        padded_tensor = torch.zeros(padded_tensor_shape, dtype=x.dtype, device=x.device)
        _x = torch.cat([x, padded_tensor], dim=-1)
        res = (_x.reshape(-1, sinv.size(-1), group_size) * sinv.unsqueeze(-1)).to(src_dtype)
        
        return res.reshape(new_shape)[..., :orig_shape[-1]]

    res = ((x.reshape(-1, group_size)) * (sinv.reshape(-1, 1))).to(src_dtype)
    return res.reshape(orig_shape)


@pytest.mark.parametrize("shape", [
    # align
    [[1024, 1024], 128],
    [[1024, 4096], 128],
    [[4096, 4096], 128],
    [[16384, 4096], 128],
    # [[16384, 16384], 128],
    # [[16384, 65536], 128],
    # [[65536, 65536], 128],
    # not align
    [[768, 640], 128],
    [[256, 65664], 128],
    [[2048, 2176], 128],
    [[80, 1024], 128],
    [[180, 4096], 128],
    [[1000, 4096], 128],
    [[1581, 16384], 128],

    [[16, 240], 128],
    [[768, 640+16], 128],
    [[256, 65664+32], 128],
    [[2048, 2176+64], 128],
    [[80, 1024+8], 128],
    [[180, 4096+16], 128],
    [[1000, 4096+32], 128],
    [[1581, 16384+64], 128],
    [[4096, (int)(128*85.5)], 128],
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
    gold_deq = composite_groupwise_uncast(gold_t, gold_sinv, group_size, src_dtype)

    musa_dst = quantizer(musa_src)
    dst_sinv = musa_dst._rowwise_scale_inv
    dst_t = musa_dst._rowwise_data.view(dst_dtype).float()

    assert torch.allclose(gold_sinv, dst_sinv, atol=1e-4, rtol=1e-4)
    assert torch.allclose(gold_t, dst_t, atol=1e-4, rtol=1e-4)

    musa_deq = musa_dst.dequantize()
    assert musa_deq.dtype == src_dtype
    assert torch.allclose(musa_deq, gold_deq, atol=1e-4, rtol=1e-4)

    musa_dst._rowwise_data.zero_()
    musa_dst._rowwise_scale_inv.zero_()
    quantizer.update_quantized(musa_src, musa_dst)
    dst_sinv = musa_dst._rowwise_scale_inv
    dst_t = musa_dst._rowwise_data.view(dst_dtype).float()

    assert torch.allclose(gold_sinv, dst_sinv, atol=1e-4, rtol=1e-4)
    assert torch.allclose(gold_t, dst_t, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("shape", [
    [[768, 1024], 128],
    [[767, 1024], 128],
    [[128, 128], 128],
    [[230, 128], 128],
    # [[16384, 65536], 128],  # for benchmark
    # [[16383, 65536], 128],  # for benchmark
    
    # K-dim unaligned cases
    [[4096, 576], 128],
])
@pytest.mark.parametrize("src_dtype", [
    torch.bfloat16,
    torch.float,
])
@pytest.mark.parametrize("dst_dtype", [
    torch.float8_e4m3fn,
    torch.float8_e5m2,
])
def test_mtfp8_groupwise_cast_transpose(shape, src_dtype, dst_dtype):
    shape, group_size = shape
    rs = create_mtfp8_groupwise_recipe_state(mode_from_th_dtype(dst_dtype), group_size)
    quantizer = rs.make_quantizers()[0]
    musa_src = torch.randn(shape, dtype = src_dtype, device = dev)
    musa_dst = quantizer(musa_src)

    # uncomment lines below to see bandwidth
    # for _ in range(5):
    #     quantizer.update_quantized(musa_src, musa_dst)
    # start = torch.musa.Event(enable_timing=True)
    # end = torch.musa.Event(enable_timing=True)
    # torch.musa.synchronize()
    # start.record()
    # nrepeats = 1
    # for _ in range(nrepeats):
    #     quantizer.update_quantized(musa_src, musa_dst)
    # end.record()
    # torch.musa.synchronize()

    # elapsed_time = start.elapsed_time(end) / nrepeats
    # # print(f"elapsed_time: {elapsed_time} ms")
    # nbytes = math.prod(shape) * (musa_src.element_size() + 1 + 1) + \
    #          math.prod(musa_dst._rowwise_scale_inv.shape) * 4 + \
    #          math.prod(musa_dst._columnwise_scale_inv.shape) * 4
    # print(f"Bandwidth: {nbytes / 1024 / 1024 / elapsed_time} GB/s")

    # gen golden
    dst_rowwise_golden, \
        scale_inv_rowwise_golden, \
            dst_columnwise_golden, \
                scale_inv_columnwise_golden = _gen_mtfp8_groupwise_cast_transpose_golden(musa_src, group_size, dst_dtype)
    
    assert torch.allclose(scale_inv_rowwise_golden.to(torch.float32), musa_dst._rowwise_scale_inv.to(torch.float32), atol=1e-4, rtol=1e-4)
    assert torch.allclose(scale_inv_columnwise_golden.to(torch.float32), musa_dst._columnwise_scale_inv.to(torch.float32), atol=1e-4, rtol=1e-4)
    assert torch.allclose(dst_rowwise_golden.to(torch.float32), musa_dst._rowwise_data.view(dst_dtype).to(torch.float32), atol=1e-4, rtol=1e-4)
    assert torch.allclose(dst_columnwise_golden.to(torch.float32), musa_dst._columnwise_data.view(dst_dtype).to(torch.float32), atol=1e-4, rtol=1e-4)

    # still failed in K-dim unaligned cases
    gold_deq = composite_groupwise_uncast_for_cast_transpose(dst_rowwise_golden.float(), scale_inv_rowwise_golden, group_size, src_dtype)
    musa_deq = musa_dst.dequantize()
    assert musa_deq.dtype == src_dtype
    assert torch.allclose(musa_deq, gold_deq, atol=1e-4, rtol=1e-4)


def _gen_mtfp8_groupwise_cast_transpose_golden(input_tensor, group_size, dst_dtype):
    fp_max = fp8_max(dst_dtype)
    input_tensor_tmp = input_tensor.reshape(-1, input_tensor.size(-1))
    nrows, ncols = input_tensor_tmp.shape

    ngroup_col = ncols // group_size
    ngroup_row = nrows // group_size

    if ((nrows % group_size == 0) and (ncols % group_size == 0)):
        # block-wise scaling along column
        tmp01 = input_tensor_tmp.reshape(nrows, ngroup_col, group_size).to(torch.float32)
        amax = torch.abs(tmp01).max(-1, keepdim=True)[0]
        amax = amax.clamp(1e-4)
        scale_inv = amax / fp_max
        dst_rowwise = (tmp01 / scale_inv).to(dst_dtype).reshape(-1, ncols)
        scale_inv_rowwise = scale_inv.reshape(nrows, ngroup_col)

        # block-wise scaling along row
        tmp02 = input_tensor_tmp.reshape(ngroup_row, group_size, ncols).to(torch.float32)
        amax = torch.abs(tmp02).max(1, keepdim=True)[0]
        amax = amax.clamp(1e-4)
        scale_inv = amax / fp_max
        dst_columnwise = (tmp02 / scale_inv).to(dst_dtype).reshape(-1, ncols)
        scale_inv_columnwise = scale_inv.reshape(ngroup_row, ncols)

    elif ((nrows % group_size != 0) and (ncols % group_size != 0)):
        # both unaligned cases seems rare
        raise NotImplementedError
    elif (nrows % group_size != 0):
        # block-wise scaling along column
        tmp01 = input_tensor_tmp.reshape(nrows, ngroup_col, group_size).to(torch.float32)
        amax = torch.abs(tmp01).max(-1, keepdim=True)[0]
        amax = amax.clamp(1e-4)
        scale_inv = amax / fp_max
        dst_rowwise = (tmp01 / scale_inv).to(dst_dtype).reshape(-1, ncols)
        scale_inv_rowwise = scale_inv.reshape(nrows, ngroup_col)

        # block-wise scaling along row 
        tmp02 = input_tensor_tmp.to(torch.float32)
        nrows_padded = ((nrows + group_size - 1) // group_size) * group_size
        ngroup_row_new = nrows_padded // group_size
        padding_tensor = torch.zeros((nrows_padded - nrows, ncols), dtype=torch.float32, device=tmp02.device)
        padded_tensor = torch.cat([tmp02, padding_tensor], 0)
        padded_tensor_reshaped = padded_tensor.reshape(ngroup_row_new, group_size, ncols)
        amax = torch.abs(padded_tensor_reshaped).max(1, keepdim=True)[0]
        amax = amax.clamp(1e-4)
        scale_inv = amax / fp_max
        dst_columnwise = (padded_tensor_reshaped / scale_inv).to(dst_dtype).reshape(-1, ncols)
        scale_inv_columnwise = scale_inv.reshape(ngroup_row_new, ncols)

        dst_columnwise = dst_columnwise[:nrows, ...]
    # elif (ncols % group_size != 0):
    else:
        # block-wise scaling along column
        tmp01 = input_tensor_tmp.to(torch.float32)
        ncols_padded = ((ncols + group_size - 1) // group_size) * group_size
        ngroup_col_new = ncols_padded // group_size
        padding_tensor = torch.zeros(nrows, (ncols_padded - ncols), dtype=torch.float32, device=tmp01.device)
        padded_tensor = torch.cat([tmp01, padding_tensor], -1)
        padded_tensor_reshaped = padded_tensor.reshape(nrows, ngroup_col_new, group_size)
        amax = torch.abs(padded_tensor_reshaped).max(-1, keepdim=True)[0]
        amax = amax.clamp(1e-4)
        scale_inv = amax / fp_max
        dst_rowwise = (padded_tensor_reshaped / scale_inv).to(dst_dtype).reshape(nrows, -1)
        scale_inv_rowwise = scale_inv.reshape(nrows, ngroup_col_new)
        dst_rowwise  = dst_rowwise[..., :ncols]

        # block-wise scaling along row
        tmp02 = input_tensor_tmp.reshape(ngroup_row, group_size, ncols).to(torch.float32)
        amax = torch.abs(tmp02).max(1, keepdim=True)[0]
        amax = amax.clamp(1e-4)
        scale_inv = amax / fp_max
        dst_columnwise = (tmp02 / scale_inv).to(dst_dtype).reshape(-1, ncols)
        scale_inv_columnwise = scale_inv.reshape(ngroup_row, ncols)

    return dst_rowwise, scale_inv_rowwise, dst_columnwise, scale_inv_columnwise


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
    x_amax = x_amax.clamp(1e-4)
    x_scaled = (x_view * (fpmax / x_amax)).to(dst_dtype)

    data = x_scaled.view_as(x_padded)[:m, :n].contiguous()
    sinv = (x_amax / fpmax).view(x_view.size(0), x_view.size(2))
    return data, sinv


def composite_blockwise_uncast(x, sinv, group_size, src_dtype):
    assert x.dim() == 2
    m, n = x.shape
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
    x_sinv = sinv.view(sinv.size(0), 1, sinv.size(1), 1)
    res = (x_view * x_sinv).to(src_dtype)
    res = res.reshape(x_padded.shape)
    return res[:m, :n].contiguous()


@pytest.mark.parametrize("shape", [
    # align
    [[1024, 1024], 128],
    [[1024, 4096], 128],
    [[4096, 4096], 128],
    [[16384, 4096], 128],
    # [[16384, 16384], 128],
    # [[16384, 65536], 128],
    # [[65536, 65536], 128],
    # not align
    [[80, 1024], 128],
    [[180, 4096], 128],
    [[1000, 4096], 128],
    [[1581, 16384], 128],

    [[80, 1024 + 64], 128],
    [[180, 4096 + 32], 128],
    [[1000, 4096 + 16], 128],
    [[1581, 16384 + 8], 128],
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
    gold_deq = composite_blockwise_uncast(gold_t, gold_sinv, group_size, src_dtype)

    musa_dst = quantizer(musa_src)
    dst_sinv = musa_dst._rowwise_scale_inv
    dst_t = musa_dst._rowwise_data.view(dst_dtype).float()

    assert torch.allclose(gold_sinv, dst_sinv, atol=1e-4, rtol=1e-4)
    assert torch.allclose(gold_t, dst_t, atol=1e-4, rtol=1e-4)

    musa_deq = musa_dst.dequantize()
    assert musa_deq.dtype == src_dtype
    assert torch.equal(musa_deq, gold_deq)

    musa_dst._rowwise_data.zero_()
    musa_dst._rowwise_scale_inv.zero_()
    quantizer.update_quantized(musa_src, musa_dst)
    dst_sinv = musa_dst._rowwise_scale_inv
    dst_t = musa_dst._rowwise_data.view(dst_dtype).float()

    assert torch.allclose(gold_sinv, dst_sinv, atol=1e-4, rtol=1e-4)
    assert torch.allclose(gold_t, dst_t, atol=1e-4, rtol=1e-4)
