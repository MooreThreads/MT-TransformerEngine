import torch, torch_musa
import pytest

import transformer_engine as te
import transformer_engine_torch as tex

from transformer_engine.pytorch.tensor.float8_tensor import (
    Float8Tensor,
)

from test_cast import (
    to_cpu_cast,
    te_dtype_from_th_dtype,
    dev,
)


@pytest.mark.parametrize("shape", [
    [[2048, 12288], True],
    [[4096, 4096], False],
    [[28672, 4096], True],
])
@pytest.mark.parametrize("dtype", [
    torch.float8_e5m2,
    torch.float8_e4m3fn,
])
def test_fp8_transpose(shape, dtype):
    shape, inv = shape
    te_dtype = te_dtype_from_th_dtype(dtype)

    cpu_src = torch.randn(shape).to(dtype)
    cpu_gold = to_cpu_cast(cpu_src.transpose(-1, -2), torch.float)

    musa_src = Float8Tensor(
        shape=shape,
        dtype=torch.float,
        data=cpu_src.view(torch.uint8).to(dev),
        fp8_scale_inv = torch.empty(1, dtype=torch.float32, device=dev),
        fp8_dtype=te_dtype,
        data_transpose=None,
        quantizer=None,
    )
    assert musa_src.dtype == torch.float
    assert musa_src._fp8_dtype == te_dtype
    assert musa_src._data.dtype == torch.uint8
    assert musa_src._transpose is None
    assert musa_src._transpose_invalid

    musa_src._create_transpose()
    cpu_dst = to_cpu_cast(musa_src._transpose.view(dtype), float)
    assert torch.equal(cpu_dst, cpu_gold)

    # out version
    musa_src._transpose.zero_()
    musa_src._create_transpose()
    cpu_dst = to_cpu_cast(musa_src._transpose.view(dtype), float)
    assert torch.equal(cpu_dst, cpu_gold)

    if inv:
        musa_src._transpose = None
        musa_src._reset_caches()
        assert musa_src._transpose is None
        assert musa_src._transpose_invalid
        musa_src._data = cpu_gold.to(dtype).view(torch.uint8).to(dev)
        cpu_gold = to_cpu_cast(cpu_src, torch.float)

        musa_src._create_transpose()
        cpu_dst = to_cpu_cast(musa_src._transpose.view(dtype), float)
        assert torch.equal(cpu_dst, cpu_gold)

        musa_src._transpose.zero_()
        musa_src._create_transpose()
        cpu_dst = to_cpu_cast(musa_src._transpose.view(dtype), float)
        assert torch.equal(cpu_dst, cpu_gold)
