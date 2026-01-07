# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import math
import os
from typing import Dict, List, Optional
import pytest
import copy
import random

import torch
import torch.nn as nn
from torch.nn import Parameter

from transformer_engine.pytorch.fp8 import (
    FP8GlobalStateManager,
    fp8_autocast,
    fp8_model_init,
)
from transformer_engine.pytorch.utils import (
    init_method_normal,
    scaled_init_method_normal,
    attention_mask_func,
    is_bf16_compatible,
)
from transformer_engine.pytorch import (
    DotProductAttention,
    LayerNormLinear,
    LayerNormMLP,
    Linear,
    GroupedLinear,
    MultiheadAttention,
    RMSNorm,
    TransformerLayer,
    LayerNorm,
    InferenceParams,
    Fp8Padding,
    Fp8Unpadding,
)
from transformer_engine.pytorch.distributed import checkpoint as te_checkpoint
from transformer_engine.pytorch.cpp_extensions import general_gemm, general_grouped_gemm
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
from transformer_engine.pytorch.module.base import get_multi_stream_cublas_workspace, get_workspace
from transformer_engine.pytorch.utils import get_device_compute_capability
from transformer_engine.common import recipe
import transformer_engine_torch as tex

# Only run FP8 tests on supported devices.
fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()
mxfp8_available, reason_for_no_mxfp8 = FP8GlobalStateManager.is_mxfp8_available()

sm_80plus = get_device_compute_capability() >= (8, 0)

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# Record initial RNG state from script run.
_cpu_rng_state = torch.get_rng_state()
_cuda_rng_state = torch.cuda.get_rng_state()


class ModelConfig:
    def __init__(self, hidden_size, eps, num_attention_heads, embed, num_layers, seq_len):
        self.hidden_size = hidden_size
        self.eps = eps
        self.num_attention_heads = num_attention_heads
        self.embed = embed
        self.num_layers = num_layers
        self.seq_len = seq_len


model_configs = {
    "small": ModelConfig(128, 1e-5, 8, 36, 4, 128),
    "126m": ModelConfig(768, 1e-5, 12, 64, 12, 2048),
}

model_configs_inference = {
    # hidden_size, eps, num_attention_heads, embed, num_layers, seq_len
    "126m": ModelConfig(768, 1e-5, 12, 64, 12, 16),
}
backends_inference = ["FlashAttention", "UnfusedAttention"]
module_inference = ["TransformerLayer", "MultiheadAttention"]
input_formats_inference = ["sbhd", "bshd"]

param_types = [torch.float32, torch.float16]
if is_bf16_compatible():  # bf16 requires sm_80 or higher
    param_types.append(torch.bfloat16)

batch_sizes = [1, 2]

all_boolean = [True, False]

all_activations = ["gelu", "relu", "reglu", "geglu", "swiglu", "qgelu", "srelu"]

all_normalizations = ["LayerNorm", "RMSNorm"]

mask_types = ["causal", "no_mask"]

fp8_recipes = [
    recipe.MXFP8BlockScaling(),
    recipe.DelayedScaling(),
]


def get_causal_attn_mask(sq: int) -> torch.Tensor:
    return torch.triu(torch.ones(sq, sq, device="cuda"), diagonal=1).bool()


def dtype_tols(dtype: torch.dtype) -> Dict[str, float]:
    """Estimated numerical error for a datatype

    Based on tolerances for torch.testing.assert_close.

    """
    if dtype == torch.float32:
        return dict(rtol=1.3e-6, atol=1e-5)
    if dtype == torch.float16:
        return dict(rtol=1e-3, atol=1e-5)
    if dtype == torch.bfloat16:
        return dict(rtol=1.6e-2, atol=1e-5)
    raise ValueError(f"Unsuppored dtype ({dtype})")


def assert_allclose(
    l1: List[torch.Tensor], l2: List[torch.Tensor], atol: float, rtol: float = None
) -> bool:
    """Ensures two lists are equal."""
    assert len(l1) == len(l2), "Unequal number of outputs."
    for i, (t1, t2) in enumerate(zip(l1, l2)):
        tols = dict(atol=atol)
        if rtol is not None:
            tols["rtol"] = rtol
        result = torch.allclose(t1, t2, **tols)
        if not result:
            diff = torch.abs(t1 - t2)
            tol = atol + (rtol * torch.abs(t2))
            exceed_mask = diff > tol
            if exceed_mask.any():
                indices = torch.nonzero(exceed_mask, as_tuple=True)
                max_diff = diff[exceed_mask].max()
                max_idx = (diff[exceed_mask] == max_diff).nonzero(as_tuple=True)[0][0]
                max_location = [idx[max_idx].item() for idx in indices]
                msg = (
                    f"Outputs not close enough in tensor at idx={i}. "
                    f"Maximum difference at location {max_location} "
                    f"with {t1[exceed_mask][max_idx].item()} vs {t2[exceed_mask][max_idx].item()} "
                    f"(diff {max_diff.item()})."
                )
            raise AssertionError(msg)


def reset_rng_states() -> None:
    """revert back to initial RNG state."""
    torch.set_rng_state(_cpu_rng_state)
    torch.cuda.set_rng_state(_cuda_rng_state)


@pytest.fixture(autouse=True)
def reset_global_fp8_state():
    yield
    FP8GlobalStateManager.reset()


class TorchScaledMaskedSoftmax(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, inp: torch.Tensor, mask: torch.Tensor, scale: Optional[float] = None
    ) -> torch.Tensor:
        dtype = inp.dtype
        inp = inp.float()

        if scale is not None:
            inp = inp * scale
        mask_output = attention_mask_func(inp, mask) if mask is not None else inp

        probs = torch.nn.Softmax(dim=-1)(mask_output)
        probs = probs.to(dtype)
        return probs


class TorchDotProductAttention(torch.nn.Module):
    def __init__(
        self,
        kv_channels: int,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.norm_factor = math.sqrt(kv_channels)
        self.scale_mask_softmax = TorchScaledMaskedSoftmax()
        self.attention_dropout = torch.nn.Dropout(attention_dropout)

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seqlen = query_layer.shape[1], query_layer.shape[0]

        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.reshape(output_size[2], output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.reshape(output_size[3], output_size[0] * output_size[1], -1)

        # preallocting result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=torch.cuda.current_device(),
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)
        attention_probs = self.attention_dropout(attention_probs)

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]
        output_size = (
            value_layer.size(1),
            value_layer.size(2),
            query_layer.size(0),
            value_layer.size(3),
        )

        # change view [sk, b * np, hn]
        value_layer = value_layer.reshape(value_layer.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        context_layer = context_layer.view(seqlen, batch_size, -1)

        return context_layer


class TorchLayerNorm(nn.Module):
    def __init__(self, in_features: int, eps: float, zero_centered_gamma: bool):
        super().__init__()
        self.eps = eps
        self.in_features = in_features
        self.zero_centered_gamma = zero_centered_gamma

        initial_value = torch.ones(in_features) if zero_centered_gamma else torch.zeros(in_features)
        self.weight = nn.Parameter(initial_value)
        self.bias = nn.Parameter(torch.zeros(in_features))
        self.register_parameter("weight", self.weight)
        self.register_parameter("bias", self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight if not self.zero_centered_gamma else 1 + self.weight
        w = w.to(torch.float32)
        b = self.bias.to(torch.float32)
        inp = x.to(torch.float32)
        out = torch.nn.functional.layer_norm(
            inp, (self.in_features,), weight=w, bias=b, eps=self.eps
        )
        return out.to(x.dtype)


# Adapted from https://github.com/bzhangGo/rmsnorm/blob/c6691f20ec0af4128c8159c903071f7575404295/rmsnorm_torch.py
class TorchRMSNorm(nn.Module):
    def __init__(self, in_features, zero_centered_gamma, eps=1e-5):
        super().__init__()

        self.eps = eps
        self.in_features = in_features
        self.zero_centered_gamma = zero_centered_gamma

        initial_value = torch.ones(in_features) if zero_centered_gamma else torch.zeros(in_features)
        self.weight = nn.Parameter(initial_value)
        self.register_parameter("weight", self.weight)

    def forward(self, x):
        norm_x2 = torch.sum(x.float() ** 2, dim=-1, keepdim=True)
        d_x = self.in_features

        rms_x2 = norm_x2 / d_x + self.eps
        r_rms_x = rms_x2 ** (-1.0 / 2)
        x_normed = x * r_rms_x

        w = self.weight.float()
        if self.zero_centered_gamma:
            w = 1 + w
        return (w * x_normed).to(x.dtype)


class TorchLayerNormLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        eps: float,
        bias: bool = True,
        normalization: str = "LayerNorm",
        zero_centered_gamma: bool = False,
    ):
        super().__init__()
        if normalization == "LayerNorm":
            self.layernorm = TorchLayerNorm(
                in_features, eps=eps, zero_centered_gamma=zero_centered_gamma
            )
        elif normalization == "RMSNorm":
            self.layernorm = TorchRMSNorm(
                in_features, eps=eps, zero_centered_gamma=zero_centered_gamma
            )
        else:
            raise RuntimeError("Unsupported normalization")

        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.layernorm(x))


class TorchMHA(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int):
        super().__init__()
        self.mhsa = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=0.1,
            bias=True,
            batch_first=False,
        )

    def forward(self, x, attention_mask=None):
        output = self.mhsa(x, x, x, attn_mask=attention_mask, need_weights=False)
        if isinstance(output, tuple):
            output = output[0]
        return output


class TorchQuickGELU(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.sigmoid(1.702 * input)


class TorchSquaredRELU(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (input > 0) * input * input


class TorchGroupedLinearWithPadding(nn.Module):

    def __init__(
        self, num_gemms, in_features, out_features, bias, params_dtype, parallel_mode, fp8
    ) -> None:
        super().__init__()

        self.padding = Fp8Padding(num_gemms)
        self.linear_fn = GroupedLinear(
            num_gemms,
            in_features,
            out_features,
            bias=bias,
            params_dtype=params_dtype,
            parallel_mode=parallel_mode,
            device="cuda",
        )
        self.unpadding = Fp8Unpadding(num_gemms)

        self.fp8 = fp8

    def forward(self, inp: torch.Tensor, m_splits: List[int]) -> torch.Tensor:
        if self.fp8:
            orig_m_splits = m_splits
            inp, m_splits = self.padding(inp, m_splits)

        out = self.linear_fn(inp, m_splits)

        if self.fp8:
            out = self.unpadding(out, orig_m_splits)

        return out


_supported_act = {
    "geglu": nn.GELU(approximate="tanh"),
    "gelu": nn.GELU(approximate="tanh"),
    "reglu": nn.ReLU(),
    "relu": nn.ReLU(),
    "swiglu": nn.SiLU(),
    "qgelu": TorchQuickGELU(),
    "srelu": TorchSquaredRELU(),
}


class TorchGLU(nn.Module):
    def __init__(self, activation: str):
        super().__init__()
        self.act = _supported_act[activation]

    def forward(self, x):
        shape = x.size(-1)
        a = x[..., : shape // 2]
        b = x[..., (shape // 2) :]
        a = self.act(a)
        return a * b


class TorchLayerNormMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        eps: float = 1e-5,
        activation="gelu",
        normalization: str = "LayerNorm",
    ):
        super().__init__()
        if normalization == "LayerNorm":
            self.ln = TorchLayerNorm(hidden_size, eps=eps, zero_centered_gamma=False)
        elif normalization == "RMSNorm":
            self.ln = TorchRMSNorm(hidden_size, eps=eps, zero_centered_gamma=False)
        else:
            raise RuntimeError("Unsupported normalization")
        if "glu" in activation:
            fc1_output_features = 2 * ffn_hidden_size
            self.gelu = TorchGLU(activation)
        else:
            fc1_output_features = ffn_hidden_size
            self.gelu = _supported_act[activation]

        self.fc1 = nn.Linear(hidden_size, fc1_output_features)
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size)

    def forward(self, x):
        t = self.gelu(self.fc1(self.ln(x)))
        return self.fc2(t)


class TorchGPT(nn.Module):
    def __init__(
        self, hidden_size: int, eps: float, num_attention_heads: int, parallel_attention_mlp: bool
    ):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size, eps=eps)
        self.causal_attn = TorchMHA(hidden_size, num_attention_heads)
        self.ln_mlp = TorchLayerNormMLP(hidden_size, 4 * hidden_size, eps)
        self.parallel_attention_mlp = parallel_attention_mlp

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        a = self.ln(x)
        b = self.causal_attn(a, attention_mask)
        if self.parallel_attention_mlp:
            n = self.ln_mlp(x)
            x = x + nn.functional.dropout(b + n, p=0.1, training=self.training)
        else:
            x = x + nn.functional.dropout(b, p=0.1, training=self.training)
            n = self.ln_mlp(x)
            x = x + nn.functional.dropout(n, p=0.1, training=self.training)
        return x


def _test_e2e_selective_recompute(
    bs, dtype, config, fp8, recipe, fp8_model_params=False, recompute=False
):
    reset_rng_states()
    FP8GlobalStateManager.reset()

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    with fp8_model_init(enabled=fp8 and fp8_model_params, recipe=recipe):
        block = TransformerLayer(
            config.hidden_size,
            4 * config.hidden_size,
            config.num_attention_heads,
            layernorm_epsilon=config.eps,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            hidden_dropout=0.1,
            attention_dropout=0.1,
            kv_channels=config.embed,
            apply_residual_connection_post_layernorm=False,
            output_layernorm=False,
            params_dtype=dtype,
            fuse_qkv_params=True,
            device="cuda",
        )

    te_inp_hidden_states = torch.randn(
        (config.seq_len, bs, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    te_inp_hidden_states.retain_grad()
    te_inp_attn_mask = get_causal_attn_mask(config.seq_len)

    with fp8_autocast(enabled=fp8, fp8_recipe=recipe):
        te_out = block(
            te_inp_hidden_states,
            attention_mask=te_inp_attn_mask,
            checkpoint_core_attention=recompute,
        )
    loss = te_out.sum()
    loss.backward()
    torch.cuda.synchronize()

    outputs = [te_out, te_inp_hidden_states.grad]
    for p in block.parameters():
        if p.requires_grad:
            outputs.append(p.grad)
    return outputs


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["126m"])
@pytest.mark.parametrize("fp8", all_boolean)
@pytest.mark.parametrize("recipe", fp8_recipes)
@pytest.mark.parametrize("fp8_model_params", all_boolean)
def test_gpt_selective_activation_recompute(dtype, bs, model, fp8, recipe, fp8_model_params):
    if fp8 and not fp8_available:
        pytest.skip(reason_for_no_fp8)
    if recipe.mxfp8() and not mxfp8_available:
        pytest.skip(reason_for_no_mxfp8)

    config = model_configs[model]

    outputs = _test_e2e_selective_recompute(
        bs, dtype, config, fp8, recipe, fp8_model_params, recompute=False
    )
    outputs_recompute = _test_e2e_selective_recompute(
        bs, dtype, config, fp8, recipe, fp8_model_params, recompute=True
    )

    # Check that results match
    tols = dtype_tols(dtype)
    if dtype in (torch.float16, torch.bfloat16):
        tols["atol"] = 1e-4
    if fp8 or fp8_model_params:
        tols.update(dict(rtol=0.125, atol=0.0675))

    for i, (ref, test) in enumerate(zip(outputs, outputs_recompute)):
        torch.testing.assert_close(
            test,
            ref,
            msg=f"Mismatch in tensor {i}",
            **tols,
        )


def _test_e2e_full_recompute(
    bs, dtype, config, fp8, recipe, fp8_model_params=False, recompute=False, use_reentrant=True
):
    reset_rng_states()
    FP8GlobalStateManager.reset()

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    with fp8_model_init(enabled=fp8 and fp8_model_params, recipe=recipe):
        block = TransformerLayer(
            config.hidden_size,
            4 * config.hidden_size,
            config.num_attention_heads,
            layernorm_epsilon=config.eps,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            hidden_dropout=0.1,
            attention_dropout=0.1,
            kv_channels=config.embed,
            apply_residual_connection_post_layernorm=False,
            output_layernorm=False,
            params_dtype=dtype,
            fuse_qkv_params=True,
            device="cuda",
        )

    te_inp_hidden_states = torch.randn(
        (config.seq_len, bs, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=use_reentrant,
    )
    if use_reentrant:
        te_inp_hidden_states.retain_grad()
    te_inp_attn_mask = get_causal_attn_mask(config.seq_len)

    with fp8_autocast(enabled=fp8, fp8_recipe=recipe):
        if recompute:
            te_out = te_checkpoint(
                block,
                te_inp_hidden_states,
                attention_mask=te_inp_attn_mask,
                checkpoint_core_attention=False,
                distribute_saved_activations=False,
                tp_group=None,
                use_reentrant=use_reentrant,
            )
        else:
            te_out = block(
                te_inp_hidden_states,
                attention_mask=te_inp_attn_mask,
                checkpoint_core_attention=False,
            )
    loss = te_out.sum()
    loss.backward()
    torch.cuda.synchronize()

    outputs = [te_out]
    names = ["output"]
    if use_reentrant:
        outputs.append(te_inp_hidden_states.grad)
        names.append("input")
    for name, p in block.named_parameters():
        if p.requires_grad:
            outputs.append(p.grad)
            names.append(name)

    return outputs, names


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["126m"])
@pytest.mark.parametrize("fp8", all_boolean)
@pytest.mark.parametrize("recipe", fp8_recipes)
@pytest.mark.parametrize("fp8_model_params", all_boolean)
@pytest.mark.parametrize("use_reentrant", all_boolean)
def test_gpt_full_activation_recompute(
    dtype, bs, model, fp8, recipe, fp8_model_params, use_reentrant
):
    if fp8 and not fp8_available:
        pytest.skip(reason_for_no_fp8)
    if recipe.mxfp8() and not mxfp8_available:
        pytest.skip(reason_for_no_mxfp8)

    config = model_configs[model]

    if not use_reentrant:
        # Non-reentrant checkpoint becomes non-deterministic with bias+GELU fusion
        os.environ["NVTE_BIAS_GELU_NVFUSION"] = "0"

    outputs, names = _test_e2e_full_recompute(
        bs,
        dtype,
        config,
        fp8,
        recipe,
        fp8_model_params,
        recompute=False,
        use_reentrant=use_reentrant,
    )
    outputs_recompute, _ = _test_e2e_full_recompute(
        bs,
        dtype,
        config,
        fp8,
        recipe,
        fp8_model_params,
        recompute=True,
        use_reentrant=use_reentrant,
    )

    if not use_reentrant:
        # Reset bias+GELU fusion flag to avoid contaminating other tests
        del os.environ["NVTE_BIAS_GELU_NVFUSION"]

    # Check that results match
    tols = dtype_tols(dtype)
    if dtype in (torch.float16, torch.bfloat16):
        tols["atol"] = 1e-3
    if fp8 or fp8_model_params:
        tols.update(dict(rtol=0.125, atol=0.0675))
    for i, (ref, test) in enumerate(zip(outputs, outputs_recompute)):
        torch.testing.assert_close(
            test,
            ref,
            msg=f"Mismatch in tensor {i}",
            **tols,
        )


def _test_e2e_checkpointing_get_model(config, dtype):
    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    return TransformerLayer(
        config.hidden_size,
        4 * config.hidden_size,
        config.num_attention_heads,
        layernorm_epsilon=config.eps,
        init_method=init_method,
        output_layer_init_method=output_layer_init_method,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        kv_channels=config.embed,
        apply_residual_connection_post_layernorm=False,
        output_layernorm=False,
        params_dtype=dtype,
        device="cuda",
    )


def _test_e2e_checkpointing(bs, dtype, config, checkpoint=False, steps=10, path="checkpoint.pt"):
    reset_rng_states()

    te_inp_hidden_states = torch.randn(
        (config.seq_len, bs, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    te_inp_hidden_states.retain_grad()

    block = _test_e2e_checkpointing_get_model(config, dtype)

    for _ in range(steps // 2):
        te_out = block(
            te_inp_hidden_states,
            None,
        )
        loss = te_out.sum()
        loss.backward()

    if checkpoint:
        # This process is necessary so that we can start afresh with
        # a new model while erasing all internal state to ensure that
        # loading from a checkpoint gives bitwise identical results.
        # Since gradients are being accumulated, it is important to
        # restore them post loading the checkpoint.
        torch.save(block.state_dict(), path)

        param_grads = []
        for p in block.parameters():
            if p.requires_grad:
                param_grads.append(p.grad.clone())

        global _cpu_rng_state, _cuda_rng_state
        _cpu_rng_state = torch.get_rng_state()
        _cuda_rng_state = torch.cuda.get_rng_state()

        del block
        block = _test_e2e_checkpointing_get_model(config, dtype)
        block.load_state_dict(torch.load(path, weights_only=False))
        reset_rng_states()

        for p in block.parameters():
            if p.requires_grad:
                p.grad = param_grads.pop(0)

        assert not param_grads, "Oops!"

    for _ in range(steps // 2):
        te_out = block(
            te_inp_hidden_states,
            None,
        )
        loss = te_out.sum()
        loss.backward()

    torch.cuda.synchronize()

    if os.path.exists(path):
        os.remove(path)

    outputs = [te_out, te_inp_hidden_states.grad]
    for p in block.parameters():
        if p.requires_grad:
            outputs.append(p.grad)
    return outputs


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["126m"])
def test_gpt_checkpointing(dtype, bs, model):
    config = model_configs[model]
    outputs = _test_e2e_checkpointing(bs, dtype, config, checkpoint=False)
    outputs_checkpoint = _test_e2e_checkpointing(bs, dtype, config, checkpoint=True)

    # Check that results match
    tols = dtype_tols(dtype)
    if dtype in (torch.float16, torch.bfloat16):
        tols.update(dict(rtol=2e-2, atol=2e-3))
    for i, (ref, test) in enumerate(zip(outputs, outputs_checkpoint)):
        torch.testing.assert_close(
            test,
            ref,
            msg=f"Mismatch in tensor {i}",
            **tols,
        )


def _test_e2e_gpt_accuracy(block, bs, dtype, config):
    reset_rng_states()

    inp_hidden_states = torch.randn(
        (config.seq_len, bs, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    inp_hidden_states.retain_grad()
    inp_attn_mask = get_causal_attn_mask(config.seq_len)

    out = block(inp_hidden_states, attention_mask=inp_attn_mask)
    loss = out.sum()
    loss.backward()

    torch.cuda.synchronize()
    outputs = [out, inp_hidden_states.grad]
    for p in block.parameters():
        if p.requires_grad:
            outputs.append(p.grad)
    return outputs


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["small"])
@pytest.mark.parametrize("parallel_attention_mlp", all_boolean)
def test_gpt_accuracy(dtype, bs, model, parallel_attention_mlp):
    config = model_configs[model]

    te_gpt = TransformerLayer(
        hidden_size=config.hidden_size,
        ffn_hidden_size=4 * config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        layernorm_epsilon=config.eps,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        params_dtype=dtype,
        fuse_qkv_params=True,
        qkv_weight_interleaved=False,
        parallel_attention_mlp=parallel_attention_mlp,
        device="cuda",
    ).eval()

    torch_gpt = (
        TorchGPT(
            config.hidden_size,
            config.eps,
            config.num_attention_heads,
            parallel_attention_mlp=parallel_attention_mlp,
        )
        .to(dtype=dtype)
        .cuda()
        .eval()
    )

    # Share params
    with torch.no_grad():
        torch_gpt.ln.weight = Parameter(
            te_gpt.self_attention.layernorm_qkv.layer_norm_weight.clone()
        )
        torch_gpt.ln.bias = Parameter(te_gpt.self_attention.layernorm_qkv.layer_norm_bias.clone())
        torch_gpt.causal_attn.mhsa.in_proj_weight = Parameter(
            te_gpt.self_attention.layernorm_qkv.weight.clone()
        )
        torch_gpt.causal_attn.mhsa.in_proj_bias = Parameter(
            te_gpt.self_attention.layernorm_qkv.bias.clone()
        )
        torch_gpt.causal_attn.mhsa.out_proj.weight = Parameter(
            te_gpt.self_attention.proj.weight.clone()
        )
        torch_gpt.causal_attn.mhsa.out_proj.bias = Parameter(
            te_gpt.self_attention.proj.bias.clone()
        )
        torch_gpt.ln_mlp.ln.weight = Parameter(te_gpt.layernorm_mlp.layer_norm_weight.clone())
        torch_gpt.ln_mlp.ln.bias = Parameter(te_gpt.layernorm_mlp.layer_norm_bias.clone())
        torch_gpt.ln_mlp.fc1.weight = Parameter(te_gpt.layernorm_mlp.fc1_weight.clone())
        torch_gpt.ln_mlp.fc1.bias = Parameter(te_gpt.layernorm_mlp.fc1_bias.clone())
        torch_gpt.ln_mlp.fc2.weight = Parameter(te_gpt.layernorm_mlp.fc2_weight.clone())
        torch_gpt.ln_mlp.fc2.bias = Parameter(te_gpt.layernorm_mlp.fc2_bias.clone())

    te_outputs = _test_e2e_gpt_accuracy(te_gpt, bs, dtype, config)
    torch_outputs = _test_e2e_gpt_accuracy(torch_gpt, bs, dtype, config)

    atol = {
        torch.float32: 5e-3,
        torch.half: 5e-2,
        torch.bfloat16: 1e-1,
    }

    # Check output.
    assert_allclose(te_outputs[0], torch_outputs[0], atol[dtype])

    # Check gradients, only for small model
    if model == "small":
        atol[torch.float32] = 5e-2
        rtol = {
            torch.float32: 1e-2,
            torch.half: 1e-2,
            torch.bfloat16: 1e-2,
        }
        for te_output, torch_output in zip(te_outputs[1:], torch_outputs[1:]):
            assert_allclose(te_output, torch_output, atol[dtype], rtol[dtype])


def _test_mha_accuracy(block, bs, dtype, config, mask_type, te=True):
    reset_rng_states()

    inp_hidden_states = torch.randn(
        (config.seq_len, bs, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    inp_hidden_states.retain_grad()
    inp_attn_mask = get_causal_attn_mask(config.seq_len) if mask_type == "causal" else None

    forward_kwargs = {}
    if te:
        forward_kwargs["attn_mask_type"] = mask_type
    forward_kwargs["attention_mask"] = inp_attn_mask

    out = block(inp_hidden_states, **forward_kwargs)
    loss = out.sum()
    loss.backward()

    torch.cuda.synchronize()
    outputs = [out, inp_hidden_states.grad]
    for p in block.parameters():
        if p.requires_grad:
            outputs.append(p.grad)
    return outputs


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["small"])
@pytest.mark.parametrize("mask_type", mask_types)
def test_mha_accuracy(dtype, bs, model, mask_type):
    config = model_configs[model]

    te_mha = MultiheadAttention(
        config.hidden_size,
        config.num_attention_heads,
        fuse_qkv_params=True,
        params_dtype=dtype,
        qkv_weight_interleaved=False,
        input_layernorm=False,
        device="cuda",
    ).eval()

    torch_mha = (
        TorchMHA(
            config.hidden_size,
            config.num_attention_heads,
        )
        .to(dtype=dtype)
        .cuda()
        .eval()
    )

    # Share params
    with torch.no_grad():
        torch_mha.mhsa.in_proj_weight = Parameter(te_mha.qkv.weight.clone())
        torch_mha.mhsa.in_proj_bias = Parameter(te_mha.qkv.bias.clone())
        torch_mha.mhsa.out_proj.weight = Parameter(te_mha.proj.weight.clone())
        torch_mha.mhsa.out_proj.bias = Parameter(te_mha.proj.bias.clone())

    te_outputs = _test_mha_accuracy(te_mha, bs, dtype, config, mask_type, te=True)
    torch_outputs = _test_mha_accuracy(torch_mha, bs, dtype, config, mask_type, te=False)

    # Check output.
    if dtype == torch.float32:
        assert_allclose(te_outputs[0], torch_outputs[0], 5e-3)
    else:
        assert_allclose(te_outputs[0], torch_outputs[0], 5e-2)

    # Check gradients, only for small model
    if model == "small":
        atol = {
            torch.float32: 5e-2,
            torch.half: 5e-2,
            torch.bfloat16: 5e-2,
        }
        rtol = {
            torch.float32: 1e-2,
            torch.half: 1e-2,
            torch.bfloat16: 1e-2,
        }
        for te_output, torch_output in zip(te_outputs[1:], torch_outputs[1:]):
            assert_allclose(te_output, torch_output, atol[dtype], rtol[dtype])


def _test_granular_accuracy(block, bs, dtype, config):
    reset_rng_states()

    inp_hidden_states = torch.randn(
        (config.seq_len, bs, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    inp_hidden_states.retain_grad()

    out = block(inp_hidden_states)
    loss = out.sum()
    loss.backward()

    torch.cuda.synchronize()
    outputs = [out, inp_hidden_states.grad]
    for p in block.parameters():
        if p.requires_grad:
            outputs.append(p.grad)
    return outputs


def _test_dpa_accuracy(block, bs, dtype, config):
    reset_rng_states()

    mask = torch.triu(
        torch.ones(config.seq_len, config.seq_len, dtype=torch.bool, device="cuda"), diagonal=1
    )
    query, key, value = [
        torch.randn(
            (config.seq_len, bs, config.num_attention_heads, config.embed),
            dtype=dtype,
            device="cuda",
            requires_grad=True,
        )
        for _ in range(3)
    ]

    query.retain_grad()
    key.retain_grad()
    value.retain_grad()

    out = block(query, key, value, attention_mask=mask)
    loss = out.sum()
    loss.backward()

    torch.cuda.synchronize()

    return [out, query.grad, key.grad, value.grad]


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["126m"])
def test_dpa_accuracy(dtype, bs, model):
    config = model_configs[model]

    te_dpa = (
        DotProductAttention(
            config.num_attention_heads,
            config.embed,
            attention_dropout=0.0,  # disable dropout, FU uses rng differently
        )
        .to(dtype=dtype)
        .cuda()
    )

    torch_dpa = (
        TorchDotProductAttention(
            config.embed,
            0.0,  # dropout
        )
        .to(dtype=dtype)
        .cuda()
    )

    te_outputs = _test_dpa_accuracy(te_dpa, bs, dtype, config)
    torch_outputs = _test_dpa_accuracy(torch_dpa, bs, dtype, config)

    # Check output.
    if dtype == torch.float32:
        assert_allclose(te_outputs[0], torch_outputs[0], 5e-3)
    else:
        assert_allclose(te_outputs[0], torch_outputs[0], 5e-2)

    for te_output, torch_output in zip(te_outputs[1:], torch_outputs[1:]):
        assert_allclose(te_output, torch_output, atol=5e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["small"])
def test_linear_accuracy(dtype, bs, model):
    config = model_configs[model]

    te_linear = Linear(
        config.hidden_size,
        4 * config.hidden_size,
        bias=True,
        params_dtype=dtype,
        device="cuda",
    ).eval()

    torch_linear = torch.nn.Linear(
        config.hidden_size,
        4 * config.hidden_size,
        bias=True,
        device="cuda",
        dtype=dtype,
    ).eval()

    # Share params
    with torch.no_grad():
        torch_linear.weight = Parameter(te_linear.weight.clone())
        torch_linear.bias = Parameter(te_linear.bias.clone())

    te_outputs = _test_granular_accuracy(te_linear, bs, dtype, config)
    torch_outputs = _test_granular_accuracy(torch_linear, bs, dtype, config)

    # Check output.
    if model == "small":
        tolerance = 5e-3 if dtype == torch.float32 else 5e-2
        rtol = {
            torch.float32: 1.3e-6,
            torch.half: 1e-2,
            torch.bfloat16: 2e-2,
        }
        for te_output, torch_output in zip(te_outputs, torch_outputs):
            assert_allclose(te_output, torch_output, tolerance, rtol[dtype])


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["126m"])
@pytest.mark.parametrize("eps", [1e-1, 1e-3, 1e-5, 1e-7])
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
def test_rmsnorm_accuracy(dtype, bs, model, eps, zero_centered_gamma):
    config = model_configs[model]

    te_rmsnorm = RMSNorm(
        config.hidden_size,
        eps=eps,
        params_dtype=dtype,
        zero_centered_gamma=zero_centered_gamma,
        device="cuda",
    ).eval()

    torch_rmsnorm = (
        TorchRMSNorm(config.hidden_size, eps=eps, zero_centered_gamma=zero_centered_gamma)
        .to(dtype=dtype)
        .cuda()
        .eval()
    )

    # Share params
    with torch.no_grad():
        torch_rmsnorm.weight = Parameter(te_rmsnorm.weight.clone())

    te_outputs = _test_granular_accuracy(te_rmsnorm, bs, dtype, config)
    torch_outputs = _test_granular_accuracy(torch_rmsnorm, bs, dtype, config)

    atol = {
        torch.float32: 1e-7,
        torch.half: 2e-3,
        torch.bfloat16: 2e-2,
    }

    # Check output.
    assert_allclose(te_outputs[0], torch_outputs[0], atol[dtype])

    atol[torch.float32] = 2e-3
    rtol = {
        torch.float32: 1.3e-6,
        torch.half: 1e-3,
        torch.bfloat16: 1.6e-2,
    }
    # Check gradients
    for te_output, torch_output in zip(te_outputs[1:], torch_outputs[1:]):
        assert_allclose(te_output, torch_output, atol[dtype], rtol[dtype])


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["126m"])
@pytest.mark.parametrize("eps", [1e-1, 1e-3, 1e-5, 1e-7])
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
def test_layernorm_accuracy(dtype, bs, model, eps, zero_centered_gamma):
    config = model_configs[model]

    te_layernorm = LayerNorm(
        config.hidden_size,
        eps=eps,
        params_dtype=dtype,
        zero_centered_gamma=zero_centered_gamma,
        device="cuda",
    ).eval()

    torch_layernorm = (
        TorchLayerNorm(config.hidden_size, eps=eps, zero_centered_gamma=zero_centered_gamma)
        .to(dtype=dtype)
        .cuda()
        .eval()
    )

    # Share params
    with torch.no_grad():
        torch_layernorm.weight = Parameter(te_layernorm.weight.clone())
        torch_layernorm.bias = Parameter(te_layernorm.bias.clone())

    te_outputs = _test_granular_accuracy(te_layernorm, bs, dtype, config)
    torch_outputs = _test_granular_accuracy(torch_layernorm, bs, dtype, config)

    atol = {
        torch.float32: 1e-7,
        torch.half: 2e-3,
        torch.bfloat16: 2e-2,
    }

    # Check output.
    assert_allclose(te_outputs[0], torch_outputs[0], atol[dtype])

    rtol = {
        torch.float32: 1.3e-6,
        torch.half: 1e-3,
        torch.bfloat16: 1.6e-2,
    }
    atol[torch.float32] = 1e-4
    # Check gradients
    for te_output, torch_output in zip(te_outputs[1:], torch_outputs[1:]):
        assert_allclose(te_output, torch_output, atol[dtype], rtol[dtype])


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["small"])
@pytest.mark.parametrize("normalization", all_normalizations)
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
def test_layernorm_linear_accuracy(dtype, bs, model, normalization, zero_centered_gamma):
    config = model_configs[model]

    te_ln_linear = LayerNormLinear(
        config.hidden_size,
        4 * config.hidden_size,
        config.eps,
        bias=True,
        normalization=normalization,
        params_dtype=dtype,
        zero_centered_gamma=zero_centered_gamma,
        device="cuda",
    ).eval()

    torch_ln_linear = (
        TorchLayerNormLinear(
            config.hidden_size,
            4 * config.hidden_size,
            config.eps,
            bias=True,
            normalization=normalization,
            zero_centered_gamma=zero_centered_gamma,
        )
        .to(dtype=dtype)
        .cuda()
        .eval()
    )

    # Share params
    with torch.no_grad():
        torch_ln_linear.layernorm.weight = Parameter(te_ln_linear.layer_norm_weight.clone())
        if normalization != "RMSNorm":
            torch_ln_linear.layernorm.bias = Parameter(te_ln_linear.layer_norm_bias.clone())
        torch_ln_linear.linear.weight = Parameter(te_ln_linear.weight.clone())
        torch_ln_linear.linear.bias = Parameter(te_ln_linear.bias.clone())

    te_outputs = _test_granular_accuracy(te_ln_linear, bs, dtype, config)
    torch_outputs = _test_granular_accuracy(torch_ln_linear, bs, dtype, config)

    atol = {
        torch.float32: 2.5e-4,
        torch.half: 2e-3,
        torch.bfloat16: 2e-2,
    }
    rtol = {
        torch.float32: 1e-3,
        torch.half: 4e-2,
        torch.bfloat16: 4e-2,
    }

    # Check output.
    assert_allclose(te_outputs[0], torch_outputs[0], atol[dtype], rtol[dtype])

    if model == "small":
        atol = {
            torch.float32: 1e-3,
            torch.half: 5e-2,
            torch.bfloat16: 5e-2,
        }
        rtol = {
            torch.float32: 1e-3,
            torch.half: 4e-2,
            torch.bfloat16: 4e-2,
        }
        # Check gradients
        for te_output, torch_output in zip(te_outputs[1:], torch_outputs[1:]):
            assert_allclose(te_output, torch_output, atol[dtype], rtol[dtype])


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["small"])
@pytest.mark.parametrize("activation", all_activations)
@pytest.mark.parametrize("normalization", all_normalizations)
def test_layernorm_mlp_accuracy(dtype, bs, model, activation, normalization):
    config = model_configs[model]

    te_ln_mlp = LayerNormMLP(
        config.hidden_size,
        4 * config.hidden_size,
        activation=activation,
        normalization=normalization,
        params_dtype=dtype,
        device="cuda",
    ).eval()

    torch_ln_mlp = (
        TorchLayerNormMLP(
            config.hidden_size,
            4 * config.hidden_size,
            activation=activation,
            normalization=normalization,
        )
        .to(dtype=dtype)
        .cuda()
        .eval()
    )

    # Share params
    with torch.no_grad():
        torch_ln_mlp.ln.weight = Parameter(te_ln_mlp.layer_norm_weight.clone())
        if normalization != "RMSNorm":
            torch_ln_mlp.ln.bias = Parameter(te_ln_mlp.layer_norm_bias.clone())
        torch_ln_mlp.fc1.weight = Parameter(te_ln_mlp.fc1_weight.clone())
        torch_ln_mlp.fc1.bias = Parameter(te_ln_mlp.fc1_bias.clone())
        torch_ln_mlp.fc2.weight = Parameter(te_ln_mlp.fc2_weight.clone())
        torch_ln_mlp.fc2.bias = Parameter(te_ln_mlp.fc2_bias.clone())

    te_outputs = _test_granular_accuracy(te_ln_mlp, bs, dtype, config)
    torch_outputs = _test_granular_accuracy(torch_ln_mlp, bs, dtype, config)

    atol = {
        torch.float32: 2e-2,
        torch.half: 5e-2,
        torch.bfloat16: 5e-2,
    }

    rtol = {
        torch.float32: 1e-3,
        torch.half: 4e-2,
        torch.bfloat16: 4e-2,
    }

    # Check output.
    assert_allclose(te_outputs[0], torch_outputs[0], atol[dtype], rtol[dtype])

    # Check gradients, only for small model
    rtol = {
        torch.float32: 1e-3,
        torch.half: 1e-2,
        torch.bfloat16: 4e-2,
    }
    atol[torch.half] = 2e-1
    atol[torch.bfloat16] = 2e-1
    if model == "small":
        for te_output, torch_output in zip(te_outputs[1:], torch_outputs[1:]):
            assert_allclose(te_output, torch_output, atol[dtype], rtol[dtype])


def _test_grouped_linear_accuracy(block, num_gemms, bs, dtype, config, recipe, fp8=False):
    reset_rng_states()
    if fp8:
        FP8GlobalStateManager.reset()

    inp_hidden_states = torch.randn(
        (config.seq_len, bs, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    inp_hidden_states.retain_grad()

    if num_gemms > 1:
        split_size = 1
        if fp8:
            if recipe.delayed():
                split_size = 16
            if recipe.mxfp8():
                split_size = 128
        m = config.seq_len // split_size
        dist = torch.sort(torch.randint(0, m, (num_gemms - 2,))).values.tolist()
        dist.append(dist[-1])  # Manually add a zero
        m_splits = torch.tensor(dist + [m]) - torch.tensor([0] + dist)
        m_splits = m_splits * split_size
        assert m_splits.sum() == config.seq_len and len(m_splits) == num_gemms
    else:
        m_splits = torch.tensor([config.seq_len])

    with fp8_autocast(enabled=fp8, fp8_recipe=recipe):
        if isinstance(block, GroupedLinear):
            m_splits = m_splits * bs
            out = block(inp_hidden_states, m_splits.tolist())
        else:
            out = torch.cat(
                [
                    block[i](inp)
                    for i, inp in enumerate(torch.split(inp_hidden_states, m_splits.tolist()))
                ]
            )
    loss = out.sum()
    loss.backward()

    torch.cuda.synchronize()
    outputs = [out, inp_hidden_states.grad]
    for p in block.parameters():
        if p.requires_grad:
            outputs.append(p.grad)
    return outputs


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("num_gemms", [3, 6])
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["126m"])
@pytest.mark.parametrize("fp8", all_boolean)
@pytest.mark.parametrize("recipe", fp8_recipes)
@pytest.mark.parametrize("fp8_model_params", all_boolean)
def test_grouped_linear_accuracy(
    dtype, num_gemms, bs, model, fp8, recipe, fp8_model_params, parallel_mode=None
):
    if fp8 and not fp8_available:
        pytest.skip(reason_for_no_fp8)
    if recipe.mxfp8() and not mxfp8_available:
        pytest.skip(reason_for_no_mxfp8)
    if fp8 and recipe.mxfp8():  # TODO(ksivamani): debug mismatches
        pytest.skip("MXFP8 unsupported for grouped linear.")

    config = model_configs[model]
    if config.seq_len % 16 != 0 and fp8:
        pytest.skip("FP8 requires sequence length to be divisible by 16.")

    with fp8_model_init(enabled=fp8 and fp8_model_params, recipe=recipe):
        grouped_linear = GroupedLinear(
            num_gemms,
            config.hidden_size,
            4 * config.hidden_size,
            bias=True,
            params_dtype=dtype,
            parallel_mode=parallel_mode,
            device="cuda",
        ).eval()
        sequential_linear = torch.nn.ModuleList(
            [
                Linear(
                    config.hidden_size,
                    4 * config.hidden_size,
                    bias=True,
                    params_dtype=dtype,
                    parallel_mode=parallel_mode,
                    device="cuda",
                ).eval()
                for _ in range(num_gemms)
            ]
        )

    # Share params
    with torch.no_grad():
        for i in range(num_gemms):
            sequential_linear[i].weight = Parameter(getattr(grouped_linear, f"weight{i}").clone())
            sequential_linear[i].bias = Parameter(getattr(grouped_linear, f"bias{i}").clone())

    outputs_ref = _test_grouped_linear_accuracy(
        sequential_linear, num_gemms, bs, dtype, config, recipe, fp8
    )
    outputs = _test_grouped_linear_accuracy(
        grouped_linear, num_gemms, bs, dtype, config, recipe, fp8
    )

    # Shoule be bit-wise match
    for i, (o, o_ref) in enumerate(zip(outputs, outputs_ref)):
        torch.testing.assert_close(o, o_ref, rtol=0, atol=0)


@pytest.mark.parametrize("parallel_mode", ["column", "row"])
@pytest.mark.parametrize("recipe", fp8_recipes)
def test_grouped_linear_accuracy_parallel_mode(parallel_mode, recipe):
    """Split the tests to save CI time"""
    test_grouped_linear_accuracy(
        dtype=torch.float32,
        num_gemms=6,
        bs=2,
        model="126m",
        fp8=True,
        recipe=recipe,
        fp8_model_params=True,
        parallel_mode=parallel_mode,
    )


@pytest.mark.parametrize("recipe", fp8_recipes)
def test_grouped_linear_accuracy_single_gemm(recipe):
    """Split the tests to save CI time"""
    test_grouped_linear_accuracy(
        dtype=torch.float32,
        num_gemms=1,
        bs=2,
        model="126m",
        fp8=True,
        recipe=recipe,
        fp8_model_params=True,
    )


def _test_padding_grouped_linear_accuracy(block, num_gemms, bs, dtype, config, recipe, fp8=False):

    def _pad_tensor_for_fp8(hidden_states, tokens_per_expert):
        """Padding tensor shapes to multiples of 16."""
        padded_tokens_per_expert = [
            (num_tokens + 15) // 16 * 16 for num_tokens in tokens_per_expert
        ]
        hidden_states = torch.split(hidden_states, tokens_per_expert)
        padded_hidden_states = []
        for hidden_state, actual_num_tokens, padded_num_tokens in zip(
            hidden_states, tokens_per_expert, padded_tokens_per_expert
        ):
            padded_hidden_states.append(hidden_state)
            if padded_num_tokens > actual_num_tokens:
                pad_tensor = torch.zeros(
                    padded_num_tokens - actual_num_tokens,
                    hidden_state.shape[1],
                    dtype=hidden_state.dtype,
                    device=hidden_state.device,
                )
                padded_hidden_states.append(pad_tensor)
        padded_hidden_states = torch.cat(padded_hidden_states, dim=0)
        return padded_hidden_states, padded_tokens_per_expert

    def _unpad_tensor_for_fp8(padded_hidden_states, actual_tokens_per_expert, tokens_per_expert):
        inputmats = torch.split(
            padded_hidden_states.view(-1, padded_hidden_states.shape[-1]), tokens_per_expert
        )
        hidden_states = torch.cat(
            [
                grad_output_mat[: actual_tokens_per_expert[i]]
                for i, grad_output_mat in enumerate(inputmats)
            ],
            dim=0,
        )

        return hidden_states

    def _generate_random_numbers(n, total_sum):
        if n <= 0:
            return []

        # reset seed
        random.seed(seed)

        breaks = sorted(random.sample(range(1, total_sum), n - 1))
        random_numbers = (
            [breaks[0]]
            + [breaks[i] - breaks[i - 1] for i in range(1, n - 1)]
            + [total_sum - breaks[-1]]
        )

        return random_numbers

    reset_rng_states()
    if fp8:
        FP8GlobalStateManager.reset()

    inp_hidden_states = torch.randn(
        (config.seq_len * bs, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    inp_hidden_states.retain_grad()

    m_splits = _generate_random_numbers(num_gemms, config.seq_len * bs)

    with fp8_autocast(enabled=fp8, fp8_recipe=recipe):
        if isinstance(block, TorchGroupedLinearWithPadding):
            out = block(inp_hidden_states, m_splits)
        else:
            if fp8:
                padded_inp_hidden_states, padding_m_splits = _pad_tensor_for_fp8(
                    inp_hidden_states, m_splits
                )
                padded_inp_hidden_states = block(padded_inp_hidden_states, padding_m_splits)
                out = _unpad_tensor_for_fp8(padded_inp_hidden_states, m_splits, padding_m_splits)
            else:
                out = block(inp_hidden_states, m_splits)

    loss = out.sum()
    loss.backward()

    torch.cuda.synchronize()
    outputs = [out, inp_hidden_states.grad]
    for p in block.parameters():
        if p.requires_grad:
            outputs.append(p.grad)
    return outputs


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("num_gemms", [3, 6])
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["126m"])
@pytest.mark.parametrize("fp8", [True])
@pytest.mark.parametrize("recipe", fp8_recipes)
@pytest.mark.parametrize("fp8_model_params", all_boolean)
def test_padding_grouped_linear_accuracy(
    dtype, num_gemms, bs, model, fp8, recipe, fp8_model_params, parallel_mode=None
):
    if fp8 and not fp8_available:
        pytest.skip(reason_for_no_fp8)
    if recipe.mxfp8() and not mxfp8_available:
        pytest.skip(reason_for_no_mxfp8)
    if fp8 and recipe.mxfp8():  # TODO(ksivamani): debug mismatches
        pytest.skip("MXFP8 unsupported for grouped linear.")

    config = model_configs[model]
    if config.seq_len % 16 != 0 and fp8:
        pytest.skip("FP8 requires sequence length to be divisible by 16.")

    with fp8_model_init(enabled=fp8 and fp8_model_params, recipe=recipe):
        grouped_linear = TorchGroupedLinearWithPadding(
            num_gemms,
            config.hidden_size,
            4 * config.hidden_size,
            bias=False,
            params_dtype=dtype,
            parallel_mode=parallel_mode,
            fp8=fp8,
        ).eval()

    with fp8_model_init(enabled=fp8 and fp8_model_params, recipe=recipe):
        ref_grouped_linear = GroupedLinear(
            num_gemms,
            config.hidden_size,
            4 * config.hidden_size,
            bias=False,
            params_dtype=dtype,
            parallel_mode=parallel_mode,
            device="cuda",
        ).eval()

    # Share params
    with torch.no_grad():
        inner_grouped_linear = grouped_linear.linear_fn
        for i in range(num_gemms):
            setattr(
                ref_grouped_linear,
                f"weight{i}",
                Parameter(getattr(inner_grouped_linear, f"weight{i}").clone()),
            )

    outputs = _test_padding_grouped_linear_accuracy(
        grouped_linear, num_gemms, bs, dtype, config, recipe, fp8
    )
    outputs_ref = _test_padding_grouped_linear_accuracy(
        ref_grouped_linear, num_gemms, bs, dtype, config, recipe, fp8
    )

    # Shoule be bit-wise match
    for i, (o, o_ref) in enumerate(zip(outputs, outputs_ref)):
        torch.testing.assert_close(o, o_ref, rtol=0, atol=0)


def _test_gpt_e2e_cuda_graph(block, bs, dtype, config, graph):
    reset_rng_states()

    # Initialize loss function and optimizer.
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(block.parameters(), lr=0.1)

    # Placeholders used for graph capture.
    static_input = torch.randn(
        config.seq_len, bs, config.hidden_size, device="cuda", dtype=dtype, requires_grad=True
    )
    static_target = torch.randn(config.seq_len, bs, config.hidden_size, device="cuda", dtype=dtype)

    real_input = torch.rand_like(static_input)
    real_target = torch.rand_like(static_target)

    # Basic training loop.
    def train_step():
        optimizer.zero_grad(set_to_none=False)
        out = block(static_input)
        loss = loss_fn(out, static_target)
        loss.backward()
        optimizer.step()
        return out

    # Warmup steps in a separate stream.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            train_step()
    torch.cuda.current_stream().wait_stream(s)

    # Capture graph.
    g = None
    static_output = None
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_output = train_step()

    # Run with new data.
    with torch.no_grad():
        static_input.copy_(real_input)
        static_target.copy_(real_target)
    if graph:
        g.replay()
    else:
        static_output = train_step()

    grads = [static_input.grad]
    for p in block.parameters():
        if p.requires_grad:
            grads.append(p.grad)

    with torch.no_grad():
        output = static_output.clone()
    return output, grads


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["126m"])
def test_gpt_cuda_graph(dtype, bs, model):
    config = model_configs[model]

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    block_args = (
        config.hidden_size,
        4 * config.hidden_size,
        config.num_attention_heads,
    )
    block_kwargs = dict(
        layernorm_epsilon=config.eps,
        init_method=init_method,
        output_layer_init_method=output_layer_init_method,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        kv_channels=config.embed,
        params_dtype=dtype,
        apply_residual_connection_post_layernorm=False,
        output_layernorm=False,
        device="cuda",
    )
    block = TransformerLayer(*block_args, **block_kwargs)
    graphed_block = TransformerLayer(*block_args, **block_kwargs)
    with torch.no_grad():
        for param1, param2 in zip(block.parameters(), graphed_block.parameters()):
            param2.copy_(param1)

    out, grads = _test_gpt_e2e_cuda_graph(block, bs, dtype, config, False)
    graphed_out, graphed_grads = _test_gpt_e2e_cuda_graph(graphed_block, bs, dtype, config, True)
    params = list(block.parameters())
    graphed_params = list(graphed_block.parameters())

    # Check that results match
    assert_allclose(out, graphed_out, 1e-3)
    assert_allclose(params, graphed_params, 1e-3)
    assert_allclose(grads, graphed_grads, 1e-3)


def _test_gpt_fp8_parameters(bs, dtype, config, fp8_model_params, recipe):
    reset_rng_states()
    FP8GlobalStateManager.reset()

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    with fp8_model_init(enabled=fp8_model_params, recipe=recipe):
        block = TransformerLayer(
            config.hidden_size,
            4 * config.hidden_size,
            config.num_attention_heads,
            layernorm_epsilon=config.eps,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            hidden_dropout=0.1,
            attention_dropout=0.1,
            kv_channels=config.embed,
            apply_residual_connection_post_layernorm=False,
            output_layernorm=False,
            params_dtype=dtype,
            fuse_qkv_params=True,
            device="cuda",
        )

    te_inp_hidden_states = torch.randn(
        (config.seq_len, bs, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    te_inp_hidden_states.retain_grad()
    te_inp_attn_mask = get_causal_attn_mask(config.seq_len)

    with fp8_autocast(enabled=True, fp8_recipe=recipe):
        te_out = block(te_inp_hidden_states, attention_mask=te_inp_attn_mask)
    loss = te_out.sum()
    loss.backward()
    torch.cuda.synchronize()

    outputs = [te_out, te_inp_hidden_states.grad]
    for p in block.parameters():
        if p.requires_grad:
            outputs.append(p.grad)
    return outputs


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["126m"])
@pytest.mark.parametrize("recipe", fp8_recipes)
def test_gpt_fp8_parameters(dtype, bs, model, recipe):
    if not fp8_available:
        pytest.skip(reason_for_no_fp8)
    if recipe.mxfp8() and not mxfp8_available:
        pytest.skip(reason_for_no_mxfp8)

    config = model_configs[model]

    outputs = _test_gpt_fp8_parameters(bs, dtype, config, False, recipe)
    outputs_fp8_params = _test_gpt_fp8_parameters(bs, dtype, config, True, recipe)

    # Check that results match
    tols = dict(rtol=0.125, atol=0.0675)
    for i, (ref, test) in enumerate(zip(outputs, outputs_fp8_params)):
        torch.testing.assert_close(
            test,
            ref,
            msg=f"Mismatch in tensor {i}",
            rtol=0.125,
            atol=0.0675,
        )


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["126m"])
def test_transformer_layer_hidden_states_format(dtype, bs, model):
    config = model_configs[model]

    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    # Set `torch.manual_seed` to make sure the weights are identical to the
    # other layer. Set `*dropout` values to 0 to make sure the forward pass
    # is identical to the other layer.
    torch.manual_seed(0)
    block_sbhd = TransformerLayer(
        config.hidden_size,
        4 * config.hidden_size,
        config.num_attention_heads,
        layernorm_epsilon=config.eps,
        init_method=init_method,
        output_layer_init_method=output_layer_init_method,
        hidden_dropout=0,
        attention_dropout=0,
        kv_channels=config.embed,
        params_dtype=dtype,
        apply_residual_connection_post_layernorm=False,
        output_layernorm=False,
        device="cuda",
        attn_input_format="sbhd",
    )

    # Set `torch.manual_seed` to make sure the weights are identical to the
    # other layer. Set `*dropout` values to 0 to make sure the forward pass
    # is identical to the other layer.
    torch.manual_seed(0)
    block_bshd = TransformerLayer(
        config.hidden_size,
        4 * config.hidden_size,
        config.num_attention_heads,
        layernorm_epsilon=config.eps,
        init_method=init_method,
        output_layer_init_method=output_layer_init_method,
        hidden_dropout=0,
        attention_dropout=0,
        kv_channels=config.embed,
        params_dtype=dtype,
        apply_residual_connection_post_layernorm=False,
        output_layernorm=False,
        device="cuda",
        attn_input_format="bshd",
    )

    torch.manual_seed(0)
    block_thd = TransformerLayer(
        config.hidden_size,
        4 * config.hidden_size,
        config.num_attention_heads,
        layernorm_epsilon=config.eps,
        init_method=init_method,
        output_layer_init_method=output_layer_init_method,
        hidden_dropout=0,
        attention_dropout=0,
        kv_channels=config.embed,
        params_dtype=dtype,
        apply_residual_connection_post_layernorm=False,
        output_layernorm=False,
        device="cuda",
        attn_input_format="thd",
        self_attn_mask_type="padding_causal",
    )

    for (n1, p1), (n2, p2), (n3, p3) in zip(
        block_bshd.named_parameters(), block_sbhd.named_parameters(), block_thd.named_parameters()
    ):
        assert torch.all(torch.eq(p1, p2) & torch.eq(p1, p3)), f"{n1}, {n2} and {n3} not identical"

    x_sbhd = torch.randn(
        (config.seq_len, bs, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )

    x_bshd = x_sbhd.transpose(0, 1).contiguous()
    x_thd = x_bshd.reshape(bs * config.seq_len, config.hidden_size).contiguous()
    x_thd_cumsum = torch.arange(bs + 1, device="cuda", dtype=torch.int32) * config.seq_len

    # To make sure forward is also identical (just in case some module decides
    # to act fancy)
    torch.manual_seed(0)
    y_sbhd = block_sbhd(x_sbhd)

    # To make sure forward is also identical (just in case some module decides
    # to act fancy)
    torch.manual_seed(0)
    y_bshd = block_bshd(x_bshd)

    # Check that results match
    torch.testing.assert_close(
        y_bshd,
        y_sbhd.transpose(0, 1).contiguous(),
    )

    # THD is not supported in float32 and on GPUs older than Ampere, skip the test here
    if dtype != torch.float32 and sm_80plus:
        # To make sure forward is also identical (just in case some module decides
        # to act fancy)
        torch.manual_seed(0)
        y_thd = block_thd(
            x_thd,
            cu_seqlens_q=x_thd_cumsum,
            cu_seqlens_kv=x_thd_cumsum,
            max_seqlen_q=config.seq_len,
            max_seqlen_kv=config.seq_len,
        )

        torch.testing.assert_close(
            y_bshd,
            y_thd.reshape(bs, config.seq_len, config.hidden_size).contiguous(),
        )


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model_key", model_configs_inference.keys())
@pytest.mark.parametrize("use_RoPE", all_boolean)
@pytest.mark.parametrize("input_format", input_formats_inference)
@pytest.mark.parametrize("module", module_inference)
@pytest.mark.parametrize("backend", backends_inference)
def test_kv_cache_accuracy(dtype, bs, model_key, use_RoPE, input_format, module, backend):
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"

    if backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    elif backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"

    config = model_configs_inference[model_key]

    S = config.seq_len
    B = bs
    H = config.num_attention_heads
    D = config.hidden_size
    head_size = config.embed
    layer_number = 1

    # Limits the max size of KV-cache
    B_max = B
    S_max = S + 2

    if module == "TransformerLayer":
        model = TransformerLayer(
            hidden_size=D,
            ffn_hidden_size=4 * D,
            num_attention_heads=H,
            attn_input_format=input_format,
            self_attn_mask_type="causal",
            enc_dec_attn_mask_type="causal",
            layer_number=layer_number,
            attention_dropout=0.0,
            params_dtype=dtype,
            device="cuda",
        ).eval()
    else:
        model = (
            MultiheadAttention(
                hidden_size=D,
                num_attention_heads=H,
                qkv_format=input_format,
                layer_number=layer_number,
                attention_dropout=0.0,
                attn_mask_type="causal",
                params_dtype=dtype,
            )
            .cuda()
            .eval()
        )

    inference_params = InferenceParams(max_batch_size=B_max, max_sequence_length=S_max)
    rotary_freqs = torch.randn((S_max, 1, 1, head_size), dtype=torch.float, device="cuda")

    input = torch.randn((S, B, D), dtype=dtype, device="cuda")
    if input_format == "bshd":
        input = input.transpose(0, 1).contiguous()

    incremental_output = torch.zeros_like(input)

    # Generate output for the entire sequence
    full_output = model(hidden_states=input, rotary_pos_emb=rotary_freqs if use_RoPE else None)

    # Incrementaly generate outputs using KV-cache
    for i in range(S):
        if input_format == "sbhd":
            incremental_input = input[i].view(1, B, D)
        else:
            incremental_input = input[:, i, :].view(B, 1, D)

        line_output = model(
            hidden_states=incremental_input,
            inference_params=inference_params,
            rotary_pos_emb=rotary_freqs if use_RoPE else None,
        )

        inference_params.sequence_len_offset += 1

        if input_format == "sbhd":
            incremental_output[i] = line_output.view(B, D)
        else:
            incremental_output[:, i, :] = line_output.view(B, D)

    if module == "TransformerLayer":
        atol = {
            torch.float32: 5e-3,
            torch.half: 5e-3,
            torch.bfloat16: 5e-2,
        }
    else:
        atol = {
            torch.float32: 1e-3,
            torch.half: 1e-3,
            torch.bfloat16: 1e-2,
        }

    # Check if the fully generated output matches the one generated incrementally
    assert_allclose(full_output, incremental_output, atol[dtype])


@pytest.mark.parametrize(
    "token",
    [
        (941, 117, 828, 821, 620, 362, 608, 625, 242, 631, 517, 711, 283, 644, 1051, 370, 110, 641, 554, 283, 1909, 997, 85, 382, 1176, 1726, 204, 364, 686, 997, 694, 662, 122, 1152, 513, 697, 1101, 833, 914, 1309, 589, 243, 893, 312, 264, 926, 880, 401),
        (886, 386, 1183, 931, 953, 229, 223, 436, 615, 725, 464, 843, 279, 819, 1595, 573, 217, 303, 217, 179, 786, 681, 93, 1020, 617, 1198, 216, 479, 326, 778, 1167, 453, 343, 954, 185, 400, 682, 705, 439, 798, 257, 273, 539, 433, 309, 981, 812, 744),
        (1397, 572, 475, 1311, 817, 265, 258, 343, 420, 979, 1143, 1480, 424, 426, 685, 339, 294, 1197, 1610, 337, 1159, 613, 59, 1114, 610, 544, 740, 658, 733, 345, 769, 199, 158, 736, 81, 644, 1023, 837, 649, 1326, 553, 232, 500, 370, 288, 928, 656, 285),
        (805, 368, 808, 1272, 629, 429, 475, 635, 318, 643, 527, 644, 501, 563, 493, 541, 293, 557, 380, 699, 2123, 343, 105, 518, 889, 1356, 221, 372, 328, 1213, 1029, 299, 299, 1623, 327, 713, 1132, 1394, 1744, 1085, 774, 641, 990, 363, 308, 718, 854, 412),
        (907, 156, 776, 1030, 509, 671, 834, 729, 140, 577, 568, 571, 265, 617, 285, 324, 146, 392, 515, 444, 1371, 488, 148, 275, 1155, 1519, 252, 356, 398, 973, 716, 414, 197, 1528, 512, 350, 811, 742, 520, 935, 881, 660, 678, 776, 256, 956, 830, 514),
        (674, 373, 546, 989, 886, 240, 314, 689, 308, 972, 535, 423, 256, 698, 320, 480, 286, 496, 696, 227, 1724, 420, 143, 416, 1118, 1057, 133, 441, 527, 1094, 1016, 501, 200, 1224, 335, 312, 1748, 1307, 444, 1150, 415, 200, 792, 437, 297, 1462, 928, 236),
        (524, 393, 428, 1614, 982, 399, 803, 763, 508, 622, 570, 653, 225, 492, 524, 751, 262, 558, 491, 312, 1270, 353, 89, 428, 516, 1334, 334, 554, 306, 1092, 1218, 299, 171, 1027, 539, 202, 1734, 660, 1000, 1373, 953, 128, 667, 385, 257, 981, 755, 390),
        (643, 310, 494, 1593, 750, 481, 888, 748, 497, 714, 444, 674, 468, 548, 380, 529, 313, 448, 536, 293, 1652, 506, 197, 525, 657, 914, 211, 419, 441, 584, 638, 370, 157, 1221, 1290, 441, 2135, 828, 416, 1238, 720, 434, 643, 495, 416, 1179, 530, 535),
        (947, 330, 835, 1470, 867, 331, 314, 948, 393, 676, 864, 565, 234, 294, 501, 265, 312, 839, 578, 262, 2041, 604, 113, 619, 1046, 1835, 308, 516, 404, 574, 798, 702, 71, 987, 104, 741, 1182, 765, 638, 1723, 515, 251, 892, 392, 255, 929, 744, 612),
        (596, 447, 745, 1167, 707, 356, 653, 740, 606, 585, 502, 701, 322, 511, 508, 455, 553, 556, 267, 478, 1188, 300, 121, 1031, 375, 1645, 331, 656, 412, 632, 652, 400, 298, 1052, 569, 261, 1294, 1208, 643, 974, 448, 468, 579, 400, 345, 919, 1059, 340),
        (1288, 286, 952, 1333, 600, 232, 292, 840, 311, 779, 1201, 433, 250, 545, 674, 453, 114, 532, 175, 376, 1693, 906, 64, 952, 1213, 2150, 470, 455, 185, 262, 1121, 1824, 343, 899, 83, 832, 1243, 404, 399, 470, 461, 195, 534, 303, 371, 622, 683, 392),
        (835, 188, 2761, 1109, 720, 549, 436, 530, 743, 626, 904, 479, 252, 484, 431, 452, 403, 1587, 433, 344, 1073, 368, 211, 371, 736, 1093, 296, 249, 882, 802, 1648, 503, 169, 786, 540, 595, 1058, 576, 908, 1210, 548, 480, 1557, 459, 161, 671, 652, 548),
        (481, 319, 608, 1147, 778, 394, 445, 906, 188, 1129, 800, 627, 426, 431, 480, 429, 294, 643, 461, 293, 1539, 318, 91, 432, 836, 2341, 249, 483, 233, 1414, 739, 485, 131, 764, 229, 519, 1048, 765, 705, 1330, 699, 390, 957, 461, 466, 1173, 643, 674),
        (877, 346, 626, 1676, 530, 491, 231, 895, 896, 1069, 1005, 823, 280, 582, 397, 601, 114, 707, 1415, 264, 1517, 413, 148, 291, 670, 1292, 304, 356, 699, 777, 929, 735, 122, 614, 158, 827, 1099, 1033, 546, 1267, 630, 249, 996, 500, 151, 1079, 1461, 761),
        (439, 429, 1168, 1185, 463, 180, 307, 870, 298, 861, 519, 577, 324, 366, 307, 597, 129, 461, 1154, 384, 1745, 582, 131, 658, 704, 1247, 2461, 723, 556, 798, 531, 465, 139, 637, 89, 690, 1126, 916, 477, 797, 569, 226, 701, 726, 219, 1123, 1006, 474),
        (726, 308, 814, 1123, 781, 586, 553, 1276, 471, 957, 620, 845, 564, 463, 1197, 430, 365, 879, 291, 354, 1095, 266, 151, 723, 599, 1115, 349, 326, 395, 612, 781, 686, 427, 1514, 268, 387, 2050, 950, 500, 519, 877, 264, 577, 586, 453, 1049, 586, 609),
        (374, 338, 475, 1203, 943, 390, 291, 510, 835, 494, 514, 489, 270, 675, 743, 381, 412, 628, 1082, 269, 2068, 342, 242, 1038, 926, 1182, 156, 379, 359, 816, 525, 400, 302, 767, 329, 395, 1039, 830, 711, 803, 736, 352, 429, 510, 145, 841, 1014, 585),
        (467, 242, 605, 1198, 1086, 1270, 722, 933, 577, 466, 459, 455, 947, 390, 546, 724, 309, 844, 860, 333, 1627, 461, 244, 624, 812, 1238, 229, 711, 366, 472, 622, 323, 111, 607, 69, 1095, 848, 930, 754, 770, 1261, 412, 642, 398, 190, 661, 1589, 825),
        (487, 304, 785, 1435, 923, 588, 621, 1008, 125, 684, 450, 354, 339, 608, 313, 339, 358, 401, 670, 970, 1096, 480, 135, 346, 692, 1153, 190, 320, 985, 710, 913, 416, 66, 1023, 101, 262, 1775, 930, 993, 1659, 455, 367, 760, 303, 699, 891, 503, 464),
        (384, 446, 772, 1577, 1096, 473, 488, 840, 200, 1086, 596, 608, 450, 270, 494, 446, 257, 507, 355, 462, 1182, 576, 177, 848, 560, 1079, 408, 251, 463, 1377, 927, 480, 97, 1357, 290, 599, 1279, 1101, 572, 1171, 835, 445, 1074, 317, 141, 1110, 929, 556),
        (359, 285, 455, 1335, 608, 644, 1046, 1159, 409, 741, 276, 772, 289, 1119, 484, 409, 500, 503, 317, 226, 1814, 589, 187, 441, 673, 1298, 196, 438, 589, 844, 749, 373, 372, 1748, 184, 341, 1317, 609, 336, 511, 711, 247, 581, 556, 274, 1103, 998, 416),
        (912, 395, 625, 1040, 759, 256, 414, 738, 600, 517, 579, 465, 247, 976, 519, 412, 166, 419, 488, 271, 2587, 495, 79, 782, 805, 741, 588, 328, 374, 575, 1190, 239, 217, 1031, 252, 1368, 1263, 1236, 620, 752, 479, 277, 466, 297, 243, 1247, 928, 438),
        (783, 596, 555, 2200, 888, 357, 481, 589, 230, 1787, 340, 887, 415, 655, 509, 560, 453, 826, 460, 384, 2275, 306, 78, 769, 582, 1391, 211, 566, 425, 889, 715, 748, 235, 963, 134, 497, 1385, 993, 462, 1143, 575, 352, 355, 442, 320, 1508, 838, 340),
        (462, 53, 775, 1288, 633, 678, 735, 1662, 298, 496, 603, 533, 501, 410, 264, 472, 206, 465, 550, 388, 1498, 681, 529, 482, 1663, 628, 459, 359, 706, 1121, 583, 378, 398, 1489, 336, 410, 1633, 519, 720, 1077, 879, 220, 927, 292, 167, 821, 631, 576),
        (533, 196, 493, 1481, 945, 449, 477, 620, 315, 957, 659, 935, 476, 371, 560, 366, 204, 554, 1312, 401, 1733, 448, 143, 420, 1496, 1139, 314, 302, 476, 819, 718, 427, 69, 1218, 219, 821, 1273, 829, 579, 983, 960, 328, 586, 430, 246, 904, 1037, 907),
        (736, 236, 341, 962, 541, 368, 250, 958, 217, 798, 1268, 398, 307, 517, 617, 272, 179, 274, 671, 693, 1350, 165, 99, 811, 760, 1033, 363, 373, 522, 796, 672, 695, 116, 585, 93, 1012, 774, 965, 778, 1447, 826, 291, 363, 252, 351, 862, 1681, 409),
        (548, 255, 846, 1382, 559, 495, 746, 940, 455, 514, 583, 857, 166, 515, 815, 266, 252, 421, 568, 465, 1336, 585, 72, 366, 404, 1775, 217, 606, 678, 630, 1456, 457, 186, 741, 651, 319, 880, 912, 449, 919, 619, 254, 1034, 364, 136, 496, 1000, 350),
        (420, 344, 732, 1472, 1145, 386, 678, 568, 365, 806, 715, 747, 451, 464, 349, 385, 291, 532, 676, 306, 1960, 441, 109, 410, 780, 1416, 398, 462, 448, 624, 712, 329, 162, 1111, 188, 508, 1556, 635, 837, 1289, 553, 227, 813, 181, 286, 1247, 889, 499),
        (388, 254, 484, 1047, 702, 593, 598, 868, 254, 798, 780, 481, 315, 349, 542, 455, 151, 275, 458, 317, 1511, 655, 168, 332, 742, 1480, 138, 332, 189, 663, 795, 617, 151, 1032, 225, 452, 1389, 501, 694, 834, 716, 346, 873, 416, 253, 1246, 433, 339),
        (864, 170, 597, 1189, 659, 505, 499, 527, 1274, 466, 525, 754, 385, 503, 1604, 676, 223, 678, 729, 259, 758, 694, 145, 477, 1880, 1066, 220, 323, 141, 1091, 912, 671, 380, 987, 455, 477, 948, 1309, 438, 1179, 427, 455, 662, 375, 239, 695, 1622, 1107),
        (699, 687, 556, 1287, 786, 295, 353, 649, 236, 629, 501, 760, 234, 550, 552, 364, 211, 587, 410, 265, 1241, 475, 144, 1163, 622, 1082, 292, 665, 392, 917, 870, 370, 418, 856, 150, 228, 1160, 666, 453, 716, 589, 324, 455, 435, 386, 651, 1133, 598),
        (411, 453, 492, 1013, 710, 394, 920, 973, 310, 1482, 771, 565, 534, 477, 917, 367, 469, 483, 704, 260, 2074, 325, 180, 490, 1419, 1100, 307, 316, 334, 662, 580, 658, 263, 1113, 218, 616, 1164, 752, 580, 785, 637, 221, 526, 572, 224, 905, 815, 623),
    ],
    ids=lambda token: sum(token),
)
@pytest.mark.parametrize("shape", [(7168, 4096), (2048, 7168)], ids=["fc1", "fc2"])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
# @pytest.mark.parametrize("layout, accumulate", [("TN", False), ("NN", False), ("NT", True)])
@pytest.mark.parametrize("layout, accumulate", [("TN", False), ("NN", False)])
def test_grouped_gemm(token, shape, dtype, layout, accumulate, request):
    k, n = shape
    m_splits = token
    z, m = len(m_splits), sum(m_splits)

    torch.manual_seed(42)
    if layout == "TN":
        A = torch.randn(z, n, k, dtype=dtype, device="musa").unbind() # weight
        # A = [torch.randn(n, k, dtype=dtype, device="musa") for _ in range(z)]  # weight
        B = torch.split(torch.randn(m, k, dtype=dtype, device="musa"), m_splits)  # input
        out = torch.split(torch.randn(m, n, dtype=dtype, device="musa"), m_splits)  # output
        grad = False
    elif layout == "NN":
        A = torch.randn(z, n, k, dtype=dtype, device="musa").unbind() # weight
        # A = [torch.randn(n, k, dtype=dtype, device="musa") for _ in range(z)]  # weight
        B = torch.split(torch.randn(m, n, dtype=dtype, device="musa"), m_splits)  # grad_output
        out = torch.split(torch.randn(m, k, dtype=dtype, device="musa"), m_splits)  # dgrad
        grad = True
    else:  # layout == "NT"
        A = torch.split(torch.randn(m, k, dtype=dtype, device="musa"), m_splits)  # input
        B = torch.split(torch.randn(m, n, dtype=dtype, device="musa"), m_splits)  # grad_output
        out = torch.randn(z, n, k, dtype=dtype, device="musa").unbind()
        # out = [torch.randn(n, k, dtype=dtype, device="musa") for _ in range(z)]  # wgrad
        grad = True

    [nn.init.normal_(a, mean=0.0, std=0.006) for a in A]
    [nn.init.normal_(b, mean=0.0, std=0.006) for b in B]
    [nn.init.normal_(o, mean=0.0, std=0.006) for o in out]

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    out_ref = [o.clone() for o in out]
    for i in range(z):
        general_gemm(
            A[i],
            B[i],
            get_workspace(),
            dtype,
            grad=grad,
            accumulate=accumulate,
            layout=layout,
            out=out_ref[i],
        )

    start.record()
    general_grouped_gemm(
        A,
        list(B),
        list(out),
        dtype,
        get_multi_stream_cublas_workspace(),
        m_splits=m_splits,  # TODO, not sure
        grad=grad,
        accumulate=accumulate,
        layout=layout,
    )
    end.synchronize()
    elapsed_time = start.elapsed_time(end) / 10
    print(f"time: {elapsed_time:.2f}ms, tflops: {2*m*n*k/elapsed_time/1e9:.2f} ", end='')

    # should be bit-wise match
    try:
        for o, o_ref in zip(out, out_ref):
            torch.testing.assert_close(o, o_ref, rtol=0, atol=0)
    except Exception as e:
        save_dir = os.environ.get("TE_TEST_OUTPUT_DIR", "test_outputs")
        test_name = request.node.name.split("test_grouped_gemm")[1].replace("[", "").replace("]", "")
        os.makedirs(save_dir, exist_ok=True)

        data = [token, shape, dtype, layout, accumulate, A, B, out]
        torch.save(data, os.path.join(save_dir, f"{test_name}.pth"))
        raise e



@pytest.mark.parametrize(
    "shape",
    [
        (1, 128, 128, 512),
        (8, 1024, 128, 512),
        (16, 4096, 128, 512),
    ],
)
@pytest.mark.parametrize("fp8_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2])
@pytest.mark.parametrize("accumulate", [False, True])
def test_fp8_grouped_gemm(shape, fp8_dtype, accumulate):
    if not fp8_available:
        pytest.skip(reason_for_no_fp8)

    z, m, k, n = shape
    m_splits = m // z

    dtype = torch.bfloat16
    A = [torch.randn(n, k, dtype=dtype, device="cuda") for _ in range(z)]  # weight
    B = torch.split(torch.randn(m, k, dtype=dtype, device="cuda"), m_splits)  # input
    out = torch.split(torch.randn(m, n, dtype=dtype, device="cuda"), m_splits)  # output
    out_ref = [o.clone() for o in out]

    # fp8 should be robust enough to this fake scale
    scale = 1 + torch.rand(1, dtype=torch.float32, device="cuda").squeeze()
    amax = torch.zeros(1, 1, dtype=torch.float32, device="cuda")

    a_quantizers = [
        Float8Quantizer(
            scale.clone(),
            amax.clone(),
            tex.DType.kFloat8E4M3,
        )
        for _ in range(z)
    ]
    b_quantizers = [
        Float8Quantizer(
            scale.clone(),
            amax.clone(),
            tex.DType.kFloat8E4M3,
        )
        for _ in range(z)
    ]

    A_fp8 = []
    B_fp8 = []

    for i in range(z):
        A_fp8.append(a_quantizers[i](A[i]))
        B_fp8.append(b_quantizers[i](B[i]))

    # baseline
    for i in range(z):
        general_gemm(
            A_fp8[i],
            B_fp8[i],
            get_workspace(),
            dtype,
            out=out_ref[i],
            accumulate=accumulate,
        )
    general_grouped_gemm(
        A_fp8,
        B_fp8,
        out,
        dtype,
        get_multi_stream_cublas_workspace(),
        m_splits=[k] * m_splits,
        accumulate=accumulate,
    )

    # should be bit-wise match
    for o, o_ref in zip(out, out_ref):
        torch.testing.assert_close(o, o_ref, rtol=0, atol=0)


def test_noncontiguous():
    def _create2modules(m, params):
        mod1 = m(*params)
        mod2 = m(*params)
        for p1, p2 in zip(mod1.parameters(), mod2.parameters()):
            p2.data = p1.data.clone()

        return mod1, mod2

    def _run_module(m, inp):
        out = m(inp)
        out.sum().backward()
        ret = [out]
        if inp.grad is not None:
            ret.append(inp.grad)

        for p in m.parameters():
            if p.requires_grad:
                ret.append(p.grad)
        return ret

    a = torch.randn((128, 256), device="cuda", requires_grad=True)
    a = a.T
    assert not a.is_contiguous(), "The test is supposed to test noncontiguous input."

    b = a.contiguous()

    # LayerNorm
    ln1, ln2 = _create2modules(LayerNorm, [128])
    outT = _run_module(ln1, a)
    out = _run_module(ln2, b)

    assert_allclose(out, outT, 1e-7)

    # RMSNorm
    ln1, ln2 = _create2modules(RMSNorm, [128])
    outT = _run_module(ln1, a)
    out = _run_module(ln2, b)

    assert_allclose(out, outT, 1e-7)

    # GEMM
    g1, g2 = _create2modules(Linear, [128, 128])
    outT = _run_module(g1, a)
    out = _run_module(g2, b)

    assert_allclose(out, outT, 1e-7)
