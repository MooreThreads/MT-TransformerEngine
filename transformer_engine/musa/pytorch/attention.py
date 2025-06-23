# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Attention."""
from contextlib import nullcontext
from importlib.metadata import version as get_pkg_version
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging
from packaging.version import Version as PkgVersion

import torch

import transformer_engine_torch as tex
from transformer_engine.pytorch.utils import get_cudnn_version

from transformer_engine.pytorch.fp8 import get_fp8_te_dtype
from transformer_engine.pytorch.float8_tensor import Float8Tensor

from transformer_engine.pytorch.constants import (
    AttnMaskTypes,
    AttnTypes,
    QKVLayouts,
    dist_group_type,
)
from transformer_engine.pytorch.distributed import (
    get_distributed_world_size,
    set_all_rng_states,
    CudaRNGStatesTracker,
    graph_safe_rng_available,
)
from transformer_engine.pytorch.graph import is_graph_capturing
_flash_attn_version = PkgVersion(get_pkg_version("flash-attn"))
_flash_attn_version_required = PkgVersion("2.0.6")
_flash_attn_max_version = PkgVersion("2.6.8")
_flash_attn_2_3_plus = _flash_attn_version >= PkgVersion("2.3")
_flash_attn_2_4_plus = _flash_attn_version >= PkgVersion("2.4")
_flash_attn_2_4_1_plus = _flash_attn_version >= PkgVersion("2.4.1")


from transformer_engine.pytorch.attention import (
    check_set_window_size,
    FusedAttention,
    UnfusedDotProductAttention,
    get_cu_seqlens,
    _get_full_cu_seqlens,
    get_alibi,
    get_qkv_layout,
    InferenceParams,
    _attention_backends,
    _PrepareQKVForFA,
    get_cu_seqlens_and_indices,
    UnpackTensor,
    get_indices,
    PackTensors,
)

from transformer_engine.pytorch.cpu_offload import CPUOffloadEnabled
_flash_attn_3_is_installed = False
_flash_attn_3_version = PkgVersion("0")
# HACK(huang.huang): recompute-variance for fa: import packages 
from transformer_engine.pytorch.attention import (
    AttentionParams, get_attention_backend,
    FlashAttention, DotProductAttention
    )
# HACK(huang.huang):


# HACK(huang.huang): recompute-variance for fa: implement flash_attn_varlen_func_variance 
# which will return [coreattention_output, lse, ...] instead of coreattention_output only; 
# and will seperate the execution of the sdp_kernel from other operations before and after it
_MIN_MUSA_DIM = 64
_MAX_MUSA_DIM = 192
def flash_attn_varlen_func_variance(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    # The input shape of varlen flash is [bs x seq_len, nheads, head_dim]
    # but the input of sdp is [bs, nheads, seq_len, head_dim]
    # seq_len = max_seqlen_q
    # bs = q.shape[0] // seq_len
    head_dim= q.shape[-1]
    if head_dim >= _MIN_MUSA_DIM and head_dim <= _MAX_MUSA_DIM:
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
            attn_output = torch.ops.aten._scaled_dot_product_attention_flash_musa(
                q,
                k,
                v,
                dropout_p=dropout_p,
                # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
                is_causal=causal #self.is_causal and attention_mask is None and q_len > 1,
            )
        return attn_output
    else:
        raise NotImplementedError(f"head_dim={head_dim} is not supported by flash_attn_varlen_func_variance")
# HACK(huang.huang)


# HACK(huang.huang): recompute-variance for fa: modify __init__ for FlashAttention and DotProductAttention, 
# just add a attr "recompute_variance" for them
def FlashAttention__init__(
    self,
    softmax_scale: float,
    attention_dropout: float = 0.0,
    attention_dropout_ctx: Optional[Callable] = nullcontext,
    attention_type: str = "self",
    layer_number: Optional[int] = None,
    deterministic: bool = False,
    recompute_variance: bool = False, # MUSA patch: support recompute_variance
) -> None:
    super(FlashAttention, self).__init__()

    assert (
        _flash_attn_version >= _flash_attn_version_required
    ), f"FlashAttention minimum version {_flash_attn_version_required} is required."
    assert (
        _flash_attn_version <= _flash_attn_max_version
    ), f"FlashAttention maximum version {_flash_attn_max_version} is supported."

    self.softmax_scale = softmax_scale
    self.attention_dropout_ctx = attention_dropout_ctx
    self.attention_dropout = attention_dropout
    self.attention_type = attention_type
    self.layer_number = 1 if layer_number is None else layer_number
    self.deterministic = deterministic
    self.recompute_variance = recompute_variance # MUSA patch: support recompute_variance


def DotProductAttention__init__(
    self,
    num_attention_heads: int,
    kv_channels: Union[int, Tuple[int, int]],
    num_gqa_groups: Optional[int] = None,
    attention_dropout: float = 0.0,
    qkv_format: str = "sbhd",
    attn_mask_type: str = "causal",
    window_size: Optional[Tuple[int, int]] = None,
    sequence_parallel: bool = False,
    tp_size: int = 1,
    get_rng_state_tracker: Optional[Callable] = None,
    tp_group: Optional[dist_group_type] = None,
    layer_number: Optional[int] = None,
    attention_type: str = "self",
    cp_group: Optional[Union[dist_group_type, List[dist_group_type]]] = None,
    cp_global_ranks: List[int] = None,
    cp_stream: torch.cuda.Stream = None,
    cp_comm_type: str = "p2p",
    softmax_scale: Optional[float] = None,
    recompute_variance: bool = False, # MUSA patch: support for variance computation
) -> None:
    super(DotProductAttention, self).__init__()

    self.logger = logging.getLogger("DotProductAttention")
    # self.logger.setLevel(_log_level)
    if not self.logger.hasHandlers():
        self.logger.addHandler(_stream_handler)
    self.qkv_format = qkv_format
    attn_mask_type = attn_mask_type.replace(",", "_")
    if attn_mask_type == "causal_padding":
        attn_mask_type = "padding_causal"
    self.attn_mask_type = attn_mask_type
    self.window_size = check_set_window_size(attn_mask_type, window_size)
    if tp_group is None:
        self.tp_size = tp_size
        if tp_size == 1:
            self.set_tensor_parallel_group(tp_group)
    else:
        self.tp_size = get_distributed_world_size(tp_group)
        self.set_tensor_parallel_group(tp_group)
    self.get_rng_state_tracker = get_rng_state_tracker
    self.num_attention_heads = num_attention_heads
    self.layer_number = 1 if layer_number is None else layer_number
    self.cp_group = cp_group
    self.cp_global_ranks = cp_global_ranks
    self.cp_stream = cp_stream
    self.cp_comm_type = cp_comm_type

    self.recompute_variance = recompute_variance # MUSA patch: support for variance computation
    self.hidden_size_per_attention_head_k = (
        kv_channels if isinstance(kv_channels, int) else kv_channels[0]
    )
    self.hidden_size_per_attention_head_v = (
        kv_channels if isinstance(kv_channels, int) else kv_channels[1]
    )

    self.num_gqa_groups = num_attention_heads if num_gqa_groups is None else num_gqa_groups
    self.num_gqa_groups_per_partition = int(self.num_gqa_groups // self.tp_size)

    assert (
        num_attention_heads % self.num_gqa_groups == 0
    ), "The number of attention heads must be divisible by the number of GQA groups!"

    self.rng_states_tracker = None
    if sequence_parallel or get_rng_state_tracker is None:
        attention_dropout_ctx = nullcontext
    else:
        self.rng_states_tracker = get_rng_state_tracker()
        set_all_rng_states(self.rng_states_tracker.get_states())
        attention_dropout_ctx = self.rng_states_tracker.fork

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(
            kv_channels if isinstance(kv_channels, int) else kv_channels[0]
        )

    self.deterministic = (
        not bool(int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")))
        or torch.are_deterministic_algorithms_enabled()
    )
    # To use the workspace optimization path for determinism, please
    # set NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT=1 for cuDNN >=8.9.5 and <9.0.0,
    # and set NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 for cuDNN >=9.0.0.
    cudnn_version = get_cudnn_version()
    if (8, 9, 5) <= cudnn_version < (9, 0, 0):
        if self.deterministic:
            os.environ["NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT"] = "1"

        # CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT
        # - unset:       enables workspace optimization when required workspace is <= 256MB
        #                or when bias gradient needs to be computed
        # - n:           enables workspace optimization when required workspace is <= n bytes
        # - -1:          enables workspace optimization always
        # - 0:           disables workspace optimization always
        if "NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT" in os.environ:
            if os.environ["NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT"] == "0":
                os.environ["CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT"] = "0"
            if os.environ["NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT"] == "1":
                os.environ["CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT"] = "-1"

    assert attention_type in AttnTypes, f"attention_type {attention_type} not supported"

    self.attention_type = attention_type
    self.attention_dropout = attention_dropout

    attn_kwargs = {
        "attention_dropout": attention_dropout,
        "attention_dropout_ctx": attention_dropout_ctx,
    }

    self.flash_attention = FlashAttention(
        softmax_scale,
        attention_type=attention_type,
        layer_number=layer_number,
        deterministic=self.deterministic,
        recompute_variance=self.recompute_variance, # MUSA patch: support for variance computation
        **attn_kwargs,
    )

    # Instantiating three types since use of flash-attn and FusedAttention
    # might be ruled out due to forward inputs.
    self.fused_attention = FusedAttention(
        softmax_scale,
        attention_type=attention_type,
        layer_number=layer_number,
        deterministic=self.deterministic,
        **attn_kwargs,
    )
    self.unfused_attention = UnfusedDotProductAttention(
        softmax_scale, **attn_kwargs, layer_number=layer_number
    )

    def remove_extra_states_check(self, incompatible_keys):  # pylint: disable=unused-argument
        """
        Temporarily remove core_attention._extra_state as a missing key
        when loading older Transformer Engine checkpoints. Will phase out
        this hook in Transformer Engine 2.0.
        """
        for key in incompatible_keys.missing_keys:
            if "core_attention._extra_state" in key:
                incompatible_keys.missing_keys.remove(key)

    self.register_load_state_dict_post_hook(remove_extra_states_check)
# HACK(huang.huang)


# HACK(huang.huang): recompute-variance for fa: add functions "forward_fa", "forward_after_fa", "forward_before_fa" for DotProductAttention
def FlashAttention_forward_after_fa(self, output, qkv_format, indices_q, batch_size, attn_mask_type, max_seqlen_q, q_shape, v_shape):
    bs = q_shape[0]
    q_seq_len = q_shape[1]
    output = output[0].transpose(1, 2).contiguous().view(bs, q_seq_len, q_shape[-2], v_shape[-1]) #core_output, args*
    if qkv_format in ["sbhd", "bshd"] and "padding" in attn_mask_type:
        output = UnpackTensor.apply(indices_q, batch_size * max_seqlen_q, output)

    if qkv_format == "sbhd":
        # (bs)hd -> bs(hd) -> sb(hd)
        output = output.view(batch_size, max_seqlen_q, -1).transpose(0, 1).contiguous()
    elif qkv_format == "bshd":
        # (bs)hd -> bs(hd)
        output = output.view(batch_size, max_seqlen_q, -1).contiguous()
    elif qkv_format == "thd":
        # thd -> t(hd)
        output = output.view(output.shape[0], -1).contiguous()
    return output

def FlashAttention_forward_fa(
    self,
    query_layer,  
    key_layer, 
    value_layer, 
    cu_seqlens_q,
    cu_seqlens_kv,
    max_seqlen_q,
    max_seqlen_kv,
    attn_mask_type,
    window_size,
    alibi_slopes,
    qkv_format, 
    indices_q, 
    batch_size,
    q_shape,
    v_shape,
    *args,
    ):
    with self.attention_dropout_ctx():
        fa_optional_forward_kwargs = {}
        if _flash_attn_2_3_plus:
            fa_optional_forward_kwargs["window_size"] = window_size
        if _flash_attn_2_4_plus:
            fa_optional_forward_kwargs["alibi_slopes"] = alibi_slopes
        if _flash_attn_2_4_1_plus:
            fa_optional_forward_kwargs["deterministic"] = self.deterministic
        output = flash_attn_varlen_func_variance(
            query_layer,
            key_layer,
            value_layer,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            self.attention_dropout if self.training else 0.0,
            softmax_scale=self.softmax_scale,
            causal="causal" in attn_mask_type,
            **fa_optional_forward_kwargs,
        ) 
    return output, qkv_format, indices_q, batch_size, attn_mask_type, max_seqlen_q, q_shape, v_shape
    
def FlashAttention_forward_before_fa(
    self,
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    qkv_layout: str = "sbh3d",
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
    attn_mask_type: str = "causal",
    window_size: Optional[Tuple[int, int]] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    cp_group: Optional[Union[dist_group_type, List[dist_group_type]]] = None,
    cp_global_ranks: List[int] = None,
    cp_stream: torch.cuda.Stream = None,
    cp_comm_type: str = "p2p",
    fp8: bool = False,
    fp8_meta: Optional[Dict[str, Any]] = None,
    quantizers=None,
) -> torch.Tensor:
    """flash-attn fprop"""

    assert all(
        x.dtype in [torch.float16, torch.bfloat16] or isinstance(x, Float8Tensor)
        for x in [query_layer, key_layer, value_layer]
    ), "FlashAttention only supports FP16 and BF16 data types, or Float8Tensors."
    assert (
        query_layer.is_cuda and key_layer.is_cuda and value_layer.is_cuda
    ), "FlashAttention currently only supports CUDA tensors."
    assert (
        qkv_layout in QKVLayouts
    ), f"FlashAttention does not support qkv_layout = {qkv_layout}!"

    cp_size = 1
    if isinstance(cp_group, dist_group_type):
        cp_size = get_distributed_world_size(cp_group)
    elif isinstance(cp_group, list):
        for group in cp_group:
            cp_size *= get_distributed_world_size(group)
    context_parallel = cp_size > 1

    qkv_format = "".join([i for i in qkv_layout.split("_")[0] if i.isalpha()])
    
    if all(not isinstance(x, Float8Tensor) for x in [query_layer, key_layer, value_layer]):
        if qkv_format == "sbhd":
            # For now just 128, will make it more general in the future
            if (
                query_layer.shape[-1] == 128
                and query_layer.shape[0] * query_layer.shape[1] >= 512
                and qkv_layout == "sbh3d"
            ):
                query_layer, key_layer, value_layer = _PrepareQKVForFA.apply(
                    query_layer, key_layer, value_layer
                )
            else:
                query_layer, key_layer, value_layer = [
                    x.transpose(0, 1) for x in (query_layer, key_layer, value_layer)
                ]
        if context_parallel:
            query_layer, key_layer, value_layer = [
                x.contiguous() for x in (query_layer, key_layer, value_layer)
            ]
    else:
        if qkv_format == "sbhd":
            query_layer._data, key_layer._data, value_layer._data = [
                x.transpose(0, 1)
                for x in (query_layer._data, key_layer._data, value_layer._data)
            ]
            query_layer, key_layer, value_layer = [
                Float8Tensor.make_like(x, data=x._data, shape=x._data.shape)
                for x in (query_layer, key_layer, value_layer)
            ]
        if context_parallel:
            query_layer._data, key_layer._data, value_layer._data = [
                x.contiguous() for x in (query_layer._data, key_layer._data, value_layer._data)
            ]

    batch_size = query_layer.shape[0]
    
    if qkv_format in ["sbhd", "bshd"]:
        max_seqlen_q, max_seqlen_kv = query_layer.shape[1], key_layer.shape[1]
        max_seqlen_q *= cp_size
        max_seqlen_kv *= cp_size
        indices_q = None
        if "padding" in attn_mask_type:
            assert not context_parallel, "Padding mask not supported with context parallelism!"
            # [b * s, h, d]
            query_layer, key_layer, value_layer = [
                x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
                for x in [query_layer, key_layer, value_layer]
            ]


            if self.attention_type == "self":
                assert (
                    max_seqlen_q == max_seqlen_kv
                ), "Maximum sequence length for Q and KV should be the same."
                if cu_seqlens_q is None:
                    assert (
                        attention_mask is not None
                    ), "Please provide attention_mask for padding!"
                    cu_seqlens_q, indices_q = get_cu_seqlens_and_indices(attention_mask)
                else:
                    indices_q = get_indices(max_seqlen_q, cu_seqlens_q)
                cu_seqlens_kv = cu_seqlens_q
                query_layer, key_layer, value_layer = PackTensors.apply(
                    indices_q, query_layer, key_layer, value_layer
                )
            else:
                if cu_seqlens_q is None or cu_seqlens_kv is None:
                    assert (
                        attention_mask is not None
                    ), "Please provide attention_mask for padding!"
                    cu_seqlens_q, indices_q = get_cu_seqlens_and_indices(attention_mask[0])
                    cu_seqlens_kv, indices_kv = get_cu_seqlens_and_indices(attention_mask[1])
                else:
                    indices_q = get_indices(max_seqlen_q, cu_seqlens_q)
                    indices_kv = get_indices(max_seqlen_kv, cu_seqlens_kv)
                query_layer = PackTensors.apply(indices_q, query_layer)
                key_layer, value_layer = PackTensors.apply(indices_kv, key_layer, value_layer)
        else:
            # Cumulative sequence lengths for unpadded data
            if cu_seqlens_q is None:
                cu_seqlens_q = _get_full_cu_seqlens(
                    batch_size,
                    max_seqlen_q,
                    query_layer.device,
                )
            if cu_seqlens_kv is None:
                cu_seqlens_kv = _get_full_cu_seqlens(
                    batch_size,
                    max_seqlen_kv,
                    key_layer.device,
                )
    elif qkv_format == "thd":
        assert (
            cu_seqlens_q is not None and cu_seqlens_kv is not None
        ), "cu_seqlens_q and cu_seqlens_kv can not be None when qkv_format = thd!"
        if max_seqlen_q is None:
            seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
            max_seqlen_q = seqlens_q.max().item()
        if max_seqlen_kv is None:
            seqlens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
            max_seqlen_kv = seqlens_kv.max().item()


    if context_parallel and all(
        not isinstance(x, Float8Tensor) for x in [query_layer, key_layer, value_layer]
    ):
        assert (
            alibi_slopes is None
        ), "Alibi slope bias addition is not supported with context parallelism."
        with self.attention_dropout_ctx():
            output = attn_forward_func_with_cp(
                self.training,
                query_layer,
                key_layer,
                value_layer,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                cu_seqlens_q if qkv_format == "thd" else None,
                cu_seqlens_kv if qkv_format == "thd" else None,
                self.attention_dropout if self.training else 0.0,
                cp_group,
                cp_global_ranks,
                cp_stream,
                cp_comm_type,
                softmax_scale=self.softmax_scale,
                qkv_format="bshd" if qkv_format == "sbhd" else qkv_format,
                attn_mask_type=attn_mask_type,
                deterministic=self.deterministic,
                window_size=window_size,
                quantizers=quantizers,
            )
    else:

        from transformer_engine.pytorch.cpu_offload import CPUOffloadEnabled

        if CPUOffloadEnabled:
            tensor_list = [query_layer, key_layer, value_layer, cu_seqlens_q, cu_seqlens_kv]
            for tensor in tensor_list:
                if tensor is not None:
                    tensor.activation_offloading = True


        # transpose before fa, which will be saved for bwd
        bs = query_layer.shape[0]
        seq_len = query_layer.shape[1]
        kv_seq_len = key_layer.shape[1]
        # seq_len = max_seqlen_q
        # bs = query_layer.shape[0] // seq_len
        q_shape = query_layer.shape
        v_shape = value_layer.shape
        query_layer = query_layer.view(bs, seq_len, query_layer.shape[-2], query_layer.shape[-1]).transpose(1, 2)
        key_layer = key_layer.view(bs, kv_seq_len, key_layer.shape[-2], key_layer.shape[-1]).transpose(1, 2)
        value_layer = value_layer.view(bs, kv_seq_len, value_layer.shape[-2], value_layer.shape[-1]).transpose(1, 2)

        return (
            query_layer,  
            key_layer, 
            value_layer, 
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            attn_mask_type,
            window_size,
            alibi_slopes,
            qkv_format,
            indices_q,
            batch_size,
            q_shape,
            v_shape)

def DotProductAttention_forward_fa(
    self,
    *args,
    ):
    return self.flash_attention.forward_fa(*args)

def DotProductAttention_forward_after_fa(self, *args):
    output = self.flash_attention.forward_after_fa(*args)
    return output

def DotProductAttention_forward_before_fa(
    self,
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    qkv_format: Optional[str] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    cu_seqlens_q_padded: Optional[torch.Tensor] = None,
    cu_seqlens_kv_padded: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
    attn_mask_type: Optional[str] = None,
    window_size: Optional[Tuple[int, int]] = None,
    checkpoint_core_attention: bool = False,
    core_attention_bias_type: str = "no_bias",
    core_attention_bias: Optional[torch.Tensor] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    fast_zero_fill: bool = True,
    inference_params: Optional[InferenceParams] = None,
) -> torch.Tensor:
    with self.prepare_forward(
        query_layer,
        num_gemms=3,
        allow_non_contiguous=True,
    ) as query_layer:
        if self.fp8:
            if self.fp8_meta["recipe"].fp8_mha:
                if not self.fp8_meta["recipe"].fp8_dpa:
                    self.fp8_meta["recipe"].fp8_dpa = True
                    self.logger.warning(
                        """Forcing fp8_meta["recipe"].fp8_dpa=True due to """
                        """fp8_meta["recipe"].fp8_mha=True"""
                    )

        if self.fp8 and self.fp8_meta["recipe"].fp8_dpa:
            forward_dtype = get_fp8_te_dtype(self.fp8_meta["recipe"], fprop_tensor=True)
            backward_dtype = get_fp8_te_dtype(self.fp8_meta["recipe"], fprop_tensor=False)
            assert forward_dtype in [
                tex.DType.kFloat8E4M3,
                tex.DType.kFloat8E5M2,
            ] and backward_dtype in [
                tex.DType.kFloat8E4M3,
                tex.DType.kFloat8E5M2,
            ], """DotProductAttention only supports "E4M3" and "E5M2" FP8 data types."""

        assert (
            query_layer.is_cuda and key_layer.is_cuda and value_layer.is_cuda
        ), "DotProductAttention only supports CUDA tensors."
        assert (
            query_layer.dtype == key_layer.dtype and query_layer.dtype == value_layer.dtype
        ), "Queries, keys and values must have the same data type!"
        assert (
            key_layer.shape[:-1] == value_layer.shape[:-1]
        ), "Keys and values must have the same batch size, sequence length and number of heads!"
        assert (
            key_layer.shape[-1] == self.hidden_size_per_attention_head_k
        ), f"Keys have head_dim = {key_layer.shape[-1]}, "
        "but expected head_dim = {self.hidden_size_per_attention_head_k}!"
        assert (
            value_layer.shape[-1] == self.hidden_size_per_attention_head_v
        ), f"Values have head_dim = {value_layer.shape[-1]}, "
        "but expected head_dim = {self.hidden_size_per_attention_head_v}!"

        if qkv_format is None:
            qkv_format = self.qkv_format

        if attn_mask_type is None:
            attn_mask_type = self.attn_mask_type
        else:
            attn_mask_type = attn_mask_type.replace(",", "_")
            if attn_mask_type == "causal_padding":
                attn_mask_type = "padding_causal"
        assert (
            attn_mask_type in AttnMaskTypes
        ), f"Attention mask type {attn_mask_type} is not supported!"
        if qkv_format == "thd":
            assert (
                "padding" in attn_mask_type
            ), "Attention mask type must be padding or padding_causal for qkv_format=thd!"

        if window_size is None:
            window_size = self.window_size
        window_size = check_set_window_size(attn_mask_type, window_size)

        if self.rng_states_tracker is not None and is_graph_capturing():
            assert isinstance(
                self.rng_states_tracker, CudaRNGStatesTracker
            ), "Unsupported RNG states tracker."
            assert (
                graph_safe_rng_available()
            ), "Upgrade PyTorch version to get RNG manipulation support for cuda graph capture."

        if inference_params is not None:
            assert self.layer_number is not None, "Layer number must be set!"

            # convert causal to causal_bottom_right in inference when KV-caching is in use
            # so users can run with the same attn_mask_type for training and inference
            if attn_mask_type in ["causal", "padding_causal"]:
                attn_mask_type = attn_mask_type + "_bottom_right"

            if qkv_format == "bshd":
                key_layer = key_layer.transpose(0, 1)
                value_layer = value_layer.transpose(0, 1)

            (
                inference_key_memory,
                inference_value_memory,
            ) = inference_params.key_value_memory_dict[self.layer_number]

            batch_start = inference_params.batch_size_offset
            batch_end = batch_start + key_layer.size(1)
            assert batch_end <= inference_key_memory.size(1)

            sequence_start = inference_params.sequence_len_offset
            sequence_end = sequence_start + key_layer.size(0)
            assert sequence_end <= inference_key_memory.size(0)

            # Copy keys and values into KV-cache
            inference_key_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = (
                key_layer
            )
            inference_value_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = (
                value_layer
            )
            key_layer = inference_key_memory[:sequence_end, batch_start:batch_end, ...]
            value_layer = inference_value_memory[:sequence_end, batch_start:batch_end, ...]

            if qkv_format == "bshd":
                key_layer = key_layer.transpose(0, 1)
                value_layer = value_layer.transpose(0, 1)

            key_layer = key_layer.contiguous()
            value_layer = value_layer.contiguous()

        assert (
            key_layer.shape[-2] == self.num_gqa_groups_per_partition
            and value_layer.shape[-2] == self.num_gqa_groups_per_partition
        ), (
            "Keys and values must have num_gqa_group ="
            f" {self.num_gqa_groups_per_partition} heads!"
        )
        assert qkv_format in [
            "sbhd",
            "bshd",
            "thd",
        ], "DotProductAttention only supports qkv_format = {'sbhd', 'bshd', 'thd'}!"

        if qkv_format == "thd":
            assert all(
                len(x.shape) == 3 for x in (query_layer, key_layer, value_layer)
            ), "Queries, keys and values must be 3D tensors when qkv_format = thd!"
            assert (
                cu_seqlens_q is not None and cu_seqlens_kv is not None
            ), "cu_seqlens_q and cu_seqlens_kv can not be None when qkv_format = thd!"
            assert (
                cu_seqlens_q.shape == cu_seqlens_kv.shape
                and len(cu_seqlens_q.shape) == 1
                and len(cu_seqlens_kv.shape) == 1
            ), "cu_seqlens_q and cu_seqlens_q must both have shape [batch_size + 1]!"
            assert (
                cu_seqlens_q.dtype == torch.int32 and cu_seqlens_kv.dtype == torch.int32
            ), "cu_seqlens_q and cu_seqlens_q must both be in dtype torch.int32!"
            batch_size = len(cu_seqlens_q) - 1
            if max_seqlen_q is None:
                if cu_seqlens_q_padded is not None:
                    seqlens_q = cu_seqlens_q_padded[1:] - cu_seqlens_q_padded[:-1]
                else:
                    seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
                max_seqlen_q = int((seqlens_q.max().item() + 63) // 64 * 64)
            if max_seqlen_kv is None:
                if cu_seqlens_kv_padded is not None:
                    seqlens_kv = cu_seqlens_kv_padded[1:] - cu_seqlens_kv_padded[:-1]
                else:
                    seqlens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
                max_seqlen_kv = int((seqlens_kv.max().item() + 63) // 64 * 64)

        cp_size = 1
        if isinstance(self.cp_group, dist_group_type):
            cp_size = get_distributed_world_size(self.cp_group)
        elif isinstance(self.cp_group, list):
            for group in self.cp_group:
                cp_size *= get_distributed_world_size(group)
        context_parallel = cp_size > 1

        if qkv_format in ["sbhd", "bshd"]:
            assert all(
                len(x.shape) == 4 for x in (query_layer, key_layer, value_layer)
            ), f"Queries, keys and values must be 4D tensors when qkv_format = {qkv_format}!"
            if qkv_format == "sbhd":
                max_seqlen_q = query_layer.shape[0] if max_seqlen_q is None else max_seqlen_q
                max_seqlen_kv = key_layer.shape[0] if max_seqlen_kv is None else max_seqlen_kv
                batch_size = query_layer.shape[1]
            else:
                max_seqlen_q = query_layer.shape[1] if max_seqlen_q is None else max_seqlen_q
                max_seqlen_kv = key_layer.shape[1] if max_seqlen_kv is None else max_seqlen_kv
                batch_size = query_layer.shape[0]
            max_seqlen_q *= cp_size
            max_seqlen_kv *= cp_size
            if cu_seqlens_q is not None:
                seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
                assert all(
                    seqlens_q <= max_seqlen_q
                ), """Sequence lengths indicated by cu_seqlens_q must be no greater than
                    the sequence dimension in 'query_layer'!"""
            if cu_seqlens_kv is not None:
                seqlens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
                assert all(
                    seqlens_kv <= max_seqlen_kv
                ), """Sequence lengths indicated by cu_seqlens_kv must be no greater than
                    the sequence dimension in 'key_layer' and 'value_layer'!"""
            if cu_seqlens_q is None or cu_seqlens_kv is None:
                if "padding" in attn_mask_type:
                    assert (
                        attention_mask is not None
                    ), "Please provide attention_mask for padding!"
                    if self.attention_type == "self":
                        cu_seqlens_q = get_cu_seqlens(attention_mask)
                        cu_seqlens_kv = cu_seqlens_q
                    else:
                        cu_seqlens_q = get_cu_seqlens(attention_mask[0])
                        cu_seqlens_kv = get_cu_seqlens(attention_mask[1])
                else:
                    cu_seqlens_q = _get_full_cu_seqlens(
                        batch_size,
                        max_seqlen_q,
                        query_layer.device,
                    )
                    cu_seqlens_kv = _get_full_cu_seqlens(
                        batch_size,
                        max_seqlen_kv,
                        key_layer.device,
                    )

        if (
            isinstance(query_layer, Float8Tensor)
            and isinstance(key_layer, Float8Tensor)
            and isinstance(value_layer, Float8Tensor)
        ):
            qkv_layout, query_layer._data, key_layer._data, value_layer._data = get_qkv_layout(
                query_layer._data, key_layer._data, value_layer._data, qkv_format=qkv_format
            )
        else:
            qkv_layout, query_layer, key_layer, value_layer = get_qkv_layout(
                query_layer, key_layer, value_layer, qkv_format=qkv_format
            )

        global _alibi_cache
        if alibi_slopes is not None:
            assert (
                core_attention_bias_type == "alibi"
            ), "core_attention_bias_type must be alibi in order to use alibi_slopes!"
            if self.layer_number == 1:
                _alibi_cache["_alibi_slopes_require_update"] = True
                _alibi_cache["_alibi_bias_require_update"] = True
        bottom_right_alignment = (attn_mask_type not in ["causal", "padding_causal"],)
        if core_attention_bias_type == "alibi":
            assert (
                core_attention_bias is None
            ), "core_attention_bias must be None when core_attention_bias_type is alibi!"
            if (
                _alibi_cache["_num_heads"] != query_layer.shape[-2]
                or _alibi_cache["_max_seqlen_q"] != max_seqlen_q
                or _alibi_cache["_max_seqlen_kv"] != max_seqlen_kv
                or _alibi_cache["_bottom_right_alignment"] != bottom_right_alignment
                or _alibi_cache["_alibi_slopes"] is None
            ):
                _alibi_cache["_alibi_slopes_require_update"] = True
                _alibi_cache["_alibi_bias_require_update"] = True

        core_attention_bias_shape = None
        if core_attention_bias is not None:
            if (
                core_attention_bias.shape[0] == batch_size
                and core_attention_bias.shape[1] == query_layer.shape[-2]
            ):
                core_attention_bias_shape = "bhss"
            elif (
                core_attention_bias.shape[0] == 1
                and core_attention_bias.shape[1] == query_layer.shape[-2]
            ):
                core_attention_bias_shape = "1hss"
            elif (
                core_attention_bias.shape[0] == batch_size and core_attention_bias.shape[1] == 1
            ):
                core_attention_bias_shape = "b1ss"
            elif core_attention_bias.shape[0] == 1 and core_attention_bias.shape[1] == 1:
                core_attention_bias_shape = "11ss"
            else:
                assert (
                    False
                ), "core_attention_bias must be in one of {bhss, 1hss, b1ss, 11ss} shapes"

        pad_between_seqs = (
            cu_seqlens_q_padded is not None
            and not torch.equal(cu_seqlens_q_padded[:-1], cu_seqlens_q[:-1])
        ) or (
            cu_seqlens_kv_padded is not None
            and not torch.equal(cu_seqlens_kv_padded[:-1], cu_seqlens_kv[:-1])
        )

        attention_params = AttentionParams(
            qkv_type=type(query_layer),
            qkv_dtype=query_layer.dtype,
            qkv_layout=qkv_layout,
            batch_size=batch_size,
            num_heads=query_layer.shape[-2],
            num_gqa_groups=key_layer.shape[-2],
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            head_dim_qk=query_layer.shape[-1],
            head_dim_v=value_layer.shape[-1],
            attn_mask_type=attn_mask_type,
            window_size=window_size,
            alibi_slopes_shape=alibi_slopes.shape if alibi_slopes is not None else None,
            core_attention_bias_type=core_attention_bias_type,
            core_attention_bias_shape=core_attention_bias_shape,
            core_attention_bias_requires_grad=(
                core_attention_bias.requires_grad if core_attention_bias is not None else False
            ),
            pad_between_seqs=pad_between_seqs,
            attention_dropout=self.attention_dropout,
            context_parallel=context_parallel,
            deterministic=self.deterministic,
            is_training=self.training,
            fp8=self.fp8,
            fp8_meta=self.fp8_meta,
        )
        global _attention_backends, _use_flash_attn_3
        if (
            _attention_backends["attention_params"] is None
            or attention_params != _attention_backends["attention_params"]
        ):
            _attention_backends["attention_params"] = attention_params
            _attention_backends["backend_selection_requires_update"] = True
        
        if _attention_backends["backend_selection_requires_update"]:
            _use_flash_attn_3 = _flash_attn_3_is_installed
            (
                use_flash_attention,
                use_fused_attention,
                fused_attention_backend,
                use_unfused_attention,
                _,
            ) = get_attention_backend(attention_params)
            if use_flash_attention:
                self.logger.info(
                    "Running with FlashAttention backend (version %s)",
                    _flash_attn_version if not _use_flash_attn_3 else _flash_attn_3_version,
                )
            elif use_fused_attention:
                self.logger.info(
                    "Running with FusedAttention backend (sub-backend %s)",
                    int(fused_attention_backend),
                )
            elif use_unfused_attention:
                self.logger.info("Running with UnfusedDotProductAttention backend")
        else:
            use_flash_attention = _attention_backends["use_flash_attention"]
            use_fused_attention = _attention_backends["use_fused_attention"]
            fused_attention_backend = _attention_backends["fused_attention_backend"]
            use_unfused_attention = _attention_backends["use_unfused_attention"]

        use_flash_attention = True #TODO:huang.huang set fa manually now!
        if use_flash_attention:
            if core_attention_bias_type == "alibi":
                alibi_slopes, _ = get_alibi(
                    query_layer.shape[-2],
                    max_seqlen_q,
                    max_seqlen_kv,
                    alibi_slopes=alibi_slopes,
                )
            return self.flash_attention.forward_before_fa(
                query_layer,
                key_layer,
                value_layer,
                attention_mask=attention_mask,
                qkv_layout=qkv_layout,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                attn_mask_type=attn_mask_type,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                cp_group=self.cp_group,
                cp_global_ranks=self.cp_global_ranks,
                cp_stream=self.cp_stream,
                cp_comm_type=self.cp_comm_type,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                fp8=self.fp8 and self.fp8_meta["recipe"].fp8_dpa,
                fp8_meta=self.fp8_meta,
                quantizers=self.quantizers,
            )
            
        raise RuntimeError("No dot product attention support for the provided inputs!")
# HACK(huang.huang)
        
from .utils import replace_attr, add_attr
replace_attr(FlashAttention,"__init__", FlashAttention__init__)
add_attr(FlashAttention, "forward_fa", FlashAttention_forward_fa)
add_attr(FlashAttention, "forward_before_fa", FlashAttention_forward_before_fa)
add_attr(FlashAttention, "forward_after_fa", FlashAttention_forward_after_fa)

replace_attr(DotProductAttention, "__init__", DotProductAttention__init__)
add_attr(DotProductAttention, "forward_fa", DotProductAttention_forward_fa)
add_attr(DotProductAttention, "forward_before_fa", DotProductAttention_forward_before_fa)
add_attr(DotProductAttention, "forward_after_fa", DotProductAttention_forward_after_fa)

