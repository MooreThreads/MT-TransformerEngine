# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Linear API"""
import os
import logging
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch

import transformer_engine_torch as tex

from transformer_engine.pytorch.module.base import (
    get_workspace,
    get_ub,
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.utils import (
    clear_tensor_data,
    requires_grad,
)
from transformer_engine.pytorch.distributed import (
    get_distributed_world_size,
    allreduce,
    reduce_scatter_along_first_dim,
    gather_along_first_dim,
)
from transformer_engine.pytorch.cpp_extensions import (
    general_gemm,
)
from transformer_engine.pytorch.jit import no_torch_dynamo
from transformer_engine.pytorch.graph import is_graph_capturing
from transformer_engine.pytorch.float8_tensor import Float8Tensor
# NVTE_DEBUG = 0/1 # disables/enables debug mode, default = 0
_NVTE_DEBUG = int(os.getenv("NVTE_DEBUG", "0"))
# NVTE_DEBUG_LEVEL = 0/1/2 # enables more and more verbose debug mode, default = 0
_NVTE_DEBUG_LEVEL = int(os.getenv("NVTE_DEBUG_LEVEL", "0"))
log_level = _NVTE_DEBUG * _NVTE_DEBUG_LEVEL
log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
logging.basicConfig(
    format="[%(levelname)-8s | %(name)-19s]: %(message)s",
    level=log_levels[log_level if log_level in [0, 1, 2] else 2],
)

from transformer_engine.pytorch.module import Linear

# HACK(huang.huang): recompute-variance for linear: add functions "backward_custom"
def backward_custom(self, input, weight, grad_output, is_first_microbatch=None,):
    logger = logging.getLogger("Linear")
    if isinstance(grad_output, Float8Tensor):
        raise RuntimeError("backward_custom is not supported with FP8 tensors now")
    with torch.cuda.nvtx.range("_Linear_backward"):
        requires_dgrad = input.requires_grad
        tensor_parallel = self.tp_size > 1
        inputmat = input.view(-1, weight.shape[-1])


        tp_world_size = get_distributed_world_size(self.tp_group)
        ub_overlap_ag = False if tp_world_size == 1 else self.ub_overlap_ag
        if ub_overlap_ag:
            dim_size = list(grad_output.size())
            dim_size[0] = dim_size[0] * tp_world_size
            ub_obj_gradout = get_ub(self.ub_name + "_dgrad")
            if ub_obj_gradout.is_atomic_gemm():
                ub_algo = tex.UbufOverlapAlgo.ATOMIC_GEMM_AG_P2P
            else:
                ub_algo = tex.UbufOverlapAlgo.SPLIT_PIPELINED_AG_P2P

        # TransformerEngineBaseModule.grad_output_preprocess
        row_parallel_mode = self.parallel_mode == "row"
        grad_output = grad_output.contiguous()
        grad_output_mat = grad_output.view(-1, grad_output.shape[-1])
        gather_grad_output = row_parallel_mode and self.sequence_parallel
        if gather_grad_output:
                if not ub_overlap_ag:
                    grad_output_mat, _ = gather_along_first_dim(grad_output_mat, self.tp_group)
                else:
                    ub_obj_gradout.copy_input_to_ubuf(grad_output, True)
                    grad_output_mat = ub_obj_gradout.get_ubuf_output(1)
        grad_output = grad_output_mat

        inputmat_total = None
        handle = None
        rs_out = None       
        ub_obj_dgrad = None
        ub_obj_wgrad = None
        ub_type_dgrad = None
        ub_type_wgrad = None
        dgrad_bulk = None
        if weight.requires_grad and self.parallel_mode == "column" and self.sequence_parallel:
            inputmat_total, handle = gather_along_first_dim(
                inputmat, self.tp_group, async_op=requires_dgrad
            )
        else:
            inputmat_total = inputmat

        if is_first_microbatch is not None:
            accumulate_wgrad_into_param_main_grad = (
                self.fuse_wgrad_accumulation and not is_first_microbatch
            )
        else:
            accumulate_wgrad_into_param_main_grad = self.fuse_wgrad_accumulation

        if self.fp8:
            raise RuntimeError("backward_custom is not supported with FP8 tensors now")
        
        if requires_dgrad:
            if self.fp8:
                raise RuntimeError("backward_custom is not supported with FP8 tensors now")
            else:
                logger.debug("Running backward in %s", self.activation_dtype)
                # dgrad, _, _ = gemm(
                #         weight,
                #         grad_output,
                #         self.activation_dtype,
                #         get_workspace(),
                #         layout="NN",
                #         grad=True,
                #         ub_algo=(
                #             tex.UbufOverlapAlgo.SPLIT_PIPELINED_AG_P2P
                #             if ub_overlap_ag
                #             else None
                #         ),
                #         ub=ub_obj_gradout if ub_overlap_ag else None,
                #     )
                dgrad, *_, rs_out = general_gemm(
                    weight,
                    grad_output,
                    get_workspace(),
                    layout="NN",
                    grad=True,
                    out=dgrad_bulk,
                    out_dtype=self.activation_dtype,
                    use_split_accumulator=_2X_ACC_DGRAD,
                    ub=ub_obj_gradout if ub_overlap_ag else None,
                    ub_type=ub_type_dgrad,
                    extra_output=rs_out,
                )
            # Overlap dgrad-RS/AR with wgrad
            if self.parallel_mode == "column" and self.sequence_parallel:
                if handle is not None:
                    handle.wait()
                dgrad, handle = reduce_scatter_along_first_dim(
                    dgrad, self.tp_group, async_op=True
                )
            elif self.parallel_mode == "column" and tensor_parallel:
                dgrad, handle = allreduce(dgrad, self.tp_group, async_op=True)


        if weight.requires_grad:
            if self.fp8:
                raise RuntimeError("backward_custom is not supported with FP8 tensors now")
            else:
                # WGRAD
                wgrad, grad_bias, _, rs_out = general_gemm(
                    inputmat_total,
                    grad_output,
                    get_workspace(),
                    layout="NT",
                    grad=True,
                    out_dtype=(
                        weight.main_grad.dtype if self.fuse_wgrad_accumulation else self.activation_dtype
                    ),
                    bias=None,
                    out=weight.main_grad if self.fuse_wgrad_accumulation else None,
                    use_split_accumulator=_2X_ACC_WGRAD,
                    accumulate=accumulate_wgrad_into_param_main_grad,
                    ub=ub_obj_wgrad,
                    ub_type=ub_type_wgrad,
                    extra_output=rs_out,
                )

            # Deallocate input tensor
            clear_tensor_data(inputmat_total)
            # clear_tensor_data(inputmat_t_total)

        # Column Parallel Linear
        if self.parallel_mode == "column" and tensor_parallel and handle is not None:
            handle.wait()

        if not self.use_bias:
            grad_bias = None

    if weight.requires_grad:
        # Handle custom DDP from mcore.
        if self.fuse_wgrad_accumulation and hasattr(weight, "grad_added_to_main_grad"):
            weight.grad_added_to_main_grad = True
            if getattr(weight, "zero_out_wgrad", False):
                wgrad = torch.zeros(
                    weight.main_grad.shape,
                    dtype=weight.dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )
            else:
                wgrad = torch.empty(
                    weight.main_grad.shape,
                    dtype=weight.dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )
            
        elif self.fuse_wgrad_accumulation:
            wgrad = None
    else:
        wgrad = None

    reduce_and_update_bwd_fp8_tensors = False
    if self.fp8 and  requires_grad(input, weight,):
        raise RuntimeError("backward_custom is not supported with FP8 tensors now")
        reduce_and_update_bwd_fp8_tensors = (
                reduce_and_update_bwd_fp8_tensors
                or FP8GlobalStateManager.is_first_fp8_module()
            )
    if reduce_and_update_bwd_fp8_tensors and not is_graph_capturing():
        FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)

    # Scatter fp8 weight buffers
    if self.fp8 and not isinstance(weight, Float8Tensor):
        raise RuntimeError("backward_custom is not supported with FP8 tensors now")
        # _fsdp_scatter_tensors(self.fsdp_group, weight_fp8)
    
    input.grad = dgrad.view(input.shape)  if requires_dgrad else None
    return  dgrad.view(input.shape) if requires_dgrad else None
# HACK(huang.huang)

from ..utils import add_attr
add_attr(Linear, "backward_custom", backward_custom)