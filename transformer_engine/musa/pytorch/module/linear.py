# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Linear API"""
import os
import logging
from typing import Any, Callable, Dict, Optional, Tuple, Union
from functools import reduce
from operator import mul as multiply_op

import torch

import transformer_engine_torch as tex

from transformer_engine.pytorch.module.base import (
    get_workspace,
    get_ub,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.utils import (
    clear_tensor_data,
    requires_grad,
    non_tn_fp8_gemm_supported,
    non_tn_fp8_gemm_supported
)
from transformer_engine.pytorch.distributed import (
    get_distributed_world_size,
    allreduce,
    reduce_scatter_along_first_dim,
    gather_along_first_dim,
    _fsdp_scatter_tensors,
)
from transformer_engine.pytorch.tensor.quantized_tensor import (
    QuantizedTensor,
    prepare_for_saving,
    restore_from_saved,
)

from transformer_engine.pytorch import cpp_extensions as ceg
from transformer_engine.pytorch.cpp_extensions import (
    general_gemm,
)
from transformer_engine.pytorch.graph import is_graph_capturing
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.module._common import noop_cat, _fix_gathered_fp8_transpose
from transformer_engine.pytorch.cpu_offload import is_cpu_offload_enabled
from transformer_engine.pytorch.utils import cast_if_needed

import torch

import transformer_engine_torch as tex

from transformer_engine.pytorch.module.base import (
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.utils import (
    cast_if_needed,
    clear_tensor_data,
    requires_grad,
)
from transformer_engine.pytorch.distributed import (
    is_fp8_activation_recompute_enabled,
    in_fp8_activation_recompute_phase,
)
from transformer_engine.pytorch.jit import no_torch_dynamo
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.graph import is_graph_capturing
from transformer_engine.pytorch.cpu_offload import is_cpu_offload_enabled

from transformer_engine.pytorch.tensor.quantized_tensor import (
    QuantizedTensor,
)

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
def backward_custom(self, inp, grad_output, is_first_microbatch=None, fp8_output=False, fp8_grad=False):
    #recompute forward before gemm
    if FP8GlobalStateManager.fp8_graph_capturing():
        skip_fp8_weight_update = FP8GlobalStateManager.get_skip_fp8_weight_update_tensor()
    else:
        skip_fp8_weight_update = None
    if skip_fp8_weight_update is not None:
        is_first_microbatch = False
    with self.prepare_forward(
            inp,
            allow_non_contiguous=isinstance(inp, QuantizedTensor),
        ) as inp:

        # Get concatenated weight and bias tensors
        unfused_weights = [getattr(self, name) for name in self.weight_names]
        if any(isinstance(w, QuantizedTensor) for w in unfused_weights):
            if self.fp8:
                if len(unfused_weights) != 1:
                    raise RuntimeError(
                        "Splitting QuantizedTensor into multiple params is not supported"
                    )
            else:
                unfused_weights = [w.dequantize() for w in unfused_weights]
        weight_tensor = noop_cat(unfused_weights)
        if self.use_bias:
            bias_tensor = noop_cat([getattr(self, name) for name in self.bias_names])
        else:
            bias_tensor = None

        (
            input_quantizer,
            weight_quantizer,
            output_quantizer,
            grad_output_quantizer,
            grad_input_quantizer,
        ) = self._get_quantizers(fp8_output, fp8_grad)

        # Make sure weight tensor has correct quantizer
        # Note: Quantizer might have changed if quantization
        # recipe changed
        if weight_quantizer is not None and isinstance(weight_tensor, QuantizedTensor):
            weight_tensor._quantizer = weight_quantizer
        
        #input args of _Linear-forward
        weight = weight_tensor 
        bias = bias_tensor if (self.apply_bias and not self.gemm_bias_unfused_add) else None
        fp8 = self.fp8
        fp8_calibration = self.fp8_calibration
        cpu_offloading = is_cpu_offload_enabled()
        tp_group = self.tp_group
        tp_size = self.tp_size
        sequence_parallel = self.sequence_parallel
        activation_dtype = self.activation_dtype
        parallel_mode = self.parallel_mode
        is_grad_enabled = torch.is_grad_enabled()
        ub_overlap_rs_fprop = self.ub_overlap_rs_fprop
        ub_overlap_ag = self.ub_overlap_ag_dgrad
        ub_overlap_ag_fprop = self.ub_overlap_ag_fprop
        ub_overlap_rs_dgrad = self.ub_overlap_rs_dgrad
        ub_bulk_dgrad = self.ub_bulk_dgrad
        ub_bulk_wgrad = self.ub_bulk_wgrad
        ub_name = self.ub_name
        fsdp_group = self.fsdp_group
        use_bias = self.use_bias

        ## code in _Linear-forward
        
        # Make sure input dimensions are compatible
        out_features, in_features = weight.shape
        inp_shape = inp.shape
        assert inp_shape[-1] == in_features, "GEMM not possible"

        tp_world_size = get_distributed_world_size(tp_group)
        backward_needs_input = is_grad_enabled and weight.requires_grad

        # Prepare input tensor
        # Note: Cast to expected dtype and perform tensor-parallel communication
        inputmat = inp
        inputmat_total = None
        with_input_all_gather_nccl = (
            parallel_mode == "column" and sequence_parallel and not ub_overlap_ag_fprop
        )
        own_quantized_input = False
        if fp8:
            if (
                any([ub_overlap_ag_fprop, ub_overlap_rs_fprop])
                and not FP8GlobalStateManager.get_fp8_recipe().delayed()
            ):
                raise NotImplementedError(
                    "Comm+GEMM overlap is only supported with FP8 delayed scaling"
                )

            if input_quantizer is None:
                raise ValueError("Missing quantizer for input tensor")
            if with_input_all_gather_nccl:
                assert not isinstance(
                    inputmat, QuantizedTensor
                ), "All gather of fp8 input is not supported"
                input_quantizer.set_usage(rowwise=True, columnwise=False)
                inputmat_total, _ = gather_along_first_dim(
                    inputmat,
                    tp_group,
                    quantizer=input_quantizer,
                )
            else:
                input_quantizer.set_usage(
                    rowwise=True,
                    columnwise=backward_needs_input,
                )
                if not isinstance(inputmat, QuantizedTensor):
                    inputmat = input_quantizer(inputmat)
                elif backward_needs_input:
                    inputmat.update_usage(rowwise_usage=True, columnwise_usage=True)
                inputmat_total = inputmat
        else:
            inputmat = cast_if_needed(inp, activation_dtype)
            if with_input_all_gather_nccl:
                inputmat_total, _ = gather_along_first_dim(inputmat, tp_group)
            else:
                inputmat_total = inputmat

        # Cast weight to expected dtype
        weightmat = weight
        if not fp8:
            weightmat = cast_if_needed(weightmat, activation_dtype)
        else:
            if not isinstance(weight, QuantizedTensor):
                # Configure quantizer
                if weight_quantizer is not None:
                    columnwise_usage = is_grad_enabled and inp.requires_grad
                    if not columnwise_usage:
                        columnwise_usage = (
                            is_fp8_activation_recompute_enabled()
                            and not in_fp8_activation_recompute_phase()
                        )
                    weight_quantizer.set_usage(rowwise=True, columnwise=columnwise_usage)

                # FP8 cast to workspace buffer
                update_workspace = is_first_microbatch is None or is_first_microbatch
                weightmat = self.get_weight_workspace(
                    tensor=weight,
                    quantizer=weight_quantizer,
                    cache_name=(None if is_first_microbatch is None else "weight"),
                    update_workspace=update_workspace,
                    skip_update_flag=skip_fp8_weight_update,
                    fsdp_group=fsdp_group,
                )

        # Cast bias to expected dtype
        bias_dtype = activation_dtype
        if fp8 and activation_dtype == torch.float32:
            bias_dtype = torch.bfloat16
        bias = cast_if_needed(bias, bias_dtype) if bias is not None else bias

        # Configure output quantizer
        if output_quantizer is not None:
            output_quantizer.set_usage(rowwise=True, columnwise=False)

        # Calibrate quantizers if needed
        if not fp8 and fp8_calibration:
            if input_quantizer is not None:
                input_quantizer.calibrate(inputmat_total)
            if weight_quantizer is not None:
                weight_quantizer.calibrate(weight)

        ub_obj = None
        ub_type = None
        rs_out = None
        out_dtype = activation_dtype
        if ub_overlap_rs_fprop:
            ub_obj = get_ub(ub_name + "_fprop")
            ub_type = tex.CommOverlapType.RS
            out_shape = [reduce(multiply_op, inp_shape[:-1]) // tp_world_size, out_features]
            rs_out = torch.empty(out_shape, dtype=activation_dtype, device=inputmat_total.device)

        elif ub_overlap_ag_fprop:
            ub_obj = get_ub(ub_name + "_fprop")
            ub_type = tex.CommOverlapType.AG
            if fp8:
                assert ub_obj.is_fp8_ubuf(), "AG overlap with FP8 GEMM inputs requires FP8 buffer."
            ub_obj.copy_into_buffer(inputmat_total, input_quantizer, local_chunk=True)
            inputmat_total = ub_obj.get_buffer(input_quantizer)
        
        tensors_to_save, tensor_objects = prepare_for_saving(
                inputmat,
                weightmat,
                weight,
                bias,
            ) #will not save actually, only to match the code format
        saved_tensors = tensors_to_save
        fuse_wgrad_accumulation = self.fuse_wgrad_accumulation
        if fuse_wgrad_accumulation and weight.requires_grad:
            main_grad = weight.main_grad
        requires_dgrad = inp.requires_grad
        requires_wgrad = weight.requires_grad
        reduce_and_update_bwd_fp8_tensors = False
        # owns_input = saved_inputmat is not inp
        owns_input = True # set True mannually, inp is not need after custom backward anyway
        # owns_input = False #set False manually now, because we do not cache any tensor in ctx, so clear_tensor_data not needed 
        is_input_fp8 = not own_quantized_input
        if fp8 and requires_grad(inp, weight, bias):
            _first_fp8_module = FP8GlobalStateManager.IS_FIRST_FP8_MODULE
            reduce_and_update_bwd_fp8_tensors = FP8GlobalStateManager.is_first_fp8_module()
            if in_fp8_activation_recompute_phase():
                FP8GlobalStateManager.IS_FIRST_FP8_MODULE = _first_fp8_module


    with torch.cuda.nvtx.range("_Linear_backward"):
        if (
            fp8
            and any(
                [
                    ub_overlap_ag,
                    ub_overlap_rs_dgrad,
                    ub_bulk_dgrad,
                    ub_bulk_wgrad,
                ]
            )
            and not FP8GlobalStateManager.get_fp8_recipe().delayed()
        ):
            raise NotImplementedError(
                "Comm+GEMM overlap is only supported with FP8 delayed scaling"
            )

        inputmat, weight_fp8, weight, bias = (  # pylint: disable=unbalanced-tuple-unpacking
            restore_from_saved(tensor_objects, saved_tensors)
        )

        # Since main_grad can be modified inplace, it should not be a part of saved_tensors
        main_grad = (
            main_grad
            if weight is not None and fuse_wgrad_accumulation and requires_wgrad
            else None
        )

        if cpu_offloading and fuse_wgrad_accumulation:
            weight = torch.nn.Parameter(weight, weight.requires_grad)
            weight.main_grad = main_grad

        # Gather intermediate/activation tensors if needed
        # NOTE: weight_fp8 = weight when fp8 == False and torch.disttributed.FSDP already
        #       shards/unshards the base weights so we don't do it ourselves
        assert fsdp_group is None, 'not support fsdp in backward_custom'

        ub_obj_gradout = None
        ub_obj_dgrad = None
        ub_obj_wgrad = None
        ub_type_dgrad = None
        ub_type_wgrad = None
        dgrad_shape = [reduce(multiply_op, inp_shape[:-1]), inp_shape[-1]]
        rs_out = None
        dgrad_bulk = None
        if ub_overlap_ag:
            # Overlap grad_output all-gather with dgrad compute
            ub_obj_gradout = get_ub(ub_name + "_dgrad")
            ub_obj_dgrad = ub_obj_gradout
            ub_type_dgrad = tex.CommOverlapType.AG

        elif ub_overlap_rs_dgrad:
            # Overlap dgrad reduce-scatter with dgrad compute
            ub_obj_gradout = get_ub(ub_name + "_dgrad")
            ub_obj_dgrad = ub_obj_gradout
            ub_type_dgrad = tex.CommOverlapType.RS
            rs_out = torch.empty(
                dgrad_shape, dtype=activation_dtype, device=grad_output.device
            )

        else:
            if ub_bulk_dgrad:
                # Overlap inputmat all-gather with dgrad compute
                # NOTE: Copying into communication buffer will always prefer rowwise data,
                #       and will copy columnwise data if rowwise does not exist. In that case,
                #       the all-gather will apply to the leading dimension of the transpose,
                #       which then needs to be interleaved correctly before WGRAD.
                ub_obj_gradout = get_ub(ub_name + "_dgrad")
                ub_obj_dgrad = ub_obj_gradout
                ub_type_dgrad = tex.CommOverlapType.AG
                ub_obj_dgrad.copy_into_buffer(inputmat, input_quantizer, local_chunk=True)

            if ub_bulk_wgrad:
                # Overlap dgrad reduce-scatter with wgrad compute
                ub_obj_wgrad = get_ub(ub_name + "_wgrad")
                ub_type_wgrad = tex.CommOverlapType.RS
                ub_obj_wgrad.set_buffer_params(grad_input_quantizer)
                dgrad_bulk = ub_obj_wgrad.get_buffer(grad_input_quantizer)

        # Prepare grad output tensor
        # Note: Cast to expected dtype and perform tensor-parallel communication
        grad_bias = None
        if grad_output_quantizer is not None:
            grad_output_quantizer.set_usage(rowwise=True, columnwise=True)
            if use_bias:
                grad_output, grad_bias = tex.bgrad_quantize(
                    grad_output, grad_output_quantizer
                    )
            else:
                # grad_output = grad_output_quantizer(grad_output) #same usage as input
                
                # usage copy from gourpedlinear
                grad_output = tex.fused_multi_quantize(
                    [grad_output],
                    None,
                    [grad_output_quantizer],
                    TE_DType[activation_dtype],
                )
                grad_output = grad_output[0]
                
        # Prepare input tensor
        # Note: Perform tensor-parallel communication if needed
        inputmat_total = None
        inputmat_total_work = None
        if (
            requires_wgrad
            and parallel_mode == "column"
            and sequence_parallel
            and not ub_bulk_dgrad
        ):
            quantizer = None
            if fp8:
                quantizer = input_quantizer
                quantizer.set_usage(rowwise=True, columnwise=True)
            inputmat_total, inputmat_total_work = gather_along_first_dim(
                inputmat,
                tp_group,
                async_op=True,
                quantizer=quantizer,
            )
        else:
            inputmat_total = inputmat

        # Check whether to output wgrad GEMM directly into main grad
        if is_first_microbatch is not None:
            accumulate_wgrad_into_param_main_grad = (
                fuse_wgrad_accumulation and not is_first_microbatch
            )
        else:
            accumulate_wgrad_into_param_main_grad = fuse_wgrad_accumulation

        # Compute grad input tensor
        dgrad = None
        dgrad_work = None
        if requires_dgrad:

            # Update quantizer
            if grad_input_quantizer is not None:
                grad_input_quantizer.set_usage(rowwise=True, columnwise=False)
            # dgrad GEMM
            dgrad, *_, rs_out = general_gemm(
                weight_fp8,
                grad_output,
                get_workspace(),
                layout="NN",
                grad=True,
                quantization_params=grad_input_quantizer,
                out=dgrad_bulk,
                out_dtype=activation_dtype,
                use_split_accumulator=_2X_ACC_DGRAD,
                ub=ub_obj_dgrad,
                ub_type=ub_type_dgrad,
                extra_output=rs_out,
                bulk_overlap=ub_bulk_dgrad,
            )

            # Launch tensor-parallel communication
            if ub_overlap_rs_dgrad:
                dgrad = rs_out
            elif parallel_mode == "column" and not ub_bulk_wgrad:
                if sequence_parallel:
                    dgrad, dgrad_work = reduce_scatter_along_first_dim(
                        dgrad,
                        tp_group,
                        async_op=True,
                    )
                else:
                    dgrad, dgrad_work = allreduce(dgrad, tp_group, async_op=True)

        # Compute grad weight tensor
        wgrad = None
        if requires_wgrad:
            if ub_bulk_dgrad:
                inputmat_total = ub_obj_dgrad.get_buffer(input_quantizer)
                if fp8:
                    if inputmat._data is None:
                        # All-gather executed on columnwise data and result is in rowwise data,
                        # so we need to fix the interleaving before WGRAD.
                        inputmat_total = _fix_gathered_fp8_transpose(
                            inputmat_total, tp_size
                        )
                    elif not non_tn_fp8_gemm_supported():
                        # FP8 GEMM on Hopper only supports TN layout so the gathered input must
                        # have a valid transpose.
                        inputmat_total._create_transpose()

            else:
                if inputmat_total_work is not None:
                    # Synchronize tensor-parallel communication
                    inputmat_total_work.wait()
                    inputmat_total_work = None

            if isinstance(grad_output, QuantizedTensor):
                # This is a no-op if platform supports non-TN FP8 GEMM or the transpose
                # already exists.
                grad_output.update_usage(rowwise_usage=True, columnwise_usage=True)

            if ub_bulk_wgrad and ub_obj_wgrad.is_fp8_ubuf():
                rs_out = torch.empty(
                    dgrad_shape, dtype=activation_dtype, device=grad_output.device
                )

            # wgrad GEMM
            # Note: Fuse with bgrad computation if needed
            wgrad, grad_bias_, _, rs_out = ceg.general_gemm(
                inputmat_total,
                grad_output,
                get_workspace(),
                layout="NT",
                grad=True,
                out_dtype=(
                    main_grad.dtype if fuse_wgrad_accumulation else activation_dtype
                ),
                bias=(bias if (grad_bias is None and not fp8) else None),
                out=main_grad if fuse_wgrad_accumulation else None,
                use_split_accumulator=_2X_ACC_WGRAD,
                accumulate=accumulate_wgrad_into_param_main_grad,
                ub=ub_obj_wgrad,
                ub_type=ub_type_wgrad,
                extra_output=rs_out,
                bulk_overlap=ub_bulk_wgrad,
            )

            if ub_bulk_wgrad:
                if ub_obj_wgrad.is_fp8_ubuf():
                    dgrad = rs_out
                else:
                    dgrad = ub_obj_wgrad.get_buffer(grad_input_quantizer, local_chunk=True)

            if grad_bias is None:
                grad_bias = grad_bias_
            del grad_bias_

            # Deallocate input tensor
            if os.getenv("ENABLE_ZERO_BUBBLE", "0") == "0":
                if owns_input:
                    clear_tensor_data(inputmat_total)

        # Don't return grad bias if not needed
        if not use_bias:
            grad_bias = None

        # Synchronize tensor parallel communication
        if inputmat_total_work is not None:
            inputmat_total_work.wait()
            inputmat_total_work = None
        if dgrad_work is not None:
            dgrad_work.wait()
            dgrad_work = None

    if requires_wgrad:
        # Handle custom DDP from mcore.
        if (
            fuse_wgrad_accumulation
            and weight is not None
            and hasattr(weight, "grad_added_to_main_grad")
        ):
            weight.grad_added_to_main_grad = True
            if getattr(weight, "zero_out_wgrad", False):
                wgrad = torch.zeros(
                    weight.main_grad.shape,
                    dtype=weight.dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )
            else:
                wgrad = None
        elif fuse_wgrad_accumulation:
            wgrad = None
    else:
        wgrad = None

    if reduce_and_update_bwd_fp8_tensors and not is_graph_capturing():
        FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)

    # Scatter fp8 weight buffers
    if fp8 and not isinstance(weight, QuantizedTensor):
        _fsdp_scatter_tensors(fsdp_group, weight_fp8)
    dgrad = dgrad.view(inp.shape) if inp.requires_grad else None
    inp.grad = dgrad #TODO: really need set grad for input? will it cause memory leak?
    #call post-backward hook mannually
    if weight.requires_grad and not self.fuse_wgrad_accumulation: 
        weight.grad = wgrad
        if weight.grad is not None and (
                not weight.grad_added_to_main_grad or getattr(weight, 'zero_out_wgrad', False)
            ):
                weight.main_grad.add_(weight.grad.data)
        weight.grad = None  
    return dgrad
# HACK(huang.huang)

from ..utils import add_attr
add_attr(Linear, "backward_custom", backward_custom)
