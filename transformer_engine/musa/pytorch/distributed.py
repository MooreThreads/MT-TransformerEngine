# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Methods needed for distributed training (DP/TP)."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import torch
from torch.utils.checkpoint import detach_variable, noop_context_fn
from torch.nn import Identity

from transformer_engine.pytorch.constants import dist_group_type
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager, fp8_autocast
from transformer_engine.pytorch.utils import safely_set_viewless_tensor_data
from transformer_engine.pytorch.distributed import (
    _get_cuda_rng_state, _get_active_autocast_contexts, activation_recompute_forward,
    gather_split_1d_tensor, split_tensor_into_1d_equal_chunks, _set_cuda_rng_state,
    has_te_modules,)

_USE_REENTRANT_ACTIVATION_RECOMPUTE = True

# HACK(huang.huang): recompute-variance for [somefunc+fa] and [somefunc+linear/groupedLinear], 
# which can save a forward for fa/linear when backward recompute 
# 2025.4.7: support list of linear as last_function, and args "mid_function" to support complex situations
class IdentityTupleOp(torch.nn.Module):
    """
    This is a placeholder for IdentityTupleOp(*args) -> args,
    """

    def __init__(self,):
        super().__init__()

    def forward(self, *args):
        return args

class _CheckpointFunctionVirance(torch.autograd.Function):
    """This function is adapted from torch.utils.checkpoint with
    two main changes:
        1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`
        2) the states in the model parallel tracker are also properly
           tracked/set/reset.
    """

    @staticmethod
    def forward(
        ctx,
        run_function: Callable,
        last_function: Union[Callable, tuple[Callable]],
        mid_function: Union[Callable, tuple[Callable]],
        distribute_saved_activations: bool,
        get_rng_state_tracker: Union[Callable, None],
        tp_group: Union[dist_group_type, None],
        context_fn: Union[Callable, None],
        kwargs: Dict[str, Any],
        *args: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        """Call forward function while saving state to be able to
        redo the computation later."""
        if not isinstance(last_function, tuple):
            last_function = (last_function, )
        mid_function = tuple(IdentityTupleOp() for _ in last_function) if mid_function is None else mid_function       
        ctx.run_function = run_function
        ctx.last_function = last_function 
        ctx.mid_function = mid_function
        ctx.distribute_saved_activations = distribute_saved_activations

        # Copy the rng states.
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        ctx.fwd_cuda_rng_state = _get_cuda_rng_state(graph_safe=False)
        if get_rng_state_tracker is not None:
            ctx.fwd_cuda_rng_state_tracker = get_rng_state_tracker().get_states()

        if context_fn is not None:
            forward_ctx, recompute_ctx = context_fn()
        else:
            forward_ctx, recompute_ctx = noop_context_fn()
        # Preserve torch autocast context for the backward pass
        torch_gpu_amp_ctx, torch_cpu_amp_ctx = _get_active_autocast_contexts()
        with torch.no_grad(), forward_ctx:
            with activation_recompute_forward(activation_recompute=True, recompute_phase=False):
                outputs = run_function(*args)
                outputs = outputs if isinstance(outputs, tuple) else (outputs, )
                total_outputs = []
                for i, func in enumerate(last_function):
                    outputs_f = mid_function[i](*outputs)
                    outputs_f = outputs_f if isinstance(outputs_f, tuple) else (outputs_f, )
                    outputs_f = func(*outputs_f)
                    total_outputs.append(outputs_f)
                if len(total_outputs)==1:
                    #maintain original behavior when only one last_function 
                    total_outputs=total_outputs[0] 
                else:
                    flat_outputs = []
                    for outputs_f in total_outputs:
                        if isinstance(outputs_f, tuple):
                            #Manually remove bias_out which is 'None', and assign 'None' to grad-bias in the corresponding backward direction
                            outputs_f = tuple([x for x in outputs_f if x is not None])         
                        flat_outputs.append(outputs_f)   
                    total_outputs = flat_outputs
                    #The reentrant version does not consider tensors in nested structures (e.g., custom objects, lists, dicts, etc) 
                    # as participating in autograd, while the non-reentrant version does
                    total_outputs = sum( [x if isinstance(x, tuple) else (x,) for x in total_outputs ], tuple()) 
        # Divide hidden states across model parallel group and only keep
        # the chunk corresponding to the current rank.
        if distribute_saved_activations:
            ctx.input_0_shape = args[0].data.shape
            safely_set_viewless_tensor_data(
                args[0],
                split_tensor_into_1d_equal_chunks(args[0].data, tp_group, new_buffer=True),
            )

        # Store everything.
        ctx.inputs = [arg if not torch.is_tensor(arg) else None for arg in args]
        tensor_inputs = [arg if torch.is_tensor(arg) else None for arg in args]
        ctx.save_for_backward(*tensor_inputs)

        fp8 = FP8GlobalStateManager.is_fp8_enabled()
        ctx.get_rng_state_tracker = get_rng_state_tracker
        ctx.tp_group = tp_group
        ctx.recompute_ctx = recompute_ctx
        ctx.torch_gpu_amp_ctx = torch_gpu_amp_ctx
        ctx.torch_cpu_amp_ctx = torch_cpu_amp_ctx
        ctx.fp8 = fp8
        ctx.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe() if fp8 else None
        ctx.kwargs = kwargs
        
        return total_outputs

    @staticmethod
    def backward(
        ctx, *args: Tuple[Union[torch.Tensor, None], ...]
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        """Call backward function with activation recomputation."""
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), please use .backward() if possible"
            )

        inputs = tuple(
            t if t is not None else arg for (t, arg) in zip(ctx.saved_tensors, ctx.inputs)
        )
        get_rng_state_tracker = ctx.get_rng_state_tracker

        if ctx.distribute_saved_activations:
            safely_set_viewless_tensor_data(
                inputs[0],
                gather_split_1d_tensor(inputs[0].data, ctx.tp_group).view(ctx.input_0_shape),
            )

        # Store the current states.
        bwd_cpu_rng_state = torch.get_rng_state()
        bwd_cuda_rng_state = _get_cuda_rng_state(graph_safe=False)
        if get_rng_state_tracker is not None:
            bwd_cuda_rng_state_tracker = get_rng_state_tracker().get_states()

        # Set the states to what it used to be before the forward pass.
        torch.set_rng_state(ctx.fwd_cpu_rng_state)
        _set_cuda_rng_state(ctx.fwd_cuda_rng_state, graph_safe=False)
        if get_rng_state_tracker is not None:
            get_rng_state_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

        # Compute the forward pass.
        detached_inputs = detach_variable(inputs)
        # ori_outputs is not requires_grad

        with torch.enable_grad(), ctx.recompute_ctx, ctx.torch_gpu_amp_ctx, ctx.torch_cpu_amp_ctx, activation_recompute_forward(
            activation_recompute=True, recompute_phase=True
        ), fp8_autocast(
            enabled=ctx.fp8, fp8_recipe=ctx.fp8_recipe
        ):
            outputs = ctx.run_function(*detached_inputs)
            outputs = outputs if isinstance(outputs, tuple) else (outputs, )
            total_outputs = []
            for i,func in enumerate(ctx.mid_function):
                outputs_f = func(*outputs)
                if isinstance(outputs_f, torch.Tensor):
                    outputs_f = [outputs_f,]
                total_outputs.append(outputs_f)
            # Set the states back to what it was at the start of this function.
            torch.set_rng_state(bwd_cpu_rng_state)
            _set_cuda_rng_state(bwd_cuda_rng_state, graph_safe=False)
            if get_rng_state_tracker is not None:
                get_rng_state_tracker().set_states(bwd_cuda_rng_state_tracker)


            #backward_custom need to be executed under this context while something like self.fp8 will change outside of context
            total_grad_input = []
            for i,func in enumerate(ctx.last_function):
                if isinstance(func, Identity):
                    grad_input_f = args[i]
                else:
                    grad_out_bias = args[i] if isinstance(args[i], tuple) else (args[i], None)
                    grad_input_f = func.backward_custom(*total_outputs[i], *grad_out_bias)
                if isinstance(grad_input_f, torch.Tensor):
                    grad_input_f = (grad_input_f,)
                total_grad_input.append(grad_input_f)


        total_outputs_with_grad = []
        total_args_with_grad = []
        for j, outputs in enumerate(total_outputs):
            outputs_with_grad = []
            args_with_grad = []
            for i, output in enumerate(outputs):
                if torch.is_tensor(output) and output.requires_grad:
                    outputs_with_grad.append(output)
                    args_with_grad.append(total_grad_input[j][i])    
            total_outputs_with_grad += outputs_with_grad
            total_args_with_grad += args_with_grad

        if len(total_outputs_with_grad) == 0:
            raise RuntimeError(
                "none of output has requires_grad=True, this checkpoint() is not necessary"
            )
        torch.autograd.backward(total_outputs_with_grad, total_args_with_grad)

        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None for inp in detached_inputs
        )
        return (None, None, None, None, None, None, None, None) + grads


@torch._disable_dynamo
def checkpointVirance(
    function: Callable,
    last_function: Callable,
    *args: Tuple[torch.Tensor, ...],
    mid_function=None,
    **kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, ...]:
    """
    Checkpoint a part of the model by trading compute for memory. This function is based on
    `torch.utils.checkpoint.checkpoint <https://pytorch.org/docs/stable/checkpoint.html>`_.

    .. warning::

        It is the user's responsibility to ensure identical behavior when calling
        :attr:`function` from the forward and backward pass. If different output is
        produced (e.g. due to global state), then the checkpointed version won't
        be numerically equivalent.

    .. warning::
        `use_reentrant=False` does not support early stopping, and will execute the entire forward
        pass for the checkpointed module when recomputing activations in the backward pass.

    Parameters
    ----------
    function: Callable
            pytorch module used to run the forward and backward passes using
            the specified :attr:`args` and :attr:`kwargs`.
    distribute_saved_activations: bool, default = False
            if set to `True` and `use_reentrant=True`, first tensor argument is distributed
            across the specified tensor parallel group (`tp_group`) before saving it for the
            backward pass. This has no effect when `use_reentrant=False`.
    get_rng_state_tracker: `Callable`, default = None
            python callable which returns an instance of :func:`CudaRNGStatesTracker`.
    tp_group : ProcessGroup, default = None
            tensor parallel process group. Used only when `distribute_saved_activations=True`
            and `use_reentrant=True`. If `None`, it falls back to the default group.
    use_reentrant : bool, default = True
            perform checkpointing in reentrant mode.
    args : tuple
            tuple of torch tensors for inputs to :attr:`function`.
    kwargs : dict
            dictionary of string keys for keyword arguments to :attr:`function`.
    """
    # Pop out te.distributed.checkpoint() arguments
    global _USE_REENTRANT_ACTIVATION_RECOMPUTE
    _USE_REENTRANT_ACTIVATION_RECOMPUTE = kwargs.pop("use_reentrant", True)
    distribute_saved_activations = kwargs.pop("distribute_saved_activations", False)
    tp_group = kwargs.pop("tp_group", None)
    get_rng_state_tracker = kwargs.pop("get_rng_state_tracker", None)

    # Ensure backward compatibility.
    if (
        len(args) > 3
        and isinstance(args[0], bool)
        and callable(args[1])
        and isinstance(args[2], None | dist_group_type)
    ):
        warnings.warn(
            "Passing non-tensor non-keyword arguments is deprecated and support will be removed in "
            "future releases of TransformerEngine. `distribute_saved_activations`, `tp_group`, and "
            "`get_rng_state_tracker` must be passed as keyword arguments to `checkpoint`.",
            DeprecationWarning,
            stacklevel=2,
        )
        distribute_saved_activations = args[0]
        get_rng_state_tracker = args[1]
        tp_group = args[2]
        args = args[3:]

    # Trigger the native PyTorch checkpoint if the function is not or does not contain a
    # Transformer Engine module.
    context_fn = kwargs.pop("context_fn", noop_context_fn)
    determinism_check = kwargs.pop("determinism_check", "default")
    debug = kwargs.pop("debug", False)
    
    assert has_te_modules(function) and has_te_modules(last_function), "only support when has te modules"

    # If this TE module is FSDP-wrapped, clear its FSDP group information because there's no need
    # to scatter/gather activations that we will recompute anyway.
    setattr(function, "fsdp_wrapped", False)
    setattr(function, "fsdp_group", None)
    if isinstance(last_function, tuple):
        for func in last_function:
            setattr(func, "fsdp_wrapped", False)
            setattr(func, "fsdp_group", None)
    else:
        setattr(last_function, "fsdp_wrapped", False)
        setattr(last_function, "fsdp_group", None)
    if mid_function is not None:
        if isinstance(mid_function, tuple):
            setattr(func, "fsdp_wrapped", False)
            setattr(func, "fsdp_group", None)
        else:
            setattr(mid_function, "fsdp_wrapped", False)
            setattr(mid_function, "fsdp_group", None)
    # Otherwise discard unused te.utils.checkpoint.checkpoint() arguments
    # and execute TE's own checkpointing
    # NOTE: This logic uses the TE checkpoint on all custom callable `function` handles because we
    #       cannot be sure there are no TE modules inside the function. It also means we might run
    #       the TE checkpoint for non-TE modules, so the TE checkpoint has to support a potential
    #       user context function.
    del determinism_check, debug
    
    if _USE_REENTRANT_ACTIVATION_RECOMPUTE:
        # If saved activations need to be distributed but there is no process group,
        # default to the world group.
        if distribute_saved_activations:
            assert torch.distributed.is_initialized(), "torch.distributed is not initialized."
            tp_group = torch.distributed.GroupMember.WORLD if tp_group is None else tp_group

        return _CheckpointFunctionVirance.apply(
            function,
            last_function,
            mid_function,
            distribute_saved_activations,
            get_rng_state_tracker,
            tp_group,
            context_fn,
            kwargs,
            *args,
        )


class _CheckpointFunctionViranceAttention(torch.autograd.Function):
    """This function is adapted from torch.utils.checkpoint with
    two main changes:
        1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`
        2) the states in the model parallel tracker are also properly
           tracked/set/reset.
    """

    @staticmethod
    def forward(
        ctx,
        run_function: Callable,
        last_function: Callable,
        distribute_saved_activations: bool,
        get_rng_state_tracker: Union[Callable, None],
        tp_group: Union[dist_group_type, None],
        context_fn: Union[Callable, None],
        kwargs: Dict[str, Any],
        *args: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        """Call forward function while saving state to be able to
        redo the computation later."""
        ctx.run_function = run_function
        ctx.last_function = last_function
        ctx.distribute_saved_activations = distribute_saved_activations

        # Copy the rng states.
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        ctx.fwd_cuda_rng_state = _get_cuda_rng_state(graph_safe=False)
        if get_rng_state_tracker is not None:
            ctx.fwd_cuda_rng_state_tracker = get_rng_state_tracker().get_states()

        if context_fn is not None:
            forward_ctx, recompute_ctx = context_fn()
        else:
            forward_ctx, recompute_ctx = noop_context_fn()

        # Preserve torch autocast context for the backward pass
        torch_gpu_amp_ctx, torch_cpu_amp_ctx = _get_active_autocast_contexts()
        with torch.no_grad(), forward_ctx:
            with activation_recompute_forward(activation_recompute=True, recompute_phase=False):
                outputs = run_function(*args)
                outputs = last_function.forward_before_fa(*outputs[:4], **outputs[4])
                outputs = last_function.forward_fa(*outputs) 
                #outputs: Union[output=Union[Tensor output, Tensor logsumexp, Tensor dropout_mask], 
                # qkv_format, indices_q, batch_size, attn_mask_type, max_seqlen_q, q_shape, v_shape]
                core_attn_out = last_function.forward_after_fa(*outputs)

        # Divide hidden states across model parallel group and only keep
        # the chunk corresponding to the current rank.
        if distribute_saved_activations:
            ctx.input_0_shape = args[0].data.shape
            safely_set_viewless_tensor_data(
                args[0],
                split_tensor_into_1d_equal_chunks(args[0].data, tp_group, new_buffer=True),
            )

        # Store everything.
        ctx.inputs = [arg if not torch.is_tensor(arg) else None for arg in args] + [None]*len(outputs[0]) #pad None to match len of tensor_inputs
        tensor_inputs = [arg if torch.is_tensor(arg) else None for arg in args]
        ctx.save_for_backward(*tensor_inputs, *outputs[0])

        fp8 = FP8GlobalStateManager.is_fp8_enabled()
        ctx.get_rng_state_tracker = get_rng_state_tracker
        ctx.tp_group = tp_group
        ctx.recompute_ctx = recompute_ctx
        ctx.torch_gpu_amp_ctx = torch_gpu_amp_ctx
        ctx.torch_cpu_amp_ctx = torch_cpu_amp_ctx
        ctx.fp8 = fp8
        ctx.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe() if fp8 else None
        ctx.kwargs = kwargs
        (ctx.qkv_format, ctx.indices_q, ctx.batch_size, 
         ctx.attn_mask_type, ctx.max_seqlen_q, ctx.q_shape, ctx.v_shape) = outputs[1:]
        
        return core_attn_out

    @staticmethod
    def backward(
        ctx, *args: Tuple[Union[torch.Tensor, None], ...]
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        """Call backward function with activation recomputation."""
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), please use .backward() if possible"
            )

        inputs = tuple(
            t if t is not None else arg for (t, arg) in zip(ctx.saved_tensors, ctx.inputs)
        )
        fa_output = inputs[-3:]
        inputs = inputs[:-3]
        get_rng_state_tracker = ctx.get_rng_state_tracker

        if ctx.distribute_saved_activations:
            safely_set_viewless_tensor_data(
                inputs[0],
                gather_split_1d_tensor(inputs[0].data, ctx.tp_group).view(ctx.input_0_shape),
            )

        # Store the current states.
        bwd_cpu_rng_state = torch.get_rng_state()
        bwd_cuda_rng_state = _get_cuda_rng_state(graph_safe=False)
        if get_rng_state_tracker is not None:
            bwd_cuda_rng_state_tracker = get_rng_state_tracker().get_states()

        # Set the states to what it used to be before the forward pass.
        torch.set_rng_state(ctx.fwd_cpu_rng_state)
        _set_cuda_rng_state(ctx.fwd_cuda_rng_state, graph_safe=False)
        if get_rng_state_tracker is not None:
            get_rng_state_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

        # Compute the forward pass.
        detached_inputs = detach_variable(inputs)
        detached_ori_outputs = detach_variable(fa_output)
        detached_ori_outputs[0].requires_grad = True #only 0 element need grad in output of FA: [Tensor output, Tensor logsumexp, Tensor dropout_mask]
        # ori_outputs is not requires_grad

        with torch.enable_grad(), ctx.recompute_ctx, ctx.torch_gpu_amp_ctx, ctx.torch_cpu_amp_ctx, activation_recompute_forward(
            activation_recompute=True, recompute_phase=True
        ), fp8_autocast(
            enabled=ctx.fp8, fp8_recipe=ctx.fp8_recipe
        ):
            outputs_before_fa = ctx.run_function(*detached_inputs)
            # outputs_before_fa: query, key, value, attention_mask, {"attn_mask_type":attn_mask_type, "attention_bias":attention_bias, "packed_seq_params":packed_seq_params}
            outputs_before_fa = ctx.last_function.forward_before_fa(*outputs_before_fa[:4], **outputs_before_fa[4])
            outputs = ctx.last_function.forward_after_fa(detached_ori_outputs, 
                                                         ctx.qkv_format, ctx.indices_q,  
                                                         ctx.batch_size, ctx.attn_mask_type, 
                                                         ctx.max_seqlen_q, ctx.q_shape, ctx.v_shape)
        # Set the states back to what it was at the start of this function.
        torch.set_rng_state(bwd_cpu_rng_state)
        _set_cuda_rng_state(bwd_cuda_rng_state, graph_safe=False)
        if get_rng_state_tracker is not None:
            get_rng_state_tracker().set_states(bwd_cuda_rng_state_tracker)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        outputs_with_grad = []
        args_with_grad = []
        for i, output in enumerate(outputs):
            if torch.is_tensor(output) and output.requires_grad:
                outputs_with_grad.append(output)
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError(
                "none of output has requires_grad=True, this checkpoint() is not necessary"
            )

        torch.autograd.backward(outputs_with_grad, args_with_grad)
        
        #costum bwd fa
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
            with torch.no_grad():
                grad_input = torch.ops.aten._scaled_dot_product_attention_flash_musa_backward(
                    # ori_outputs[0][0].grad,
                    detached_ori_outputs[0].grad,
                    *outputs_before_fa[:3], #q, k, v
                    *detached_ori_outputs, #(Tensor output, Tensor logsumexp, Tensor dropout_mask)
                    is_causal="causal" in ctx.attn_mask_type, #causal same as fwd
                ) 

        #bwd before fa: for qkv
        torch.autograd.backward(outputs_before_fa[:3], grad_input)

        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None for inp in detached_inputs
        )
        return (None, None, None, None, None, None, None) + grads


@torch._disable_dynamo
def checkpointViranceAttention(
    function: Callable,
    last_function: Callable,
    *args: Tuple[torch.Tensor, ...],
    **kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, ...]:
    """
    Checkpoint a part of the model by trading compute for memory. This function is based on
    `torch.utils.checkpoint.checkpoint <https://pytorch.org/docs/stable/checkpoint.html>`_.

    .. warning::

        It is the user's responsibility to ensure identical behavior when calling
        :attr:`function` from the forward and backward pass. If different output is
        produced (e.g. due to global state), then the checkpointed version won't
        be numerically equivalent.

    .. warning::
        `use_reentrant=False` does not support early stopping, and will execute the entire forward
        pass for the checkpointed module when recomputing activations in the backward pass.

    Parameters
    ----------
    function: Callable
            pytorch module used to run the forward and backward passes using
            the specified :attr:`args` and :attr:`kwargs`.
    distribute_saved_activations: bool, default = False
            if set to `True` and `use_reentrant=True`, first tensor argument is distributed
            across the specified tensor parallel group (`tp_group`) before saving it for the
            backward pass. This has no effect when `use_reentrant=False`.
    get_rng_state_tracker: `Callable`, default = None
            python callable which returns an instance of :func:`CudaRNGStatesTracker`.
    tp_group : ProcessGroup, default = None
            tensor parallel process group. Used only when `distribute_saved_activations=True`
            and `use_reentrant=True`. If `None`, it falls back to the default group.
    use_reentrant : bool, default = True
            perform checkpointing in reentrant mode.
    args : tuple
            tuple of torch tensors for inputs to :attr:`function`.
    kwargs : dict
            dictionary of string keys for keyword arguments to :attr:`function`.
    """
    # Pop out te.distributed.checkpoint() arguments
    global _USE_REENTRANT_ACTIVATION_RECOMPUTE
    _USE_REENTRANT_ACTIVATION_RECOMPUTE = kwargs.pop("use_reentrant", True)
    distribute_saved_activations = kwargs.pop("distribute_saved_activations", False)
    tp_group = kwargs.pop("tp_group", None)
    get_rng_state_tracker = kwargs.pop("get_rng_state_tracker", None)

    # Ensure backward compatibility.
    if (
        len(args) > 3
        and isinstance(args[0], bool)
        and callable(args[1])
        and isinstance(args[2], None | dist_group_type)
    ):
        warnings.warn(
            "Passing non-tensor non-keyword arguments is deprecated and support will be removed in "
            "future releases of TransformerEngine. `distribute_saved_activations`, `tp_group`, and "
            "`get_rng_state_tracker` must be passed as keyword arguments to `checkpoint`.",
            DeprecationWarning,
            stacklevel=2,
        )
        distribute_saved_activations = args[0]
        get_rng_state_tracker = args[1]
        tp_group = args[2]
        args = args[3:]

    # Trigger the native PyTorch checkpoint if the function is not or does not contain a
    # Transformer Engine module.
    context_fn = kwargs.pop("context_fn", noop_context_fn)
    determinism_check = kwargs.pop("determinism_check", "default")
    debug = kwargs.pop("debug", False)
    
    assert has_te_modules(function) and has_te_modules(last_function), "only support when has te modules"

    # If this TE module is FSDP-wrapped, clear its FSDP group information because there's no need
    # to scatter/gather activations that we will recompute anyway.
    setattr(function, "fsdp_wrapped", False)
    setattr(function, "fsdp_group", None)
    setattr(last_function, "fsdp_wrapped", False)
    setattr(last_function, "fsdp_group", None)
    # Otherwise discard unused te.utils.checkpoint.checkpoint() arguments
    # and execute TE's own checkpointing
    # NOTE: This logic uses the TE checkpoint on all custom callable `function` handles because we
    #       cannot be sure there are no TE modules inside the function. It also means we might run
    #       the TE checkpoint for non-TE modules, so the TE checkpoint has to support a potential
    #       user context function.
    del determinism_check, debug
    
    if _USE_REENTRANT_ACTIVATION_RECOMPUTE:
        # If saved activations need to be distributed but there is no process group,
        # default to the world group.
        if distribute_saved_activations:
            assert torch.distributed.is_initialized(), "torch.distributed is not initialized."
            tp_group = torch.distributed.GroupMember.WORLD if tp_group is None else tp_group

        return _CheckpointFunctionViranceAttention.apply(
            function,
            last_function,
            distribute_saved_activations,
            get_rng_state_tracker,
            tp_group,
            context_fn,
            kwargs,
            *args,
        )
# HACK(huang.huang)

from .utils import add_attr
import transformer_engine
add_attr(transformer_engine.pytorch.distributed, "checkpointViranceAttention", checkpointViranceAttention)
add_attr(transformer_engine.pytorch.distributed, "checkpointVirance", checkpointVirance)