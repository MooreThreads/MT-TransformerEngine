# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Functionality for CPU offloading of tensors saved for backward pass."""
from __future__ import annotations
from contextlib import nullcontext
from typing import Any, Dict, Optional
import math
import torch

from .tensor.float8_tensor import Float8Tensor

__all__ = [
    "get_cpu_offload_context",
    "get_fine_grained_offload_handler",
    "get_fine_grained_offload_context",
    "LaunchReloadFunction",
    "WaitReloadFunction",
    "FineGrainedOffloadLayerCounter",
]

CPUOffloadEnabled = False


def set_offloading_param(tensor, param_name, value):
    """Set the type of the offloading needed for a tensor."""
    assert param_name in ["weight_offloading", "activation_offloading", "fine_grained_offloading"]
    if tensor is None:
        return
    if type(tensor) in [torch.Tensor, torch.nn.Parameter]:
        setattr(tensor, param_name, value)
    else:
        data_tensors = tensor.get_data_tensors()
        for tensor in data_tensors:
            if tensor is not None:
                setattr(tensor, param_name, value)

def has_acivation_offloading_param(tensor):
    for param_name in ["activation_offloading", "fine_grained_offloading"]:
        if hasattr(tensor, param_name):
            return True
    return False


def is_cpu_offload_enabled() -> bool:
    """Check if CPU offloading is currently enabled."""
    return CPUOffloadEnabled


class CpuOffloadSavedTensorHook:
    """Contex-manager that executes a pair of pack/unpack hooks for saved tensors.

    In this context, the ``on_save_for_backward`` method will be called every time
    a tensor is saved for backward (this includes intermediary results saved using
    :func:`~torch.autograd.function._ContextMethodMixin.save_for_backward` but
    also those recorded by a PyTorch-defined operation).

    The ``on_get_saved_tensors`` method will be called when the backward function
    of this op attempts to retrieve the saved tensor from context (this includes
    :func: `torch.Tensor.backward()` or :func: `torch.autograd.grad()`. It takes the
    as input the return value of the ``on_save_for_backward``, and is meant to return
    an identical copy of the tensor being saved by ``on_save_for_backward`` in terms of
    size, device and element values.

    Example:

        >>> import torch
        >>> from typing import Any
        >>>
        >>> class DummyHook(CpuOffloadSavedTensorHook):
        ...
        ...     def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        ...         logging.info("On save", tensor)
        ...         return (tensor,)
        ...
        ...     def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        ...         logging.info("On get", saved_state)
        ...         tensor, = saved_state
        ...         return tensor
        ...
        >>> a = torch.ones(5, requires_grad=True)
        >>> b = torch.ones(5, requires_grad=True) * 2
        >>> with DummyHook():
        ...     y = a * b
        ...
        On save tensor([1., 1., 1., 1., 1.], requires_grad=True)
        On save tensor([2., 2., 2., 2., 2.], grad_fn=<MulBackward0>)
        >>> y.sum().backward()
        On get (tensor([1., 1., 1., 1., 1.], requires_grad=True),)
        On get (tensor([2., 2., 2., 2., 2.], grad_fn=<MulBackward0>),)

    """

    def __init__(self) -> None:
        self.inside_context = False

    def __enter__(self):
        global CPUOffloadEnabled
        CPUOffloadEnabled = True

        self.inside_context = True
        torch._C._autograd._push_saved_tensors_default_hooks(
            self.on_save_for_backward, self.on_get_saved_tensor
        )

    def __exit__(self, *args: Any):
        global CPUOffloadEnabled
        CPUOffloadEnabled = False

        self.inside_context = False
        torch._C._autograd._pop_saved_tensors_default_hooks()

    def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        """On save for backward."""
        raise NotImplementedError(
            "`on_save_for_backward: Callable[[torch.Tensor], Any]`"
            "is not implemented in CpuOffloadHook class. Inherit "
            "this class and implement your custom hooks"
        )

    def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        """On get saved tensor."""
        raise NotImplementedError(
            "`on_get_saved_tensors: Callable[[Any], torch.Tensor]`"
            "is not implemented in CpuOffloadHook class. Inherit "
            "this class and implement your custom hooks"
        )


class CpuOffloadHookWithOffloadHandler(CpuOffloadSavedTensorHook):
    """Context-manager that offloads/recovers tensors through an offload hander.

    The hook just offloads/recovers the tensor object to the handler through `tensor_push`
    and `tensor_pop` interface. How the offload-handler manages the offloading, recovering
    or prefetching timing is transparent to this hook.
    """

    def __init__(
        self,
        offload_handler: OffloadHandler,
        handler_extra_kwargs: Optional[Dict[str, Any]] = None,
        debug: bool = False,
    ) -> None:
        if handler_extra_kwargs is None:
            handler_extra_kwargs = {}
        self.debug: bool = debug
        self.offload_handler: OffloadHandler = offload_handler
        self.handler_extra_kwargs: Dict[str, Any] = handler_extra_kwargs
        super().__init__()

    def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        retrieve_identifier = self.offload_handler.tensor_push(tensor, **self.handler_extra_kwargs)
        return retrieve_identifier

    def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        tensor = self.offload_handler.tensor_pop(saved_state, **self.handler_extra_kwargs)
        return tensor


class OffloadHandler:
    """A base class for CPU offload-handler."""

    def __init__(self) -> None:
        pass

    def tensor_push(self, tensor: torch.Tensor, **kwargs) -> Any:
        """Tensor push."""
        raise NotImplementedError(
            "`tensor_push is not implented in OffloadHandler class. "
            "Inherit this class and implement your custom tensor_push."
        )

    def tensor_pop(self, tensor_tag: Any, **kwargs):
        """Tensor pop."""
        raise NotImplementedError(
            "`tensor_pop is not implented in OffloadHandler class. "
            "Inherit this class and implement your custom tensor_pop."
        )


class GroupCommitFunction(torch.autograd.Function):
    """this is a dummy op with output identical to input.
    However, it is necessary for marking a timepoint for offload handler to
    accomplish all synchronizations. Implementing it as a function is necessary
    because we need to actions in both forward and backward.
    """

    @staticmethod
    def forward(ctx, tensor, cpu_offload_handler):
        # pylint: disable=missing-function-docstring
        cpu_offload_handler.on_group_commit_forward()
        ctx.cpu_offload_handler = cpu_offload_handler
        # return the identical tensor
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        # pylint: disable=missing-function-docstring
        cpu_offload_handler = ctx.cpu_offload_handler
        cpu_offload_handler.on_group_commit_backward()
        return grad_output, None


group_prefetch_offload_commit = GroupCommitFunction.apply


class SynchronizedGroupOffloadHandler(OffloadHandler):
    """Offload Handler that offloads/reloads in a synchronized way.
    The device-to-host and host-to-device copying happen in the same stream
    as the computation kernels, thus the copying will block computation.
    """

    def __init__(
        self, num_offload_group, tensor_need_offloading_checker=(lambda _: True), debug=False
    ) -> None:
        super().__init__()

        self.num_offload_group = num_offload_group
        self.tensor_need_offloading_checker = tensor_need_offloading_checker
        self.debug = debug

        self.groupid_reset()

    def groupid_reset(self):
        """Groupid reset."""
        # Data structures to label saved tensors and book-keep their cpu copies.
        # Currently, on push, create a new cpu tensor and copies; on pop, copies
        # the tensor back to gpu and deletes the cpu tensor.
        # These will increment whenever `group_commit()` is invoked
        self.current_group, self.tensor_count_current_group = (0, 0)
        self.torch_tensor_count = 0
        self.tensor_tag_to_state = {}

    def on_group_commit_forward(self):
        """On group commit forward."""
        # finishing up with updating current group and tensor count
        self.current_group += 1  # increment
        self.tensor_count_current_group = 0  # reset

    def on_group_commit_backward(self):
        """On group commit backward."""
        self.current_group -= 1
        assert self.current_group >= 0

    @staticmethod
    def offload(src_tensor, pin_memory=True):
        """Offload."""
        fp8_offload = isinstance(src_tensor, Float8Tensor)

        cpu_backup = torch.empty(
            src_tensor.size(),
            dtype=torch.uint8 if fp8_offload else src_tensor.dtype,
            layout=src_tensor.layout,
            device="cpu",
            pin_memory=pin_memory,
        )

        if fp8_offload:
            cpu_backup = Float8Tensor.make_like(src_tensor, data=cpu_backup)

        cpu_backup.copy_(src_tensor, non_blocking=pin_memory)
        state = (src_tensor.device, cpu_backup)
        return state

    @staticmethod
    def reload(state, non_blocking=None):
        """Reload."""
        dev, cpu_backup = state
        if non_blocking is None:
            non_blocking = cpu_backup.is_pinned()
        return cpu_backup.to(dev, non_blocking=non_blocking)

    def tensor_push(self, tensor: torch.Tensor, **kwargs):
        """Tensor push."""
        # obtain a unique tensor tag
        tensor_tag = (self.current_group, self.tensor_count_current_group)
        self.tensor_count_current_group += 1
        assert tensor_tag not in self.tensor_tag_to_state
        if self.current_group < self.num_offload_group and self.tensor_need_offloading_checker(
            tensor
        ):
            state = SynchronizedGroupOffloadHandler.offload(tensor)
            self.tensor_tag_to_state[tensor_tag] = state
        else:
            # will be offloaded together after group commit
            self.tensor_tag_to_state[tensor_tag] = tensor

        return tensor_tag

    def tensor_pop(self, tensor_tag, **kwargs):
        """Tensor pop."""
        assert tensor_tag in self.tensor_tag_to_state
        state = self.tensor_tag_to_state.pop(tensor_tag)
        if isinstance(state, tuple):
            tensor = SynchronizedGroupOffloadHandler.reload(state)
        else:
            tensor = state
        return tensor


class AsyncDoubleBufferGroupOffloadHandler(SynchronizedGroupOffloadHandler):
    """Compared to synchronize, this uses more memory because of the buffer but
    achieves better performance due to the overlapping. D2h and h2d copying are
    completely hidden behind computation if computation time of a layer is longer
    than host-device communication time. Bulk offloading with delay and bulk reloading
    with prefetch are implemented."""

    def __init__(
        self,
        num_offload_group,  # must be <= actual number of groups (number of commits)
        num_model_group,
        tensor_need_offloading_checker=(lambda t: True),
        debug=False,
    ) -> None:
        super().__init__(
            num_offload_group=num_offload_group,
            tensor_need_offloading_checker=tensor_need_offloading_checker,
            debug=debug,
        )
        # Number of layers in the model
        self.num_layers = num_model_group
        # Data Structure to maintain reference to activation tensors
        self.tensor_tag_to_buf = {}
        # Tracking the number of layers offloaded
        self.offloaded_group_count = 0
        # Core data structure that decides the window for offloading
        self.layer_window_map = {}

        # Logic to make offloading load balance across computation
        # for optimal CPU/GPU interconnect usage
        constant = 0
        for i in range(self.num_offload_group):
            self.layer_window_map[i] = ((self.num_layers // self.num_offload_group) * (i + 1)) - 1
            if i < (self.num_layers % self.num_offload_group):
                self.layer_window_map[i] += i + 1
                constant = i + 1
            else:
                self.layer_window_map[i] += constant

        # allocate streams and events for synchronization
        self.d2h_stream = torch.cuda.Stream()
        self.h2d_stream = torch.cuda.Stream()

    def tensor_push(self, tensor: torch.Tensor, **kwargs) -> Any:

        torch_stray_tensor = isinstance(
            tensor,
            (
                torch._subclasses.fake_tensor.FakeTensor,
                torch._subclasses.functional_tensor.FunctionalTensor,
            ),
        )

        if not torch_stray_tensor:
            # obtain a unique tensor tag
            tensor_tag = (self.current_group, self.tensor_count_current_group)
            self.tensor_count_current_group += 1
            assert tensor_tag not in self.tensor_tag_to_state

            self.tensor_tag_to_state[tensor_tag] = tensor

            if self.current_group < self.num_offload_group and self.tensor_need_offloading_checker(
                tensor
            ):
                self.tensor_tag_to_buf[tensor_tag] = tensor
        else:
            tensor_tag = (-1, self.torch_tensor_count)
            self.torch_tensor_count += 1
            self.tensor_tag_to_state[tensor_tag] = tensor

        return tensor_tag

    def tensor_pop(self, tensor_tag, **kwargs):
        """Tensor pop."""
        assert tensor_tag in self.tensor_tag_to_state
        tensor = self.tensor_tag_to_state.pop(tensor_tag)
        self.tensor_tag_to_buf.pop(tensor_tag, None)
        # the tensor should have been copied back in on_group_commit_backward()
        # which invokes bulk_reload_group.
        assert not isinstance(tensor, tuple)
        return tensor

    def bulk_offload_group(self, group_to_offload):
        """Bulk offload group."""
        with torch.cuda.stream(self.d2h_stream):
            for tensor_tag, state in self.tensor_tag_to_state.items():
                group_id, _ = tensor_tag
                if group_id == group_to_offload:
                    assert not isinstance(state, tuple)
                    tensor_on_device = state

                    # if offload, return the reference to cpu copy
                    if self.tensor_need_offloading_checker(tensor_on_device):
                        state = SynchronizedGroupOffloadHandler.offload(tensor_on_device)
                        self.tensor_tag_to_state[tensor_tag] = state
                        tensor_on_device.data = torch.Tensor()  # Force to release memory

    def synchronize_on_group_commit_forward(self, current_group):
        """Synchronize on group commit forward."""

        # For the first group, kickstart the offload after we have
        # the first compute completion
        if current_group == 0:
            self.d2h_stream.wait_stream(torch.cuda.current_stream())
            self.bulk_offload_group(current_group)

        # Window map data structure helps us synchronize based on number
        # of layers offloaded
        if self.layer_window_map[self.offloaded_group_count] == current_group:

            # Stream synchronization both ways
            self.d2h_stream.wait_stream(torch.cuda.current_stream())
            torch.cuda.current_stream().wait_stream(self.d2h_stream)

            # Time to free the activation memory after usage
            for tensor_tag, _ in self.tensor_tag_to_buf.items():
                if tensor_tag[0] == self.offloaded_group_count:
                    self.tensor_tag_to_buf[tensor_tag] = None

            # Time to offload the next group
            if self.offloaded_group_count < (self.num_offload_group - 1):
                self.bulk_offload_group(self.offloaded_group_count + 1)

            # Increment the offload group count to keep track
            self.offloaded_group_count += 1

    def on_group_commit_forward(self):
        """This function will cause host device synchronization"""
        # handle synchronization events
        self.synchronize_on_group_commit_forward(self.current_group)

        super().on_group_commit_forward()

    def bulk_reload_group(self, group_to_reload):
        """Bulk reload group."""
        assert group_to_reload < self.num_offload_group

        with torch.cuda.stream(self.h2d_stream):
            # move back tensors
            for tensor_label, state in self.tensor_tag_to_state.items():
                group_id, _ = tensor_label
                if group_id == group_to_reload:
                    if isinstance(state, tuple):
                        recovered_tensor = SynchronizedGroupOffloadHandler.reload(state)
                        self.tensor_tag_to_state[tensor_label] = recovered_tensor

    def on_group_commit_backward(self):
        # first decrement the current group.
        # after last commit in forward, the group will +1; in backward it -1.
        # Finally it should be decremented to 0.
        self.current_group -= 1
        assert self.current_group >= 0

        # Layer window data structure helps us to reload at right times
        if self.layer_window_map[self.offloaded_group_count - 1] == self.current_group:

            # Stream synchronization both ways
            self.h2d_stream.wait_stream(torch.cuda.current_stream())
            torch.cuda.current_stream().wait_stream(self.h2d_stream)

            # Time to reload the next group
            self.bulk_reload_group(self.offloaded_group_count - 1)

            # Decrease the offloading group counter
            self.offloaded_group_count -= 1 if self.offloaded_group_count > 1 else 0

        # Last group computation needs to wait till all the reloads complete
        if self.current_group == 0:
            torch.cuda.current_stream().wait_stream(self.h2d_stream)
            self.offloaded_group_count = 0


def get_cpu_offload_context(
    enabled: bool = False,
    num_layers: int = 1,
    model_layers: int = 1,
    offload_activations: bool = True,
    offload_weights: bool = True,
):
    """
    This function returns the CPU Offload context and the synchronizer function that needs to be
    used after every transformer layer. Returns `nullcontext()` if offloading is not enabled.

    Usage:

    .. code-block:: python

        cpu_offload_context, cpu_offload_synchronizer = get_cpu_offload_context(enabled=True)

        with cpu_offload_context:
            te_layer.forward(inp_tensor)
        cpu_offload_synchronizer()

    Parameters
    ----------
    enabled: bool, default = `False`
             When set to True, CPU Offloading functionality is enabled.
    num_layers: int, default = 1
                Determines the number of transformer layers
                you want to offload activations/weights for.
    model_layers: int, default = 1
                  Number of layers in the model that will be used under this context.
    offload_activations: bool, default = `True`
                         When set to `True`, offloads the activations for the TE layer.
    offload_weights: bool, default = `True`
                     When set to `True`, offloads the weights for the TE layer.

    """

    def tensor_need_offloading_checker_activations(tensor):
        return hasattr(tensor, "activation_offloading")

    # This includes the Gradient Accumulation Buffer
    def tensor_need_offloading_checker_weights(tensor):
        return hasattr(tensor, "weight_offloading")

    def tensor_need_offloading_checker_all(tensor):
        return hasattr(tensor, "activation_offloading") or hasattr(tensor, "weight_offloading")

    if offload_activations and offload_weights:
        tensor_need_offloading_checker = tensor_need_offloading_checker_all
    elif offload_activations:
        tensor_need_offloading_checker = tensor_need_offloading_checker_activations
    elif offload_weights:
        tensor_need_offloading_checker = tensor_need_offloading_checker_weights
    else:
        raise ValueError(
            "CPU Offloading is enabled while it is not "
            "mentioned what to offload (weights/activations)"
        )

    cpu_offload_handler = AsyncDoubleBufferGroupOffloadHandler(
        num_offload_group=num_layers,
        num_model_group=model_layers,
        tensor_need_offloading_checker=tensor_need_offloading_checker,
    )

    def group_prefetch_offload_commit_async(tensor):
        return group_prefetch_offload_commit(tensor, cpu_offload_handler)

    if enabled:
        return (
            CpuOffloadHookWithOffloadHandler(offload_handler=cpu_offload_handler),
            group_prefetch_offload_commit_async,
        )
    return nullcontext(), group_prefetch_offload_commit_async

import queue
class _FineGrainedAsyncDoubleBufferGroupOffloadHandler(OffloadHandler):

    def __init__(self) -> None:
        # Data Structure to maintain reference to activation tensors
        self.tensor_tag_to_state = {}
        # Tracking the number of layers offloaded
        self.current_layer_id = 0
        # Tracking the number of microbatches offloaded
        self.current_microbatch_id = 0

        self.reloading_tensor = {}
        self.to_offload_tensor_tag_queue_dict = {}
        self.to_offload_tensor_queue_dict = {}
        self.to_release_tensor_queue_dict = {}

        # allocate streams and events for synchronization
        self.d2h_stream = None
        self.h2d_stream = None

        self.OFFLOAD_TENSOR_ATTR_KEY = 'fine_grained_offloading'

        self.num_layers = None # num of layers in a PP/VPP stage
        self.pp_size = None
        self.pp_rank = None
        self.num_microbatches = None
        self.available_pin_memory_tensor_pool_id_queue_dict = {}
        self.tensor_tag_to_pin_memory_id = {}

        self.pin_memory_tensor_pool = {}
        self.pin_memory_tensor_pool_metadata = {}
        self.pin_memory_tensor_pool_released = False
        
        self.moe_layer_pattern = []
        self.num_model_chunks = None # vpp size
        
        # (microbatch_id, chunk_id) -> virtual_batch_id_backward (list index of (microbatch_id, chunk_id) in self.schedule_table_backward)
        self.microbatch_id_chunk_id_to_virtual_batch_id_backward = {} 
        
        self.schedule_table_forward = [] # schedule_table [(microbatch_id_, chunk_id)] for forward
        self.schedule_table_backward = [] # schedule_table [(microbatch_id_, chunk_id)] for backward
        self.schedule_table_forward_backward = [] # [(microbatch_id_, chunk_id, "forward" / "backward")]
        
        # (current_microbatch_id, current_layer_id) -> True/False (should offload)
        self.cur_batch_id_cur_layer_id_to_should_offload = {} 
        
        
    def init_by_config(self, config):
        self.config = config
        from megatron.core.parallel_state import get_pipeline_model_parallel_rank, get_pipeline_model_parallel_world_size
        self.pp_rank = get_pipeline_model_parallel_rank()
        self.pp_size = get_pipeline_model_parallel_world_size()
        
        if config.overlap_moe_expert_parallel_comm:
            from megatron.core.pipeline_parallel.utils import get_comm_stream
            self.a2a_stream = get_comm_stream()
            
        import queue
        if config.offload_moe_fc1_input:
            self.available_pin_memory_tensor_pool_id_queue_dict['moe_fc1_input'] = queue.Queue()
            self.to_offload_tensor_tag_queue_dict['moe_fc1_input'] = queue.Queue()
            self.to_offload_tensor_queue_dict['moe_fc1_input'] = queue.Queue()
            self.to_release_tensor_queue_dict['moe_fc1_input'] = queue.Queue()
        if config.offload_moe_fused_swiglu_input:
            self.available_pin_memory_tensor_pool_id_queue_dict['moe_fused_swiglu_input'] = queue.Queue()
            self.to_offload_tensor_tag_queue_dict['moe_fused_swiglu_input'] = queue.Queue()
            self.to_offload_tensor_queue_dict['moe_fused_swiglu_input'] = queue.Queue()
            self.to_release_tensor_queue_dict['moe_fused_swiglu_input'] = queue.Queue()

    def release_cpu_pinmem_pool(self):
        torch.cuda.synchronize()
        self.tensor_tag_to_state.clear()

        while self.pin_memory_tensor_pool:
            key, buffer = self.pin_memory_tensor_pool.popitem()
            self.pin_memory_tensor_pool_metadata[key] = (
                buffer.shape, 
                buffer.dtype,
                buffer.layout)
            import sys
            del buffer
        self.pin_memory_tensor_pool.clear()
        
        import gc
        gc.collect()
        import torch_musa
        torch_musa._MUSAC._host_emptyCache()
        torch.distributed.barrier()
        self.pin_memory_tensor_pool_released = True
    
    
    def maybe_resume_cpu_pinmem_pool(self):
        torch.cuda.synchronize()
        self.pin_memory_tensor_pool = {}
        if self.pin_memory_tensor_pool_released:
            torch.distributed.barrier()
            for key in self.pin_memory_tensor_pool_metadata.keys():
                metadata = self.pin_memory_tensor_pool_metadata[key]
                self.pin_memory_tensor_pool[key] = torch.empty(
                        metadata[0],
                        dtype=metadata[1],
                        layout=metadata[2],
                        device="cpu",
                        pin_memory=True,
                    )
            torch.distributed.barrier()
            self.pin_memory_tensor_pool_released = False
    
    def print_pin_memory_tensor_pool_memory_usage(self, tag=""):
        import os
        total_bytes = 0
        for buffer in self.pin_memory_tensor_pool.values():
            if buffer is not None:
                total_bytes += buffer.untyped_storage().nbytes()

        local_rank = os.getenv("LOCAL_RANK")
        if local_rank is None and torch.distributed.is_available() and torch.distributed.is_initialized():
            local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", os.getenv("GPUS_PER_NODE", "8")))
            local_rank = str(torch.distributed.get_rank() % local_world_size)
        if local_rank is None:
            local_rank = "unknown"

        tag_text = f" tag={tag}" if tag else ""
        print(
            f"[cpu_act_offload_pinmem_pool] local_rank={local_rank}{tag_text} "
            f"total_memory_gb={total_bytes / (1024 ** 3):.4f}",
            flush=True,
        )
        return total_bytes
    
    
    def is_last_2_pipeline_parallel_stage(self):
        return (self.pp_rank == (self.pp_size - 1)) or (self.pp_rank == (self.pp_size - 2))
    
    
    def is_last_batch_last_layer(self):
        # print(f"self.current_microbatch_id = {self.current_microbatch_id}, self.num_microbatches = {self.num_microbatches}, self.current_layer_id = {self.current_layer_id}, self.num_layers = {self.num_layers}")
        return self.current_microbatch_id >= self.num_microbatches - 1 and self.current_layer_id >= len(self.moe_layer_pattern) - 1
    
    
    def make_should_offload_table(self, num_warmup_microbatches):
        def make_schedule_table_forward_backward():
            virtual_batch_id_fwd = 0
            virtual_batch_id_bwd = 0
            self.schedule_table_forward_backward = []
            for i in range(num_warmup_microbatches):
                self.schedule_table_forward_backward.append((self.schedule_table_forward[virtual_batch_id_fwd][0], self.schedule_table_forward[virtual_batch_id_fwd][1], "forward"))
                virtual_batch_id_fwd +=1
            
            while True:
                if virtual_batch_id_fwd < len(self.schedule_table_forward):
                    self.schedule_table_forward_backward.append((self.schedule_table_forward[virtual_batch_id_fwd][0], self.schedule_table_forward[virtual_batch_id_fwd][1], "forward"))
                    virtual_batch_id_fwd +=1
                if virtual_batch_id_bwd < len(self.schedule_table_backward):
                    self.schedule_table_forward_backward.append((self.schedule_table_backward[virtual_batch_id_bwd][0], self.schedule_table_backward[virtual_batch_id_bwd][1], "backward"))
                    virtual_batch_id_bwd +=1
                if virtual_batch_id_fwd == len(self.schedule_table_forward) and virtual_batch_id_bwd == len(self.schedule_table_backward):
                    break

        def is_moe_layer(layer_id):
            return layer_id < len(self.moe_layer_pattern) and self.moe_layer_pattern[layer_id] == 1
        
        make_schedule_table_forward_backward()
        # (current_microbatch_id, current_layer_id) -> True/False (should offload)
        self.cur_batch_id_cur_layer_id_to_should_offload = {}
        judging_microbatch_id = None
        judging_layer_id = None
        for i in range(len(self.schedule_table_forward_backward)):
            microbatch_id = self.schedule_table_forward_backward[i][0]
            chunk_id = self.schedule_table_forward_backward[i][1]
            is_forward = 1 if self.schedule_table_forward_backward[i][2] == "forward" else 0
            for j in range(self.num_layers):
                layer_id = j + chunk_id * self.num_layers
                if is_moe_layer(layer_id):
                    if is_forward:
                        if judging_microbatch_id == None and judging_layer_id == None:
                            judging_microbatch_id = microbatch_id
                            judging_layer_id = layer_id
                        if judging_microbatch_id != microbatch_id or judging_layer_id != layer_id:
                            self.cur_batch_id_cur_layer_id_to_should_offload[(judging_microbatch_id, judging_layer_id)] = 1
                            judging_microbatch_id = microbatch_id
                            judging_layer_id = layer_id
                    else:
                        if judging_microbatch_id == microbatch_id and judging_layer_id == layer_id:
                            self.cur_batch_id_cur_layer_id_to_should_offload[(judging_microbatch_id, judging_layer_id)] = 0
                            judging_microbatch_id = None
                            judging_layer_id = None

        # The last MoE layer in a (microbatch, chunk) is the first MoE layer whose
        # activation is needed during that chunk's backward.  It can only be
        # reloaded early if another MoE backward happens after its forward and
        # before its own backward.
        last_moe_layer_id_by_chunk_id = {}
        chunk_ids = set()
        for _, chunk_id, _ in self.schedule_table_forward_backward:
            chunk_ids.add(chunk_id)
        for chunk_id in chunk_ids:
            for j in range(self.num_layers):
                layer_id = j + chunk_id * self.num_layers
                if is_moe_layer(layer_id):
                    last_moe_layer_id_by_chunk_id[chunk_id] = layer_id

        moe_events = []
        for microbatch_id, chunk_id, direction in self.schedule_table_forward_backward:
            if direction == "forward":
                layer_range = range(self.num_layers)
            else:
                layer_range = range(self.num_layers - 1, -1, -1)
            for j in layer_range:
                layer_id = j + chunk_id * self.num_layers
                if is_moe_layer(layer_id):
                    moe_events.append((microbatch_id, chunk_id, layer_id, direction))

        last_moe_forward_event_index = {}
        backward_event_index = {}
        for event_index, (microbatch_id, chunk_id, layer_id, direction) in enumerate(moe_events):
            key = (microbatch_id, layer_id)
            if direction == "forward" and last_moe_layer_id_by_chunk_id.get(chunk_id) == layer_id:
                last_moe_forward_event_index[key] = event_index
            elif direction == "backward":
                backward_event_index[key] = event_index

        for key, forward_event_index in last_moe_forward_event_index.items():
            backward_event_index_for_key = backward_event_index.get(key)
            if backward_event_index_for_key == None:
                continue
            has_reload_window = False
            for event_index in range(forward_event_index + 1, backward_event_index_for_key):
                if moe_events[event_index][3] == "backward":
                    has_reload_window = True
                    break
            if not has_reload_window:
                self.cur_batch_id_cur_layer_id_to_should_offload[key] = 0
                judging_microbatch_id, judging_layer_id = key
        
        if self.pp_rank != (self.pp_size - 1):
            batch_id = 0
            layer_id = self.num_model_chunks * self.num_layers - 1
            self.cur_batch_id_cur_layer_id_to_should_offload[(batch_id, layer_id)] = 1
        
        # print(f"self.cur_batch_id_cur_layer_id_to_should_offload = {self.cur_batch_id_cur_layer_id_to_should_offload}")
            
            
    def should_offload(self):
        if self.is_last_batch_last_layer():
            return False
        if self.config.virtual_pipeline_model_parallel_size != None and self.config.virtual_pipeline_model_parallel_size > 1:
            # vpp
            
            whether_offload = self.cur_batch_id_cur_layer_id_to_should_offload.get((self.current_microbatch_id, self.current_layer_id))
            if whether_offload == None or whether_offload == 0:
                return False
            else:
                return True
            # if self.pp_rank == (self.pp_size - 1):
            #     cur_chunk_id = int(self.current_layer_id / self.num_layers)
            #     cur_virtual_layer_id = self.current_layer_id % self.num_layers
            #     if cur_chunk_id == (self.num_model_chunks - 1) and cur_virtual_layer_id == (self.num_layers - 1):
            #         return False
            #     else:
            #         return True
            # return True
            
        else:
            # 1F1B
            return not self.is_last_2_pipeline_parallel_stage()


    def register_offload(self, src_tensor):
        assert hasattr(src_tensor, self.OFFLOAD_TENSOR_ATTR_KEY)
        tensor_name = getattr(src_tensor, self.OFFLOAD_TENSOR_ATTR_KEY)
        tensor_tag = (self.current_microbatch_id, self.current_layer_id, tensor_name)
        self.to_offload_tensor_tag_queue_dict[tensor_name].put(tensor_tag)
        self.to_offload_tensor_queue_dict[tensor_name].put(src_tensor)
        return tensor_tag


    def get_tag_from_name(self, tensor_name):
        tensor_tag = (self.current_microbatch_id, self.current_layer_id, tensor_name)
        return tensor_tag

    
    def launch_offload(self, tensor_name, offloading_microbatch_id = None, offloading_layer_id = None):
        # print(f"[launch_offload] call with tensor_name:{tensor_name}, offloading_microbatch_id:{offloading_microbatch_id}, offloading_layer_id:{offloading_layer_id}")
        # print(f"[launch_offload] current_layer_id={self.current_layer_id}")
        if self.d2h_stream is None:
            self.d2h_stream = torch.cuda.Stream()
        
        if self.current_layer_id >= len(self.moe_layer_pattern) or self.moe_layer_pattern[self.current_layer_id] == 0 or self.to_offload_tensor_queue_dict[tensor_name].empty():
            return

        src_tensor = self.to_offload_tensor_queue_dict[tensor_name].get()
        tensor_tag = self.to_offload_tensor_tag_queue_dict[tensor_name].get()
        copy_done_event = torch.cuda.Event()
        # print(f"[SUCCES launch_offload] on tensor tag (offloading_microbatch_id, offloading_layer_id, tensor_name) : {tensor_tag}")

        token_num = src_tensor.size(0)
        hidden_dim = src_tensor.size()[1:]
                
        if not self.config.overlap_moe_expert_parallel_comm:
            self.d2h_stream.wait_stream(torch.cuda.current_stream())
        
        if self.available_pin_memory_tensor_pool_id_queue_dict[tensor_name].empty():
            pin_memory_id = len(self.pin_memory_tensor_pool)
        else:
            pin_memory_id = self.available_pin_memory_tensor_pool_id_queue_dict[tensor_name].get()
            
        pin_memory_tag = (pin_memory_id, tensor_name)
        self.tensor_tag_to_pin_memory_id[tensor_tag] = pin_memory_id
        
        with torch.cuda.stream(self.d2h_stream):
            existing_buffer = self.pin_memory_tensor_pool.get(pin_memory_tag)
            # existing_buffer = self.pin_memory_tensor_pool.get(tensor_tag)
            if existing_buffer is None or existing_buffer.size() < src_tensor.size():
                buffer_shape = [math.ceil(token_num * 1.1)] + list(hidden_dim)
                new_buffer = torch.empty(
                    buffer_shape,
                    dtype=src_tensor.dtype,
                    layout=src_tensor.layout,
                    device="cpu",
                    pin_memory=True,
                )
                self.pin_memory_tensor_pool[pin_memory_tag] = new_buffer
                # self.pin_memory_tensor_pool[tensor_tag] = new_buffer

            # buffer = self.pin_memory_tensor_pool[tensor_tag]
            buffer = self.pin_memory_tensor_pool[pin_memory_tag]
            buffer[:token_num, ...].copy_(src_tensor.detach(), non_blocking=True)
            cpu_backup = buffer[:token_num, ...]

            copy_done_event.record(stream=self.d2h_stream)

        self.to_release_tensor_queue_dict[tensor_name].put((copy_done_event, src_tensor))

        state = (src_tensor, cpu_backup, copy_done_event, src_tensor.untyped_storage().size())
        self.tensor_tag_to_state[tensor_tag] = state

        return tensor_tag
    
    
    def wait_offload(self, tensor_name, offloading_microbatch_id = None, offloading_layer_id = None):
        # print(f"[wait_offload] call with tensor_name:{tensor_name}, offloading_microbatch_id:{offloading_microbatch_id}, offloading_layer_id:{offloading_layer_id}")
        # print(f"[wait_offload] current_layer_id={self.current_layer_id}")
        if self.current_layer_id >= len(self.moe_layer_pattern) or self.moe_layer_pattern[self.current_layer_id] == 0 or self.to_release_tensor_queue_dict[tensor_name].empty():
            return
        
        copy_done_event, release_src_tensor = self.to_release_tensor_queue_dict[tensor_name].get()
        copy_done_event.synchronize() # TODO: use .wait() to check the stream with copy engine (d2h / h2d / all2all)
        release_src_tensor.untyped_storage().resize_(0)

    
    def get_reloading_microbatch_id_layer_id_from_table(self):
        chunk_id = int(self.current_layer_id / self.num_layers)
        virtual_batch_id_backward = self.microbatch_id_chunk_id_to_virtual_batch_id_backward[(self.current_microbatch_id, chunk_id)]
        cur_virtual_layer_id = self.current_layer_id % self.num_layers
        for virtual_layer_id in range(cur_virtual_layer_id - 1, -1, -1):
            layer_id = virtual_layer_id + chunk_id * self.num_layers
            if layer_id < len(self.moe_layer_pattern) and self.moe_layer_pattern[layer_id] == 1:
                return (self.current_microbatch_id, layer_id)
        
        reloading_virtual_batch_id_backward = virtual_batch_id_backward
        while reloading_virtual_batch_id_backward + 1 < len(self.microbatch_id_chunk_id_to_virtual_batch_id_backward):
            reloading_virtual_batch_id_backward += 1
            reloading_microbatch_id, reloading_chunk_id = self.schedule_table_backward[reloading_virtual_batch_id_backward]
            if reloading_chunk_id == chunk_id:
                return (self.current_microbatch_id + 1, self.current_layer_id)
            else:
                for virtual_layer_id in range(self.num_layers - 1, -1, -1):
                    if self.pp_rank == (self.pp_size - 1) and reloading_chunk_id == (self.num_model_chunks - 1) and virtual_layer_id == (self.num_layers - 1):
                        continue
                    layer_id = virtual_layer_id + reloading_chunk_id * self.num_layers
                    if layer_id < len(self.moe_layer_pattern) and self.moe_layer_pattern[layer_id] == 1:
                        return (reloading_microbatch_id, layer_id)
        # print("get_reloading_microbatch_id_layer_id_from_table returning (-1,-1)")
        return (-1, -1)
            

    def launch_reload(self, tensor_name, reloading_microbatch_id = None, reloading_layer_id = None):
        # print(f"[launch_reload] call with tensor_name:{tensor_name}, reloading_microbatch_id:{reloading_microbatch_id}, reloading_layer_id:{reloading_layer_id}")
        # print(f"[launch_reload] self.current_layer_id = {self.current_layer_id}")
        if reloading_microbatch_id == None and (self.current_layer_id >= len(self.moe_layer_pattern) or self.moe_layer_pattern[self.current_layer_id] == 0):
            return
        
        if self.h2d_stream is None:
            if self.d2h_stream is None:
                self.d2h_stream = torch.cuda.Stream()
            self.h2d_stream = self.d2h_stream
                
        if self.num_model_chunks == None:
            # 1F1B
            # reload actication of layer i-1 in the bwd of layer i (when i > 0)
            # reload actication of last layer of next microbatch in the bwd of layer 0 
            if reloading_microbatch_id == None and reloading_layer_id == None:
                if self.current_layer_id == 0:
                    reloading_microbatch_id = self.current_microbatch_id + 1
                    reloading_layer_id = self.num_layers - 1
                else:
                    reloading_microbatch_id = self.current_microbatch_id
                    reloading_layer_id = self.current_layer_id - 1
                    
            for i in range(len(self.moe_layer_pattern) - 1):
                if self.moe_layer_pattern[reloading_layer_id] == 0:
                    # is dense layer
                    if reloading_layer_id == 0:
                        reloading_microbatch_id = reloading_microbatch_id + 1
                        reloading_layer_id = self.num_layers - 1
                    else:
                        reloading_layer_id = reloading_layer_id - 1
                else:
                    # is moe layer
                    break
        else:
            # interleaved 1F1B (VPP)
            if reloading_microbatch_id == None and reloading_layer_id == None:
                reloading_microbatch_id, reloading_layer_id = self.get_reloading_microbatch_id_layer_id_from_table()    
        
            if reloading_microbatch_id < 0 or reloading_layer_id < 0 or reloading_layer_id >= len(self.moe_layer_pattern) or self.moe_layer_pattern[reloading_layer_id] == 0:
                return
            
        tensor_tag = (reloading_microbatch_id, reloading_layer_id, tensor_name)
        if not tensor_tag in self.tensor_tag_to_state:
            # in the bwd of layer 0 of the last mircobatch
            # print(f"launch reload : not tensor_tag in self.tensor_tag_to_state")
            return
        (src_tensor, cpu_backup, copy_done_event, untyped_size) = self.tensor_tag_to_state.pop(tensor_tag)
        # print(f"[SUCCES launch reload] on tensor_tag (reloading_microbatch_id, reloading_layer_id, tensor_name): {tensor_tag}")

        copy_done_event = torch.cuda.Event()
        
        if not self.config.overlap_moe_expert_parallel_comm:
            self.h2d_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.h2d_stream):
            src_tensor.untyped_storage().resize_(untyped_size)
            src_tensor.copy_(cpu_backup, non_blocking=True)
            copy_done_event.record(stream=self.h2d_stream)
            state = (copy_done_event, src_tensor)
            self.reloading_tensor[tensor_tag] = state
                

    def wait_reload(self, tensor_name, reloading_microbatch_id = None, reloading_layer_id = None):
        # print(f"[wait_reload] current_layer_id={self.current_layer_id}")
        if reloading_microbatch_id == None and (self.current_layer_id >= len(self.moe_layer_pattern) or self.moe_layer_pattern[self.current_layer_id] == 0):
            return
        
        if self.num_model_chunks == None:
            # 1F1B
            # reload actication of layer i-1 in the bwd of layer i (when i > 0)
            # reload actication of last layer of next microbatch in the bwd of layer 0 
            if reloading_microbatch_id == None and reloading_layer_id == None:
                if self.current_layer_id == 0:
                    reloading_microbatch_id = self.current_microbatch_id + 1
                    reloading_layer_id = self.num_layers - 1
                else:
                    reloading_microbatch_id = self.current_microbatch_id
                    reloading_layer_id = self.current_layer_id - 1
                    
            for i in range(len(self.moe_layer_pattern) - 1):
                if self.moe_layer_pattern[reloading_layer_id] == 0:
                    # is dense layer
                    if reloading_layer_id == 0:
                        reloading_microbatch_id = reloading_microbatch_id + 1
                        reloading_layer_id = self.num_layers - 1
                    else:
                        reloading_layer_id = reloading_layer_id - 1
                else:
                    # is moe layer
                    break
        else:
            # interleaved 1F1B (VPP)
            if reloading_microbatch_id == None and reloading_layer_id == None:
                reloading_microbatch_id, reloading_layer_id = self.get_reloading_microbatch_id_layer_id_from_table()    
        
            if reloading_microbatch_id < 0 or reloading_layer_id < 0 or reloading_layer_id >= len(self.moe_layer_pattern) or self.moe_layer_pattern[reloading_layer_id] == 0:
                return      
                
        tensor_tag = (reloading_microbatch_id, reloading_layer_id, tensor_name)
        if not tensor_tag in self.reloading_tensor:
            # in the bwd of layer 0 of the last mircobatch
            # print(f"wait reload : not tensor_tag in self.reloading_tensor")
            return
        # print(f"[SUCCES wait_reload] on tensor_tag (reloading_microbatch_id, reloading_layer_id, tensor_name): {tensor_tag}")
        (copy_done_event, device_tensor) = self.reloading_tensor[tensor_tag]
        copy_done_event.synchronize() # TODO: use .wait() to check the stream with copy engine (d2h / h2d / all2all)
        pin_memory_id = self.tensor_tag_to_pin_memory_id[tensor_tag]
        self.available_pin_memory_tensor_pool_id_queue_dict[tensor_name].put(pin_memory_id)
        return
    
    
    def get_reloaded(self, tensor_tag):
        (copy_done_event, device_tensor) = self.reloading_tensor.pop(tensor_tag)
        return device_tensor
    
    
    def tensor_push(self, tensor: torch.Tensor, **kwargs) -> Any:
        if hasattr(tensor, self.OFFLOAD_TENSOR_ATTR_KEY):
            return self.register_offload(tensor)
        return tensor
    
    
    def tensor_pop(self, tensor_tag, **kwargs):
        if tensor_tag in self.reloading_tensor:
            return self.wait_reload(tensor_tag)
        return tensor_tag


    def start_microbatch_forward(self, current_microbatch_id, current_layer_id = 0):
        # print(f"[start_microbatch_forward] batch id: {current_microbatch_id}, from layer current_layer_id = {current_layer_id}")
        self.current_microbatch_id = current_microbatch_id
        self.current_layer_id = current_layer_id


    def start_microbatch_backward(self, current_microbatch_id, current_layer_id = None):
        # print(f"[start_microbatch_backward] batch id: {current_microbatch_id}, from layer current_layer_id = {current_layer_id}")
        self.current_microbatch_id = current_microbatch_id
        if current_layer_id == None:
            self.current_layer_id = self.num_layers
        else:
            self.current_layer_id = current_layer_id



_fg_offload_handler_instance = _FineGrainedAsyncDoubleBufferGroupOffloadHandler()


def get_fine_grained_offload_handler():
    return _fg_offload_handler_instance



# class CpuOffloadHookWithFineGrainedOffloadHandler(CpuOffloadSavedTensorHook):

#     def __init__(self):
#         self.offload_handler = get_fine_grained_offload_handler()
#         self.handler_extra_kwargs = {}
#         super().__init__()

#     def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
#         retrieve_identifier = self.offload_handler.tensor_push(tensor, **self.handler_extra_kwargs)
#         return retrieve_identifier
    
#     def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
#         tensor = self.offload_handler.tensor_pop(saved_state, **self.handler_extra_kwargs)
#         return tensor
    

class LaunchReloadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, fine_grained_tensor_name):
        ctx.fine_grained_tensor_name = fine_grained_tensor_name
        # print(f"LaunchReloadFunction fwd on {fine_grained_tensor_name}")
        return tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        cpu_offload_handler = get_fine_grained_offload_handler()
        # print(f"LaunchReloadFunction bwd on {ctx.fine_grained_tensor_name}")
        cpu_offload_handler.launch_reload(ctx.fine_grained_tensor_name)
        return grad_output, None
    

class WaitReloadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, fine_grained_tensor_name):
        ctx.fine_grained_tensor_name = fine_grained_tensor_name
        # print(f"WaitReloadFunction fwd on {fine_grained_tensor_name}")
        return tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        cpu_offload_handler = get_fine_grained_offload_handler()
        # print(f"WaitReloadFunction bwd on {ctx.fine_grained_tensor_name}")
        cpu_offload_handler.wait_reload(ctx.fine_grained_tensor_name)
        return grad_output, None


class FineGrainedOffloadLayerCounter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        cpu_offload_handler = get_fine_grained_offload_handler()
        # print(f'[LayerCounter] finish layer forward: current_microbatch_id = {cpu_offload_handler.current_microbatch_id}, current_layer_id = {cpu_offload_handler.current_layer_id}')
        cpu_offload_handler.current_layer_id += 1
        return tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        cpu_offload_handler = get_fine_grained_offload_handler()
        cpu_offload_handler.current_layer_id -= 1
        # print(f'[LayerCounter] start layer backward: current_microbatch_id = {cpu_offload_handler.current_microbatch_id}, current_layer_id = {cpu_offload_handler.current_layer_id}')
        return grad_output, None


# def get_fine_grained_offload_context():
#     return CpuOffloadHookWithFineGrainedOffloadHandler()
