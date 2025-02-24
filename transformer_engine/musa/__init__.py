import sys
import torch
import torch.utils
import torch.utils.data
import torch_musa

def patch_before_import_te():
    pass

def patch_after_import_torch():
    torch.cuda.is_available = torch.musa.is_available
    torch.cuda.current_device = lambda : f'musa:{torch.musa.current_device()}'
    torch.cuda.device_count = torch.musa.device_count
    torch.cuda.set_device = torch.musa.set_device
    torch.cuda.DoubleTensor = torch.musa.DoubleTensor
    torch.cuda.FloatTensor = torch.musa.FloatTensor
    torch.cuda.LongTensor = torch.musa.LongTensor
    torch.cuda.HalfTensor = torch.musa.HalfTensor
    torch.cuda.BFloat16Tensor = torch.musa.BFloat16Tensor
    torch.cuda.IntTensor = torch.musa.IntTensor
    torch.cuda.synchronize = torch.musa.synchronize
    torch.cuda.get_rng_state = torch.musa.get_rng_state
    torch.cuda.set_rng_state = torch.musa.set_rng_state
    torch.cuda.synchronize = torch.musa.synchronize
    torch.cuda.empty_cache = torch.musa.empty_cache
    torch.Tensor.cuda = torch.Tensor.musa
    torch.cuda.manual_seed = torch.musa.manual_seed
    torch.cuda.Event = torch.musa.Event
    torch.cuda.Stream = torch.musa.Stream
    torch.cuda.current_stream = torch.musa.current_stream
    torch.cuda.set_stream = torch.musa.set_stream
    torch.cuda.get_device_properties = torch.musa.get_device_properties

    torch.cuda.memory_allocated = torch.musa.memory_allocated
    torch.cuda.max_memory_allocated = torch.musa.memory_allocated
    torch.cuda.memory_reserved = torch.musa.memory_reserved
    torch.cuda.max_memory_reserved = torch.musa.max_memory_reserved

    original_tensor = torch.tensor
    def patched_tensor(*args, **kwargs):
        if 'device' in kwargs and kwargs['device'] == 'cuda':
            kwargs['device'] = 'musa'
        result = original_tensor(*args, **kwargs)
        return result
    torch.tensor = patched_tensor

    orig_type = torch.Tensor.type
    def musa_type(*args, **kwargs):
        result = orig_type(*args, **kwargs)
        return result.replace("musa", "cuda")
    torch.Tensor.type = musa_type

    original_zeros = torch.zeros
    def patched_zeros(*args, **kwargs):
        if 'device' in kwargs and kwargs['device'] == 'cuda':
            kwargs['device'] = 'musa'
        result = original_zeros(*args, **kwargs)
        return result
    torch.zeros = patched_zeros

    original_ones = torch.ones
    def patched_ones(*args, **kwargs):
        if 'device' in kwargs:
            v = kwargs['device']
            if isinstance(v, str) and v.startswith("cuda"):
                v = v.replace('cuda', 'musa')
            elif isinstance(v, torch.device) and v.type == "cuda":
                v = torch.device("musa", v.index)
            kwargs['device'] = v
        result = original_ones(*args, **kwargs)
        return result
    torch.ones = patched_ones

    original_empty = torch.empty
    def patched_empty(*args, **kwargs):
        if 'device' in kwargs and kwargs['device'] == 'cuda':
            kwargs['device'] = 'musa'
        result = original_empty(*args, **kwargs)
        return result
    torch.empty = patched_empty

    original_rand = torch.rand
    def patched_rand(*args, **kwargs):
        if 'device' in kwargs and kwargs['device'] == 'cuda':
            kwargs['device'] = 'musa'
        result = original_rand(*args, **kwargs)
        return result
    torch.rand = patched_rand

    original_is_cuda = torch.Tensor.is_cuda
    def always_cuda(self):
        return True
    torch.Tensor.is_cuda = property(always_cuda)

    origin_init_process_group = torch.distributed.init_process_group
    def patched_init_process_group(*args, **kwargs):
        if 'backend' in kwargs and kwargs['backend'] == 'nccl':
            kwargs['backend'] = 'mccl'
        result = origin_init_process_group(*args, **kwargs)
        return result
    torch.distributed.init_process_group = patched_init_process_group

    # def pin_memory(data, device=None):
    #     return data
    # torch.utils.data._utils.pin_memory.pin_memory = pin_memory

    def _pass_pvtx(*args, **kwargs):
        return
    torch.cuda.nvtx.range_push = _pass_pvtx
    torch.cuda.nvtx.range_pop = _pass_pvtx

    torch.cuda.is_current_stream_capturing = lambda: False

    origin_module_to = torch.nn.Module.to
    def patched_module_to(self, *args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, str) and arg.startswith("cuda"):
                new_args.append(arg.replace("cuda", "musa"))
            if isinstance(arg, torch.device) and arg.type == "cuda":
                new_args.append(torch.device("musa", arg.index))
            else:
                new_args.append(arg)
        return origin_module_to(self, *new_args, **kwargs)
    torch.nn.Module.to = patched_module_to

    origin_tensor_to = torch.Tensor.to
    def patched_tensor_to(self, *args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, str) and arg.startswith("cuda"):
                new_args.append(arg.replace("cuda", "musa"))
            if isinstance(arg, torch.device) and arg.type == "cuda":
                new_args.append(torch.device("musa", arg.index))
            else:
                new_args.append(arg)
        return origin_tensor_to(self, *new_args, **kwargs)
    torch.Tensor.to = patched_tensor_to

    import os
    os.environ["NVTE_TORCH_COMPILE"] = "0"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

def py_patch():
    if sys.version_info >= (3.9, 0):
        return
    import math
    def lcm(a, b):
        return abs(a * b) // math.gcd(a, b)
    math.lcm = lcm
    return

py_patch()
patch_before_import_te()
patch_after_import_torch()
