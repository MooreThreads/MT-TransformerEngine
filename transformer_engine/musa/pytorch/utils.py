from typing import List
import torch


def wrap_name(src):
    return f"_orig_{src}"

def add_attr(module, name, target):
    setattr(module, name, target)

def wrap_attr(module, name, wrapper):
    target = getattr(module, name)
    setattr(module, wrap_name(name), target)
    setattr(module, name, wrapper)

def replace_attr(module, name, target):
    wrap_attr(module, name, target)


def musa_assert_dim_for_fp8_exec(*tensors: List[torch.Tensor]) -> None:
    return
# TODO(yehua.zhang) do not work
import sys
for k in sys.modules:
    if 'utils' in k:
        for target in ['assert_dim_for_fp8_exec']:
            if getattr(sys.modules[k], target, None):
                setattr(sys.modules[k], target, musa_assert_dim_for_fp8_exec)