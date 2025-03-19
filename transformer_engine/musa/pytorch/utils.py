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

def assert_dim_for_fp8_exec(*tensors: List[torch.Tensor]) -> None:
    return

from transformer_engine.pytorch import utils
replace_attr(utils, "assert_dim_for_fp8_exec", assert_dim_for_fp8_exec)