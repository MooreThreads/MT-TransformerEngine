import sys
import torch
import torch.utils
import torch.utils.data
import torch_musa

def patch_before_import_te():
    pass

def patch_after_import_torch():
    pass
    
def py_patch():
    pass

py_patch()
patch_before_import_te()
patch_after_import_torch()
