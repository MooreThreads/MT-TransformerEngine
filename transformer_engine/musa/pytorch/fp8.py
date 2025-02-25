from typing import Tuple

from transformer_engine.pytorch.utils import get_device_compute_capability

def check_fp8_support() -> Tuple[bool, str]:
    """Return if fp8 support is available"""
    if get_device_compute_capability() >= (3, 1):
        return True, ""
    return False, "Device compute capability 3.1 or higher required for FP8 execution."


import sys
for k in sys.modules:
    if k.startswith('transformer_engine'):
        for target in ['check_fp8_support']:
            if getattr(sys.modules[k], target, None):
                setattr(sys.modules[k], target, check_fp8_support)
