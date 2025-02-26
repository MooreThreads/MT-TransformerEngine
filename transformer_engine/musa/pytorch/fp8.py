from typing import Tuple

from transformer_engine.pytorch.utils import get_device_compute_capability


def musa_check_fp8_support() -> Tuple[bool, str]:
    """Return if fp8 support is available"""
    if get_device_compute_capability() >= (3, 1):
        return True, ""
    return False, "Device compute capability 3.1 or higher required for FP8 execution."


def replace_musa_check_fp8_support():
    from transformer_engine.pytorch import fp8
    setattr(fp8, "check_fp8_support", musa_check_fp8_support)


replace_musa_check_fp8_support()
