# AGENTS.md - Guide for Coding Agents

This document provides essential information for AI coding agents working in the MT-TransformerEngine repository.

## Project Overview

MT-TransformerEngine is a high-performance deep learning framework for FP8 training on Moore Threads GPUs. It is built upon TransformerEngine and torch_musa.

Supported frameworks: PyTorch (`pytorch`), JAX (`jax`), MUSA (`musa`).

---

## Build Commands

### Setting the Framework

```bash
export NVTE_FRAMEWORK=pytorch  # or: jax, musa, none
```

### Build and Install

```bash
pip install . -v                    # Standard build
pip install . -v --no-deps          # Build without dependencies
pip wheel --no-build-isolation -v -w ./dist .  # Build wheel
```

### Clean Build

```bash
pip uninstall transformer_engine -y
rm -rf build transformer_engine.egg-info
pip install . -v
```

---

## Test Commands

### Running Python Tests (Pytest)

```bash
pytest -v -s tests/pytorch/test_sanity.py                    # Single test file
pytest -v -s tests/pytorch/test_numerics.py::test_func_name  # Single test function
pytest -v -s tests/pytorch/test_sanity.py -k "test_name"     # Run tests matching pattern
pytest -v -s tests/pytorch/test_recipe.py                    # Recipe tests
pytest -v -s tests/pytorch/test_fused_rope.py                # Fused RoPE tests
pytest -v -s tests/pytorch/test_float8tensor.py              # Float8 tensor tests
pytest -v -s tests/pytorch/test_cuda_graphs.py              # CUDA graphs tests
pytest -v -s tests/pytorch/test_jit.py                       # JIT tests
pytest -v -s tests/pytorch/test_gqa.py                       # Grouped query attention tests
pytest -v -s tests/pytorch/test_fused_optimizer.py           # Optimizer tests
pytest -v -s tests/pytorch/test_multi_tensor.py             # Multi-tensor tests
pytest -v -s tests/pytorch/test_fusible_ops.py              # Fusible ops tests
pytest -v -s tests/pytorch/test_permutation.py              # Permutation tests
pytest -v -s tests/pytorch/fused_attn/test_fused_attn.py    # Fused attention tests
pytest -v -s tests/jax/test_sanity_import.py                # JAX sanity tests
pytest -v -s tests/musa/operators/test_gemm.py              # MUSA GEMM tests
pytest -v -s tests/musa/operators/test_cast.py              # MUSA cast tests
pytest -v -s tests/musa/operators/test_rms_norm.py          # MUSA RMS norm tests
pytest -v -s tests/musa/operators/test_transpose.py         # MUSA transpose tests
```

### Tests with Special Environment Variables

```bash
PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 pytest -v -s tests/pytorch/test_numerics.py
NVTE_CUDNN_MXFP8_NORM=0 PYTORCH_JIT=0 NVTE_TORCH_COMPILE=0 pytest -v -s tests/pytorch/test_cuda_graphs.py
NVTE_TORCH_COMPILE=0 NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 pytest -o log_cli=true --log-cli-level=INFO -v -s tests/pytorch/fused_attn/test_fused_attn.py
```

### Running Distributed Tests

```bash
torchrun --nproc-per-node=8 tests/pytorch/distributed/run_gemm_with_overlap.py --comm-type rs --verbose --dtype bf16
torchrun --nproc-per-node=8 tests/pytorch/distributed/test_fp8_gemm_with_meta_tensor.py
```

---

## Lint Commands

### Python Linting

```bash
pip install pylint==3.3.1
pylint --recursive=y transformer_engine/common transformer_engine/pytorch
pylint --recursive=y transformer_engine/musa  # For MUSA code
```

### C++ Linting

```bash
pip install cpplint==1.6.0
cpplint --root transformer_engine/common/include --recursive transformer_engine/common/include
cpplint --recursive transformer_engine/common transformer_engine/pytorch
cpplint --recursive transformer_engine/musa/common transformer_engine/musa/pytorch
```

### Format Python Code

```bash
pip install black==24.4.2
black --line-length=100 --preview --enable-unstable-feature=string_processing <file.py>
black --line-length=100 --preview --enable-unstable-feature=string_processing .
```

### Format C++ Code

```bash
clang-format -i -style=file <file.cpp>
clang-format -i -style=file <file.cu>
clang-format -i -style=file <file.h>
clang-format -i -style=file <file.cuh>
```

---

## Code Style Guidelines

### Python Code Style

- Line length: 100 characters maximum
- Formatter: black with `--preview --enable-unstable-feature=string_processing`
- Linter: pylint with custom pylintrc configuration
- Quotes: Use double quotes for docstrings, single quotes for regular strings preferred
- Type hints: Use from `typing` module (Optional, List, Dict, Tuple, Union, Callable, Any)
- No inline comments unless absolutely necessary

### Python Imports Order

Order imports following standard conventions:

```python
# Standard library imports first
import math
import os
from typing import Callable, Dict, List, Optional, Tuple, Union

# Third-party imports next
import numpy as np
import torch
import torch.nn.functional as F
import pytest  # In test files

# Local imports last
import transformer_engine_torch as tex
from transformer_engine.pytorch import Linear, LayerNormMLP
from transformer_engine.common import recipe
```

### Python Naming Conventions

- Functions: `snake_case` (e.g., `get_device_compute_capability`)
- Classes: `PascalCase` (e.g., `TransformerLayer`, `FP8GlobalStateManager`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `NVTE_FRAMEWORK`)
- Private functions: prefix with `_` (e.g., `_get_thd_freqs_on_this_cp_rank`)
- Module-level dunder names: `__version__`, `__all__`

### Python Docstrings

Use descriptive docstrings when functions are complex:

```python
def apply_rotary_pos_emb_thd(
    t: torch.Tensor,
    cu_seqlens: torch.Tensor,
    freqs: torch.Tensor,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> torch.Tensor:
    """Apply RoPE for `thd` format.

    Args:
        t (Tensor): Input tensor of shape [t, h, d]
        cu_seqlens (Tensor): Cumulative sum of sequence lengths, shape [b + 1]
        freqs (Tensor): Rotary positional embedding, shape [max_s, 1, 1, d]
        cp_size (int): Context parallelism size
        cp_rank (int): Context parallelism rank

    Returns:
        Tensor: Output tensor of shape [t, h, d]
    """
```

### Error Handling

Use assertions and exceptions appropriately:

```python
assert condition, "Error message"
raise ValueError(f"Invalid value: {value}")
raise RuntimeError(f"Operation failed: {reason}")
```

### C++ Code Style

Use clang-format with repository `.clang-format` configuration:

- Based on Google style with custom modifications
- Column limit: 100 characters
- Indent width: 2 spaces
- Pointer alignment: Left
- Sort includes: CaseSensitive

### C++ Naming Conventions

- Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `kCamelCase` or `UPPER_SNAKE_CASE`
- Member variables: `snake_case_` with trailing underscore
- Namespace: `snake_case`

### File Headers

All files must include the NVIDIA copyright header:

```python
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
```

```cpp
// Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// See LICENSE for license information.
```

---

## Testing Conventions

### Test File Organization

- PyTorch tests: `tests/pytorch/test_*.py`
- JAX tests: `tests/jax/test_*.py`
- MUSA tests: `tests/musa/operators/test_*.py`
- Distributed tests: `tests/pytorch/distributed/test_*.py`

### Test Function Naming

Use descriptive names with `test_` prefix:

```python
def test_linear_forward_fp8():
    ...

def test_gemm_with_overlap_rs():
    ...

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_attention(dtype):
    ...
```

### Test Fixtures and Utilities

Use shared utilities from test files when needed:

```python
from test_numerics import reset_rng_states, dtype_tols
from test_cast import dev, te_dtype_from_th_dtype
```

### Pytest Markers

Use parametrize for testing multiple configurations:

```python
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [768, 1024])
def test_feature(dtype, hidden_size):
    ...
```

---

## Directory Structure

```
/data/MT-TransformerEngine/
├── transformer_engine/
│   ├── common/           # Shared C++/CUDA code
│   ├── pytorch/          # PyTorch bindings
│   ├── jax/              # JAX bindings
│   ├── musa/             # MUSA (Moore Threads) bindings
│   └── __init__.py
├── tests/
│   ├── pytorch/          # PyTorch tests
│   ├── jax/              # JAX tests
│   ├── cpp/              # C++ tests
│   └── musa/             # MUSA tests
├── qa/                   # Quality assurance scripts
│   ├── L0_pytorch_unittest/
│   ├── L0_pytorch_lint/
│   ├── L0_jax_unittest/
│   └── L0_jax_lint/
├── build_tools/          # Build utilities
├── docs/                 # Documentation
├── examples/             # Example scripts
└── benchmarks/           # Performance benchmarks
```

---

## Key Modules

| Module | Description |
|--------|-------------|
| `transformer_engine.pytorch` | PyTorch layer implementations |
| `transformer_engine.pytorch.fp8` | FP8 autocast and state management |
| `transformer_engine.pytorch.attention` | Attention mechanisms including fused attention |
| `transformer_engine.pytorch.distributed` | Distributed primitives |
| `transformer_engine.common.recipe` | FP8 recipes |
| `transformer_engine.musa.pytorch` | MUSA-specific PyTorch implementations |

---

## Pre-commit Hooks

Pre-commit is configured in `.pre-commit-config.yaml`. Install hooks with:

```bash
pip install pre-commit
pre-commit install
```

Hooks run on commit:
- `check-merge-conflict`: Check for merge conflicts
- `check-added-large-files`: Prevent large files
- `end-of-file-fixer`: Fix missing newlines
- `trailing-whitespace`: Remove trailing whitespace
- `black`: Format Python code
- `clang-format`: Format C/C++ code

---

## Common Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `NVTE_FRAMEWORK` | pytorch, jax, musa, none | Target framework |
| `NVTE_TORCH_COMPILE` | 0, 1 | Enable/disable Torch compile |
| `NVTE_DEBUG` | 0, 1 | Enable debug logging |
| `NVTE_DEBUG_LEVEL` | 1, 2, ... | Debug verbosity |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | 0, 1 | Allow non-deterministic algorithms |
| `PYTORCH_JIT` | 0, 1 | Enable/disable PyTorch JIT |
| `MAX_JOBS` | 1, 2, ... | Parallel build jobs |
