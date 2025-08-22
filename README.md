# Introduction
MT-TransformerEngine is a high-performance deep learning framework developed by the Moore Threads AI-Infra Team. Built upon [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) and [torch_musa](https://github.com/MooreThreads/torch_musa), MT-TransformerEngine delivers optimized support for FP8 training on Moore Threads GPUs. When integrated with [MT-Megatron](https://github.com/MooreThreads/MT-MegatronLM/tree/main), MT-TransformerEngine enables:
- FP8 training recipe on Moore Threads GPUs. And we provide the same FP8 training strategy as the DeepSeek-v3 with the **MTFP8BlockScalingRecipeState** in transformer_engine/musa/pytorch/fp8.py.
- Scalable large-model training across clusters of thousands of GPUs. For detailed introduction on large model training, refer to the [MT-Megatron](https://github.com/MooreThreads/MT-MegatronLM/tree/main).

# Installation
Install MT-TransformerEngine via the provided installation script.

```bash
bash install.sh
```

The script will compile MUSA kernels and C++ source files from **transformer_engine/musa/common** and **transformer_engine/musa/pytorch/csrc**

# MUSA Example
To execute CUDA-compatible training on Moore Threads GPUs:
1. Import torch and [torch_musa](https://github.com/MooreThreads/torch_musa)
2. Replace cuda device strings with musa

```python
import torch
import torch_musa
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# Set dimensions.
in_features = 768
out_features = 3072
hidden_size = 2048

# Initialize model and inputs.
model = te.Linear(in_features, out_features, bias=True)
inp = torch.randn(hidden_size, in_features, device="musa")

# Create an FP8 recipe. Note: All input args are optional.
fp8_recipe = recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.E4M3)

# Enable autocasting for the forward pass
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    out = model(inp)

loss = out.sum()
loss.backward()
```

# Feature

| Feature               | Availability |
| :---                  |    :----:    |
| per-tensor fp8        | &#10004;     |
| per-block fp8         | &#10004;     |
| tp overlap (with fp8) | &#10004;     |
| moe recompute         | &#10004;     |
| zero bubble           | &#10004;     |
| fp8 alltoall          | Coming Soon  |

# Community
### Issue Reporting
If you find any problems for large model training using MT-TE, please open an issue.

### Contributions
**Welcome any form of contribution of code and document!**
