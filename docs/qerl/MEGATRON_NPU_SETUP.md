# Megatron-LM + MindSpeed Setup on Ascend NPU

**Date**: 2025-12-29
**Container**: `verl-a3cloud-16-ranks`
**NPU Count**: 16 (Ascend 910C)

## Summary

Successfully set up Megatron-LM with MindSpeed for MoE training on Ascend NPU.

| Component | Version | Status |
|-----------|---------|--------|
| Megatron-LM | core_v0.12.1 | ✅ Installed |
| MindSpeed | 0.12.1 | ✅ Installed |
| Tensor Parallelism | Tested TP=2,8 | ✅ Working |
| Expert Parallelism | Tested EP=4 | ✅ Working |
| HCCL Communication | AllReduce | ✅ Working |

## Installation Steps

### 1. Checkout Megatron-LM to MindSpeed-compatible version

```bash
cd /home/z00637938/workspace/Megatron-LM
git checkout core_v0.12.1
```

### 2. Install megatron-core

```bash
source /home/z00637938/setup_proxy.sh  # For network access
cd /home/z00637938/workspace/Megatron-LM
pip install -e .
```

### 3. Install MindSpeed

```bash
cd /home/z00637938/workspace/MindSpeed
pip install -e .
```

### 4. Configure PYTHONPATH

The full `megatron` package (including `megatron.training`) needs to be in PYTHONPATH:

```bash
export PYTHONPATH=/home/z00637938/workspace/Megatron-LM:$PYTHONPATH
```

A helper script is available:
```bash
source /home/z00637938/setup_megatron_env.sh
```

## Usage

### Python Import Order

```python
import os
import sys

# 1. Set device visibility (CRITICAL for multi-NPU)
local_rank = int(os.environ.get("LOCAL_RANK", 0))
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(local_rank)

# 2. Add Megatron-LM to path
sys.path.insert(0, "/home/z00637938/workspace/Megatron-LM")

# 3. Import torch and set device
import torch
import torch_npu
torch.npu.set_device(0)

# 4. Import MindSpeed adaptor (patches Megatron for NPU)
import mindspeed.megatron_adaptor

# 5. Initialize distributed
import torch.distributed as dist
dist.init_process_group(backend="hccl")

# 6. Initialize Megatron parallel state
from megatron.core import parallel_state as mpu
mpu.initialize_model_parallel(
    tensor_model_parallel_size=TP,
    expert_model_parallel_size=EP,
)
```

### Running Multi-NPU Scripts

```bash
PYTHONPATH=/home/z00637938/workspace/Megatron-LM:$PYTHONPATH \
torchrun --nproc_per_node=8 --master_port=29500 your_script.py
```

## Tested Configurations

### TP=2 (2 NPUs)
```
✅ Megatron TP=2 initialized on NPU
   World size: 2
   TP world size: 2
   DP world size: 1
✅ Basic allreduce test passed
```

### TP=8 (8 NPUs)
```
✅ Megatron TP=8 initialized on NPU
   World size: 8
   TP world size: 8
   DP world size: 1
✅ Basic allreduce test passed
```

### TP=2, EP=4 (8 NPUs)
```
✅ Megatron TP=2, EP=4 initialized on NPU
   World size: 8
   TP world size: 2
   EP world size: 4
✅ TP communication test passed
✅ EP communication test passed
```

## Known Warnings (Non-Fatal)

These warnings are expected and can be ignored:

1. **torch_npu transfer warnings** - Expected on NPU
2. **"Failed to import transformer engine plugin"** - TransformerEngine not available on NPU
3. **"Failed to import megatron plugin"** - modelopt plugin compatibility
4. **"torch.npu.get_device_capability isn't implemented"** - NPU doesn't report CUDA capabilities
5. **"Apex is not installed"** - Using PyTorch fallbacks

## verl Integration

verl's MegatronPPOActor can be imported with this setup:

```python
import sys
sys.path.insert(0, "/home/z00637938/workspace/Megatron-LM")
import mindspeed.megatron_adaptor
sys.path.insert(0, "/verl")
from verl.workers.actor.megatron_actor import MegatronPPOActor
# Shows: UserWarning: NPU not support router replay for now.
# This is expected - router replay (R3) is not yet supported on NPU
```

## MoE Training Parallelism Recommendations

For Qwen3-30B-A3B (128 experts, 8 active per token):

| NPU Count | Recommended Config | Notes |
|-----------|-------------------|-------|
| 8 NPUs | TP=2, EP=4 | 32 experts per EP rank |
| 16 NPUs | TP=2, EP=8 | 16 experts per EP rank |

## Next Steps

1. **Test MoE model loading** with Megatron + MindSpeed
2. **Run verl training** with Megatron backend on NPU
3. **Verify gradient computation** for MoE layers
4. **Test vLLM + Megatron hybrid** (vLLM for rollout, Megatron for training)

## References

- MindSpeed README: `/home/z00637938/workspace/MindSpeed/README.md`
- verl router_replay example: `/verl/examples/router_replay/`
- Megatron-LM: `/home/z00637938/workspace/Megatron-LM`
