# Ascend 910C Baseline for VERL + MoE Training

**Created**: 2025-12-25
**Purpose**: Technical baseline for running QeRL MoE training on Ascend 910C NPU
**Status**: Investigation Complete

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [VERL's Existing Ascend Support](#2-verls-existing-ascend-support)
3. [MindSpeed-LLM MoE Capabilities](#3-mindspeed-llm-moe-capabilities)
4. [vLLM-Ascend Integration](#4-vllm-ascend-integration)
5. [CANN/torch_npu Stack Overview](#5-canntorch_npu-stack-overview)
6. [MindSpeed-RL Comparison](#6-mindspeed-rl-comparison)
7. [Key Integration Points for QeRL MoE](#7-key-integration-points-for-qerl-moe)
8. [Recommended Configuration](#8-recommended-configuration)
9. [Known Limitations](#9-known-limitations)

---

## 1. Executive Summary

### Readiness Assessment

| Component | Status | Notes |
|-----------|--------|-------|
| **VERL Ascend Support** | Production-Ready | Full NPU support with MindSpeed backend |
| **MoE Training** | Production-Ready | EP/TP/PP parallelism, multiple routing strategies |
| **vLLM-Ascend** | Production-Ready | V1 engine with MoE support and EPLB |
| **CANN/torch_npu** | Mature | 200+ operators, comprehensive PyTorch integration |
| **QeRL Integration** | Requires Work | Need to verify noise injection + quantization on NPU |

### Key Takeaways

1. **VERL has comprehensive Ascend support** - Production recipes exist for DeepSeek-V3 on 128 A3 NPUs
2. **MindSpeed-LLM provides full MoE stack** - Qwen MoE, DeepSeek, Mixtral all supported
3. **vLLM-Ascend handles inference** - Hardware plugin with Expert Parallel Load Balancing
4. **torch_npu mirrors CUDA API** - `.npu()` works like `.cuda()` with HCCL backend
5. **MindSpeed-RL is an alternative** - Native Ascend RL framework with shared-card training

---

## 2. VERL's Existing Ascend Support

### 2.1 Device Abstraction Architecture

**Location**: `/home/zheng/workspace/verl/verl/utils/device.py`

```python
# Key Functions
is_torch_npu_available()     # Detects torch_npu availability
get_device_name()            # Returns "cuda", "npu", or "cpu"
get_nccl_backend()           # Returns "nccl" for CUDA, "hccl" for NPU
get_torch_device()           # Returns torch.cuda or torch.npu namespace
auto_set_ascend_device_name() # Auto-configures trainer.device to "npu"
get_visible_devices_keyword() # CUDA_VISIBLE_DEVICES or ASCEND_RT_VISIBLE_DEVICES
```

### 2.2 Backend Configurations

| Backend | Device Support | Registration Location |
|---------|---------------|----------------------|
| **MindSpeed** (Megatron on NPU) | NPU only | `verl/workers/engine/mindspeed/transformer_impl.py` |
| **FSDP** | CUDA + NPU | `verl/workers/engine/fsdp/transformer_impl.py` |
| **VeOmni** | CUDA + NPU | `verl/workers/engine/veomni/transformer_impl.py` |

### 2.3 NPU-Specific Kernel Optimizations

**Location**: `/home/zheng/workspace/verl/verl/models/transformers/npu_patch.py`

**Monkey Patches Applied**:
- `torch_npu.npu_rms_norm()` - Optimized RMSNorm
- `torch_npu.npu_swiglu()` - Fused SiLU/MLP activation
- `torch_npu.npu_rotary_mul()` - Optimized RoPE
- `torch_npu.npu_grouped_matmul()` - MoE expert routing
- `torch_npu.npu_moe_token_permute()` / `npu_moe_token_unpermute()` - Token dispatch

**Patched Models**: Qwen2, Qwen2.5-VL, Qwen3, Qwen3-MoE, Qwen3-VL, Qwen3-VL-MoE

### 2.4 Production Recipes

**Location**: `/home/zheng/workspace/verl/recipe/r1_ascend/`

| File | Purpose |
|------|---------|
| `megatron_workers.py` | Megatron worker with torch.compile handling |
| `engine_core.py` | Custom KV cache initialization patch |
| `vllm_rollout_spmd.py` | vLLM-Ascend rollout integration |
| `vllm_parallel_state.py` | HCCL backend parallel group management |

**DeepSeek-V3 Configuration (128 A3 NPUs)**:
- TP2 EP256 for rollout
- EP32 PP8 for actor
- ~95.5 tokens/sec per A3 NPU throughput

### 2.5 Version Requirements

| Component | Version |
|-----------|---------|
| Python | 3.10-3.11 |
| CANN | 8.3.RC1 |
| PyTorch | 2.7.1 |
| torch_npu | 2.7.1 |
| MindSpeed | commit f2b0977e |
| vLLM | v0.11.0 |
| vLLM-Ascend | v0.11.0rc1 |

---

## 3. MindSpeed-LLM MoE Capabilities

### 3.1 Supported MoE Models

| Model | Architecture | Status |
|-------|-------------|--------|
| **Qwen2 MoE** (57B-A14B) | 64 experts, top-8 | Pass |
| **Qwen3 MoE** (30B-A3B, 235B-A22B) | 128 experts | Pass |
| **Mixtral** (8x7B, 8x22B) | 8 experts | Pass |
| **DeepSeek-V2/V2.5** (236B, 16B) | MLA + MoE | Pass |
| **DeepSeek-V3** (671B) | 64x8 cluster | Pass |
| **MiniCPM-MoE** (8x2B) | 8 experts | Pass |
| **Phi-3.5-MoE** | Variable | Pass |

### 3.2 MoE Layer Implementation

**Location**: `/home/zheng/workspace/MindSpeed-LLM/mindspeed_llm/core/transformer/moe/`

| File | Purpose | Lines |
|------|---------|-------|
| `moe_layer.py` | Main MoE layer with shared expert support | 395 |
| `router.py` | Router/gating with multiple strategies | 594 |
| `layers.py` | Shared expert linear layers | 241 |
| `moe_utils.py` | Routing and load balancing utilities | 242 |

### 3.3 Routing Strategies

| Strategy | Flag | Description |
|----------|------|-------------|
| `aux_loss` | `--moe-router-load-balancing-type aux_loss` | Auxiliary loss-based balancing |
| `softmax_topk` | `--moe-router-load-balancing-type softmax_topk` | Softmax then top-k |
| `sinkhorn` | Available in base router | Sinkhorn algorithm |
| `group_limited_greedy` | `--moe-router-load-balancing-type group_limited_greedy` | Device-limited routing |
| `sparsemixer_topk` | `--moe-router-load-balancing-type sparsemixer_topk` | SparseMixer top-2 |

### 3.4 Expert Parallelism (EP)

**Location**: `/home/zheng/workspace/MindSpeed/mindspeed/lite/distributed/expert_parallel/`

**Dispatchers**:
- `eager` - Token-by-token processing
- `fused` - Grouped GEMM operations
- `mc2` - MC2 algorithm (EP >= 16)
- `allgather` - AllGather-based dispatch
- `alltoall` - AllToAll-based dispatch
- `alltoall_seq` - Sequence parallel AllToAll

**TP-Extend-EP** (`--moe-tp-extend-ep`): Uses TP group to extend EP instead of weight sharding

### 3.5 Communication Optimizations

| Optimization | Flag | Benefit |
|--------------|------|---------|
| AllToAll Overlap | `--moe-alltoall-overlap-comm` | Hides communication latency |
| AllGather Overlap | `--moe-allgather-overlap-comm` | Overlaps with computation |
| Async Permutation | `--moe-permutation-async-comm` | Non-blocking token permute |
| Fused Permute | `--use-fused-moe-token-permute-and-unpermute` | Reduced kernel launches |

### 3.6 Memory Optimizations

| Option | Values | Purpose |
|--------|--------|---------|
| `--moe-zero-memory` | disable/level0/level1 | Activation memory saving |
| `--moe-grouped-gemm` | Boolean | Batched expert computation |
| `--moe-layer-recompute` | Boolean | Activation recomputation |
| `--gemm-gradient-accumulation-fusion` | Boolean | Fused gradient accumulation |

---

## 4. vLLM-Ascend Integration

### 4.1 Architecture

**Location**: `/home/zheng/workspace/vllm-ascend/vllm_ascend/`

```
vLLM Core v0.11.0
        ↓
vllm-ascend Plugin (Hardware Layer)
  ├── NPUPlatform (Device abstraction)
  ├── AscendConfig (Configuration)
  └── Model/Attention/Ops registration
        ↓
torch-npu 2.7.1
        ↓
CANN 8.3+ / triton-ascend 3.2.x
        ↓
Ascend Hardware (Atlas 910C)
```

### 4.2 Supported MoE Models

| Model | Status | Features |
|-------|--------|----------|
| Qwen3 MoE | Functional | Full EP support |
| DeepSeek-V2/V3 | Functional | MoE with EPLB |
| Pangu Pro MoE | Functional | Multi-NPU |
| Kimi K2 | Functional | MoE layers |

### 4.3 Expert Parallel Communication

```python
class FusedMoEState(Enum):
    AllGather = 0        # All2All gather
    All2All = 1          # Standard all-to-all
    MC2 = 2              # MC2 (EP >= 16)
    AllGatherEP = 3      # DeepSeek V3/R1 only
    NaiveMulticast = 4   # Prefill multicast
    All2AllSeq = 5       # Sequential all-to-all
```

### 4.4 EPLB (Expert Parallel Load Balancing)

```python
ascend_config = {
    "init_redundancy_expert": 0,           # Redundant expert count
    "dynamic_eplb": False,                 # Enable dynamic balancing
    "num_iterations_eplb_update": 400,     # Update interval
    "gate_eplb": False,                    # Gate-based EPLB
}
```

### 4.5 Attention Backends

| Backend | Model Type | Implementation |
|---------|-----------|----------------|
| ASCEND (default) | Standard (Qwen, Llama) | CANN Flash Attention |
| ASCEND_SFA | DeepSeek-V2/V3 | Sparse Flash Attention |
| ASCEND_MLA | DeepSeek MLA variants | Multi-head Latent Attention |

### 4.6 Key Limitations

- **V1 Engine Only** - V0 scheduler not supported
- **Fixed Block Size** - KV cache blocks fixed at 128 tokens
- **No torch.compile()** - Uses eager mode or ACL Graph
- **Limited speculative decoding** - Basic support only

---

## 5. CANN/torch_npu Stack Overview

### 5.1 Software Stack

```
PyTorch Application
        ↓
torch_npu (Ascend Extension)
        ↓
CANN (Compute Architecture for Neural Networks)
  ├── opbase: Foundational operator framework
  ├── ops-nn: Neural network operators
  ├── ops-math: Mathematical operators
  ├── ops-transformer: Transformer operators
  └── ascend-transformer-boost: Optimized implementations
        ↓
Ascend NPU Hardware
```

### 5.2 torch_npu API (mirrors CUDA)

```python
# Device Management
torch.npu.is_available()
torch.npu.device_count()
torch.npu.set_device(device_id)
torch.npu.current_device()

# Memory Management
torch.npu.memory_allocated()
torch.npu.memory_reserved()
torch.npu.empty_cache()
torch.npu.memory_stats()

# Synchronization
torch.npu.synchronize()
torch.npu.Stream()
torch.npu.Event()

# Operations
torch_npu.npu_format_cast(tensor, format)
torch_npu.npu_dynamic_quant(tensor)
torch_npu.npu_grouped_matmul_*(...)
```

### 5.3 CANN Operator Coverage

**200+ operators** including:
- MatMul: `aclnnMatMul`, `aclnnBatchMatMul`, `aclnnAddmm`
- Normalization: `aclnnLayerNorm`, `aclnnRmsNorm`, `aclnnGroupNorm`
- Activation: `aclnnRelu`, `aclnnGelu`, `aclnnFastGelu`, `aclnnSigmoid`
- Quantization: `aclnnAscendQuantV3`, `aclnnDynamicQuantV3`
- Fusion: `aclnnAddLayerNorm`, `aclnnAddRmsNorm`, `aclnnAddRmsNormQuant`

### 5.4 Key Differences from CUDA

| Aspect | CUDA | NPU |
|--------|------|-----|
| Device Type | `torch.device('cuda:0')` | `torch.device('npu:0')` |
| Communication | NCCL | HCCL |
| Memory Hierarchy | Global → Shared → Registers | UB (32KB) → L1 → L2 → DDR |
| Graph Capture | CUDA Graphs | NPUGraph |
| Env Variable | `CUDA_VISIBLE_DEVICES` | `ASCEND_RT_VISIBLE_DEVICES` |

### 5.5 Triton-Ascend

**Location**: `/home/zheng/workspace/triton-ascend/`

- Supports 85%+ of Triton Python API
- Enables custom kernel development on NPU
- Tutorials: VectorAdd, Softmax, LayerNorm, FlashAttention, Matmul

---

## 6. MindSpeed-RL Comparison

### 6.1 Supported Algorithms

| Algorithm | MindSpeed-RL Status | VERL Status |
|-----------|---------------------|-------------|
| GRPO | Released | Supported |
| PPO | Preview | Mature |
| DAPO | Preview | Supported |
| DPO | Preview | Supported |

### 6.2 Architectural Differences

| Aspect | MindSpeed-RL | VERL |
|--------|-------------|------|
| Target Hardware | Ascend NPU exclusive | GPU/TPU agnostic |
| Shared Card Training | Native (Released) | Not primary design |
| Reference Model | Swappable weights | Separate loaded copy |
| Memory Management | Param/grad/optimizer offloading | Checkpoint activation |
| Rollout Engine | vLLM-Ascend patched | Native vLLM/SGLang |

### 6.3 Key MindSpeed-RL Features

**Shared Card Training (训推共卡)**:
- Single model for both inference and training
- Dynamic weight resharding between modes
- Memory optimization through offloading

**TransferDock Pattern**:
```
Dataset → Prompts → [Rollout (vLLM)] → Responses
  → [Reward Worker] → Rewards
  → [Reference Worker] → Log Probs
  → [Advantage Computation] → Advantages
  → [Training Worker] → Updated Weights
```

### 6.4 MindSpeed-RL Key Files

| File | Purpose |
|------|---------|
| `mindspeed_rl/trainer/grpo_trainer_hybrid.py` | GRPO trainer |
| `mindspeed_rl/trainer/ppo_trainer_hybrid.py` | PPO trainer |
| `mindspeed_rl/models/actor.py` | Actor base class |
| `mindspeed_rl/models/rollout/vllm_engine.py` | Rollout inference |
| `mindspeed_rl/workers/actor_hybrid_worker.py` | Hybrid actor worker |

---

## 7. Key Integration Points for QeRL MoE

### 7.1 QeRL Components to Port

| Component | Current (CUDA) | Ascend Equivalent | Status |
|-----------|---------------|-------------------|--------|
| Noise Injection | Custom CUDA kernel | torch_npu ops or Triton-Ascend | **Needs Implementation** |
| NVFP4 Quantization | llmcompressor | CANN quantization ops | **Needs Verification** |
| MoE Training | Megatron | MindSpeed-LLM MoE | Available |
| vLLM Inference | vLLM CUDA | vLLM-Ascend | Available |
| Actor-Critic | VERL workers | VERL MindSpeed backend | Available |

### 7.2 Noise Injection Options

**Option 1: torch_npu Native Ops**
```python
# Use existing torch_npu random operations
noise = torch.randn_like(weights).npu()
weights_noisy = weights + noise * scale
```

**Option 2: Triton-Ascend Custom Kernel**
```python
# Write custom Triton kernel for efficient noise injection
@triton.jit
def noise_injection_kernel(weights_ptr, noise_ptr, scale, ...):
    ...
```

**Option 3: CANN Operator**
```python
# Use CANN's aclnn operators directly
# Requires C++ extension
```

### 7.3 Quantization on Ascend

**CANN Native Quantization**:
- `aclnnAscendQuantV3` - Per-tensor/per-channel quantization
- `aclnnDynamicQuantV3` - Dynamic quantization
- `aclnnAscendAntiQuant` - Dequantization

**vLLM-Ascend Quantization Support**:
- W8A8 (Dynamic) - Stable
- W8A8 (Static) - Stable
- W4A8 (Dynamic) - Available
- MXFP8 - Experimental

**Note**: NVFP4 format may need custom implementation or mapping to Ascend-supported formats.

### 7.4 Recommended Integration Path

```
1. Port VERL to Ascend (mostly done)
   └── Use MindSpeed backend
   └── Use vLLM-Ascend for rollout

2. Port QeRL noise injection
   └── Option A: torch_npu native ops (simplest)
   └── Option B: Triton-Ascend kernel (optimal)

3. Adapt quantization
   └── Map NVFP4 to W4A8 or MXFP8
   └── Verify accuracy with Ascend quantization

4. MoE-specific optimizations
   └── Configure EP/TP for target model
   └── Enable EPLB for load balancing
   └── Use overlapped communication
```

---

## 8. Recommended Configuration

### 8.1 Environment Setup

```bash
# Device visibility
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# vLLM V1 engine (required)
export VLLM_USE_V1=1

# Optional optimizations
export VLLM_ASCEND_ENABLE_NZ=1
export VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP=1
```

### 8.2 Model Parallelism for MoE

**Small MoE (Qwen3-30B-A3B)**:
```yaml
tensor_parallel_size: 2
expert_parallel_size: 4
pipeline_parallel_size: 1
```

**Large MoE (Qwen3-235B-A22B)**:
```yaml
tensor_parallel_size: 4
expert_parallel_size: 16
pipeline_parallel_size: 2
```

### 8.3 MoE Training Arguments

```bash
MOE_ARGS="
    --num-experts 128
    --moe-router-topk 8
    --n-shared-experts 8
    --shared-expert-gate
    --moe-router-load-balancing-type aux_loss
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall_seq
    --moe-aux-loss-coeff 0.001
    --moe-permutation-async-comm
    --moe-alltoall-overlap-comm
    --use-fused-moe-token-permute-and-unpermute
"
```

### 8.4 Memory Optimization

```yaml
# For large models
moe_zero_memory: level1
offload_train_param: true
offload_train_grad: true
use_remove_padding: true
use_dynamic_bsz: true
```

---

## 9. Known Limitations

### 9.1 Hardware Constraints

| Limitation | Impact | Workaround |
|-----------|--------|-----------|
| Fixed KV cache block (128) | Less flexibility | None (architectural) |
| No native FP8 | Performance gap | Use W8A8 |
| Graph mode experimental | Some models slower | Use eager mode |
| EP cross-node limitation | Large model constraints | Use TP-Extend-EP |

### 9.2 Feature Gaps vs CUDA

| Feature | CUDA VERL | Ascend VERL |
|---------|-----------|-------------|
| SGLang backend | Supported | Not available |
| Speculative decoding | Full | Basic |
| LoRA | Full | V1 compatible |
| Multi-modal | Extensive | Limited |

### 9.3 QeRL-Specific Concerns

1. **Noise Injection Kernel**: Needs custom implementation
2. **NVFP4 Format**: May need adaptation to Ascend quantization
3. **Training-Inference Consistency**: Verify quantized model behavior matches
4. **Performance Profiling**: Use MindStudio tools (msProf) instead of NVIDIA profilers

---

## References

### Key Source Files

- VERL Ascend Support: `verl/utils/device.py`, `verl/workers/engine/mindspeed/`
- NPU Patches: `verl/models/transformers/npu_patch.py`
- Production Recipes: `recipe/r1_ascend/`
- MindSpeed-LLM MoE: `MindSpeed-LLM/mindspeed_llm/core/transformer/moe/`
- vLLM-Ascend: `vllm-ascend/vllm_ascend/`
- torch_npu: `pytorch/torch_npu/`
- MindSpeed-RL: `MindSpeed-RL/mindspeed_rl/`

### Documentation

- VERL Ascend Tutorial: `verl/docs/ascend_tutorial/ascend_quick_start.rst`
- MindSpeed-LLM MoE Docs: `MindSpeed-LLM/docs/pytorch/models/moe_model.md`
- vLLM-Ascend Docs: `vllm-ascend/docs/source/`
- Triton-Ascend: `triton-ascend/docs/`
