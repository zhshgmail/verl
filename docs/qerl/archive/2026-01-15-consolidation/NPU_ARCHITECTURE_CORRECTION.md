# CRITICAL CORRECTION: Ascend 910C Architecture

**Date**: 2025-12-26 09:10 UTC
**Severity**: CRITICAL - Fundamental misunderstanding

---

## The Truth About 910C NPUs

**CORRECT**: 16 separate NPUs, each with 64GB HBM
- Total: 16 × 64GB = **1024GB (1TB) of memory**
- Each NPU is independent
- Full TP=4 × EP=4 = 16 workers possible

**WRONG** (what I thought):
- ❌ "8 physical NPUs with 2 logical ranks sharing 64GB"
- ❌ "Only 8 × 64GB = 512GB available"
- ❌ "n_gpus_per_node should be 8"

---

## Impact of This Mistake

I configured for **8 NPUs** when **16 NPUs** are available:
- Used only HALF the available hardware
- Caused OOM by cramming everything into half resources
- Wasted significant debugging time

**Memory impact**:
- With 8 NPUs: Each NPU has model_size/8 → Tight fit, OOM
- With 16 NPUs: Each NPU has model_size/16 → Plenty of room

---

## Correct Configuration

### For Qwen1.5-MoE-A2.7B (60 experts, 16 heads)

```yaml
trainer:
  n_gpus_per_node: 16  # Use all 16 NPUs
  nnodes: 1
  device: npu

actor.megatron:
  tensor_model_parallel_size: 4   # 16 heads ÷ 4 = 4 per rank
  expert_model_parallel_size: 4   # 60 experts ÷ 4 = 15 per rank
  # TP × EP = 4 × 4 = 16 NPUs ✓

rollout:
  tensor_model_parallel_size: 4
  gpu_memory_utilization: 0.4  # Can be generous with 16 NPUs

ref.megatron:
  tensor_model_parallel_size: 4
  expert_model_parallel_size: 4
```

### For Qwen3-30B-A3B (128 experts, 32 heads)

```yaml
trainer:
  n_gpus_per_node: 16

actor.megatron:
  tensor_model_parallel_size: 4   # 32 heads ÷ 4 = 8 per rank
  expert_model_parallel_size: 4   # 128 experts ÷ 4 = 32 per rank
  # TP × EP = 4 × 4 = 16 NPUs ✓

# Alternative:
  tensor_model_parallel_size: 8   # 32 heads ÷ 8 = 4 per rank
  expert_model_parallel_size: 2   # 128 experts ÷ 2 = 64 per rank
  # TP × EP = 8 × 2 = 16 NPUs ✓
```

---

## How to Verify

```bash
# Check torch view
docker exec verl-a3cloud python3 -c "import torch_npu; print(torch_npu.npu.device_count())"
# Output: 16

# Check device files
docker exec verl-a3cloud ls /dev/davinci* | wc -l
# Output: 17 (davinci0-15 + davinci_manager)

# Check Ray resources
docker exec verl-a3cloud ray status | grep NPU
# Output: Should show 12.0 or 16.0 NPU
```

---

## Why I Got This Wrong

1. **Misunderstood "2 ranks per NPU"**: Thought this meant 2 logical devices sharing memory
   - **Reality**: It's about HCCL communication ranks, not device count
   - Each NPU still has its own 64GB

2. **Cargo-culted from baseline**: Saw n_gpus_per_node=8 and didn't question it
   - **Reality**: Baseline might have been for different setup or older config

3. **Didn't validate with user**: Should have asked "How many NPUs does this cluster have?"

---

## Lesson for Future Agents

**ALWAYS verify hardware facts directly with user**:
- "This cluster has X NPUs, each with Y GB memory, correct?"
- Don't assume based on:
  - torch.device_count() output
  - Existing configs
  - Documentation that might be for different hardware

**For distributed training**: TP × EP × PP × DP must equal total workers needed
- If n_gpus_per_node = 16, then TP × EP × PP should use factors of 16

---

## Updated in NPU_LESSONS_LEARNED.md

This correction is added to the main lessons learned document.

---

**Status**: CORRECTED - Now using all 16 NPUs properly
