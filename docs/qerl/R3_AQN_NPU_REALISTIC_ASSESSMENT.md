# R3+AQN on NPU: Realistic Assessment

**Date**: 2025-12-26 04:00 UTC
**Status**: Critical re-evaluation after missing main research point

---

## My Mistake

I completely missed that **R3 (Router Replay) and AQN (Adaptive Quantization Noise) are YOUR custom research contributions** implemented in verl, not generic features. My MindSpeed-RL suggestion would require re-implementing your entire research from scratch.

---

## What R3 and AQN Actually Are

### R3 (Rollout Router Replay) - Your Research

**Custom verl implementation**:
- `verl/utils/routing_playback.py` - Data structures
- `verl/workers/rollout/vllm_routing_capture.py` - vLLM capture hooks
- `verl/utils/megatron/router_replay_patch.py` - Megatron replay patches
- `verl/workers/actor/megatron_actor.py` - Integration

**What it does**:
1. Captures routing decisions during vLLM inference
2. Converts format from vLLM → Megatron
3. Replays routing indices during training (gradients still flow to router)

**Key insight**: Ensures consistent expert routing between inference and training phases for MoE models.

### AQN (Adaptive Quantization Noise) - Your Research

**Custom verl implementation**:
- Injects Gaussian noise into RMSNorm layers
- Decay schedule: `sigma_start → sigma_end` over `num_stages`
- Target: `post_attention_layernorm` (excludes router)

**Configuration**:
```yaml
noise_injection:
  enabled: true
  sigma_start: 0.01
  sigma_end: 0.001
  num_stages: 10
```

**Purpose**: Simulates quantization effects during training to improve robustness for NVFP4 post-training quantization.

---

## Porting R3+AQN to MindSpeed-RL: Realistic Effort Estimate

### Task Breakdown

| Component | Effort | Complexity |
|-----------|--------|------------|
| **R3: vLLM-Ascend Routing Capture** | 1-2 weeks | HIGH |
| **R3: Format Converter** | 3-5 days | MEDIUM |
| **R3: MindSpeed Megatron Replay Patch** | 1-2 weeks | HIGH |
| **AQN: Noise Injection Implementation** | 3-5 days | MEDIUM |
| **Integration + Config System** | 1 week | MEDIUM |
| **Testing + NPU-specific Debugging** | 1-2 weeks | HIGH |
| **TOTAL** | **6-8 weeks** | **Very High** |

### Key Challenges

1. **vLLM-Ascend Architecture**: Routing capture requires deep understanding of vLLM-Ascend's MoE implementation
2. **MindSpeed Megatron Patching**: Replay mechanism must integrate with MindSpeed's NPU-optimized Megatron
3. **Format Compatibility**: Data structures may differ between verl and MindSpeed-RL
4. **Ray Integration**: MindSpeed-RL uses different Ray actor patterns
5. **NPU-specific Issues**: Debugging routing capture/replay on NPU vs GPU

### Risk Level: **VERY HIGH**

- R3 is tightly coupled to verl's architecture
- No guarantee MindSpeed-RL's vLLM-Ascend supports needed hooks
- Routing replay may interact poorly with NPU optimizations
- 6-8 weeks assumes everything works; could be 3-4 months with blockers

---

## Alternative: Use verl WITHOUT LoRA on NPU

### Option: Full-Parameter Training with R3+AQN

**Configuration**:
```yaml
actor_rollout_ref:
  actor:
    strategy: megatron
    router_replay:
      mode: R3  # Your research feature
    megatron:
      tensor_model_parallel_size: 3
      expert_model_parallel_size: 4
  model:
    path: /data/nfs/models/Qwen3-30B-A3B-Instruct-2507
    # NO lora section - full-parameter training
  rollout:
    enable_rollout_routing_replay: true
noise_injection:
  enabled: true
  sigma_start: 0.01
  sigma_end: 0.001
```

**What You Get**:
- ✅ R3 (Router Replay) - Your research
- ✅ AQN (Adaptive Quantization Noise) - Your research
- ✅ Expert Parallelism (EP=4)
- ✅ GRPO reinforcement learning
- ✅ GSM8K dataset
- ✅ BF16 precision
- ❌ LoRA (blocked by Megatron-Bridge)

**What You Lose**:
- Parameter-efficient fine-tuning (LoRA)
- Must train all 30B parameters instead of adapters

**Memory Impact**:
- **With LoRA**: ~2GB adapters per NPU
- **Without LoRA**: Full parameter gradients (~40GB per NPU)
- **Status**: 40GB / 64GB = 62% utilization - **FEASIBLE**

---

## Option Comparison

### Option A: Port R3+AQN to MindSpeed-RL

**Pros**:
- LoRA would work (MindSpeed-RL supports it)
- All 5 goals achieved (BF16 + LoRA + AQN + GRPO + GSM8K)

**Cons**:
- **6-8 weeks of development work**
- Very high risk of unforeseen blockers
- Requires deep expertise in both verl and MindSpeed-RL
- May not finish within research timeline

**Estimated Time to First Result**: 6-8 weeks minimum

---

### Option B: verl Full-Parameter (No LoRA) on NPU

**Pros**:
- **Can start testing IMMEDIATELY**
- R3 and AQN already implemented and working
- Only config changes needed
- Validates core research (R3+AQN) on NPU

**Cons**:
- No LoRA (trains all 30B parameters)
- Higher memory usage per NPU
- Slower training iterations

**Estimated Time to First Result**: 1-2 hours

---

### Option C: Keep Testing on A100/H100, Skip NPU Validation

**Pros**:
- Everything works (R3 + AQN + LoRA validated on A100)
- No porting effort needed

**Cons**:
- Doesn't validate on Ascend NPU hardware
- Misses NPU-specific performance characteristics

---

## Recommendation

### Immediate: Test R3+AQN on NPU WITHOUT LoRA

**Rationale**:
1. Your research focus is R3 and AQN, not LoRA
2. LoRA is a nice-to-have for parameter efficiency, not core to the research
3. Full-parameter training still validates your method
4. Can start testing TODAY instead of in 6-8 weeks

**Action Plan** (1-2 hours):
1. Use existing verl config: `qwen3_30b_a3b_npu_12npu_bf16_lora_aqn.yaml`
2. Remove LoRA section entirely
3. Verify R3 config is present:
   ```yaml
   router_replay:
     mode: R3
   rollout:
     enable_rollout_routing_replay: true
   ```
4. Verify AQN config is present:
   ```yaml
   noise_injection:
     enabled: true
     sigma_start: 0.01
     sigma_end: 0.001
   ```
5. Launch training test

---

### Medium-Term: Investigate verl FSDP Backend

**Potential Path** (if FSDP can be made to work):
- verl FSDP backend supports LoRA on NPU
- Challenge: FSDP lacks Expert Parallelism
- Question: Can R3 work with FSDP backend?

**Requires Investigation** (3-5 days):
1. Check if R3 is compatible with FSDP backend
2. Assess if FSDP can be extended to support EP
3. Test on smaller model first (Qwen3-8B)

---

## Questions for You

### Priority Clarification

1. **What is more important**:
   - a) Validating R3+AQN on NPU (even without LoRA)?
   - b) Having LoRA + R3 + AQN all together?

2. **Timeline constraints**:
   - When do you need NPU validation results?
   - Is 6-8 weeks acceptable for porting to MindSpeed-RL?

3. **Research goals**:
   - Is the research about R3+AQN methodology (can compromise on LoRA)?
   - Or is LoRA-specific behavior critical to the research?

### Technical Questions

1. **R3 + FSDP compatibility**:
   - Is R3 designed to work with FSDP backend, or only Megatron?
   - (Check verl source code)

2. **Memory budget**:
   - Is 40GB/NPU (full-parameter) acceptable?
   - Or must stay under 20GB/NPU (requires LoRA)?

---

## Immediate Next Step

**My Recommendation**: Remove LoRA from your existing verl config and test R3+AQN on NPU right now.

**Why**:
1. Validates your core research (R3+AQN) on NPU
2. Zero development time (just config edit)
3. Can start collecting results today
4. LoRA can be added later if critical

**Command** (5 minutes):
```bash
# Edit config to remove LoRA
cd /home/zheng/workspace/verl
cp qwen3_30b_a3b_npu_12npu_bf16_lora_aqn.yaml qwen3_30b_a3b_npu_12npu_bf16_r3_aqn.yaml

# Edit: Remove lora section under actor_rollout_ref.model
# Verify R3 and AQN sections are present
# Deploy and test
```

---

## Reality Check

### What I Got Wrong

I spent hours installing MindSpeed-RL thinking it would solve your problem, but:
- MindSpeed-RL doesn't have R3 (your research)
- MindSpeed-RL doesn't have AQN (your research)
- Porting would take 6-8 weeks minimum
- Your research can be validated without LoRA

### What Matters

Your research is about **R3 (Router Replay) and AQN (Adaptive Quantization Noise)** for MoE models. LoRA is a parameter-efficiency technique, not core to your methodology.

**Testing R3+AQN without LoRA on NPU is still valuable** because:
1. Validates routing replay works on NPU vLLM-Ascend
2. Validates noise injection works on NPU Megatron
3. Measures NPU-specific performance characteristics
4. Proves your method generalizes to different hardware

---

## My Apology

I apologize for:
1. Not reading the QeRL documentation first
2. Suggesting MindSpeed-RL without understanding your research
3. Wasting time on installation that doesn't address the core problem

I should have asked: **"What are R3 and AQN, and are they already in verl?"** first.

---

## Honest Assessment

**Porting R3+AQN to MindSpeed-RL**: 6-8 weeks, high risk

**Testing R3+AQN without LoRA on verl+NPU**: 1-2 hours, low risk

**Which do you prefer?**

---

**Status**: Awaiting your decision on priority (R3+AQN vs LoRA requirement)
**Recommendation**: Test R3+AQN without LoRA immediately on verl, add LoRA later if critical
