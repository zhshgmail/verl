# HANDOVER: AQN Degradation Fix for W4A4 + LoRA Experiments

**Date**: 2026-01-18
**Status**: CRITICAL - Context running low, need new agent to continue
**Priority**: P0 - This is the FIRST time we have REAL step 29 W4A4 validation results

---

## Problem Summary

When adding AQN (Adaptive Quantization Noise) to W4A4 + LoRA training, **performance DEGRADES from step 20 to step 29** instead of improving.

| Experiment | Step 20 | Step 29 | Change | vs Baseline |
|------------|---------|---------|--------|-------------|
| **E13h** (Baseline, no AQN) | 60.73% | **71.42%** | ✅ +10.69% | - |
| **E13j_v2** (σ=0.05→0.0005) | 70.20% | **65.96%** | ⚠️ **-4.24%** | -5.46% |
| **E13k_v2** (σ=0.01→0.0001) | 69.45% | **67.55%** | ⚠️ **-1.90%** | -3.87% |

**Key Insight**: AQN HELPS at step 20 but HURTS at step 29!

---

## Root Cause (From Expert Analysis)

### 1. Noise-Dependent Overfitting
The model learns behaviors **optimized for the noise regime** during early training (high sigma), then **fails to generalize** when noise is removed (low sigma).

### 2. Exponential Decay Creates "Noise Cliff"
```
E13j_v2 sigma schedule:
  Step 3:  σ = 0.050000 (HIGH)
  Step 20: σ = 0.002812 (18x lower)
  Step 29: σ = 0.000500 (100x lower - effectively no noise)
```

### 3. CRITICAL: AQN Is Applied During VALIDATION Too!
The noise injection happens during weight sync to vLLM, which occurs for **both training AND validation**. This means:
- Step 20 validation: evaluated with σ=0.002812 noise
- Step 29 validation: evaluated with σ=0.000500 noise (82% less)

The model sees different noise distributions during different validations!

---

## Recommended Experiments (Priority Order)

### P0: E13j_v3 - Constant Sigma (NO CODE CHANGE NEEDED)
**Hypothesis**: Constant noise eliminates noise-dependent overfitting.

```bash
# In test script, change:
++trainer.noise_injection.sigma_start=0.01
++trainer.noise_injection.sigma_end=0.01   # SAME AS START - no decay
++trainer.noise_injection.num_stages=1     # Single stage
```

**Script to create**: `scripts/test_mxfp4_w4a4_e13j_v3_constant_sigma.sh`

### P0: E13l_v3 - Training-Only AQN (REQUIRES CODE CHANGE)
**Hypothesis**: Validation should be noise-free to get true model performance.

**Code change needed** in `/verl/workers/rollout/vllm_rollout/vllm_rollout.py`:
```python
# Around line 354, add validation check before noise injection:
if hasattr(self, 'noise_injection_config') and self.noise_injection_config.get('enabled', False):
    is_validation = meta_info.get('validate', False) if meta_info else False
    if is_validation:
        print(f"[AQN] Skipping noise injection during validation")
    else:
        # ... existing noise injection code ...
```

### P1: E13n_v3 - Higher Ending Sigma
**Hypothesis**: Keep meaningful noise throughout to prevent distribution shift.

```bash
++trainer.noise_injection.sigma_start=0.05
++trainer.noise_injection.sigma_end=0.01   # 10x higher than before
++trainer.noise_injection.num_stages=10
```

### P1: E13m_v3 - Early Stop AQN at Step 20
**Hypothesis**: Let model "fine-tune" without noise in final steps.

This requires modifying the total_steps parameter in noise scheduling.

---

## Current Status

### Completed
- ✅ E13j_v2: 65.96% @ step 29 (2h 22min)
- ✅ E13k_v2: 67.55% @ step 29 (2h 23min)

### Running
- 🏃 E13l_v2: Started 17:54, expected ~20:15 completion

### Queued (BUT SHOULD PAUSE FOR v3 EXPERIMENTS)
- ⏸️ E13m_v2, E13n_v2, E14a_v2

**Recommendation**: PAUSE remaining v2 reruns. Run v3 experiments first to fix the fundamental degradation issue.

---

## Files Modified Recently

1. `verl/trainer/ppo/ray_trainer.py` - Removed skip validation bug
2. `docs/qerl/ALL_EXPERIMENTS_SUMMARY.md` - Updated with v2 results
3. `docs/qerl/V2_RERUNS_SUMMARY.md` - Created with timing info
4. `scripts/test_mxfp4_w4a4_e13j_v2_global_aqn.sh` - v2 rerun script
5. `scripts/test_mxfp4_w4a4_e13k_aqn_qerl_sigma_v2.sh` - v2 rerun script

---

## Server Info

- **Server**: root@90.90.102.18
- **Container**: verl-r3-test
- **GPUs**: 8 × A100
- **CRITICAL RULE**: ALWAYS restart container before each experiment!

```bash
# Container restart procedure:
ssh root@90.90.102.18
docker restart verl-r3-test
sleep 15
docker exec verl-r3-test bash -c 'ps aux | grep defunct | wc -l'  # Check zombies
docker exec -it verl-r3-test bash
cd /home/z00637938/workspace/verl
ray start --head --port=6379 --num-gpus=8 --disable-usage-stats
```

---

## Key Code Locations

1. **Noise Injection**: `verl/utils/noise_injection.py`
   - `add_noise_to_model()` - main noise injection function
   - Sigma schedule calculation

2. **vLLM Weight Sync**: `verl/workers/rollout/vllm_rollout/vllm_rollout.py`
   - `_load_vllm_model_to_gpu_async()` - where noise is applied during sync
   - Lines 353-407

3. **Training Loop**: `verl/trainer/ppo/ray_trainer.py`
   - Main training loop with validation logic

4. **Experiment Scripts**: `scripts/test_mxfp4_w4a4_e13*.sh`

---

## Next Steps for New Agent

1. **Check E13l_v2 status** - should be complete or nearly complete
2. **Create E13j_v3 script** with constant sigma (no code change needed)
3. **Run E13j_v3** and compare step 20 vs step 29
4. If E13j_v3 still shows degradation, implement training-only AQN code change
5. Update ALL_EXPERIMENTS_SUMMARY.md with v3 results

---

## Git Info

- **Branch**: feature/npu-aqn-test
- **Remotes**: personal (zhshgmail), team (EdisonAILab)
- **Latest commit**: Step 20 column addition to main table

---

## Critical Documents

- `docs/qerl/ALL_EXPERIMENTS_SUMMARY.md` - Main results table
- `docs/qerl/V2_RERUNS_SUMMARY.md` - v2 timing and results
- `docs/qerl/A100_CONTAINER_AND_DEV_WORKFLOW.md` - Server procedures
- `docs/qerl/EXPERIMENT_BUG_IMPACT_ANALYSIS.md` - Bug documentation
