# E13h Execution Plan: MXFP4 W4A4 with STE Fix

**Date**: 2026-01-15
**Status**: Ready to Execute
**Branch**: `feature/npu-aqn-test`

---

## 1. Experiment Overview

### 1.1 What is E13h?

**E13h**: MXFP4 W4A4 quantization with STE fix (baseline, no RIN/AQN)

This experiment tests whether the STE fix that worked for NVFP4 (E13g: 60.88%) also works for MXFP4. It's a baseline experiment before adding RIN noise.

### 1.2 Comparison with E13g

| Aspect | E13g (NVFP4) | E13h (MXFP4) |
|--------|--------------|--------------|
| Quantization format | NVFP4 (~15% rel error) | MXFP4 (~21% rel error) |
| STE enabled | ✅ Yes | ✅ Yes |
| RIN/AQN | ❌ No | ❌ No |
| Expected accuracy | 60.88% (actual) | 55-60% |
| Status | ✅ Completed | ⏳ Ready to run |

### 1.3 Motivation

**Why run E13h?**

1. **Validate STE fix generalizes**: E13g proved STE fixes W4A4 training for NVFP4. We need to confirm it works for MXFP4 too.
2. **Establish MXFP4 baseline**: Before adding RIN (E13i/j/k), we need a clean baseline.
3. **NPU target format**: MXFP4 is the target format for Ascend NPU deployment.

**Expected outcome**: 55-60% accuracy (slightly lower than E13g due to higher MXFP4 quantization error)

---

## 2. Experiment Configuration

### 2.1 Key Parameters

```yaml
# Model
model: Qwen2.5-1.5B-Instruct
algorithm: DAPO
epochs: 1
total_steps: 29

# LoRA (16-bit adapters)
lora_rank: 32
lora_alpha: 16
target_modules: all-linear

# Batch size
train_batch_size: 128
gen_batch_size: 256
ppo_mini_batch_size: 32
n_resp_per_prompt: 8

# Learning rate
lr: 1e-5
warmup_steps: 10

# W4A4 Quantization (CRITICAL: injection_point=both)
hw_error_injection:
  enabled: true
  error_type: mxfp4                    # MXFP4 format (~21% rel error)
  injection_point: both                # W4A4: quantize weights AND activations
  apply_during: both                   # Both rollout and training
  target_modules: ["linear"]
  exclude_modules: ["lm_head", "embed_tokens", "lora_A", "lora_B",
                    "layers.0", "layers.27", "base_layer"]
  use_ste: true                        # CRITICAL: STE for gradient flow

# NO RIN/AQN (baseline)
noise_injection:
  enabled: false
```

### 2.2 Critical Settings

**STE (Straight-Through Estimator)**:
- `use_ste: true` - Enables gradient flow through quantized activations
- Without STE: E13a-f failed with 7-10% accuracy
- With STE: E13g succeeded with 60.88%

**Exclusions**:
- `lm_head`, `embed_tokens`: Keep in FP16 for numerical stability
- `lora_A`, `lora_B`: LoRA adapters stay in BF16
- `layers.0`, `layers.27`: First/last layers more sensitive
- `base_layer`: Exclude base layer weights (LoRA-specific)

---

## 3. Execution Plan

### 3.1 Pre-Execution Checklist

- [x] Script ready: `scripts/test_mxfp4_w4a4_ste_fix_e13h.sh`
- [x] STE fix verified in E13g
- [ ] SSH to A100 server
- [ ] Check disk space in /tmp
- [ ] Check no other experiments running
- [ ] Source proxy (if needed for any pip installs)

### 3.2 Execution Steps

```bash
# 1. SSH to A100
ssh root@90.90.102.18
docker exec -it verl-r3-test bash
cd /home/z00637938/workspace/verl

# 2. Check current branch
git branch  # Should be on feature/npu-aqn-test

# 3. Check no other experiments running
ps aux | grep main_dapo

# 4. Check disk space
df -h /tmp

# 5. Run E13h
bash scripts/test_mxfp4_w4a4_ste_fix_e13h.sh

# 6. Monitor progress
tail -f /tmp/mxfp4_w4a4_ste_fix_e13h/training.log

# 7. Check for step 20 validation results
grep 'val-core/openai/gsm8k/acc/mean@1' /tmp/mxfp4_w4a4_ste_fix_e13h/training.log
```

### 3.3 Monitoring

**Key metrics to watch**:
- Step 0 accuracy: Should be ~8% (baseline with quantization)
- Step 20 accuracy: Target 55-60%
- Training scores (critic/score/mean): Should increase from ~20% to ~50-60%
- Response length: Should be ~200-250 tokens

**Expected timeline**:
- Total training time: ~90-120 minutes
- Step 20 validation: ~60 minutes into training
- Steps 1-19: ~3 minutes per step
- Step 20 (validation): ~5 minutes

**Red flags** (if these occur, STOP and investigate):
- Step 20 accuracy < 20%: Training failure, STE not working
- Loss diverging or NaN: Numerical instability
- OOM errors: Memory configuration issue

---

## 4. Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| **Step 20 accuracy** | 55-60% | val-core/openai/gsm8k/acc/mean@1 |
| **vs E13g (NVFP4)** | -5% to -0% | E13h vs E13g difference |
| **vs E13a (old MXFP4)** | +50% | E13h 60% vs E13a 7.43% |
| **Training stability** | No divergence | Monitor loss/entropy |

**Pass/Fail**:
- ✅ **PASS**: Step 20 accuracy ≥ 50%
- ⚠️ **MARGINAL**: 40-50% accuracy (investigate but acceptable)
- ❌ **FAIL**: Step 20 accuracy < 40% (STE fix doesn't work for MXFP4)

---

## 5. Post-Execution Analysis

### 5.1 Data to Collect

After E13h completes:

```bash
# 1. Extract step 20 validation result
grep 'step:20' /tmp/mxfp4_w4a4_ste_fix_e13h/training.log | grep 'val-core'

# 2. Extract training progression
grep 'critic/score/mean' /tmp/mxfp4_w4a4_ste_fix_e13h/training.log

# 3. Copy log to permanent location
cp /tmp/mxfp4_w4a4_ste_fix_e13h/training.log \
   /home/z00637938/workspace/verl/logs/w4a4_experiments/e13h_mxfp4_w4a4_ste_fix_<accuracy>.log

# 4. Check checkpoint saved (if applicable)
ls -lh /tmp/mxfp4_w4a4_ste_fix_e13h/checkpoints/
```

### 5.2 Documentation Updates

After collecting results:

1. **Update E13_W4A4_EXPERIMENT_LOG.md**:
   - Add E13h result to comparison table
   - Add detailed E13h section with configuration and results

2. **Update ALL_EXPERIMENTS_SUMMARY.md**:
   - Add E13h entry with final accuracy
   - Update "Latest Experiments" section

3. **Commit and push**:
   ```bash
   git add docs/qerl/E13_W4A4_EXPERIMENT_LOG.md
   git add docs/qerl/ALL_EXPERIMENTS_SUMMARY.md
   git commit -m "docs: add E13h MXFP4 W4A4 results"
   git push personal feature/npu-aqn-test
   git push team feature/npu-aqn-test
   ```

---

## 6. Next Steps (After E13h)

### 6.1 If E13h Succeeds (≥50% accuracy)

**Next experiments**: Add RIN to MXFP4 W4A4

| ID | Config | Purpose |
|----|--------|---------|
| **E13i** | MXFP4 W4A4 + Global AQN | Baseline with static noise |
| **E13j** | MXFP4 W4A4 + RIN-targeted | SRDD-guided binary targeting |
| **E13k** | MXFP4 W4A4 + RIN-variable | SRDD-guided variable multipliers |

Expected improvements:
- E13i (global AQN): +2-4% over E13h
- E13j (RIN-targeted): +1-2% over E13i
- E13k (RIN-variable): +1-2% over E13j (best expected)

### 6.2 If E13h Fails (<40% accuracy)

**Investigate**:
1. Check if MXFP4 implementation differs from NVFP4
2. Check if STE is applied correctly to MXFP4
3. Compare MXFP4 vs NVFP4 quantization characteristics
4. Consider if MXFP4 error (~21%) is too high for W4A4 training

**Potential solutions**:
- Lower MXFP4 group size (32→16)
- Add RIN immediately (don't wait for baseline)
- Use hybrid W4A8 (4-bit weights, 8-bit activations)

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| E13h fails (<40%) | Medium | High | Have NVFP4 as fallback |
| OOM during training | Low | Medium | Same config as E13g (worked) |
| Training too slow | Low | Low | Expected ~2 hours |
| Log file lost | Medium | Medium | Copy to permanent location ASAP |
| /tmp disk full | Medium | High | Check disk space before starting |

---

## 8. Execution Log

### 8.1 Execution Timestamp

**Started**: 2026-01-15 03:30:15 UTC (PID: 27860)
**Completed**: 2026-01-15 04:32:34 UTC
**Duration**: ~62 minutes (step 0-20)

### 8.2 Key Events

- [x] E13h script execution attempted (multiple tries)
- [x] Ray cluster started (8 GPUs)
- [x] Training initialization started
- [x] **Logger configuration issue resolved** (changed `trainer.logger=null` to `trainer.logger='["console"]'`)
- [x] vLLM initialized with LoRA support
- [x] 182 HW error hooks registered for MXFP4 W4A4
- [x] Step 0 validation completed: **7.66% accuracy**
- [x] Step 20 validation completed: **56.41% accuracy** ✅

### 8.3 Configuration Issues Encountered and Resolved

**Issue 1: PyTorch Distributed Backend Error** (First attempt)
- **Error**: `ValueError: Duplicate device type cpu in backend string`
- **Resolution**: Container restart to kill zombie processes

**Issue 2: WandB API Key Error** (Second attempt)
- **Error**: `wandb.errors.errors.UsageError: api_key not configured`
- **Resolution**: Changed `trainer.logger=null` to `trainer.logger='["console"]'`

**Issue 3: TypeError with logger=null** (Third attempt)
- **Error**: `TypeError: 'NoneType' object is not iterable` at `verl/utils/tracking.py:53`
- **Root Cause**: `trainer.logger=null` creates NoneType, but tracking.py expects string or list
- **Resolution**: Updated script to use `trainer.logger='["console"]'` (console-only logging)

### 8.4 Results Summary

**Status**: **COMPLETED SUCCESSFULLY** ✅

**Step 0 Validation Results** (Baseline with MXFP4 W4A4):
- **Accuracy**: 7.66% (`val-core/openai/gsm8k/acc/mean@1`)
- **Reward**: 7.66% (`val-aux/openai/gsm8k/reward/mean@1`)
- This confirms MXFP4 W4A4 quantization is active and working

**Step 20 Validation Results** (After Training):
- **Accuracy**: **56.41%** (`val-core/openai/gsm8k/acc/mean@1`) ✅
- **Reward**: 56.41% (`val-aux/openai/gsm8k/reward/mean@1`)
- **Training score**: 41.89% (critic/score/mean)
- **Improvement**: 7.66% → 56.41% = **+48.75 percentage points**

**Comparison with E13g (NVFP4 W4A4)**:
| Metric | E13g (NVFP4) | E13h (MXFP4) | Difference |
|--------|--------------|--------------|------------|
| Step 20 accuracy | 60.88% | 56.41% | -4.47% |
| Quantization error | ~15% rel | ~21% rel | +6% |
| STE enabled | ✅ Yes | ✅ Yes | Same |
| Result | PASS | PASS | Both succeed |

**Configuration Confirmed**:
- Error type: MXFP4
- Injection point: both (W4A4 mode - quantize weights AND activations)
- Apply during: both (rollout and training)
- STE enabled: True ✅
- Target modules: ['linear']
- Exclude modules: ['lm_head', 'embed_tokens', 'lora_A', 'lora_B', 'layers.0', 'layers.27', 'base_layer']
- 182 quantization hooks registered

**Final Verdict**: ✅ **PASS**
- Step 20 accuracy 56.41% exceeds 50% threshold
- STE fix **works for MXFP4 W4A4**
- Performance gap vs NVFP4 (-4.47%) is acceptable given higher MXFP4 quantization error
- Ready to proceed with RIN experiments (E13i/j/k)

---

## References

- E13g success log: `/home/zheng/workspace/verl/logs/w4a4_experiments/e13g_nvfp4_w4a4_ste_fix_60.88.log`
- E13 experiment log: `docs/qerl/E13_W4A4_EXPERIMENT_LOG.md`
- STE fix commit: `a04eacda` - W4A4 activation quantization must use STE
- Script location: `scripts/test_mxfp4_w4a4_ste_fix_e13h.sh`
