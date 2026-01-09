# MXFP4 + AQN Next Steps Plan

**Date**: 2026-01-08
**Status**: Ready for Execution

---

## Executive Summary

Expert analysis identified **5 critical issues** explaining why AQN hurts performance (-2.57% vs MXFP4-only):

| Issue | Impact | Fix |
|-------|--------|-----|
| **Target Mismatch** | AQN targets RMSNorm, MXFP4 targets Linear | Align AQN to Linear layers |
| **Sigma Too Weak** | 0.05 sigma for 21% error (should be 0.15-0.20) | Scale sigma 3-4x |
| **Training Too Short** | 58 steps, ~6 steps/stage | Increase to 174+ steps (3 epochs) |
| **Error Too High** | MXFP4 21% vs NVFP4 1% (21x worse) | Consider MXFP8 fallback |
| **No Time to Adapt** | Model can't converge in 6 steps/stage | More steps per stage |

---

## Root Cause: QeRL vs Our Implementation

| Aspect | QeRL (NVIDIA) | Our Implementation | Gap |
|--------|---------------|---------------------|-----|
| Format | NVFP4 (E4M3, 16-elem) | MXFP4 (E8M0, 32-elem) | 21x worse error |
| Quant Error | ~1% | ~21% | **21x** |
| AQN Target | LayerNorm (propagates through Linear) | RMSNorm (separate path) | **Mismatch** |
| AQN Sigma | 0.05 for 1% error | 0.05 for 21% error | **20x too weak** |
| Training Steps | ~200-300 | 58 | **~5x too short** |

---

## Prioritized Experiment Plan

### TIER 1: Critical Fixes (Execute First)

#### Experiment 1A: Target Alignment + Longer Training
**Goal**: Fix the two most critical issues together

```yaml
# Configuration
trainer.total_epochs: 3  # 174 steps (was 1 epoch = 58 steps)

trainer.hw_error_injection:
  enabled: true
  error_type: mxfp4
  injection_point: weight
  target_modules: ['linear']
  apply_during: both

trainer.noise_injection:
  enabled: true
  sigma_start: 0.05      # Keep QeRL default first
  sigma_end: 0.0005
  num_stages: 10
  target_modules: ['linear']  # CRITICAL: Match MXFP4 target
```

**Expected**: 70-72% (vs 67.48% current)

---

#### Experiment 1B: Scaled Sigma (if 1A shows improvement)
**Goal**: Scale AQN sigma proportional to MXFP4 error

```yaml
trainer.noise_injection:
  enabled: true
  sigma_start: 0.15      # 3x original (for 21% error)
  sigma_end: 0.0015
  num_stages: 10
  target_modules: ['linear']
```

**Expected**: 72-74% (if target alignment works)

---

### TIER 2: Format Alternatives (If TIER 1 < 72%)

#### Experiment 2A: MXFP8 (Lower Error Format)
**Goal**: Reduce quantization error to trainable level

```yaml
trainer.hw_error_injection:
  enabled: true
  error_type: mxfp8  # ~8-10% error vs 21% for MXFP4
  injection_point: weight
  target_modules: ['linear']
```

**Expected**: 74-75% (MXFP8 has ~4x less error)

---

#### Experiment 2B: Mixed Precision (Exclude Worst Layers)
**Goal**: Keep layers 10-17 (worst SQNR) in BF16

```yaml
trainer.hw_error_injection:
  enabled: true
  error_type: mxfp4
  target_layers: [0,1,2,3,4,5,6,7,8,9,18,19,20,21,22,23,24,25,26,27]
  # Excludes layers 10-17 (highest error from SRDD)
```

**Expected**: 73-75%

---

### TIER 3: Diagnostics (If Unexpected Results)

#### Experiment 3A: AQN-Only (Verify Implementation)
```yaml
trainer.hw_error_injection.enabled: false
trainer.noise_injection.enabled: true
trainer.noise_injection.target_modules: ['linear']
```

**Expected**: ≥75% (AQN alone shouldn't hurt)

---

## Sigma Sweep Guide

| sigma_start | sigma_end | Ratio to Error | Use Case |
|-------------|-----------|----------------|----------|
| 0.05 | 0.0005 | 0.24x (QeRL default) | NVFP4 only |
| 0.10 | 0.001 | 0.48x | Conservative |
| **0.15** | **0.0015** | **0.71x** | **Recommended** |
| 0.20 | 0.002 | 0.95x | Aggressive |
| 0.30 | 0.003 | 1.4x | Maximum |

---

## Success Metrics

### Accuracy Targets

| Outcome | Accuracy | Interpretation |
|---------|----------|----------------|
| **Success** | ≥72% | AQN working, continue optimization |
| **Partial** | 70-72% | Needs more sigma or epochs |
| **No Change** | 67-70% | Target mismatch not fixed |
| **Regression** | <67% | Bug in implementation |

### SRDD Improvement Targets

| Metric | Pre-Training | Target Post-Training |
|--------|--------------|----------------------|
| SQNR (dB) | 16.91 | ≥18.0 (+1 dB) |
| Deadzone % | 22.88% | ≤20% (-2%) |
| Relative Error % | 36.41% | ≤34% (-2%) |

---

## Execution Order

1. **Experiment 1A**: Target alignment + 3 epochs (~45 min)
2. **SRDD scan** on checkpoint (compare metrics)
3. **Experiment 1B**: If 1A shows improvement, try scaled sigma
4. **Experiment 2A/2B**: If still <72%, try MXFP8 or mixed precision

---

## Script Commands

```bash
# Experiment 1A: Target Alignment + Longer Training
bash scripts/test_mxfp4_exp1a_aligned.sh 8

# Experiment 1B: Scaled Sigma
bash scripts/test_mxfp4_exp1b_scaled_sigma.sh 8

# Experiment 2A: MXFP8
bash scripts/test_mxfp4_exp2a_mxfp8.sh 8
```

---

## Files to Modify

1. `verl/utils/noise_injection.py` - Add `target_modules` parameter for AQN
2. `verl/workers/fsdp_workers.py` - Pass target_modules to noise injection
3. `scripts/test_mxfp4_exp1a_aligned.sh` - New experiment script

---

## Expected Timeline

| Day | Task | Expected Result |
|-----|------|-----------------|
| 1 | Implement AQN target_modules | Code ready |
| 1 | Run Experiment 1A | 70-72% accuracy |
| 2 | Run SRDD comparison | Metrics improvement data |
| 2 | Run Experiment 1B (if 1A works) | 72-74% accuracy |
| 3 | Run TIER 2 (if needed) | 73-75% accuracy |

---

## Key Insight

**The fundamental issue is that AQN and MXFP4 operate on different computational paths.**

- MXFP4: Quantizes Linear layer weights → affects output
- AQN: Adds noise to RMSNorm weights → affects normalization only

**Fix**: Either move AQN to Linear layers OR move MXFP4 to operate after RMSNorm.
