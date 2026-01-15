# CRITICAL: test_freq Configuration Issue

**Discovered**: 2026-01-15
**Severity**: **HIGH** - Wasting compute and missing accuracy measurements
**Status**: ⚠️ **AFFECTS ALL EXPERIMENTS** ⚠️

---

## Problem Summary

**Current configuration** (all E13 experiments):
```bash
trainer.total_epochs=1       # → 29 training steps
trainer.test_freq=20         # → Validation ONLY at step 20
trainer.val_before_train=True  # → Validation at step 0
```

**What happens**:
1. Step 0: Validation runs (baseline)
2. Steps 1-20: Training + validation at step 20
3. **Steps 21-28: Training continues BUT NO VALIDATION** ❌
4. Training ends at step 28 or 29 WITHOUT final validation ❌

**Impact**:
- We report step 20 results (e.g., E13h: 56.41%)
- Training continues for 8-9 more steps (~15 minutes compute)
- We NEVER measure the final accuracy after those additional steps
- Potential accuracy gains are UNKNOWN

---

## Evidence from E13h

### Training Score Progression (critic/score/mean)

| Step | Critic Score | Val Accuracy | Status |
|------|--------------|--------------|--------|
| 0 | - | 7.66% | ✓ Validated |
| 1 | 19.92% | - | No validation |
| 20 | 41.89% | **56.41%** | ✓ Validated **(reported)** |
| 26 | **61.04%** | **UNKNOWN** | ❌ No validation |
| 27 | 58.40% | **UNKNOWN** | ❌ No validation |
| 28-29 | ? | **UNKNOWN** | ❌ Training ended, no validation |

**Key Observation**: Critic score increased from 41.89% (step 20) to 61.04% (step 26), suggesting **validation accuracy likely improved too!**

### Training Progress Indicators

From E13h log:
```
Training Progress:  83%|████████▎ | 24/29 [1:06:31<13:52, 166.45s/it]
Training Progress:  86%|████████▌ | 25/29 [1:09:13<11:00, 165.20s/it]
Training Progress:  90%|████████▉ | 26/29 [1:12:03<08:19, 166.59s/it]
Training Progress:  93%|█████████▎| 27/29 [1:14:51<05:34, 167.14s/it]
Training Progress:  97%|█████████▋| 28/29 [1:17:38<02:46, 166.91s/it]
```

Training reached 97% (28/29 steps), but **step 28 results are not logged**, and there's **NO final validation**.

---

## Compute Waste Analysis

### Per Experiment

- Steps 21-28: ~8 steps × 170s/step = **~23 minutes** wasted
- GPU utilization: 8 × A100 GPUs
- Total waste: **~3 GPU-hours per experiment**

### Across All Experiments

Affected experiments: E13a, E13b, E13c, E13d, E13e, E13f, E13g, **E13h**
- Total waste: **~8 experiments × 3 GPU-hours = 24 GPU-hours**
- This is equivalent to running **~1.5 additional full experiments**

---

## Missed Insights

### 1. Unknown Final Accuracy

We report step 20 accuracy, but the model trained for 30-40% more steps. The final accuracy could be:
- **Higher**: If training continued to improve (likely given critic score 41% → 61%)
- **Similar**: If model plateaued
- **Lower**: If model overfit (unlikely with LoRA)

**E13h Speculation**:
- Step 20: 56.41% (reported)
- Step 28: **58-60%?** (based on critic score improvement from 42% to 61%)
- **Potential 2-4% accuracy underestimation**

### 2. Training Dynamics

Without final validation, we don't know:
- Does accuracy improve linearly or plateau?
- Is there overfitting in late stages?
- Should we train for fewer steps (if plateaued at step 20)?
- Should we train longer (if still improving at step 28)?

### 3. Experiment Comparisons

Comparing E13h (56.41% at step 20) vs E13g (60.88% at step 20) might be unfair if:
- E13h's final accuracy (step 28) is actually 59-60%
- The 4.47% gap might actually be only 0-1%

---

## Root Cause

### Verl Trainer Validation Logic

The verl trainer validates when:
1. `val_before_train=True` → Step 0
2. `step % test_freq == 0` → Every test_freq steps
3. **NO automatic final validation**

There is **NO `val_after_train` flag** or equivalent in the config.

### Why test_freq=20?

Likely copied from examples where:
- Total steps might be 100+
- Validating every 20 steps makes sense (5 validations total)
- Final step might be a multiple of 20

But for 29 steps:
- test_freq=20 means validation at step 0, 20, 40...
- Step 29 is NOT a multiple of 20
- **No final validation**

---

## Solutions

### Solution 1: Set test_freq to Final Step (Recommended for 1-epoch)

```bash
trainer.total_epochs=1
trainer.test_freq=29  # Or total_steps - 1
```

**Pros**: Guaranteed final validation
**Cons**: Only 2 validations total (step 0, step 29)

### Solution 2: Use Divisor of Total Steps

```bash
trainer.total_epochs=1
trainer.test_freq=14  # Validates at steps 0, 14, 28
# OR
trainer.test_freq=9   # Validates at steps 0, 9, 18, 27
```

**Pros**: Multiple validation points + near-final validation
**Cons**: Requires calculating divisors

### Solution 3: Add 1 Extra Step

```bash
trainer.total_epochs=1
trainer.test_freq=20
# Adjust data so total_steps = 30 (multiple of 20)
# This requires changing batch sizes
```

**Pros**: Clean multiples of 20
**Cons**: Artificial, not aligned with actual data

### Solution 4: Implement val_after_train (Code Change)

Modify verl trainer to add:
```python
if config.val_after_train and step == total_steps - 1:
    run_validation()
```

**Pros**: Best solution long-term
**Cons**: Requires code changes

---

## Immediate Actions

### For Completed Experiments (E13a-h)

1. **Document limitation**: All reported accuracies are at step 20, not final
2. **Acknowledge uncertainty**: Final accuracy (step 28-29) is UNKNOWN
3. **Consider re-running**: If E13h is critical, re-run with test_freq=29

### For Future Experiments (E13i-j-k...)

**MUST FIX BEFORE RUNNING**:

```bash
# Option A: Single final validation (fast, simple)
trainer.total_epochs=1
trainer.test_freq=29
trainer.val_before_train=True

# Option B: Multiple validations (better insights)
trainer.total_epochs=1
trainer.test_freq=10  # Validates at 0, 10, 20
# Then manually add validation at step 29 OR use solution 4
```

---

## Updated Experiment Protocol

### Standard Configuration (1 Epoch, 29 Steps)

```bash
# CORRECTED configuration for E13i onwards
trainer.total_epochs=1
trainer.test_freq=10  # Validates at steps 0, 10, 20
trainer.val_before_train=True

# TODO: Check if can add val_after_train flag
# OR: Set test_freq=29 for final validation only
```

### Validation Schedule

| Config | Step 0 | Step 10 | Step 20 | Step 29 | Total Validations |
|--------|--------|---------|---------|---------|-------------------|
| **Current (broken)** | ✓ | ✗ | ✓ | ✗ | 2 (missing final!) |
| **test_freq=29** | ✓ | ✗ | ✗ | ✓ | 2 (final only) |
| **test_freq=10** | ✓ | ✓ | ✓ | ✗ | 3 (missing final!) |
| **test_freq=14** | ✓ | ✗ | ✓ (14) | ✓ (28) | 3 (near-final) |
| **Best: test_freq=9** | ✓ | ✓ (9) | ✓ (18) | ✓ (27) | 4 (near-final) |

**Recommendation**: Use `test_freq=9` for 3 intermediate validations + near-final at step 27

---

## Re-run Decision Matrix

| Experiment | Importance | Re-run? | Reason |
|------------|------------|---------|--------|
| E13a-f | Low | ❌ No | All failed anyway (<10%), final validation won't change conclusion |
| E13g | Medium | ~ Maybe | 60.88% at step 20, might be 62-63% at step 29, but not critical |
| **E13h** | **High** | **✓ Yes** | **Baseline for RIN experiments**, need accurate final accuracy |

### E13h Re-run Justification

E13h (56.41% at step 20) is the baseline for ALL RIN experiments (E13i/j/k). If actual final accuracy is 58-60%, then:
- All RIN target accuracies shift up by 2-4%
- Comparison with E13g (60.88%) changes significantly
- Hypothesis confidence levels need adjustment

**Recommendation**: Re-run E13h with `test_freq=9` or `test_freq=29` to get accurate final validation.

---

## Long-term Fix

### Code Change Proposal

File: `verl/trainer/ppo_trainer.py` (or equivalent)

```python
# Add to config
val_after_train: bool = True  # NEW FLAG

# In training loop
for step in range(total_steps):
    # ... training ...

    # Existing validation
    if step % test_freq == 0:
        validate()

    # NEW: Final validation
    if config.val_after_train and step == total_steps - 1:
        validate()
```

This ensures validation ALWAYS runs at the final step, regardless of test_freq.

---

## Checklist for Future Experiments

Before running ANY experiment:

- [ ] Calculate total_steps from batch sizes and epochs
- [ ] Choose test_freq that divides total_steps OR equals total_steps-1
- [ ] Verify validation schedule (step 0, intermediate, final)
- [ ] Document validation steps in experiment plan
- [ ] After completion, check log for final step validation

---

## Impact on Current Work

### RIN Experiment Plan

**ALL planned experiments (E13i/j/k/l/m) MUST use corrected configuration**:

```bash
# In scripts/test_mxfp4_w4a4_rin_*.sh
trainer.total_epochs=1
trainer.test_freq=9  # Validates at 0, 9, 18, 27 (near-final)
# OR
trainer.test_freq=29  # Validates at 0, 29 (final only)
```

**Update RIN_EXPERIMENT_PLAN_SYSTEMATIC.md**:
- Add note about test_freq configuration
- Adjust expected outcomes if E13h is re-run
- Document validation schedule for each experiment

---

## Lessons Learned

1. **Always verify validation schedule** before running experiments
2. **Check log outputs** during early experiments to catch config issues
3. **Don't blindly copy configuration** from examples without understanding
4. **Document configuration reasoning** in experiment plans
5. **Calculate validation steps** based on total_steps, not guessing

---

## References

- **E13h Log**: `/home/z00637938/workspace/verl/logs/w4a4_experiments/e13h_mxfp4_w4a4_ste_fix_56.41.log`
- **E13h Script**: `scripts/test_mxfp4_w4a4_ste_fix_e13h.sh`
- **Config File**: `verl/trainer/config/ppo_trainer.yaml`

---

## Status Tracking

- [ ] E13h re-run decision made
- [ ] All future scripts updated with corrected test_freq
- [ ] RIN experiment plan updated
- [ ] Code change proposal submitted (val_after_train flag)
- [ ] All team members notified of this issue
