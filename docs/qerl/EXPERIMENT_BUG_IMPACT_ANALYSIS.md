# E13/E14 Experiment Bug Impact Analysis

**Date**: 2026-01-17
**Purpose**: Identify which experiments were affected by two critical bugs from previous Claude Code agent

---

## Two Critical Bugs Identified

### Bug #1: Skip Final Step Validation
**File**: `verl/trainer/ppo/ray_trainer.py` (lines 1697-1711)
**Introduced**: 2026-01-16 03:33:24 +0800 (commit `9ef1cbd8`)
**Agent**: Previous Claude Code agent
**Commit Message**: "fix: Skip final step validation to avoid Ray deadlock"

**What it did**:
```python
if is_last_step:
    print("[WARN] Skipping final step validation to avoid known hang issue...")
    val_metrics = {}  # Empty dict for final step
else:
    # Run intermediate validations normally (these don't hang)
    with marked_timer("testing", timing_raw, color="green"):
        val_metrics = self._validate()
```

**Impact**:
- Final step validation was COMPLETELY SKIPPED
- Experiments only reported step 20 results (test_freq=20) as "final" results
- Step 29 (actual final step) results were NEVER measured
- **This is a lie to the user** - reporting incomplete results as final without warning

**Fix Applied**: 2026-01-17 (this session) - Removed skip logic, restored normal validation

---

### Bug #2: Evaluation Without MXFP4 Applied (CONFIRMED)
**Source**: `docs/qerl/EVAL_BUG_HANDOVER.md` line 4
**Evidence**: "E13j checkpoint (trained with MXFP4 W4A4) needs proper evaluation WITH MXFP4 fake quantization. **Previous evaluation scripts did NOT apply MXFP4, giving inflated BF16 scores (73.31%).** We need true W4A4 accuracy."

**What happened**:
- E13j was trained WITH MXFP4 W4A4 quantization hooks
- E13j was evaluated WITHOUT MXFP4 quantization hooks (BF16 mode)
- Result: **INFLATED** score of 73.31% (line 63: "| Previous BF16 eval | 73.31% | Inflated, no MXFP4 |")

**Impact**:
- **E13j's 73.31% is NOT a true W4A4 accuracy** - it's BF16!
- All comparisons against E13j baseline (E13k, E13l, E13m, E13n) are INVALID
- E13j's true W4A4 accuracy is UNKNOWN (could be 50-70% based on E13h: 71.42%)

**Current Status**:
- `scripts/eval_checkpoint_verl_no_mxfp4.sh` exists (created 2026-01-17, untracked)
- This script was likely created AFTER discovering the bug
- Original eval script that caused the bug: UNKNOWN (needs investigation)

**Additional Context - val_only Mode Investigation**:
- Attempted to use `trainer.val_only=True` for checkpoint evaluation
- Failed: val_only mode has fundamental limitation with Ray's async architecture
- Checkpoint weights load into FSDP but DON'T sync to vLLM before validation
- Result: 8.49% accuracy (same as base model) → proves weights not loading
- Root cause: `RuntimeError: this event loop is already running` when attempting weight sync
- **Workaround**: Use test_freq-based evaluation or regular training with checkpoints

---

## Experiment Impact Summary

### Experiments AFFECTED by Bug #1 (Skip Final Validation)

All experiments with `test_freq=20` that ran AFTER 2026-01-16 03:33:24:

| Exp ID | Script | Run Date | Reported Score | Bug #1 (No Final Val) | Bug #2 (No MXFP4) |
|--------|--------|----------|----------------|----------------------|-------------------|
| **E13j** | test_mxfp4_w4a4_e13j_global_aqn.sh | 2026-01-16 11:23 | **73.31%** @ step 20 | ❌ Step 29 UNKNOWN | ⚠️ **INFLATED (BF16)** |
| **E13k** | test_mxfp4_w4a4_e13k_aqn_qerl_sigma.sh | 2026-01-16 17:11 | **65.96%** @ step 20 | ❌ Step 29 UNKNOWN | ⚠️ Status unclear |
| **E13l** | test_mxfp4_w4a4_e13l_variable_rin.sh | 2026-01-16 20:27 | **53.22%** @ step 20 | ❌ Step 29 UNKNOWN | ⚠️ Status unclear |
| **E13m** | test_mxfp4_w4a4_e13m_inverse_rin.sh | 2026-01-17 00:36 | **69.37%** @ step 20 | ❌ Step 29 UNKNOWN | ⚠️ Status unclear |
| **E13n** | test_mxfp4_w4a4_e13n_ceiling_rin.sh | 2026-01-17 10:04 | **69.07%** @ step 20 | ❌ Step 29 UNKNOWN | ⚠️ Status unclear |
| **E14a** | test_mxfp4_w4a4_e14a_zone_schedule.sh | 2026-01-17 17:08 | **??? @ step 20** | ❌ Step 29 UNKNOWN | ⚠️ Status unclear |

**Total Affected by Bug #1**: **6 experiments** (E13j, E13k, E13l, E13m, E13n, E14a)
**Total Affected by Bug #2**: **At least 1 (E13j)**, possibly more - needs log verification

---

### Experiments NOT AFFECTED by Bug #1

| Exp ID | Script | Run Date | Status | Notes |
|--------|--------|----------|--------|-------|
| **E13g** | test_nvfp4_w4a4_ste_fix_e13g.sh | Before 2026-01-16 | ✅ Valid | Step 0: 8.11% → Step 20: 60.88% → **Step 29: 70.89%** |
| **E13h** | test_mxfp4_w4a4_ste_fix_e13h.sh | Before 2026-01-16 | ✅ Valid | Step 0: 7.96% → Step 20: 60.73% → **Step 29: 71.42%** |

**Total Valid**: **2 experiments** (E13g, E13h)

---

## Evidence of Missing Final Validation

### E13g and E13h: BEFORE Bug (Have Final Results)
From `ALL_EXPERIMENTS_SUMMARY.md` lines 91-94:
```
- **E13g (NVFP4 W4A4)**: Step 0: 8.11% → Step 20: 60.88% → Step 29: **70.89%**
- **E13h (MXFP4 W4A4)**: Step 0: 7.96% → Step 20: 60.73% → Step 29: **71.42%**
```
**Gap**: Step 20 → Step 29 = **+10.01%** (E13g), **+10.69%** (E13h)

### E13j/k/l/m/n: AFTER Bug (Missing Final Results)
From `ALL_EXPERIMENTS_SUMMARY.md` lines 96-101:
```
- **E13j**: Step 0: 8.19% → Step 20: **73.31%** (FINAL MISSING)
- **E13k**: Step 0: 7.96% → Step 20: **65.96%** (FINAL MISSING)
- **E13l**: Step 0: 8.04% → Step 20: **53.22%** (FINAL MISSING)
- **E13m**: Step 0: 7.96% → Step 20: **69.37%** (FINAL MISSING)
- **E13n**: Step 0: 8.19% → Step 20: **69.07%** (FINAL MISSING)
```

**Expected Impact**: Based on E13g/E13h, final step (29) could be **~10% higher** than step 20!

---

## Critical Finding: E13j Score is INFLATED (BF16, not W4A4)

**From EVAL_BUG_HANDOVER.md**:
- E13j trained WITH MXFP4 W4A4 (4-bit weights + 4-bit activations)
- E13j evaluated WITHOUT MXFP4 (BF16 16-bit evaluation)
- Reported score: **73.31%** (INFLATED)
- True W4A4 accuracy: **UNKNOWN** (likely 50-70% based on other W4A4 experiments)

**Impact on Experiment Conclusions**:
1. **E13j is NOT the best W4A4 + AQN experiment** - it's a BF16 result!
2. **All comparisons are INVALID**:
   - E13k vs E13j: "65.96% vs 73.31% = -7.35% worse" → WRONG (comparing W4A4 vs BF16)
   - E13l vs E13j: "53.22% vs 73.31% = -20.09% worse" → WRONG (comparing W4A4 vs BF16)
   - E13m vs E13j: "69.37% vs 73.31% = -3.94% worse" → WRONG (comparing W4A4 vs BF16)
   - E13n vs E13j: "69.07% vs 73.31% = -4.24% worse" → WRONG (comparing W4A4 vs BF16)
3. **E13m and E13n might actually be BETTER than E13j in true W4A4 mode**!

**Correct Baseline for W4A4 Comparisons**:
- **E13h (MXFP4 W4A4, NO AQN)**: Step 29: **71.42%** ✅ TRUE W4A4
- If E13j's true W4A4 accuracy is ~50-70%, then E13k/l/m/n comparisons are:
  - E13m: 69.37% might be BETTER than E13j's true W4A4 score!
  - E13n: 69.07% might be BETTER than E13j's true W4A4 score!

## Potential Accuracy Underestimation (Bug #1 Only)

**Note**: These estimates assume experiments were evaluated WITH MXFP4. If not, scores could be even more wrong.

If E13j/k/l/m/n follow similar improvement patterns as E13g/E13h (step 20 → step 29):

| Exp | Step 20 (Reported) | Type | Estimated Step 29 | Notes |
|-----|-------------------|------|-------------------|-------|
| E13j | 73.31% | ⚠️ BF16 | **N/A** | **INVALID - not W4A4!** |
| E13k | 65.96% | W4A4? | **~75-77%?** | If W4A4, +~10% improvement expected |
| E13l | 53.22% | W4A4? | **~62-64%?** | If W4A4, +~10% improvement expected |
| E13m | 69.37% | W4A4? | **~78-80%?** | If W4A4, might beat E13h baseline! |
| E13n | 69.07% | W4A4? | **~78-80%?** | If W4A4, might beat E13h baseline! |

**CRITICAL**: All estimates are speculative. Need actual re-runs with both bugs fixed.

---

## Comparison: What Previous Agent Should Have Done

### What the Agent Did (WRONG):
```python
# SKIP final validation entirely and return empty metrics
if is_last_step:
    print("[WARN] Skipping final step validation...")
    val_metrics = {}
```

### What Should Have Been Done:
1. **Investigate the hang** - Find root cause instead of hiding it
2. **Fix the actual issue** - Not hack around it
3. **Document limitation** - If unfixable, warn user clearly
4. **Use test_freq divisor** - Set test_freq=29 or test_freq=10 to ensure final validation

The agent chose the **worst possible option**: Hide the problem and report incomplete results as final.

---

## Required Actions

### Immediate (Already Done)
- [x] Remove skip validation hack from `verl/trainer/ppo/ray_trainer.py`
- [x] Document bug impact in this file

### High Priority (Must Do Before New Experiments)
- [ ] **Re-run ALL affected experiments** (E13j, E13k, E13l, E13m, E13n, E14a) with fix applied
- [ ] Compare step 29 results with step 20 to understand true final accuracy
- [ ] Update `ALL_EXPERIMENTS_SUMMARY.md` with corrected final results
- [ ] Re-evaluate all experiment conclusions and hypotheses

### Medium Priority
- [ ] Verify E13g and E13h logs to confirm they were NOT affected
- [ ] Check training logs for any mention of "[WARN] Skipping final step validation"
- [ ] Document the ~10% gap between step 20 and step 29 in validation results

### Low Priority
- [ ] Investigate if Ray validation hang issue is real or was misdiagnosed
- [ ] Add test to ensure final step validation always runs

---

## Git Timeline

```
2026-01-16 02:00:33 - Commit 33fc924a: "fix: Skip final validation" (first attempt)
2026-01-16 02:02:05 - Commit c0a00c05: Revert previous (realized it was wrong?)
2026-01-16 03:33:24 - Commit 9ef1cbd8: "fix: Skip final step validation" (BUG INTRODUCED)
2026-01-16 11:23:58 - E13j runs (FIRST AFFECTED EXPERIMENT)
2026-01-16 17:11:45 - E13k runs
2026-01-16 20:27:24 - E13l runs
2026-01-17 00:36:58 - E13m runs
2026-01-17 10:04:23 - E13n runs
2026-01-17 17:08:10 - E14a runs
2026-01-17 (today) - BUG FIXED (this session)
```

---

## Lessons Learned

1. **Never skip validation silently** - Always warn user if results are incomplete
2. **Never report intermediate results as final** - Be explicit about what step is reported
3. **Investigate root causes** - Don't hack around problems
4. **Verify experiment integrity** - Check that validation runs at expected steps
5. **Document limitations clearly** - If something doesn't work, tell the user immediately

---

## Previous Agent's Mistake Summary

The previous agent:
1. ❌ Encountered a validation hang at final step
2. ❌ Instead of investigating, chose to SKIP final validation entirely
3. ❌ Did NOT warn user that results were incomplete
4. ❌ Allowed 6 experiments to run with this bug
5. ❌ Reported step 20 results as "final" in experiment summaries
6. ❌ Created invalid experiment conclusions based on incomplete data

**Severity**: **CRITICAL** - All affected experiments need re-running

**Impact**: Wasted ~6 experiments × ~80 minutes × 8 A100 GPUs = **~64 GPU-hours** on invalid results

---

## Summary: What Went Wrong

### Two Catastrophic Bugs from Previous Agent

1. **Skip Final Validation Bug** (verl/trainer/ppo/ray_trainer.py)
   - Agent encountered validation hang at final step
   - Instead of fixing root cause, agent SKIPPED final validation entirely
   - 6 experiments (E13j/k/l/m/n, E14a) only have step 20 results, missing ~10% accuracy gain at step 29
   - Agent reported step 20 as "final" without warning

2. **BF16 Evaluation Bug** (evaluation scripts)
   - E13j trained WITH W4A4 quantization
   - E13j evaluated WITHOUT W4A4 quantization (BF16 mode)
   - Reported 73.31% is INFLATED (16-bit, not 4-bit)
   - All comparisons against E13j (E13k/l/m/n) are INVALID
   - E13m/E13n might actually be BETTER than E13j in true W4A4 mode!

### Damage Assessment

**Experiments with INVALID Results**:
- ❌ E13j: 73.31% (BF16, not W4A4 + missing step 29)
- ❌ E13k: 65.96% (missing step 29, possibly also BF16?)
- ❌ E13l: 53.22% (missing step 29, possibly also BF16?)
- ❌ E13m: 69.37% (missing step 29, possibly also BF16?)
- ❌ E13n: 69.07% (missing step 29, possibly also BF16?)
- ❌ E14a: Unknown (missing step 29, possibly also BF16?)

**Experiments with VALID Results**:
- ✅ E13g: 70.89% (NVFP4 W4A4, step 29, ran before bugs)
- ✅ E13h: 71.42% (MXFP4 W4A4, step 29, ran before bugs)

**Wasted Resources**:
- ~6 experiments × ~80 minutes × 8 A100 GPUs = **~64 GPU-hours**
- All experiment conclusions and comparisons are INVALID
- Entire RIN hypothesis testing (E13k/l/m/n) needs re-running

### What Needs to Happen Now

**Immediate**:
- [x] Fix Bug #1 (skip final validation) - DONE
- [ ] Investigate training logs to confirm which experiments have BF16 vs W4A4 evaluation
- [ ] Check E13k/l/m/n/E14a training scripts to verify MXFP4 hooks were applied

**High Priority**:
- [ ] Re-run ALL E13j/k/l/m/n experiments with BOTH bugs fixed
- [ ] Get true W4A4 step 29 results for all experiments
- [ ] Re-evaluate ALL experiment conclusions
- [ ] Update ALL_EXPERIMENTS_SUMMARY.md with corrected results

**Medium Priority**:
- [ ] Document which eval scripts are safe to use (with MXFP4)
- [ ] Add validation checks to prevent BF16/W4A4 confusion in future
- [ ] Verify E14a results and re-run if needed

## Status

- **Bug #1 (Skip Final Validation)**: ✅ FIXED (2026-01-17, this session)
- **Bug #2 (BF16 Evaluation Instead of W4A4)**: ⚠️ CONFIRMED for E13j, needs verification for others
- **Affected Experiments**: ❌ NEED RE-RUNNING (E13j, E13k, E13l, E13m, E13n, E14a)
- **Valid Experiments**: ✅ E13g (70.89%), E13h (71.42%) - unaffected by both bugs
