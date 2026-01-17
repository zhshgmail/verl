# E13k_v2 Re-Run Status

**Experiment**: E13k_v2 - Global AQN with QeRL sigma values
**Date**: 2026-01-17
**Server**: root@90.90.102.18, container: verl-r3-test
**Output**: `/tmp/mxfp4_w4a4_e13k_v2_aqn_qerl_sigma/training.log`

---

## Current Status (Last Update: 2026-01-18 02:52 server time)

**ATTEMPT 2 - IN PROGRESS**:
**Progress**: ⏳ **RUNNING** (1/29 steps)
**Step 0**: ✅ **VALIDATED** - 8.34%
**Status**: Baseline validated, training progressing

**ATTEMPT 1 - FAILED**:
**Step 0**: ✅ 7.73% (validated)
**Step 1**: ❌ HUNG at 10:35 on 2026-01-17 (16 hours wasted)

---

## Configuration

**Key Differences from E13j_v2**:
- E13j_v2: sigma_start=0.05, sigma_end=0.0005 (QeRL code default)
- E13k_v2: sigma_start=0.01, sigma_end=0.0001 (QeRL training scripts actual values)

**Purpose**: Test QeRL's actual sigma values from their training scripts (dapo_qwen*.sh). Lower sigma (0.01 vs 0.05) may provide more stable training with less aggressive noise.

**Expected Behavior**:
- Original E13h (no AQN): 71.42%
- Original E13j (sigma=0.05): step 20: 68.84%, step 29: SKIPPED
- E13j_v2 (sigma=0.05): step 20: 70.20%, step 29: 65.96%
- E13k_v2 (sigma=0.01): TBD

---

## Results So Far

| Step | Result | Type | Notes |
|------|--------|------|-------|
| 0 | 7.73% | Baseline | ✅ Confirmed (vs E13j_v2: 7.96%) |
| 20 | ⏳ Pending | Validation | Expected ~11:50 |
| 29 | ⏳ Pending | FINAL VALIDATION | Expected ~13:30 |

---

## Monitoring Commands

**Check step 0 baseline:**
```bash
ssh root@90.90.102.18 "docker exec verl-r3-test bash -c 'grep \"step:0\" /tmp/mxfp4_w4a4_e13k_v2_aqn_qerl_sigma/training.log | grep \"val-core\"'"
```

**Check step 20 validation:**
```bash
ssh root@90.90.102.18 "docker exec verl-r3-test bash -c 'grep \"step:20\" /tmp/mxfp4_w4a4_e13k_v2_aqn_qerl_sigma/training.log | grep \"val-core\"'"
```

**Check step 29 FINAL validation (CRITICAL):**
```bash
ssh root@90.90.102.18 "docker exec verl-r3-test bash -c 'grep \"step:29\" /tmp/mxfp4_w4a4_e13k_v2_aqn_qerl_sigma/training.log | grep \"val-core\"'"
```

**Extract final score:**
```bash
ssh root@90.90.102.18 "docker exec verl-r3-test bash -c 'grep \"step:29\" /tmp/mxfp4_w4a4_e13k_v2_aqn_qerl_sigma/training.log | grep -o \"val-core/openai/gsm8k/acc/mean@1:[^-]*\"'"
```

---

## Comparison Context

**E13j_v2 Unexpected Finding**: Step 29 (65.96%) was LOWER than step 20 (70.20%), suggesting:
1. Possible overfitting after step 20
2. Validation variance in trajectory sampling
3. AQN noise effects on later training stages

**E13k_v2 Hypothesis**: Lower sigma (0.01 vs 0.05) should provide:
- Less aggressive noise injection
- More stable training progression
- Potentially better step 29 performance

**Key Question**: Will E13k_v2 show the same step 29 < step 20 pattern, or will lower sigma change the dynamics?

---

## Next Steps

### After Step 29 Completes:

1. Extract final score and document
2. Compare with E13j_v2 to understand sigma impact:
   - Does lower sigma (0.01) improve step 29 performance?
   - Does it prevent the step 29 < step 20 degradation?
3. Update ALL_EXPERIMENTS_SUMMARY.md
4. **Launch next experiment**: E13l_v2 (Variable RIN)

---

## Container Info

**Container restarted**: 2026-01-17 10:25
**Zombie processes**: 2 (acceptable)
**Ray cluster**: Started at 10:25
**Experiment started**: 10:26

**Next experiment will require container restart** (CRITICAL RULE #1)

---

## FAILURE ANALYSIS - FIRST ATTEMPT

**What Happened**:
- Experiment started: 2026-01-17 10:26
- Step 0 validated successfully: 7.73% @ 10:37
- **Training HUNG after step 1 @ 10:35**
- Log file last modified: Jan 17 10:35 (119K size)
- Process still running but frozen for 16 hours
- Current time: 2026-01-18 02:36

**Impact**:
- ❌ No step 20 validation result
- ❌ No step 29 final validation result
- ❌ 16 hours × 8 GPUs = 128 GPU-hours wasted
- ❌ CRITICAL: This is the SECOND failed attempt (after E13j's bugs)

**Root Cause Investigation**:
- Zombie processes: Only 2 (same as after restart, not the cause)
- Log shows normal W4A4 quantization messages, then stops
- Progress bar stuck at "1/29 [04:54<2:17:38, 294.95s/it]"
- No error messages in log
- Process still alive (PID 1103) but not progressing

**This is the exact hang issue the user warned about!**
> "Sometime it will hang but after the test score provided. If no, it means we waste another 3 hours of 8 GPU."

In this case, it hung BEFORE any useful validation results, making it a complete failure.

---

## ATTEMPT 2 - CURRENT RUN

**Started**: 2026-01-18 02:39 (server time)
**Actions Taken**:
1. ✅ Killed hung process from attempt 1
2. ✅ Restarted container
3. ✅ Preserved failed logs → `/tmp/mxfp4_w4a4_e13k_v2_aqn_qerl_sigma_FAILED_ATTEMPT1/`
4. ✅ Re-launched experiment with fresh environment
5. ✅ Step 0 validated: 8.34% @ 02:52

**Results So Far (Attempt 2)**:
| Step | Result | Time | Notes |
|------|--------|------|-------|
| 0 | 8.34% | 02:52 | ✅ Confirmed (vs E13j_v2: 7.96%, attempt 1: 7.73%) |
| 20 | ⏳ Pending | ~04:15 | Expected ~85 minutes from start |
| 29 | ⏳ Pending | ~05:45 | **CRITICAL** - final validation |

**Monitoring Schedule**:
- ✅ Step 0: Validated @ 02:52
- [ ] Check @ 03:30: Verify step 5-10 progress
- [ ] Check @ 04:15: **CRITICAL** - Verify step 20 validation logged
- [ ] Check @ 05:00: Verify step 25+ progress
- [ ] Check @ 05:45: **CRITICAL** - Extract step 29 final score

**DO NOT ASSUME SUCCESS UNTIL STEP 29 SCORE IS CONFIRMED IN LOG!**
