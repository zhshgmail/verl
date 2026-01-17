# E13k_v2 Re-Run Status

**Experiment**: E13k_v2 - Global AQN with QeRL sigma values
**Date**: 2026-01-17
**Server**: root@90.90.102.18, container: verl-r3-test
**Output**: `/tmp/mxfp4_w4a4_e13k_v2_aqn_qerl_sigma/training.log`

---

## Current Status (Last Update: 2026-01-17 10:37 server time)

**Progress**: ⏳ **RUNNING** (1/29 steps)
**Step 0**: ✅ **VALIDATED** - 7.73%
**Status**: Baseline validated, training in progress

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

## Status: TRAINING IN PROGRESS

**Step 0 completed**: 10:37 (server time) - 7.73% baseline ✅
**Next checkpoint**: Step 20 @ ~11:50
**Final validation**: Step 29 @ ~13:30 (CRITICAL)

**DO NOT ASSUME SUCCESS UNTIL STEP 29 SCORE IS CONFIRMED IN LOG!**
