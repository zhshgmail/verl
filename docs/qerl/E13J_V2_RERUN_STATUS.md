# E13j_v2 Re-Run Status

**Experiment**: E13j_v2 - Global AQN baseline with bug fixes
**Date**: 2026-01-17
**Server**: root@90.90.102.18, container: verl-r3-test
**Output**: `/tmp/mxfp4_w4a4_e13j_v2_global_aqn/training.log`

---

## Current Status (Last Update: 2026-01-17 02:07 server time)

**Progress**: 90% (26/29 steps completed)
**Step 29 ETA**: ~02:20 (in ~14 minutes from 02:07)

---

## Results So Far

| Step | Result | Type | Notes |
|------|--------|------|-------|
| 0 | 7.96% | Baseline | ✅ Confirmed |
| 20 | 70.20% | Validation | ✅ Confirmed (better than original 68.84%) |
| 29 | **⏳ PENDING** | **FINAL VALIDATION** | **CRITICAL - WATCH FOR THIS** |

---

## CRITICAL: Step 29 Monitoring Instructions

**Why This Matters:**
- Step 29 is the FIRST time we get true W4A4 final validation (bug fixed)
- Previous agent's skip validation bug caused step 29 to be SKIPPED
- Original E13j "step 29: 73.31%" was BF16 FAKE from separate eval

**What to Watch For:**

1. ✅ **Step 29 validation score appears in log**
   ```bash
   grep "step:29" /tmp/mxfp4_w4a4_e13j_v2_global_aqn/training.log | grep "val-core"
   ```

2. ⚠️ **May hang AFTER step 29 validation**
   - This is ACCEPTABLE if the validation score was logged
   - Score is what matters, post-validation hang is known issue

3. ❌ **If step 29 never appears = FAILURE**
   - Means experiment hung BEFORE validation
   - Wasted 3 hours of 8 × A100 GPUs

**Extract Score Command:**
```bash
ssh root@90.90.102.18 "docker exec verl-r3-test bash -c 'grep \"step:29\" /tmp/mxfp4_w4a4_e13j_v2_global_aqn/training.log | grep -o \"val-core/openai/gsm8k/acc/mean@1:[^-]*\"'"
```

---

## Comparison with Original E13j

| Metric | Original E13j (INVALID) | E13j_v2 (CORRECTED) | Status |
|--------|------------------------|---------------------|--------|
| Step 0 | 8.04% | 7.96% | ✅ Similar baseline |
| Step 20 | 68.84% (W4A4) | **70.20%** (W4A4) | ✅ +1.36% improvement |
| Step 29 | SKIPPED (bug) | **⏳ RUNNING NOW** | Waiting... |
| "Step 29" | 73.31% (BF16 FAKE!) | N/A | Invalid |

**Expected E13j_v2 Step 29**: ~78-80% (based on E13g/E13h +10% pattern)

---

## Next Steps After Step 29 Completes

### If Step 29 Success (score logged):
1. Extract final score and document
2. Compare with original E13j (68.84% @ step 20 vs v2 @ step 29)
3. Update ALL_EXPERIMENTS_SUMMARY.md with corrected results
4. **Launch next experiment**: E13k_v2 (High priority)

### If Step 29 Fails (hangs before validation):
1. Document failure in this file
2. Investigate cause (check for zombie processes, memory issues)
3. Restart container
4. Re-run E13j_v2 again with fresh environment

---

## Running Experiments Queue

After E13j_v2 completes:

1. **E13k_v2** (High) - QeRL sigma test
2. **E13l_v2** (High) - Variable RIN
3. **E13m_v2** (High) - Inverse RIN
4. **E13n_v2** (High) - Ceiling RIN
5. **E14a_v2** (Medium) - Zone-based scheduling

**Total runtime**: ~8 hours for all 6 experiments

---

## Container Info

**Must restart container before next experiment** (CRITICAL RULE #1):
```bash
ssh root@90.90.102.18
docker restart verl-r3-test
sleep 15
docker exec verl-r3-test bash -c 'ps aux | grep defunct | wc -l'  # Check zombies
docker exec -it verl-r3-test bash
cd /home/z00637938/workspace/verl
ray start --head --port=6379 --num-gpus=8 --disable-usage-stats
bash scripts/test_mxfp4_w4a4_e13k_v2_aqn_qerl_sigma.sh 8
```

---

## Status: WAITING FOR STEP 29 VALIDATION

**Current time**: 02:07 (server time)
**Expected completion**: 02:20
**Monitoring**: Every 5 minutes

**DO NOT ASSUME SUCCESS UNTIL STEP 29 SCORE IS CONFIRMED IN LOG!**
