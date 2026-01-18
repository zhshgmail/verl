# E13/E14 v2 Re-Runs Summary

**Purpose**: Get TRUE step 29 W4A4 validation results after fixing the skip validation bug

**Bug Fixed**: Previous agent's code skipped final step validation, only reporting step 20 results

---

## Completed Experiments

| Exp ID | Config | Step 29 Result | Training Time | Status | Notes |
|--------|--------|----------------|---------------|--------|-------|
| E13j_v2 | Global AQN (σ=0.05→0.0005) | **65.96%** | ~2h 22min | ✅ Complete | Lower than step 20 (70.20%) |
| E13k_v2 | Global AQN (σ=0.01→0.0001) | **67.55%** | ~2h 23min | ✅ Complete | Lower than step 20 (69.45%), Attempt 1 hung |
| E13l_v2 | Variable RIN | ⏳ Pending | ⏳ Running | 🏃 In Progress | Started 2026-01-17 17:54 |
| E13m_v2 | Inverse RIN | ⏳ Pending | - | ⏸️ Queued | - |
| E13n_v2 | Ceiling RIN | ⏳ Pending | - | ⏸️ Queued | - |
| E14a_v2 | Zone-based schedule | ⏳ Pending | - | ⏸️ Queued | - |

---

## Detailed Results

### E13j_v2 - Global AQN (sigma 0.05)

**Configuration**:
- AQN sigma: 0.05 → 0.0005 (QeRL code default)
- Target layers: RMSNorm
- 10 decay stages

**Results**:
- Step 0: 7.96% (baseline)
- Step 20: 70.20%
- **Step 29: 65.96%** ← FINAL

**Training Time**: ~2h 22min (2:17:33 for 28 steps + ~5min for step 29)

**Key Finding**: Performance DEGRADED by 4.24% from step 20 to step 29

---

### E13k_v2 - Global AQN (sigma 0.01)

**Configuration**:
- AQN sigma: 0.01 → 0.0001 (QeRL training scripts actual values)
- Target layers: RMSNorm
- 10 decay stages

**Results**:
- Step 0: 8.34% (baseline)
- Step 20: 69.45%
- **Step 29: 67.55%** ← FINAL

**Training Time**: ~2h 23min (2:18:08 for 28 steps + ~5min for step 29)

**Issues**: Attempt 1 hung after step 1 (wasted 16 hours), had to restart and re-run

**Key Finding**: Performance DEGRADED by 1.90% from step 20 to step 29 (less severe than E13j_v2)

---

### E13l_v2 - Variable RIN

**Configuration**:
- Variable RIN: High reward → MORE noise (opposite of typical)
- Target layers: RMSNorm

**Status**: Running (started 2026-01-17 17:54)

**Results**: Pending

---

### E13m_v2 - Inverse RIN

**Configuration**:
- Inverse RIN: High reward → LESS noise
- Target layers: RMSNorm

**Status**: Queued

**Results**: Pending

---

### E13n_v2 - Ceiling RIN

**Configuration**:
- Ceiling RIN: Noise limited by ceiling threshold
- Target layers: RMSNorm

**Status**: Queued

**Results**: Pending

---

### E14a_v2 - Zone-based Schedule

**Configuration**:
- Zone-based noise scheduling
- Target layers: RMSNorm

**Status**: Queued

**Results**: Pending

---

## Critical Findings So Far

### 1. Step 29 < Step 20 Degradation Pattern

Both completed experiments show **performance degradation** from step 20 to step 29:

| Experiment | Sigma | Step 20 | Step 29 | Degradation |
|------------|-------|---------|---------|-------------|
| E13j_v2 | 0.05 | 70.20% | 65.96% | **-4.24%** |
| E13k_v2 | 0.01 | 69.45% | 67.55% | **-1.90%** |

**Lower sigma (0.01) reduces degradation** but doesn't eliminate it.

### 2. Comparison with No-AQN Baseline

- **E13h baseline (no AQN)**: 71.42% @ step 29
- **E13j_v2 (σ=0.05)**: 65.96% @ step 29 → **-5.46% vs baseline**
- **E13k_v2 (σ=0.01)**: 67.55% @ step 29 → **-3.87% vs baseline**

**Both AQN experiments WORSE than no-AQN baseline!**

### 3. Training Time

All experiments take **~2h 20min** on 8 × A100 GPUs:
- E13j_v2: 2:22
- E13k_v2: 2:23 (attempt 2)
- Expected for remaining: ~2:20 each

**Total estimated time for all 6 experiments**: ~14 hours (if no hangs)

---

## Next Steps

1. ✅ E13j_v2 complete
2. ✅ E13k_v2 complete
3. 🏃 E13l_v2 running (ETA: ~2h from start)
4. ⏸️ E13m_v2 queued
5. ⏸️ E13n_v2 queued
6. ⏸️ E14a_v2 queued

**After all complete**: Analyze why AQN/RIN approaches underperform baseline
