# QeRL-Style Experiments: E6b-qerl & E12-qerl

**Date**: 2026-01-12
**Branch**: `feature/npu-aqn-test`
**Goal**: Test if QeRL's small-batch/many-steps configuration improves AQN effectiveness

---

## Background

Our previous AQN experiments showed marginal improvement (+0.31% for E6b). Analysis of QeRL source code revealed a critical configuration difference:

| Parameter | QeRL | Our Previous | This Experiment |
|-----------|------|--------------|-----------------|
| **Batch Size** | 2 | 256 | **2** |
| **Steps/Epoch** | ~3,736 | ~29 | **~3,736** |
| **Epochs** | 1 | 2 | **1** |
| **Total Steps** | ~3,736 | ~58 | **~3,736** |
| **Test Freq** | 50 (save) | 10 | **200** |
| **Epoch-Aware** | No | Mixed | **No** (step-based) |

**Hypothesis**: AQN's exploration benefit accumulates over many training steps. With 60x more steps, we should see larger AQN improvement.

---

## Experiment Plan

### E6b-qerl: Standard Sigma AQN

| Setting | Value |
|---------|-------|
| Exp ID | E6b-qerl |
| TensorBoard Name | `LoRA_MXFP4_DAPO_qerl_AQN_std` |
| Quantization | MXFP4 (~21% error) |
| LoRA | rank=32, alpha=16 |
| AQN σ | 0.01 → 0.0001 (10 stages, step-based) |
| Batch Size | gen=2, train=2 |
| Epochs | 1 |
| Expected Steps | ~3,736 |
| Test Freq | 200 (~18 evals + final) |

### E12-qerl: High Sigma AQN

| Setting | Value |
|---------|-------|
| Exp ID | E12-qerl |
| TensorBoard Name | `LoRA_MXFP4_DAPO_qerl_AQN_high` |
| Quantization | MXFP4 (~21% error) |
| LoRA | rank=32, alpha=16 |
| AQN σ | 0.05 → 0.0005 (10 stages, step-based) |
| Batch Size | gen=2, train=2 |
| Epochs | 1 |
| Expected Steps | ~3,736 |
| Test Freq | 200 (~18 evals + final) |

---

## Key Changes from Previous Experiments

1. **Batch Size**: 256 → 2 (matches QeRL exactly)
2. **Epochs**: 2 → 1 (matches QeRL, avoids epoch_aware complexity)
3. **Test Freq**: 10 → 200 (reduces eval overhead for 3700+ steps)
4. **Epoch-Aware**: Disabled (use QeRL's step-based K-stage decay)
5. **Final Eval**: Guaranteed by recent fix to `dapo_ray_trainer.py`

---

## Expected Results

| Experiment | Previous Result | Expected with QeRL Config |
|------------|-----------------|---------------------------|
| E6b (std σ) | 73.24% (2ep, 58 steps) | Higher if AQN helps |
| E12 (high σ) | 72.93% (2ep, 58 steps) | Higher if high σ helps |

**Success Criteria**:
- If E6b-qerl > 73.24%: QeRL config improves AQN effectiveness
- If E12-qerl > E6b-qerl: High sigma beneficial with more steps
- If both ≈ previous: Step count alone doesn't explain QeRL's results

---

## Execution Plan

1. Restart A100 container (kill zombies)
2. Run E6b-qerl first (standard sigma baseline)
3. Run E12-qerl second (high sigma comparison)
4. Monitor via TensorBoard
5. Update ALL_EXPERIMENTS_SUMMARY.md with results

---

## Scripts

- `scripts/test_mxfp4_qerl_e6b_std_sigma.sh` - Standard sigma AQN
- `scripts/test_mxfp4_qerl_e12_high_sigma.sh` - High sigma AQN

---

## Notes

- With batch_size=2, each step processes 2 samples
- 7,473 training samples / 2 = 3,736 steps per epoch
- Test freq 200 = 18-19 evaluations (3736/200)
- Final step evaluation guaranteed by `dapo_ray_trainer.py` fix
- No epoch_aware = sigma decays linearly across all 3736 steps
