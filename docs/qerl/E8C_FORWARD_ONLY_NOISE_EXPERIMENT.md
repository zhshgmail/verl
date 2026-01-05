# E8c: Forward-Only Noise Experiment Record

**Date**: 2026-01-05
**Status**: V2 Training In Progress (with checkpoints)
**Branch**: `feature/npu-aqn-test`

---

## 1. Experiment Overview

### 1.1 Purpose

Test the theory that **forward noise (activations)** and **backward noise (gradients)** have different effects:
- Forward noise → Affects inference robustness
- Backward noise → Provides training regularization

### 1.2 Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen2.5-1.5B-Instruct |
| Dataset | GSM8K |
| Hardware | 8x A100 |
| Noise type | Relative Gaussian |
| Noise scale | 5% |
| Forward noise | **Enabled** |
| Backward noise | **Disabled** |
| AQN | Epoch-aware scheduling |
| Total epochs | 2 |

### 1.3 Environment Variables

```bash
VERL_NOISY_OPS_ENABLED=1
VERL_NOISY_OPS_SCALE=5e-2
VERL_NOISY_OPS_TYPE=relative_gaussian
VERL_NOISY_OPS_FORWARD_ONLY=1  # Key: only forward pass noise
```

---

## 2. E8c V1 Results (No Checkpoint)

**Run date**: 2026-01-04
**Duration**: ~1h47m

### 2.1 Validation Accuracy Progression

| Step | Accuracy |
|------|----------|
| 0 | 9.1% |
| 20 | 63.1% |
| 40 | 65.0% |
| 60 | 66.3% |
| 80 | **70.5%** (peak) |
| 100 | 69.2% |
| 116 | 69.4% (final) |

### 2.2 Comparison with E5b

| Metric | E5b (both noise) | E8c v1 (forward-only) | Delta |
|--------|------------------|----------------------|-------|
| Clean accuracy | ~78% | 69.4% | **-8.6%** |
| Peak accuracy | ~80% | 70.5% | **-9.5%** |
| Checkpoint | Available | Not saved | |

### 2.3 V1 Conclusion

**Partial theory validation**:
- **Backward noise helps training**: -8.6% accuracy drop confirms gradient noise provides regularization benefit
- **Robustness evaluation blocked**: No checkpoint saved (`save_freq=-1`)

---

## 3. E8c V2 (With Checkpoints)

**Started**: 2026-01-05 05:56 UTC
**Commit**: `5b2f21e5`
**Log file**: `/tmp/e8c_forward_v2.log`

### 3.1 Changes from V1

| Parameter | V1 | V2 |
|-----------|----|----|
| `save_freq` | -1 | 58 |
| Checkpoint | Not saved | Saves at epoch end |

### 3.2 Expected Outputs

1. **Checkpoints**:
   - `checkpoints/e8c_forward_only_5e-2/epoch_0/`
   - `checkpoints/e8c_forward_only_5e-2/epoch_1/`

2. **Training metrics** (expected similar to V1):
   - Clean accuracy: ~69-70%
   - Peak accuracy: ~70-71%

### 3.3 Monitoring Commands

```bash
# Check training progress
ssh root@90.90.102.18 "tmux capture-pane -t e8c_forward -p | tail -30"

# Check log file
ssh root@90.90.102.18 "docker exec verl-r3-test tail -50 /tmp/e8c_forward_v2.log"

# Check validation accuracy
ssh root@90.90.102.18 "docker exec verl-r3-test grep 'val-core' /tmp/e8c_forward_v2.log"
```

---

## 4. Post-Training Evaluation Plan

After V2 training completes:

### 4.1 Robustness Evaluation

```bash
# Evaluate at 0%, 5%, 10% noise levels
python scripts/eval_checkpoint_robustness.py \
    --checkpoint checkpoints/e8c_forward_only_5e-2/epoch_1/ \
    --noise-levels 0.0 0.05 0.10 \
    --n-samples 100
```

### 4.2 Success Criteria

**Theory CONFIRMED if**:
- E8c shows better % retention at noisy inference than E5b
- i.e., `(E8c @ 5%) / (E8c @ 0%) > (E5b @ 5%) / (E5b @ 0%)`

| Metric | E5b (both) | E8c Target | Interpretation |
|--------|------------|------------|----------------|
| Clean (0%) | 78% | ~70% | Expected lower (less regularization) |
| Robustness @ 5% | 64% | **>70%** | If higher = theory supported |
| Retention rate | 82% | **>90%?** | Better retention = forward noise helps |

### 4.3 Comparison Matrix

| Experiment | Forward Noise | Backward Noise | Clean Acc | Robustness (5%) |
|------------|---------------|----------------|-----------|-----------------|
| E5 (baseline) | OFF | OFF | 68.16% | TBD |
| E5b (both) | ON | ON | ~78% | 64% |
| **E8c (forward)** | ON | **OFF** | ~70% | **TBD** |
| E8d (backward) | OFF | ON | TBD | TBD |

---

## 5. Theory Implications

### 5.1 If E8c Shows Better Robustness

**Conclusion**: Forward noise is the key to inference robustness
- Training with forward-only noise teaches model to handle noisy activations
- Backward noise is unnecessary for robustness (just regularization)

**Implication for hardware deployment**:
- Use forward-only AQN for NPU deployment preparation
- Can use full precision for gradients, quantized for forward pass

### 5.2 If E8c Shows Similar/Worse Robustness

**Conclusion**: Forward noise alone is insufficient
- May need combined approach for robustness
- Or robustness requires different mechanism entirely

---

## 6. Related Experiments

| Experiment | Description | Status |
|------------|-------------|--------|
| E5 | Noise only (no AQN) | Completed |
| E5a | Noise + global AQN | Completed |
| E5b | Noise + epoch-aware AQN | Completed |
| E6 | All-ops mode | Completed |
| E7c | 7B model verification | Completed |
| **E8c** | Forward-only noise | **V2 In Progress** |
| E8d | Backward-only noise | Planned |

---

## 7. Commits

| Commit | Description |
|--------|-------------|
| `d6643add` | Add forward/backward phase control to noisy_ops |
| `4b21a970` | Add env var support for FORWARD_ONLY mode |
| `ec12c783` | Enable checkpoint saving in E8c script |
| `5b2f21e5` | Add selective layer injection API |

---

**Last updated**: 2026-01-05 06:00 UTC
