# Gradient Noise vs Activation Noise: Theory and Verification Plan

**Date**: 2026-01-04
**Status**: Draft - Pending Verification
**Branch**: `feature/npu-aqn-test`

---

## 1. Background

### 1.1 Observed Phenomenon

Our AQN experiments showed unexpected results:

| Model | Training Improvement | Inference Robustness (5% noise) |
|-------|---------------------|--------------------------------|
| E5b (1.5B) | +2.42% | -14% degradation |
| E7c (7B) | +0.80% | 0% degradation |

**Key question**: Why does training with noise improve training stability but NOT inference robustness (for smaller models)?

### 1.2 Current Implementation Analysis

Our `noisy_ops.py` implementation injects noise in **BOTH** forward and backward passes:

```python
# From verl/utils/noisy_ops.py - NoisyMatMul class

class NoisyMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, other):
        result = torch.matmul(input, other)
        if _NOISY_OPS_ENABLED:
            error = _compute_error(result)  # Forward noise
            result = result + error
        return result

    @staticmethod
    def backward(ctx, grad_output):
        # ... gradient computation ...
        if _NOISY_OPS_ENABLED:
            error_a = _compute_error(grad_a)  # Backward noise (GRADIENT)
            error_b = _compute_error(grad_b)  # Backward noise (GRADIENT)
            grad_a = grad_a + error_a
            grad_b = grad_b + error_b
        return grad_a, grad_b
```

---

## 2. Proposed Theory

### 2.1 Core Hypothesis

**Training noise and inference noise require DIFFERENT skills:**

| Noise Type | When Applied | What It Affects | Skill Learned |
|------------|--------------|-----------------|---------------|
| **Forward noise** | Training & Inference | Activations | Handle noisy computations |
| **Backward noise** | Training only | Gradients | Optimize despite noisy signals |

### 2.2 Why Current AQN Doesn't Transfer to Inference Robustness

```
Current Training:
  Forward:  y = f(x) + noise_fwd     ← Activation noise
  Backward: grad = g(y) + noise_bwd  ← Gradient noise (DOMINANT effect)

  Model learns: "Find weights that converge despite noisy gradients"
  This is REGULARIZATION, not ROBUSTNESS LEARNING.

Inference Testing:
  Forward:  y = f(x) + noise_fwd     ← Activation noise only

  Model tested: "Can you compute correctly with noisy activations?"
  This is a DIFFERENT SKILL than what was learned!
```

### 2.3 Predicted Outcomes

If we train with **forward-only noise** (no backward noise):

| Scenario | Prediction | Reasoning |
|----------|------------|-----------|
| Training stability | Slightly worse | Less regularization from gradient noise |
| Inference robustness | Better | Training noise matches inference noise |

If we train with **backward-only noise** (no forward noise):

| Scenario | Prediction | Reasoning |
|----------|------------|-----------|
| Training stability | Similar to current | Gradient regularization preserved |
| Inference robustness | Poor | No activation noise during training |

---

## 3. Verification Experiment Design

### 3.1 Experiment Matrix

| Experiment | Forward Noise | Backward Noise | Expected Training | Expected Robustness |
|------------|---------------|----------------|-------------------|---------------------|
| **E8a** (baseline) | OFF | OFF | Baseline | Baseline |
| **E8b** (current AQN) | ON (5%) | ON (5%) | +2.42% | -14% @ 5% |
| **E8c** (forward-only) | ON (5%) | OFF | +1-2%? | **Better?** |
| **E8d** (backward-only) | OFF | ON (5%) | +2%? | Poor |

### 3.2 Implementation Changes Required

#### Option A: Environment Variable Control (Recommended)

Add environment variables to control noise injection phases:

```python
# In verl/utils/noisy_ops.py

_NOISY_OPS_FORWARD_ENABLED = True   # Default: inject in forward
_NOISY_OPS_BACKWARD_ENABLED = True  # Default: inject in backward

def set_noise_phases(forward=True, backward=True):
    global _NOISY_OPS_FORWARD_ENABLED, _NOISY_OPS_BACKWARD_ENABLED
    _NOISY_OPS_FORWARD_ENABLED = forward
    _NOISY_OPS_BACKWARD_ENABLED = backward

class NoisyMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, other):
        result = torch.matmul(input, other)
        if _NOISY_OPS_ENABLED and _NOISY_OPS_FORWARD_ENABLED:
            error = _compute_error(result)
            result = result + error
        return result

    @staticmethod
    def backward(ctx, grad_output):
        # ... gradient computation ...
        if _NOISY_OPS_ENABLED and _NOISY_OPS_BACKWARD_ENABLED:
            error_a = _compute_error(grad_a)
            error_b = _compute_error(grad_b)
            grad_a = grad_a + error_a
            grad_b = grad_b + error_b
        return grad_a, grad_b
```

#### Option B: Separate Noise Scales

```python
_NOISY_OPS_FORWARD_SCALE = 0.05
_NOISY_OPS_BACKWARD_SCALE = 0.05  # Set to 0 for forward-only
```

### 3.3 Training Configuration

**Model**: Qwen2.5-1.5B-Instruct (same as E5 series for comparison)
**Dataset**: GSM8K
**Hardware**: 8x A100
**Epochs**: 2
**Noise Scale**: 5% (relative Gaussian)

### 3.4 Evaluation Protocol

For each trained model:

1. **Clean accuracy** (0% noise inference)
2. **5% noise inference** (matching training noise level)
3. **10% noise inference** (stress test)

All evaluations using native PyTorch with injection count verification.

### 3.5 Success Criteria

**Theory is VERIFIED if:**

1. E8c (forward-only) shows **better** inference robustness than E8b (current AQN)
2. E8d (backward-only) shows **worse** inference robustness than E8c
3. E8b and E8d show similar training stability (gradient noise is the regularizer)

**Theory is FALSIFIED if:**

1. E8c shows similar or worse robustness than E8b
2. Forward noise has no effect on inference robustness

---

## 4. Implementation Plan

### Phase 1: Code Modification (1-2 hours)

1. Add `_NOISY_OPS_FORWARD_ENABLED` and `_NOISY_OPS_BACKWARD_ENABLED` flags
2. Add `set_noise_phases(forward, backward)` function
3. Add environment variable support: `VERL_NOISY_OPS_FORWARD_ONLY=1`
4. Update all NoisyXXX autograd functions (NoisyMatMul, NoisyBMM, NoisyLinear)

### Phase 2: Training Experiments (8-12 hours per experiment)

1. **E8a**: Baseline (no noise) - may reuse existing baseline
2. **E8b**: Current AQN (forward+backward) - may reuse E5b
3. **E8c**: Forward-only noise - NEW EXPERIMENT
4. **E8d**: Backward-only noise - NEW EXPERIMENT

### Phase 3: Robustness Evaluation (1-2 hours per model)

1. Native PyTorch evaluation at 0%, 5%, 10% noise
2. n=100 samples minimum for statistical significance
3. Injection count verification for all runs

### Phase 4: Analysis and Documentation

1. Compare results against predictions
2. Update theory document with verified/falsified status
3. If verified, merge into main HW_ERROR_INJECTION_EXPERIMENTS.md

---

## 5. Risk Assessment

### 5.1 Potential Issues

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Forward-only training is unstable | Medium | Monitor loss curves, reduce noise if needed |
| Results are inconclusive | Low | Increase sample size, run multiple seeds |
| Code changes break existing functionality | Low | Add unit tests, keep backward compatibility |

### 5.2 Resource Requirements

- **GPU hours**: ~40 hours (4 experiments x ~10 hours each)
- **Storage**: ~50GB for checkpoints
- **Time**: 2-3 days end-to-end

---

## 6. Alternative Hypotheses to Test

If the primary theory is falsified, consider:

### 6.1 Model Capacity Hypothesis

**Theory**: Larger models are robust due to redundant pathways, not training method.

**Test**: Train 3B model with same setup, test robustness.

### 6.2 Task-Specific Hypothesis

**Theory**: GSM8K (math) is naturally robust to noise due to discrete answers.

**Test**: Evaluate on continuous output tasks (dialogue, translation).

### 6.3 Noise Distribution Mismatch

**Theory**: Training noise distribution differs subtly from inference noise.

**Test**: Use identical RNG seeds for training and inference noise.

---

## 7. Expected Timeline

| Day | Task |
|-----|------|
| Day 1 | Code modifications + E8c training start |
| Day 2 | E8c training complete + E8d training start |
| Day 3 | E8d training complete + All evaluations |
| Day 4 | Analysis + Documentation update |

---

## 8. References

- [Noise Injection Node Regularization (OpenReview)](https://openreview.net/forum?id=gmSZ-GPNY6)
- [Training with Gaussian Noise (2024)](https://arxiv.org/abs/2405.18499)
- [Bayes-Optimized Noise Injection (Nature 2023)](https://www.nature.com/articles/s44172-023-00074-3)

---

## Appendix A: Code Diff Preview

```diff
# verl/utils/noisy_ops.py

+ _NOISY_OPS_FORWARD_ENABLED = True
+ _NOISY_OPS_BACKWARD_ENABLED = True

+ def set_noise_phases(forward: bool = True, backward: bool = True):
+     """Control which phases have noise injection."""
+     global _NOISY_OPS_FORWARD_ENABLED, _NOISY_OPS_BACKWARD_ENABLED
+     _NOISY_OPS_FORWARD_ENABLED = forward
+     _NOISY_OPS_BACKWARD_ENABLED = backward
+     logger.info(f"[NoisyOps] Phases: forward={forward}, backward={backward}")

class NoisyMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, other):
        result = torch.matmul(input, other)
-       if _NOISY_OPS_ENABLED:
+       if _NOISY_OPS_ENABLED and _NOISY_OPS_FORWARD_ENABLED:
            error = _compute_error(result)
            result = result + error
        return result

    @staticmethod
    def backward(ctx, grad_output):
        # ... gradient computation ...
-       if _NOISY_OPS_ENABLED:
+       if _NOISY_OPS_ENABLED and _NOISY_OPS_BACKWARD_ENABLED:
            error_a = _compute_error(grad_a)
            error_b = _compute_error(grad_b)
            grad_a = grad_a + error_a
            grad_b = grad_b + error_b
        return grad_a, grad_b
```

---

**Document Status**: APPROVED WITH CONDITIONS - Implementing

---

## 9. QA Review Results (2026-01-04)

### 9.1 Decision: **CONDITIONAL GO**

QA review identified 4 blockers that must be addressed:

| Blocker | Priority | Status |
|---------|----------|--------|
| No unit tests for phase flags | CRITICAL | Pending |
| Sample size too small (n=100) | CRITICAL | Will use n=200 |
| Thread safety not addressed | HIGH | Pending |
| Quantitative thresholds undefined | HIGH | Defined below |

### 9.2 Quantitative Success Criteria

| Criterion | Threshold | Statistical Test |
|-----------|-----------|------------------|
| "Better robustness" | E8c ≥ E8b + 5% at 5% noise | t-test p<0.05 |
| "Similar training" | \|E8c - E8b\| < 1% clean accuracy | Within variance |
| "Poor robustness" | E8d ≤ E8a + 2% at 5% noise | Worse than baseline |

### 9.3 Additional Experiments Recommended

| Experiment | Purpose | Priority |
|------------|---------|----------|
| E8e (2.5% noise) | Dose-response control | MEDIUM |
| E8f (forward-only, test@0%) | Clean performance impact | MEDIUM |
| Pre-test: gradient magnitude analysis | Validate theory before 40hr investment | HIGH |

### 9.4 Alternative Hypotheses to Monitor

1. **Noise scheduling mismatch**: Training uses 5%→0.1% decay, test uses fixed 5%
2. **Model capacity**: 7B robust regardless of noise type (1.5B lacks capacity)
3. **Task-specific**: GSM8K math sensitive to numerical errors

### 9.5 Updated Implementation Plan

**Day 0 (Today):**
- [x] Implement phase flags in noisy_ops.py ✅
- [x] Add thread safety locks ✅
- [x] Update all 6 operators (not just 3) ✅
- [x] Write unit tests ✅ (ALL TESTS PASSED)
- [ ] Run pre-test: gradient vs activation magnitude analysis

**Day 1-2:**
- [ ] E8c training (forward-only noise)

**Day 3-4:**
- [ ] E8d training (backward-only noise)

**Day 5:**
- [ ] Evaluations (n=200 each, 0%/5%/10% noise)
- [ ] Analysis and documentation

---

## 10. Unit Tests Required

```python
# tests/utils/test_noisy_ops_phases.py

def test_forward_only():
    """Verify backward injection count is 0 when backward disabled."""
    set_noise_phases(forward=True, backward=False)
    enable_noisy_ops(error_scale=0.05)
    # ... run forward + backward pass
    stats = get_injection_stats()
    assert stats['total_forward'] > 0
    assert stats['total_backward'] == 0

def test_backward_only():
    """Verify forward injection count is 0 when forward disabled."""
    set_noise_phases(forward=False, backward=True)
    enable_noisy_ops(error_scale=0.05)
    # ... run forward + backward pass
    stats = get_injection_stats()
    assert stats['total_forward'] == 0
    assert stats['total_backward'] > 0

def test_noise_magnitude():
    """Verify noise is approximately 5% of signal magnitude."""
    # ... measure noise ratio across multiple runs
    assert 0.03 < noise_ratio < 0.07

def test_phase_persistence():
    """Verify phase flags survive enable/disable cycles."""
    set_noise_phases(forward=True, backward=False)
    disable_noisy_ops()
    enable_noisy_ops(error_scale=0.05)
    # Phase flags should still be forward=True, backward=False
```

---

**Document Status**: APPROVED WITH CONDITIONS - Implementing
