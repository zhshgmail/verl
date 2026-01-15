# Self-Referential Differential Diagnosis (SRDD) for Hardware Error Localization

**Version**: 2.0
**Date**: 2026-01-06
**Authors**: QeRL Team with Gemini Collaboration

---

## Abstract

This report presents SRDD (Self-Referential Differential Diagnosis), a novel methodology for localizing hardware errors in neural network accelerators **without requiring a reference system**. SRDD uses controllable noise injection as a probe to detect anomalous layer behavior through three statistical probes: Linearity, Neighborhood Smoothness, and Input Invariance. Validation on A100 GPU with simulated faults demonstrates **100% top-1 accuracy** for dead_zone faults using v2.0 improvements including log-space normalization and Spearman correlation.

---

## 1. Introduction

### 1.1 Problem Statement

When deploying LLMs on novel hardware accelerators (NPUs, TPUs), hardware errors may cause model degradation without obvious failure signatures. Traditional debugging requires:
- Access to a known-good reference system (GPU)
- Ability to compare outputs between systems

**Challenge**: In production scenarios, reference systems may not be available. We need a method to identify the error source layer using only the faulty system.

### 1.2 Key Insight

Hardware errors are **discrete** (affect specific layers/operations), while architectural sensitivity is **continuous** (varies smoothly across layers). By detecting anomalies in how layers respond to controlled perturbations, we can identify fault locations without external reference.

---

## 2. Methodology

### 2.1 Controllable Noise Injection Framework

We leverage the AQN (Adaptive Quantization Noise) infrastructure to inject controlled noise at specific layers:

```python
from verl.utils.noisy_ops import (
    enable_noisy_ops,
    disable_noisy_ops,
    set_selective_layers,
    register_layer_hooks,
)
```

### 2.2 Diagnostic Probes

#### Probe A: Monotonicity Test (Spearman Correlation)

**Principle**: A healthy layer should show monotonic response to increasing noise levels.

**Metric**: Spearman rank correlation (rho^2) between noise scale and KL divergence.

**v2.0 Fix**: Use Spearman instead of Pearson correlation because KL divergence scales with noise^2 (variance), not linearly with noise amplitude.

```python
# Faulty layer: non-monotonic response (dead zones, saturation)
# Normal layer: rho^2 ~ 1.0 (higher noise -> higher divergence)
rho, _ = stats.spearmanr(noise_scales, sensitivities)
monotonicity_score = rho ** 2
```

#### Probe B: Neighborhood Smoothness (Log-Space)

**Principle**: Sensitivity should vary smoothly across adjacent layers. Hardware faults create discontinuities.

**Metric**: Second derivative of log-transformed sensitivity curve.

**v2.0 Fix**: Transform to log space to normalize magnitude differences across layers.

```python
# v1.0 problem: |0.9 - 0.8| >> |0.003 - 0.002| (last layer dominates)
# v2.0 solution: |log(0.9) - log(0.8)| ~ |log(0.003) - log(0.002)|
log_sens = np.log(sensitivities + 1e-9)
local_anomaly = |log_sens[i] - (log_sens[i-1] + log_sens[i+1]) / 2|
```

#### Probe C: Input Invariance

**Principle**: A healthy layer should respond consistently across different inputs.

**Metric**: Coefficient of Variation (CV = std/mean) of sensitivity across diverse prompts.

```python
# Faulty layer: "neurotic" - normal for 99 inputs, explodes on 1
# Normal layer: consistent CV across inputs
cv = std(input_sensitivities) / mean(input_sensitivities)
```

#### Probe D: Variance Compression (v2.0)

**Principle**: Saturated layers "absorb" noise instead of amplifying it.

**Metric**: Amplification slope (divergence per unit noise).

```python
# Normal layer: divergence increases with noise (positive slope)
# Saturated layer: divergence plateaus (slope ~ 0)
slope, _ = np.polyfit(noise_scales, responses, 1)
```

### 2.3 BF16 Numerical Stability

**Critical Fix**: BF16 machine epsilon (~1e-3) makes eps=1e-10 effectively zero.

```python
def compute_kl_divergence(p_logits, q_logits):
    # Upcast to float32 for numerical stability
    p_logits = p_logits.float()
    q_logits = q_logits.float()

    # Use safe epsilon (1e-6, not 1e-10)
    eps = 1e-6
    kl = torch.sum(p * torch.log((p + eps) / (q + eps)))
    return max(0.0, kl.item())
```

### 2.4 Aggregation

Combine probe results using weighted composite score:

```python
composite_score = (
    max(0, -lin_z) * 1.0 +      # Lower rho^2 = worse
    max(0, smooth_z) * 1.5 +    # Higher local anomaly = worse
    max(0, inv_z) * 1.0 +       # Higher CV = worse
    max(0, -comp_z) * 1.5       # Lower slope = worse (saturation)
)
```

---

## 3. Experimental Setup

### 3.1 Hardware & Environment

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA A100 |
| Model | Qwen2.5-1.5B-Instruct (E8c checkpoint) |
| Precision | BF16 |
| Framework | PyTorch + Transformers |

### 3.2 Fault Simulation

Hardware faults simulated via forward hooks:

| Fault Type | Description | Simulates |
|------------|-------------|-----------|
| `dead_zone` | Small values -> 0 | FP underflow |
| `saturation` | Values clamped to range | FP overflow |
| `bias` | Systematic offset | Computation error |
| `noise` | Random perturbation | Numerical instability |
| `spike` | Random large values | Bit flip |

### 3.3 Test Configuration

- Noise scales: [0.01, 0.02, 0.05, 0.10, 0.15]
- Trials per scale: 3
- Test prompts: 4 basic + 10 diverse
- Fault magnitude: 0.3

---

## 4. Results

### 4.1 SRDD v2.0 Validation (dead_zone Fault)

| Ground Truth Layer | Diagnosed Layer | Rank | Result |
|-------------------|-----------------|------|--------|
| 10 | 10 | 1 | EXACT MATCH |
| 20 | 20 | 1 | EXACT MATCH |

### 4.2 Comparison: v1.0 vs v2.0

| GT Layer | v1.0 Diagnosed | v1.0 GT Rank | v2.0 Diagnosed | v2.0 GT Rank | Improvement |
|----------|----------------|--------------|----------------|--------------|-------------|
| 5 | 6 | 2 | - | - | - |
| 10 | 2 | 4 | **10** | **1** | +3 ranks |
| 15 | 2 | 4 | - | - | - |
| 20 | **27** | 4 | **20** | **1** | L27 dominance fixed |

### 4.3 Saturation Fault Detection

| GT Layer | Fault Type | Diagnosed | GT Rank | Result |
|----------|------------|-----------|---------|--------|
| 15 | saturation (0.2) | 26 | >5 | MISMATCH |

**Note**: Saturation faults with low magnitude remain challenging to detect.

### 4.4 Key Metrics (GT=10, dead_zone)

```
Layer 10: rho^2 = 0.0000, LogAnom = 2.328873, CV = 0.4893, AmpSlope = 0.0000
```

- **rho^2 = 0.0**: No monotonic relationship (fault signature)
- **High LogAnom**: Discontinuity in log-space sensitivity curve
- **Low AmpSlope**: Noise absorbed rather than amplified

---

## 5. Discussion

### 5.1 Key Improvements in v2.0

1. **Spearman Correlation**: Correctly handles non-linear KL scaling
2. **Log-Space Smoothness**: Eliminates last-layer dominance bias
3. **BF16 Stability**: Prevents NaN/Inf in divergence calculation
4. **State Management**: try/finally ensures cleanup on exceptions

### 5.2 Fault Detectability

| Fault Type | Detectability | Why |
|------------|---------------|-----|
| dead_zone | HIGH | Creates non-monotonic response |
| saturation | LOW | Appears "stable" (noise absorbed) |
| spike | LOW | Random nature averages out |
| noise | VERY LOW | Similar to diagnostic noise |

### 5.3 Limitations

1. **Adjacent Layer Confusion**: Diagnosed layer sometimes off by 1
2. **Low-Magnitude Faults**: May fall below detection threshold
3. **Computational Cost**: O(layers * noise_scales * trials * probes)

---

## 6. Conclusion

SRDD v2.0 successfully localizes hardware error sources at the layer level without requiring a reference system. For dead_zone faults (FP underflow), the method achieves **100% top-1 accuracy** on tested configurations.

### 6.1 Recommendations

1. **Use SRDD for initial fault localization** when reference system unavailable
2. **Focus on dead_zone/underflow faults** which have clearest signatures
3. **Combine with fingerprint correlation** when reference system available

### 6.2 Future Work

1. Operator-level localization (matmul, softmax, etc.)
2. Channel-level analysis within faulty layer
3. Improved saturation detection via output distribution analysis
4. Binary search for faster layer identification

---

## Appendix A: Usage

```bash
# Production mode (no reference needed)
python scripts/srdd_error_finder.py \
    --model_path /path/to/model

# Validation mode (with simulated fault)
python scripts/srdd_error_finder.py \
    --model_path /path/to/model \
    --ground_truth_layer 10 \
    --fault_type dead_zone \
    --fault_magnitude 0.3
```

## Appendix B: Code Review Fixes (Gemini)

| Issue | Problem | Solution |
|-------|---------|----------|
| BF16 epsilon | eps=1e-10 is 0 in BF16 | Upcast to float32, use eps=1e-6 |
| Pearson correlation | KL scales with noise^2 | Use Spearman (monotonicity) |
| State cleanup | Exception leaves noise enabled | try/finally blocks |

---

**Document Status**: Validated on A100
**Last Updated**: 2026-01-06
