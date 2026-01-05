# AQN as Diagnostic Probe: Layer-wise Noise Sensitivity Analysis

**Date**: 2026-01-05
**Status**: Research Complete - Implementation Pending
**Branch**: `feature/npu-aqn-test`

---

## Executive Summary

This document outlines how to use Adaptive Quantization Noise (AQN) as a **self-diagnostic mechanism** to identify where major system noise comes from during training and inference. The approach is:

1. **Theoretically Sound**: Based on information bottleneck theory and differential sensitivity analysis
2. **Empirically Validated**: E5b/E7c results show dramatic differences in layer robustness
3. **Practically Implementable**: Extends existing `noisy_ops.py` with selective injection

---

## 1. The Core Insight

### 1.1 What E5b/E7c Results Tell Us

| Model | Training Improvement | Inference Robustness (5% noise) |
|-------|---------------------|--------------------------------|
| **1.5B (E5b)** | +2.42% | **-14%** (brittle) |
| **7B (E7c)** | +0.80% | **0%** (robust!) |

**Key observation**: The 7B model has **architectural redundancy** that buffers noise. This means:
- Different layers contribute differently to robustness
- Noise probing can identify which layers are critical

### 1.2 The Diagnostic Principle

```
If you inject noise in layer X and accuracy drops significantly:
  → Layer X is a "bottleneck" for noise propagation
  → Hardware errors in layer X will cause problems

If accuracy stays stable:
  → Layer X has redundancy or error-correction capabilities
  → Safe to use lower precision hardware for this layer
```

---

## 2. Implementation: Selective Layer Noise Injection

### 2.1 API Design

```python
# verl/utils/noisy_ops.py - New API

# Configure which layers receive noise
set_selective_layers(layer_ids=[0, 5, 10])  # Only inject in layers 0, 5, 10
set_selective_layers(None)                   # Inject in all layers (default)

# Configure which operations receive noise
set_selective_ops(['matmul', 'softmax'])     # Only matmul and softmax
set_selective_ops(None)                       # All ops (default)

# Get per-layer injection stats
get_layer_injection_stats()  # Returns {layer_id: {forward: N, backward: M}, ...}
```

### 2.2 How Layer Detection Works

```python
# For transformer models, layer ID is extracted from module name:
# 'model.layers.12.self_attn.q_proj' → layer_id = 12
# 'model.layers.5.mlp.down_proj' → layer_id = 5

# Implementation uses context variables set during forward pass
_CURRENT_LAYER_ID = None

def register_layer_hooks(model):
    """Register hooks to track current layer during forward pass."""
    for name, module in model.named_modules():
        if '.layers.' in name:
            layer_id = int(name.split('.layers.')[1].split('.')[0])
            module.register_forward_pre_hook(
                lambda m, inp, lid=layer_id: set_current_layer(lid)
            )
```

### 2.3 Concrete Implementation

```python
# In NoisyMatMul.forward:
@staticmethod
def forward(ctx, a, b):
    result = _ORIGINAL_MATMUL(a, b)

    if _NOISY_OPS_ENABLED and _NOISY_OPS_FORWARD_ENABLED:
        layer_id = get_current_layer()

        # Check if we should inject for this layer
        if should_inject_for_layer(layer_id):
            error = _compute_error(result)
            result = result + error

            # Track per-layer stats
            _LAYER_INJECTION_STATS[layer_id]['forward'] += 1

    ctx.save_for_backward(a, b)
    return result

def should_inject_for_layer(layer_id):
    """Check if noise should be injected for this layer."""
    if _SELECTIVE_LAYERS is None:
        return True  # All layers
    return layer_id in _SELECTIVE_LAYERS
```

---

## 3. Diagnostic Protocol

### 3.1 Layer Sensitivity Profiling

**Goal**: Identify which layers are most sensitive to noise

**Protocol**:
```
For each layer L in [0, num_layers):
    For each noise_level in [1%, 5%, 10%]:
        1. Enable noise ONLY for layer L (forward-only)
        2. Run inference on test set (n=100 samples)
        3. Record accuracy
        4. Calculate degradation = (baseline - accuracy) / baseline

Output: Sensitivity heatmap [layer x noise_level]
```

**Interpretation**:
- High degradation (>10%) = **Critical layer** - needs robust hardware
- Low degradation (<2%) = **Robust layer** - can tolerate lower precision

### 3.2 Hardware Error Localization

**Goal**: Find where GPU→NPU divergence causes problems

**Protocol**:
```
1. Run same model on GPU and NPU
2. Capture activations at each layer boundary
3. Compute divergence metrics:
   - MAE (mean absolute error)
   - MAPE (mean absolute percentage error)
   - Correlation coefficient
4. Cross-reference with sensitivity profile

High divergence + High sensitivity = CRITICAL BOTTLENECK
```

### 3.3 Operation-Type Profiling

**Goal**: Identify which operations (matmul, softmax, layernorm) are most sensitive

**Protocol**:
```
For each operation type in [matmul, softmax, silu, layernorm]:
    1. Enable noise ONLY for this operation type
    2. Run inference
    3. Record degradation

Output: Operation sensitivity ranking
```

---

## 4. Key Metrics

### 4.1 Sensitivity Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Relative Degradation** | `(acc_clean - acc_noisy) / acc_clean` | % accuracy loss |
| **Noise Tolerance Index** | `1 / relative_degradation` | Higher = more robust |
| **Critical Threshold** | Min noise causing >10% drop | Brittleness measure |

### 4.2 Per-Layer Metrics

```python
{
    'snr': activation.abs().mean() / noise_scale,  # Signal-to-noise ratio
    'sparsity': (activation.abs() < 1e-6).float().mean(),  # Sparsity
    'dynamic_range': activation.max() - activation.min(),  # Value range
    'grad_magnitude': activation.grad.abs().mean(),  # Gradient size
}
```

### 4.3 Hardware Divergence Metrics

```python
{
    'mae': torch.abs(gpu - npu).mean(),
    'mape': (torch.abs(gpu - npu) / (gpu.abs() + 1e-8)).mean(),
    'correlation': torch.corrcoef(gpu.flat, npu.flat)[0,1],
    'max_error': torch.abs(gpu - npu).max(),
}
```

---

## 5. Expected Results and Applications

### 5.1 Typical Sensitivity Pattern (Hypothesis)

Based on transformer architecture:

| Layer Region | Expected Sensitivity | Reasoning |
|--------------|---------------------|-----------|
| Early layers (0-5) | Medium | Feature extraction, some redundancy |
| Middle layers (6-20) | Low | High redundancy, learned abstractions |
| Late layers (21-27) | HIGH | Output formation, less redundancy |
| Attention (all) | HIGH | Softmax is numerically sensitive |
| FFN (all) | Medium | ReLU/SiLU provides some robustness |

### 5.2 Applications

#### A. Adaptive AQN Training

```python
# Apply stronger noise to robust layers, weaker to brittle layers
def get_layer_noise_scale(layer_id, base_scale=0.05):
    sensitivity = SENSITIVITY_PROFILE[layer_id]
    if sensitivity > 0.10:  # Critical layer
        return base_scale * 0.5  # Reduce noise
    elif sensitivity < 0.02:  # Robust layer
        return base_scale * 2.0  # Increase noise
    else:
        return base_scale
```

#### B. Hardware Migration Planning

```python
# Prioritize which layers need validation during GPU→NPU migration
critical_layers = [L for L in range(num_layers)
                   if sensitivity[L] > 0.10 and divergence[L] > 0.01]
print(f"Validate these layers first: {critical_layers}")
```

#### C. Mixed Precision Strategy

```python
# Determine per-layer precision based on sensitivity
def get_layer_precision(layer_id):
    if sensitivity[layer_id] > 0.15:
        return 'fp16'  # High sensitivity → full precision
    elif sensitivity[layer_id] > 0.05:
        return 'bf16'  # Medium → bfloat16
    else:
        return 'fp8'   # Low sensitivity → aggressive quantization
```

---

## 6. Limitations and Caveats

### 6.1 What This CAN Tell You

- Layer-wise sensitivity ranking
- Operation-level bottlenecks
- Hardware error correlation
- Adaptive noise targeting opportunities

### 6.2 What This CANNOT Tell You

- **Root cause** of hardware errors (identifies WHERE, not WHY)
- **Non-linear interactions** (multiple small errors combining)
- **Temporal effects** (error accumulation over long sequences)
- **Training vs inference mismatch** (E5b: +2.42% training, -14% inference)

### 6.3 Forward vs Backward Noise (Critical!)

Your E8c experiment revealed:
- **Forward noise** → Diagnoses inference robustness
- **Backward noise** → Diagnoses training stability

**Always specify which you're testing!**

---

## 7. Implementation Roadmap

### Phase 1: POC (1-2 days)
- [ ] Add `set_selective_layers()` to noisy_ops.py
- [ ] Test on 5 layers (early, middle, late, attention, FFN)
- [ ] Validate reproducibility

### Phase 2: Full Diagnostic Suite (1 week)
- [ ] Full layer sweep (all layers)
- [ ] Operation-level breakdown
- [ ] Generate sensitivity heatmap
- [ ] Create diagnostic script

### Phase 3: Hardware Localization (1 week)
- [ ] Implement activation capture
- [ ] GPU vs NPU comparison
- [ ] Cross-reference with sensitivity

### Phase 4: Adaptive AQN (2 weeks)
- [ ] Layer-specific noise scaling
- [ ] Training experiments
- [ ] Publication preparation

---

## 8. Related Work

- **Bayes-Optimized Noise Injection** (Nature 2023): 10-100x robustness via targeted noise
- **Hardware-Aware Training** (Nature 2023): Hardware noise distributions > generic Gaussian
- **Neural Architecture Search with Noise**: Finds robust architectures
- **Fisher Information Matrix**: Theoretical sensitivity analysis
- **Layer-wise Relevance Propagation**: Identifies important layers

---

## 9. Conclusion

Using AQN as a diagnostic probe is **feasible and valuable**. The approach:

1. **Leverages existing infrastructure** (noisy_ops.py)
2. **Provides actionable insights** (which layers need protection)
3. **Enables optimizations** (adaptive noise, mixed precision)
4. **Represents novel research** (publishable contribution)

**Next step**: Implement `set_selective_layers()` API and run proof-of-concept diagnostic.
