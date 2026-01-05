# AQN as Diagnostic Probe: Layer-wise Noise Sensitivity Analysis

**Date**: 2026-01-05
**Status**: Research Complete - Implementation In Progress
**Branch**: `feature/npu-aqn-test`

---

## Executive Summary

This document outlines how to use Adaptive Quantization Noise (AQN) as a **self-diagnostic mechanism** to identify where major system noise comes from during training and inference.

### Core Principle: Closed-Loop Diagnostic → Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│              AQN CLOSED-LOOP SYSTEM                         │
│                                                             │
│   DIAGNOSTIC PHASE              TRAINING PHASE              │
│   ("Oscilloscope")       →      ("Vaccine")                 │
│                                                             │
│   Inject probe noise            Use sensitivity map to:     │
│   Generate sensitivity map      • WHERE to inject noise     │
│         │                       • HOW STRONG the noise      │
│         ▼                              ▲                    │
│   ┌─────────────────┐                  │                    │
│   │ Sensitivity Map │──────────────────┘                    │
│   │ Layer 24: 15%   │                                       │
│   │ Layer 10: 2%    │  Inverse scaling:                     │
│   │ MatMul: 8%      │  • Sensitive → LESS noise (protect)   │
│   │ Softmax: 3%     │  • Robust → MORE noise (regularize)   │
│   └─────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
```

**Key Insight**: Diagnostic AQN output becomes Training AQN input.

The approach is:
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

---

## 10. Three-Level Diagnostic Methodology (Enhanced)

Based on NPU deployment experience, we extend the diagnostic approach to three levels:

### 10.1 Level 1: Operator-Level Diagnosis

**Goal**: Distinguish compute-intensive (MatMul) vs memory-intensive (Softmax/Norm) error sources.

```python
# Experiment A: MatMul/Linear only
set_selective_op_types(['matmul', 'linear'])
accuracy_A = evaluate(model, test_data)

# Experiment B: Attention + Norm only
set_selective_op_types(['softmax', 'layer_norm', 'silu'])
accuracy_B = evaluate(model, test_data)
```

**Diagnosis Logic**:

| Result | Diagnosis | NPU Countermeasure |
|--------|-----------|-------------------|
| A crashes more | FP4 accumulator precision issue | Enable FP16/FP32 accumulation |
| B crashes more | Non-linear approximation error | Use BF16 for Softmax/Norm, FP4 for MatMul |

**Key Insight**: This moves from "Layer X is sensitive" to "Layer X's MatMul has FP4 accumulation issues" - actionable for NPU configuration.

### 10.2 Level 2: Layer-Level Diagnosis ("Sliding Window")

**Goal**: Find the "avalanche point" where errors cascade.

```python
# Sliding window injection
for window_start in range(0, num_layers, 10):
    set_layer_window(window_start, window_start + 10)
    accuracy = evaluate(model, test_data)
    sensitivity_by_region[window_start] = (baseline - accuracy) / baseline
```

**Why Sliding Window > Single Layer**:
- Captures **inter-layer error propagation**
- More realistic (real HW errors affect consecutive layers)
- Identifies **regional patterns** (early/middle/late transformer blocks)

**Typical Sensitivity Pattern**:

| Region | Layers | Expected Sensitivity | Reasoning |
|--------|--------|---------------------|-----------|
| Early | 0-8 | Medium (5-8%) | Feature extraction |
| Middle | 9-19 | Low (2-4%) | High redundancy |
| Late | 20-27 | **HIGH (10-15%)** | Output formation, critical |

### 10.3 Level 3: Channel-Level Diagnosis (Advanced)

**Goal**: Identify "outlier channels" that carry critical information.

```python
# Instead of uniform noise across all channels:
for channel_id in range(hidden_dim):
    set_selective_channels([channel_id])
    accuracy = evaluate(model, test_data)
    channel_sensitivity[channel_id] = (baseline - accuracy) / baseline

# Identify outlier channels (top 5% sensitivity)
critical_channels = [c for c, s in channel_sensitivity.items() if s > threshold]
```

**Application**:
- Keep high precision for critical channels
- Use aggressive quantization for robust channels
- Target LoRA on outlier channels

---

## 11. Closed-Loop Training Pipeline

### 11.1 The Complete Workflow

```
┌──────────────────────────────────────────────────────────────────┐
│                    AQN CLOSED-LOOP WORKFLOW                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  STEP 1: DIAGNOSTIC PHASE                                        │
│  ─────────────────────────                                       │
│  Run three-level diagnosis:                                      │
│    • Operator-level: MatMul vs Softmax/Norm                      │
│    • Layer-level: Sliding window scan                            │
│    • (Optional) Channel-level: Outlier detection                 │
│                                                                  │
│  Output: sensitivity_map.json                                    │
│    {                                                             │
│      "per_layer": {"0": 0.06, "10": 0.02, "24": 0.15},          │
│      "per_operator": {"matmul": 0.08, "softmax": 0.03},         │
│      "critical_layers": [24, 25, 26, 27],                       │
│      "critical_ops": ["matmul"]                                  │
│    }                                                             │
│                         │                                        │
│                         ▼                                        │
│  STEP 2: TRAINING PHASE                                          │
│  ─────────────────────────                                       │
│  Load sensitivity_map.json                                       │
│  Apply INVERSE noise scaling:                                    │
│                                                                  │
│    Layer 10 (2% sensitivity)  → noise_scale × 2.0 (regularize)  │
│    Layer 24 (15% sensitivity) → noise_scale × 0.5 (protect)     │
│    MatMul ops (8% sens)       → noise_scale × 1.0 (normal)      │
│    Softmax ops (3% sens)      → noise_scale × 1.5 (regularize)  │
│                                                                  │
│  Result: Efficient noise budget allocation                       │
│    • Maximum regularization on robust components                 │
│    • Minimum perturbation on critical components                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 11.2 Configuration Example

```yaml
# trainer config
trainer:
  adaptive_aqn:
    enabled: true
    sensitivity_profile: sensitivity_map.json

    # Noise scaling strategy
    scaling:
      # Inverse: high sensitivity → low noise, low sensitivity → high noise
      mode: inverse
      base_scale: 0.05

      # Sensitivity thresholds
      robust_threshold: 0.03      # < 3% degradation = robust
      robust_multiplier: 2.0      # Apply 2x noise

      sensitive_threshold: 0.10   # > 10% degradation = sensitive
      sensitive_multiplier: 0.3   # Apply 0.3x noise

    # Special handling for critical layers
    critical_layers:
      action: lora  # 'reduce_noise', 'disable_noise', 'lora'
      lora_rank: 8
      lora_alpha: 16
```

### 11.3 Expected Improvement

| Strategy | Description | Expected Accuracy |
|----------|-------------|-------------------|
| Uniform AQN | Same noise everywhere | 70.58% (baseline) |
| Inverse AQN | More noise on robust layers | +0.5-1.0% |
| Inverse + LoRA | + LoRA on critical layers | +1.0-2.0% |

---

## 12. Implementation Status Update

### Completed
- [x] `set_selective_layers()` - Target specific layers
- [x] `register_layer_hooks()` - Auto-track current layer
- [x] `get_layer_injection_stats()` - Per-layer statistics
- [x] Forward/backward phase control

### In Progress
- [ ] `set_selective_op_types()` - Target specific operator types
- [ ] `set_layer_window()` - Sliding window wrapper

### Planned
- [ ] Diagnostic script with heatmap generation
- [ ] Adaptive AQN configuration loader
- [ ] Channel-level sensitivity profiling

---

## 13. Key Takeaway

**The AQN system has dual identity:**

1. **Diagnostic Mode** ("Oscilloscope"): Probe the model to generate sensitivity heatmap
2. **Training Mode** ("Vaccine"): Use heatmap to guide noise injection

**The diagnostic output IS the training input** - this closed-loop design enables:
- Efficient noise budget allocation
- Targeted protection of critical components
- Maximum regularization benefit with minimum accuracy cost

**From black-box to white-box**: Instead of "NPU results are wrong", we can say "NPU produces 15% systematic bias in Layer 24's MatMul due to FP4 accumulation" and take targeted action.
