# Using Noise Injection as a Diagnostic Probe

**Date**: 2026-01-05
**Status**: Concept Validated, Implementation Ready
**Detailed doc**: `docs/qerl/AQN_DIAGNOSTIC_PROBE.md`

---

## 1. Core Idea

Use **selective noise injection** to identify which parts of a model are most sensitive to computational errors, enabling:

1. **Layer sensitivity profiling** - Which layers are critical?
2. **Hardware error localization** - Where do GPU/NPU differences matter?
3. **Adaptive training** - Apply stronger noise to robust layers

---

## 2. Why This Works

### 2.1 Evidence from E5b/E7c

| Model | Training Benefit | Inference Robustness (5% noise) |
|-------|-----------------|--------------------------------|
| 1.5B (E5b) | +2.42% | **-14%** (brittle) |
| 7B (E7c) | +0.80% | **0%** (robust!) |

**Key insight**: The 7B model has architectural redundancy that buffers noise. By probing layer-by-layer, we can identify WHERE this redundancy exists.

### 2.2 Theoretical Basis

- **Differential Sensitivity**: Different layers have different robustness
- **Information Bottleneck**: Attention/normalization are typically more sensitive
- **Gradient Flow**: Forward vs backward noise have different effects

---

## 3. Implementation (Ready)

### 3.1 New API in `noisy_ops.py`

```python
from verl.utils.noisy_ops import (
    enable_noisy_ops,
    set_selective_layers,
    register_layer_hooks,
    get_layer_injection_stats
)

# 1. Register hooks to track current layer
model = load_model(...)
register_layer_hooks(model)

# 2. Enable noise for specific layers only
set_selective_layers([0, 5, 10])  # Only layers 0, 5, 10
enable_noisy_ops(error_scale=0.05)

# 3. Run evaluation
accuracy = evaluate(model, test_data)

# 4. Get per-layer statistics
stats = get_layer_injection_stats()
# {0: {'forward': 1000, 'backward': 0}, 5: {...}, 10: {...}}
```

### 3.2 Diagnostic Protocol

```
For layer L in [0, num_layers):
    set_selective_layers([L])
    accuracy_L = evaluate(model, test_data)
    sensitivity[L] = (baseline - accuracy_L) / baseline

# Output: sensitivity heatmap showing which layers are critical
```

---

## 4. Use Cases

### 4.1 Pre-deployment Robustness Assessment

Before deploying to NPU:
1. Run layer sensitivity profiling on GPU
2. Identify critical layers (sensitivity > 10%)
3. Ensure those layers have sufficient numerical precision on NPU

### 4.2 Hardware Migration Planning

When moving GPU → NPU:
1. Compare activations layer-by-layer
2. Cross-reference with sensitivity profile
3. High divergence + High sensitivity = CRITICAL (fix first)

### 4.3 Adaptive Training (Future)

```python
def get_layer_noise_scale(layer_id, base_scale=0.05):
    if sensitivity[layer_id] > 0.10:  # Critical
        return base_scale * 0.5  # Less noise
    elif sensitivity[layer_id] < 0.02:  # Robust
        return base_scale * 2.0  # More noise
    return base_scale
```

---

## 5. Key Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Relative Degradation | `(acc_clean - acc_noisy) / acc_clean` | % accuracy loss |
| Critical Threshold | Min noise causing >10% drop | Brittleness |
| SNR | `activation.abs().mean() / noise_scale` | Signal strength |

---

## 6. Limitations

**CAN identify**:
- Which layers are sensitive
- Which operations are bottlenecks
- Where hardware differences matter

**CANNOT identify**:
- Root cause of sensitivity
- Non-linear interactions
- Error accumulation over time

---

## 7. Implementation Status

| Component | Status |
|-----------|--------|
| `set_selective_layers()` | Implemented |
| `register_layer_hooks()` | Implemented |
| `get_layer_injection_stats()` | Implemented |
| All 6 operators updated | Implemented |
| Diagnostic script | Planned |
| Visualization tools | Planned |

**Commit**: `5b2f21e5`

---

## 8. Next Steps

1. **POC** (1-2 days): Test on 5-10 layers of 1.5B model
2. **Full profile** (1 week): All layers, generate heatmap
3. **Hardware comparison** (1 week): GPU vs NPU divergence
4. **Publication** (optional): Novel layer-wise noise profiling for LLMs

---

**See detailed methodology**: `docs/qerl/AQN_DIAGNOSTIC_PROBE.md`
