# Hardware Error Source Localization via SRDD

**Version**: 5.2
**Date**: 2026-01-06
**Status**: ALL 3 fault types 100% accurate

---

## Overview

**SRDD (Self-Referential Differential Diagnosis)** is a method to locate hardware error sources in LLMs **without a reference system**. Unlike fingerprint correlation (which requires a known-good GPU), SRDD works by analyzing layer behavior anomalies using local measurements.

| Approach | Requires Reference | Accuracy | Script |
|----------|-------------------|----------|--------|
| Fingerprint Correlation | Yes | 100% | `error_source_finder.py` |
| **SRDD v5.2** | **No** | **100%** | `srdd_error_finder.py` |

---

## Quick Start

```bash
# Production mode (no ground truth)
python scripts/srdd_error_finder.py \
    --model_path /path/to/model

# Validation mode (with simulated fault)
python scripts/srdd_error_finder.py \
    --model_path /path/to/model \
    --ground_truth_layer 15 \
    --fault_type saturation \
    --fault_magnitude 0.3
```

Supported fault types: `dead_zone`, `saturation`, `noise`, `bias`, `spike`

---

## SRDD v5.2 Detection Methods

SRDD v5.2 uses three complementary detection methods, each targeting a specific fault type:

### Method 1: Local Gain Scan (Dead Zone Detection)

**Target**: Dead zone faults where small values are zeroed out.

**Principle**: A healthy layer has gain ≈ 1.0 (output changes proportionally to input). A dead zone layer has gain ≈ 0 (signal lost).

**How it works**:
1. Use `forward_pre_hook` to add noise to layer INPUT
2. Use `forward_hook` to capture layer OUTPUT
3. Calculate: `gain = std(output_change) / std(input_noise)`

| Layer State | Gain |
|-------------|------|
| Normal | ≈ 1.0 |
| Dead Zone | ≈ 0.02 |

**Detection**: Z-score of gain < -2.0 indicates dead zone fault.

---

### Method 2: Instability Scan + Edge Detection (Noise Detection)

**Target**: Noise faults where hardware adds random values.

**Principle**: A healthy layer is deterministic (same input → same output). A noise fault causes different outputs across trials.

**How it works**:
1. Run the same input multiple times
2. Capture layer OUTPUT on each trial
3. Calculate: `instability = std(output_trial_i - output_trial_0)`
4. Use edge detection (first derivative) to find fault SOURCE

| Layer State | Instability |
|-------------|-------------|
| Normal | ≈ 0 |
| Noise Fault | >> 0 |

**Why edge detection**: Noise propagates downstream. The FIRST layer with high instability is the source, not the layer with maximum instability.

**Detection**: First layer where `z_score(instability_jump) > 2.0`

---

### Method 3: Kurtosis Scan + Edge Detection (Saturation Detection)

**Target**: Saturation faults where values are clamped at a ceiling.

**Principle**: LLM activations are naturally "spiky" (high kurtosis ≈ 3500). Saturation clips these spikes, causing kurtosis to DROP.

**Why this works**:
- LayerNorm has scale invariance, so input perturbations get normalized away
- Kurtosis is a passive measurement (no injection needed)
- Directly measures the distribution shape change caused by clamping

**How it works**:
1. Capture layer OUTPUT (no injection)
2. Calculate kurtosis of flattened tensor using `scipy.stats.kurtosis`
3. Use edge detection on log-kurtosis to find DROP point

| Layer State | Kurtosis |
|-------------|----------|
| Normal | ≈ 3500 |
| Saturated | ≈ 3000 (significant drop) |

**Detection**: First layer (excluding L0/L1) where `z_score(kurtosis_drop) < -2.0`

---

### Hierarchical Detection Strategy

To prevent interference between methods:

1. **Check dead zone first**: If any layer has gain z-score < -2.0, diagnose as dead zone
2. **Check noise second**: If instability edge detected, diagnose as noise
3. **Check saturation last**: Only apply kurtosis detection if no other fault found

---

## Validation Results

### v5.2 Results (A100)

| Fault Type | GT Layer | Diagnosed | Result | Method |
|-----------|----------|-----------|--------|--------|
| dead_zone | L10 | **L10** | EXACT MATCH | Local Gain (z=-38.6) |
| saturation | L15 | **L15** | EXACT MATCH | Kurtosis Edge (drop_z=-541.2) |
| noise | L10 | **L10** | EXACT MATCH | Instability Edge (jump_z=29.96) |

### Fault Type Summary

| Fault Type | Detection Method | Key Signature |
|-----------|-----------------|---------------|
| Dead Zone | Local Gain | gain << 1.0 |
| Saturation | Kurtosis Edge | First kurtosis DROP |
| Noise | Instability Edge | First instability SPIKE |

---

## Implementation

### Core APIs

```python
from verl.utils.noisy_ops import (
    enable_noisy_ops,
    disable_noisy_ops,
    set_selective_layers,
    register_layer_hooks,
)
```

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/srdd_error_finder.py` | SRDD diagnosis (no reference needed) |
| `scripts/error_source_finder.py` | Fingerprint correlation (needs reference) |

### Key Code Patterns

**Local Gain Measurement**:
```python
# Perturb input via forward_pre_hook
def perturb_input(module, args):
    hidden = args[0]
    noise = torch.randn_like(hidden) * scale * torch.std(hidden)
    return (hidden + noise,) + args[1:]

# Capture output via forward_hook
def capture_output(module, input, output):
    outputs.append(output[0].detach().clone())

# Calculate gain
gain = std(perturbed_output - baseline_output) / std(input_noise)
```

**Instability Measurement**:
```python
# Run same input multiple times
for trial in range(num_trials):
    output = model(input)
    trial_outputs.append(layer_output.clone())

# Compare outputs
instability = std(trial_outputs[1] - trial_outputs[0])
```

**Kurtosis Measurement**:
```python
from scipy.stats import kurtosis
k = kurtosis(layer_output.flatten(), fisher=True)
```

---

## A100 Test Environment

### Connection

```bash
ssh root@90.90.102.18
docker exec -it verl-r3-test bash
cd /home/z00637938/workspace/verl
```

### Model Path

```bash
MODEL_PATH="/home/z00637938/workspace/verl/checkpoints/noisy_ops_e8c_forward_only/e8c_forward_only_5e-2/global_step_116/merged_hf"
```

### Run Tests

```bash
# Test all fault types
python scripts/srdd_error_finder.py --model_path $MODEL_PATH --ground_truth_layer 10 --fault_type dead_zone
python scripts/srdd_error_finder.py --model_path $MODEL_PATH --ground_truth_layer 15 --fault_type saturation
python scripts/srdd_error_finder.py --model_path $MODEL_PATH --ground_truth_layer 10 --fault_type noise
```

---

## Theoretical Background

### Why Local Measurement?

Previous versions (v1-v4) used end-to-end (E2E) probing: inject at layer L[i], measure at final output. This failed due to **propagation masking** - the signal passes through 18+ healthy layers that normalize/mask the anomaly.

**Solution**: Measure at the layer itself ("inject at L[i], measure at L[i]").

### Why Kurtosis for Saturation?

LayerNorm has **scale invariance** - it normalizes input perturbations before they can trigger saturation. Active injection methods fail.

Kurtosis works because:
1. It's a passive measurement (no injection)
2. It measures distribution shape, not magnitude
3. Clamping fundamentally changes the shape (removes spikes)

### Edge Detection Principle

Both noise and saturation effects **propagate downstream**:
- Noise at L10 causes instability at L10, L11, ..., L27
- Saturation at L15 causes kurtosis drop at L15, L16, ..., L27

Using first derivative (edge detection) finds the **source** layer, not just affected layers.

---

## References

- Gemini collaboration for SRDD v2.0-v5.2 methodology
- Information Bottleneck: [arXiv:2106.12912](https://arxiv.org/abs/2106.12912)

---

**Document Status**: v5.2 validated on A100 - ALL 3 fault types 100% accurate
**Last Updated**: 2026-01-06
