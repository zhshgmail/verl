# Hardware Error Source Localization via SRDD

**Version**: 5.3
**Date**: 2026-01-06
**Status**: Dense faults 100% accurate; Sparse fault limits identified

---

## Overview

**SRDD (Self-Referential Differential Diagnosis)** is a method to locate hardware error sources in LLMs **without a reference system**. Unlike fingerprint correlation (which requires a known-good GPU), SRDD works by analyzing layer behavior anomalies using local measurements.

| Approach | Requires Reference | Accuracy | Script |
|----------|-------------------|----------|--------|
| Fingerprint Correlation | Yes | 100% | `error_source_finder.py` |
| **SRDD v5.3** | **No** | **100% (dense)** | `srdd_error_finder.py` |

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

```bash
# v5.3: Test sparse fault (e.g., 1% of neurons affected)
python scripts/srdd_error_finder.py \
    --model_path /path/to/model \
    --ground_truth_layer 10 \
    --fault_type dead_zone \
    --sparsity 0.01
```

---

## SRDD v5.3 Detection Methods

SRDD v5.3 uses three complementary detection methods, each targeting a specific fault type:

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

## v5.3: Sparse Fault Analysis

### Background (Gemini Collaboration)

v5.2 achieves 100% accuracy on **dense faults** (affecting ALL tensor elements). However, real hardware faults are often **sparse/local**:

- Single bad GPU core → affects only 1/128 of neurons (~0.8%)
- Bad memory bank → affects specific address ranges
- Bit flip in ALU → affects specific computation paths

**Question**: How sparse can a fault be before SRDD misses it?

### Sparse Fault Simulation

v5.3 adds a `--sparsity` parameter to simulate local faults:

```python
# In HardwareFaultSimulator
if self.sparsity < 1.0:
    sparse_mask = torch.rand_like(hidden_states) < self.sparsity
    hidden_states = torch.where(sparse_mask, faulty_states, original_states)
```

- `sparsity=1.0`: Dense fault (ALL elements affected) - default
- `sparsity=0.01`: Sparse fault (only 1% of elements affected)

### Sparse Fault Detection Results (A100, Qwen2.5-1.5B)

| Fault Type | 100% | 10% | 5% | 1% |
|-----------|------|-----|-----|-----|
| **noise** | ✓ EXACT | ✓ EXACT | ✓ EXACT | ✓ EXACT |
| **dead_zone** | ✓ EXACT | ✓ EXACT | ~33% fail | ✓ EXACT* |
| **saturation** | ✓ EXACT | ✗ MISS | ~50% fail | ✗ MISS |

*Note: 1% dead_zone success may be due to random mask placement

### Analysis

**Why noise detection is robust to sparsity:**
- Instability scan measures **variance across trials**
- Even sparse random noise creates detectable trial-to-trial variation
- Edge detection finds the FIRST layer with instability spike

**Why saturation detection fails on sparse faults:**
- Kurtosis is a **global statistic** (mean of 4th moments)
- Sparse clipping (1-10% of values) barely affects global kurtosis
- The signal gets "averaged out" by the 90-99% healthy values

**Why dead_zone is intermediate:**
- At high sparsity (100%), gain drops dramatically (detected via Local Gain)
- At low sparsity (<10%), sparse zeroing creates instability (detected via Noise method)
- At medium sparsity (5%), neither signal is strong enough

### Detection Threshold Summary

| Fault Type | Reliable Detection | Method Used |
|-----------|-------------------|-------------|
| Noise | Down to **1%** sparsity | Instability Scan |
| Dead Zone | Down to **~7%** sparsity | Local Gain / Instability |
| Saturation | **Dense only** (100%) | Kurtosis Scan |

### Known Limitations

1. **Saturation blind spot**: Cannot detect sparse saturation (<10% of neurons)
2. **Potential fix**: Use L∞ norm (max-error) instead of kurtosis for sparse clipping detection
3. **ALU bugs**: Deterministic logic errors (e.g., `2*3=5`) are not detected by any current method

### v6.0 Experiment: Max Gain Scan (Attempted Fix)

**Hypothesis**: Saturation clips HIGH values, so we should look at max_gain (99.9th percentile) instead of min_gain (0.1th percentile).

**Implementation**:
```python
# Inject DC bias (+50) to layer input
# Measure per-element output shift
# Look at max_gain = quantile(shift/bias, 0.999)
# Saturated neurons can't shift as much (hit ceiling)
```

**Results**: The max_gain approach also failed to detect sparse saturation.

| Sparsity | GT Layer | Diagnosed | Result |
|----------|----------|-----------|--------|
| 100% | L10 | L2 | Rank 2 |
| 10% | L10 | L2 | MISS |
| 5% | L10 | L2 | MISS |

**Analysis**: The DC bias test doesn't work because:
1. Saturation clips values ABOVE a threshold
2. Not all neurons exceed the threshold, so not all get clipped
3. With 10% sparsity AND partial activation, effective affected neurons << 10%
4. The signal is too weak to distinguish from baseline variation

### Fundamental Challenge: Sparse Saturation

**Why sparse saturation is hard to detect**:

1. **Global statistics fail**: Kurtosis, min_gain, max_gain all measure aggregate behavior. When 99% of neurons are healthy, the aggregate looks normal.

2. **The "needle in haystack" problem**: We're looking for 1-10% anomalous neurons among millions.

3. **Activation-dependent**: Saturation only affects neurons that EXCEED the clamp threshold. Many neurons may not reach it.

**Potential approaches (not yet implemented)**:
- Histogram analysis: Look for a "spike" at the saturation ceiling
- Per-neuron tracking: Compare activation distributions before/after layer
- Synthetic probe: Generate inputs designed to trigger specific neuron ranges

### Future Work

- Investigate histogram-based saturation detection
- Add structured fault patterns (row/column dead, addressing errors)
- Investigate deterministic ALU bug detection

---

## Validation Results

### v5.3 Dense Fault Results (A100, Qwen2.5-1.5B)

| Fault Type | GT Layer | Diagnosed | Result | Method |
|-----------|----------|-----------|--------|--------|
| dead_zone | L10 | **L10** | EXACT MATCH | Local Gain (z=-27.3) |
| saturation | L10 | **L10** | EXACT MATCH | Kurtosis Edge (drop_z=-434.2) |
| noise | L10 | **L10** | EXACT MATCH | Instability Edge |

### v5.3 Sparse Fault Results (A100, Qwen2.5-1.5B)

| Fault Type | Sparsity | GT Layer | Diagnosed | Result |
|-----------|----------|----------|-----------|--------|
| noise | 1% | L10 | **L10** | EXACT MATCH |
| dead_zone | 10% | L10 | **L10** | EXACT MATCH |
| dead_zone | 5% | L10 | L11/L18 | UNRELIABLE |
| saturation | 10% | L10 | L20 | MISS |

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

- Gemini collaboration for SRDD v2.0-v5.3 methodology
- Information Bottleneck: [arXiv:2106.12912](https://arxiv.org/abs/2106.12912)

---

**Document Status**: v5.3 validated on A100 - Dense faults 100% accurate; Sparse fault limits documented
**Last Updated**: 2026-01-06
