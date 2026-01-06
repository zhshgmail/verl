# Hardware Error Source Localization via SRDD

**Version**: 8.1
**Date**: 2026-01-07
**Status**: Dense faults 100% | Sparse 60% (3/5 fault types: dead_zone, noise, spike detectable at 10%; saturation, bias fail)

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

| Fault Type | 100% (Dense) | 10% (Sparse) | Detection Method |
|-----------|--------------|--------------|------------------|
| **dead_zone** | ✓ EXACT | ✓ EXACT | Gain Scan (gain << 1.0) + Min Gain |
| **noise** | ✓ EXACT | ✓ EXACT | Instability Scan (trial variance) |
| **spike** | ✓ EXACT | ✓ EXACT | Instability Scan (large random values) |
| **saturation** | ✓ EXACT | ✗ MISS (L2) | Kurtosis + Histogram (dense only) |
| **bias** | ✓ EXACT | ✗ MISS (L2) | Kurtosis drop (weak signal) |

**Summary**: 3 of 5 fault types detectable at 10% sparsity (60% coverage)

**Key Finding**: Only **saturation** and **bias** fail at sparse levels. These faults don't create instability (deterministic) and don't dramatically change gain (values still flow through).

### Analysis

**Why noise/spike detection is robust to sparsity:**
- Instability scan measures **variance across trials**
- Even sparse random noise/spikes create detectable trial-to-trial variation
- Edge detection finds the FIRST layer with instability spike
- Spike faults are essentially extreme noise - easily detected

**Why dead_zone detection works at 10% sparsity:**
- Min Gain scan detects "weakest link" - ANY neurons with gain ≈ 0
- Discrete scan (v7.0) counts neurons failing to shift by expected amount
- At 10%, the 154 faulty neurons create a clear 10% failure rate signal

**Why saturation detection fails on sparse faults:**
- Kurtosis is a **global statistic** (mean of 4th moments)
- Sparse clipping (10% of values) barely affects global kurtosis
- Histogram pile-up is INSIDE the distribution, not at edges (undetectable)
- The signal gets "averaged out" by the 90% healthy values

**Why bias detection fails on sparse faults:**
- Bias adds a systematic offset, but doesn't create instability
- Doesn't change gain dramatically (values still flow through)
- Only 10% of neurons affected = negligible kurtosis impact
- No clear distinguishing signature compared to natural layer variations

### Detection Threshold Summary

| Fault Type | Reliable Detection | Method Used |
|-----------|-------------------|-------------|
| Noise | Down to **1%** sparsity | Instability Scan |
| Spike | Down to **10%** sparsity | Instability Scan |
| Dead Zone | Down to **10%** sparsity | Min Gain + Discrete Scan |
| Saturation | **Dense only** (100%) | Kurtosis Scan |
| Bias | **Dense only** (100%) | Kurtosis Scan |

### Known Limitations

1. **Saturation/Bias blind spot**: Cannot detect sparse saturation/bias (<10% of neurons)
2. **Root cause**: These faults are deterministic and don't change gain/instability
3. **Potential fix**: Not yet found - histogram pile-up only works for dense faults
4. **ALU bugs**: Deterministic logic errors (e.g., `2*3=5`) are not detected by any current method

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

### v7.0 Experiment: Discrete Outlier Counting (Attempted Fix)

**Hypothesis**: Instead of measuring aggregate statistics (percentiles), COUNT neurons that fail to shift. Even 1 stuck neuron gives count=1, not gain=0.9999.

**Implementation**:
```python
# Inject DC bias (+50) to layer input
# Calculate per-neuron shift
# Determine expected shift (median of layer)
# COUNT neurons shifting < 85% of median (outliers)
```

**Results**: The discrete counting approach also failed to detect sparse saturation.

| Test Case | L10 Failure Rate | L21-27 Failure Rates | Detection |
|-----------|------------------|---------------------|-----------|
| **Baseline (no fault)** | 0.000% | 0.24-13.7% | L2 false positive |
| **10% sparse saturation** | 0.000% | 0.24-13.7% | **IDENTICAL to baseline** |

**Critical finding**: Sparse saturation produces **zero additional signal** above baseline noise.

**Root cause**: The saturation uses a **DYNAMIC threshold** (70% of max output):
```python
max_val = hidden_states.abs().max() * (1.0 - magnitude)
hidden_states = hidden_states.clamp(-max_val, max_val)
```

When we inject DC bias:
1. All outputs increase → max increases
2. Threshold = 0.7 × max also increases
3. Saturated neurons shift WITH the new threshold (not "stuck")
4. shift ≈ 0.7 × delta (healthy shift ≈ delta)

The 30% difference might be detectable in theory, but in practice:
- LayerNorm partially absorbs the DC bias
- Residual connections complicate the shift propagation
- Baseline noise at high layers (L21-27) masks any sparse fault signal

### v7.1 Experiment: Absolute Saturation Threshold

**Hypothesis**: The dynamic threshold defeats detection. Test with FIXED threshold.

**Implementation**: `--absolute_saturation` flag computes threshold ONCE and caches.

**Results**: Tested absolute saturation at different thresholds:

| Test Case | Threshold | Kurtosis Change | Detection |
|-----------|-----------|-----------------|-----------|
| Dense, absolute 70% | 5040 | L10: 3580→2996 (DROP) | L10 rank 2 ✓ |
| Dense, absolute 30% | 2160 | L10: 3580→1506 (DROP) | L10 rank 2 ✓ |
| **Sparse 10%, absolute 30%** | 2160 | L10: 3580→**4110** (INCREASE!) | MISS |
| Sparse 10%, dynamic 70% | varies | L10: 3580→3580 (unchanged) | MISS |

**Critical discovery**: Sparse and dense saturation have **OPPOSITE** kurtosis effects:

- **Dense saturation**: Clips many values → extreme tail removed → kurtosis **DROPS**
- **Sparse saturation**: Clips only 10% → extreme tail mostly intact → kurtosis **INCREASES** (σ decreases more than E[(x-μ)⁴])

This explains why kurtosis-based detection fundamentally cannot detect sparse saturation - it's looking for the wrong direction of change!

### v8.0 Experiment: Histogram Pile-up Detection

**Hypothesis** (from Gemini): Saturation creates pile-up at the clamp threshold. Look for histogram spike.

**Implementation**: `local_histogram_scan` - detect edge pile-up in activation histograms.

**CRITICAL FIX over Gemini's proposal**:
- Gemini looked at top 1% of MAX value (0.99 * max)
- But saturation pile-up is at THRESHOLD, not MAX
- With threshold = 30% of max, pile-up is at 0.3*max, not 0.99*max!

v8.0.1 Method:
- Look at EDGE bins directly (first/last bins of histogram)
- Compare edge bin count to 3rd/4th neighbor bins
- Saturation clamps values to threshold → pile-up at histogram max

**Results**:

| Test Case | Pile-up at L10 | Max Value | Result |
|-----------|----------------|-----------|--------|
| Dense absolute (30%) | **ratio=3.0** | **2160.0** | **EXACT MATCH** ✓ |
| 50% sparse absolute | ratio=1.0 | 5760.0 | MISS |
| 10% sparse absolute | ratio=1.0 | 5216.0 | MISS |

**Why histogram pile-up fails for sparse saturation**:
1. With sparse mask (10-50%), healthy neurons (50-90%) still exist
2. Healthy neurons have values > threshold (up to ~7200)
3. So histogram max = ~7200 (from healthy neurons), not 2160 (threshold)
4. The pile-up at 2160 is **INSIDE** the histogram, not at the edge
5. Edge detection looks at wrong location!

**To detect sparse saturation histogram pile-up**:
- Need to scan **interior** of histogram for unusual peaks
- Compare each bin to smoothed neighbors
- Look for spike at ANY location, not just edges
- Challenge: distinguish saturation spike from natural variation

### Fundamental Challenge: Sparse Saturation

**Why sparse saturation is hard to detect**:

1. **Global statistics fail**: Kurtosis, min_gain, max_gain all measure aggregate behavior. When 99% of neurons are healthy, the aggregate looks normal.

2. **The "needle in haystack" problem**: We're looking for 1-10% anomalous neurons among millions.

3. **Activation-dependent**: Saturation only affects neurons that EXCEED the clamp threshold. Many neurons may not reach it.

**Potential approaches (not yet implemented)**:

1. **Histogram spike detection**: Saturated neurons all have the SAME clamped value, creating a visible spike in the output distribution. Look for histogram peaks at specific values.

2. **Bidirectional kurtosis change**: Instead of looking only for kurtosis DROP, check for ANY significant change (up or down). Sparse saturation causes INCREASE, dense causes DECREASE.

3. **Range compression**: Compare (max - min) range before/after suspected layer. Saturation reduces the range.

4. **Per-neuron tracking**: Run multiple different inputs and track which neurons consistently hit the same ceiling value.

5. **Gradient-based probing**: In training scenarios, saturated neurons will have zero gradient. This is detectable without reference.

### Future Work

- Implement histogram-based saturation detection (spike at ceiling)
- Add bidirectional kurtosis change detection
- Add structured fault patterns (row/column dead, addressing errors)
- Investigate deterministic ALU bug detection
- Explore gradient-based fault detection for training scenarios

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
