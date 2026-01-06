# Noise Injection Diagnostic Methodology

**Version**: 2.0
**Date**: 2026-01-06
**Status**: Validated on A100

---

## Overview

This document describes using **noise injection as a diagnostic probe** to locate sources of numerical errors in LLMs. This is distinct from AQN training for robustness (see `HW_ERROR_INJECTION_EXPERIMENTS.md`).

| Approach | Purpose | Document |
|----------|---------|----------|
| **Diagnostic Probe** | Find error sources (oscilloscope) | This document |
| **Robustness Training** | Tolerate errors (vaccine) | `HW_ERROR_INJECTION_EXPERIMENTS.md` |

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Theoretical Foundations](#2-theoretical-foundations)
3. [Three-Level Diagnostic Protocol](#3-three-level-diagnostic-protocol)
4. [SRDD: Reference-Free Diagnosis](#4-srdd-reference-free-diagnosis)
5. [Validation Results](#5-validation-results)
6. [Implementation](#6-implementation)
7. [A100 Environment](#7-a100-environment)

---

## 1. Quick Start

### 1.1 With Reference System (Fingerprint Correlation)

When you have a known-good GPU for comparison:

```bash
python scripts/error_source_finder.py \
    --model_path /path/to/model \
    --method fingerprint \
    --hw_error_scale 0.20
```

**Result**: 100% exact match accuracy (GT layer = diagnosed layer).

### 1.2 Without Reference System (SRDD)

When NO reference GPU is available:

```bash
# Production mode
python scripts/srdd_error_finder.py \
    --model_path /path/to/model

# Validation mode (with simulated fault)
python scripts/srdd_error_finder.py \
    --model_path /path/to/model \
    --ground_truth_layer 10 \
    --fault_type dead_zone \
    --fault_magnitude 0.3
```

**Result**: 100% top-1 accuracy for dead_zone faults (v2.0).

---

## 2. Theoretical Foundations

### 2.1 Key Insight

- **Hardware errors are DISCRETE**: Affect specific layers/operations
- **Architectural sensitivity is CONTINUOUS**: Varies smoothly across layers
- **Detection method**: Find anomalies in layer response to controlled noise

### 2.2 Information Bottleneck Theory

Layers go through two phases:
- **Fitting phase**: Extracting features → Sensitive to noise
- **Compression phase**: Redundant information → Robust to noise

This explains why larger models (7B) are more robust than smaller ones (1.5B).

### 2.3 Fisher Information Matrix

FIM quantifies parameter sensitivity:
- Higher FIM = parameter significantly affects output
- Used for quantization loss estimation and pruning decisions

---

## 3. Three-Level Diagnostic Protocol

### 3.1 Level 1: Operator-Level

Identify which operations are most sensitive:

```python
from verl.utils.noisy_ops import set_selective_operators

# Test each operator type
for op in ['matmul', 'softmax', 'layer_norm']:
    set_selective_operators([op])
    enable_noisy_ops(error_scale=0.05)
    accuracy = evaluate(model, test_data)
    print(f"{op}: {accuracy:.2%}")
```

### 3.2 Level 2: Layer-Level (Sliding Window)

Find "avalanche points" where errors compound:

```python
# Sliding window analysis
window_size = 3
for start in range(0, num_layers - window_size + 1, window_size):
    layers = list(range(start, start + window_size))
    set_selective_layers(layers)
    enable_noisy_ops(error_scale=0.05)
    accuracy = evaluate(model, test_data)
    degradation = (baseline - accuracy) / baseline
    print(f"L{start}-{start+window_size-1}: {degradation:+.1%}")
```

### 3.3 Level 3: Channel-Level

Detect outlier channels (10-100x magnitude):

```python
# After identifying faulty layer
activations = capture_layer_output(model, layer_id, input)
channel_magnitudes = activations.abs().mean(dim=(0, 1))
median = channel_magnitudes.median()
outliers = (channel_magnitudes / median) > 10.0
```

---

## 4. SRDD: Reference-Free Diagnosis

### 4.1 Overview

SRDD (Self-Referential Differential Diagnosis) locates error sources **without a reference system** using four statistical probes.

### 4.2 Probe A: Monotonicity Test

**Principle**: Healthy layers show monotonic response to increasing noise.

**Metric**: Spearman rank correlation (rho^2)

```python
# KL scales with noise^2, use Spearman not Pearson
rho, _ = stats.spearmanr(noise_scales, sensitivities)
monotonicity = rho ** 2  # 1.0 = perfect monotonicity
```

- Normal layer: rho^2 ~ 1.0
- Faulty layer: rho^2 << 1.0 (non-monotonic)

### 4.3 Probe B: Neighborhood Smoothness

**Principle**: Sensitivity should vary smoothly across adjacent layers.

**Metric**: Second derivative in log-space

```python
# Log-space fixes "last layer dominance" problem
log_sens = np.log(sensitivities + 1e-9)
local_anomaly = |log_sens[i] - (log_sens[i-1] + log_sens[i+1]) / 2|
```

### 4.4 Probe C: Input Invariance

**Principle**: Healthy layers respond consistently across different inputs.

**Metric**: Coefficient of Variation (CV = std/mean)

```python
# "Neurotic" layers: normal for 99 inputs, explodes on 1
cv = std(sensitivities_per_input) / mean(sensitivities_per_input)
```

### 4.5 Probe D: Variance Compression

**Principle**: Saturated layers "absorb" noise instead of amplifying it.

**Metric**: Amplification slope

```python
# Normal: divergence increases with noise (positive slope)
# Saturated: divergence plateaus (slope ~ 0)
slope, _ = np.polyfit(noise_scales, responses, 1)
```

### 4.6 Aggregation

```python
composite_score = (
    max(0, -lin_z) * 1.0 +      # Lower rho^2 = worse
    max(0, smooth_z) * 1.5 +    # Higher local anomaly = worse
    max(0, inv_z) * 1.0 +       # Higher CV = worse
    max(0, -comp_z) * 1.5       # Lower slope = worse
)
```

### 4.7 Critical Implementation Details

**BF16 Numerical Stability**:
```python
def compute_kl_divergence(p_logits, q_logits):
    # BF16 machine epsilon ~1e-3, so eps=1e-10 is effectively 0!
    p_logits = p_logits.float()  # Upcast to float32
    q_logits = q_logits.float()
    eps = 1e-6  # Safe for float32
    # ...
```

**State Management**:
```python
try:
    for layer_id in range(num_layers):
        set_selective_layers([layer_id])
        enable_noisy_ops(...)
        # ... measurement ...
finally:
    disable_noisy_ops()
    set_selective_layers(None)
```

---

## 5. Validation Results

### 5.1 SRDD v2.0 Results (A100)

| GT Layer | Fault Type | Diagnosed | Rank | Result |
|----------|------------|-----------|------|--------|
| 10 | dead_zone (0.3) | 10 | 1 | EXACT MATCH |
| 20 | dead_zone (0.3) | 20 | 1 | EXACT MATCH |
| 15 | saturation (0.2) | 26 | >5 | MISMATCH |

### 5.2 v1.0 vs v2.0 Comparison

| GT Layer | v1.0 Rank | v2.0 Rank | Improvement |
|----------|-----------|-----------|-------------|
| 10 | 4 | **1** | +3 ranks |
| 20 | 4 (L27 dominated) | **1** | L27 dominance fixed |

### 5.3 Fault Type Detectability

| Fault Type | Detectability | Signature |
|------------|---------------|-----------|
| dead_zone | HIGH | Non-monotonic response |
| saturation | LOW | Appears "stable" |
| spike | LOW | Averages out |
| noise | VERY LOW | Similar to diagnostic noise |

### 5.4 Fingerprint Correlation Results (with reference)

| GT Layer | Diagnosed | Similarity | Result |
|----------|-----------|------------|--------|
| 5 | 5 | 1.0000 | EXACT MATCH |
| 10 | 10 | 1.0000 | EXACT MATCH |
| 15 | 15 | 1.0000 | EXACT MATCH |
| 20 | 20 | 1.0000 | EXACT MATCH |
| 25 | 25 | 1.0000 | EXACT MATCH |

**100% accuracy** when reference system available.

---

## 6. Implementation

### 6.1 Core APIs

```python
from verl.utils.noisy_ops import (
    enable_noisy_ops,
    disable_noisy_ops,
    set_selective_layers,
    set_selective_operators,
    register_layer_hooks,
    get_layer_injection_stats,
)
```

### 6.2 Scripts

| Script | Purpose |
|--------|---------|
| `scripts/srdd_error_finder.py` | Reference-free SRDD diagnosis |
| `scripts/error_source_finder.py` | Fingerprint correlation (needs reference) |

---

## 7. A100 Environment

### 7.1 Connection

```bash
ssh root@90.90.102.18
docker exec -it verl-r3-test bash
cd /home/z00637938/workspace/verl
```

### 7.2 Model Paths

```bash
MODEL_PATH="/home/z00637938/workspace/verl/checkpoints/noisy_ops_e8c_forward_only/e8c_forward_only_5e-2/global_step_116/merged_hf"
TOKENIZER_PATH="/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
```

### 7.3 Run Tests

```bash
# Install scipy if needed
pip install scipy

# Run SRDD validation
python scripts/srdd_error_finder.py \
    --model_path $MODEL_PATH \
    --ground_truth_layer 10 \
    --fault_type dead_zone \
    --fault_magnitude 0.3
```

---

## References

- Information Bottleneck: [arXiv:2106.12912](https://arxiv.org/abs/2106.12912)
- Fisher Information for Quantization: BRECQ, FIMA-Q papers
- Gemini collaboration for SRDD v2.0 improvements

---

**Document Status**: Validated on A100
**Last Updated**: 2026-01-06
