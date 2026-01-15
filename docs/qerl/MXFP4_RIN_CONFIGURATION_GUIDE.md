# MXFP4 W4A4 RIN Configuration Guide

**Date**: 2026-01-15
**Based on**: SRDD analysis of MXFP4 activation quantization errors
**Goal**: Configure RIN (Resilient-Improving Noise) for MXFP4 W4A4 experiments (E13i/j/k)

---

## Executive Summary

SRDD analysis reveals that **ALL 28 layers** suffer from high MXFP4 quantization error:
- **Mean relative error**: 36.4% (far above 10% threshold)
- **Mean deadzone**: 22.9% (values falling to zero)
- **Pattern**: Middle layers (10-17) have significantly worse error (38-43%) than edge layers (28-33%)

**Implication**: E13h's 56.41% accuracy vs E13g's 60.88% (-4.47%) is directly attributable to MXFP4's high quantization error. RIN can help bridge this gap by training the model to be more robust to quantization noise.

---

## SRDD Analysis Results

### Tool Capabilities and Limitations

**What the SRDD tool does:**
- Analyzes **activation quantization error only** (not weights)
- Hooks into layer outputs and measures MXFP4 quantization fidelity
- Computes SQNR, deadzone ratio, saturation ratio, relative error

**What it does NOT do:**
- Does NOT analyze weight quantization error
- Does NOT support combined W4A4 analysis (would need separate tool)

**Why this is still useful for RIN:**
- Activation errors directly impact gradient flow and model convergence
- High activation error layers are prime candidates for noise injection
- RIN helps model learn to be robust to these quantization errors

### Overall Statistics

| Metric | Min | Max | Mean±Std | Threshold | Status |
|--------|-----|-----|----------|-----------|--------|
| **SQNR (dB)** | 16.0 | 18.1 | 17.0±0.4 | <20 dB | ❌ All fail |
| **Deadzone %** | 15.7 | 28.7 | 22.9±3.4 | >5% | ❌ All fail |
| **Saturation %** | 0.02 | 0.12 | 0.03±0.02 | >1% | ✅ Pass |
| **Relative Error %** | 28.6 | 42.7 | 36.4±3.3 | >10% | ❌ All fail |

**Key Finding**: 100% of layers (28/28) are problematic for MXFP4 quantization.

---

## Layer-by-Layer Analysis

### Top 10 Worst Layers (Ranked by Relative Error)

| Rank | Layer ID | Relative Error | Deadzone % | SQNR (dB) | Zone |
|------|----------|----------------|------------|-----------|------|
| 1 | **15** | **42.65%** | 28.71% | 17.2 | Middle |
| 2 | **14** | **41.94%** | 27.98% | 17.2 | Middle |
| 3 | **16** | **41.85%** | 27.98% | 17.2 | Middle |
| 4 | **17** | **40.77%** | 27.07% | 17.2 | Middle |
| 5 | **12** | **40.60%** | 26.53% | 17.2 | Middle |
| 6 | **13** | **40.36%** | 26.44% | 17.2 | Middle |
| 7 | **11** | **39.94%** | 26.04% | 17.2 | Middle |
| 8 | **18** | **39.50%** | 25.78% | 17.2 | Middle |
| 9 | **19** | **38.96%** | 25.36% | 17.2 | Middle |
| 10 | **10** | **38.57%** | 24.71% | 17.1 | Middle |

**Pattern**: Layers 10-19 form a contiguous high-error zone in the middle of the network.

### Top 5 Worst by SQNR (Signal Quality)

| Rank | Layer ID | SQNR (dB) | Relative Error | Zone |
|------|----------|-----------|----------------|------|
| 1 | 26 | 16.0 | 28.61% | Last |
| 2 | 27 | 16.1 | 32.03% | Last |
| 3 | 3 | 16.5 | 33.67% | First |
| 4 | 2 | 16.8 | 33.74% | First |
| 5 | 0 | 16.9 | 30.08% | First |

**Pattern**: First 3 layers and last 2 layers have lowest SQNR, but relative error is still lower than middle layers.

### Error Distribution by Zone

| Zone | Layers | Mean Rel Error | Mean Deadzone | Characteristics |
|------|--------|----------------|---------------|-----------------|
| **First** (0-3) | 4 | 32.5% | 18.9% | Input processing, lower error |
| **Early-Mid** (4-9) | 6 | 35.6% | 22.4% | Transition zone |
| **Middle** (10-19) | 10 | **40.4%** | **26.4%** | **WORST ZONE** |
| **Late-Mid** (20-25) | 6 | 35.6% | 21.9% | Transition zone |
| **Last** (26-27) | 2 | 30.3% | 17.1% | Output generation, lower error |

---

## RIN Configuration Strategies

### Strategy 1: RIN-targeted (Binary On/Off)

**Concept**: Apply RIN only to the worst layers (binary selection)

**Configuration**:
```python
trainer.noise_injection.enabled = True
trainer.noise_injection.sigma_start = 0.05
trainer.noise_injection.sigma_end = 0.0005
trainer.noise_injection.num_stages = 10
trainer.noise_injection.target_layers = [15, 14, 16, 17, 12, 13, 11, 10, 18, 19]  # Top 10
trainer.noise_injection.exclude_patterns = ["lm_head", "embed_tokens", "lora_"]
```

**Pros**:
- Simple to implement
- Targets the worst offenders
- Clear ablation for effectiveness

**Cons**:
- Ignores error gradient (treats 42% and 38% error the same)
- Arbitrary cutoff at top 10

**Expected Impact**: +2-3% accuracy improvement (56.41% → 58-59%)

---

### Strategy 2: RIN-variable (Scaled by Relative Error)

**Concept**: Scale noise intensity proportional to each layer's relative error

**Configuration**:
```python
# Compute layer-specific multipliers based on relative error
# multiplier = layer_error / mean_error
# Layer 15: 42.65% / 36.4% = 1.17x
# Layer 0: 30.08% / 36.4% = 0.83x

layer_multipliers = {
    0: 0.83, 1: 0.89, 2: 0.93, 3: 0.93, 4: 0.93,
    5: 0.98, 6: 0.99, 7: 1.01, 8: 1.00, 9: 1.02,
    10: 1.06, 11: 1.10, 12: 1.12, 13: 1.11, 14: 1.15,
    15: 1.17, 16: 1.15, 17: 1.12, 18: 1.09, 19: 1.07,
    20: 1.02, 21: 0.99, 22: 0.96, 23: 0.94, 24: 0.91,
    25: 0.90, 26: 0.79, 27: 0.88,
}

trainer.noise_injection.enabled = True
trainer.noise_injection.sigma_start = 0.05
trainer.noise_injection.sigma_end = 0.0005
trainer.noise_injection.num_stages = 10
trainer.noise_injection.use_variable_sigma = True
trainer.noise_injection.layer_multipliers = layer_multipliers
```

**Pros**:
- Fine-grained control matching actual error distribution
- No arbitrary cutoffs
- Best theoretical alignment with SRDD data

**Cons**:
- More complex implementation
- Harder to ablate (many variables)
- May need code changes to support layer-specific σ

**Expected Impact**: +3-5% accuracy improvement (56.41% → 59-61%)

---

### Strategy 3: Zone-based (3-tier)

**Concept**: Group layers into 3 zones with different noise levels

**Configuration**:
```python
# High-error zone: Layers 10-19 (40.4% mean error)
high_error_layers = list(range(10, 20))
high_sigma = (0.05, 0.001)  # Aggressive decay

# Medium-error zone: Layers 4-9, 20-25 (35.6% mean error)
medium_error_layers = list(range(4, 10)) + list(range(20, 26))
medium_sigma = (0.03, 0.0005)  # Moderate decay

# Low-error zone: Layers 0-3, 26-27 (31.4% mean error)
# No RIN or minimal

# Apply in 3 separate experiments or use multi-config
```

**Pros**:
- Balances simplicity and granularity
- Clear zone definitions
- Easy to implement with existing noise injection code

**Cons**:
- Loses within-zone variation
- Requires running multiple configs or custom implementation

**Expected Impact**: +2-4% accuracy improvement (56.41% → 58-60%)

---

## Recommended Experiment Plan

### E13i: RIN-targeted (Top 10 Layers)

**Experiment ID**: E13i
**Config**: MXFP4 W4A4 + STE + RIN-targeted
**Target layers**: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19] (top 10 by error)
**Noise schedule**: σ = 0.05 → 0.0005 (10 stages)
**Baseline**: E13h (56.41%)
**Target**: 58-59% (+2-3%)

### E13j: RIN-variable (Error-proportional)

**Experiment ID**: E13j
**Config**: MXFP4 W4A4 + STE + RIN-variable
**Layer multipliers**: Based on relative_error / mean_error (see Strategy 2)
**Noise schedule**: Base σ = 0.05 → 0.0005, scaled per layer
**Baseline**: E13h (56.41%)
**Target**: 59-61% (+3-5%)
**Note**: May require code changes to support variable σ per layer

### E13k: Zone-based (High-error zone only)

**Experiment ID**: E13k
**Config**: MXFP4 W4A4 + STE + RIN (layers 10-19 only)
**Target layers**: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19] (middle zone)
**Noise schedule**: σ = 0.05 → 0.001 (more aggressive)
**Baseline**: E13h (56.41%)
**Target**: 58-60% (+2-4%)

---

## Implementation Notes

### Current Code Support

The existing noise injection framework in `verl/trainer/config/ppo_trainer.yaml` supports:
```yaml
noise_injection:
  enabled: false
  sigma_start: 0.05
  sigma_end: 0.0005
  num_stages: 10
  target_modules: []  # Empty = auto-detect
  exclude_patterns: []  # Empty = auto-detect
```

**Supported**: Binary on/off via `target_layers` parameter (if implemented)
**NOT supported yet**: Variable σ per layer (requires code changes)

### Required Code Changes for RIN-variable

If implementing Strategy 2 (RIN-variable), need to modify:

1. **Config** (`verl/trainer/config/ppo_trainer.yaml`):
```yaml
noise_injection:
  use_variable_sigma: false
  layer_multipliers: {}  # Dict[int, float]
```

2. **Noise injector** (`verl/utils/noise_injector.py` or similar):
```python
if self.config.use_variable_sigma:
    layer_id = self._get_layer_id_from_module(module)
    multiplier = self.config.layer_multipliers.get(layer_id, 1.0)
    sigma = base_sigma * multiplier
else:
    sigma = base_sigma
```

---

## Expected Outcomes

### Success Criteria

| Experiment | Target Accuracy | vs E13h | vs E13g | Status |
|------------|-----------------|---------|---------|--------|
| E13i (targeted) | 58-59% | +2-3% | -2 to -1% | TBD |
| E13j (variable) | 59-61% | +3-5% | -1 to +1% | TBD |
| E13k (zone-based) | 58-60% | +2-4% | -2 to 0% | TBD |

**Goal**: Close the gap between MXFP4 (56.41%) and NVFP4 (60.88%) using RIN.

### Comparison with E12 (W4A16 + RIN)

E12 achieved **72.48%** with MXFP4 W4A16 + RIN-variable (high σ).

**Key differences**:
- E12: W4A16 (only weights quantized) + high σ (0.1 start)
- E13i/j/k: W4A4 (weights AND activations) + moderate σ (0.05 start)
- Expected W4A4 ceiling: ~60-62% (due to activation quantization overhead)

---

## Next Steps

1. **Confirm code support**: Check if `target_layers` parameter exists in noise injection framework
2. **Start with E13i**: Simplest implementation, validates RIN helps MXFP4 W4A4
3. **If E13i succeeds** (58%+): Implement E13j for maximum performance
4. **Document results**: Update E13_W4A4_EXPERIMENT_LOG.md with findings

---

## References

- **SRDD Results**: `/home/z00637938/workspace/verl/logs/srdd_analysis/mxfp4_activation_scan_20260115.json`
- **E13h Baseline**: 56.41% (MXFP4 W4A4 + STE, no RIN)
- **E13g Comparison**: 60.88% (NVFP4 W4A4 + STE, no RIN)
- **E12 Reference**: 72.48% (MXFP4 W4A16 + RIN-variable)
