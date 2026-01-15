# SRDD Co-Library Validation Report

**Date**: 2026-01-11
**Branch**: `feature/npu-aqn-test`
**Status**: COMPLETED

---

## Executive Summary

This validation effort discovered that:
1. **verl's NVFP4 implementation is CORRECT** - it matches manual reference implementation perfectly
2. **The SRDD scanner's "relative error" metric is MISLEADING** for comparing MXFP4 vs NVFP4
3. **By standard quantization metrics (SQNR, MSE), NVFP4 is actually slightly BETTER than MXFP4**
4. The "~1% error" claim in `verl/utils/nvfp4_quant.py` docstring is WRONG

---

## Test Results

### 1. Implementation Verification

Compared verl's NVFP4 implementation against manual reference:

| Implementation | Relative Error |
|---------------|----------------|
| Manual NVFP4 (group_size=16) | 24.17% |
| verl NVFP4 | 24.17% |
| **Difference** | **0.00%** |

**Conclusion**: verl's NVFP4 implementation is correct.

### 2. Multi-Metric Comparison (Qwen2.5-1.5B-Instruct, 196 layers)

| Metric | NVFP4 | MXFP4 | Winner |
|--------|-------|-------|--------|
| **SRDD rel_error** | 26.51% | 21.77% | MXFP4 |
| **MSE** | 0.000011 | 0.000012 | **NVFP4** |
| **RMSE** | 0.003242 | 0.003351 | **NVFP4** |
| **SQNR (dB)** | 18.83 | 18.59 | **NVFP4** |
| **Cosine Similarity** | 0.9949 | 0.9945 | **NVFP4** |
| **Max Error** | 0.042 | 0.085 | **NVFP4** |
| **Deadzone Ratio** | 13.23% | 9.43% | MXFP4 |

### 3. Per-Layer Analysis

- Total layers tested: 196
- MXFP4 wins (by rel_error): 192 layers
- NVFP4 wins (by rel_error): 4 layers

---

## Root Cause Analysis

### Why Does SRDD Show NVFP4 as Worse?

The SRDD scanner uses this formula for relative error:
```python
rel_error = (|quantized - original| / (|original| + 1e-10)).mean()
```

**Problem**: This metric is heavily biased by small values:
1. NVFP4 uses smaller group_size (16 vs 32), creating **more zeros** (13.23% vs 9.43%)
2. When small values (e.g., 0.001) become 0, the relative error is ~100%
3. These outliers inflate the mean significantly

### Standard Quantization Metrics Tell Different Story

| Metric | Meaning | NVFP4 Result |
|--------|---------|--------------|
| **SQNR** | Signal-to-Quantization-Noise Ratio | NVFP4 18.83 dB > MXFP4 18.59 dB |
| **MSE** | Mean Squared Error | NVFP4 0.000011 < MXFP4 0.000012 |
| **Cosine Sim** | Direction preservation | NVFP4 0.9949 > MXFP4 0.9945 |

By these metrics, **NVFP4 is slightly BETTER**.

---

## Recommendations

### 1. Update SRDD Scanner Metric

Change from relative error to SQNR or use better formulation:
```python
# Option 1: Use SQNR (dB)
signal_power = (original ** 2).mean()
noise_power = ((quantized - original) ** 2).mean()
sqnr_db = 10 * log10(signal_power / noise_power)

# Option 2: Use MSE-based relative error
mse_rel = sqrt(mse) / original.std()
```

### 2. Fix Docstring in nvfp4_quant.py

The current docstring claims:
```
NVFP4 has ~1% relative error vs ~21% for MXFP4
```

This is **WRONG**. Should be updated to reflect actual results:
```
NVFP4 and MXFP4 have similar quantization quality:
- SQNR: NVFP4 ~18.8 dB, MXFP4 ~18.6 dB (NVFP4 slightly better)
- Relative Error (mean): NVFP4 ~26%, MXFP4 ~22% (metric bias from deadzone)
- NVFP4 creates more zeros due to smaller group_size (16 vs 32)
```

### 3. Re-evaluate Experiment Conclusions

The previous conclusion that "NVFP4+LoRA (63.84%) performed worse than MXFP4+LoRA (65.88%)" may NOT be due to higher quantization error. Other factors to investigate:
- Different group sizes affect gradient flow differently
- Smaller groups may cause different training dynamics
- Need to test with SQNR-based SRDD scanning

---

## llm-compressor Comparison Notes

The llm-compressor test showed 0% error because:
1. llm-compressor uses **lazy quantization** - weights are not modified in place
2. It stores `weight_scale` separately and applies quantization during forward pass
3. Our comparison script compared the same (unmodified) weights

This is a test methodology issue, not an implementation bug.

---

## Files Changed/Created

| File | Purpose |
|------|---------|
| `scripts/srdd_colibrary_comparison.py` | Original comparison script |
| `docs/qerl/SRDD_COLIBRARY_VALIDATION.md` | This document |

---

## Next Steps

1. [ ] Update SRDD scanner to use SQNR or improved metric
2. [ ] Fix nvfp4_quant.py docstring
3. [ ] Re-run LoRA experiments with SQNR-based SRDD
4. [ ] Investigate why NVFP4+LoRA performs worse despite better SQNR

---

## Session Info

- A100 Container: `verl-r3-test`
- Model: Qwen2.5-1.5B-Instruct
- Python packages: torch, transformers, llmcompressor (0.9.1.dev34)
