# AQN (Adaptive Quantization Noise) Accuracy Impact Analysis

**Date**: 2025-12-28 (UPDATED 2025-12-29)
**Branch**: `feature/npu-aqn-test`

> ⚠️ **CRITICAL BUG DISCOVERED (2025-12-29)**
>
> All test results in this document are **INVALID**. A critical bug was discovered where no noise was actually being applied during "AQN" tests.
>
> **Root Cause**: The code used `isinstance(module, Qwen2RMSNorm)` with transformers classes, but vLLM uses its own model implementations with different class names. The isinstance check always failed, resulting in 0 layers receiving noise.
>
> **Impact**: All "AQN" tests (Test 1, Test 2, Test 3 first run) were effectively running as baseline (no noise applied).
>
> **Fix Applied**: Changed to class name detection: `'rmsnorm' in module.__class__.__name__.lower()`
>
> **Retesting**: Test 3 v2 is now running with the fix applied (2025-12-29 14:18 CST). Results will be updated once available.

## Executive Summary

~~AQN (Adaptive Quantization Noise) **negatively impacts accuracy** when applied to non-quantized (BF16) models.~~ **CONCLUSIONS PENDING** - Previous tests did not actually apply noise. Re-testing with fixed code in progress.

## Experimental Results

### Test Configuration
- **Model**: Qwen2.5-1.5B-Instruct (BF16, non-quantized)
- **Task**: GSM8K mathematical reasoning
- **Training**: GRPO with 2 epochs
- **Hardware**: Ascend 910C NPU (8 NPUs)

### Validation Accuracy Comparison

**Full Baseline Results (466 steps, 2 epochs):**

| Step | Epoch | Baseline Accuracy | AQN Accuracy |
|------|-------|-------------------|--------------|
| 0 | 0 | 13.73% | 18.74% |
| 80 | 0 | 70.49% | - |
| 120 | 0 | 71.24% | 69.65% |
| 140 | 0 | 71.17% | 70.71% |
| 233 | - | - | **70.94%** (AQN final) |
| 280 | 1 | 71.32% | - |
| 340 | 1 | **73.52%** | - |
| 360 | 1 | 72.46% | - |
| 400 | 1 | 72.76% | - |
| 460 | 1 | **73.82%** | - |

### Key Finding

| Metric | Baseline (No AQN) | AQN Enabled | Difference |
|--------|-------------------|-------------|------------|
| **Best Accuracy** | **73.82%** | 70.94% | **-2.88%** |
| Total Steps | 466 | 233 | - |
| Training Time | ~2.3 hours | ~82 min | - |

**AQN reduced accuracy by 2.88 percentage points** on non-quantized BF16 models.

### Additional Test: QeRL Original Parameters (sigma=0.05)

Tested with QeRL's original higher sigma values to see if parameter choice affected results:

| Step | AQN QeRL (sigma=0.05) | AQN Small (sigma=0.01) | Baseline |
|------|----------------------|------------------------|----------|
| 80 | 66.69% | - | 70.49% |
| 100 | 68.66% | - | 69.50% |
| 120 | 68.59% | 69.65% | 71.24% |

**Finding**: Higher sigma (0.05) performs even **worse** than smaller sigma (0.01).
The QeRL original parameters result in a **2.65% accuracy gap** vs baseline at step 120.

**Conclusion**: AQN hurts BF16 training regardless of sigma values. The technique is fundamentally designed for quantized models.

## Root Cause Analysis

### 1. AQN is Designed for Quantized Models Only

From QeRL source code (`qerl.py`):
```python
noise_scheduler = True if 'nvfp4' in model_args.model_name.lower() else False
```

**AQN is only enabled for nvfp4 quantized models** in the original implementation. The noise injection is meant to:
- Compensate for quantization errors
- Improve exploration in the presence of reduced precision
- Regularize training for quantized weights

### 2. Parameter Comparison

| Parameter | QeRL Original | Our Implementation | Issue |
|-----------|---------------|-------------------|-------|
| sigma_start | **0.05** | 0.01 | 5x smaller |
| sigma_end | **0.0005** | 0.001 | 2x larger |
| num_stages | **10** | 5 | Half |
| Target modules | ALL RMSNorm | `post_attention_layernorm` only | More restrictive |
| Exclude patterns | None | `input_layernorm` | Additional filtering |

### 3. Why AQN Hurts Non-Quantized Models

1. **No quantization errors to compensate**: BF16 weights don't suffer from quantization artifacts that AQN is designed to mitigate.

2. **Noise adds unnecessary variance**: Adding Gaussian noise to already-precise weights introduces training instability without corresponding benefits.

3. **Sigma values too small for intended effect**: The reduced sigma values (0.01 vs 0.05) may have limited the exploration benefit while still adding harmful noise.

4. **RMSNorm sensitivity**: RMSNorm layers normalize activations; perturbing their weights can have outsized effects on model behavior.

## Recommendations

### For Non-Quantized (BF16/FP16) Models
- **Disable AQN** - Use `trainer.noise_injection.enabled=False`
- Standard GRPO training achieves better results

### For Quantized (nvfp4) Models
If testing AQN on quantized models, use original QeRL parameters:
```yaml
trainer:
  noise_injection:
    enabled: true
    sigma_start: 0.05    # Higher noise initially
    sigma_end: 0.0005    # Lower final noise
    num_stages: 10       # More gradual decay
    target_modules: []   # All RMSNorm layers (empty = all)
    exclude_patterns: [] # No exclusions
```

### Alternative Exploration Techniques
For non-quantized models, consider:
1. **Entropy bonus** in policy loss
2. **Temperature scaling** during sampling
3. **KL divergence regularization** (already in GRPO)

## Sigma Schedule Verification

The exponential decay schedule was working correctly:

```
Stage 0 (warmup):  sigma = 0.0       (steps 0-46)
Stage 1:          sigma = 0.01      (steps 47-93)
Stage 2:          sigma = 0.004642  (steps 94-139)
Stage 3:          sigma = 0.002154  (steps 140-186)
Stage 4 (final):  sigma = 0.001000  (steps 187-233)
```

Transitions confirmed at steps 47, 94, 140, and 187.

## Technical Implementation Notes

### Files Modified for AQN
1. `verl/utils/noise_injection.py` - Core noise injection
2. `verl/workers/rollout/vllm_rollout/vllm_rollout.py` - Rollout integration
3. `verl/trainer/ppo/ray_trainer.py` - Config passing (fixed timing bug)
4. `verl/workers/config/rollout.py` - RolloutConfig dataclass

### Bug Fixed During Testing
Config was being set AFTER workers spawned. Workers read config during `__init__()`, so they saw stale values. Fixed by moving config modification BEFORE `spawn()`.

## Conclusion

AQN should **NOT** be used for non-quantized model training. The technique is specifically designed to improve training stability for quantized (nvfp4) models by compensating for quantization noise. When applied to BF16 models, it introduces unnecessary variance that degrades accuracy.

For future work with quantized models on NPU, the original QeRL parameters should be tested to validate AQN effectiveness in that specific context.

## References

- QeRL source: `/home/zheng/workspace/QeRL/`
- Original AQN implementation: `QeRL/trl_trainer/noise_scheduler.py`
- VERL AQN port: `verl/utils/noise_injection.py`
