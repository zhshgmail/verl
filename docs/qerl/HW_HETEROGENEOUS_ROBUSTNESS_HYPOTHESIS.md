# AQN for Hardware Heterogeneous Error Robustness

**Date**: 2025-12-29
**Branch**: `feature/npu-aqn-test`
**Status**: Hypothesis Under Validation

## Executive Summary

We propose a novel application of Adaptive Quantization Noise (AQN) beyond its original purpose. While QeRL designed AQN for **quantization robustness**, we hypothesize it may also improve **Hardware Heterogeneous Error robustness** when training on different hardware than the base model.

## Background

### Original QeRL Purpose
QeRL (Quantization-aware Reinforcement Learning) uses Gaussian noise injection to:
- Simulate quantization errors during BF16 training
- Make models robust to precision reduction when deployed as nvfp4
- Target: GPU training → Quantized GPU inference

### Our Use Case
- Base model: Qwen2.5-1.5B-Instruct (trained on NVIDIA GPUs)
- Post-training: GRPO on Ascend 910C NPUs
- Target: GPU pre-trained → NPU post-trained

## Research Hypothesis

**Gaussian noise injection may improve model robustness to Hardware Heterogeneous Errors** - the subtle numerical differences that arise when training continues on different hardware than the original pre-training.

### Analogy

| Aspect | Quantization Error | HW Heterogeneous Error |
|--------|-------------------|------------------------|
| **Cause** | Precision reduction (FP32→nvfp4) | GPU↔NPU numerical differences |
| **Nature** | Small distributed perturbations | Small distributed perturbations |
| **Location** | Every layer | Every layer |
| **Character** | Systematic but hard to model | Systematic but hard to model |

### Sources of HW Heterogeneous Error

| Component | GPU (NVIDIA) | NPU (Ascend) | Potential Difference |
|-----------|--------------|--------------|---------------------|
| BF16 implementation | CUDA BF16 | CANN BF16 | Rounding modes, denormal handling |
| MatMul | Tensor Cores | Cube Units | Accumulation order, internal precision |
| Activation functions | cuDNN | CANN kernels | tanh, gelu, silu precision |
| Memory layout | NCHW optimized | Potentially different | Data arrangement effects |
| Reduction operations | CUDA reduce | CANN reduce | Summation order differences |

## Theoretical Justification

### Why Gaussian Noise Might Help

1. **Perturbation robustness generalizes**: A model trained to be robust to random perturbations may also be robust to systematic perturbations from HW differences.

2. **Similar error characteristics**: Both quantization and HW heterogeneity introduce small, layer-wise perturbations that accumulate through the network.

3. **Regularization effect**: Noise injection acts as a form of regularization, potentially helping the model find flatter minima that are less sensitive to small numerical variations.

### Theoretical Limitations

1. **Gaussian ≠ HW Error Distribution**: The actual distribution of HW heterogeneous errors is unknown and likely not Gaussian.

2. **Error magnitude unknown**: We don't know if the sigma values from QeRL (0.05-0.0005) are appropriate for HW errors.

3. **Error correlation**: HW errors may be correlated across layers, while Gaussian noise is independent.

## Experimental Validation

### Test 3 v2 (2025-12-29)

| Test | Sigma Range | Purpose |
|------|-------------|---------|
| Test 3a v2 | 0.05 → 0.0005 | QeRL original parameters |
| Test 3b v2 | 0.10 → 0.001 | 2x stronger noise |

**Configuration:**
- Model: Qwen2.5-1.5B-Instruct (BF16, NOT quantized)
- Hardware: 8x Ascend 910C NPUs
- Task: GSM8K GRPO training
- Baseline: Previous tests without noise (Test 2)

**Key Metrics:**
- GSM8K validation accuracy at steps 20, 40, 60, 80, 100, 116
- Compare AQN vs Baseline (no noise) at same steps

### Critical Bug Fix

A critical bug was discovered and fixed on 2025-12-29:
- **Bug**: `isinstance(module, Qwen2RMSNorm)` failed because vLLM uses different classes
- **Impact**: All previous "AQN" tests had NO noise applied (0 layers affected)
- **Fix**: Changed to class name detection: `'rmsnorm' in class_name.lower()`
- **Verification**: Now applying noise to 57 RMSNorm layers per step

## Expected Outcomes

### If Hypothesis is Validated (AQN improves accuracy)
- Novel application of noise injection for cross-hardware training
- Practical benefit for GPU→NPU model transfer
- Potential research contribution

### If Hypothesis is Rejected (AQN hurts or no effect)
- Confirms noise injection is specific to quantization use case
- HW heterogeneous errors may be too small to benefit from regularization
- Or sigma values need different tuning for HW errors

## Future Work

1. **Measure actual HW errors**: Compare GPU vs NPU outputs for same inputs to characterize the actual error distribution.

2. **Targeted noise**: If HW errors are layer-specific, apply noise only to affected layers.

3. **Adaptive sigma**: Tune sigma based on measured HW error magnitude rather than quantization-derived values.

4. **Other regularization**: Compare with other techniques (dropout, weight decay) for HW robustness.

## Results

### Test 3 v2 Results (To be updated)

| Step | Test 3a v2 (σ=0.05) | Test 3b v2 (σ=0.10) | Baseline (Test 2) |
|------|---------------------|---------------------|-------------------|
| 0 | (pending) | (pending) | 23.96% |
| 20 | (pending) | (pending) | 74.60% |
| 40 | (pending) | (pending) | 74.83% |
| 60 | (pending) | (pending) | 76.04% |
| 80 | (pending) | (pending) | 74.22% |
| 100 | (pending) | (pending) | 75.82% |
| 116 | (pending) | (pending) | 76.42% |

## References

- QeRL: Original quantization-aware RL paper
- VERL: `verl/utils/noise_injection.py`
- Test logs: `/tmp/test3a_qerl_v2.log`, `/tmp/test3b_strong_v2.log`
