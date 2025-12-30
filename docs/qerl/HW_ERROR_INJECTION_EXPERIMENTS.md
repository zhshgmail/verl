# HW Heterogeneous Error Injection Experiments

**Date**: 2025-12-30
**Branch**: `feature/npu-aqn-test`
**Status**: In Progress

## Executive Summary

This document tracks experiments using **synthetic HW heterogeneous error injection** on GPU to simulate GPU/NPU numerical differences. The goal is to test whether AQN (Adaptive Quantization Noise) can improve model robustness to hardware heterogeneous errors.

## Background

### Problem Statement
- NPU baseline and GPU baseline show no significant difference in GSM8K accuracy
- AQN (Adaptive Quantization Noise) didn't show meaningful impact in our tests
- **Hypothesis**: The natural HW differences between GPU/NPU may be too small to observe

### Proposed Solution
Inject **synthetic HW heterogeneous errors** on GPU during training to:
1. Create a controlled environment to study error robustness
2. Test if AQN can help models become robust to these errors
3. Avoid needing NPU hardware for initial experiments

### Analogy to Fake Quantization
Similar to **quantization simulation** (fake quantization):
- Real: GPU doesn't support NVFP4 natively
- Simulation: Add quant/dequant error to BF16 computation
- Our case: Add "NPU-like errors" to GPU computation

## Implementation

### Module: `verl/utils/hw_error_injection.py`

**Key Features:**
1. **Two injection points** (like quant_compute's fake quantization):
   - `injection_point='input'`: Inject error to operator INPUT (default)
     - Like fake quant: `y = operator(x + error)`
     - Uses `register_forward_pre_hook`
   - `injection_point='output'`: Inject error to operator OUTPUT
     - Like: `y = operator(x) + error`
     - Uses `register_forward_hook`

2. **Three error types:**
   - `relative_gaussian`: `error = randn() * |x| * scale` (HW relative error model)
   - `absolute_gaussian`: `error = randn() * scale` (fixed-point noise)
   - `systematic_bias`: `error = sign(x) * scale` (rounding bias)

3. **Phase control:**
   - Apply during rollout only, training only, or both

### Configuration
```yaml
trainer:
  hw_error_injection:
    enabled: true
    error_scale: 1e-5  # or 1e-4 for aggressive testing
    error_type: relative_gaussian
    injection_point: input
    apply_during: rollout
    target_modules: ['linear']  # or ['rmsnorm'], ['rmsnorm', 'linear']
```

### Command Line Usage
```bash
# Using test script (recommended)
bash scripts/test_hw_error_injection_a100.sh linear 8 1e-5

# Direct command
python3 -m verl.trainer.main_ppo \
    --config-name=ppo_trainer \
    trainer.hw_error_injection.enabled=True \
    trainer.hw_error_injection.error_scale=1e-5 \
    "trainer.hw_error_injection.target_modules=['linear']" \
    ...
```

## Operator Scope Analysis

### Qwen2.5-1.5B Architecture

| Operator Type | Count | FLOPs % | HW Difference Risk |
|---------------|-------|---------|-------------------|
| **Linear (MatMul)** | 197 | ~95% | High (accumulation order) |
| FlashAttention | 28 | ~3% | High (tiling differences) |
| RMSNorm | 57 | <1% | Medium |
| SiLU | 28 | <1% | Medium (approximation) |

### Target Module Options

| Scope | Target Modules | Hooks | FLOPs Coverage |
|-------|----------------|-------|----------------|
| **minimal** | `['rmsnorm']` | 57 | <1% |
| **mlp** | `['rmsnorm', 'down_proj']` | 85 | ~15% |
| **linear** | `['linear']` | 197 | **~95%** |
| **all** | `['rmsnorm', 'linear']` | 254 | ~96% |

**Recommendation**: Use `['linear']` for realistic simulation - Linear layers are where most computation (and potential HW error) occurs.

## Experiment Results

### E2: RMSNorm Only, Error Scale 1e-5 (Completed)

**Configuration:**
- Model: Qwen2.5-1.5B-Instruct
- Dataset: GSM8K (7473 train, 1319 test)
- GPUs: 8x A100-SXM4-80GB
- Error Scale: 1e-5 (relative_gaussian)
- Target Modules: RMSNorm only (57 hooks)
- Total Steps: 58

**Results:**
| Metric | Value |
|--------|-------|
| Initial OOD accuracy (step 0) | 8.26% |
| Final OOD accuracy (step 58) | **74.75%** |
| Final ID reward (step 58) | **80.31%** |

**Training Progression:**
| Step Range | ID Reward Range |
|------------|-----------------|
| Steps 1-10 | 11.56% → 67.81% |
| Steps 11-20 | 65.63% → 74.53% |
| Steps 21-30 | 75.94% → 77.81% |
| Steps 31-40 | 77.50% → 79.38% |
| Steps 41-50 | 77.81% → 80.31% |
| Steps 51-58 | 79.38% → 80.31% |

**Comparison with GPU Baseline:**
| Step | GPU Baseline | HW Error (1e-5) | Delta |
|------|--------------|-----------------|-------|
| 20 | 73.77% | 73.69% | **-0.08%** |

**Conclusion:**
- **1e-5 error scale with RMSNorm-only scope is too small to observe measurable impact**
- Training behaves identically to baseline
- Two possible causes:
  1. Error scale too small
  2. Operator scope too narrow (RMSNorm = <1% of FLOPs)

### E3: Linear Layers, Error Scale 1e-5 (Pending)

**Configuration:**
- Target Modules: All Linear layers (197 hooks, ~95% FLOPs)
- Error Scale: 1e-5

**Hypothesis:** With 95% FLOPs coverage, accumulated error should be more significant.

### E4: Linear Layers, Error Scale 1e-4 (Pending)

**Configuration:**
- Target Modules: All Linear layers (197 hooks, ~95% FLOPs)
- Error Scale: 1e-4 (10x larger)

**Hypothesis:** More aggressive error scale should show measurable accuracy degradation.

## Test Plan

### Phase 1: Operator Scope Impact (Current)

| Test | Target | Scale | Purpose |
|------|--------|-------|---------|
| E2 | RMSNorm | 1e-5 | Baseline narrow scope (DONE) |
| **E3** | **Linear** | **1e-5** | **Broad scope, same scale** |
| E3b | All | 1e-5 | Maximum coverage |

### Phase 2: Error Scale Impact

| Test | Target | Scale | Purpose |
|------|--------|-------|---------|
| E4 | Linear | 1e-4 | 10x larger error |
| E5 | Linear | 1e-3 | 100x larger (may destabilize) |

### Phase 3: AQN Robustness Test

| Test | HW Error | AQN | Purpose |
|------|----------|-----|---------|
| E6 | Linear 1e-4 | σ=0.05 | Test if AQN helps with HW errors |
| E7 | Linear 1e-4 | σ=0.025 | Milder AQN |

## How to Run Tests

### Test Script
```bash
# Run on GPU machine with A100s
cd /home/zheng/workspace/verl

# Linear layers with 1e-5 error (recommended first)
bash scripts/test_hw_error_injection_a100.sh linear 8 1e-5

# Linear layers with 1e-4 error (aggressive)
bash scripts/test_hw_error_injection_a100.sh linear 8 1e-4

# Run both scales in parallel (4 GPUs each)
bash scripts/test_hw_error_injection_a100.sh comparison
```

### Verify Hook Registration
Look for output like:
```
[HW Error] Registered 197 input hooks (scale=1e-05, type=relative_gaussian, targets=['linear'])
```

### Metrics to Collect
1. **OOD accuracy**: `val-core/openai/gsm8k/acc/mean@1`
2. **ID reward**: `critic/rewards/mean`
3. Compare at steps: 0, 20, 40, 60, 80, 100, 116

## Key Findings

### Finding 1: RMSNorm scope is insufficient
- 57 RMSNorm hooks cover <1% of FLOPs
- Error injection at this scope doesn't produce measurable impact
- **Action**: Expand to Linear layers (197 hooks, 95% FLOPs)

### Finding 2: 1e-5 error scale may be too conservative
- Relative error of ~8e-6 is within normal floating point variance
- Real GPU/NPU differences may be larger in specific operations
- **Action**: Test with 1e-4 scale

### Finding 3: Implementation is working correctly
- Hooks register and fire as expected
- Error injection statistics confirm expected magnitudes
- Hooks survive weight sync cycles

## References

- Related doc: [HW_HETEROGENEOUS_ROBUSTNESS_HYPOTHESIS.md](HW_HETEROGENEOUS_ROBUSTNESS_HYPOTHESIS.md)
- Related doc: [AQN_ACCURACY_ANALYSIS.md](AQN_ACCURACY_ANALYSIS.md)
- Implementation: `verl/utils/hw_error_injection.py`
- Test script: `scripts/test_hw_error_injection_a100.sh`
- quant_compute library: Fake quantization reference implementation
