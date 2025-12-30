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

### E3: Linear Layers, Error Scale 1e-5 (Completed)

**Configuration:**
- Model: Qwen2.5-1.5B-Instruct
- Dataset: GSM8K (7473 train, 1319 test)
- GPUs: 8x A100-SXM4-80GB
- Error Scale: 1e-5 (relative_gaussian)
- Target Modules: Linear layers (112 vLLM Linear hooks, ~95% FLOPs)
- Total Steps: 116 (2 epochs)
- Config: Matches `run_gpu_baseline.sh` exactly (batch=128, lr=5e-7, n=5)

**Results:**
| Step | OOD Accuracy |
|------|--------------|
| 0 | 8.19% |
| 20 | 73.54% |
| 40 | 74.37% |
| 60 | 76.27% |
| 80 | 76.27% |
| 100 | 76.65% |
| 116 | **76.35%** |

**Comparison with GPU Baseline:**
| Metric | GPU Baseline | HW Error (Linear 1e-5) | Delta |
|--------|--------------|------------------------|-------|
| Final OOD accuracy | **76.88%** | **76.35%** | **-0.53%** |
| Total Steps | 116 | 116 | Same |

**Conclusion:**
- **1e-5 error scale with Linear layers (112 hooks, ~95% FLOPs) still shows minimal impact** (-0.53%)
- Training converges normally with similar trajectory to baseline
- Error injection is working (verified hook registration and injection messages)
- The 1e-5 relative error is too small to cause observable degradation even with high FLOPs coverage

**Key Insight:**
The error scale matters more than operator coverage. Need to test with larger scales (1e-4, 1e-3).

### E4: Linear Layers, Error Scale 1e-4 (Pending)

**Configuration:**
- Target Modules: All Linear layers (197 hooks, ~95% FLOPs)
- Error Scale: 1e-4 (10x larger)

**Hypothesis:** More aggressive error scale should show measurable accuracy degradation.

## Test Plan

### Phase 1: Operator Scope Impact (Current)

| Test | Target | Scale | Purpose | Status |
|------|--------|-------|---------|--------|
| E2 | RMSNorm | 1e-5 | Baseline narrow scope | **DONE** |
| E3 | Linear | 1e-5 | Broad scope, same scale | **DONE** (-0.53% vs baseline) |
| E3b | All | 1e-5 | Maximum coverage | Pending |

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

### Finding 2: 1e-5 error scale is too conservative
- Relative error of ~8e-6 is within normal floating point variance
- Even with 95% FLOPs coverage (Linear layers), only -0.53% accuracy drop
- Real GPU/NPU differences may be larger in specific operations
- **Action**: Test with 1e-4 scale

### Finding 3: Implementation is working correctly
- Hooks register and fire as expected
- Error injection statistics confirm expected magnitudes
- Hooks survive weight sync cycles

### Finding 4: Module-level injection has fundamental limitations
The current approach using PyTorch module hooks has critical limitations:

| Limitation | Current Approach | Real HW Errors |
|------------|------------------|----------------|
| **Scope** | Module forward only | All operators |
| **Backward pass** | Not affected | Gradients also have errors |
| **Phase control** | Rollout vs training | Always present |
| **Granularity** | Layer boundaries | Operator level (MatMul, etc.) |

**Why this matters:**
1. Real HW errors occur at **operator level** (MatMul kernel), not layer level
2. Errors affect **both forward AND backward** passes
3. Training gradients should also be "noisy" - this affects convergence
4. No artificial phase distinction - errors happen everywhere

### Finding 5: Ray workers require environment variable approach
Monkey-patching `torch.matmul` in the main process doesn't propagate to Ray workers because they are separate processes. The solution is to use environment variables that are inherited by child processes:

1. Set `VERL_NOISY_OPS_ENABLED=1` before launching training
2. `verl/__init__.py` imports `noisy_ops` module
3. `noisy_ops._auto_enable_from_env()` is called at import time
4. Each Ray worker will auto-enable noisy ops when it imports verl

## Next Steps: Operator-Level Error Injection

### Problem with Current Approach

```
Current (Module-level):
┌─────────────┐     ┌─────────────┐
│ Linear Layer│ ──► │ Forward Hook│ ──► Error injected
└─────────────┘     └─────────────┘
                    (only forward, only rollout)

Desired (Operator-level):
┌─────────────┐     ┌─────────────┐
│ torch.matmul│ ──► │  Wrapper    │ ──► Error in forward + backward
└─────────────┘     └─────────────┘
                    (all phases, all operations)
```

### Proposed Implementation: Custom Autograd Function

Replace `torch.matmul` / `F.linear` with a noisy version:

```python
class NoisyMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        result = torch.matmul(a, b)
        # Inject error in forward
        error = torch.randn_like(result) * result.abs() * ERROR_SCALE
        ctx.save_for_backward(a, b)
        return result + error

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = torch.matmul(grad_output, b.transpose(-1, -2))
        grad_b = torch.matmul(a.transpose(-1, -2), grad_output)
        # Inject error in backward too
        grad_a += torch.randn_like(grad_a) * grad_a.abs() * ERROR_SCALE
        grad_b += torch.randn_like(grad_b) * grad_b.abs() * ERROR_SCALE
        return grad_a, grad_b
```

### Implementation Plan

| Step | Task | Description | Status |
|------|------|-------------|--------|
| 1 | Create `verl/utils/noisy_ops.py` | Noisy autograd functions for MatMul | **DONE** |
| 2 | Add global enable/disable | Context manager for activating noisy ops | **DONE** |
| 3 | Monkey-patch torch ops | Replace `F.linear`, `torch.matmul` globally | **DONE** |
| 4 | Integrate into trainer | Add config, phase tracking, summary | **DONE** |
| 5 | Fix Ray worker isolation | Enable via env vars for all processes | **DONE** |
| 6 | Test E4b | Operator-level 1e-4 scale test | **In Progress** |
| 7 | Compare with module-level | Does operator-level show more degradation? | Pending |

### Configuration

**Environment Variables (Recommended for Ray workers):**
```bash
export VERL_NOISY_OPS_ENABLED=1
export VERL_NOISY_OPS_SCALE=1e-4
export VERL_NOISY_OPS_TYPE=relative_gaussian
```

**Hydra config (main process only):**
```yaml
# In trainer config or command line:
trainer:
  noisy_ops:
    enabled: true
    error_scale: 1e-4        # 10x larger than module-level test
    error_type: relative_gaussian
```

**Important:** Use environment variables to ensure noisy ops is enabled in ALL processes (including Ray workers). The env vars are read at `import verl` time via `_auto_enable_from_env()`.

### Test Script

```bash
# Run operator-level noisy ops test on A100
bash scripts/test_noisy_ops_a100.sh 1e-4 8
```

**Expected Output:** You should see `[NoisyOps] Auto-enabled from environment:` messages from Ray workers, confirming noisy ops is active in all processes.

### Expected Behavior Differences

| Aspect | Module-level (current) | Operator-level (planned) |
|--------|------------------------|--------------------------|
| Forward errors | Yes | Yes |
| Backward errors | No | **Yes** |
| Gradient noise | None | **Present** |
| Training stability | Normal | May be affected |
| Error accumulation | Per-layer | **Per-operation** |

## References

- Related doc: [HW_HETEROGENEOUS_ROBUSTNESS_HYPOTHESIS.md](HW_HETEROGENEOUS_ROBUSTNESS_HYPOTHESIS.md)
- Related doc: [AQN_ACCURACY_ANALYSIS.md](AQN_ACCURACY_ANALYSIS.md)
- Implementation: `verl/utils/hw_error_injection.py`
- Test script: `scripts/test_hw_error_injection_a100.sh`
- quant_compute library: Fake quantization reference implementation
