# HW Heterogeneous Error Injection Experiments

**Date**: 2025-12-30
**Branch**: `feature/npu-aqn-test`
**Status**: In Progress

---

## Quick Reference: Environment & Reproduction

### A100 Remote Machine Access
```bash
# SSH to A100 machine
ssh root@90.90.102.18

# Enter verl container
docker exec -it verl-r3-test bash

# Working directory inside container
cd /home/z00637938/workspace/verl
```

### Model & Data Paths (Inside Container)
| Resource | Path |
|----------|------|
| **Model** | `/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| **Train Data** | `/data/z00637938/gsm8k/train.parquet` |
| **Val Data** | `/data/z00637938/gsm8k/test.parquet` |

### Training Scripts
| Script | Purpose | Location |
|--------|---------|----------|
| `run_gpu_baseline.sh` | GPU baseline (no error injection) | `scripts/run_gpu_baseline.sh` |
| `test_noisy_ops_a100.sh` | Operator-level noisy ops test (matmul only) | `scripts/test_noisy_ops_a100.sh` |
| `test_noisy_ops_all_ops.sh` | ALL operators noisy test (E5) | `scripts/test_noisy_ops_all_ops.sh` |
| `test_hw_error_injection_a100.sh` | Module-level error injection test | `scripts/test_hw_error_injection_a100.sh` |

### Log Locations (Inside Container)
| Experiment | Log File |
|------------|----------|
| E4b (1e-4) | `/tmp/noisy_ops_1e-4.log` |
| E4c (1e-3) | `/tmp/noisy_ops_1e-3.log` |
| E4d (1e-2) | `/tmp/noisy_ops_1e-2.log` (pending) |
| E5 (5e-2 FP4-realistic) | `/tmp/noisy_ops_5e-2.log` |

### Monitoring Commands
```bash
# Check if training is running
ssh root@90.90.102.18 "docker exec verl-r3-test pgrep -a python | grep trainer"

# Check training progress
ssh root@90.90.102.18 "docker exec verl-r3-test grep 'Training Progress' /tmp/noisy_ops_1e-4.log | tail -3"

# Check validation results
ssh root@90.90.102.18 "docker exec verl-r3-test grep val-core /tmp/noisy_ops_1e-4.log"

# Full log tail
ssh root@90.90.102.18 "docker exec verl-r3-test tail -100 /tmp/noisy_ops_1e-4.log"
```

### Running a New Test
```bash
# SSH and enter container
ssh root@90.90.102.18
docker exec -it verl-r3-test bash
cd /home/z00637938/workspace/verl

# Run with custom model/data paths
MODEL_PATH=/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
TRAIN_DATA=/data/z00637938/gsm8k/train.parquet \
VAL_DATA=/data/z00637938/gsm8k/test.parquet \
nohup bash scripts/test_noisy_ops_a100.sh 1e-3 8 > /tmp/noisy_ops_1e-3.log 2>&1 &
```

---

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
| 6 | Fix torch.compile conflict | Use `enforce_eager=True` for vLLM | **DONE** |
| 7 | Test E4b | Operator-level 1e-4 scale test | **In Progress** |
| 8 | Compare with module-level | Does operator-level show more degradation? | Pending |

### Finding 6: torch.compile Conflict and Solution

**Problem:** vLLM V1 uses `torch.compile` by default, which is incompatible with custom `autograd.Function` classes that use `@torch.compiler.disable`.

```
torch._dynamo.exc.Unsupported: Skip inlining `torch.compiler.disable()`d function
```

**Root Cause:**
- Our `NoisyMatMul` autograd function uses `@torch.compiler.disable` to prevent tracing
- torch.compile cannot trace through disabled functions
- vLLM's model compilation fails when it encounters our patched `F.linear`

**Solution:** Set `enforce_eager=True` in vLLM config to disable torch.compile:

```yaml
actor_rollout_ref:
  rollout:
    enforce_eager: True  # Disables torch.compile in vLLM
```

**Comparison of Approaches:**

| Approach | How it works | torch.compile compatible |
|----------|--------------|--------------------------|
| Module-level hooks (`hw_error_injection.py`) | `register_forward_pre_hook` | ✅ Yes - hooks don't interfere |
| Operator-level patching (`noisy_ops.py`) | Monkey-patch `F.linear` with custom autograd | ❌ No - needs `enforce_eager=True` |

### Configuration

**Environment Variables (Required for Ray workers):**
```bash
export VERL_NOISY_OPS_ENABLED=1
export VERL_NOISY_OPS_SCALE=1e-4
export VERL_NOISY_OPS_TYPE=relative_gaussian
```

**Hydra config:**
```yaml
# Rollout config - MUST have enforce_eager=True for noisy_ops compatibility
actor_rollout_ref:
  rollout:
    enforce_eager: True  # Critical: disables torch.compile in vLLM

# Trainer config
trainer:
  noisy_ops:
    enabled: true
    error_scale: 1e-4
    error_type: relative_gaussian
```

**Important:**
- Use environment variables to ensure noisy ops is enabled in ALL processes (including Ray workers)
- The env vars are read at `import verl` time via `_auto_enable_from_env()`
- `enforce_eager=True` is required to disable torch.compile in vLLM

### Test Script

```bash
# Run operator-level noisy ops test on A100
# This script sets env vars and enforce_eager=True automatically
bash scripts/test_noisy_ops_a100.sh 1e-4 8
```

**Expected Output:**
```
[NoisyOps] Auto-enabled from environment: scale=0.0001, type=relative_gaussian
[NoisyOps] Enabled: scale=0.0001, type=relative_gaussian, affects forward+backward passes globally
```

You should see these messages from ALL Ray workers (including vLLM workers), confirming noisy ops is active everywhere.

### Behavior Summary

| Phase | Module-level (hw_error_injection.py) | Operator-level (noisy_ops.py) |
|-------|--------------------------------------|-------------------------------|
| **Rollout Forward** | ✅ Yes (hook on Linear) | ✅ Yes (patched F.linear) |
| **Rollout Backward** | ❌ No | ✅ **Yes** |
| **Training Forward** | ✅ Yes | ✅ Yes |
| **Training Backward** | ❌ No | ✅ **Yes** |
| **Gradient noise** | ❌ None | ✅ **Present** |
| **vLLM compatibility** | ✅ Works with torch.compile | ⚠️ Needs enforce_eager=True |

### E4b: Operator-Level, Error Scale 1e-4 (Completed)

**Configuration:**
- Model: Qwen2.5-1.5B-Instruct
- Dataset: GSM8K (7473 train, 1319 test)
- GPUs: 8x A100-SXM4-80GB
- Error Scale: 1e-4 (relative_gaussian)
- Implementation: `verl/utils/noisy_ops.py` (operator-level)
- Noise in: **ALL phases** (rollout + training, forward + backward)
- Total Steps: 116 (2 epochs)
- Config: `enforce_eager=True` to disable torch.compile

**Results:**
| Step | OOD Accuracy | ID Reward |
|------|--------------|-----------|
| 0 | 7.88% | - |
| 20 | 73.01% | 75.0% |
| 40 | 75.82% | 78.6% |
| 60 | 77.41% | 81.9% |
| 80 | 77.33% | 79.4% |
| 100 | 77.18% | 81.7% |
| 116 | **77.33%** | **82.5%** |

**Comparison with GPU Baseline:**
| Metric | GPU Baseline | E4b (Noisy ops 1e-4) | Delta |
|--------|--------------|---------------------|-------|
| Final OOD accuracy | **76.88%** | **77.33%** | **+0.45%** |

**Conclusion:**
- **1e-4 error scale with operator-level injection (ALL phases) does NOT degrade performance**
- Final accuracy is actually slightly better than baseline (+0.45%)
- This suggests:
  1. Run-to-run variance may be larger than the noise effect
  2. The noise may act as regularization (similar to dropout)
  3. 1e-4 scale is still too small to cause observable degradation
- **Action**: Test with 1e-3 scale to see if degradation appears

### E4c: Operator-Level, Error Scale 1e-3 (Completed)

**Configuration:**
- Same as E4b, but with 10x larger error scale (1e-3)
- All other settings identical

**Results:**
| Step | OOD Accuracy | ID Reward |
|------|--------------|-----------|
| 0 | 8.49% | - |
| 20 | 73.84% | 74.8% |
| 40 | 74.22% | 77.2% |
| 60 | 75.74% | 79.8% |
| 80 | 75.06% | 80.3% |
| 100 | 76.42% | 83.0% |
| 116 | **77.18%** | **84.5%** |

**Comparison with E4b (1e-4) and GPU Baseline:**
| Metric | GPU Baseline | E4b (1e-4) | E4c (1e-3) |
|--------|--------------|------------|------------|
| Final OOD | 76.88% | 77.33% | **77.18%** |
| Delta vs baseline | - | +0.45% | **+0.30%** |

**Progressive Degradation Analysis:**
| Step | E4b (1e-4) | E4c (1e-3) | Delta |
|------|------------|------------|-------|
| 40 | 75.82% | 74.22% | **-1.60%** |
| 60 | 77.41% | 75.74% | **-1.67%** |
| 80 | 77.33% | 75.06% | **-2.27%** (max) |
| 100 | 77.18% | 76.42% | -0.76% |
| 116 | 77.33% | 77.18% | **-0.15%** |

**Key Observations:**
1. **Mid-training degradation**: E4c shows ~2% lower accuracy than E4b during steps 40-80
2. **Recovery by end**: The gap narrows to only 0.15% by step 116
3. **Above baseline**: Both E4b and E4c end above the GPU baseline (76.88%)
4. **Regularization effect**: The noise may act as regularization (similar to dropout)

**Conclusion:**
- **1e-3 noise scale (10x larger) still does NOT cause significant final degradation**
- Shows temporary mid-training impact (~2%) but recovers
- Suggests models are robust to this level of computational noise
- **Action**: Consider 1e-2 scale or adding noise to more operators (bmm, softmax, activations)

### Finding 7: Training Robustness to Computational Noise

The experiments reveal that transformer models are surprisingly robust to relative computational noise:

| Noise Scale | Coverage | Final OOD Accuracy | vs Baseline |
|-------------|----------|-------------------|-------------|
| 1e-5 | Module-level Linear | 76.35% | -0.53% |
| 1e-4 | Operator-level (all matmul, fwd+bwd) | 77.33% | **+0.45%** |
| 1e-3 | Operator-level (all matmul, fwd+bwd) | 77.18% | **+0.30%** |

**Implications:**
1. Relative noise up to 0.1% (1e-3) in all matmul ops doesn't significantly harm convergence
2. The noise may provide regularization benefits (like dropout/noise injection)
3. To simulate HW errors that DO cause degradation, may need:
   - Even higher noise scales (1e-2 or higher)
   - Noise in additional operators (softmax, activations)
   - Systematic bias rather than Gaussian noise

---

## Planned Experiments

### E4d: Operator-Level, Error Scale 1e-2 (Pending)

**Configuration:**
- Same as E4b/E4c, but with 100x larger error scale (1e-2 = 1% relative error)
- May destabilize training

**Purpose:** Determine if 1% relative error causes significant degradation.

**Command:**
```bash
ssh root@90.90.102.18
docker exec -it verl-r3-test bash
cd /home/z00637938/workspace/verl

MODEL_PATH=/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
TRAIN_DATA=/data/z00637938/gsm8k/train.parquet \
VAL_DATA=/data/z00637938/gsm8k/test.parquet \
nohup bash scripts/test_noisy_ops_a100.sh 1e-2 8 > /tmp/noisy_ops_1e-2.log 2>&1 &
```

**Monitor:**
```bash
ssh root@90.90.102.18 "docker exec verl-r3-test grep val-core /tmp/noisy_ops_1e-2.log"
```

### E5: FP4-Realistic Error Scale (5e-2) - Noise Only Baseline (Running)

**Rationale:**
Analysis of `QeRL/llm-compressor` and `quant_compute` revealed:

1. **NVFP4 quantization error is ~5-15%**, not 0.1% (1e-3)
2. **Only Linear layers are quantized** in QeRL (ignore lm_head)
3. **Softmax, activations, LayerNorm are NOT quantized** - kept in BF16

Therefore, to properly simulate QeRL NVFP4:
- Use **5e-2 (5%)** error scale
- Apply to **matmul/linear ONLY** (not softmax, silu, gelu, layer_norm)

**Configuration:**
```bash
# Environment variables - FP4-realistic simulation
export VERL_NOISY_OPS_ENABLED=1
export VERL_NOISY_OPS_SCALE=5e-2  # 5% - matches FP4 error level
export VERL_NOISY_OPS_TYPE=relative_gaussian
# DO NOT set VERL_NOISY_OPS_ALL_OPS=1 (only matmul/linear)
# NO AQN - this is baseline under noise
```

**Command:**
```bash
ssh root@90.90.102.18
docker exec -it verl-r3-test bash
cd /home/z00637938/workspace/verl

MODEL_PATH=/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
TRAIN_DATA=/data/z00637938/gsm8k/train.parquet \
VAL_DATA=/data/z00637938/gsm8k/test.parquet \
nohup bash scripts/test_noisy_ops_a100.sh 5e-2 8 > /tmp/noisy_ops_5e-2.log 2>&1 &
```

**Monitor:**
```bash
ssh root@90.90.102.18 "docker exec verl-r3-test grep val-core /tmp/noisy_ops_5e-2.log"
```

**Purpose:** Establish baseline for "training on noisy HW without mitigation". Shows how much FP4-level noise degrades accuracy.

**Status:** ✅ **COMPLETE** (2025-12-30)

**Final Results:**
| Step | OOD Accuracy | vs Baseline |
|------|--------------|-------------|
| 0 | 9.25% | -15.47% |
| 20 | 61.64% | -12.13% |
| 40 | 66.34% | -9.10% |
| 60 | 66.19% | -8.03% |
| 80 | 68.08% | -6.60% |
| 100 | 69.14% | -8.34% |
| **116** | **68.16%** | **-8.72%** |

**Conclusion:** 5% noise (FP4-realistic) causes **8.72% accuracy degradation** compared to GPU baseline (76.88%). This establishes the baseline that E5a (with AQN) needs to improve upon.

**Note:** ALL_OPS mode implementation is available (`VERL_NOISY_OPS_ALL_OPS=1`) but NOT used for E5 since QeRL only quantizes Linear layers.

### E5a: FP4-Realistic Error Scale (5e-2) + AQN (Pending)

**Purpose:** Test if AQN can mitigate noise degradation. This is the key experiment to validate the HW heterogeneous robustness hypothesis.

**Configuration:**
```bash
# Same noise as E5
export VERL_NOISY_OPS_ENABLED=1
export VERL_NOISY_OPS_SCALE=5e-2
export VERL_NOISY_OPS_TYPE=relative_gaussian

# PLUS AQN (QeRL original parameters)
# trainer.noise_injection.enabled=true
# trainer.noise_injection.sigma_start=0.05
# trainer.noise_injection.sigma_end=0.0005
```

**Hypothesis:**
- E5a OOD accuracy > E5 OOD accuracy (AQN helps model learn despite noise)
- E5a checkpoint passes robustness eval (clean ≈ noisy accuracy)

**Expected Experimental Flow:**
```
GPU Baseline: 76.88%
       │
       ▼
E5 (Noise only): 68.16% ◄── Degradation from 5% noise (-8.72%)
       │
       ▼
E5a (Noise + AQN): > 68.16%? ◄── AQN mitigates degradation
       │
       ▼
Robustness eval on E5a:
  Clean ≈ Noisy = Model is ROBUST ✓
```

**Success Criteria:**
1. E5a final OOD > 68.16% (E5) - AQN provides benefit under noise
2. E5a robustness eval: |Clean - Noisy| < 1% (model is robust)

**If Successful:** This proves AQN works for **HW heterogeneous errors**, not just quantization - a novel finding beyond QeRL's original scope.

**Command:**
```bash
ssh root@90.90.102.18
docker exec -it verl-r3-test bash
cd /home/z00637938/workspace/verl

MODEL_PATH=/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
TRAIN_DATA=/data/z00637938/gsm8k/train.parquet \
VAL_DATA=/data/z00637938/gsm8k/test.parquet \
nohup bash scripts/test_noisy_ops_aqn.sh 5e-2 8 > /tmp/noisy_ops_aqn_5e-2.log 2>&1 &
```

**Monitor:**
```bash
ssh root@90.90.102.18 "docker exec verl-r3-test grep val-core /tmp/noisy_ops_aqn_5e-2.log"
```

**Status:** 🔄 Running (started 2025-12-31)

**AQN Configuration Verified:**
| Setting | Value | Status |
|---------|-------|--------|
| Model type | Dense (auto-detected) | ✅ |
| Target modules | ALL RMSNorm (57 layers) | ✅ |
| Warmup phase | Steps 0-11 (sigma=0) | ✅ |
| Active phase | Steps 12+ (sigma=0.05→0.0005) | ✅ |
| Total stages | 10 (9 active + 1 warmup) | ✅ |

**Early Progress (Step 16):**
- AQN activated at step 12 with sigma=0.05
- Training scores progressing: step 15 = 47.0%, step 16 = 41.4%
- Step 20 validation pending (~4 mins)

**Implementation Notes:**
- `get_sigma_by_step()` returns sigma=0 for interval_id=0 (warmup)
- With 116 total steps and 10 stages: steps_per_interval = 11.6
- AQN applies noise to RMSNorm weights after each weight sync in vLLM rollout

### E6: Systematic Bias Instead of Gaussian (Pending)

**Rationale:**
Real HW differences often have systematic patterns (e.g., consistent rounding direction) rather than random Gaussian noise.

**Error Types to Test:**
| Type | Formula | Simulates |
|------|---------|-----------|
| `relative_gaussian` | `randn() * |x| * scale` | Random HW variance |
| `systematic_bias` | `sign(x) * |x| * scale` | Consistent rounding bias |
| `truncation` | `floor(x / scale) * scale` | Truncation error |

### E7: AQN + Noisy Ops (Pending)

**Prerequisites:**
- E4d or E5 shows significant degradation (>2-3%)

**Purpose:**
Test if AQN (Adaptive Quantization Noise) during training can improve robustness to HW errors during inference.

**Configuration:**
```yaml
trainer:
  aqn:
    enabled: true
    sigma: 0.05
  noisy_ops:
    enabled: true
    error_scale: <scale that showed degradation>
```

---

## QeRL/quant_compute Quantization Analysis

### Key Finding: Error Scale Mismatch

Our initial experiments (E4b, E4c) used error scales of 1e-4 and 1e-3, which are **50-150x smaller** than actual FP4 quantization error.

| Format | Typical Relative Error | Our Test Scale | Gap |
|--------|----------------------|----------------|-----|
| **NVFP4** (E4M3 + E2M1) | **5-15%** | 0.1% (1e-3) | ~100x |
| **MXFP4** (E8 + E2M1) | **5-15%** | 0.1% (1e-3) | ~100x |
| **HiF4** (3-level scaling) | **3-10%** | 0.1% (1e-3) | ~50x |

**Conclusion**: To simulate FP4-level degradation, use **5e-2 (5%)** error scale.

### QeRL Layer Coverage Analysis

Analysis of `QeRL/llm-compressor` shows which layers are quantized:

**Quantized (FP4 error applied):**
```python
# From QeRL/llm-compressor/quantize_nvfp4.py
recipe = QuantizationModifier(targets="Linear", scheme="NVFP4A16", ignore=["lm_head"])
```

| Layer Type | Quantized | Error Level |
|------------|-----------|-------------|
| Linear projections (q/k/v/o_proj, gate/up/down_proj) | ✅ Yes | ~5-15% |

**NOT Quantized (kept in BF16):**

| Layer Type | Ignore Pattern | Reason |
|------------|----------------|--------|
| **lm_head** | `ignore=["lm_head"]` | Output precision critical |
| **Embeddings** | Not in `targets="Linear"` | Input precision critical |
| **RMSNorm/LayerNorm** | Not Linear | Normalization precision |
| **Softmax** | Not quantized | Attention precision |
| **SiLU/GELU** | Not quantized | Activation functions |

### Comparison: noisy_ops vs QeRL

| Operator | QeRL NVFP4 | noisy_ops (matmul-only) | noisy_ops (ALL_OPS) |
|----------|------------|-------------------------|---------------------|
| **Linear/matmul** | ✅ ~5-15% | ✅ Configurable | ✅ Configurable |
| **lm_head** | ❌ Ignored | ✅ Included | ✅ Included |
| **RMSNorm/LayerNorm** | ❌ No | ❌ No | ✅ **Over-coverage** |
| **Softmax** | ❌ No | ❌ No | ✅ **Over-coverage** |
| **SiLU/GELU** | ❌ No | ❌ No | ✅ **Over-coverage** |

### Recommended Test Configuration

To properly simulate QeRL NVFP4 quantization:

1. **Error scale**: **5e-2 (5%)** - matches FP4 quantization error
2. **Operators**: **matmul/linear ONLY** (not softmax, silu, gelu, layer_norm)
3. **all_ops_mode**: **False** (default)

```bash
# Correct simulation of QeRL NVFP4
export VERL_NOISY_OPS_ENABLED=1
export VERL_NOISY_OPS_SCALE=5e-2  # 5% - matches FP4 error
export VERL_NOISY_OPS_TYPE=relative_gaussian
# DO NOT set VERL_NOISY_OPS_ALL_OPS=1
```

### Methodology Note: Gaussian Noise vs Quantization Error

**Important:** Our noise injection differs fundamentally from real quantization error:

| Property | Real Quantization Error | Our Gaussian Noise |
|----------|------------------------|-------------------|
| **Deterministic** | ✅ Same input → same error | ❌ Random each forward pass |
| **Bounded** | ✅ Limited by quantization step | ❌ Unbounded (Gaussian tail) |
| **Systematic bias** | ✅ Often present (truncation/rounding) | ❌ Zero mean, no bias |
| **Learnable pattern** | ✅ Model can adapt to consistent errors | ❌ Cannot learn random noise |
| **Train-inference consistency** | ✅ Same error pattern | ❌ Different noise each time |

**Implications for our experiments:**

1. **Random noise is HARDER than real quantization:**
   - Model cannot learn to compensate for consistent error patterns
   - Gradients are randomly perturbed (optimizer can't work around it)
   - Each forward pass sees completely different errors

2. **Real quantization is CONSISTENT:**
   - `quant(x)` always produces the same output for the same `x`
   - Model can naturally adapt to the fixed error pattern during training
   - Inference error matches training error exactly

3. **Conservative test interpretation:**
   - If a model trained with random Gaussian noise shows robustness, it is likely **over-robust** compared to what's needed for deterministic quantization
   - Random noise acts as a **stronger regularizer** than deterministic quantization error
   - **Success with Gaussian noise implies success with real FP4 quantization**

**Conclusion:** Our Gaussian noise injection is a conservative upper bound on quantization difficulty. Results from E5/E5a experiments should transfer (or exceed) to real NVFP4 quantization scenarios.

---

## Experiment Summary Table

| Test | Error Scale | Operator Coverage | AQN | Status | Final OOD | vs Baseline |
|------|-------------|-------------------|-----|--------|-----------|-------------|
| Baseline | - | - | No | Done | 76.88% | - |
| E2 | 1e-5 | Module-level RMSNorm | No | Done | 74.75% | -2.13% |
| E3 | 1e-5 | Module-level Linear | No | Done | 76.35% | -0.53% |
| E4b | 1e-4 | All matmul (fwd+bwd) | No | Done | 77.33% | +0.45% |
| E4c | 1e-3 | All matmul (fwd+bwd) | No | Done | 77.18% | +0.30% |
| E4d | 1e-2 | All matmul (fwd+bwd) | No | Pending | - | - |
| **E5** | **5e-2** | **matmul-only (FP4-realistic)** | **No** | **Done** | **68.16%** | **-8.72%** |
| **E5a** | **5e-2** | **matmul-only (FP4-realistic)** | **Yes** | **Running** | - | - |
| E6 | TBD | Systematic bias | No | Pending | - | - |

## Robustness Evaluation Methodology

### The Robustness Question

When training with noisy ops, there are two different measurements:

| Measurement | Training | Evaluation | What it Shows |
|-------------|----------|------------|---------------|
| **Performance under noise** | Noisy | Noisy | Accuracy when deployed on noisy HW |
| **Robustness** | Noisy | **Clean** | Model's learned noise tolerance |

### Why This Matters

A **robust** model should:
1. Learn from noisy training data
2. **Maintain accuracy when noise is removed**

This is the true test of robustness - if a model trained with noise maintains performance on clean hardware, it has learned to generalize despite the noise.

### Clean Evaluation Scripts

Two scripts are provided for robustness evaluation:

**1. Python script for single evaluation:**
```bash
# Clean evaluation (no noise)
VERL_NOISY_OPS_ENABLED=0 python scripts/clean_eval_checkpoint.py \
    --model_path /path/to/checkpoint \
    --data_path /data/gsm8k/test.parquet \
    --n_samples 5

# Noisy evaluation (for comparison)
VERL_NOISY_OPS_ENABLED=1 VERL_NOISY_OPS_SCALE=5e-2 python scripts/clean_eval_checkpoint.py ...
```

**2. Bash script for robustness comparison:**
```bash
# Runs BOTH clean and noisy evaluation, computes robustness delta
bash scripts/eval_robustness.sh \
    /path/to/checkpoint \
    /data/gsm8k/test.parquet \
    5e-2  # error scale for noisy eval
```

### Interpreting Results

| Clean - Noisy Delta | Interpretation |
|---------------------|----------------|
| < 1% | ✓ Model is **ROBUST** |
| 1-3% | ⚠ Model is **MODERATELY ROBUST** |
| > 3% | ✗ Model is **NOT ROBUST** |

### Expected Outcomes by Training Type

| Training Method | Expected Clean Accuracy | Expected Robustness |
|-----------------|------------------------|---------------------|
| Baseline (no noise) | High | Low (sensitive to noise) |
| Noisy training | Lower | **High** (tolerant to noise) |
| AQN + Noisy training | Medium | **Very High** (best robustness) |

### How to Use in Experiments

**Important:** Robustness evaluation is most meaningful for **E5a** (trained with AQN), not E5.

- **E5 checkpoint**: Trained with noise, no AQN → expected to be NOT robust
- **E5a checkpoint**: Trained with noise + AQN → expected to BE robust

```bash
# After E5a completes, evaluate the checkpoint for robustness
CHECKPOINT=/data/checkpoints/noisy_ops_aqn_5e-2/global_step_116/actor
DATA=/data/gsm8k/test.parquet

bash scripts/eval_robustness.sh ${CHECKPOINT} ${DATA} 5e-2
```

This will produce:
- `results_clean.json` - accuracy without noise (simulates clean HW deployment)
- `results_noisy.json` - accuracy with noise (simulates noisy HW deployment)
- Robustness summary printed to console

### Full Experimental Validation

To fully validate the HW heterogeneous robustness hypothesis:

```bash
# 1. E5 (no AQN) - expect degradation, NOT robust
bash scripts/eval_robustness.sh /path/to/E5/checkpoint /data/gsm8k/test.parquet 5e-2
# Expected: Clean >> Noisy (NOT robust)

# 2. E5a (with AQN) - expect better accuracy, IS robust
bash scripts/eval_robustness.sh /path/to/E5a/checkpoint /data/gsm8k/test.parquet 5e-2
# Expected: Clean ≈ Noisy (ROBUST) AND Clean > E5's noisy accuracy
```

---

## References

### Related Documentation
- **[NOISE_INJECTION_LITERATURE_REVIEW.md](NOISE_INJECTION_LITERATURE_REVIEW.md)** - Survey of related work on noise injection for robustness (RL, LLM, Embodied AI, Hardware)
- [HW_HETEROGENEOUS_ROBUSTNESS_HYPOTHESIS.md](HW_HETEROGENEOUS_ROBUSTNESS_HYPOTHESIS.md)
- [AQN_ACCURACY_ANALYSIS.md](AQN_ACCURACY_ANALYSIS.md)
- Module-level implementation: `verl/utils/hw_error_injection.py`
- Operator-level implementation: `verl/utils/noisy_ops.py`
- GPU baseline script: `scripts/run_gpu_baseline.sh`
- Module-level test script: `scripts/test_hw_error_injection_a100.sh`
- Operator-level test script (matmul only): `scripts/test_noisy_ops_a100.sh`
- Operator-level test script (ALL ops): `scripts/test_noisy_ops_all_ops.sh`
- Robustness evaluation script: `scripts/eval_robustness.sh`
- Clean evaluation Python script: `scripts/clean_eval_checkpoint.py`
- quant_compute library: Fake quantization reference implementation
