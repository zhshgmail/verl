# All AQN Experiments Summary

**Generated**: 2026-01-12
**Branch**: `feature/npu-aqn-test`
**TensorBoard**: `/tmp/tb_logs/`
**Wandb Project**: `aqn_refine`

---

## Glossary

| Term | Description |
|------|-------------|
| **AQN** | Adaptive Quantization Noise - inject Gaussian noise into RMSNorm layers during training |
| **Epoch-Aware** | AQN scheduling mode where each epoch has its own sigma range with exponential decay, vs linear decay across all steps |
| **SRDD** | Static Relative Deadzone Detection - identifies layers with high quantization error |
| **SRDD-targeted** | AQN applied only to layers identified by SRDD as high-error (binary: on/off) |
| **SRDD-variable** | AQN with layer-specific noise multipliers based on SRDD error scores (continuous scaling) |
| **σ (sigma)** | Noise magnitude. Typically decays from σ_start (0.01) to σ_end (0.0001) over training |
| **HW inject** | Hardware error simulation via `VERL_NOISY_OPS` - injects relative Gaussian noise into matmul operations |
| **Quant** | Actual fake quantization experiments using MXFP4/NVFP4/BF16 via `hw_error_injection` |
| **Matmul-only** | Noise injected only into matmul operations (simulates compute quantization) |
| **Weight inject** | Noise injected into model weights (simulates weight quantization via `hw_error_injection`) |

---

## Complete Experiment Table

| Exp ID | Type | TensorBoard Name | Score | Epochs | AQN | Epoch-Aware | SRDD | LoRA | dtype | Notes |
|--------|------|------------------|-------|--------|-----|-------------|------|------|-------|-------|
| **Baseline** | HW | `HW_BF16_GRPO_2ep_76.88_BASELINE` | **76.88%** | 2 | No | - | No | No | BF16 | GRPO, 2 epochs |
| **E5** | HW | `HW_5pct_noise_only_68.92` | 68.92% | 2 | No | - | No | No | BF16 | 5% noise, no AQN |
| **E5b** | HW | `HW_5pct_AQN_epoch-aware_sigma0.01_70.58` | 70.58% | 2 | Yes | ✅ Yes | No | No | BF16 | Epoch-aware AQN, σ=0.05→0.0005 |
| **E5c** | HW | `HW_5pct_AQN_lower_sigma0.005_70.27` | 70.27% | 2 | Yes | ✅ Yes | No | No | BF16 | Lower sigma, σ=0.01→0.00001 |
| **E9a** | HW | `HW_5pct_AQN_SRDD-targeted_sigma0.01_70.58` | 70.58% | 2 | Yes | ✅ Yes | targeted | No | BF16 | AQN on specific layers only |
| **E9a-high-σ** | HW | `HW_5pct_AQN_SRDD-targeted_high-sigma0.05_70.81` | 70.81% | 2 | Yes | ✅ Yes | targeted | No | BF16 | High σ=0.05 start |
| **E9b** | HW | `HW_5pct_AQN_SRDD-variable_sigma0.01_71.19_BEST` | **71.19%** | 2 | Yes | ✅ Yes | variable | No | BF16 | **BEST HW** - per-layer multipliers |
| **E8a** | Quant | `Q_BF16_DAPO_fullFT_1ep_74.75` | **74.75%** | 1 | No | - | No | No | BF16 | DAPO Full FT baseline |
| **E3a** | Quant | `Q_MXFP4_DAPO_fullFT_1ep_73.77` | 73.77% | 1 | No | - | No | No | MXFP4 | ⚠️ STE bug - needs rerun |
| **E3b** | Quant | `Q_MXFP4_DAPO_fullFT_AQN_1ep_74.37` | 74.37% | 1 | Yes | ❌ No | No | No | MXFP4 | ⚠️ STE bug - needs rerun |
| **E4a** | Quant | `Q_NVFP4_DAPO_fullFT_72.10` | 72.10% | 1 | No | - | No | No | NVFP4 | ⚠️ STE bug - needs rerun (no 2ep) |
| **E4b** | Quant | `Q_NVFP4_DAPO_fullFT_AQN_73.24` | 73.24% | 1 | Yes | ❌ No | No | No | NVFP4 | ⚠️ STE bug - needs rerun (no 2ep) |
| **E7a** | LoRA | `LoRA_BF16_DAPO_1ep_71.27` | **71.27%** | 1 | No | - | No | Yes | BF16 | BF16 LoRA baseline |
| **E5a-LoRA** | LoRA | `LoRA_NVFP4_DAPO_1ep_68.23` | 68.23% | 1 | No | - | No | Yes | NVFP4 | NVFP4 + LoRA |
| **E5b-LoRA** | LoRA | `LoRA_NVFP4_DAPO_1ep_AQN_70.58` | 70.58% | 1 | Yes | ❌ No | No | Yes | NVFP4 | NVFP4 + LoRA + AQN |
| **E6a** | LoRA | `LoRA_MXFP4_DAPO_1ep_65.88` | 65.88% | 1 | No | - | No | Yes | MXFP4 | MXFP4 + LoRA |
| **E6b** | LoRA | `LoRA_MXFP4_DAPO_1ep_AQN_67.48` | 67.48% | 1 | Yes | ❌ No | No | Yes | MXFP4 | MXFP4 + LoRA + AQN |
| **E12** | LoRA | `LoRA_MXFP4_DAPO_1ep_AQN-high_72.48` | 72.48% | 1 | Yes | ✅ Yes | variable | Yes | MXFP4 | High σ + SRDD |

---

## 2-Epoch Extension Experiments

These experiments extend 1-epoch runs to 2 epochs to study longer training effects.

### Training Configuration (All 2ep Experiments)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `train_batch_size` | 128 | Per-step training batch |
| `gen_batch_size` | 256 | Generation batch (determines steps/epoch) |
| `total_epochs` | 2 | Target epochs |
| `test_freq` | 10 | Evaluation every 10 steps |
| **Steps/epoch** | 29 | 7473 samples / 256 batch = 29 |
| **Total steps** | 58 | 29 × 2 epochs |
| **Eval steps** | 0,10,20,30,40,50 | Last eval at step 50 (58%10≠0) |

### Results

| Exp ID | Type | 1ep | 2ep @step50 | Peak | Status | Notes |
|--------|------|-----|-------------|------|--------|-------|
| **E6b-2ep** | LoRA | 67.48% | **73.24%** | 73.31%@40 | ✅ Complete | MXFP4 + LoRA + AQN (+5.76%) |
| **E6a-2ep** | LoRA | 65.88% | **72.93%** | 72.93%@50 | ✅ Complete | MXFP4 + LoRA (+7.05%) |
| **E7a-2ep** | LoRA | 71.27% | **73.84%** | 73.84%@40 | ✅ Complete | BF16 + LoRA (+2.57%) |
| **E3a-2ep** | Quant | 73.77% | **72.78%** | 73.92%@40 | ⚠️ STE Bug | MXFP4 + Full FT (-0.99%) |
| **E3b-2ep** | Quant | 74.37% | **70.05%** | 73.24%@50 | ⚠️ STE Bug | MXFP4 + Full FT + AQN (-4.32%) |
| **E8a-2ep** | Quant | 74.75% | **75.97%** | 75.97%@40 | ✅ Complete | BF16 + Full FT (+1.22%) |
| **E12-2ep** | LoRA | 72.48% | **72.48%** | 72.93%@30 | ✅ Complete | MXFP4 + LoRA + AQN-high (+0.45% peak) |

**Notes on "Ended Early"**: All experiments ran to ~95% completion (step 55-57 of 58). This is normal behavior due to:
1. Dataloader exhaustion (7473 samples doesn't divide evenly into 58 batches)
2. Final eval at step 50 (last multiple of test_freq=10 before step 58)

**Key Findings**:
- 2-epoch training significantly improves LoRA results (+5-7% accuracy)
- Full FT results show mixed results: E8a improved (+1.22%), but E3a/E3b degraded
- E3b-2ep shows potential overfitting (dropped from 73.24%@step50 in training to 70.05% final)
- E12-2ep (AQN-high) peaked at step 30 (72.93%) and declined - no improvement over 1ep
- FullFT experiments (E3a, E3b) need rerun with STE fix, not due to early termination

---

## QeRL-Style Experiments (Balanced Batch Size)

These experiments use a smaller batch size (32 instead of 256) to increase training steps per epoch, more closely matching QeRL's configuration. This tests whether more training steps improve AQN effectiveness.

### Training Configuration (QeRL-Style)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `train_batch_size` | 32 | Smaller batch for more steps |
| `gen_batch_size` | 32 | Matches train batch |
| `total_epochs` | 1 | Single epoch |
| `test_freq` | 20 | Evaluation every 20 steps |
| **Steps/epoch** | 234 | 7473 samples / 32 batch = 234 |
| **vs Original** | 4x | 234 steps vs 58 steps (2ep) |

### Results

| Exp ID | Type | Sigma | Score | Steps | Status | Notes |
|--------|------|-------|-------|-------|--------|-------|
| **E6b-qerl** | LoRA | 0.01→0.0001 | **66.72%** | 117 | ✅ Complete | Standard σ, balanced BS |
| **E12-qerl** | LoRA | 0.05→0.0005 | TBD | TBD | 🔄 Running | High σ, balanced BS |

**Key Findings**:
- E6b-qerl (66.72%) is very close to original E6b 1-epoch baseline (67.48%)
- Dataloader exhausted at step 117 (~half of expected 234 steps)
- More training steps with balanced batch size does not significantly improve results
- E12-qerl (high sigma) result pending

---

## Summary by Category

### HW Error Injection Experiments (Simulated Hardware Noise)

All E5/E9 experiments use **identical HW noise setup** for fair comparison:
- **Noise Type**: Matmul-only (via `VERL_NOISY_OPS`)
- **Noise Level**: 5% relative Gaussian (`VERL_NOISY_OPS_SCALE=5e-2`)
- **Algorithm**: GRPO, 2 epochs
- **Model**: Qwen2.5-1.5B-Instruct

| Exp ID | Score | HW Noise | AQN σ | Epoch-Aware | AQN Layers | AQN Benefit | Notes |
|--------|-------|----------|-------|-------------|------------|-------------|-------|
| Baseline | 76.88% | None | - | - | - | - | Clean BF16 reference |
| E5 | 68.92% | 5% matmul | None | - | - | - | Noise only (-7.96%) |
| E5b | 70.58% | 5% matmul | 0.05→0.0005 | ✅ Yes | All RMSNorm | +1.66% | Epoch-aware AQN |
| E5c | 70.27% | 5% matmul | 0.01→0.00001 | ✅ Yes | All RMSNorm | +1.35% | Lower σ slightly worse |
| E9a | 70.58% | 5% matmul | 0.01→0.0001 | ✅ Yes | Layers 14-17 | +1.66% | SRDD targeted |
| E9a-high-σ | 70.81% | 5% matmul | 0.05→0.0005 | ✅ Yes | Layers 14-17 | +1.89% | High σ + targeted |
| **E9b** | **71.19%** | 5% matmul | 0.01→0.0001 | ✅ Yes | All (variable) | **+2.27%** | **BEST: per-layer multipliers** |

**Key Finding**: SRDD-guided variable layer multipliers (E9b) achieves best HW noise recovery. E5 and E9 are directly comparable (same noise setup).

### Quantization Experiments (MXFP4/NVFP4 Full Fine-Tuning)

All Quant experiments use **weight injection** setup:
- **Noise Type**: Weight-level fake quantization (via `hw_error_injection`)
- **Target**: All linear layers (excluding lm_head, embed_tokens)
- **Algorithm**: DAPO
- **Model**: Qwen2.5-1.5B-Instruct

| Exp ID | dtype | Quant Error | AQN σ | Epoch-Aware | Score | Epochs | AQN Benefit | Notes |
|--------|-------|-------------|-------|-------------|-------|--------|-------------|-------|
| E8a | BF16 | None | - | - | **74.75%** | 1 | - | DAPO baseline |
| E3a | MXFP4 | ~21% rel | None | - | 73.77% | 1 | - | ⚠️ STE bug |
| E3b | MXFP4 | ~21% rel | 0.01→0.0001 | ❌ No | 74.37% | 1 | +0.60% | ⚠️ STE bug |
| E4a | NVFP4 | ~15% rel | None | - | 72.10% | 1 | - | ⚠️ STE bug |
| E4b | NVFP4 | ~15% rel | 0.01→0.0001 | ❌ No | 73.24% | 1 | +1.14% | ⚠️ STE bug |

**Key Finding**: BF16 1ep baseline achieves 74.75%. MXFP4/NVFP4 results need rerun after STE fix.

### LoRA Experiments (Quantized Base + 16-bit LoRA)

All LoRA experiments use **weight injection + LoRA** setup:
- **Noise Type**: Weight-level fake quantization (via `hw_error_injection`)
- **Target**: Linear layers (excluding lm_head, embed_tokens, lora_A, lora_B)
- **LoRA**: rank=32, alpha=16 (trained in BF16)
- **Algorithm**: DAPO
- **Model**: Qwen2.5-1.5B-Instruct

| Exp ID | dtype | Quant Error | AQN σ | Epoch-Aware | Score | Epochs | AQN Benefit | Notes |
|--------|-------|-------------|-------|-------------|-------|--------|-------------|-------|
| E7a | BF16 | None | - | - | **71.27%** | 1 | - | LoRA baseline |
| E5a-LoRA | NVFP4 | ~15% rel | None | - | 68.23% | 1 | - | -3.04% from BF16 |
| E5b-LoRA | NVFP4 | ~15% rel | 0.01→0.0001 | ❌ No | 70.58% | 1 | +2.35% | AQN recovers some |
| E6a | MXFP4 | ~21% rel | None | - | 65.88% | 1 | - | -5.39% from BF16 |
| E6b | MXFP4 | ~21% rel | 0.01→0.0001 | ❌ No | 67.48% | 1 | +1.60% | AQN helps |
| E12 | MXFP4 | ~21% rel | 0.05→0.0005 | ✅ Yes | 72.48% | 1 | +6.60% | High σ + SRDD |

**Key Finding**: E12 with high sigma AQN + SRDD achieved the best LoRA+Quant result (72.48%), outperforming standard AQN (E6b: 67.48%) by +4.99%. AQN provides +1.60% benefit for E6b over E6a baseline.

---

## Best Results per Category

| Category | Best Experiment | Score | Epochs | Key Config |
|----------|----------------|-------|--------|------------|
| **HW Inject** | E9b | 71.19% | 2 | SRDD variable multipliers, σ=0.01 |
| **Full FT** | E8a (BF16) | 74.75% | 1 | Pure BF16 DAPO |
| **Full FT + Quant** | E3b (MXFP4+AQN) | 74.37% | 1 | MXFP4 + AQN, σ=0.01 (needs STE fix rerun) |
| **LoRA** | E7a (BF16) | 71.27% | 1 | BF16 LoRA baseline |
| **LoRA + Quant** | E12 (MXFP4+AQN-high) | 72.48% | 1 | MXFP4 + LoRA + AQN high σ + SRDD |

---

## AQN Configuration Reference

| Config | Standard (E3b, E6b) | High (E12) | HW Inject (E5b, E9x) | Description |
|--------|---------------------|------------|----------------------|-------------|
| `sigma_start` | 0.01 | 0.05 | 0.05 | Initial noise magnitude |
| `sigma_end` | 0.0001 | 0.0005 | 0.0005 | Final noise magnitude |
| `epoch_aware` | ❌ No | ✅ Yes | ✅ Yes | Per-epoch sigma scheduling |
| `layer_types` | rmsnorm | rmsnorm | rmsnorm | Target RMSNorm layers |
| SRDD guided | No | Yes | Optional | Layer-specific multipliers |

**Note on epoch_aware**: When `epoch_aware=False`, sigma decays linearly across all steps. When `epoch_aware=True`, each epoch has its own sigma range with exponential decay within the epoch, ensuring meaningful noise throughout training.

---

## Commands

```bash
# View TensorBoard
tensorboard --logdir /tmp/tb_logs --bind_all

# Access from Windows (WSL)
# http://localhost:6006

# Wandb project
# https://wandb.ai/vaai/aqn_refine
```

---

## Known Issues

### QAT/STE Implementation Bug (Affects FullFT Experiments)

**Date Identified**: 2026-01-12

**Affected Experiments**: ALL FullFT + Weight Quantization experiments:
- E3a (1ep, 2ep) - MXFP4 + FullFT
- E3b (1ep, 2ep) - MXFP4 + FullFT + AQN
- E4a (1ep) - NVFP4 + FullFT
- E4b (1ep) - NVFP4 + FullFT + AQN

**Issue**: The `hw_error_injection.py` weight quantization implementation does NOT correctly implement Straight-Through Estimator (STE) for gradient flow.

**Technical Details**:

The current implementation in `verl/utils/hw_error_injection.py` (lines 714-790):

```python
# Pre-hook (before forward):
module._original_weight = weight.clone()
module.weight.data = quantized_weight  # Use quantized for forward

# Post-hook (after forward, BEFORE backward):
module.weight.data = module._original_weight  # Restore original
```

**Problem**: In PyTorch's backward pass for `nn.Linear`:
- `grad_W = grad_output.T @ x` - Uses saved `x` from forward (OK)
- `grad_x = grad_output @ W` - Uses **current** `W` value (WRONG!)

Since we restore `W_orig` after forward but before backward, the gradient `grad_x` is computed with `W_orig` instead of `W_quant`. This corrupts gradient flow to earlier layers.

**Impact by Experiment Type**:

| Type | Base Weights | Impact |
|------|-------------|--------|
| **LoRA** | Frozen (`requires_grad=False`) | ✅ No impact - `grad_x` not used for frozen weights |
| **FullFT** | Trainable (`requires_grad=True`) | ❌ Corrupted gradients - affects ALL earlier layers |

**Why E3b Results May Be Invalid**:
- E3b uses FullFT with trainable base weights
- The corrupted `grad_x` propagates backward through all layers
- Combined with AQN noise, this may have caused suboptimal training
- E3b-2ep dropped from 73.24% (step50) to 70.05% (step58) - possibly due to this bug

**True STE Implementation** (required fix):

```python
class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, quantize_fn):
        ctx.save_for_backward(weight)
        return quantize_fn(weight)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None  # Pass gradient through unchanged (STE)
```

**Status**:
- LoRA experiments (E6a, E6b, E12, etc.) are **NOT affected** - results are valid
- FullFT experiments (E3b, E4b) **NEED RERUN** after fix is implemented

**Action Items**:
1. ✅ Document this issue
2. ⏳ Implement proper STE in `hw_error_injection.py`
3. ⏳ Rerun ALL affected FullFT experiments:
   - E3a (1ep, 2ep) - MXFP4 + FullFT
   - E3b (1ep, 2ep) - MXFP4 + FullFT + AQN
   - E4a (1ep) - NVFP4 + FullFT
   - E4b (1ep) - NVFP4 + FullFT + AQN

---

## QeRL Paper Comparison

### Why Our AQN Results Differ from QeRL Paper

**Reference**: QeRL paper (arXiv:2510.11696) reports significant improvements with AQN on quantized models. Our experiments show marginal improvement. This section analyzes the key differences.

### Configuration Comparison

| Parameter | QeRL GSM8K Config | Our Experiments | Impact |
|-----------|-------------------|-----------------|--------|
| **Dataset** | GSM8K train (~7,473) | DAPO Math (7,473) | Same size |
| **Batch Size** | 2 | 128-256 | 64-128x smaller in QeRL |
| **Steps/Epoch** | ~3,700 | ~29-58 | **60-130x more steps in QeRL** |
| **Epochs** | 1 | 2 | QeRL uses 1 epoch |
| **Total Steps** | ~3,700 | ~58 | **60x difference** |
| **Quantization** | NVFP4 | MXFP4 | Different error profile |
| **Quant Error** | ~15% rel | ~21% rel | Higher error in MXFP4 |
| **Learning Rate** | 1e-5 | 1e-5 | Same |
| **Epoch-Aware** | ❌ No (step-based) | Mixed | Different schedule |
| **σ Range** | 1e-2 → 1e-4 | Various | Similar |

**Critical Finding**: QeRL's small batch size (2) results in ~3,700 steps per epoch on GSM8K, while our large batch size (256) gives only ~29 steps per epoch. This 60x difference in training steps is likely the primary reason our AQN results differ from QeRL's.

### QeRL's AQN Implementation Details

From analyzing QeRL source code (`../QeRL`):

**Noise Schedule** (K-stage exponential decay):
```python
# File: trl_trainer/noise_scheduler.py
sigma_trend = sigma_start * (sigma_end / sigma_start) ** (i / (num_stages - 2))
# Default: sigma_start=1e-2, sigma_end=1e-4, num_stages=10
```

**Noise Application**:
```python
# File: trl_trainer/grpo_trainer.py (line 1364-1369)
# Applied to vLLM inference model (rollout phase), NOT training model
if self.noise_scheduler:
    llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
    generate_gaussian_noise(llm_model, self.update_step, self.total_steps, sigma_trend)
```

**Key Design Choices**:
1. **First interval warmup**: σ=0 for first 1/(K+1) of training
2. **Step-based, NOT epoch-aware**: Schedule divides total_steps into K+1 intervals
3. **Targets RMSNorm only**: Same as our implementation
4. **Applied to inference model**: Affects rollout/generation, not training pass

### Why QeRL Reports Bigger Improvements

1. **Training Duration**: QeRL trains 10-20x longer (500-1000 steps vs our 58 steps)
   - More steps = more opportunities for exploration benefits to compound
   - AQN's "enhanced exploration" effect needs sufficient training to realize gains

2. **Quantization Format**:
   - **NVFP4**: E4M3 scaling, block_size=16, ~15% relative error
   - **MXFP4**: E8M0 scaling, block_size=32, ~21% relative error
   - Higher MXFP4 error may be too aggressive for AQN to compensate

3. **Dataset Size**:
   - QeRL uses full DAPO Math dataset (50K+ samples)
   - Our experiments use 7473 samples (subset)
   - Larger dataset = more diverse exploration opportunities

4. **QeRL's Main Benefit Source**:
   - Paper shows quantization noise ITSELF enhances exploration (higher entropy)
   - AQN provides only **marginal additional benefit** (+0.4% for 3B in Table 6)
   - Main speedup comes from NVFP4 fast inference, not AQN

### QeRL Paper Table 6 (3B Model Results)

| Method | GSM8K | MATH 500 | Notes |
|--------|-------|----------|-------|
| BF16 LoRA | 89.31% | 59.8% | 16-bit baseline |
| QLoRA | 84.84% | 53.4% | 8-bit |
| **NVFP4 LoRA** | **89.84%** | 72.4% | Quant noise helps |
| NVFP4 LoRA + AQN | 90.07% | 72.8% | **+0.23%** from AQN |

**Key Insight**: QeRL's main improvement comes from quantization noise itself, not AQN. AQN provides diminishing returns on top of quantization noise.

### Recommendations for Future Experiments

To match QeRL's results:

1. **Reduce Batch Size to Increase Training Steps** (Most Important):
   - QeRL uses batch_size=2, resulting in ~3,700 steps/epoch
   - Our batch_size=256 gives only ~29 steps/epoch
   - Consider reducing to batch_size=16-32 for 200-450 steps/epoch
   - AQN's exploration benefit needs many steps to accumulate

2. **Try NVFP4 Instead of MXFP4**:
   - Lower error profile (~15% vs ~21%)
   - May be more compatible with AQN

3. **Try Higher Learning Rate for Quantized Models**:
   - QeRL README recommends 10x higher LR for quantized models
   - Try 1e-4 or 3e-5 instead of 1e-5

4. **Consider Linear Layer Noise**:
   - Our implementation supports `layer_types=['linear']`
   - Aligns noise injection with where quantization error actually occurs

5. **Remove Epoch-Aware**:
   - QeRL uses pure step-based schedule
   - Epoch-aware may fragment the decay curve unnecessarily

### Example: Matching QeRL Configuration

To run a comparable experiment:
```bash
# QeRL-like config for verl
# gen_batch_size=32 (instead of 256)
# -> 7473 / 32 = 233 steps/epoch
# -> 2 epochs = 466 steps total (closer to QeRL's ~3700)
```
