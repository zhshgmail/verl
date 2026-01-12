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
| **E8a** | Quant | `Q_BF16_DAPO_fullFT_2ep_75.97` | **75.97%** | 2 | No | - | No | No | BF16 | DAPO Full FT baseline |
| **E3a** | Quant | `Q_MXFP4_DAPO_fullFT_2ep_72.78` | 72.78% | 2 | No | - | No | No | MXFP4 | ⚠️ STE bug - needs rerun |
| **E3b** | Quant | `Q_MXFP4_DAPO_fullFT_AQN_2ep_70.05` | 70.05% | 2 | Yes | ❌ No | No | No | MXFP4 | ⚠️ STE bug - needs rerun |
| **E4a** | Quant | `Q_NVFP4_DAPO_fullFT_72.10` | 72.10% | 1 | No | - | No | No | NVFP4 | ⚠️ STE bug - needs rerun (no 2ep) |
| **E4b** | Quant | `Q_NVFP4_DAPO_fullFT_AQN_73.24` | 73.24% | 1 | Yes | ❌ No | No | No | NVFP4 | ⚠️ STE bug - needs rerun (no 2ep) |
| **E7a** | LoRA | `LoRA_BF16_DAPO_2ep_73.84` | **73.84%** | 2 | No | - | No | Yes | BF16 | BF16 LoRA baseline |
| **E5a-LoRA** | LoRA | `LoRA_NVFP4_DAPO_1ep_68.23` | 68.23% | 1 | No | - | No | Yes | NVFP4 | NVFP4 + LoRA |
| **E5b-LoRA** | LoRA | `LoRA_NVFP4_DAPO_1ep_AQN_70.58` | 70.58% | 1 | Yes | ❌ No | No | Yes | NVFP4 | NVFP4 + LoRA + AQN |
| **E6a** | LoRA | `LoRA_MXFP4_DAPO_2ep_72.93` | **72.93%** | 2 | No | - | No | Yes | MXFP4 | MXFP4 + LoRA |
| **E6b** | LoRA | `LoRA_MXFP4_DAPO_2ep_AQN_73.24` | **73.24%** | 2 | Yes | ❌ No | No | Yes | MXFP4 | MXFP4 + LoRA + AQN |
| **E12** | LoRA | `LoRA_MXFP4_DAPO_2ep_AQN-high_72.93` | **72.93%** | 2 | Yes | ✅ Yes | variable | Yes | MXFP4 | High σ + SRDD (peak@step30) |

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
| E8a | BF16 | None | - | - | **75.97%** | 2 | - | DAPO baseline |
| E3a | MXFP4 | ~21% rel | None | - | 73.77% | 1 | - | ⚠️ STE bug |
| E3b | MXFP4 | ~21% rel | 0.01→0.0001 | ❌ No | 74.37% | 1 | +0.60% | ⚠️ STE bug |
| E4a | NVFP4 | ~15% rel | None | - | 72.10% | 1 | - | ⚠️ STE bug |
| E4b | NVFP4 | ~15% rel | 0.01→0.0001 | ❌ No | 73.24% | 1 | +1.14% | ⚠️ STE bug |

**Key Finding**: BF16 2ep baseline achieves 75.97%. MXFP4/NVFP4 results need rerun after STE fix.

### LoRA Experiments (Quantized Base + 16-bit LoRA)

All LoRA experiments use **weight injection + LoRA** setup:
- **Noise Type**: Weight-level fake quantization (via `hw_error_injection`)
- **Target**: Linear layers (excluding lm_head, embed_tokens, lora_A, lora_B)
- **LoRA**: rank=32, alpha=16 (trained in BF16)
- **Algorithm**: DAPO
- **Model**: Qwen2.5-1.5B-Instruct

| Exp ID | dtype | Quant Error | AQN σ | Epoch-Aware | Score | Epochs | AQN Benefit | Notes |
|--------|-------|-------------|-------|-------------|-------|--------|-------------|-------|
| E7a | BF16 | None | - | - | **73.84%** | 2 | - | LoRA baseline |
| E5a-LoRA | NVFP4 | ~15% rel | None | - | 68.23% | 1 | - | -5.61% from BF16 |
| E5b-LoRA | NVFP4 | ~15% rel | 0.01→0.0001 | ❌ No | 70.58% | 1 | +2.35% | AQN recovers some |
| E6a | MXFP4 | ~21% rel | None | - | **72.93%** | 2 | - | -0.91% from BF16 |
| E6b | MXFP4 | ~21% rel | 0.01→0.0001 | ❌ No | **73.24%** | 2 | +0.31% | AQN helps |
| E12 | MXFP4 | ~21% rel | 0.05→0.0005 | ✅ Yes | **72.93%** | 2 | +0.00% | High σ + SRDD, peaked early |

**Key Finding**: With 2 epochs, MXFP4+LoRA (E6a: 72.93%, E6b: 73.24%) nearly matches BF16 baseline (E7a: 73.84%). AQN provides marginal benefit (+0.31%) at 2 epochs.

---

## Best Results per Category

| Category | Best Experiment | Score | Epochs | Key Config |
|----------|----------------|-------|--------|------------|
| **HW Inject** | E9b | 71.19% | 2 | SRDD variable multipliers, σ=0.01 |
| **Full FT** | E8a (BF16) | 75.97% | 2 | Pure BF16 DAPO |
| **Full FT + Quant** | E3b (MXFP4+AQN) | 74.37% | 1 | MXFP4 + AQN, σ=0.01 (needs STE fix rerun) |
| **LoRA** | E7a (BF16) | 73.84% | 2 | BF16 LoRA baseline |
| **LoRA + Quant** | E6b (MXFP4+AQN) | 73.24% | 2 | MXFP4 + LoRA + AQN |

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
