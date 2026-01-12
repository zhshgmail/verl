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

| Exp ID | Type | TensorBoard Name | Score | Epochs | AQN | SRDD | LoRA | dtype | Notes |
|--------|------|------------------|-------|--------|-----|------|------|-------|-------|
| **Baseline** | HW | `HW_BF16_GRPO_2ep_76.88_BASELINE` | **76.88%** | 2 | No | No | No | BF16 | GRPO, 2 epochs |
| **E5** | HW | `HW_5pct_noise_only_68.92` | 68.92% | 2 | No | No | No | BF16 | 5% noise, no AQN |
| **E5b** | HW | `HW_5pct_AQN_epoch-aware_sigma0.01_70.58` | 70.58% | 2 | Yes | No | No | BF16 | Epoch-aware AQN, σ=0.01 |
| **E5c** | HW | `HW_5pct_AQN_lower_sigma0.005_70.27` | 70.27% | 2 | Yes | No | No | BF16 | Lower sigma, σ=0.005 |
| **E9a** | HW | `HW_5pct_AQN_SRDD-targeted_sigma0.01_70.58` | 70.58% | 2 | Yes | targeted | No | BF16 | AQN on specific layers only |
| **E9a-high-σ** | HW | `HW_5pct_AQN_SRDD-targeted_high-sigma0.05_70.81` | 70.81% | 2 | Yes | targeted | No | BF16 | High σ=0.05 start |
| **E9b** | HW | `HW_5pct_AQN_SRDD-variable_sigma0.01_71.19_BEST` | **71.19%** | 2 | Yes | variable | No | BF16 | **BEST HW** - per-layer multipliers |
| **E8a** | Quant | `Q_BF16_DAPO_fullFT_1ep_74.75_BASELINE` | **74.75%** | 1 | No | No | No | BF16 | DAPO Full FT baseline |
| **E3a** | Quant | `Q_MXFP4_DAPO_fullFT_73.77` | 73.77% | 1 | No | No | No | MXFP4 | DAPO + MXFP4 |
| **E3b** | Quant | `Q_MXFP4_DAPO_fullFT_AQN_74.37` | 74.37% | 1 | Yes | No | No | MXFP4 | DAPO + MXFP4 + AQN |
| **E4a** | Quant | `Q_NVFP4_DAPO_fullFT_72.10` | 72.10% | 1 | No | No | No | NVFP4 | DAPO + NVFP4 |
| **E4b** | Quant | `Q_NVFP4_DAPO_fullFT_AQN_73.24` | 73.24% | 1 | Yes | No | No | NVFP4 | DAPO + NVFP4 + AQN |
| **E7a** | LoRA | `LoRA_BF16_DAPO_1ep_71.27` | **71.27%** | 1 | No | No | Yes | BF16 | BF16 LoRA baseline |
| **E5a-LoRA** | LoRA | `LoRA_NVFP4_DAPO_1ep_68.23` | 68.23% | 1 | No | No | Yes | NVFP4 | NVFP4 + LoRA |
| **E5b-LoRA** | LoRA | `LoRA_NVFP4_DAPO_1ep_AQN_70.58` | 70.58% | 1 | Yes | No | Yes | NVFP4 | NVFP4 + LoRA + AQN |
| **E6a** | LoRA | `LoRA_MXFP4_DAPO_1ep_65.88` | 65.88% | 1 | No | No | Yes | MXFP4 | MXFP4 + LoRA |
| **E6b** | LoRA | `LoRA_MXFP4_DAPO_1ep_AQN_67.48` | 67.48% | 1 | Yes | No | Yes | MXFP4 | MXFP4 + LoRA + AQN |
| **E12** | LoRA | `LoRA_MXFP4_DAPO_1ep_AQN-high_72.48` | **72.48%** | 1 | Yes | variable | Yes | MXFP4 | **BEST LoRA** - High σ + SRDD |

---

## 2-Epoch Extension Experiments (In Progress)

These experiments extend 1-epoch runs to 2 epochs to study longer training effects.

| Exp ID | Type | Original 1ep | 2ep Score | Status | Notes |
|--------|------|--------------|-----------|--------|-------|
| **E6b-2ep** | LoRA | 67.48% | **73.24%** | ✅ Complete | MXFP4 + LoRA + AQN |
| **E6a-2ep** | LoRA | 65.88% | **72.93%** | ✅ Complete | MXFP4 + LoRA |
| **E7a-2ep** | LoRA | 71.27% | **73.84%** @step40 | ⚠️ Needs Rerun | BF16 + LoRA (ended early, investigate) |
| **E12-2ep** | LoRA | 72.48% | - | ⏳ Queued | MXFP4 + LoRA + AQN-high (BEST 1ep) |
| **E3a-2ep** | Quant | 73.77% | **73.92%** @step40 | 🔄 Running | MXFP4 + Full FT (step 49/58) |
| **E3b-2ep** | Quant | 74.37% | - | ⏳ Queued | MXFP4 + Full FT + AQN |
| **E8a-2ep** | Quant | 74.75% | - | ⏳ Queued | BF16 + Full FT |

**Key Finding**: 2-epoch training significantly improves LoRA results (+5-7% accuracy).

---

## Summary by Category

### HW Error Injection Experiments (Simulated Hardware Noise)

All E5/E9 experiments use **identical HW noise setup** for fair comparison:
- **Noise Type**: Matmul-only (via `VERL_NOISY_OPS`)
- **Noise Level**: 5% relative Gaussian (`VERL_NOISY_OPS_SCALE=5e-2`)
- **Algorithm**: GRPO, 2 epochs
- **Model**: Qwen2.5-1.5B-Instruct

| Exp ID | Score | HW Noise | AQN σ | AQN Layers | AQN Benefit | Notes |
|--------|-------|----------|-------|------------|-------------|-------|
| Baseline | 76.88% | None | - | - | - | Clean BF16 reference |
| E5 | 68.92% | 5% matmul | None | - | - | Noise only (-7.96%) |
| E5b | 70.58% | 5% matmul | 0.05→0.0005 | All RMSNorm | +1.66% | Epoch-aware AQN |
| E5c | 70.27% | 5% matmul | 0.01→0.00001 | All RMSNorm | +1.35% | Lower σ slightly worse |
| E9a | 70.58% | 5% matmul | 0.01→0.0001 | Layers 14-17 | +1.66% | SRDD targeted (high-error only) |
| E9a-high-σ | 70.81% | 5% matmul | 0.05→0.0005 | Layers 14-17 | +1.89% | High σ + targeted |
| **E9b** | **71.19%** | 5% matmul | 0.01→0.0001 | All (variable) | **+2.27%** | **BEST: per-layer multipliers** |

**Key Finding**: SRDD-guided variable layer multipliers (E9b) achieves best HW noise recovery. E5 and E9 are directly comparable (same noise setup).

### Quantization Experiments (MXFP4/NVFP4 Full Fine-Tuning)

All Quant experiments use **weight injection** setup:
- **Noise Type**: Weight-level fake quantization (via `hw_error_injection`)
- **Target**: All linear layers (excluding lm_head, embed_tokens)
- **Algorithm**: DAPO, 1 epoch
- **Model**: Qwen2.5-1.5B-Instruct

| Exp ID | dtype | Quant Error | AQN σ | Score | AQN Benefit | Notes |
|--------|-------|-------------|-------|-------|-------------|-------|
| E8a | BF16 | None | - | 74.75% | - | DAPO baseline |
| E3a | MXFP4 | ~21% rel | None | 73.77% | - | -0.98% from BF16 |
| E3b | MXFP4 | ~21% rel | 0.01→0.0001 | 74.37% | +0.60% | AQN helps |
| E4a | NVFP4 | ~15% rel | None | 72.10% | - | -2.65% from BF16 |
| E4b | NVFP4 | ~15% rel | 0.01→0.0001 | 73.24% | +1.14% | AQN helps more |

**Key Finding**: AQN provides +0.60% for MXFP4, +1.14% for NVFP4. BF16 baseline (74.75%) > MXFP4+AQN (74.37%).

### LoRA Experiments (Quantized Base + 16-bit LoRA)

All LoRA experiments use **weight injection + LoRA** setup:
- **Noise Type**: Weight-level fake quantization (via `hw_error_injection`)
- **Target**: Linear layers (excluding lm_head, embed_tokens, lora_A, lora_B)
- **LoRA**: rank=32, alpha=16 (trained in BF16)
- **Algorithm**: DAPO, 1 epoch
- **Model**: Qwen2.5-1.5B-Instruct

| Exp ID | dtype | Quant Error | AQN σ | AQN Type | Score | AQN Benefit | Notes |
|--------|-------|-------------|-------|----------|-------|-------------|-------|
| E7a | BF16 | None | - | - | 71.27% | - | LoRA baseline |
| E5a-LoRA | NVFP4 | ~15% rel | None | - | 68.23% | - | -3.04% from BF16 |
| E5b-LoRA | NVFP4 | ~15% rel | 0.01→0.0001 | Standard | 70.58% | +2.35% | AQN recovers most |
| E6a | MXFP4 | ~21% rel | None | - | 65.88% | - | -5.39% from BF16 |
| E6b | MXFP4 | ~21% rel | 0.01→0.0001 | Standard | 67.48% | +1.60% | AQN helps |
| **E12** | MXFP4 | ~21% rel | 0.05→0.0005 | High + SRDD | **72.48%** | **+6.60%** | **BEST: exceeds BF16!** |

**Key Finding**: E12 (MXFP4+LoRA+AQN-high) exceeds BF16 baseline (72.48% > 71.27%)!

---

## Best Results per Category

| Category | Best Experiment | Score | Key Config |
|----------|----------------|-------|------------|
| **HW Inject** | E9b | 71.19% | SRDD variable multipliers, σ=0.01 |
| **Full FT** | E8a (BF16) | 74.75% | Pure BF16 DAPO baseline |
| **Full FT + Quant** | E3b (MXFP4+AQN) | 74.37% | MXFP4 + AQN, σ=0.01 |
| **LoRA** | E12 | 72.48% | MXFP4 + LoRA + AQN-high (σ=0.05) |

---

## AQN Configuration Reference

| Config | Standard | High (E12) | Description |
|--------|----------|------------|-------------|
| `sigma_start` | 0.01 | 0.05 | Initial noise magnitude |
| `sigma_end` | 0.0001 | 0.0005 | Final noise magnitude |
| `epoch_aware` | Yes | Yes | Reduce σ over training |
| `layer_types` | rmsnorm | rmsnorm | Target RMSNorm layers |
| SRDD guided | Optional | Yes | Layer-specific multipliers |

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
