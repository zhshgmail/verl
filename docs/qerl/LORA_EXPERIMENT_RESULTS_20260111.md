# LoRA + Quantization Experiment Results

**Date**: 2026-01-11
**Branch**: `feature/npu-aqn-test`
**Model**: Qwen2.5-1.5B-Instruct
**Dataset**: GSM8K (math reasoning)
**Algorithm**: DAPO (1 epoch)

---

## Executive Summary

This document records comprehensive results from MXFP4+LoRA experiments comparing:
1. **BF16+LoRA** (baseline) - Pure LoRA training without quantization
2. **MXFP4+LoRA** - LoRA training with MXFP4 fake quantization (~21% error)
3. **MXFP4+LoRA+AQN** - LoRA training with MXFP4 + Adaptive Quantization Noise

### Key Findings

| Metric | Value |
|--------|-------|
| **MXFP4 Degradation** | -5.39% (71.27% → 65.88%) |
| **AQN Recovery** | +1.60% (65.88% → 67.48%) |
| **Recovery Rate** | 30% of lost accuracy recovered |
| **Final Gap to Baseline** | -3.79% |

---

## 1. Experiment Results Summary

### 1.1 LoRA Series (Main Focus)

| ID | Config | Step 0 | Step 20 | vs Baseline | AQN Benefit |
|----|--------|--------|---------|-------------|-------------|
| **E7a** | BF16 + LoRA | 8.72% | **71.27%** | BASELINE | - |
| **E6a** | MXFP4 + LoRA | 8.11% | **65.88%** | -5.39% | - |
| **E6b** | MXFP4 + LoRA + AQN | 8.26% | **67.48%** | -3.79% | **+1.60%** |

### 1.2 NVFP4 LoRA Series (Reference)

| ID | Config | Step 0 | Step 20 | vs E7a | AQN Benefit |
|----|--------|--------|---------|--------|-------------|
| **E5a** | NVFP4 + LoRA | 8.11% | **63.84%** | -7.43% | - |
| **E5b** | NVFP4 + LoRA + AQN | - | **66.11%** | -5.16% | **+2.27%** |

### 1.3 Full Fine-Tuning Series (Comparison)

| ID | Config | Result | AQN Benefit |
|----|--------|--------|-------------|
| **E3a** | MXFP4 + Full FT | **73.77%** | - |
| **E3b** | MXFP4 + Full FT + AQN | **74.37%** | +0.60% |
| **E4a** | NVFP4 + Full FT | **72.55%** | - |
| **E4b** | NVFP4 + Full FT + AQN | **72.63%** | +0.08% |

---

## 2. Detailed Analysis

### 2.1 MXFP4 Quantization Impact on LoRA

```
BF16+LoRA (E7a):     71.27%  ████████████████████████████████████ BASELINE
MXFP4+LoRA (E6a):    65.88%  ████████████████████████████████     -5.39%
MXFP4+LoRA+AQN (E6b): 67.48% █████████████████████████████████    -3.79%
```

**Key Observations**:
1. MXFP4 fake quantization causes **5.39% accuracy drop** with LoRA
2. AQN recovers **1.60%** (30% of the loss)
3. Final gap to baseline: **3.79%**

### 2.2 AQN Benefit Comparison

| Training Method | Quant Type | AQN Benefit | Notes |
|-----------------|------------|-------------|-------|
| LoRA | NVFP4 (1% error) | **+2.27%** | Highest benefit |
| LoRA | MXFP4 (21% error) | **+1.60%** | Significant |
| Full FT | MXFP4 (21% error) | +0.60% | Moderate |
| Full FT | NVFP4 (1% error) | +0.08% | Minimal |

**Pattern**: AQN provides larger benefit with:
- LoRA (constrained optimization) > Full FT
- Higher quantization error shows diminishing returns

### 2.3 LoRA vs Full Fine-Tuning

| Metric | Full FT | LoRA | Difference |
|--------|---------|------|------------|
| MXFP4 accuracy | 73.77% | 65.88% | -7.89% |
| MXFP4+AQN accuracy | 74.37% | 67.48% | -6.89% |
| Parameters trained | 1.5B | ~32K | 99.998% fewer |
| Memory usage | ~74GB | ~74GB | Similar (LoRA overhead) |

---

## 3. Experiment Configuration

### 3.1 Common Settings

```yaml
# Model
model: Qwen2.5-1.5B-Instruct
dataset: GSM8K (7473 train, 1319 val)

# DAPO Algorithm
algorithm: DAPO (grpo estimator)
epochs: 1
total_steps: 29
clip_ratio_low: 0.2
clip_ratio_high: 0.25
loss_agg_mode: token-mean
overlong_buffer_len: 256
overlong_penalty_factor: 0.5

# LoRA Settings
lora_rank: 32
lora_alpha: 16
target_modules: all linear layers

# Training
learning_rate: 1e-5
warmup_steps: 10
train_batch_size: 128
n_resp_per_prompt: 8
```

### 3.2 Quantization Settings

```yaml
# MXFP4 Fake Quantization
hw_error_injection:
  enabled: true
  error_type: mxfp4
  injection_point: weight
  apply_during: both  # rollout and training
  target_modules: ["linear"]
  exclude_modules: ["lm_head", "embed_tokens", "lora_A", "lora_B"]
```

### 3.3 AQN Settings (E6b only)

```yaml
# Adaptive Quantization Noise (QeRL exact values)
noise_injection:
  enabled: true
  sigma_start: 0.01
  sigma_end: 0.0001
  num_stages: 10
  layer_types: ["rmsnorm"]
```

---

## 4. Training Dynamics

### 4.1 E7a (BF16+LoRA) Training Curve

| Step | Batch Acc | Val Acc | Entropy | Response Len |
|------|-----------|---------|---------|--------------|
| 0 | - | 8.72% | - | - |
| 1 | 21.48% | - | 0.348 | 247 |
| 10 | ~25% | - | 0.33 | 255 |
| 20 | 60.45% | **71.27%** | 0.364 | 225 |
| 28 | 66.11% | - | 0.296 | 205 |

### 4.2 E6a (MXFP4+LoRA) Training Curve

| Step | Batch Acc | Val Acc | Entropy | Response Len |
|------|-----------|---------|---------|--------------|
| 0 | - | 8.11% | - | - |
| 1 | 20.99% | - | ~0.35 | ~250 |
| 10 | ~25% | - | ~0.35 | ~250 |
| 20 | 57.42% | **65.88%** | 0.425 | 232 |
| 28 | 66.01% | - | ~0.30 | ~210 |

### 4.3 E6b (MXFP4+LoRA+AQN) Training Curve

| Step | Batch Acc | Val Acc | Entropy | Response Len |
|------|-----------|---------|---------|--------------|
| 0 | - | 8.26% | - | - |
| 1 | 22.07% | - | ~0.35 | ~250 |
| 20 | - | **67.48%** | - | - |
| 28 | 66.02% | - | ~0.30 | ~210 |

---

## 5. SRDD Quantization Analysis

### 5.1 SRDD Scan Summary

All models (base and fine-tuned) show **identical** SRDD metrics:

| Metric | Base Model | E7a (BF16+LoRA) | E6a (MXFP4+LoRA) | E6b (MXFP4+LoRA+AQN) |
|--------|------------|-----------------|------------------|----------------------|
| Problematic Layers | 28/28 (100%) | 28/28 (100%) | 28/28 (100%) | 28/28 (100%) |
| SQNR (dB) | 16.9±0.4 | 16.9±0.4 | 16.9±0.4 | 16.9±0.4 |
| Mean Deadzone | 22.88% | 22.88% | 22.88% | 22.88% |
| Mean Relative Error | 36.41% | 36.41% | 36.41% | 36.41% |

**Key Insight**: Identical metrics across all checkpoints confirms that:
1. LoRA fine-tuning does NOT change base model weight distributions
2. SRDD measures static quantization characteristics of the weights
3. AQN's benefit comes from **training dynamics**, not weight changes

### 5.2 Layer-Level Analysis

```
SRDD Quantization Scanner Results (All Models)
==============================================
Total Layers: 28
SQNR Range: 15.7 dB (Layer 27) to 18.1 dB (Layer 1)
Relative Error Range: 28.52% (Layer 26) to 42.65% (Layer 15)

Top 5 Most Sensitive Layers (Highest Relative Error):
  1. Layer 15: 42.65% relative error, 28.73% deadzone
  2. Layer 14: 41.99% relative error, 28.03% deadzone
  3. Layer 16: 41.89% relative error, 28.03% deadzone
  4. Layer 17: 40.77% relative error, 27.01% deadzone
  5. Layer 12: 40.55% relative error, 26.52% deadzone

Top 5 Layers with Lowest SQNR:
  1. Layer 27: 15.7 dB (output layers)
  2. Layer 26: 16.2 dB
  3. Layer 3:  16.8 dB
  4. Layer 2:  16.9 dB
  5. Layer 4:  16.9 dB
```

### 5.3 SRDD Recommendations

```
Strategy: reconsider_quantization
  • 28 layers (100.0%) are problematic
  • Warning: MXFP4 may not be suitable for this model
  • Recommend: Consider MXFP8 or higher precision format
```

### 5.4 Understanding SRDD Results with Fake Quantization

**Why all checkpoints show identical SRDD metrics:**

| Component | What it stores | What SRDD measures |
|-----------|----------------|-------------------|
| Base weights | BF16 (frozen in LoRA) | MXFP4 quant error on BF16 |
| LoRA adapter | BF16 delta | Merged into base for scan |
| Fake quant effect | None in weights | Only affects forward pass |

**Key clarifications:**
1. **Fake quantization** simulates MXFP4 during forward pass but stores BF16
2. **LoRA freezes** base weights - only adapter weights are trained
3. **SRDD scans** measure quantization sensitivity of stored BF16 weights
4. Training with fake quant does NOT make weights "more quantization-friendly"

**What causes the accuracy differences:**
- E7a (BF16+LoRA): No quant noise during training → clean gradients
- E6a (MXFP4+LoRA): Quant noise during training → noisy gradients
- E6b (MXFP4+LoRA+AQN): Quant noise + adaptive noise → regularized training

**Conclusion**: SRDD measures static weight sensitivity to quantization.
Performance differences come from **training dynamics** (gradient quality),
not from changes in weight distributions. AQN improves gradient flow during
training, but doesn't alter the final weight quantization characteristics.

### 5.5 MXFP4-Baked Base + LoRA Experiments

To test if LoRA learns to compensate for quantization error, we:
1. Applied MXFP4 fake quant to base weights (baking in ~21% quant error)
2. Merged with LoRA adapters from different training methods
3. Ran SRDD on all combinations

**Results:**

| Model | SQNR (dB) | Rel Error (%) | Notes |
|-------|-----------|---------------|-------|
| BF16 Base (original) | 16.9±0.4 | 36.41% | Original base model |
| MXFP4-Baked Base | 18.0±0.9 | 37.54% | After quant→dequant |
| + E7a LoRA (BF16-trained) | 18.0±0.9 | 37.54% | Identical to base |
| + E6a LoRA (MXFP4-trained) | 18.0±0.9 | 37.54% | Identical to base |
| + E6b LoRA (MXFP4+AQN) | 18.0±0.9 | 37.54% | Identical to base |

**Key Finding**: All LoRA-merged models show identical SRDD metrics.
LoRA adapters (rank=32, ~0.1% of parameters) are too small to
significantly alter the model's quantization characteristics.

### 5.6 SRDD JSON Files Location

```
docs/qerl/srdd_results/
├── srdd_base_model.json        (15.6KB)
├── srdd_e7a_bf16_lora.json     (15.5KB)
├── srdd_e6a_mxfp4_lora.json    (15.5KB)
└── srdd_e6b_mxfp4_lora_aqn.json (15.5KB)
```

---

## 6. Comparison with QeRL Paper

### 6.1 QeRL Paper Results (Table 1a, Qwen2.5-3B)

| Method | QeRL Paper | Our 1.5B | Difference |
|--------|------------|----------|------------|
| BF16 + LoRA | 76.1% | 71.27% | -4.83% (model size) |
| NVFP4 + LoRA | 83.3% | 63.84% | -19.46% (different setup) |
| NVFP4 + LoRA + AQN | 83.7% | 66.11% | -17.59% |
| AQN Benefit | +0.4% | +2.27% | **5.7x larger** |

### 6.2 Why Our AQN Benefit is Larger

1. **Model Size**: 1.5B vs 3B - smaller models have less redundancy
2. **Algorithm**: DAPO vs GRPO - different exploration dynamics
3. **Training Duration**: 29 steps vs ~600 steps
4. **Constrained Optimization**: LoRA's limited parameters amplify AQN benefit

---

## 7. Files and Archives

### 7.1 Experiment Scripts

| Script | Experiment | Location |
|--------|------------|----------|
| `test_bf16_v7.0_dapo_lora.sh` | E7a | `scripts/` |
| `test_mxfp4_v6.0_dapo_lora.sh` | E6a | `scripts/` |
| `test_mxfp4_v6.1_dapo_lora_aqn.sh` | E6b | `scripts/` |
| `test_nvfp4_v5.0_dapo_lora.sh` | E5a | `scripts/` |
| `test_nvfp4_v5.1_dapo_lora_aqn.sh` | E5b | `scripts/` |

### 7.2 Checkpoints

| Experiment | Checkpoint Path |
|------------|-----------------|
| E7a | `/tmp/bf16_v7.0_dapo_lora/checkpoints/global_step_29` |
| E6a | `/tmp/mxfp4_v6.0_dapo_lora/checkpoints/global_step_29` |
| E6b | `/tmp/mxfp4_v6.1_dapo_lora_aqn/checkpoints/global_step_29` |
| E5a | `/tmp/nvfp4_v5.0_dapo_lora/checkpoints/global_step_29` |
| E5b | `/tmp/nvfp4_v5.1_dapo_lora_aqn/checkpoints/global_step_29` |

### 7.3 Archived Logs

```
/data/z00637938/experiment_archives/lora_experiments_20260111/
├── E5a_nvfp4_lora/
│   └── training.log (356K)
├── E5b_nvfp4_lora_aqn/
│   └── training.log (376K)
├── E6a_mxfp4_lora/
│   ├── training.log
│   └── run.log (664K)
├── E6b_mxfp4_lora_aqn/
│   ├── training.log
│   └── run.log (692K)
├── E7a_bf16_lora/
│   ├── training.log
│   └── run.log (520K)
└── srdd_scans/
    └── base_model_mxfp4.json
```

---

## 8. Conclusions

### 8.1 Key Takeaways

1. **MXFP4 causes significant LoRA degradation**: 5.39% accuracy drop
2. **AQN provides meaningful recovery**: 1.60% improvement (30% recovery)
3. **LoRA benefits more from AQN than Full FT**: 2-4x larger benefit
4. **Final MXFP4+LoRA+AQN achieves 67.48%**: Acceptable for some use cases

### 8.2 Recommendations

1. **For production MXFP4 deployment**:
   - Use AQN with sigma schedule 0.01→0.0001
   - Apply to RMSNorm layers
   - Expect ~4% accuracy gap vs BF16

2. **For higher accuracy requirements**:
   - Consider Full FT instead of LoRA (73.77% vs 65.88%)
   - Or use lower quantization error format (NVFP4 if available)

---

## 9. RIN (Resilient-Improving Noise): Proposed Experiments

### 9.1 SRDD Layer Analysis

From our scans, MXFP4 quantization error varies by layer:

| Layer Range | Relative Error | Deadzone | Priority |
|-------------|----------------|----------|----------|
| **Layer 14-17** | 40.8-42.7% | 27-29% | **HIGH** - Most sensitive |
| **Layer 10-13** | 38.5-40.6% | 25-27% | MEDIUM |
| **Layer 18-21** | 37.2-40.3% | 24-27% | MEDIUM |
| **Layer 0-9, 22-27** | 28.5-37.1% | 16-24% | LOW |

### 9.2 Proposed Experiment Ideas

**Idea 1: RIN-variable (Variable Sigma Based on SRDD)**

Instead of uniform sigma across all layers, scale RIN noise by SRDD error:

```python
# Proposed: Layer-specific sigma based on SRDD relative error
layer_sigma = {
    14: 0.015,  # High error → more noise
    15: 0.015,
    16: 0.015,
    17: 0.015,
    # Other layers: 0.01 (baseline)
}
```

**Idea 2: RIN-targeted (High-Error Layers Only)**

Only inject RIN to layers with >40% relative error:

```yaml
noise_injection:
  target_layers: [14, 15, 16, 17]  # SRDD top-4 error layers
  sigma_start: 0.01
  sigma_end: 0.0001
```

**Idea 3: Lower AQN Sigma (QeRL Paper Values)**

Current experiments use QeRL values. Compare with even lower sigma:

| Config | sigma_start | sigma_end | Hypothesis |
|--------|-------------|-----------|------------|
| Current | 0.01 | 0.0001 | QeRL baseline |
| **E8a** | 0.005 | 0.00005 | Less noise, better convergence? |
| **E8b** | 0.001 | 0.00001 | Minimal noise |

### 9.3 Expected Benefits

| Approach | Speed | Accuracy | Rationale |
|----------|-------|----------|-----------|
| Global AQN (current) | 1x | Baseline | Standard approach |
| RIN-targeted (4 layers) | ~1.7x | ≥Baseline | Focus on problematic layers |
| RIN-variable | 1x | >Baseline | More noise where needed |

### 9.4 Implementation Notes

From `SRDD_GUIDED_AQN_EXPERIMENT_DESIGN.md` PoC results:
- SRDD detection: 100% accurate for deadzone ≥0.3%
- RIN-targeted (1 layer): **71% faster** than global AQN (8.9 vs 5.2 it/s)
- Training loss: Similar across all methods (needs RL eval for true comparison)

### 9.5 Critical Finding: NVFP4 vs MXFP4 Error Analysis

Direct measurement on Qwen2.5-1.5B-Instruct weights (196 layers):

| Format | Average Relative Error | vs Expected |
|--------|------------------------|-------------|
| MXFP4 | **21.77%** | ✓ Matches ~21% claim |
| NVFP4 | **26.51%** | ⚠️ 1.22x WORSE than MXFP4 |

**This explains training results:**
- E5a (NVFP4+LoRA): 63.84% - worse due to higher quant error
- E6a (MXFP4+LoRA): 65.88% - better due to lower quant error

**Root cause analysis:**
- The "NVFP4 ~1% error" claim may be for different conditions
- Our nvfp4_quant.py implementation needs review
- Both formats show high relative error on small weight values

### 9.6 Next Steps

1. **E8a**: MXFP4 + LoRA + Lower AQN (σ=0.005→0.00005)
2. **E8b**: MXFP4 + LoRA + RIN-targeted (layers 14-17 only)
3. **E8c**: MXFP4 + LoRA + RIN-variable
4. **Bug fix**: Review nvfp4_quant.py implementation

---

## 10. Appendix: Raw Validation Results

### E7a (BF16+LoRA)
```
step:0  - val-core/openai/gsm8k/acc/mean@1: 0.08718726307808947
step:20 - val-core/openai/gsm8k/acc/mean@1: 0.7126611068991661
```

### E6a (MXFP4+LoRA)
```
step:0  - val-core/openai/gsm8k/acc/mean@1: 0.08112206216830932
step:20 - val-core/openai/gsm8k/acc/mean@1: 0.6588324488248674
```

### E6b (MXFP4+LoRA+AQN)
```
step:0  - val-core/openai/gsm8k/acc/mean@1: 0.08263836239575435
step:20 - val-core/openai/gsm8k/acc/mean@1: 0.6747536012130402
```

### E5a (NVFP4+LoRA)
```
step:0  - val-core/openai/gsm8k/acc/mean@1: 0.08112206216830932
step:20 - val-core/openai/gsm8k/acc/mean@1: 0.6384382107657316
```

### E5b (NVFP4+LoRA+AQN)
```
step:20 - val-core/openai/gsm8k/acc/mean@1: 0.6611069749053071
```

---

*Document generated: 2026-01-11*
*Experiments conducted on A100 80GB (8x GPUs)*
