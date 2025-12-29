# AQN for Hardware Heterogeneity: Experimental Analysis

**Date**: 2025-12-29
**Authors**: Zheng + Claude Code
**Branch**: `feature/npu-aqn-test`

## Executive Summary

This document analyzes whether Adaptive Quantization Noise (AQN) can help overcome hardware heterogeneity errors between GPU and NPU platforms. Our experiments show **AQN does not provide benefits in BF16 training** on either platform, and the expected hardware heterogeneity gap was not observed in our setup.

## Background

### Original QeRL Hypothesis

AQN (from QeRL paper) was designed to:
1. Inject Gaussian noise into RMSNorm layers during training
2. Make models robust to quantization errors during low-precision inference
3. Bridge the gap between high-precision training and low-precision deployment

### Hardware Heterogeneity Hypothesis

GPU and NPU have inherently different:
- Operator implementations (matmul, softmax, layernorm, etc.)
- Numerical precision behaviors
- Rounding/accumulation patterns

**Expected behavior**: A model trained on GPU could achieve 99% on GPU but only 93% on NPU due to these differences. AQN might help the model be robust to these variations.

### Key Consideration

The base model (Qwen2.5-1.5B-Instruct) was **pre-trained on GPU**. Therefore:
- GPU RL training: Model runs on same hardware as pre-training (consistent numerical behavior)
- NPU RL training: Model encounters different hardware from pre-training (potential mismatch)

This means the training reward/critic trajectory should theoretically differ between GPU and NPU.

## Experimental Setup

### Hardware
- **GPU**: NVIDIA A100 80GB
- **NPU**: Ascend 910C 64GB (8x per node)

### Model & Task
- **Model**: Qwen2.5-1.5B-Instruct (BF16)
- **Task**: GSM8K mathematical reasoning
- **Training**: GRPO with 2 epochs (116 steps)

### AQN Configurations Tested

| Test | sigma_start | sigma_end | Description |
|------|-------------|-----------|-------------|
| Baseline | - | - | No noise injection |
| AQN-QeRL (Test 3a) | 0.05 | 0.0005 | QeRL original parameters |
| AQN-Mild (Test 4a) | 0.025 | 0.00025 | Half of QeRL |
| AQN-Strong (Test 3b) | 0.10 | 0.001 | Double QeRL (crashed) |

## Results

### Final Accuracy Comparison

| Run | Platform | Final Accuracy | vs NPU Baseline |
|-----|----------|----------------|-----------------|
| GPU_baseline | GPU (A100) | **76.88%** | +0.46% |
| NPU_baseline | NPU (910C) | **76.42%** | - |
| NPU_AQN-QeRL | NPU (910C) | 75.97% | **-0.45%** |
| NPU_AQN-Mild | NPU (910C) | 74.75% | **-1.67%** |

**Key Finding**: AQN variants performed WORSE than baseline on NPU.

### Training Trajectory: GPU vs NPU

| Step | GPU Baseline | NPU Baseline | Delta |
|------|--------------|--------------|-------|
| 0 | 24.72% | 23.96% | -0.76% |
| 20 | 73.77% | 74.60% | **+0.83%** |
| 40 | 75.44% | 74.83% | -0.61% |
| 60 | 74.22% | 76.04% | **+1.82%** |
| 80 | 74.68% | 75.89% | **+1.21%** |
| 100 | 77.48% | 75.97% | -1.51% |
| 116 | 76.88% | 76.42% | -0.46% |

**Observation**: NPU actually outperforms GPU at steps 20-80, then GPU catches up. Trajectories stay within ~2% throughout training.

### AQN Impact on NPU Training

| Step | NPU Baseline | AQN-QeRL (σ=0.05) | AQN-Mild (σ=0.025) |
|------|--------------|-------------------|---------------------|
| 0 | 23.96% | 24.26% | 22.44% |
| 20 | 74.60% | 67.63% (-6.97%) | 71.72% (-2.88%) |
| 40 | 74.83% | 74.75% | 75.06% |
| 60 | 76.04% | 75.89% | **76.35%** (peak) |
| 80 | 75.89% | 75.89% | 76.19% |
| 100 | 75.97% | 75.97% | 75.66% |
| 116 | **76.42%** | 75.97% | 74.75% |

**Pattern**: AQN causes early disruption (step 20), recovers mid-training, but degrades in epoch 2.

## Analysis

### Why Hardware Heterogeneity Was Not Observed

1. **Minimal Step 0 Gap**: Base model shows only 0.76% difference between GPU (24.72%) and NPU (23.96%) at step 0, suggesting the pre-trained model transfers well to NPU.

2. **Similar Training Trajectories**: GPU and NPU baselines track within ~2% throughout training, indicating comparable learning dynamics.

3. **Possible Explanations**:
   - **Model robustness**: Qwen2.5-1.5B may be inherently robust to numerical differences
   - **BF16 precision**: Both platforms use BF16, which may mask low-level operator differences
   - **Task simplicity**: GSM8K accuracy might not be sensitive enough to expose subtle numerical errors
   - **RL adaptation**: The RL process may quickly re-calibrate the model to new hardware

### Why AQN Did Not Help

1. **No Error to Overcome**: Without a measurable heterogeneity gap, AQN has nothing to fix.

2. **Wrong Experimental Design**: We tested:
   - GPU: Train on GPU → Eval on GPU
   - NPU: Train on NPU → Eval on NPU

   We should have tested:
   - Train on GPU → Eval on NPU (measure degradation)
   - Train with AQN on GPU → Eval on NPU (test if AQN reduces degradation)

3. **AQN Design Purpose**: AQN was designed for precision mismatch (train FP32/BF16, deploy INT8/INT4), not same-precision cross-hardware deployment.

## Conclusions

### What We Learned

1. **AQN does not improve BF16 training** on non-quantized models
2. **GPU and NPU achieve comparable accuracy** (~76-77%) on GSM8K with Qwen2.5-1.5B
3. **Hardware heterogeneity gap is minimal** for this model/task/precision combination
4. **AQN adds harmful noise** when there's no quantization error to overcome

### Limitations of This Study

1. **Same-hardware evaluation**: We didn't test cross-hardware transfer (train GPU → deploy NPU)
2. **Single model size**: Qwen2.5-1.5B may be too small or too robust to show heterogeneity
3. **Single task**: GSM8K may not be sensitive enough to numerical differences
4. **BF16 only**: Lower precision (INT8) might show more hardware differences

### Future Work to Properly Test Hypothesis

To test AQN for hardware heterogeneity, need:

1. **Cross-hardware deployment test**:
   ```
   Train on GPU → Eval on GPU (reference)
   Train on GPU → Eval on NPU (measure drop)
   Train with AQN on GPU → Eval on NPU (test AQN benefit)
   ```

2. **Scenarios with measurable gap**:
   - Larger models (more numerical sensitivity)
   - Harder tasks (where small errors compound)
   - Lower precision (INT8 where rounding matters more)
   - Tasks known to be sensitive to numerical precision

3. **Quantization experiments**:
   - Train BF16 → Deploy INT8 on NPU
   - This is closer to AQN's intended use case

## Wandb Links

**Project**: https://wandb.ai/vaai/qerl

| Run | Description | URL |
|-----|-------------|-----|
| GPU_baseline | A100 baseline (76.88%) | https://wandb.ai/vaai/qerl/runs/jk4vl0xy |
| NPU_baseline | 910C baseline (76.42%) | https://wandb.ai/vaai/qerl/runs/zxl5x3it |
| NPU_AQN-QeRL_sigma0.05 | Test 3a (75.97%) | https://wandb.ai/vaai/qerl/runs/4o6jt9vg |
| NPU_AQN-Mild_sigma0.025 | Test 4a (74.75%) | https://wandb.ai/vaai/qerl/runs/h5oz00k1 |

## References

- QeRL Paper: Original AQN implementation for quantization robustness
- VERL AQN Implementation: `verl/utils/noise_injection.py`
- Full accuracy analysis: `docs/qerl/AQN_ACCURACY_ANALYSIS.md`
