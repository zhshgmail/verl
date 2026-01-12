# Wandb Metrics Upload Guide

## Overview

This guide explains how to upload VERL training metrics to Weights & Biases for visualization and comparison.

## Setup

### Prerequisites
- Conda environment: `wandb_upload`
- Script: `scripts/upload_log_to_wandb.py`

### Create Conda Environment (if needed)
```bash
conda create -n wandb_upload python=3.10 -y
conda activate wandb_upload
pip install wandb
```

## Wandb Configuration

- **Project**: `qerl`
- **Entity**: `vaai`
- **API Key**: Set via environment variable (see local temp docs for key)

## Usage

### Basic Upload
```bash
export WANDB_API_KEY='your_api_key_here'
export WANDB_ENTITY='vaai'
conda activate wandb_upload

python scripts/upload_log_to_wandb.py \
    --log /path/to/training.log \
    --run-name "descriptive-run-name" \
    --project qerl \
    --entity vaai \
    --tags tag1 tag2
```

### Dry Run (Parse Only)
```bash
python scripts/upload_log_to_wandb.py \
    --log /path/to/training.log \
    --run-name "test" \
    --dry-run
```

## Naming Convention

| Run Type | Name Format | Example |
|----------|-------------|---------|
| Baseline | `baseline-{test_id}-{config}` | `baseline-test2-hbm-optimized` |
| AQN Test | `{test_id}-aqn-sigma{value}` | `test3a-aqn-sigma0.05` |
| Strong AQN | `{test_id}-aqn-strong-sigma{value}` | `test3b-aqn-strong-sigma0.10` |

## Uploaded Runs (Official wandb.ai)

### Project: qerl (NPU experiments)
**URL**: https://wandb.ai/vaai/qerl

| Run Name | Description | Final Accuracy | URL |
|----------|-------------|----------------|-----|
| GPU_baseline | A100 GPU baseline | 76.88% | https://wandb.ai/vaai/qerl/runs/jk4vl0xy |
| NPU_baseline | NPU 910C baseline | 76.42% | https://wandb.ai/vaai/qerl/runs/zxl5x3it |
| NPU_AQN-QeRL_sigma0.05 | Test 3a AQN (QeRL params) | 75.97% | https://wandb.ai/vaai/qerl/runs/4o6jt9vg |
| NPU_AQN-Mild_sigma0.025 | Test 4a AQN (half QeRL) | 74.75% | https://wandb.ai/vaai/qerl/runs/h5oz00k1 |

### Project: aqn (AQN validation experiments)
**URL**: https://wandb.ai/vaai/aqn

#### HW Error Simulation Experiments
| Run Name | Description | Final Accuracy | URL |
|----------|-------------|----------------|-----|
| GPU_baseline_76.88 | A100 GPU baseline (no noise) | 76.88% | https://wandb.ai/vaai/aqn/runs/mow31kkd |
| E5_5pct_noise_only_68.16 | 5% matmul noise, no AQN | 68.16% | https://wandb.ai/vaai/aqn/runs/r759ouw1 |
| E5b_AQN_epoch_aware_70.58 | 5% noise + Epoch-Aware AQN (σ=0.05→0.0005) | 70.58% | https://wandb.ai/vaai/aqn/runs/23fkkt0d |
| E5c_AQN_lower_sigma_67.48 | 5% noise + Lower AQN (σ=0.01→0.00001) | 67.48% | https://wandb.ai/vaai/aqn/runs/iwqcvbca |
| E9a_SRDD_targeted_68.54 | 5% noise + SRDD-targeted (layers 14-17, low σ) | 68.54% | https://wandb.ai/vaai/aqn/runs/j33mvc0e |
| **E9b_SRDD_variable_71.19_BEST** | 5% noise + SRDD-variable sigma (**BEST**) | **71.19%** | https://wandb.ai/vaai/aqn/runs/zeazqz2n |

#### Quantization Experiments (Full Fine-Tuning)
| Run Name | Description | Final Accuracy | URL |
|----------|-------------|----------------|-----|
| E3a_MXFP4_DAPO_baseline | MXFP4 + DAPO baseline | 73.77% | https://wandb.ai/vaai/aqn/runs/0027cbho |
| E4a_NVFP4_DAPO_baseline | NVFP4 + DAPO baseline | 72.55% | https://wandb.ai/vaai/aqn/runs/fztg4c9r |

#### LoRA Experiments
| Run Name | Description | Final Accuracy | URL |
|----------|-------------|----------------|-----|
| E7a_LoRA_BF16_baseline_71.27 | BF16 + LoRA baseline | 71.27% | https://wandb.ai/vaai/aqn/runs/gmvnxbft |
| E5a_LoRA_NVFP4_baseline_63.84 | NVFP4 + LoRA (no AQN) | 63.84% | https://wandb.ai/vaai/aqn/runs/6ysmaz46 |
| E5b_LoRA_NVFP4_AQN_66.11 | NVFP4 + LoRA + AQN (+2.27%) | 66.11% | https://wandb.ai/vaai/aqn/runs/63tmy4nm |
| **E12_MXFP4_LoRA_highσ_AQN_72.48** | MXFP4 + LoRA + high-σ AQN (**+1.21% vs BF16!**) | **72.48%** | https://wandb.ai/vaai/aqn/runs/1cgmoipq |

## Log File Locations

### Local Logs
- `/tmp/baseline_test_2k.log` - Test 2 baseline
- `/tmp/aqn_test_2k.log` - Test 2 "AQN" (INVALID - bug meant no noise applied)

### Remote Logs (on A3Cloud host)
```bash
# Test 3a v2 (verl-a3cloud container)
ssh root@7.150.12.17 "docker exec verl-a3cloud cat /tmp/test3a_qerl_v2.log" > /tmp/test3a_qerl_v2.log

# Test 3b v3 (verl-a3cloud-2 container) - CRASHED
ssh root@7.150.12.17 "docker exec verl-a3cloud-2 cat /tmp/test3b_strong_v3.log" > /tmp/test3b_strong_v3.log
```

## Metrics Parsed

The upload script extracts these key metrics from VERL logs:
- `val-core/openai/gsm8k/acc/mean@1` - GSM8K validation accuracy
- `actor/entropy` - Policy entropy
- `actor/grad_norm` - Gradient norm
- `actor/kl_loss` - KL divergence loss
- `critic/score/mean` - Training reward score
- `response_length/mean` - Average response length
- `perf/throughput` - Training throughput

## Notes

1. **Duplicate Prevention**: The script removes duplicate steps (keeps last occurrence)
2. **ANSI Cleanup**: Color codes are automatically stripped from logs
3. **Step Alignment**: Metrics are logged by training step for easy comparison
