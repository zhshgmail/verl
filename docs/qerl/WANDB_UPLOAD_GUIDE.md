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

**Project**: https://wandb.ai/vaai/qerl

| Run Name | Description | Final Accuracy | URL |
|----------|-------------|----------------|-----|
| GPU_baseline | A100 GPU baseline | 76.88% | https://wandb.ai/vaai/qerl/runs/jk4vl0xy |
| NPU_baseline | NPU 910C baseline | 76.42% | https://wandb.ai/vaai/qerl/runs/zxl5x3it |
| NPU_AQN-QeRL_sigma0.05 | Test 3a AQN (QeRL params) | 75.97% | https://wandb.ai/vaai/qerl/runs/4o6jt9vg |
| NPU_AQN-Mild_sigma0.025 | Test 4a AQN (half QeRL) | 74.75% | https://wandb.ai/vaai/qerl/runs/h5oz00k1 |

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
