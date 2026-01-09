#!/bin/bash
# MXFP4 Fake Quantization + AQN Training Experiment
#
# Purpose: Test if AQN training can improve model robustness to MXFP4 quantization
#
# Experiment Design:
# 1. Run SRDD scan BEFORE training (baseline metrics)
# 2. Train with MXFP4 fake quantization + AQN (gamma decays to 0)
# 3. Run SRDD scan AFTER training (compare metrics)
#
# Usage:
#   bash scripts/test_mxfp4_aqn_training.sh [N_GPUS] [SIGMA_START] [SIGMA_END] [NUM_STAGES]
#
# Examples:
#   bash scripts/test_mxfp4_aqn_training.sh 8         # 8 GPUs, QeRL defaults (0.05→0.0005, 10 stages)
#   bash scripts/test_mxfp4_aqn_training.sh 4 0.05   # 4 GPUs, custom sigma_start
#   bash scripts/test_mxfp4_aqn_training.sh 4 0.1 0.001 5  # Custom all params

set -x

# Disable WandB online logging
export WANDB_MODE=offline

# Configuration
N_GPUS=${1:-8}
# AQN sigma values (QeRL defaults: sigma_start=0.05, sigma_end=0.0005)
# Note: sigma_end must be > 0 to avoid breaking exponential decay schedule
AQN_SIGMA_START=${2:-0.05}
AQN_SIGMA_END=${3:-0.0005}
AQN_NUM_STAGES=${4:-10}

# Model and data paths (A100 server)
MODEL_PATH=${MODEL_PATH:-"/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"}
TRAIN_DATA=${TRAIN_DATA:-"/data/z00637938/gsm8k/train.parquet"}
VAL_DATA=${VAL_DATA:-"/data/z00637938/gsm8k/test.parquet"}

# Output directory for checkpoints and logs
OUTPUT_DIR="/tmp/mxfp4_aqn_experiment"
mkdir -p ${OUTPUT_DIR}

echo "============================================================"
echo "MXFP4 + AQN Training Experiment"
echo "============================================================"
echo "  Model: ${MODEL_PATH}"
echo "  Train data: ${TRAIN_DATA}"
echo "  Val data: ${VAL_DATA}"
echo "  N GPUs: ${N_GPUS}"
echo "  AQN sigma_start: ${AQN_SIGMA_START}"
echo "  AQN sigma_end: ${AQN_SIGMA_END}"
echo "  AQN num_stages: ${AQN_NUM_STAGES}"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================================"

# Step 1: Run SRDD scan BEFORE training
echo ""
echo "=== Step 1: SRDD Scan BEFORE Training ==="
python scripts/srdd_quant_scanner.py \
    --model_path ${MODEL_PATH} \
    --quant_type mxfp4 \
    --output ${OUTPUT_DIR}/srdd_scan_before.json \
    2>&1 | tee ${OUTPUT_DIR}/srdd_scan_before.log

# Step 2: Train with MXFP4 + AQN
echo ""
echo "=== Step 2: GRPO Training with MXFP4 + AQN ==="

# Common training args
COMMON_ARGS="
    data.train_files=${TRAIN_DATA}
    data.val_files=${VAL_DATA}
    data.train_batch_size=128
    data.max_prompt_length=1024
    data.max_response_length=1024
    data.filter_overlong_prompts=True
    data.truncation=error
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.actor.optim.lr=5e-7
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.ppo_mini_batch_size=32
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8
    actor_rollout_ref.rollout.n=5
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    trainer.critic_warmup=0
    trainer.total_epochs=1
    trainer.test_freq=20
    trainer.project_name=mxfp4_aqn_test
    trainer.n_gpus_per_node=${N_GPUS}
    trainer.val_before_train=True
    trainer.save_freq=50
    trainer.default_local_dir=${OUTPUT_DIR}/checkpoints
"

# MXFP4 Fake Quantization config (applies to both rollout and training)
# Note: Default target_modules=['rmsnorm'] from config will be used
# For linear layers targeting, we need to override via command line
MXFP4_ARGS="
    trainer.hw_error_injection.enabled=True
    trainer.hw_error_injection.error_type=mxfp4
    trainer.hw_error_injection.apply_during=both
"

# AQN (Adaptive Quantization Noise) config
# Uses trainer.noise_injection with sigma_start -> sigma_end decay over num_stages
# QeRL defaults: sigma_start=0.05, sigma_end=0.0005, num_stages=10
# Note: Use ++ prefix to override existing config keys (not + which adds new keys)
AQN_ARGS="
    ++trainer.noise_injection.enabled=True
    ++trainer.noise_injection.sigma_start=${AQN_SIGMA_START}
    ++trainer.noise_injection.sigma_end=${AQN_SIGMA_END}
    ++trainer.noise_injection.num_stages=${AQN_NUM_STAGES}
"

python3 -m verl.trainer.main_ppo \
    --config-name=ppo_trainer \
    ${COMMON_ARGS} \
    ${MXFP4_ARGS} \
    'trainer.hw_error_injection.target_modules=["linear"]' \
    ${AQN_ARGS} \
    trainer.experiment_name=mxfp4_aqn_sigma${AQN_SIGMA_START} \
    2>&1 | tee ${OUTPUT_DIR}/training.log

# Step 3: Run SRDD scan AFTER training
echo ""
echo "=== Step 3: SRDD Scan AFTER Training ==="

# Find the latest checkpoint
CHECKPOINT_DIR=$(ls -td ${OUTPUT_DIR}/checkpoints/*/actor* 2>/dev/null | head -1)
if [ -n "${CHECKPOINT_DIR}" ]; then
    echo "Using checkpoint: ${CHECKPOINT_DIR}"
    python scripts/srdd_quant_scanner.py \
        --model_path ${CHECKPOINT_DIR} \
        --quant_type mxfp4 \
        --output ${OUTPUT_DIR}/srdd_scan_after.json \
        2>&1 | tee ${OUTPUT_DIR}/srdd_scan_after.log
else
    echo "No checkpoint found, scanning original model again for comparison"
    python scripts/srdd_quant_scanner.py \
        --model_path ${MODEL_PATH} \
        --quant_type mxfp4 \
        --output ${OUTPUT_DIR}/srdd_scan_after.json \
        2>&1 | tee ${OUTPUT_DIR}/srdd_scan_after.log
fi

# Step 4: Compare results
echo ""
echo "=== Step 4: Compare Before/After Results ==="
python -c "
import json
import sys

try:
    with open('${OUTPUT_DIR}/srdd_scan_before.json') as f:
        before = json.load(f)
    with open('${OUTPUT_DIR}/srdd_scan_after.json') as f:
        after = json.load(f)

    print('SRDD Scan Comparison:')
    print('=' * 60)

    stats_before = before['report']['statistics']
    stats_after = after['report']['statistics']

    for metric in ['sqnr_db', 'deadzone_ratio', 'relative_error']:
        b_mean = stats_before[metric]['mean']
        a_mean = stats_after[metric]['mean']
        change = (a_mean - b_mean) / b_mean * 100 if b_mean != 0 else 0

        print(f'{metric}:')
        print(f'  Before: {b_mean:.4f}')
        print(f'  After:  {a_mean:.4f}')
        print(f'  Change: {change:+.2f}%')
        print()

    prob_before = before['report']['summary']['problematic_layers']
    prob_after = after['report']['summary']['problematic_layers']
    print(f'Problematic layers:')
    print(f'  Before: {prob_before}')
    print(f'  After:  {prob_after}')

except Exception as e:
    print(f'Error comparing results: {e}')
    sys.exit(1)
"

echo ""
echo "============================================================"
echo "EXPERIMENT COMPLETE"
echo "============================================================"
echo "Results saved to: ${OUTPUT_DIR}"
echo "  - srdd_scan_before.json: Pre-training SRDD metrics"
echo "  - srdd_scan_after.json: Post-training SRDD metrics"
echo "  - training.log: Training logs"
echo "============================================================"
