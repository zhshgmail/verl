#!/bin/bash
# MXFP4 W4A16 (Weight Quantization) + AQN Training Experiment
#
# Purpose: Test if W4A16 (quantize weights, keep activations FP16) works better than W16A4
#
# Key difference from previous experiment:
# - Previous (W16A4): injection_point=input  -> quantize activations
# - This (W4A16):     injection_point=weight -> quantize weights (QeRL style)
#
# Expected:
# - W4A16 should have much better accuracy than W16A4
# - QeRL achieves 90.8% on GSM8k with NVFP4 (W4A16)
# - Our W16A4 achieved only 8-10% due to activation quantization
#
# Usage:
#   bash scripts/test_mxfp4_w4a16_training.sh [N_GPUS]

set -x

# Disable WandB online logging
export WANDB_MODE=offline

# Configuration
N_GPUS=${1:-8}

# Model and data paths (A100 server)
MODEL_PATH=${MODEL_PATH:-"/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"}
TRAIN_DATA=${TRAIN_DATA:-"/data/z00637938/gsm8k/train.parquet"}
VAL_DATA=${VAL_DATA:-"/data/z00637938/gsm8k/test.parquet"}

# Output directory
OUTPUT_DIR="/tmp/mxfp4_w4a16_experiment"
mkdir -p ${OUTPUT_DIR}

echo "============================================================"
echo "MXFP4 W4A16 Training Experiment (Weight Quantization)"
echo "============================================================"
echo "  Model: ${MODEL_PATH}"
echo "  Train data: ${TRAIN_DATA}"
echo "  Val data: ${VAL_DATA}"
echo "  N GPUs: ${N_GPUS}"
echo "  Injection point: WEIGHT (W4A16 - QeRL style)"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================================"

# Step 1: Quick baseline test without quantization
echo ""
echo "=== Step 1: Baseline Test (No Quantization) ==="

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
    trainer.project_name=mxfp4_w4a16_test
    trainer.n_gpus_per_node=${N_GPUS}
    trainer.val_before_train=True
    trainer.save_freq=50
    trainer.default_local_dir=${OUTPUT_DIR}/checkpoints
"

# W4A16 MXFP4 config - KEY CHANGE: injection_point=weight
MXFP4_W4A16_ARGS="
    trainer.hw_error_injection.enabled=True
    trainer.hw_error_injection.error_type=mxfp4
    trainer.hw_error_injection.injection_point=weight
    trainer.hw_error_injection.apply_during=both
"

# AQN config (QeRL defaults)
AQN_ARGS="
    ++trainer.noise_injection.enabled=True
    ++trainer.noise_injection.sigma_start=0.05
    ++trainer.noise_injection.sigma_end=0.0005
    ++trainer.noise_injection.num_stages=10
"

# Step 2: Train with W4A16 MXFP4 + AQN
echo ""
echo "=== Step 2: GRPO Training with W4A16 MXFP4 + AQN ==="

python3 -m verl.trainer.main_ppo \
    --config-name=ppo_trainer \
    ${COMMON_ARGS} \
    ${MXFP4_W4A16_ARGS} \
    'trainer.hw_error_injection.target_modules=["linear"]' \
    ${AQN_ARGS} \
    trainer.experiment_name=mxfp4_w4a16 \
    2>&1 | tee ${OUTPUT_DIR}/training.log

echo ""
echo "============================================================"
echo "W4A16 EXPERIMENT COMPLETE"
echo "============================================================"
echo "Results saved to: ${OUTPUT_DIR}"
echo "  - training.log: Training logs"
echo "============================================================"
