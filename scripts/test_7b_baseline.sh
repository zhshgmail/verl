#!/bin/bash
# E7a: 7B Baseline (No noise, No AQN)
#
# Purpose: Establish clean baseline for Qwen2.5-7B-Instruct on GSM8K
# This is the control experiment for E7b and E7c comparison.
#
# Usage: bash scripts/test_7b_baseline.sh [N_GPUS]

set -x

# Disable WandB online logging
export WANDB_MODE=offline

# Configuration
N_GPUS=${1:-8}

# NO noise injection
export VERL_NOISY_OPS_ENABLED=0

# Model and data paths
MODEL_PATH=${MODEL_PATH:-"/data/g30067331/Qwen2.5-7B-Instruct"}
TRAIN_DATA=${TRAIN_DATA:-"/data/z00637938/gsm8k/train.parquet"}
VAL_DATA=${VAL_DATA:-"/data/z00637938/gsm8k/test.parquet"}

# Training args (same as E7c for fair comparison)
COMMON_ARGS="
    data.train_files=${TRAIN_DATA}
    data.val_files=${VAL_DATA}
    data.train_batch_size=64
    data.max_prompt_length=1024
    data.max_response_length=512
    data.filter_overlong_prompts=True
    data.truncation=error
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.actor.optim.lr=1e-7
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.ppo_mini_batch_size=16
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.tensor_model_parallel_size=2
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7
    actor_rollout_ref.rollout.n=5
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    trainer.critic_warmup=0
    trainer.logger=console
    trainer.total_epochs=2
    trainer.test_freq=20
    trainer.save_freq=-1
    trainer.nnodes=1
    trainer.project_name=7b_baseline_test
    trainer.n_gpus_per_node=${N_GPUS}
    trainer.val_before_train=True
"

echo "=== E7a: 7B Baseline (No Noise, No AQN) ==="
echo "Model: Qwen2.5-7B-Instruct"
echo "N GPUs: ${N_GPUS}"
echo ""
echo "Configuration:"
echo "  - NO noise injection"
echo "  - NO AQN"
echo "  - This establishes clean baseline for 7B"
echo ""

python3 -m verl.trainer.main_ppo \
    --config-name=ppo_trainer \
    ${COMMON_ARGS} \
    trainer.experiment_name=7b_baseline

echo "=== E7a Baseline Complete ==="
