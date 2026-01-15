#!/bin/bash
# Test Noisy Ops + SRDD-Guided Targeted AQN (E9a) on A100
#
# E9a: Targeted AQN - Only inject noise to high-error layers (14-17)
# This tests whether targeted AQN can speed up training while maintaining robustness.
#
# Previous results:
#   E5 (noise only): 68.16% accuracy
#   E5b (noise + epoch-aware AQN σ=0.05→0.0005): 70.58%
#   E5c (noise + lower AQN σ=0.01→0.00001): TBD
#
# E9a hypothesis:
#   Targeting only high-error layers (14-17) based on SRDD analysis
#   may reduce training overhead while maintaining robustness benefit.
#
# SRDD Analysis (Qwen2.5-1.5B-Instruct):
#   - Layers 14-17: 40.8-42.7% relative error (highest)
#   - Other layers: 28-40% relative error
#
# Usage: bash scripts/test_noisy_ops_srdd_targeted.sh [ERROR_SCALE] [N_GPUS]

set -x

# Disable WandB online logging (avoids API key requirement)
export WANDB_MODE=offline

# Configuration
ERROR_SCALE=${1:-5e-2}
N_GPUS=${2:-8}

# Enable operator-level noisy ops via environment variables
export VERL_NOISY_OPS_ENABLED=1
export VERL_NOISY_OPS_SCALE=${ERROR_SCALE}
export VERL_NOISY_OPS_TYPE=relative_gaussian

# Model and data paths (adjust for your setup)
MODEL_PATH=${MODEL_PATH:-"/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"}
TRAIN_DATA=${TRAIN_DATA:-"/data/z00637938/gsm8k/train.parquet"}
VAL_DATA=${VAL_DATA:-"/data/z00637938/gsm8k/test.parquet"}

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
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8
    actor_rollout_ref.rollout.n=5
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    trainer.critic_warmup=0
    trainer.logger=console
    trainer.total_epochs=2
    trainer.test_freq=20
    trainer.save_freq=58
    trainer.nnodes=1
    trainer.project_name=noisy_ops_srdd_targeted_test
    trainer.n_gpus_per_node=${N_GPUS}
    trainer.val_before_train=True
"

echo "=== Running Noisy Ops + SRDD-Guided Targeted AQN Test (E9a) ==="
echo "Error scale: ${ERROR_SCALE}"
echo "N GPUs: ${N_GPUS}"
echo ""
echo "Environment variables set:"
echo "  VERL_NOISY_OPS_ENABLED=${VERL_NOISY_OPS_ENABLED}"
echo "  VERL_NOISY_OPS_SCALE=${VERL_NOISY_OPS_SCALE}"
echo "  VERL_NOISY_OPS_TYPE=${VERL_NOISY_OPS_TYPE}"
echo ""
echo "E9a: SRDD-Guided TARGETED AQN"
echo "  - Only inject noise to layers 14-17 (high-error layers)"
echo "  - Other layers: NO AQN (multiplier=0)"
echo "  - sigma_start: 0.01, sigma_end: 0.00001"
echo ""
echo "Comparison with previous experiments:"
echo "  E5 (noise only): 68.16%"
echo "  E5b (noise + epoch-aware AQN): 70.58%"
echo "  E5c (noise + lower AQN): TBD"
echo "  E9a (noise + targeted AQN layers 14-17): TBD"
echo ""

# SRDD-guided layer sigma config for targeted AQN (only layers 14-17)
# Using Hydra's nested key syntax for layer_multipliers
python3 -m verl.trainer.main_ppo \
    --config-name=ppo_trainer \
    ${COMMON_ARGS} \
    ++trainer.noisy_ops.enabled=True \
    ++trainer.noisy_ops.error_scale=${ERROR_SCALE} \
    ++trainer.noisy_ops.error_type=relative_gaussian \
    ++trainer.noise_injection.enabled=True \
    ++trainer.noise_injection.epoch_aware=True \
    ++trainer.noise_injection.sigma_start=0.01 \
    ++trainer.noise_injection.sigma_end=0.00001 \
    ++trainer.noise_injection.stages_per_epoch=5 \
    ++trainer.noise_injection.layer_sigma_config.enabled=True \
    ++trainer.noise_injection.layer_sigma_config.default_multiplier=0.0 \
    '++trainer.noise_injection.layer_sigma_config.layer_multipliers={14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0}' \
    trainer.experiment_name=noisy_ops_srdd_targeted_${ERROR_SCALE}

echo "=== E9a Test Complete ==="
echo "Compare result with:"
echo "  E5b (epoch-aware AQN): 70.58%"
echo "  E5c (lower AQN): TBD"
