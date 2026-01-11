#!/bin/bash
# Test Noisy Ops + SRDD-Guided Variable Sigma AQN (E9b) on A100
#
# E9b: Variable sigma AQN - Scale sigma by SRDD error per layer
# This tests whether scaling noise by quantization error improves training.
#
# Previous results:
#   E5 (noise only): 68.16% accuracy
#   E5b (noise + epoch-aware AQN σ=0.05→0.0005): 70.58%
#   E5c (noise + lower AQN σ=0.01→0.00001): TBD
#   E9a (noise + targeted AQN layers 14-17): TBD
#
# E9b hypothesis:
#   Scaling sigma by SRDD error may provide better robustness:
#   - High-error layers (14-17): 1.5x sigma
#   - Medium-error layers (10-13, 18-21): 1.2x sigma
#   - Low-error layers (0-9, 22-27): 1.0x sigma
#
# Usage: bash scripts/test_noisy_ops_srdd_variable.sh [ERROR_SCALE] [N_GPUS]

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
    trainer.project_name=noisy_ops_srdd_variable_test
    trainer.n_gpus_per_node=${N_GPUS}
    trainer.val_before_train=True
"

echo "=== Running Noisy Ops + SRDD-Guided Variable Sigma AQN Test (E9b) ==="
echo "Error scale: ${ERROR_SCALE}"
echo "N GPUs: ${N_GPUS}"
echo ""
echo "Environment variables set:"
echo "  VERL_NOISY_OPS_ENABLED=${VERL_NOISY_OPS_ENABLED}"
echo "  VERL_NOISY_OPS_SCALE=${VERL_NOISY_OPS_SCALE}"
echo "  VERL_NOISY_OPS_TYPE=${VERL_NOISY_OPS_TYPE}"
echo ""
echo "E9b: SRDD-Guided VARIABLE SIGMA AQN"
echo "  - High-error layers (14-17): sigma × 1.5"
echo "  - Medium-error layers (10-13, 18-21): sigma × 1.2"
echo "  - Low-error layers (0-9, 22-27): sigma × 1.0"
echo "  - sigma_start: 0.01, sigma_end: 0.00001"
echo ""
echo "Comparison with previous experiments:"
echo "  E5 (noise only): 68.16%"
echo "  E5b (noise + epoch-aware AQN): 70.58%"
echo "  E5c (noise + lower AQN): TBD"
echo "  E9a (noise + targeted AQN): TBD"
echo "  E9b (noise + variable sigma AQN): TBD"
echo ""

# SRDD-guided layer sigma config with variable multipliers
# All 28 layers of Qwen2.5-1.5B with SRDD-based multipliers
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
    ++trainer.noise_injection.layer_sigma_config.default_multiplier=1.0 \
    '++trainer.noise_injection.layer_sigma_config.layer_multipliers={"10": 1.2, "11": 1.2, "12": 1.2, "13": 1.2, "14": 1.5, "15": 1.5, "16": 1.5, "17": 1.5, "18": 1.2, "19": 1.2, "20": 1.2, "21": 1.2}' \
    trainer.experiment_name=noisy_ops_srdd_variable_${ERROR_SCALE}

echo "=== E9b Test Complete ==="
echo "Compare result with:"
echo "  E5b (epoch-aware AQN): 70.58%"
echo "  E5c (lower AQN): TBD"
echo "  E9a (targeted AQN): TBD"
