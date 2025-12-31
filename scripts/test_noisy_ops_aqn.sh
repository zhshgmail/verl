#!/bin/bash
# Test Noisy Ops + AQN (Adaptive Quantization Noise) on A100
#
# This is E5a in the experimental series - testing if AQN can mitigate
# the accuracy degradation caused by FP4-level noise injection.
#
# Key hypothesis:
#   E5 (noise only): 68.16% accuracy (-8.72% vs baseline)
#   E5a (noise + AQN): Should be > 68.16% if AQN helps
#
# Usage: bash scripts/test_noisy_ops_aqn.sh [ERROR_SCALE] [N_GPUS]
#
# AQN Configuration (QeRL original parameters):
#   sigma_start: 0.05 (high noise early for exploration)
#   sigma_end: 0.0005 (low noise late for exploitation)
#   num_stages: 10 (gradual decay)

set -x

# Disable WandB online logging (avoids API key requirement)
export WANDB_MODE=offline

# Configuration
ERROR_SCALE=${1:-5e-2}
N_GPUS=${2:-8}

# Enable operator-level noisy ops via environment variables
# This simulates FP4-level quantization error in all matmul operations
export VERL_NOISY_OPS_ENABLED=1
export VERL_NOISY_OPS_SCALE=${ERROR_SCALE}
export VERL_NOISY_OPS_TYPE=relative_gaussian

# Model and data paths (adjust for your setup)
MODEL_PATH=${MODEL_PATH:-"/data/models/Qwen2.5-1.5B-Instruct"}
TRAIN_DATA=${TRAIN_DATA:-"/data/datasets/gsm8k/train.parquet"}
VAL_DATA=${VAL_DATA:-"/data/datasets/gsm8k/test.parquet"}

# Common training args (matching run_gpu_baseline.sh exactly)
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
    trainer.save_freq=-1
    trainer.nnodes=1
    trainer.project_name=noisy_ops_aqn_test
    trainer.n_gpus_per_node=${N_GPUS}
    trainer.val_before_train=True
"

echo "=== Running Noisy Ops + AQN Test (E5a) ==="
echo "Error scale: ${ERROR_SCALE}"
echo "N GPUs: ${N_GPUS}"
echo ""
echo "Environment variables set:"
echo "  VERL_NOISY_OPS_ENABLED=${VERL_NOISY_OPS_ENABLED}"
echo "  VERL_NOISY_OPS_SCALE=${VERL_NOISY_OPS_SCALE}"
echo "  VERL_NOISY_OPS_TYPE=${VERL_NOISY_OPS_TYPE}"
echo ""
echo "This test combines:"
echo "  1. Noisy ops: 5% error in ALL matmul operations (simulates FP4)"
echo "  2. AQN: Adaptive noise injection in RMSNorm (QeRL technique)"
echo ""
echo "Expected outcome:"
echo "  E5 (noise only): 68.16%"
echo "  E5a (noise + AQN): > 68.16% if AQN helps"
echo ""

python3 -m verl.trainer.main_ppo \
    --config-name=ppo_trainer \
    ${COMMON_ARGS} \
    ++trainer.noisy_ops.enabled=True \
    ++trainer.noisy_ops.error_scale=${ERROR_SCALE} \
    ++trainer.noisy_ops.error_type=relative_gaussian \
    ++trainer.noise_injection.enabled=True \
    ++trainer.noise_injection.sigma_start=0.05 \
    ++trainer.noise_injection.sigma_end=0.0005 \
    ++trainer.noise_injection.num_stages=10 \
    trainer.experiment_name=noisy_ops_aqn_${ERROR_SCALE}

echo "=== Test Complete ==="
