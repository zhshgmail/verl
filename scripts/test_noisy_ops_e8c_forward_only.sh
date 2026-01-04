#!/bin/bash
# E8c: Forward-Only Noise + Epoch-Aware AQN
#
# This experiment tests the theory that forward noise (activations) is
# responsible for inference robustness, while backward noise (gradients)
# provides training regularization.
#
# Configuration:
#   - Model: 1.5B (same as E5b)
#   - Noise: 5% forward-only (no backward noise)
#   - AQN: Epoch-aware scheduling
#
# Expected outcome based on theory:
#   - Training stability: Slightly worse than E5b (less gradient regularization)
#   - Inference robustness: Better than E5b (training noise matches inference noise)
#
# Usage: bash scripts/test_noisy_ops_e8c_forward_only.sh

set -x

# Disable WandB online logging
export WANDB_MODE=offline

# Configuration
ERROR_SCALE=${1:-5e-2}
N_GPUS=${2:-8}

# Enable operator-level noisy ops with FORWARD-ONLY mode
export VERL_NOISY_OPS_ENABLED=1
export VERL_NOISY_OPS_SCALE=${ERROR_SCALE}
export VERL_NOISY_OPS_TYPE=relative_gaussian
export VERL_NOISY_OPS_FORWARD_ONLY=1  # KEY: Only inject noise in forward pass

# Model and data paths
MODEL_PATH=${MODEL_PATH:-"/data/models/Qwen2.5-1.5B-Instruct"}
TRAIN_DATA=${TRAIN_DATA:-"/data/datasets/gsm8k/train.parquet"}
VAL_DATA=${VAL_DATA:-"/data/datasets/gsm8k/test.parquet"}

# Common training args (same as E5b)
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
    trainer.project_name=noisy_ops_e8c_forward_only
    trainer.n_gpus_per_node=${N_GPUS}
    trainer.val_before_train=True
"

echo "=== E8c: Forward-Only Noise + Epoch-Aware AQN ==="
echo ""
echo "Theory being tested:"
echo "  Forward noise (activations) -> Inference robustness"
echo "  Backward noise (gradients)  -> Training regularization"
echo ""
echo "Configuration:"
echo "  Error scale: ${ERROR_SCALE}"
echo "  N GPUs: ${N_GPUS}"
echo "  VERL_NOISY_OPS_ENABLED=${VERL_NOISY_OPS_ENABLED}"
echo "  VERL_NOISY_OPS_SCALE=${VERL_NOISY_OPS_SCALE}"
echo "  VERL_NOISY_OPS_TYPE=${VERL_NOISY_OPS_TYPE}"
echo "  VERL_NOISY_OPS_FORWARD_ONLY=${VERL_NOISY_OPS_FORWARD_ONLY}"
echo ""
echo "Expected results:"
echo "  E5b (both): Training +2.42%, Robustness -14%"
echo "  E8c (forward-only): Training +1-2%?, Robustness BETTER?"
echo ""

python3 -m verl.trainer.main_ppo \
    --config-name=ppo_trainer \
    ${COMMON_ARGS} \
    ++trainer.noisy_ops.enabled=True \
    ++trainer.noisy_ops.error_scale=${ERROR_SCALE} \
    ++trainer.noisy_ops.error_type=relative_gaussian \
    ++trainer.noise_injection.enabled=True \
    ++trainer.noise_injection.epoch_aware=True \
    ++trainer.noise_injection.sigma_start=0.05 \
    ++trainer.noise_injection.sigma_end=0.0005 \
    ++trainer.noise_injection.stages_per_epoch=5 \
    trainer.experiment_name=e8c_forward_only_${ERROR_SCALE}

echo "=== E8c Complete ==="
echo "Next step: Run robustness evaluation with native PyTorch"
