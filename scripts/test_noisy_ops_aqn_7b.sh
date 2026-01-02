#!/bin/bash
# E7: Test Noisy Ops + Epoch-Aware AQN on Qwen2.5-7B-Instruct
#
# Purpose: Validate AQN effectiveness at 7B scale
# This extends E5b findings (1.5B) to larger model scale.
#
# Key differences from 1.5B:
#   - Model: Qwen2.5-7B-Instruct (~4.5x larger)
#   - Reduced batch sizes for memory constraints
#   - Using TP=2 for vLLM rollout
#
# Expected results (based on 1.5B findings):
#   - AQN should improve accuracy by ~2% over noise-only baseline
#   - Model should show <1% degradation in robustness testing
#
# Usage: bash scripts/test_noisy_ops_aqn_7b.sh [ERROR_SCALE] [N_GPUS]

set -x

# Disable WandB online logging
export WANDB_MODE=offline

# Configuration
ERROR_SCALE=${1:-5e-2}
N_GPUS=${2:-8}

# Enable operator-level noisy ops (matmul only, matching E5b)
export VERL_NOISY_OPS_ENABLED=1
export VERL_NOISY_OPS_SCALE=${ERROR_SCALE}
export VERL_NOISY_OPS_TYPE=relative_gaussian

# Model and data paths
MODEL_PATH=${MODEL_PATH:-"/data/g30067331/Qwen2.5-7B-Instruct"}
TRAIN_DATA=${TRAIN_DATA:-"/data/z00637938/gsm8k/train.parquet"}
VAL_DATA=${VAL_DATA:-"/data/z00637938/gsm8k/test.parquet"}

# Training args adjusted for 7B model
# Key changes from 1.5B:
#   - Smaller batch sizes to fit in memory
#   - TP=2 for vLLM rollout (7B needs more memory)
#   - Lower GPU memory utilization for stability
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
    trainer.project_name=noisy_ops_aqn_7b_test
    trainer.n_gpus_per_node=${N_GPUS}
    trainer.val_before_train=True
"

echo "=== E7: Noisy Ops + Epoch-Aware AQN on Qwen2.5-7B ==="
echo "Model: Qwen2.5-7B-Instruct"
echo "Error scale: ${ERROR_SCALE}"
echo "N GPUs: ${N_GPUS}"
echo ""
echo "Environment variables set:"
echo "  VERL_NOISY_OPS_ENABLED=${VERL_NOISY_OPS_ENABLED}"
echo "  VERL_NOISY_OPS_SCALE=${VERL_NOISY_OPS_SCALE}"
echo "  VERL_NOISY_OPS_TYPE=${VERL_NOISY_OPS_TYPE}"
echo ""
echo "Key config differences from 1.5B:"
echo "  - train_batch_size: 64 (was 128)"
echo "  - ppo_mini_batch_size: 16 (was 32)"
echo "  - ppo_micro_batch_size_per_gpu: 2 (was 4)"
echo "  - tensor_model_parallel_size: 2 (was 1)"
echo "  - lr: 1e-7 (was 5e-7)"
echo "  - max_response_length: 512 (was 1024)"
echo ""
echo "Epoch-aware AQN schedule:"
echo "  Epoch 1: sigma 0.05 → 0.01"
echo "  Epoch 2: sigma 0.01 → 0.0005"
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
    trainer.experiment_name=noisy_ops_aqn_7b_${ERROR_SCALE}

echo "=== E7 Test Complete ==="
