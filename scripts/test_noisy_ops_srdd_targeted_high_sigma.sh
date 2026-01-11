#!/bin/bash
# E9a-high-σ: SRDD-Targeted AQN with HIGH Sigma (Isolate Layer Targeting Effect)
#
# Experiment ID: E9a-high-σ
# Date: 2026-01-12
#
# Critical Validation:
# - E5b (high σ=0.05, all layers): 70.58%
# - E9a (low σ=0.01, targeted layers 14-17): 68.54%
# - Problem: E5b vs E9a differs in TWO variables!
#
# This experiment isolates the layer targeting effect by:
# - Using HIGH sigma (0.05→0.0005) matching E5b
# - Targeting only layers 14-17 (like E9a)
#
# Comparison:
# - E5b (high σ, all layers): 70.58%
# - E9a (low σ, targeted): 68.54%
# - E9a-high-σ (high σ, targeted): ???
#
# Hypothesis:
# - If E9a-high-σ ≈ E5b: Layer targeting provides no additional benefit
# - If E9a-high-σ > E5b: Targeting high-error layers IS beneficial
# - If E9a-high-σ < E5b: All-layer AQN is better than targeted
#
# Usage: bash scripts/test_noisy_ops_srdd_targeted_high_sigma.sh [ERROR_SCALE] [N_GPUS]

set -x

export WANDB_MODE=offline

# Configuration
ERROR_SCALE=${1:-5e-2}
N_GPUS=${2:-8}

# Enable operator-level noisy ops
export VERL_NOISY_OPS_ENABLED=1
export VERL_NOISY_OPS_SCALE=${ERROR_SCALE}
export VERL_NOISY_OPS_TYPE=relative_gaussian

MODEL_PATH=${MODEL_PATH:-"/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"}
TRAIN_DATA=${TRAIN_DATA:-"/data/z00637938/gsm8k/train.parquet"}
VAL_DATA=${VAL_DATA:-"/data/z00637938/gsm8k/test.parquet"}

OUTPUT_DIR="/tmp/noisy_ops_srdd_targeted_high_sigma"
mkdir -p ${OUTPUT_DIR}

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
    trainer.project_name=noisy_ops_srdd_targeted_high_sigma_test
    trainer.n_gpus_per_node=${N_GPUS}
    trainer.val_before_train=True
"

echo "=== E9a-high-σ: SRDD-Targeted AQN with HIGH Sigma ==="
echo ""
echo "CRITICAL VALIDATION:"
echo "  - E5b (high σ=0.05, all layers): 70.58%"
echo "  - E9a (low σ=0.01, targeted): 68.54%"
echo "  - Problem: These differ in TWO variables!"
echo ""
echo "This experiment isolates layer targeting effect by:"
echo "  - HIGH sigma 0.05→0.0005 (matching E5b)"
echo "  - TARGETED layers 14-17 only (matching E9a)"
echo ""
echo "Error scale: ${ERROR_SCALE}"
echo "N GPUs: ${N_GPUS}"
echo ""
echo "Environment variables:"
echo "  VERL_NOISY_OPS_ENABLED=${VERL_NOISY_OPS_ENABLED}"
echo "  VERL_NOISY_OPS_SCALE=${VERL_NOISY_OPS_SCALE}"
echo ""
echo "Expected outcomes:"
echo "  - E9a-high-σ ≈ E5b (70.58%): Targeting provides NO additional benefit"
echo "  - E9a-high-σ > E5b: Targeting IS beneficial"
echo "  - E9a-high-σ < E5b: All-layer AQN is better"
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
    ++trainer.noise_injection.layer_sigma_config.enabled=True \
    ++trainer.noise_injection.layer_sigma_config.default_multiplier=0.0 \
    '++trainer.noise_injection.layer_sigma_config.layer_multipliers={14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0}' \
    trainer.experiment_name=e9a_high_sigma_srdd_targeted_${ERROR_SCALE} \
    2>&1 | tee ${OUTPUT_DIR}/training.log

echo ""
echo "=== E9a-high-σ Complete ==="
echo "Results in: ${OUTPUT_DIR}"
echo ""
echo "COMPARISON TABLE:"
echo "  E5b  (high σ, all layers):     70.58%"
echo "  E9a  (low σ, targeted):        68.54%"
echo "  E9a-high-σ (high σ, targeted): CHECK LOG"
echo ""
echo "CONCLUSION:"
echo "  - If result ≈ 70.58%: Sigma matters, targeting doesn't"
echo "  - If result > 70.58%: Targeting + high sigma is OPTIMAL"
echo "  - If result < 70.58%: All-layer AQN is preferred"
