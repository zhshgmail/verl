#!/bin/bash
# W4A4 MXFP4 + LoRA Baseline (No AQN)
#
# Experiment ID: E13a
# Date: 2026-01-14
#
# Configuration:
# - W4A4 MXFP4 fake quantization (~35-50% error, both weights AND activations)
# - 16-bit LoRA (rank=32, alpha=16) on all linear layers
# - lm_head, embed_tokens, lora_A, lora_B EXCLUDED from quantization
# - NO AQN - baseline degradation measurement
# - DAPO algorithm, 1 epoch
#
# Purpose: Measure how severe W4A4 degradation is compared to W4A16 (E6a: 65.88%)
# Expected: 60-62% (worse than W4A16 due to activation quantization)
#
# Usage:
#   bash scripts/test_mxfp4_w4a4_lora_baseline.sh [N_GPUS]

set -x

N_GPUS=${1:-8}

export WANDB_MODE=offline

MODEL_PATH=${MODEL_PATH:-"/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"}
TRAIN_DATA=${TRAIN_DATA:-"/data/z00637938/gsm8k/train.parquet"}
VAL_DATA=${VAL_DATA:-"/data/z00637938/gsm8k/test.parquet"}

OUTPUT_DIR="/tmp/mxfp4_w4a4_lora_baseline_e13a"
mkdir -p ${OUTPUT_DIR}

# DAPO-specific settings
adv_estimator=grpo
use_kl_in_reward=False
use_kl_loss=False
kl_loss_coef=0.0

# Asymmetric clipping (DAPO core)
clip_ratio_low=0.2
clip_ratio_high=0.25

# Response length settings
max_prompt_length=1024
max_response_length=1024

# Overlong buffer penalty
enable_overlong_buffer=True
overlong_buffer_len=256
overlong_penalty_factor=0.5

# Token-level loss
loss_agg_mode="token-mean"

# Dynamic sampling / filter groups
enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=5

# Batch sizes (same as E6a/E6b)
train_batch_size=128
gen_batch_size=256
ppo_mini_batch_size=32
n_resp_per_prompt=8

# LoRA settings
lora_rank=32
lora_alpha=16

echo "=== W4A4 MXFP4 + LoRA Baseline (E13a) ==="
echo "Key settings:"
echo "  - W4A4 MXFP4 fake quantization (weights + activations)"
echo "  - injection_point=both (NEW - W4A4 mode)"
echo "  - 16-bit LoRA: rank=${lora_rank}, alpha=${lora_alpha}"
echo "  - exclude_modules=['lm_head', 'embed_tokens', 'lora_A', 'lora_B']"
echo "  - NO AQN - measuring baseline degradation"
echo "  - 1 epoch"
echo ""
echo "Expected: 60-62% (worse than W4A16 E6a: 65.88%)"

python3 -m recipe.dapo.main_dapo \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.train_batch_size=${train_batch_size} \
    data.gen_batch_size=${gen_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.lora_rank=${lora_rank} \
    actor_rollout_ref.model.lora_alpha=${lora_alpha} \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    reward_model.overlong_buffer.log=True \
    trainer.critic_warmup=0 \
    trainer.total_epochs=1 \
    trainer.test_freq=20 \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.val_before_train=True \
    trainer.save_freq=50 \
    trainer.hw_error_injection.enabled=True \
    trainer.hw_error_injection.error_type=mxfp4 \
    trainer.hw_error_injection.injection_point=both \
    trainer.hw_error_injection.apply_during=both \
    'trainer.hw_error_injection.target_modules=["linear"]' \
    '++trainer.hw_error_injection.exclude_modules=["lm_head", "embed_tokens", "lora_A", "lora_B"]' \
    ++trainer.hw_error_injection.use_ste=True \
    trainer.default_local_dir=${OUTPUT_DIR}/checkpoints \
    trainer.project_name=w4a4_e13a_baseline \
    trainer.experiment_name=w4a4_mxfp4_lora_baseline_1ep \
    2>&1 | tee ${OUTPUT_DIR}/training.log

echo ""
echo "E13a W4A4 baseline experiment complete!"
echo "Results in: ${OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "  1. Run E13b (global AQN): scripts/test_mxfp4_w4a4_lora_global_aqn.sh"
echo "  2. Run E13d (SRDD-variable AQN): scripts/test_mxfp4_w4a4_lora_srdd_variable.sh"
