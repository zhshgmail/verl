#!/bin/bash
# MXFP4 v3.0: DAPO with overlong penalty (1 epoch)
#
# Experiment ID: v3.0 (E3a)
# Date: 2026-01-09
#
# Key Changes from v2.x (PPO/GRPO):
# - Uses DAPO algorithm with overlong buffer penalty
# - Asymmetric clipping (clip_low=0.2, clip_high=0.25)
# - Token-level loss aggregation
# - Dynamic sampling with filter_groups
# - 1 epoch only (DAPO paper standard)
#
# DAPO Analysis:
# - Overlong buffer penalty: Prevents length explosion (207->410 tokens)
# - Does NOT prevent entropy collapse (0.27->0.11)
# - MXFP4's 21% error compounds over epochs, so 1 epoch is safer
#
# Configuration:
# - MXFP4 W4A16 quantization
# - lm_head and embed_tokens EXCLUDED
# - DAPO overlong penalty: buffer=256, penalty=0.5
# - 1 epoch (DAPO standard)
#
# Usage:
#   bash scripts/test_mxfp4_v3.0_dapo.sh [N_GPUS]

set -x

N_GPUS=${1:-8}

export WANDB_MODE=offline

MODEL_PATH=${MODEL_PATH:-"/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"}
TRAIN_DATA=${TRAIN_DATA:-"/data/z00637938/gsm8k/train.parquet"}
VAL_DATA=${VAL_DATA:-"/data/z00637938/gsm8k/test.parquet"}

OUTPUT_DIR="/tmp/mxfp4_v3.0_dapo"
mkdir -p ${OUTPUT_DIR}

# DAPO-specific settings
adv_estimator=grpo
use_kl_in_reward=False
use_kl_loss=False
kl_loss_coef=0.0

# Asymmetric clipping (DAPO core) - reduced for MXFP4 stability
clip_ratio_low=0.2
clip_ratio_high=0.25  # DAPO default is 0.28, reduced for MXFP4

# Response length settings
max_prompt_length=1024
max_response_length=1024

# Overlong buffer penalty (DAPO's length hack prevention)
enable_overlong_buffer=True
overlong_buffer_len=256  # 25% of max_response_length
overlong_penalty_factor=0.5  # Gentle for MXFP4 (DAPO default is 1.0)

# Token-level loss (DAPO standard)
loss_agg_mode="token-mean"

# Dynamic sampling / filter groups
enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=5  # Reduced from DAPO's 10 for MXFP4

# Batch sizes
train_batch_size=128
gen_batch_size=256  # 2x train for dynamic sampling
ppo_mini_batch_size=32
n_resp_per_prompt=8  # DAPO uses 16, reduced for 1.5B model

echo "=== MXFP4 v3.0: DAPO with Overlong Penalty, 1 Epoch ==="
echo "Key settings:"
echo "  - MXFP4 W4A16 quantization"
echo "  - exclude_modules=['lm_head', 'embed_tokens']"
echo "  - DAPO overlong penalty: buffer=${overlong_buffer_len}, penalty=${overlong_penalty_factor}"
echo "  - Asymmetric clipping: low=${clip_ratio_low}, high=${clip_ratio_high}"
echo "  - Token-level loss: ${loss_agg_mode}"
echo "  - Dynamic sampling: filter_groups enabled"
echo "  - 1 epoch (DAPO standard)"
echo ""
echo "Expected improvements over v2.x:"
echo "  - No response length explosion (overlong penalty)"
echo "  - Stable training (asymmetric clipping)"
echo "  - Better sample efficiency (dynamic sampling)"

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
    actor_rollout_ref.actor.optim.lr=5e-7 \
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
    trainer.hw_error_injection.injection_point=weight \
    trainer.hw_error_injection.apply_during=both \
    'trainer.hw_error_injection.target_modules=["linear"]' \
    trainer.default_local_dir=${OUTPUT_DIR}/checkpoints \
    trainer.project_name=mxfp4_v3.0_dapo \
    trainer.experiment_name=mxfp4_v3.0_dapo_1ep \
    2>&1 | tee ${OUTPUT_DIR}/training.log

echo ""
echo "MXFP4 v3.0 DAPO experiment complete!"
echo "Results in: ${OUTPUT_DIR}"
echo ""
echo "Compare with:"
echo "  - E2a (v2.0, no AQN, 2ep): 65.96% final, 73.16% peak"
echo "  - E2b (v2.1, RMSNorm AQN, 2ep): 68.84% final, 73.24% peak"
echo "  - v3.0 target: Match or exceed peak (~73%) without epoch-2 decline"
