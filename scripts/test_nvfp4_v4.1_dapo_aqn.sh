#!/bin/bash
# NVFP4 v4.1: DAPO + RMSNorm AQN (1 epoch)
#
# Experiment ID: v4.1 (E4b)
# Date: 2026-01-09
#
# Key Changes from v4.0:
# - Adds RMSNorm AQN noise injection (sigma 0.05 -> 0.0005)
#
# Configuration:
# - NVFP4 W4A16 quantization (~1% error, 21x better than MXFP4)
# - lm_head and embed_tokens EXCLUDED
# - DAPO overlong penalty: buffer=256, penalty=0.5
# - RMSNorm AQN: sigma 0.05 -> 0.0005 (10 stages)
# - 1 epoch (DAPO standard)
#
# Compare with:
# - E4a (NVFP4 + DAPO, no AQN): TBD
# - E3a (MXFP4 + DAPO): 73.77%
# - E3b (MXFP4 + DAPO + AQN): 74.22%
#
# Usage:
#   bash scripts/test_nvfp4_v4.1_dapo_aqn.sh [N_GPUS]

set -x

N_GPUS=${1:-8}

export WANDB_MODE=offline

MODEL_PATH=${MODEL_PATH:-"/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"}
TRAIN_DATA=${TRAIN_DATA:-"/data/z00637938/gsm8k/train.parquet"}
VAL_DATA=${VAL_DATA:-"/data/z00637938/gsm8k/test.parquet"}

OUTPUT_DIR="/tmp/nvfp4_v4.1_dapo_aqn"
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

# Overlong buffer penalty (DAPO's length hack prevention)
enable_overlong_buffer=True
overlong_buffer_len=256
overlong_penalty_factor=0.5

# Token-level loss (DAPO standard)
loss_agg_mode="token-mean"

# Dynamic sampling / filter groups
enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=5

# Batch sizes
train_batch_size=128
gen_batch_size=256
ppo_mini_batch_size=32
n_resp_per_prompt=8

echo "=== NVFP4 v4.1: DAPO + RMSNorm AQN, 1 Epoch ==="
echo "Key settings:"
echo "  - NVFP4 W4A16 quantization (~1% error)"
echo "  - exclude_modules=['lm_head', 'embed_tokens']"
echo "  - DAPO overlong penalty: buffer=${overlong_buffer_len}, penalty=${overlong_penalty_factor}"
echo "  - Asymmetric clipping: low=${clip_ratio_low}, high=${clip_ratio_high}"
echo "  - Token-level loss: ${loss_agg_mode}"
echo "  - Dynamic sampling: filter_groups enabled"
echo "  - RMSNorm AQN: sigma 0.05 -> 0.0005 (10 stages)"
echo "  - 1 epoch (DAPO standard)"
echo ""
echo "Compare with MXFP4 DAPO experiments:"
echo "  - E3a (MXFP4 + DAPO, no AQN): 73.77%"
echo "  - E3b (MXFP4 + DAPO + AQN): 74.22%"

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
    trainer.hw_error_injection.error_type=nvfp4 \
    trainer.hw_error_injection.injection_point=weight \
    trainer.hw_error_injection.apply_during=both \
    'trainer.hw_error_injection.target_modules=["linear"]' \
    ++trainer.noise_injection.enabled=True \
    ++trainer.noise_injection.sigma_start=0.05 \
    ++trainer.noise_injection.sigma_end=0.0005 \
    ++trainer.noise_injection.num_stages=10 \
    '++trainer.noise_injection.layer_types=["rmsnorm"]' \
    trainer.default_local_dir=${OUTPUT_DIR}/checkpoints \
    trainer.project_name=nvfp4_v4.1_dapo_aqn \
    trainer.experiment_name=nvfp4_v4.1_dapo_aqn_1ep \
    2>&1 | tee ${OUTPUT_DIR}/training.log

echo ""
echo "NVFP4 v4.1 DAPO + AQN experiment complete!"
echo "Results in: ${OUTPUT_DIR}"
echo ""
echo "Compare with:"
echo "  - E4a (NVFP4 + DAPO, no AQN): TBD"
echo "  - E3a (MXFP4 + DAPO, no AQN): 73.77%"
echo "  - E3b (MXFP4 + DAPO + AQN): 74.22%"
