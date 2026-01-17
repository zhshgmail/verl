#!/bin/bash
# W4A4 MXFP4 + LoRA + Zone-Based SRDD AQN (E14a)
#
# Experiment ID: E14a
# Date: 2026-01-17
#
# Configuration:
# - W4A4 MXFP4 fake quantization (both weights AND activations)
# - 16-bit LoRA (rank=32, alpha=16)
# - Zone-Based Scheduling AQN:
#   - Phase 1 (Steps 0-14): Uniform noise on ALL layers
#   - Phase 2 (Steps 15-29): Reduced noise on MIDDLE zone (6-22)
# - DAPO algorithm, 1 epoch
#
# SRDD Zone Analysis:
#   - EDGE zones (layers 0-5, 23-27): 32.72% avg error - RESILIENT
#   - MIDDLE zone (layers 6-22): 38.78% avg error - VULNERABLE
#
# Hypothesis:
#   - Maintain gradient consistency during rapid learning (early phase)
#   - Protect vulnerable middle layers during convergence (late phase)
#
# Baseline: E13j = 73.31%
# Expected: Match or exceed E13j
#
# Usage:
#   bash scripts/test_mxfp4_w4a4_e14a_zone_schedule.sh [N_GPUS]

set -x

N_GPUS=${1:-8}

export WANDB_MODE=offline

MODEL_PATH=${MODEL_PATH:-"/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"}
TRAIN_DATA=${TRAIN_DATA:-"/data/z00637938/gsm8k/train.parquet"}
VAL_DATA=${VAL_DATA:-"/data/z00637938/gsm8k/test.parquet"}

OUTPUT_DIR="/tmp/mxfp4_w4a4_e14a_zone_schedule"
mkdir -p ${OUTPUT_DIR}

# DAPO settings (same as E13j)
adv_estimator=grpo
use_kl_in_reward=False
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.25
max_prompt_length=1024
max_response_length=1024
enable_overlong_buffer=True
overlong_buffer_len=256
overlong_penalty_factor=0.5
loss_agg_mode="token-mean"

enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=0

# Batch sizes
train_batch_size=128
gen_batch_size=256
ppo_mini_batch_size=32
n_resp_per_prompt=8

# LoRA settings
lora_rank=32
lora_alpha=16

echo "=== W4A4 MXFP4 + LoRA + Zone-Based SRDD AQN (E14a) ==="
echo "Key settings:"
echo "  - W4A4 MXFP4 fake quantization (injection_point=both)"
echo "  - 16-bit LoRA: rank=${lora_rank}, alpha=${lora_alpha}"
echo "  - Zone-Based AQN: sigma 0.05 -> 0.0005 on ALL layers (RMSNorm)"
echo "  - Schedule:"
echo "    - Phase 1 (Steps 0-14): Uniform noise (EDGE=1.0, MIDDLE=1.0)"
echo "    - Phase 2 (Steps 15-29): Protect middle (EDGE=1.0, MIDDLE=0.5)"
echo "  - 10 decay stages"
echo "  - 1 epoch (29 steps)"
echo ""
echo "SRDD Zone Analysis:"
echo "  - EDGE (layers 0-5, 23-27): 32.72% error - RESILIENT"
echo "  - MIDDLE (layers 6-22): 38.78% error - VULNERABLE"
echo ""
echo "Baseline: E13j = 73.31%"
echo "Expected: Match or exceed E13j"

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
    '++trainer.hw_error_injection.exclude_modules=["lm_head", "embed_tokens", "lora_A", "lora_B", "layers.0", "layers.27", "base_layer"]' \
    ++trainer.hw_error_injection.use_ste=True \
    ++trainer.noise_injection.enabled=True \
    ++trainer.noise_injection.sigma_start=0.05 \
    ++trainer.noise_injection.sigma_end=0.0005 \
    ++trainer.noise_injection.num_stages=10 \
    '++trainer.noise_injection.layer_types=["rmsnorm"]' \
    '++trainer.noise_injection.exclude_patterns=["lm_head", "embed_tokens"]' \
    '++trainer.noise_injection.layer_sigma_config.enabled=True' \
    '++trainer.noise_injection.layer_sigma_config.default_multiplier=1.0' \
    '++trainer.noise_injection.layer_sigma_config.zone_schedule.enabled=True' \
    '++trainer.noise_injection.layer_sigma_config.zone_schedule.edge_layers=[0,1,2,3,4,5,23,24,25,26,27]' \
    '++trainer.noise_injection.layer_sigma_config.zone_schedule.middle_layers=[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]' \
    '++trainer.noise_injection.layer_sigma_config.zone_schedule.phase_boundary_ratio=0.5' \
    '++trainer.noise_injection.layer_sigma_config.zone_schedule.early_middle_mult=1.0' \
    '++trainer.noise_injection.layer_sigma_config.zone_schedule.late_middle_mult=0.5' \
    trainer.default_local_dir=${OUTPUT_DIR}/checkpoints \
    trainer.project_name=w4a4_e14a_zone_schedule \
    trainer.experiment_name=mxfp4_w4a4_zone_aqn_1ep \
    trainer.logger='["console"]' \
    2>&1 | tee ${OUTPUT_DIR}/training.log

echo ""
echo "E14a W4A4 + Zone-Based AQN experiment complete!"
echo "Results in: ${OUTPUT_DIR}"
echo ""
echo "Monitor: tail -f ${OUTPUT_DIR}/training.log"
echo "Grep scores: grep 'score/mean\|val-core' ${OUTPUT_DIR}/training.log"
