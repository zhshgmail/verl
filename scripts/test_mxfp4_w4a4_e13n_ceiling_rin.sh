#!/bin/bash
# W4A4 MXFP4 + LoRA + Ceiling-Constrained Variable RIN (E13n)
#
# Experiment ID: E13n
# Date: 2026-01-17
#
# Configuration:
# - W4A4 MXFP4 fake quantization (both weights AND activations)
# - 16-bit LoRA (rank=32, alpha=16)
# - Ceiling-Constrained Variable RIN: SRDD-guided layer multipliers (NORMALIZED)
#   - ALL multipliers divided by max (1.21) to respect σ=0.05 ceiling
#   - Preserves relative differences (~1.46x between max/min)
#   - Layer 26 (best error): 1.00x (at ceiling, σ_max=0.05)
#   - Layer 15 (worst error): 0.69x (LEAST noise, σ=0.0345)
#   - Middle layers (10-19): 0.69-0.78x (HIGH error, LESS noise)
#   - Edge layers (0-9, 20-27): 0.81-1.00x (LOWER error, MORE noise)
# - Base: sigma 0.05 -> 0.0005, 10 stages (same as E13j/E13m)
# - DAPO algorithm, 1 epoch
# - BUG FIXES: filter_groups disabled, RMSNorm targeting
#
# Purpose: Test if respecting σ=0.05 ceiling improves Variable RIN performance
# Hypothesis: E13m failed because it exceeded proven ceiling (0.0605 max vs 0.05)
#
# Comparison:
#   - E13h baseline (no AQN): 71.42%
#   - E13j (uniform σ=0.05): 73.31% (+1.89%) ← BEST so far, CEILING REFERENCE
#   - E13k (uniform σ=0.01): 65.96% (-5.46%) ← Too low
#   - E13l (high error → MORE noise, max=0.0585): 53.22% (-20.09%) ← CATASTROPHIC
#   - E13m (high error → LESS noise, max=0.0605): 69.37% (-3.94%) ← Exceeded ceiling
#   - E13n (ceiling-constrained, max=0.05): TBD - respects proven ceiling
#
# Expected: Respecting ceiling should close -3.94% gap with E13j
#
# Usage:
#   bash scripts/test_mxfp4_w4a4_e13n_ceiling_rin.sh [N_GPUS]

set -x

N_GPUS=${1:-8}

export WANDB_MODE=offline

MODEL_PATH=${MODEL_PATH:-"/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"}
TRAIN_DATA=${TRAIN_DATA:-"/data/z00637938/gsm8k/train.parquet"}
VAL_DATA=${VAL_DATA:-"/data/z00637938/gsm8k/test.parquet"}

OUTPUT_DIR="/tmp/mxfp4_w4a4_e13n_ceiling_rin"
mkdir -p ${OUTPUT_DIR}

# DAPO settings (same as E13j/E13m)
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

# ✅ BUG FIX: Disable filter_groups
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

echo "=== W4A4 MXFP4 + LoRA + Ceiling-Constrained Variable RIN (E13n) ==="
echo "Key settings:"
echo "  - W4A4 MXFP4 fake quantization (injection_point=both)"
echo "  - 16-bit LoRA: rank=${lora_rank}, alpha=${lora_alpha}"
echo "  - Ceiling-Constrained Variable RIN: SRDD-guided layer multipliers (NORMALIZED)"
echo "    - Based on MXFP4 activation quantization error analysis"
echo "    - ALL multipliers NORMALIZED by max (1.21) to respect σ=0.05 ceiling"
echo "    - Preserves relative differences (~1.46x ratio between max/min)"
echo "    - Layer 26 (best 28.61% error): 1.00x (at ceiling, σ_max=0.05)"
echo "    - Layer 0: 0.97x (σ=0.0485)"
echo "    - Layer 15 (worst 42.65% error): 0.69x (LEAST noise, σ=0.0345)"
echo "    - Middle layers (10-19): 0.69-0.78x (high error ~40%, LESS noise)"
echo "    - Edge layers (0-9, 20-27): 0.81-1.00x (lower error ~32%, MORE noise)"
echo "  - Base sigma: 0.05 -> 0.0005, 10 stages (same as E13j)"
echo "  - Target: RMSNorm layers (QeRL approach)"
echo "  - 1 epoch (29 steps)"
echo ""
echo "Hypothesis: Respecting σ=0.05 ceiling will close -3.94% gap with E13j"
echo "Strategy: High-error layers get LESS noise, low-error layers get MORE"
echo "          BUT no layer exceeds proven σ=0.05 ceiling (E13k showed σ=0.01 too low)"
echo "Target: Match or beat E13j's 73.31% to prove Variable RIN benefit"
echo ""
echo "BUG FIXES APPLIED:"
echo "  ✅ filter_groups.enable=False (no rejection loop)"
echo "  ✅ layer_types=['rmsnorm'] (match QeRL proven approach)"
echo "  ✅ Normalized multipliers (respect σ=0.05 ceiling, fix E13m issue)"
echo ""
echo "Baseline comparison:"
echo "  - E13h (no AQN): 71.42%"
echo "  - E13j (uniform σ=0.05): 73.31% (BEST so far, ceiling reference)"
echo "  - E13k (uniform σ=0.01): 65.96% (too low)"
echo "  - E13l (Variable RIN, max=0.0585): 53.22% (exceeded ceiling, catastrophic)"
echo "  - E13m (Inverse RIN, max=0.0605): 69.37% (exceeded ceiling, underperforms)"
echo ""
echo "Rationale for E13n:"
echo "  E13m's -3.94% gap may be due to exceeding σ=0.05 ceiling (max=0.0605)"
echo "  E13j proved σ=0.05 is optimal (vs E13k's σ=0.01: 65.96%)"
echo "  E13n normalizes all multipliers to ensure max=0.05 (no ceiling violation)"
echo "  If ceiling was the issue, E13n should close the gap with E13j"

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
    ++trainer.noise_injection.layer_sigma_config.layer_multipliers='{0: 0.97, 1: 0.92, 2: 0.88, 3: 0.88, 4: 0.88, 5: 0.84, 6: 0.83, 7: 0.82, 8: 0.83, 9: 0.81, 10: 0.78, 11: 0.74, 12: 0.73, 13: 0.74, 14: 0.70, 15: 0.69, 16: 0.70, 17: 0.73, 18: 0.75, 19: 0.77, 20: 0.81, 21: 0.83, 22: 0.86, 23: 0.88, 24: 0.90, 25: 0.91, 26: 1.00, 27: 0.93}' \
    trainer.default_local_dir=${OUTPUT_DIR}/checkpoints \
    trainer.project_name=w4a4_e13n_ceiling_rin \
    trainer.experiment_name=mxfp4_w4a4_ceiling_rin_1ep \
    trainer.logger='["console"]' \
    2>&1 | tee ${OUTPUT_DIR}/training.log

echo ""
echo "E13n W4A4 + Ceiling-Constrained Variable RIN experiment complete!"
echo "Results in: ${OUTPUT_DIR}"
echo ""
echo "Monitor: tail -f ${OUTPUT_DIR}/training.log"
echo "Grep scores: grep 'score/mean\|val-core' ${OUTPUT_DIR}/training.log"
