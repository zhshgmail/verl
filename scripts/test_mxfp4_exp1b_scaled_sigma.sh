#!/bin/bash
# MXFP4 Experiment 1B: Scaled Sigma (if 1A shows improvement)
#
# Key changes from Experiment 1A:
# 1. sigma_start=0.15 (3x QeRL default, for 21% MXFP4 error)
# 2. sigma_end=0.0015 (3x end value)
# 3. Same aligned targets and 3 epochs
#
# Rationale: MXFP4 has ~21% relative error vs NVFP4's ~1%
#            QeRL's sigma was calibrated for 1% error
#            Scaling sigma 3x should better match MXFP4 error magnitude
#
# Expected: 72-74% accuracy (vs 70-72% for 1A)
#
# Usage:
#   bash scripts/test_mxfp4_exp1b_scaled_sigma.sh [N_GPUS]

set -x

N_GPUS=${1:-8}

export WANDB_MODE=offline

MODEL_PATH=${MODEL_PATH:-"/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"}
TRAIN_DATA=${TRAIN_DATA:-"/data/z00637938/gsm8k/train.parquet"}
VAL_DATA=${VAL_DATA:-"/data/z00637938/gsm8k/test.parquet"}

OUTPUT_DIR="/tmp/mxfp4_exp1b_scaled_sigma"
mkdir -p ${OUTPUT_DIR}

echo "=== Experiment 1B: MXFP4 W4A16 + AQN (Scaled Sigma 3x) ==="
echo "Key changes from 1A:"
echo "  - sigma_start=0.15 (was: 0.05)"
echo "  - sigma_end=0.0015 (was: 0.0005)"
echo "  - Rationale: Scale sigma to match MXFP4's 21% error (vs NVFP4's 1%)"
echo "  - Expected: 72-74%"

python3 -m verl.trainer.main_ppo \
    --config-name=ppo_trainer \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.total_epochs=3 \
    trainer.test_freq=20 \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.val_before_train=True \
    trainer.save_freq=50 \
    trainer.hw_error_injection.enabled=True \
    trainer.hw_error_injection.error_type=mxfp4 \
    trainer.hw_error_injection.injection_point=weight \
    trainer.hw_error_injection.apply_during=both \
    'trainer.hw_error_injection.target_modules=["linear"]' \
    ++trainer.noise_injection.enabled=True \
    ++trainer.noise_injection.sigma_start=0.15 \
    ++trainer.noise_injection.sigma_end=0.0015 \
    ++trainer.noise_injection.num_stages=10 \
    '++trainer.noise_injection.layer_types=["linear"]' \
    trainer.default_local_dir=${OUTPUT_DIR}/checkpoints \
    trainer.project_name=mxfp4_exp1b_scaled_sigma \
    trainer.experiment_name=mxfp4_w4a16_aqn_scaled_3x_3ep \
    2>&1 | tee ${OUTPUT_DIR}/training.log

echo ""
echo "Experiment 1B complete!"
echo "Results in: ${OUTPUT_DIR}"
echo "Expected accuracy: 72-74%"
