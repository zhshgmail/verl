#!/bin/bash
# NVFP4 W4A16 Training Experiment
#
# NVFP4 (NVIDIA FP4) has ~1% relative error (vs ~21% for MXFP4).
# This makes it 21x better for quantization-aware training.
#
# This experiment tests NVFP4 + AQN to see if:
# 1. NVFP4's lower error allows stable training
# 2. AQN can improve quantization robustness
#
# Expected: Better results than MXFP4 experiments due to lower error.
#
# Usage:
#   bash scripts/test_nvfp4_w4a16_training.sh [N_GPUS]

set -x

N_GPUS=${1:-8}

export WANDB_MODE=offline

MODEL_PATH=${MODEL_PATH:-"/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"}
TRAIN_DATA=${TRAIN_DATA:-"/data/z00637938/gsm8k/train.parquet"}
VAL_DATA=${VAL_DATA:-"/data/z00637938/gsm8k/test.parquet"}

OUTPUT_DIR="/tmp/nvfp4_w4a16_aqn"
mkdir -p ${OUTPUT_DIR}

echo "=== NVFP4 W4A16 + AQN Experiment ==="
echo "Key differences from MXFP4:"
echo "  - Relative error: ~1% (vs ~21% for MXFP4)"
echo "  - Block size: 16 (vs 32)"
echo "  - Scale format: E4M3 (vs E8M0)"
echo "  - Expected: Stable training, ~74% accuracy"

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
    trainer.project_name=nvfp4_w4a16_aqn \
    trainer.experiment_name=nvfp4_w4a16_aqn_3ep \
    2>&1 | tee ${OUTPUT_DIR}/training.log

echo ""
echo "NVFP4 experiment complete!"
echo "Results in: ${OUTPUT_DIR}"
echo "Expected accuracy: 74-76% (near baseline due to low error)"
