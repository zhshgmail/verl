#!/bin/bash
# NVFP4 v2.3: Baseline (no AQN) with lm_head fix
#
# Experiment ID: v2.3
# Date: 2026-01-09
#
# Configuration:
# - NVFP4 W4A16 quantization (~1% error, 21x better than MXFP4)
# - lm_head and embed_tokens EXCLUDED (fix applied)
# - NO AQN noise injection (baseline test)
# - 2 epochs
#
# Compare with MXFP4 v2.0 to measure format difference.
# Previous NVFP4 v1 collapsed at 7.66% due to lm_head bug.
#
# Usage:
#   bash scripts/test_nvfp4_v2.3_baseline.sh [N_GPUS]

set -x

N_GPUS=${1:-8}

export WANDB_MODE=offline

MODEL_PATH=${MODEL_PATH:-"/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"}
TRAIN_DATA=${TRAIN_DATA:-"/data/z00637938/gsm8k/train.parquet"}
VAL_DATA=${VAL_DATA:-"/data/z00637938/gsm8k/test.parquet"}

OUTPUT_DIR="/tmp/nvfp4_v2.3_baseline"
mkdir -p ${OUTPUT_DIR}

echo "=== NVFP4 v2.3: Baseline (no AQN), lm_head excluded, 2 epochs ==="
echo "Key settings:"
echo "  - NVFP4 W4A16 quantization (~1% error)"
echo "  - exclude_modules=['lm_head', 'embed_tokens'] (FIX!)"
echo "  - NO AQN noise injection"
echo "  - 2 epochs"
echo "  - Compare with MXFP4 v2.0 (65.96%)"
echo "  - Previous NVFP4 v1 collapsed at 7.66% (lm_head bug)"

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
    trainer.total_epochs=2 \
    trainer.test_freq=20 \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.val_before_train=True \
    trainer.save_freq=50 \
    trainer.hw_error_injection.enabled=True \
    trainer.hw_error_injection.error_type=nvfp4 \
    trainer.hw_error_injection.injection_point=weight \
    trainer.hw_error_injection.apply_during=both \
    'trainer.hw_error_injection.target_modules=["linear"]' \
    trainer.default_local_dir=${OUTPUT_DIR}/checkpoints \
    trainer.project_name=nvfp4_v2.3_baseline \
    trainer.experiment_name=nvfp4_v2.3_baseline_2ep \
    2>&1 | tee ${OUTPUT_DIR}/training.log

echo ""
echo "NVFP4 v2.3 experiment complete!"
echo "Results in: ${OUTPUT_DIR}"
echo ""
echo "Compare with:"
echo "  - MXFP4 v2.0 (baseline, no AQN): 65.96%"
echo "  - NVFP4 v1 (collapsed, lm_head bug): 7.66%"
