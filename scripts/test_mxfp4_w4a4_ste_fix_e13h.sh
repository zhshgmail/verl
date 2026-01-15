#!/bin/bash
# E13h: MXFP4 W4A4 with STE FIX (baseline, no RIN/AQN)
#
# This experiment tests MXFP4 W4A4 with STE fix (similar to E13g but with MXFP4).
# NO RIN (Resilient-Improving Noise) or AQN applied.
#
# Comparison:
# - E13g (NVFP4 W4A4 + STE): 60.88% @ step 20 ✓
# - E13h (MXFP4 W4A4 + STE): TBD
#
# Expected: Similar to E13g, around 55-60% accuracy
# MXFP4 has higher quantization error (~21% rel) vs NVFP4 (~15% rel)
# so expect slightly lower accuracy than E13g

cd /home/z00637938/workspace/verl

# Configuration
OUTPUT_DIR="/tmp/mxfp4_w4a4_ste_fix_e13h"
LOG_FILE="${OUTPUT_DIR}/training.log"
mkdir -p "${OUTPUT_DIR}"

echo "E13h-mxfp4-w4a4 started in $(hostname) (PID: $$)" > /tmp/mxfp4_w4a4_e13h_status.txt

# Run training with STE fix for activation quantization
# Set WANDB_MODE=disabled to skip WandB logging
export WANDB_MODE=disabled
nohup python3 -m recipe.dapo.main_dapo \
    data.train_files=/data/z00637938/gsm8k/train.parquet \
    data.val_files=/data/z00637938/gsm8k/test.parquet \
    data.train_batch_size=128 \
    data.gen_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path=/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.25 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric=acc \
    algorithm.filter_groups.max_num_gen_batches=5 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=True \
    reward_model.overlong_buffer.len=256 \
    reward_model.overlong_buffer.penalty_factor=0.5 \
    reward_model.overlong_buffer.log=True \
    trainer.critic_warmup=0 \
    trainer.total_epochs=1 \
    trainer.test_freq=20 \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=True \
    trainer.save_freq=50 \
    trainer.hw_error_injection.enabled=True \
    trainer.hw_error_injection.error_type=mxfp4 \
    trainer.hw_error_injection.injection_point=both \
    trainer.hw_error_injection.apply_during=both \
    trainer.hw_error_injection.target_modules='["linear"]' \
    ++trainer.hw_error_injection.exclude_modules='["lm_head", "embed_tokens", "lora_A", "lora_B", "layers.0", "layers.27", "base_layer"]' \
    ++trainer.hw_error_injection.use_ste=True \
    trainer.default_local_dir="${OUTPUT_DIR}/checkpoints" \
    trainer.project_name=w4a4_e13h_mxfp4 \
    trainer.experiment_name=mxfp4_w4a4_ste_fix_1ep \
    trainer.logger='["console"]' \
    > "${LOG_FILE}" 2>&1 &

echo "MXFP4 W4A4 with STE FIX started (PID: $!)"
echo "This is E13h - MXFP4 W4A4 baseline (no RIN/AQN)"
echo "Monitor: tail -f ${LOG_FILE}"
echo "Grep scores: grep 'score/mean\|val-core' ${LOG_FILE}"
