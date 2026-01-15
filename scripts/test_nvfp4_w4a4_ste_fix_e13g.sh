#!/bin/bash
# E13g: NVFP4 W4A4 with STE FIX for activation quantization
#
# CRITICAL FIX: Previous experiments failed because activation quantization
# was NOT using STEQuantizeActivation.apply(), which blocked gradient flow!
#
# The STE bug fix:
# 1. Activation quantization now uses STEQuantizeActivation.apply()
# 2. Removed @torch.no_grad() from quantization functions
# 3. This allows gradients to flow backward through quantized activations
#
# Previous experiments (all failed at ~7-10%):
# - E13a-mxfp4: MXFP4 POST-hook: 7.43% @ step 20
# - E13a-nvfp4: NVFP4 POST-hook large batch: 9.02% @ step 20
# - E13b-nvfp4: NVFP4 POST-hook small batch: 8.34% @ step 20
# - E13c-nvfp4: NVFP4 POST-hook training-only: 9.02% @ step 20
# - E13d-nvfp4: NVFP4 PRE-hook: N/A
# - E13e-nvfp4: NVFP4 PRE-hook + exclude base_layer: N/A
# - E13f-nvfp4: NVFP4 PRE-hook + column-wise: 10.39% @ step 20
#
# Root cause analysis:
# - W4A16 experiments (E3-E7) succeeded (60-74%) because only weights quantized
# - W4A4 experiments (E13) failed because activation quant blocked gradients
# - LoRA adapters couldn't learn due to corrupted/blocked backward gradients
#
# Expected: With STE fix, accuracy should improve to ~60% (matching QeRL)

cd /home/z00637938/workspace/verl

# Configuration
OUTPUT_DIR="/tmp/nvfp4_w4a4_ste_fix_e13g"
LOG_FILE="${OUTPUT_DIR}/training.log"
mkdir -p "${OUTPUT_DIR}"

echo "E13g-ste-fix started in $(hostname) (PID: $$)" > /tmp/nvfp4_w4a4_e13g_status.txt

# Run training with STE fix for activation quantization
nohup env WANDB_MODE=disabled python3 -m recipe.dapo.main_dapo \
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
    trainer.hw_error_injection.error_type=nvfp4 \
    trainer.hw_error_injection.injection_point=both \
    trainer.hw_error_injection.apply_during=both \
    trainer.hw_error_injection.target_modules='["linear"]' \
    ++trainer.hw_error_injection.exclude_modules='["lm_head", "embed_tokens", "lora_A", "lora_B", "layers.0", "layers.27", "base_layer"]' \
    ++trainer.hw_error_injection.use_ste=True \
    trainer.default_local_dir="${OUTPUT_DIR}/checkpoints" \
    trainer.project_name=w4a4_e13g_ste_fix \
    trainer.experiment_name=nvfp4_w4a4_ste_fix_1ep \
    > "${LOG_FILE}" 2>&1 &

echo "NVFP4 W4A4 with STE FIX started (PID: $!)"
echo "CRITICAL FIX: Activation quantization now uses STEQuantizeActivation.apply() for gradient flow"
echo "Monitor: tail -f ${LOG_FILE}"
echo "Grep scores: grep 'score/mean\|val-core' ${LOG_FILE}"
