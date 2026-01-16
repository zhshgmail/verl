#!/bin/bash
# Validation-only evaluation for E13j step 29 checkpoint
# Uses verl's fast vLLM infrastructure

set -x

N_GPUS=8

# Paths (same as training)
MODEL_PATH=${MODEL_PATH:-"/home/z00637938/workspace/verl/checkpoints/noisy_ops_e8c_forward_only/e8c_forward_only_5e-2/global_step_116/merged_hf"}
TRAIN_DATA=${TRAIN_DATA:-"/data/z00637938/gsm8k/train.parquet"}
VAL_DATA=${VAL_DATA:-"/data/z00637938/gsm8k/test.parquet"}

# IMPORTANT: Use same output dir as training to load checkpoint
OUTPUT_DIR="/tmp/mxfp4_w4a4_e13j_global_aqn"

# Key parameters for validation-only mode
VAL_ONLY=True  # This is the critical flag!
VAL_BEFORE_TRAIN=True
TOTAL_EPOCHS=0  # Not needed since val_only exits early

python3 -m verl.trainer.main_ppo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=128 \
    data.val_batch_size=1319 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    ++data.truncation="error" \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${N_GPUS} \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.path=${MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size=4 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.grad_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    reward_model.enable_rm=False \
    reward_model.enable_default_compute_score=True \
    trainer.val_only=${VAL_ONLY} \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    trainer.critic_warmup=0 \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.resume_mode=auto \
    trainer.default_local_dir=${OUTPUT_DIR}/checkpoints \
    trainer.project_name=mxfp4_w4a4_e13j_eval \
    trainer.experiment_name=e13j_step29_validation \
    trainer.logger=['console','tracking'] \
    trainer.hw_error_injection.enabled=True \
    trainer.hw_error_injection.error_type=mxfp4 \
    trainer.hw_error_injection.injection_point=both \
    trainer.hw_error_injection.apply_during=both \
    'trainer.hw_error_injection.target_modules=["linear"]' \
    '++trainer.hw_error_injection.exclude_modules=["lm_head", "embed_tokens", "lora_A", "lora_B", "layers.0", "layers.27", "base_layer"]' \
    ++trainer.hw_error_injection.use_ste=True
