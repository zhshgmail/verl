#!/bin/bash
# Evaluate a checkpoint using verl's val_only mode with MXFP4
# This uses vLLM for fast inference with MXFP4 hooks applied
#
# Usage: bash scripts/eval_checkpoint_verl.sh <checkpoint_path> <experiment_name>
# Example: bash scripts/eval_checkpoint_verl.sh /tmp/mxfp4_w4a4_e13j_global_aqn/checkpoints/global_step_29 e13j

set -x

CHECKPOINT_PATH=${1:-"/tmp/mxfp4_w4a4_e13j_global_aqn/checkpoints/global_step_29"}
EXP_NAME=${2:-"e13j_eval"}
N_GPUS=${3:-8}

export WANDB_MODE=offline

MODEL_PATH=${MODEL_PATH:-"/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"}
TRAIN_DATA=${TRAIN_DATA:-"/data/z00637938/gsm8k/train.parquet"}
VAL_DATA=${VAL_DATA:-"/data/z00637938/gsm8k/test.parquet"}

OUTPUT_DIR="/tmp/${EXP_NAME}_eval"
mkdir -p ${OUTPUT_DIR}

echo "=== Evaluating checkpoint with verl val_only mode ==="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Using MXFP4 W4A4 fake quantization"
echo "Output: $OUTPUT_DIR"
echo ""

# LoRA settings (must match training config)
lora_rank=32
lora_alpha=16

python3 -m recipe.dapo.main_dapo \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.train_batch_size=128 \
    data.gen_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.lora_rank=${lora_rank} \
    actor_rollout_ref.model.lora_alpha=${lora_alpha} \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.temperature=0.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    reward_model.reward_manager=dapo \
    trainer.critic_warmup=0 \
    trainer.total_epochs=1 \
    trainer.test_freq=1 \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=${CHECKPOINT_PATH} \
    trainer.hw_error_injection.enabled=True \
    trainer.hw_error_injection.error_type=mxfp4 \
    trainer.hw_error_injection.injection_point=both \
    trainer.hw_error_injection.apply_during=both \
    'trainer.hw_error_injection.target_modules=["linear"]' \
    '++trainer.hw_error_injection.exclude_modules=["lm_head", "embed_tokens", "lora_A", "lora_B", "layers.0", "layers.27", "base_layer"]' \
    ++trainer.hw_error_injection.use_ste=False \
    trainer.default_local_dir=${OUTPUT_DIR}/checkpoints \
    trainer.project_name=${EXP_NAME} \
    trainer.experiment_name=val_only \
    trainer.logger='["console"]' \
    2>&1 | tee ${OUTPUT_DIR}/eval.log

echo ""
echo "Evaluation complete!"
echo "Results in: ${OUTPUT_DIR}/eval.log"
grep -E "score/mean|val-core|accuracy" ${OUTPUT_DIR}/eval.log | tail -5
