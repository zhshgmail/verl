#!/bin/bash
# GSM8K training with Qwen2.5-1.5B + AQN (Adaptive Quantization Noise) on NPU
# This script demonstrates AQN noise injection during GRPO training
set -x

# Source Ascend environment (adjust paths as needed)
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi

# Critical environment variables for NPU
export VLLM_ASCEND_ENABLE_NZ=0
export WANDB_MODE=offline
export TASK_QUEUE_ENABLE=2
export HCCL_CONNECT_TIMEOUT=1800
export RAY_USAGE_STATS_DISABLE=1

# Model path - adjust to your local model path
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-1.5B-Instruct"}

# Data paths - adjust to your local data paths
TRAIN_DATA=${TRAIN_DATA:-"data/gsm8k/train.parquet"}
VAL_DATA=${VAL_DATA:-"data/gsm8k/test.parquet"}

# GSM8K training with AQN noise injection
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=verl/utils/reward_score/gsm8k.py \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name=gsm8k_aqn_test \
    trainer.experiment_name=qwen15b_gsm8k_aqn \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    trainer.total_epochs=2 \
    trainer.val_before_train=True \
    +trainer.noise_injection.enabled=True \
    +trainer.noise_injection.sigma_start=0.01 \
    +trainer.noise_injection.sigma_end=0.001 \
    +trainer.noise_injection.num_stages=10 \
    '+trainer.noise_injection.target_modules=["post_attention_layernorm"]' \
    '+trainer.noise_injection.exclude_patterns=["input_layernorm"]'
