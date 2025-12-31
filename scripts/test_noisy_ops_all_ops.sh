#!/bin/bash
# Test ALL Operators Noisy Mode on A100 with Qwen2.5-1.5B + GSM8K
# This extends E4 by injecting noise into ALL operators:
# - matmul, bmm, linear (core ops - always enabled)
# - softmax, silu, gelu, layer_norm (additional ops - enabled by ALL_OPS_MODE)
#
# Usage: bash scripts/test_noisy_ops_all_ops.sh [ERROR_SCALE] [N_GPUS]
#
# This is E5 in the experimental series - testing broader operator coverage
# to observe measurable accuracy degradation.

set -x

# Disable WandB online logging (avoids API key requirement)
export WANDB_MODE=offline

# Configuration
ERROR_SCALE=${1:-1e-3}
N_GPUS=${2:-8}

# Enable operator-level noisy ops via environment variables
# Key difference from E4: VERL_NOISY_OPS_ALL_OPS=1 enables noise in ALL operators
export VERL_NOISY_OPS_ENABLED=1
export VERL_NOISY_OPS_SCALE=${ERROR_SCALE}
export VERL_NOISY_OPS_TYPE=relative_gaussian
export VERL_NOISY_OPS_ALL_OPS=1

# Model and data paths (adjust for your setup)
MODEL_PATH=${MODEL_PATH:-"/data/models/Qwen2.5-1.5B-Instruct"}
TRAIN_DATA=${TRAIN_DATA:-"/data/datasets/gsm8k/train.parquet"}
VAL_DATA=${VAL_DATA:-"/data/datasets/gsm8k/test.parquet"}

# Common training args (matching run_gpu_baseline.sh exactly)
COMMON_ARGS="
    data.train_files=${TRAIN_DATA}
    data.val_files=${VAL_DATA}
    data.train_batch_size=128
    data.max_prompt_length=1024
    data.max_response_length=1024
    data.filter_overlong_prompts=True
    data.truncation=error
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.actor.optim.lr=5e-7
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.ppo_mini_batch_size=32
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8
    actor_rollout_ref.rollout.n=5
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    trainer.critic_warmup=0
    trainer.logger=console
    trainer.total_epochs=2
    trainer.test_freq=20
    trainer.save_freq=-1
    trainer.nnodes=1
    trainer.project_name=noisy_ops_all_ops_test
    trainer.n_gpus_per_node=${N_GPUS}
    trainer.val_before_train=True
"

echo "=== Running ALL Operators Noisy Mode Test (E5) ==="
echo "Error scale: ${ERROR_SCALE}"
echo "N GPUs: ${N_GPUS}"
echo ""
echo "Environment variables set:"
echo "  VERL_NOISY_OPS_ENABLED=${VERL_NOISY_OPS_ENABLED}"
echo "  VERL_NOISY_OPS_SCALE=${VERL_NOISY_OPS_SCALE}"
echo "  VERL_NOISY_OPS_TYPE=${VERL_NOISY_OPS_TYPE}"
echo "  VERL_NOISY_OPS_ALL_OPS=${VERL_NOISY_OPS_ALL_OPS}"
echo ""
echo "This test injects errors into ALL operators:"
echo "  - Core ops: matmul, bmm, linear"
echo "  - Additional ops: softmax, silu, gelu, layer_norm"
echo "  - BOTH forward AND backward passes"
echo "  - ALL phases (rollout + training)"
echo ""

python3 -m verl.trainer.main_ppo \
    --config-name=ppo_trainer \
    ${COMMON_ARGS} \
    +trainer.noisy_ops.enabled=True \
    +trainer.noisy_ops.error_scale=${ERROR_SCALE} \
    +trainer.noisy_ops.error_type=relative_gaussian \
    +trainer.noisy_ops.all_ops_mode=True \
    trainer.experiment_name=noisy_ops_all_ops_${ERROR_SCALE}

echo "=== Test Complete ==="
