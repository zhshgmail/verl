#!/bin/bash
# Test Operator-Level Noisy Ops on A100 with Qwen2.5-1.5B + GSM8K
# This is a more realistic HW error simulation that affects:
# - ALL matmul operations (not just specific layers)
# - BOTH forward AND backward passes (affects gradients)
# - ALL phases (rollout and training, no phase distinction)
#
# Usage: bash scripts/test_noisy_ops_a100.sh [ERROR_SCALE] [N_GPUS]
#
# Error scale recommendations:
# - 1e-5: Conservative, baseline comparison
# - 1e-4: Aggressive, should show observable accuracy differences
# - 1e-3: Very aggressive, may destabilize training
#
# Examples:
#   bash scripts/test_noisy_ops_a100.sh 1e-4 8   # 1e-4 scale, 8 GPUs
#   bash scripts/test_noisy_ops_a100.sh 1e-5 8   # 1e-5 scale, 8 GPUs (baseline comparison)

set -x

# Disable WandB online logging (avoids API key requirement)
export WANDB_MODE=offline

# Configuration
ERROR_SCALE=${1:-1e-4}
N_GPUS=${2:-8}

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
    trainer.project_name=noisy_ops_test
    trainer.n_gpus_per_node=${N_GPUS}
    trainer.val_before_train=True
"

echo "=== Running Operator-Level Noisy Ops Test ==="
echo "Error scale: ${ERROR_SCALE}"
echo "N GPUs: ${N_GPUS}"
echo ""
echo "This test injects errors into:"
echo "  - ALL torch.matmul / F.linear operations"
echo "  - BOTH forward AND backward passes"
echo "  - ALL phases (rollout + training)"
echo ""

python3 -m verl.trainer.main_ppo \
    --config-name=ppo_trainer \
    ${COMMON_ARGS} \
    +trainer.noisy_ops.enabled=True \
    +trainer.noisy_ops.error_scale=${ERROR_SCALE} \
    +trainer.noisy_ops.error_type=relative_gaussian \
    trainer.experiment_name=noisy_ops_${ERROR_SCALE}

echo "=== Test Complete ==="
