#!/bin/bash
# Test ALL_OPS Noisy + Epoch-Aware AQN (E5d) on A100
#
# This is E5d in the experimental series - testing epoch-aware AQN scheduling
# with noise injection on ALL operators (not just matmul).
#
# Key results so far:
#   E5 (matmul noise only, no AQN): 68.16%
#   E5b (matmul noise + epoch-aware AQN): 70.58%
#   E5c (ALL_OPS noise, no AQN): 69.07% (+0.91% vs E5!)
#   E5d (ALL_OPS noise + epoch-aware AQN): ??? (this experiment)
#
# Hypothesis: If AQN improved matmul-only from 68.16% to 70.58% (+2.42%),
# can it also improve ALL_OPS from 69.07% to ~71.5%?
#
# Epoch-aware schedule (Option C):
#   Epoch 1: sigma 0.05 -> 0.01 (high exploration)
#   Epoch 2: sigma 0.01 -> 0.0005 (refinement)
#
# Usage: bash scripts/test_noisy_ops_all_ops_aqn_epoch_aware.sh [ERROR_SCALE] [N_GPUS]

set -x

# Disable WandB online logging (avoids API key requirement)
export WANDB_MODE=offline

# Configuration
ERROR_SCALE=${1:-5e-2}
N_GPUS=${2:-8}

# Enable operator-level noisy ops via environment variables
# KEY DIFFERENCE: ALL_OPS=1 enables noise on softmax, silu, gelu, layer_norm too
export VERL_NOISY_OPS_ENABLED=1
export VERL_NOISY_OPS_SCALE=${ERROR_SCALE}
export VERL_NOISY_OPS_TYPE=relative_gaussian
export VERL_NOISY_OPS_ALL_OPS=1

# Model and data paths (adjust for your setup)
MODEL_PATH=${MODEL_PATH:-"/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"}
TRAIN_DATA=${TRAIN_DATA:-"/data/z00637938/gsm8k/train.parquet"}
VAL_DATA=${VAL_DATA:-"/data/z00637938/gsm8k/test.parquet"}

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
    trainer.save_freq=58
    trainer.nnodes=1
    trainer.project_name=noisy_ops_all_ops_aqn_epoch_aware_test
    trainer.n_gpus_per_node=${N_GPUS}
    trainer.val_before_train=True
"

echo "=== Running ALL_OPS Noisy + Epoch-Aware AQN Test (E5d) ==="
echo "Error scale: ${ERROR_SCALE}"
echo "N GPUs: ${N_GPUS}"
echo ""
echo "Environment variables set:"
echo "  VERL_NOISY_OPS_ENABLED=${VERL_NOISY_OPS_ENABLED}"
echo "  VERL_NOISY_OPS_SCALE=${VERL_NOISY_OPS_SCALE}"
echo "  VERL_NOISY_OPS_TYPE=${VERL_NOISY_OPS_TYPE}"
echo "  VERL_NOISY_OPS_ALL_OPS=${VERL_NOISY_OPS_ALL_OPS}"
echo ""
echo "This test combines:"
echo "  1. ALL_OPS noise: 5% error in matmul, softmax, silu, gelu, layer_norm"
echo "  2. Epoch-aware AQN (Option C):"
echo "     - Epoch 1: sigma 0.05 -> 0.01 (exploration)"
echo "     - Epoch 2: sigma 0.01 -> 0.0005 (refinement)"
echo ""
echo "Previous results:"
echo "  E5 (matmul noise only, no AQN): 68.16%"
echo "  E5b (matmul noise + epoch-aware AQN): 70.58%"
echo "  E5c (ALL_OPS noise, no AQN): 69.07%"
echo "  E5d target: > 69.07% (AQN should help)"
echo ""

python3 -m verl.trainer.main_ppo \
    --config-name=ppo_trainer \
    ${COMMON_ARGS} \
    ++trainer.noisy_ops.enabled=True \
    ++trainer.noisy_ops.error_scale=${ERROR_SCALE} \
    ++trainer.noisy_ops.error_type=relative_gaussian \
    ++trainer.noisy_ops.all_ops_mode=True \
    ++trainer.noise_injection.enabled=True \
    ++trainer.noise_injection.epoch_aware=True \
    ++trainer.noise_injection.sigma_start=0.05 \
    ++trainer.noise_injection.sigma_end=0.0005 \
    ++trainer.noise_injection.stages_per_epoch=5 \
    trainer.experiment_name=noisy_ops_all_ops_aqn_epoch_aware_${ERROR_SCALE}

echo "=== Test Complete ==="
