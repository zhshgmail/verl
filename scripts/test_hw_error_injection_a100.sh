#!/bin/bash
# Test HW Error Injection on A100 with Qwen2.5-1.5B + GSM8K
# Usage: bash scripts/test_hw_error_injection_a100.sh [minimal|mlp|both]
#
# Test configurations:
# - minimal: Only RMSNorm layers
# - mlp: RMSNorm + down_proj
# - both: Run both in parallel (4 GPUs each)

set -x

# Configuration
MODE=${1:-minimal}  # minimal, mlp, or both
N_GPUS=${2:-4}
ERROR_SCALE=${3:-1e-5}

# Model and data paths (adjust for your setup)
MODEL_PATH=${MODEL_PATH:-"/data/models/Qwen2.5-1.5B-Instruct"}
TRAIN_DATA=${TRAIN_DATA:-"/data/datasets/gsm8k/train.parquet"}
VAL_DATA=${VAL_DATA:-"/data/datasets/gsm8k/test.parquet"}

# Common training args
COMMON_ARGS="
    data.train_files=${TRAIN_DATA}
    data.val_files=${VAL_DATA}
    data.train_batch_size=32
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=4
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.enforce_eager=True
    algorithm.adv_estimator=grpo
    trainer.total_epochs=3
    trainer.test_freq=1
    trainer.project_name=hw_error_test
    trainer.n_gpus_per_node=${N_GPUS}
    trainer.val_before_train=True
"

# HW Error Injection args
HW_ERROR_BASE="
    trainer.hw_error_injection.enabled=True
    trainer.hw_error_injection.error_scale=${ERROR_SCALE}
    trainer.hw_error_injection.error_type=relative_gaussian
    trainer.hw_error_injection.injection_point=input
    trainer.hw_error_injection.apply_during=rollout
"

run_minimal() {
    echo "=== Running Minimal HW Error Injection (RMSNorm only) ==="
    python3 -m verl.trainer.main_ppo \
        --config-name=ppo_trainer \
        ${COMMON_ARGS} \
        ${HW_ERROR_BASE} \
        "trainer.hw_error_injection.target_modules=['rmsnorm']" \
        trainer.experiment_name=hw_error_minimal_${ERROR_SCALE} \
        2>&1 | tee /tmp/hw_error_minimal.log
}

run_mlp() {
    echo "=== Running MLP-focused HW Error Injection (RMSNorm + down_proj) ==="
    python3 -m verl.trainer.main_ppo \
        --config-name=ppo_trainer \
        ${COMMON_ARGS} \
        ${HW_ERROR_BASE} \
        "trainer.hw_error_injection.target_modules=['rmsnorm','down_proj']" \
        trainer.experiment_name=hw_error_mlp_${ERROR_SCALE} \
        2>&1 | tee /tmp/hw_error_mlp.log
}

run_both() {
    echo "=== Running Both Tests in Parallel (4 GPUs each) ==="

    # Minimal on GPUs 0-3
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main_ppo \
        --config-name=ppo_trainer \
        ${COMMON_ARGS} \
        ${HW_ERROR_BASE} \
        "trainer.hw_error_injection.target_modules=['rmsnorm']" \
        trainer.experiment_name=hw_error_minimal_${ERROR_SCALE} \
        trainer.n_gpus_per_node=4 \
        2>&1 | tee /tmp/hw_error_minimal.log &
    PID1=$!

    # MLP on GPUs 4-7
    CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m verl.trainer.main_ppo \
        --config-name=ppo_trainer \
        ${COMMON_ARGS} \
        ${HW_ERROR_BASE} \
        "trainer.hw_error_injection.target_modules=['rmsnorm','down_proj']" \
        trainer.experiment_name=hw_error_mlp_${ERROR_SCALE} \
        trainer.n_gpus_per_node=4 \
        2>&1 | tee /tmp/hw_error_mlp.log &
    PID2=$!

    echo "Started minimal test (PID: $PID1) and MLP test (PID: $PID2)"
    echo "Logs: /tmp/hw_error_minimal.log, /tmp/hw_error_mlp.log"

    # Wait for both
    wait $PID1 $PID2
    echo "Both tests completed."
}

case $MODE in
    minimal)
        run_minimal
        ;;
    mlp)
        run_mlp
        ;;
    both)
        run_both
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [minimal|mlp|both] [n_gpus] [error_scale]"
        exit 1
        ;;
esac
