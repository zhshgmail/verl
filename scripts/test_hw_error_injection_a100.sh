#!/bin/bash
# Test HW Error Injection on A100 with Qwen2.5-1.5B + GSM8K
# Usage: bash scripts/test_hw_error_injection_a100.sh [MODE] [N_GPUS] [ERROR_SCALE]
#
# Test configurations (by operator scope):
# - minimal: Only RMSNorm layers (57 hooks, <1% FLOPs)
# - mlp: RMSNorm + down_proj (85 hooks, ~15% FLOPs)
# - linear: All Linear layers (197 hooks, ~95% FLOPs) [RECOMMENDED]
# - all: RMSNorm + all Linear (254 hooks, ~96% FLOPs)
#
# Error scale recommendations:
# - 1e-5: Conservative, may not show measurable impact
# - 1e-4: Aggressive, should show observable accuracy differences
#
# Examples:
#   bash scripts/test_hw_error_injection_a100.sh linear 8 1e-5   # Linear layers, 8 GPUs, 1e-5 scale
#   bash scripts/test_hw_error_injection_a100.sh linear 8 1e-4   # Linear layers, 8 GPUs, 1e-4 scale (aggressive)

set -x

# Disable WandB online logging (avoids API key requirement)
export WANDB_MODE=offline

# Configuration
MODE=${1:-linear}  # minimal, mlp, linear, all
N_GPUS=${2:-8}
ERROR_SCALE=${3:-1e-5}

# Model and data paths (adjust for your setup)
MODEL_PATH=${MODEL_PATH:-"/data/models/Qwen2.5-1.5B-Instruct"}
TRAIN_DATA=${TRAIN_DATA:-"/data/datasets/gsm8k/train.parquet"}
VAL_DATA=${VAL_DATA:-"/data/datasets/gsm8k/test.parquet"}

# Common training args (matching baseline config: batch_size=64 → 116 steps/epoch)
COMMON_ARGS="
    data.train_files=${TRAIN_DATA}
    data.val_files=${VAL_DATA}
    data.train_batch_size=64
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=16
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.enforce_eager=True
    algorithm.adv_estimator=grpo
    trainer.total_epochs=1
    trainer.test_freq=20
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

run_linear() {
    echo "=== Running Linear Layer HW Error Injection (197 hooks, ~95% FLOPs) ==="
    python3 -m verl.trainer.main_ppo \
        --config-name=ppo_trainer \
        ${COMMON_ARGS} \
        ${HW_ERROR_BASE} \
        "trainer.hw_error_injection.target_modules=['linear']" \
        trainer.experiment_name=hw_error_linear_${ERROR_SCALE} \
        2>&1 | tee /tmp/hw_error_linear.log
}

run_all() {
    echo "=== Running All Operators HW Error Injection (RMSNorm + Linear, ~96% FLOPs) ==="
    python3 -m verl.trainer.main_ppo \
        --config-name=ppo_trainer \
        ${COMMON_ARGS} \
        ${HW_ERROR_BASE} \
        "trainer.hw_error_injection.target_modules=['rmsnorm','linear']" \
        trainer.experiment_name=hw_error_all_${ERROR_SCALE} \
        2>&1 | tee /tmp/hw_error_all.log
}

run_comparison() {
    echo "=== Running Comparison: Linear 1e-5 vs Linear 1e-4 (4 GPUs each) ==="

    # Linear 1e-5 on GPUs 0-3
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main_ppo \
        --config-name=ppo_trainer \
        ${COMMON_ARGS} \
        trainer.hw_error_injection.enabled=True \
        trainer.hw_error_injection.error_scale=1e-5 \
        trainer.hw_error_injection.error_type=relative_gaussian \
        trainer.hw_error_injection.injection_point=input \
        trainer.hw_error_injection.apply_during=rollout \
        "trainer.hw_error_injection.target_modules=['linear']" \
        trainer.experiment_name=hw_error_linear_1e-5 \
        trainer.n_gpus_per_node=4 \
        2>&1 | tee /tmp/hw_error_linear_1e-5.log &
    PID1=$!

    # Linear 1e-4 on GPUs 4-7
    CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m verl.trainer.main_ppo \
        --config-name=ppo_trainer \
        ${COMMON_ARGS} \
        trainer.hw_error_injection.enabled=True \
        trainer.hw_error_injection.error_scale=1e-4 \
        trainer.hw_error_injection.error_type=relative_gaussian \
        trainer.hw_error_injection.injection_point=input \
        trainer.hw_error_injection.apply_during=rollout \
        "trainer.hw_error_injection.target_modules=['linear']" \
        trainer.experiment_name=hw_error_linear_1e-4 \
        trainer.n_gpus_per_node=4 \
        2>&1 | tee /tmp/hw_error_linear_1e-4.log &
    PID2=$!

    echo "Started Linear 1e-5 (PID: $PID1) and Linear 1e-4 (PID: $PID2)"
    echo "Logs: /tmp/hw_error_linear_1e-5.log, /tmp/hw_error_linear_1e-4.log"

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
    linear)
        run_linear
        ;;
    all)
        run_all
        ;;
    comparison)
        run_comparison
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [minimal|mlp|linear|all|comparison] [n_gpus] [error_scale]"
        echo ""
        echo "Modes:"
        echo "  minimal    - RMSNorm only (57 hooks, <1% FLOPs)"
        echo "  mlp        - RMSNorm + down_proj (85 hooks, ~15% FLOPs)"
        echo "  linear     - All Linear layers (197 hooks, ~95% FLOPs) [RECOMMENDED]"
        echo "  all        - RMSNorm + Linear (254 hooks, ~96% FLOPs)"
        echo "  comparison - Run Linear 1e-5 vs 1e-4 in parallel (4 GPUs each)"
        exit 1
        ;;
esac
