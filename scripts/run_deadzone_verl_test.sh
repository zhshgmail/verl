#!/bin/bash
# Run verl PPO training with deadzone injection for SRDD-guided AQN experiment
#
# This script tests that deadzone injection works consistently in both:
# 1. vLLM rollout (inference)
# 2. FSDP training (forward pass)
#
# Usage:
#   # Dry run (just verify config)
#   ./scripts/run_deadzone_verl_test.sh --dry-run
#
#   # Full test (1 training step)
#   ./scripts/run_deadzone_verl_test.sh
#
# Prerequisites:
#   - 2+ A100 GPUs
#   - Qwen2.5-0.5B model (or similar small model)
#   - GSM8K data (or other RL dataset)

set -e

# Configuration
MODEL_PATH="${MODEL_PATH:-/path/to/Qwen2.5-0.5B}"
DATA_PATH="${DATA_PATH:-/path/to/gsm8k}"
TARGET_LAYER="${TARGET_LAYER:-15}"
DEADZONE_THRESHOLD="${DEADZONE_THRESHOLD:-0.01}"
N_GPUS="${N_GPUS:-2}"

# Parse arguments
DRY_RUN=false
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
    esac
done

echo "=============================================="
echo "verl Deadzone Injection Test"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Target layer: $TARGET_LAYER"
echo "Deadzone threshold: $DEADZONE_THRESHOLD"
echo "GPUs: $N_GPUS"
echo ""

# First run basic verification
echo "Step 1: Running basic verification..."
python scripts/test_verl_deadzone_injection.py --mode verify_hooks
if [ $? -ne 0 ]; then
    echo "Basic verification failed!"
    exit 1
fi

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "Dry run complete. To run full test, remove --dry-run flag."
    exit 0
fi

# Check model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    echo "Please set MODEL_PATH environment variable"
    exit 1
fi

# Check data exists
if [ ! -f "$DATA_PATH/train.parquet" ]; then
    echo "Error: Data path does not exist: $DATA_PATH/train.parquet"
    echo "Please set DATA_PATH environment variable"
    exit 1
fi

echo ""
echo "Step 2: Running verl PPO with deadzone injection..."
echo ""

# Run verl with deadzone enabled
# Note: This uses the default ppo_trainer.yaml with overrides
python -m verl.trainer.main_ppo \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    data.train_files="$DATA_PATH/train.parquet" \
    data.val_files="$DATA_PATH/test.parquet" \
    data.train_batch_size=64 \
    data.max_prompt_length=256 \
    data.max_response_length=256 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.total_epochs=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.logger="['console']" \
    trainer.hw_error_injection.enabled=true \
    trainer.hw_error_injection.error_type=deadzone \
    trainer.hw_error_injection.injection_point=output \
    trainer.hw_error_injection.target_layers="[$TARGET_LAYER]" \
    trainer.hw_error_injection.deadzone_threshold=$DEADZONE_THRESHOLD \
    trainer.hw_error_injection.apply_during=both \
    actor_rollout_ref.rollout.hw_error_injection_enabled=true \
    'actor_rollout_ref.rollout.hw_error_injection_config.error_type=deadzone' \
    'actor_rollout_ref.rollout.hw_error_injection_config.injection_point=output' \
    "actor_rollout_ref.rollout.hw_error_injection_config.target_layers=[$TARGET_LAYER]" \
    "actor_rollout_ref.rollout.hw_error_injection_config.deadzone_threshold=$DEADZONE_THRESHOLD" \
    'actor_rollout_ref.rollout.hw_error_injection_config.apply_during=rollout' \
    2>&1 | tee /tmp/verl_deadzone_test.log

echo ""
echo "=============================================="
echo "Test completed!"
echo "=============================================="
echo ""
echo "Check log for:"
echo "  1. '[DEADZONE] First injection on' - confirms injection is active"
echo "  2. Consistent injection in both rollout and training"
echo ""
echo "Log saved to: /tmp/verl_deadzone_test.log"
echo ""
echo "To grep for deadzone messages:"
echo "  grep -i 'deadzone\|hw.*error' /tmp/verl_deadzone_test.log"
