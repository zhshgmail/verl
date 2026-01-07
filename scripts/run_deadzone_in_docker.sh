#!/bin/bash
# Run verl deadzone test inside existing Docker container on A100
#
# Usage on A100 (90.90.102.18):
#   # SSH to A100 and enter existing container
#   ssh root@90.90.102.18
#   docker exec -it verl-r3-test bash
#   cd /home/z00637938/workspace/verl
#
#   # Pull latest code
#   git fetch personal
#   git checkout feature/npu-aqn-test
#   git pull personal feature/npu-aqn-test
#
#   # Run test
#   ./scripts/run_deadzone_in_docker.sh --dry-run  # verify only
#   ./scripts/run_deadzone_in_docker.sh            # full test

set -e

# Configuration for A100 server (90.90.102.18) - verl-r3-test container
VERL_DIR="${VERL_DIR:-/home/z00637938/workspace/verl}"
MODEL_PATH="${MODEL_PATH:-/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}"
DATA_DIR="${DATA_DIR:-/data/z00637938/gsm8k}"
N_GPUS="${N_GPUS:-2}"
TARGET_LAYER="${TARGET_LAYER:-10}"
DEADZONE_THRESHOLD="${DEADZONE_THRESHOLD:-0.01}"

# Parse arguments
DRY_RUN=false
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --dry-run      Only run verification, not full training"
            echo "  --help         Show this help"
            echo ""
            echo "Environment variables:"
            echo "  MODEL_PATH     Path to model"
            echo "  DATA_DIR       Path to data directory (contains train.parquet, test.parquet)"
            echo "  N_GPUS         Number of GPUs (default: 2)"
            echo "  TARGET_LAYER   Layer to inject deadzone (default: 10)"
            echo "  DEADZONE_THRESHOLD  Deadzone threshold (default: 0.01)"
            exit 0
            ;;
    esac
done

echo "=============================================="
echo "verl Deadzone Injection Test"
echo "=============================================="
echo "MODEL_PATH: $MODEL_PATH"
echo "DATA_DIR: $DATA_DIR"
echo "N_GPUS: $N_GPUS"
echo "TARGET_LAYER: $TARGET_LAYER"
echo "DEADZONE_THRESHOLD: $DEADZONE_THRESHOLD"
echo ""

# Verify we're in the right directory
if [ ! -f "verl/utils/hw_error_injection.py" ]; then
    echo "Error: hw_error_injection.py not found."
    echo "Make sure you're in the verl directory and on feature/npu-aqn-test branch"
    echo ""
    echo "Run:"
    echo "  cd $VERL_DIR"
    echo "  git fetch personal && git checkout feature/npu-aqn-test"
    exit 1
fi

# Check if deadzone support exists
if ! grep -q "deadzone" verl/utils/hw_error_injection.py; then
    echo "Error: deadzone support not found in hw_error_injection.py"
    echo "Make sure you pulled the latest code from feature/npu-aqn-test"
    exit 1
fi

# Install verl in editable mode if needed
echo "Step 0: Ensuring verl is installed..."
pip install -e . --no-deps -q 2>/dev/null || true

echo ""
echo "Step 1: Running basic verification..."
python scripts/test_verl_deadzone_injection.py --mode verify_hooks
if [ $? -ne 0 ]; then
    echo "Basic verification failed!"
    exit 1
fi

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "=============================================="
    echo "Dry run complete!"
    echo "=============================================="
    echo ""
    echo "To run full test:"
    echo "  $0"
    exit 0
fi

# Check model and data exist
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "Error: Train data not found: $DATA_DIR/train.parquet"
    exit 1
fi

echo ""
echo "Step 2: Running verl PPO with deadzone injection..."
echo ""

# Run verl PPO with deadzone enabled
python -m verl.trainer.main_ppo \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=64 \
    data.max_prompt_length=256 \
    data.max_response_length=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.ppo_mini_batch_size=64 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.total_epochs=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    "trainer.logger=['console']" \
    trainer.hw_error_injection.enabled=true \
    trainer.hw_error_injection.error_type=deadzone \
    trainer.hw_error_injection.injection_point=output \
    "trainer.hw_error_injection.target_layers=[$TARGET_LAYER]" \
    trainer.hw_error_injection.deadzone_threshold=$DEADZONE_THRESHOLD \
    trainer.hw_error_injection.apply_during=both \
    actor_rollout_ref.rollout.hw_error_injection_enabled=true \
    actor_rollout_ref.rollout.hw_error_injection_config.error_type=deadzone \
    actor_rollout_ref.rollout.hw_error_injection_config.injection_point=output \
    "actor_rollout_ref.rollout.hw_error_injection_config.target_layers=[$TARGET_LAYER]" \
    actor_rollout_ref.rollout.hw_error_injection_config.deadzone_threshold=$DEADZONE_THRESHOLD \
    actor_rollout_ref.rollout.hw_error_injection_config.apply_during=rollout \
    2>&1 | tee /tmp/verl_deadzone_test.log

echo ""
echo "=============================================="
echo "Test completed!"
echo "=============================================="
echo ""
echo "Check for [DEADZONE] messages:"
echo "  grep -i 'deadzone' /tmp/verl_deadzone_test.log"
echo ""
echo "Expected output:"
echo "  [DEADZONE] First injection on model.layers.$TARGET_LAYER: ..."
