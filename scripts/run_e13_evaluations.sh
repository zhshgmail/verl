#!/bin/bash
# Run all E13 evaluations with MXFP4 fake quantization
# This script runs on the remote server with nohup to survive client disconnection

set -e

LOG_DIR="/tmp/e13_eval_results"
mkdir -p $LOG_DIR

BASE_MODEL="/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DATA_PATH="/data/z00637938/gsm8k/test.parquet"

cd /home/z00637938/workspace/verl

# Start Ray cluster
echo "Starting Ray cluster..."
ray start --head --port=6379 --num-gpus=8 --disable-usage-stats

sleep 10

# Define experiments and their checkpoint paths
declare -A EXPERIMENTS
EXPERIMENTS["e13j"]="/tmp/mxfp4_w4a4_e13j_global_aqn/checkpoints/global_step_29/actor/lora_adapter"
EXPERIMENTS["e13k"]="/tmp/mxfp4_w4a4_e13k_aqn_qerl_sigma/checkpoints/global_step_29/actor/lora_adapter"
EXPERIMENTS["e13l"]="/tmp/mxfp4_w4a4_e13l_variable_rin/checkpoints/global_step_29/actor/lora_adapter"
EXPERIMENTS["e13m"]="/tmp/mxfp4_w4a4_e13m_inverse_rin/checkpoints/global_step_29/actor/lora_adapter"
EXPERIMENTS["e13n"]="/tmp/mxfp4_w4a4_e13n_ceiling_rin/checkpoints/global_step_29/actor/lora_adapter"

echo "=== E13 Evaluation Suite with MXFP4 W4A4 ==="
echo "Base model: $BASE_MODEL"
echo "Data: $DATA_PATH (GSM8K test set)"
echo "Results will be saved to: $LOG_DIR"
echo ""

# Run each evaluation
for exp in e13j e13k e13l e13m e13n; do
    CKPT_PATH="${EXPERIMENTS[$exp]}"
    LOG_FILE="$LOG_DIR/${exp}_mxfp4_eval.log"

    echo "========================================"
    echo "Evaluating $exp"
    echo "Checkpoint: $CKPT_PATH"
    echo "Log: $LOG_FILE"
    echo "========================================"

    if [ ! -d "$CKPT_PATH" ]; then
        echo "WARNING: Checkpoint not found: $CKPT_PATH"
        echo "SKIPPED: $exp - checkpoint not found" >> $LOG_DIR/summary.txt
        continue
    fi

    python scripts/eval_lora_checkpoint_parallel.py \
        --mxfp4 \
        --mxfp4_injection_point both \
        --base_model_path "$BASE_MODEL" \
        --lora_adapter_path "$CKPT_PATH" \
        --data_path "$DATA_PATH" \
        --num_workers 8 \
        2>&1 | tee "$LOG_FILE"

    # Extract result
    RESULT=$(grep -E "Accuracy:" "$LOG_FILE" | tail -1)
    echo "$exp: $RESULT" >> $LOG_DIR/summary.txt
    echo ""
done

echo "=== All Evaluations Complete ==="
echo "Summary:"
cat $LOG_DIR/summary.txt

ray stop
