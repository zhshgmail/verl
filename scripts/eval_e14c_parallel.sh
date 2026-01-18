#!/bin/bash
# E14c checkpoint evaluation - Parallel across 8 GPUs
#
# Runs 8 independent Python processes, each evaluating a slice of the data.
# Simple and reliable - no Ray required.
#
# Usage:
#   bash scripts/eval_e14c_parallel.sh [--nvfp4]

set -e

NVFP4_FLAG=""
if [[ "$1" == "--nvfp4" ]]; then
    NVFP4_FLAG="--nvfp4"
    echo "Mode: NVFP4 W4A4"
else
    echo "Mode: BF16 (no quantization)"
fi

TOTAL_SAMPLES=1319
NUM_GPUS=8
SAMPLES_PER_GPU=$(( (TOTAL_SAMPLES + NUM_GPUS - 1) / NUM_GPUS ))

OUTPUT_DIR="/tmp/eval_e14c_step29"
mkdir -p ${OUTPUT_DIR}

echo ""
echo "=================================================="
echo "E14c Step 29 Checkpoint Parallel Evaluation"
echo "=================================================="
echo "Total samples: ${TOTAL_SAMPLES}"
echo "Samples per GPU: ~${SAMPLES_PER_GPU}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

# Launch 8 processes in parallel
for gpu_id in $(seq 0 $(($NUM_GPUS - 1))); do
    start_idx=$(($gpu_id * $SAMPLES_PER_GPU))

    # Last GPU gets remaining samples
    if [ $gpu_id -eq $(($NUM_GPUS - 1)) ]; then
        n_samples=$(($TOTAL_SAMPLES - $start_idx))
    else
        n_samples=$SAMPLES_PER_GPU
    fi

    echo "[GPU $gpu_id] Starting: samples $start_idx to $(($start_idx + $n_samples - 1))"

    CUDA_VISIBLE_DEVICES=$gpu_id python3 /home/z00637938/workspace/verl/scripts/eval_e14c_slice.py \
        --start_idx $start_idx \
        --n_samples $n_samples \
        --gpu_id $gpu_id \
        --output_dir ${OUTPUT_DIR} \
        ${NVFP4_FLAG} \
        > "${OUTPUT_DIR}/gpu${gpu_id}.log" 2>&1 &
done

echo ""
echo "All GPU processes launched. Waiting for completion..."
echo ""

# Wait for all background jobs
wait

echo ""
echo "All processes completed. Collecting results..."
echo ""

# Aggregate results
total_correct=0
total_samples=0

for gpu_id in $(seq 0 $(($NUM_GPUS - 1))); do
    result_file="${OUTPUT_DIR}/gpu${gpu_id}_result.txt"
    if [ -f "${result_file}" ]; then
        result=$(cat "${result_file}")
        correct=$(echo $result | cut -d'/' -f1)
        samples=$(echo $result | cut -d'/' -f2)
        total_correct=$(($total_correct + $correct))
        total_samples=$(($total_samples + $samples))
        echo "[GPU $gpu_id] ${correct}/${samples} correct"
    else
        echo "[GPU $gpu_id] ERROR: result file not found"
        echo "Check log: ${OUTPUT_DIR}/gpu${gpu_id}.log"
    fi
done

echo ""
echo "=================================================="
if [ $total_samples -gt 0 ]; then
    accuracy=$(echo "scale=4; $total_correct * 100 / $total_samples" | bc)
    echo "FINAL RESULT:"
    echo "  Accuracy: ${accuracy}%"
    echo "  Correct: ${total_correct}/${total_samples}"
    echo "=================================================="
    echo ""
    echo "COMPARISON:"
    echo "  E13j_v4 (MXFP4): Step 20=70.28%, Step 29=66.11% (-4.17%)"
    echo "  E14c (NVFP4): Step 20=70.74%, Step 29=${accuracy}%"
    echo ""

    # Parse accuracy as integer for comparison
    acc_int=$(echo "$accuracy" | cut -d'.' -f1)
    if [ "$acc_int" -lt 71 ]; then
        echo "VERDICT: E14c step 29 < step 20 (70.74%) -> BUG in our code"
    else
        echo "VERDICT: E14c step 29 >= step 20 -> Issue is MXFP4-specific"
    fi
else
    echo "ERROR: No results collected"
fi
echo "=================================================="
