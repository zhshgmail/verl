#!/bin/bash
# Parallel evaluation using simple shell-based process spawning
# Each GPU gets its own independent Python process

BASE_MODEL="/home/z00637938/workspace/verl/checkpoints/noisy_ops_e8c_forward_only/e8c_forward_only_5e-2/global_step_116/merged_hf"
LORA_ADAPTER="/tmp/mxfp4_w4a4_e13j_global_aqn/checkpoints/global_step_29/actor/lora_adapter"
DATA_PATH="/data/z00637938/gsm8k/test.parquet"
OUTPUT_DIR="/tmp/mxfp4_w4a4_e13j_global_aqn"

NUM_GPUS=8
TOTAL_SAMPLES=1319
SAMPLES_PER_GPU=$((($TOTAL_SAMPLES + $NUM_GPUS - 1) / $NUM_GPUS))

echo "Starting parallel evaluation on $NUM_GPUS GPUs"
echo "Total samples: $TOTAL_SAMPLES"
echo "Samples per GPU: ~$SAMPLES_PER_GPU"

# Launch 8 processes in parallel, each on its own GPU
for gpu_id in $(seq 0 $(($NUM_GPUS - 1))); do
    start_idx=$(($gpu_id * $SAMPLES_PER_GPU))

    # Calculate samples for this GPU
    if [ $gpu_id -eq $(($NUM_GPUS - 1)) ]; then
        # Last GPU gets remaining samples
        n_samples=$(($TOTAL_SAMPLES - $start_idx))
    else
        n_samples=$SAMPLES_PER_GPU
    fi

    echo "[GPU $gpu_id] Processing samples $start_idx to $(($start_idx + $n_samples - 1))"

    # Launch Python process with CUDA_VISIBLE_DEVICES set
    CUDA_VISIBLE_DEVICES=$gpu_id python3 /home/z00637938/workspace/verl/scripts/eval_lora_checkpoint_single_gpu.py \
        --base_model_path "$BASE_MODEL" \
        --lora_adapter_path "$LORA_ADAPTER" \
        --data_path "$DATA_PATH" \
        --start_idx $start_idx \
        --n_samples $n_samples \
        --gpu_id $gpu_id \
        > "$OUTPUT_DIR/eval_gpu${gpu_id}.log" 2>&1 &
done

echo "All GPU processes launched. Waiting for completion..."

# Wait for all background jobs
wait

echo "All processes completed. Collecting results..."

# Aggregate results
total_correct=0
total_samples=0

for gpu_id in $(seq 0 $(($NUM_GPUS - 1))); do
    if [ -f "$OUTPUT_DIR/eval_gpu${gpu_id}_result.txt" ]; then
        result=$(cat "$OUTPUT_DIR/eval_gpu${gpu_id}_result.txt")
        correct=$(echo $result | cut -d'/' -f1)
        samples=$(echo $result | cut -d'/' -f2)
        total_correct=$(($total_correct + $correct))
        total_samples=$(($total_samples + $samples))
        echo "[GPU $gpu_id] $correct/$samples correct"
    fi
done

accuracy=$(echo "scale=4; $total_correct * 100 / $total_samples" | bc)

echo "=================================================="
echo "Final Results:"
echo "  Accuracy: ${accuracy}%"
echo "  Correct: $total_correct/$total_samples"
echo "=================================================="
