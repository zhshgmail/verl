#!/bin/bash
# Evaluate E13l checkpoint across 8 GPUs in parallel
# GSM8K test set: 1319 samples

set -x

BASE_MODEL="/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
LORA_ADAPTER="/tmp/mxfp4_w4a4_e13l_variable_rin/checkpoints/global_step_29/actor/lora_adapter"
DATA_PATH="/data/z00637938/gsm8k/test.parquet"
OUTPUT_DIR="/tmp/e13l_eval"

mkdir -p ${OUTPUT_DIR}

# Total samples: 1319
# Samples per GPU: 165 (last GPU gets remainder)
SAMPLES_PER_GPU=165

echo "Starting E13l evaluation on 8 GPUs..."
echo "Base model: ${BASE_MODEL}"
echo "LoRA adapter: ${LORA_ADAPTER}"
echo "Data: ${DATA_PATH}"
echo ""

# Launch 8 evaluation processes in parallel
for gpu_id in {0..7}; do
    start_idx=$((gpu_id * SAMPLES_PER_GPU))

    # Last GPU gets all remaining samples
    if [ $gpu_id -eq 7 ]; then
        n_samples=$((1319 - start_idx))
    else
        n_samples=${SAMPLES_PER_GPU}
    fi

    echo "[GPU ${gpu_id}] Processing samples ${start_idx}-$((start_idx + n_samples - 1))"

    CUDA_VISIBLE_DEVICES=${gpu_id} python3 scripts/eval_lora_checkpoint_single_gpu.py \
        --base_model_path ${BASE_MODEL} \
        --lora_adapter_path ${LORA_ADAPTER} \
        --data_path ${DATA_PATH} \
        --start_idx ${start_idx} \
        --n_samples ${n_samples} \
        --gpu_id ${gpu_id} \
        --max_tokens 512 \
        > ${OUTPUT_DIR}/eval_gpu${gpu_id}.log 2>&1 &
done

echo ""
echo "All evaluation processes launched. Waiting for completion..."
wait

echo ""
echo "All evaluations complete! Aggregating results..."

# Aggregate results
total_correct=0
total_samples=0

for gpu_id in {0..7}; do
    result_file="/tmp/mxfp4_w4a4_e13j_global_aqn/eval_gpu${gpu_id}_result.txt"
    if [ -f "$result_file" ]; then
        result=$(cat $result_file)
        correct=$(echo $result | cut -d'/' -f1)
        samples=$(echo $result | cut -d'/' -f2)
        total_correct=$((total_correct + correct))
        total_samples=$((total_samples + samples))

        acc=$(awk "BEGIN {printf \"%.2f\", ($correct / $samples) * 100}")
        echo "[GPU ${gpu_id}] ${correct}/${samples} = ${acc}%"
    else
        echo "[GPU ${gpu_id}] ERROR: Result file not found!"
    fi
done

echo ""
echo "========================================="
if [ $total_samples -gt 0 ]; then
    final_acc=$(awk "BEGIN {printf \"%.2f\", ($total_correct / $total_samples) * 100}")
    echo "E13l Final Accuracy: ${final_acc}%"
    echo "Correct: ${total_correct}/${total_samples}"
else
    echo "ERROR: No samples processed!"
fi
echo "========================================="

# Save summary
cat > ${OUTPUT_DIR}/summary.txt << EOF
E13l Evaluation Summary
=======================
Checkpoint: ${LORA_ADAPTER}
Test Set: ${DATA_PATH}
Total Samples: ${total_samples}
Correct: ${total_correct}
Accuracy: ${final_acc}%

Per-GPU Results:
EOF

for gpu_id in {0..7}; do
    result_file="/tmp/mxfp4_w4a4_e13j_global_aqn/eval_gpu${gpu_id}_result.txt"
    if [ -f "$result_file" ]; then
        result=$(cat $result_file)
        echo "GPU ${gpu_id}: ${result}" >> ${OUTPUT_DIR}/summary.txt
    fi
done

echo ""
echo "Summary saved to: ${OUTPUT_DIR}/summary.txt"
