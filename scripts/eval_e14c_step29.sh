#!/bin/bash
# Evaluate E14c NVFP4 W4A4 step 29 checkpoint
#
# This uses direct HuggingFace + PEFT loading (NOT verl's val_only mode)
# which bypasses the Ray event loop issues.
#
# Usage:
#   bash scripts/eval_e14c_step29.sh [NUM_WORKERS]

set -e

NUM_WORKERS=${1:-8}

BASE_MODEL="/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
LORA_ADAPTER="/tmp/nvfp4_w4a4_e14c_sigma_decay/checkpoints/global_step_29/actor/lora_adapter"
DATA_PATH="/data/z00637938/gsm8k/test.parquet"

echo "=================================================="
echo "E14c NVFP4 W4A4 Step 29 Checkpoint Evaluation"
echo "=================================================="
echo ""
echo "Configuration:"
echo "  - Base model: ${BASE_MODEL}"
echo "  - LoRA adapter: ${LORA_ADAPTER}"
echo "  - Test data: ${DATA_PATH}"
echo "  - Workers: ${NUM_WORKERS}"
echo "  - Quantization: NVFP4 W4A4"
echo ""
echo "Context:"
echo "  - E13j_v4 (MXFP4): Step 20=70.28%, Step 29=66.11% (-4.17% degradation)"
echo "  - E14c (NVFP4): Step 20=70.74%, Step 29=???"
echo ""
echo "If E14c step 29 < step 20 -> BUG in our code"
echo "If E14c step 29 >= step 20 -> MXFP4-specific issue"
echo ""

# Check if checkpoint exists
if [ ! -d "${LORA_ADAPTER}" ]; then
    echo "ERROR: LoRA adapter not found at ${LORA_ADAPTER}"
    echo "Make sure E14c experiment completed and checkpoint was saved."
    exit 1
fi

echo "Starting evaluation..."
echo ""

python3 /home/z00637938/workspace/verl/scripts/eval_lora_checkpoint_general.py \
    --base_model_path "${BASE_MODEL}" \
    --lora_adapter_path "${LORA_ADAPTER}" \
    --data_path "${DATA_PATH}" \
    --num_workers ${NUM_WORKERS} \
    --quant_type nvfp4 \
    --injection_point both \
    --exclude_modules lm_head embed_tokens layers.0 layers.27

echo ""
echo "=================================================="
echo "COMPARISON:"
echo "  - E13j_v4 (MXFP4): 70.28% -> 66.11% (-4.17% degradation)"
echo "  - E14c (NVFP4): 70.74% -> [SEE ABOVE]"
echo ""
echo "VERDICT:"
echo "  - If E14c step 29 accuracy < 70.74%: BUG in our code"
echo "  - If E14c step 29 accuracy >= 70.74%: Issue is MXFP4-specific"
echo "=================================================="
