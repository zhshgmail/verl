#!/bin/bash
# Robustness Evaluation Script
#
# Evaluates a checkpoint with BOTH clean and noisy inference to measure robustness.
# A robust model should maintain similar accuracy whether noise is present or not.
#
# Usage:
#   bash scripts/eval_robustness.sh <CHECKPOINT_PATH> <DATA_PATH> [ERROR_SCALE]
#
# Example:
#   bash scripts/eval_robustness.sh \
#       /data/checkpoints/noisy_ops_5e-2/global_step_116/actor \
#       /data/datasets/gsm8k/test.parquet \
#       5e-2
#
# Output:
#   - results_clean.json: Accuracy without noise (clean hardware simulation)
#   - results_noisy.json: Accuracy with noise (noisy hardware simulation)
#   - Comparison summary printed to console

set -e

CHECKPOINT_PATH=${1:?Error: CHECKPOINT_PATH required}
DATA_PATH=${2:?Error: DATA_PATH required}
ERROR_SCALE=${3:-5e-2}
N_SAMPLES=${4:-5}

OUTPUT_DIR="${CHECKPOINT_PATH}/eval_robustness"
mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "Robustness Evaluation"
echo "============================================================"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Data: ${DATA_PATH}"
echo "Error scale: ${ERROR_SCALE}"
echo "Output dir: ${OUTPUT_DIR}"
echo "============================================================"

# 1. Clean evaluation (no noise)
echo ""
echo ">>> Running CLEAN evaluation (VERL_NOISY_OPS_ENABLED=0)..."
VERL_NOISY_OPS_ENABLED=0 python3 scripts/clean_eval_checkpoint.py \
    --model_path "${CHECKPOINT_PATH}" \
    --data_path "${DATA_PATH}" \
    --n_samples ${N_SAMPLES} \
    --output_file "${OUTPUT_DIR}/results_clean.json"

# 2. Noisy evaluation (with noise)
echo ""
echo ">>> Running NOISY evaluation (VERL_NOISY_OPS_ENABLED=1, scale=${ERROR_SCALE})..."
VERL_NOISY_OPS_ENABLED=1 \
VERL_NOISY_OPS_SCALE=${ERROR_SCALE} \
VERL_NOISY_OPS_TYPE=relative_gaussian \
python3 scripts/clean_eval_checkpoint.py \
    --model_path "${CHECKPOINT_PATH}" \
    --data_path "${DATA_PATH}" \
    --n_samples ${N_SAMPLES} \
    --output_file "${OUTPUT_DIR}/results_noisy.json"

# 3. Summary
echo ""
echo "============================================================"
echo "ROBUSTNESS SUMMARY"
echo "============================================================"

CLEAN_ACC=$(python3 -c "import json; print(json.load(open('${OUTPUT_DIR}/results_clean.json'))['accuracy_best_of_n'])")
NOISY_ACC=$(python3 -c "import json; print(json.load(open('${OUTPUT_DIR}/results_noisy.json'))['accuracy_best_of_n'])")

echo "Clean accuracy:  ${CLEAN_ACC}%"
echo "Noisy accuracy:  ${NOISY_ACC}%"
echo ""
echo "Robustness = Clean - Noisy"
python3 -c "
clean = ${CLEAN_ACC}
noisy = ${NOISY_ACC}
diff = clean - noisy
print(f'Difference: {diff:+.2f}%')
if abs(diff) < 1.0:
    print('✓ Model is ROBUST (< 1% degradation)')
elif abs(diff) < 3.0:
    print('⚠ Model is MODERATELY ROBUST (1-3% degradation)')
else:
    print('✗ Model is NOT ROBUST (> 3% degradation)')
"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  - ${OUTPUT_DIR}/results_clean.json"
echo "  - ${OUTPUT_DIR}/results_noisy.json"
