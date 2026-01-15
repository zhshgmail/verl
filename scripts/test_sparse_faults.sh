#!/bin/bash
# Comprehensive Sparse Fault Detection Test Suite
# Tests SRDD's ability to detect sparse (10%) variants of all fault types
#
# Usage: bash scripts/test_sparse_faults.sh
#
# Expected results:
# - Dense faults: All detectable (baseline)
# - Sparse saturation: Known limitation (fails)
# - Sparse dead_zone/noise/bias/spike: TBD

MODEL_PATH="/home/z00637938/workspace/verl/checkpoints/noisy_ops_e8c_forward_only/e8c_forward_only_5e-2/global_step_116/merged_hf"
GROUND_TRUTH_LAYER=10
SPARSITY=0.1  # 10% of neurons affected
FAULT_MAG=0.3

echo "======================================"
echo "SRDD Sparse Fault Detection Test Suite"
echo "======================================"
echo "Model: $MODEL_PATH"
echo "Ground truth layer: $GROUND_TRUTH_LAYER"
echo "Sparsity: ${SPARSITY} ($(echo "$SPARSITY * 100" | bc)% of neurons)"
echo "Fault magnitude: $FAULT_MAG"
echo ""

# Test each fault type
for FAULT_TYPE in dead_zone noise saturation bias spike; do
    echo ""
    echo "========================================"
    echo "Testing SPARSE $FAULT_TYPE (${SPARSITY} sparsity)"
    echo "========================================"

    python scripts/srdd_error_finder.py \
        --model_path "$MODEL_PATH" \
        --ground_truth_layer $GROUND_TRUTH_LAYER \
        --fault_type "$FAULT_TYPE" \
        --fault_magnitude $FAULT_MAG \
        --sparsity $SPARSITY

    echo ""
    echo "--- Finished $FAULT_TYPE ---"
    echo ""
done

echo ""
echo "======================================"
echo "All sparse fault tests completed"
echo "======================================"
