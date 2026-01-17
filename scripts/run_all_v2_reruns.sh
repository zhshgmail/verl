#!/bin/bash
# Master script to run all v2 re-run experiments in sequence
# This ensures all experiments complete with bug fixes applied

set -e

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N_GPUS=${1:-8}

echo "=========================================="
echo "E13/E14 v2 RE-RUN EXPERIMENTS"
echo "=========================================="
echo ""
echo "This will run 6 experiments in sequence:"
echo "  1. E13j_v2 (CRITICAL) - Global AQN baseline"
echo "  2. E13k_v2 (High)     - QeRL sigma test"
echo "  3. E13l_v2 (High)     - Variable RIN (high→MORE noise)"
echo "  4. E13m_v2 (High)     - Inverse RIN (high→LESS noise)"
echo "  5. E13n_v2 (High)     - Ceiling RIN"
echo "  6. E14a_v2 (Medium)   - Zone-based scheduling"
echo ""
echo "Bug fixes applied:"
echo "  ✅ Skip final validation removed"
echo "  ✅ MXFP4 verified active during validation"
echo "  ✅ New IDs preserve original logs/checkpoints"
echo ""
echo "Using ${N_GPUS} GPUs per experiment"
echo "Estimated time: ~6-8 hours total (80 min × 6)"
echo ""
read -p "Press Enter to start, or Ctrl+C to cancel..."
echo ""

# Function to run experiment with error handling
run_experiment() {
    local script=$1
    local exp_name=$2
    local priority=$3

    echo "=========================================="
    echo "Starting: $exp_name [$priority priority]"
    echo "Script: $(basename $script)"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""

    if bash "$script" "$N_GPUS"; then
        echo ""
        echo "✅ $exp_name completed successfully!"
        echo ""
    else
        echo ""
        echo "❌ $exp_name FAILED!"
        echo "Check logs for errors"
        echo ""
        read -p "Continue with next experiment? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Run experiments in priority order
run_experiment "$SCRIPTS_DIR/test_mxfp4_w4a4_e13j_v2_global_aqn.sh" "E13j_v2" "CRITICAL"
run_experiment "$SCRIPTS_DIR/test_mxfp4_w4a4_e13k_aqn_qerl_sigma_v2.sh" "E13k_v2" "High"
run_experiment "$SCRIPTS_DIR/test_mxfp4_w4a4_e13l_variable_rin_v2.sh" "E13l_v2" "High"
run_experiment "$SCRIPTS_DIR/test_mxfp4_w4a4_e13m_inverse_rin_v2.sh" "E13m_v2" "High"
run_experiment "$SCRIPTS_DIR/test_mxfp4_w4a4_e13n_ceiling_rin_v2.sh" "E13n_v2" "High"
run_experiment "$SCRIPTS_DIR/test_mxfp4_w4a4_e14a_zone_schedule_v2.sh" "E14a_v2" "Medium"

echo ""
echo "=========================================="
echo "ALL RE-RUNS COMPLETE!"
echo "=========================================="
echo ""
echo "Results summary:"
echo "  E13j_v2: /tmp/mxfp4_w4a4_e13j_v2_global_aqn/training.log"
echo "  E13k_v2: /tmp/mxfp4_w4a4_e13k_v2_aqn_qerl_sigma/training.log"
echo "  E13l_v2: /tmp/mxfp4_w4a4_e13l_v2_variable_rin/training.log"
echo "  E13m_v2: /tmp/mxfp4_w4a4_e13m_v2_inverse_rin/training.log"
echo "  E13n_v2: /tmp/mxfp4_w4a4_e13n_v2_ceiling_rin/training.log"
echo "  E14a_v2: /tmp/mxfp4_w4a4_e14a_v2_zone_schedule/training.log"
echo ""
echo "Extract final validation results:"
echo "  for log in /tmp/mxfp4_w4a4_e13*_v2_*/training.log; do"
echo "    echo \"=== \$(basename \$(dirname \$log)) ===\";"
echo "    grep 'val-core' \$log | tail -2;"
echo "  done"
echo ""
echo "Next steps:"
echo "  1. Compare v2 step 29 results with original step 20 results"
echo "  2. Update docs/qerl/ALL_EXPERIMENTS_SUMMARY.md"
echo "  3. Update docs/qerl/EXPERIMENT_BUG_IMPACT_ANALYSIS.md with findings"
