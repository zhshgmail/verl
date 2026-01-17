#!/bin/bash
# Script to create v2 versions of all affected experiments
# This preserves original logs and ensures proper re-run with bug fixes

set -e

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Creating v2 re-run scripts for all affected experiments..."
echo ""

# Function to create v2 version
create_v2_script() {
    local original=$1
    local exp_id=$2
    local priority=$3

    if [ ! -f "$original" ]; then
        echo "ERROR: Original script not found: $original"
        return 1
    fi

    local v2_script="${original%.sh}_v2.sh"

    # Skip if already exists
    if [ -f "$v2_script" ]; then
        echo "  ✓ $v2_script already exists, skipping"
        return 0
    fi

    echo "  Creating: $(basename $v2_script) [$priority priority]"

    # Create v2 by modifying original
    sed "s|OUTPUT_DIR=\"/tmp/mxfp4_w4a4_${exp_id}_|OUTPUT_DIR=\"/tmp/mxfp4_w4a4_${exp_id}_v2_|g" "$original" | \
    sed "s|Experiment ID: ${exp_id}|Experiment ID: ${exp_id}_v2 (RE-RUN with bug fixes)|" | \
    sed "s|Date: 2026-01-1[6-7]|Date: 2026-01-17 (RE-RUN)|" | \
    sed "s|project_name=w4a4_${exp_id}_|project_name=w4a4_${exp_id}_v2_|" | \
    sed "s|experiment_name=|experiment_name=${exp_id}_v2_|" | \
    sed '1 a\
# RE-RUN with bug fixes applied (2026-01-17)\
# - Bug #1 FIXED: Skip final validation removed\
# - Bug #2 VERIFIED: MXFP4 hooks active during validation\
# - Uses v2 ID to preserve original logs/checkpoints\
' > "$v2_script"

    chmod +x "$v2_script"
}

# E13j_v2 - Already created manually, skip
if [ -f "$SCRIPTS_DIR/test_mxfp4_w4a4_e13j_v2_global_aqn.sh" ]; then
    echo "✓ E13j_v2 already exists (created manually)"
else
    create_v2_script "$SCRIPTS_DIR/test_mxfp4_w4a4_e13j_global_aqn.sh" "e13j" "CRITICAL"
fi

# E13k_v2
create_v2_script "$SCRIPTS_DIR/test_mxfp4_w4a4_e13k_aqn_qerl_sigma.sh" "e13k" "High"

# E13l_v2
create_v2_script "$SCRIPTS_DIR/test_mxfp4_w4a4_e13l_variable_rin.sh" "e13l" "High"

# E13m_v2
create_v2_script "$SCRIPTS_DIR/test_mxfp4_w4a4_e13m_inverse_rin.sh" "e13m" "High"

# E13n_v2
create_v2_script "$SCRIPTS_DIR/test_mxfp4_w4a4_e13n_ceiling_rin.sh" "e13n" "High"

# E14a_v2
create_v2_script "$SCRIPTS_DIR/test_mxfp4_w4a4_e14a_zone_schedule.sh" "e14a" "Medium"

echo ""
echo "✅ All v2 re-run scripts created successfully!"
echo ""
echo "Re-run scripts:"
ls -1 "$SCRIPTS_DIR"/test_mxfp4_w4a4_e13*_v2*.sh "$SCRIPTS_DIR"/test_mxfp4_w4a4_e14*_v2*.sh 2>/dev/null || true
echo ""
echo "To run all experiments in sequence:"
echo "  bash scripts/run_all_v2_reruns.sh"
