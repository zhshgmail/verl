#!/bin/bash
# Archive experiment logs from A100 server to local
#
# Usage:
#   bash scripts/archive_experiment_logs.sh
#
# This script:
# 1. Fetches training logs from the A100 server
# 2. Archives them to logs/mxfp4_nvfp4_experiments/
# 3. Names files according to experiment results

set -e

REMOTE_HOST="root@90.90.102.18"
CONTAINER="verl-r3-test"
LOCAL_DIR="/home/zheng/workspace/verl/logs/mxfp4_nvfp4_experiments"

mkdir -p ${LOCAL_DIR}

echo "=== Archiving MXFP4/NVFP4 Experiment Logs ==="
echo "Remote: ${REMOTE_HOST} (container: ${CONTAINER})"
echo "Local: ${LOCAL_DIR}"
echo ""

# Function to fetch a log file
fetch_log() {
    local remote_path=$1
    local local_name=$2

    echo "Fetching: ${remote_path} -> ${local_name}"
    ssh ${REMOTE_HOST} "docker exec ${CONTAINER} cat ${remote_path}" > "${LOCAL_DIR}/${local_name}" 2>/dev/null || {
        echo "  WARNING: Failed to fetch ${remote_path}"
        return 1
    }
    local size=$(wc -c < "${LOCAL_DIR}/${local_name}")
    echo "  OK: ${size} bytes"
}

# Baseline
echo ""
echo "--- Baseline ---"
fetch_log "/tmp/mxfp4_exp_baseline/training.log" "baseline_clean_75.97.log" || true

# MXFP4-only
echo ""
echo "--- MXFP4-only ---"
fetch_log "/tmp/mxfp4_exp_mxfp4only/training.log" "mxfp4_only_70.05.log" || true

# Exp 1A (collapsed)
echo ""
echo "--- Exp 1A (Linear sigma=0.05, collapsed) ---"
fetch_log "/tmp/mxfp4_exp1a_aligned/training.log" "exp1a_linear_collapsed.log" || true

# Exp 1C (collapsed)
echo ""
echo "--- Exp 1C (Linear sigma=0.005, collapsed) ---"
fetch_log "/tmp/mxfp4_exp1c_small_sigma/training.log" "exp1c_linear_collapsed.log" || true

# Exp 1D (stable)
echo ""
echo "--- Exp 1D (Linear sigma=0.001, stable) ---"
fetch_log "/tmp/mxfp4_exp1d_tiny_sigma/training.log" "exp1d_linear_tiny_66.49.log" || true

# Exp 1E (RMSNorm)
echo ""
echo "--- Exp 1E (RMSNorm AQN) ---"
fetch_log "/tmp/mxfp4_exp1e_rmsnorm/training.log" "exp1e_rmsnorm_62.32.log" || true

# Original MXFP4+AQN experiments
echo ""
echo "--- Original MXFP4+AQN ---"
fetch_log "/tmp/mxfp4_aqn_experiment/training.log" "mxfp4_aqn_orig_67.48.log" || true
fetch_log "/tmp/mxfp4_aqn_experiment_v2/training.log" "mxfp4_aqn_v2.log" || true

# NVFP4 v1
echo ""
echo "--- NVFP4 v1 (in progress) ---"
fetch_log "/tmp/nvfp4_w4a16_aqn/training.log" "nvfp4_v1_aqn_inprogress.log" || true

echo ""
echo "=== Archive Complete ==="
echo "Files in ${LOCAL_DIR}:"
ls -la ${LOCAL_DIR}/
