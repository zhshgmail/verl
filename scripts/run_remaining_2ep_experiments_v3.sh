#!/bin/bash
# Run remaining 2-epoch experiments (v3)
# Includes: E7a-2ep rerun, E12-2ep (NEW), E3b-2ep, E8a-2ep
# Skips: E6b-2ep (73.24%), E6a-2ep (72.93%), E3a-2ep (running/done)
#
# Date: 2026-01-12
# Estimated time: ~10 hours (4 experiments x ~2.5h each)

set -x

LOG_DIR="/tmp/2ep_experiments_master"
mkdir -p ${LOG_DIR}

echo "=== Starting Remaining 2-Epoch Experiments (v3) ===" | tee -a ${LOG_DIR}/master_v3.log
echo "Start time: $(date)" | tee -a ${LOG_DIR}/master_v3.log
echo "" | tee -a ${LOG_DIR}/master_v3.log
echo "Completed experiments (skipping):" | tee -a ${LOG_DIR}/master_v3.log
echo "  - E6b-2ep: 73.24% (MXFP4 + LoRA + AQN)" | tee -a ${LOG_DIR}/master_v3.log
echo "  - E6a-2ep: 72.93% (MXFP4 + LoRA)" | tee -a ${LOG_DIR}/master_v3.log
echo "  - E3a-2ep: 73.92%@step40 (MXFP4 + Full FT) - check if complete" | tee -a ${LOG_DIR}/master_v3.log
echo "" | tee -a ${LOG_DIR}/master_v3.log

cd /home/z00637938/workspace/verl

# 1. E7a-2ep RERUN: LoRA_BF16 (ended early at step 40)
echo "=== [1/4] E7a-2ep RERUN: BF16 + LoRA ===" | tee -a ${LOG_DIR}/master_v3.log
echo "Previous run ended early at step 40 with 73.84%" | tee -a ${LOG_DIR}/master_v3.log
echo "Start: $(date)" | tee -a ${LOG_DIR}/master_v3.log
# Backup old log if exists
if [ -f /tmp/bf16_v7.0_dapo_lora_2ep/training.log ]; then
    mv /tmp/bf16_v7.0_dapo_lora_2ep/training.log /tmp/bf16_v7.0_dapo_lora_2ep/training.log.bak.$(date +%Y%m%d_%H%M%S)
fi
bash scripts/test_bf16_v7.0_dapo_lora_2ep.sh 8
echo "End: $(date)" | tee -a ${LOG_DIR}/master_v3.log
grep "val-core/openai/gsm8k/acc/mean" /tmp/bf16_v7.0_dapo_lora_2ep/training.log | tail -1 >> ${LOG_DIR}/master_v3.log
echo "" | tee -a ${LOG_DIR}/master_v3.log

# 2. E12-2ep: LoRA_MXFP4_AQN-high (BEST 1ep config - 72.48%)
echo "=== [2/4] E12-2ep: MXFP4 + LoRA + AQN-high (BEST 1ep) ===" | tee -a ${LOG_DIR}/master_v3.log
echo "1ep achieved 72.48% - exceeded BF16 baseline!" | tee -a ${LOG_DIR}/master_v3.log
echo "Expected 2ep: ~75-77%" | tee -a ${LOG_DIR}/master_v3.log
echo "Start: $(date)" | tee -a ${LOG_DIR}/master_v3.log
bash scripts/test_mxfp4_v6.2_dapo_lora_aqn_high_sigma_2ep.sh 8
echo "End: $(date)" | tee -a ${LOG_DIR}/master_v3.log
grep "val-core/openai/gsm8k/acc/mean" /tmp/mxfp4_v6.2_dapo_lora_aqn_high_sigma_2ep/training.log | tail -1 >> ${LOG_DIR}/master_v3.log
echo "" | tee -a ${LOG_DIR}/master_v3.log

# 3. E3b-2ep: Q_MXFP4_fullFT_AQN
echo "=== [3/4] E3b-2ep: MXFP4 + Full FT + AQN ===" | tee -a ${LOG_DIR}/master_v3.log
echo "1ep achieved 74.37%" | tee -a ${LOG_DIR}/master_v3.log
echo "Start: $(date)" | tee -a ${LOG_DIR}/master_v3.log
bash scripts/test_mxfp4_v3.1_dapo_aqn_2ep.sh 8
echo "End: $(date)" | tee -a ${LOG_DIR}/master_v3.log
grep "val-core/openai/gsm8k/acc/mean" /tmp/mxfp4_v3.1_dapo_aqn_2ep/training.log | tail -1 >> ${LOG_DIR}/master_v3.log
echo "" | tee -a ${LOG_DIR}/master_v3.log

# 4. E8a-2ep: Q_BF16_fullFT
echo "=== [4/4] E8a-2ep: BF16 + Full FT ===" | tee -a ${LOG_DIR}/master_v3.log
echo "1ep achieved 74.75% - BF16 baseline" | tee -a ${LOG_DIR}/master_v3.log
echo "Start: $(date)" | tee -a ${LOG_DIR}/master_v3.log
bash scripts/test_bf16_v8.0_dapo_fullft_2ep.sh 8
echo "End: $(date)" | tee -a ${LOG_DIR}/master_v3.log
grep "val-core/openai/gsm8k/acc/mean" /tmp/bf16_v8.0_dapo_fullft_2ep/training.log | tail -1 >> ${LOG_DIR}/master_v3.log
echo "" | tee -a ${LOG_DIR}/master_v3.log

echo "=== ALL REMAINING EXPERIMENTS COMPLETED ===" | tee -a ${LOG_DIR}/master_v3.log
echo "End time: $(date)" | tee -a ${LOG_DIR}/master_v3.log
echo "" | tee -a ${LOG_DIR}/master_v3.log

echo "=== FINAL RESULTS SUMMARY ===" | tee -a ${LOG_DIR}/master_v3.log
echo "" | tee -a ${LOG_DIR}/master_v3.log
echo "LoRA Experiments:" | tee -a ${LOG_DIR}/master_v3.log
for dir in mxfp4_v6.1_dapo_lora_aqn_2ep mxfp4_v6.0_dapo_lora_2ep bf16_v7.0_dapo_lora_2ep mxfp4_v6.2_dapo_lora_aqn_high_sigma_2ep; do
    acc=$(grep "val-core/openai/gsm8k/acc/mean" /tmp/${dir}/training.log 2>/dev/null | tail -1 | grep -oP 'np.float64\(\K[0-9.]+')
    echo "  ${dir}: ${acc:-N/A}" | tee -a ${LOG_DIR}/master_v3.log
done
echo "" | tee -a ${LOG_DIR}/master_v3.log
echo "Full FT Experiments:" | tee -a ${LOG_DIR}/master_v3.log
for dir in mxfp4_v3.0_dapo_2ep mxfp4_v3.1_dapo_aqn_2ep bf16_v8.0_dapo_fullft_2ep; do
    acc=$(grep "val-core/openai/gsm8k/acc/mean" /tmp/${dir}/training.log 2>/dev/null | tail -1 | grep -oP 'np.float64\(\K[0-9.]+')
    echo "  ${dir}: ${acc:-N/A}" | tee -a ${LOG_DIR}/master_v3.log
done
