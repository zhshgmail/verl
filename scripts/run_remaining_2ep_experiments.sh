#!/bin/bash
# Run remaining 5 2-epoch experiments sequentially (E6b-2ep already running)
# Date: 2026-01-12

set -x

LOG_DIR="/tmp/2ep_experiments_master"
mkdir -p ${LOG_DIR}

echo "=== Starting Remaining 2-Epoch Experiments ===" | tee ${LOG_DIR}/master.log
echo "Start time: $(date)" | tee -a ${LOG_DIR}/master.log
echo "Note: E6b-2ep already running separately" | tee -a ${LOG_DIR}/master.log
echo "" | tee -a ${LOG_DIR}/master.log

cd /home/z00637938/workspace/verl

# Wait for E6b-2ep to complete (check every 2 minutes)
echo "Waiting for E6b-2ep to complete..." | tee -a ${LOG_DIR}/master.log
while ! grep -q "experiment complete" /tmp/mxfp4_v6.1_dapo_lora_aqn_2ep/training.log 2>/dev/null; do
    sleep 120
    echo "  Still waiting... $(date)" >> ${LOG_DIR}/master.log
done
echo "E6b-2ep completed! $(date)" | tee -a ${LOG_DIR}/master.log
grep "val-core/openai/gsm8k/acc/mean" /tmp/mxfp4_v6.1_dapo_lora_aqn_2ep/training.log | tail -1 >> ${LOG_DIR}/master.log

# 2. E6a-2ep: LoRA_MXFP4
echo "=== [2/6] E6a-2ep: MXFP4 + LoRA ===" | tee -a ${LOG_DIR}/master.log
echo "Start: $(date)" | tee -a ${LOG_DIR}/master.log
bash scripts/test_mxfp4_v6.0_dapo_lora_2ep.sh 8
echo "End: $(date)" | tee -a ${LOG_DIR}/master.log
grep "val-core/openai/gsm8k/acc/mean" /tmp/mxfp4_v6.0_dapo_lora_2ep/training.log | tail -1 >> ${LOG_DIR}/master.log

# 3. E7a-2ep: LoRA_BF16
echo "=== [3/6] E7a-2ep: BF16 + LoRA ===" | tee -a ${LOG_DIR}/master.log
echo "Start: $(date)" | tee -a ${LOG_DIR}/master.log
bash scripts/test_bf16_v7.0_dapo_lora_2ep.sh 8
echo "End: $(date)" | tee -a ${LOG_DIR}/master.log
grep "val-core/openai/gsm8k/acc/mean" /tmp/bf16_v7.0_dapo_lora_2ep/training.log | tail -1 >> ${LOG_DIR}/master.log

# 4. E3a-2ep: Q_MXFP4_fullFT
echo "=== [4/6] E3a-2ep: MXFP4 + Full FT ===" | tee -a ${LOG_DIR}/master.log
echo "Start: $(date)" | tee -a ${LOG_DIR}/master.log
bash scripts/test_mxfp4_v3.0_dapo_2ep.sh 8
echo "End: $(date)" | tee -a ${LOG_DIR}/master.log
grep "val-core/openai/gsm8k/acc/mean" /tmp/mxfp4_v3.0_dapo_2ep/training.log | tail -1 >> ${LOG_DIR}/master.log

# 5. E3b-2ep: Q_MXFP4_fullFT_AQN
echo "=== [5/6] E3b-2ep: MXFP4 + Full FT + AQN ===" | tee -a ${LOG_DIR}/master.log
echo "Start: $(date)" | tee -a ${LOG_DIR}/master.log
bash scripts/test_mxfp4_v3.1_dapo_aqn_2ep.sh 8
echo "End: $(date)" | tee -a ${LOG_DIR}/master.log
grep "val-core/openai/gsm8k/acc/mean" /tmp/mxfp4_v3.1_dapo_aqn_2ep/training.log | tail -1 >> ${LOG_DIR}/master.log

# 6. E8a-2ep: Q_BF16_fullFT
echo "=== [6/6] E8a-2ep: BF16 + Full FT ===" | tee -a ${LOG_DIR}/master.log
echo "Start: $(date)" | tee -a ${LOG_DIR}/master.log
bash scripts/test_bf16_v8.0_dapo_fullft_2ep.sh 8
echo "End: $(date)" | tee -a ${LOG_DIR}/master.log
grep "val-core/openai/gsm8k/acc/mean" /tmp/bf16_v8.0_dapo_fullft_2ep/training.log | tail -1 >> ${LOG_DIR}/master.log

echo "" | tee -a ${LOG_DIR}/master.log
echo "=== ALL 6 EXPERIMENTS COMPLETED ===" | tee -a ${LOG_DIR}/master.log
echo "End time: $(date)" | tee -a ${LOG_DIR}/master.log
