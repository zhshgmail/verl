#!/bin/bash
# Full Robustness Testing Script for E5b Checkpoints
#
# This script:
# 1. Converts FSDP checkpoints to HuggingFace format using verl.model_merger
# 2. Runs evaluation at different noise levels (0%, 5%, 10%)
# 3. Compares robustness between epoch 1 and epoch 2 checkpoints
#
# Usage: bash scripts/full_robustness_test.sh

set -x

# Configuration
CHECKPOINT_BASE="/home/dpsk_a2a/DeepEP/checkpoints/noisy_ops_aqn_epoch_aware_test/noisy_ops_aqn_epoch_aware_ckpt_5e-2"
MODEL_BASE="/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
VAL_DATA="/data/z00637938/gsm8k/test.parquet"
OUTPUT_DIR="/tmp/robustness_test_results"

# Disable WandB
export WANDB_MODE=offline

mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "ROBUSTNESS TESTING FOR E5b CHECKPOINTS"
echo "=========================================="
echo ""

# Step 1: Merge FSDP checkpoints to HuggingFace format
echo "### STEP 1: Converting FSDP checkpoints to HuggingFace format ###"

for STEP in 58 116; do
    CKPT_PATH="${CHECKPOINT_BASE}/global_step_${STEP}/actor"
    MERGED_PATH="${CHECKPOINT_BASE}/global_step_${STEP}/merged_hf"

    if [ ! -f "${MERGED_PATH}/model.safetensors" ] && [ ! -f "${MERGED_PATH}/pytorch_model.bin" ]; then
        echo ""
        echo "Merging checkpoint: global_step_${STEP}"
        python -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "${CKPT_PATH}" \
            --target_dir "${MERGED_PATH}"
    else
        echo "Checkpoint global_step_${STEP} already merged, skipping..."
    fi
done

# Step 2: Run evaluation at different noise levels
echo ""
echo "### STEP 2: Running evaluations at different noise levels ###"

# Function to run evaluation
run_eval() {
    local STEP=$1
    local NOISE_SCALE=$2
    local NOISE_LABEL=$3

    local MERGED_PATH="${CHECKPOINT_BASE}/global_step_${STEP}/merged_hf"
    local RESULT_FILE="${OUTPUT_DIR}/step${STEP}_noise${NOISE_LABEL}.txt"

    echo ""
    echo "=== Testing step ${STEP} @ ${NOISE_LABEL} noise ==="

    # Set noise environment
    if [ "$NOISE_SCALE" = "0" ]; then
        export VERL_NOISY_OPS_ENABLED=0
    else
        export VERL_NOISY_OPS_ENABLED=1
        export VERL_NOISY_OPS_SCALE=${NOISE_SCALE}
        export VERL_NOISY_OPS_TYPE=relative_gaussian
    fi

    # Run evaluation with vLLM
    python3 << EOF | tee "${RESULT_FILE}"
import os
import torch
import re
import pandas as pd
from vllm import LLM, SamplingParams

model_path = "${MERGED_PATH}"
tokenizer_path = "${MODEL_BASE}"
val_data_path = "${VAL_DATA}"
noise_scale = ${NOISE_SCALE}
step = ${STEP}

print(f"\\n=== Evaluation: step {step}, noise {noise_scale*100:.0f}% ===")
print(f"Model: {model_path}")

# Load data
df = pd.read_parquet(val_data_path)
n_samples = 200
df = df.head(n_samples)
print(f"Loaded {len(df)} samples")

# Initialize vLLM
llm = LLM(
    model=model_path,
    tokenizer=tokenizer_path,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.6,
    trust_remote_code=True,
    dtype="bfloat16",
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=512,
)

# Prepare prompts and answers
prompts = []
answers = []

for idx, row in df.iterrows():
    if 'conversations' in df.columns:
        convs = row['conversations']
        if isinstance(convs, list) and len(convs) > 0:
            prompt = convs[0].get('content', '')
        else:
            prompt = str(convs)
    else:
        prompt = row.get('question', row.get('prompt', ''))
    prompts.append(prompt)

    if 'reward_model' in df.columns and isinstance(row.get('reward_model'), dict):
        gt = str(row['reward_model'].get('ground_truth', ''))
    else:
        gt = str(row.get('answer', ''))
    answers.append(gt)

# Generate
print("Generating responses...")
outputs = llm.generate(prompts, sampling_params)

# Score
correct = 0
for i, output in enumerate(outputs):
    text = output.outputs[0].text
    numbers = re.findall(r'[-+]?\d+(?:\\.\\d+)?', text.replace(',', ''))
    pred = numbers[-1] if numbers else ''
    gt = answers[i]

    try:
        if abs(float(pred) - float(gt)) < 0.01:
            correct += 1
    except:
        if pred.strip() == gt.strip():
            correct += 1

accuracy = correct / len(outputs) * 100

print(f"\\n{'='*50}")
print(f"RESULT: step={step}, noise={noise_scale*100:.0f}%, accuracy={accuracy:.2f}%")
print(f"{'='*50}\\n")

# Cleanup
del llm
torch.cuda.empty_cache()
EOF
}

# Test both checkpoints at all noise levels
echo ""
echo "### Testing Step 58 (End of Epoch 1) ###"
run_eval 58 0 "0pct"
run_eval 58 0.05 "5pct"
run_eval 58 0.1 "10pct"

echo ""
echo "### Testing Step 116 (End of Epoch 2) ###"
run_eval 116 0 "0pct"
run_eval 116 0.05 "5pct"
run_eval 116 0.1 "10pct"

# Step 3: Generate summary report
echo ""
echo "### STEP 3: Generating Summary Report ###"

python3 << EOF
import os

output_dir = "${OUTPUT_DIR}"
results = {}

for step in [58, 116]:
    results[step] = {}
    for noise_label in ["0pct", "5pct", "10pct"]:
        result_file = f"{output_dir}/step{step}_noise{noise_label}.txt"
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                content = f.read()
                # Extract accuracy from result
                import re
                match = re.search(r'accuracy=(\d+\.\d+)%', content)
                if match:
                    results[step][noise_label] = float(match.group(1))

print("\\n" + "=" * 70)
print("ROBUSTNESS TEST SUMMARY")
print("=" * 70)
print(f"\\n{'Checkpoint':<20} {'0% Noise':<15} {'5% Noise':<15} {'10% Noise':<15}")
print("-" * 65)

for step in [58, 116]:
    epoch = 1 if step == 58 else 2
    row = f"Step {step} (Epoch {epoch})"
    row = f"{row:<20}"
    for noise in ["0pct", "5pct", "10pct"]:
        acc = results.get(step, {}).get(noise, "N/A")
        if isinstance(acc, float):
            row += f"{acc:.2f}%{'':<9}"
        else:
            row += f"{acc:<15}"
    print(row)

print("-" * 65)

# Calculate degradation
if all(noise in results.get(58, {}) and noise in results.get(116, {}) for noise in ["0pct", "5pct", "10pct"]):
    print(f"\\n{'Degradation Analysis:'}")
    for step in [58, 116]:
        epoch = 1 if step == 58 else 2
        clean = results[step].get("0pct", 0)
        noise5 = results[step].get("5pct", 0)
        noise10 = results[step].get("10pct", 0)
        deg5 = clean - noise5
        deg10 = clean - noise10
        print(f"  Step {step} (Epoch {epoch}): 5% noise degradation = {deg5:.2f}%, 10% noise degradation = {deg10:.2f}%")

print("\\n" + "=" * 70)
EOF

echo ""
echo "=========================================="
echo "ROBUSTNESS TESTING COMPLETE"
echo "=========================================="
echo "Results saved in: ${OUTPUT_DIR}"
