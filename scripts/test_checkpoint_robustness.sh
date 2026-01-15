#!/bin/bash
# Robustness Testing Script for E5b Checkpoints
#
# Tests checkpoints at different noise levels to evaluate robustness:
# - 0% noise (clean evaluation)
# - 5% noise (training noise level)
# - 10% noise (stress test)
#
# Usage: bash scripts/test_checkpoint_robustness.sh <checkpoint_path> <noise_scale>

set -x

# Configuration
CHECKPOINT_PATH=${1:-"/home/dpsk_a2a/DeepEP/checkpoints/noisy_ops_aqn_epoch_aware_test/noisy_ops_aqn_epoch_aware_ckpt_5e-2/global_step_116"}
NOISE_SCALE=${2:-0}  # 0 for clean, 5e-2 for 5%, 1e-1 for 10%
N_GPUS=${3:-8}

# Disable WandB
export WANDB_MODE=offline

# Set noisy ops based on noise scale
if [ "$NOISE_SCALE" = "0" ]; then
    export VERL_NOISY_OPS_ENABLED=0
    echo "=== Running CLEAN evaluation (no noise) ==="
else
    export VERL_NOISY_OPS_ENABLED=1
    export VERL_NOISY_OPS_SCALE=${NOISE_SCALE}
    export VERL_NOISY_OPS_TYPE=relative_gaussian
    echo "=== Running evaluation with ${NOISE_SCALE} noise ==="
fi

# Data paths
VAL_DATA=${VAL_DATA:-"/data/z00637938/gsm8k/test.parquet"}

# Extract model path from checkpoint
ACTOR_PATH="${CHECKPOINT_PATH}/actor"

echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Noise scale: ${NOISE_SCALE}"
echo "Actor path: ${ACTOR_PATH}"

# Run evaluation using vLLM generation + reward scoring
python3 -c "
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import pandas as pd
import re

# Configuration
checkpoint_path = '${CHECKPOINT_PATH}'
actor_path = '${ACTOR_PATH}'
val_data_path = '${VAL_DATA}'
noise_scale = float('${NOISE_SCALE}')

print(f'Loading checkpoint from: {actor_path}')

# Check if HuggingFace format exists
hf_path = os.path.join(actor_path, 'huggingface')
if os.path.exists(hf_path):
    model_path = hf_path
else:
    # Need to merge sharded weights
    print('HuggingFace format not found, using sharded weights...')
    model_path = actor_path

# Load validation data
print(f'Loading validation data from: {val_data_path}')
df = pd.read_parquet(val_data_path)
print(f'Loaded {len(df)} validation samples')

# Use a subset for quick testing
n_samples = min(100, len(df))
df_subset = df.head(n_samples)

# Initialize vLLM
print('Initializing vLLM...')
try:
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        stop=['</s>', '<|im_end|>', '<|endoftext|>'],
    )

    # Prepare prompts
    prompts = []
    for idx, row in df_subset.iterrows():
        if 'prompt' in row:
            prompts.append(row['prompt'])
        elif 'question' in row:
            prompts.append(row['question'])

    print(f'Generating responses for {len(prompts)} prompts...')
    outputs = llm.generate(prompts, sampling_params)

    # Extract answers and compute accuracy
    correct = 0
    total = len(outputs)

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        # Extract numerical answer
        numbers = re.findall(r'[-+]?\d*\.?\d+', generated_text)
        if numbers:
            pred_answer = numbers[-1]
        else:
            pred_answer = ''

        # Get ground truth
        if 'answer' in df_subset.iloc[i]:
            gt_answer = str(df_subset.iloc[i]['answer'])
        elif 'reward' in df_subset.iloc[i]:
            gt_answer = str(df_subset.iloc[i]['reward'])
        else:
            gt_answer = ''

        # Compare
        if pred_answer == gt_answer:
            correct += 1

    accuracy = correct / total * 100
    print(f'\\n=== ROBUSTNESS TEST RESULT ===')
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Noise scale: {noise_scale}')
    print(f'Samples: {total}')
    print(f'Correct: {correct}')
    print(f'Accuracy: {accuracy:.2f}%')
    print(f'==============================\\n')

except Exception as e:
    print(f'Error during evaluation: {e}')
    import traceback
    traceback.print_exc()
"

echo "=== Robustness Test Complete ==="
