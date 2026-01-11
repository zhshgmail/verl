#!/bin/bash
# Comprehensive Robustness Testing for E5b Checkpoints
#
# Tests both checkpoints (step 58 and step 116) at multiple noise levels
#
# Usage: bash scripts/run_robustness_tests.sh

set -x

CHECKPOINT_BASE="/home/dpsk_a2a/DeepEP/checkpoints/noisy_ops_aqn_epoch_aware_test/noisy_ops_aqn_epoch_aware_ckpt_5e-2"
VAL_DATA="/data/z00637938/gsm8k/test.parquet"
MODEL_BASE="/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

# Disable WandB
export WANDB_MODE=offline

# Function to run evaluation
run_eval() {
    local CHECKPOINT=$1
    local NOISE_SCALE=$2
    local LABEL=$3

    echo ""
    echo "=============================================="
    echo "Testing: $LABEL"
    echo "Checkpoint: $CHECKPOINT"
    echo "Noise scale: $NOISE_SCALE"
    echo "=============================================="

    if [ "$NOISE_SCALE" = "0" ]; then
        export VERL_NOISY_OPS_ENABLED=0
        unset VERL_NOISY_OPS_SCALE
    else
        export VERL_NOISY_OPS_ENABLED=1
        export VERL_NOISY_OPS_SCALE=$NOISE_SCALE
        export VERL_NOISY_OPS_TYPE=relative_gaussian
    fi

    # Run quick evaluation using vLLM generate
    python3 << EOF
import os
import torch
import json
from pathlib import Path

checkpoint = "$CHECKPOINT"
noise_scale = "$NOISE_SCALE"
val_data = "$VAL_DATA"
model_base = "$MODEL_BASE"

print(f"\\n=== Robustness Test: {checkpoint.split('/')[-1]} @ noise={noise_scale} ===")

# Check checkpoint contents
ckpt_path = Path(checkpoint) / "actor"
if ckpt_path.exists():
    files = list(ckpt_path.glob("*.pt"))
    print(f"Found {len(files)} checkpoint files in {ckpt_path}")

    # Check for huggingface format
    hf_path = ckpt_path / "huggingface"
    if hf_path.exists():
        print(f"HuggingFace format available at: {hf_path}")

        # Try to load and evaluate
        try:
            from vllm import LLM, SamplingParams
            import pandas as pd
            import re

            # Load test data
            df = pd.read_parquet(val_data)
            n_samples = 200  # Quick test with 200 samples
            df = df.head(n_samples)

            print(f"Loaded {len(df)} test samples")

            # Initialize vLLM with HuggingFace checkpoint
            llm = LLM(
                model=str(hf_path),
                tokenizer=model_base,  # Use base tokenizer
                tensor_parallel_size=1,
                gpu_memory_utilization=0.5,
                trust_remote_code=True,
                dtype="bfloat16",
            )

            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=512,
            )

            # Prepare prompts (extract from data format)
            prompts = []
            answers = []
            for idx, row in df.iterrows():
                # Assuming verl parquet format
                if 'data_source' in df.columns:
                    # Extract prompt from conversations
                    data = row.get('conversations', row.get('prompt', ''))
                    if isinstance(data, list):
                        prompt = data[0].get('content', '') if data else ''
                    else:
                        prompt = str(data)
                else:
                    prompt = row.get('question', row.get('prompt', ''))
                prompts.append(prompt)

                # Get answer
                ans = row.get('reward_model', {}).get('ground_truth', row.get('answer', ''))
                answers.append(str(ans))

            print(f"Running inference on {len(prompts)} prompts...")
            outputs = llm.generate(prompts, sampling_params)

            # Score outputs
            correct = 0
            for i, output in enumerate(outputs):
                text = output.outputs[0].text
                # Extract last number as answer
                numbers = re.findall(r'[-+]?\\d+(?:\\.\\d+)?', text.replace(',', ''))
                pred = numbers[-1] if numbers else ''
                gt = answers[i]

                # Normalize comparison
                try:
                    if float(pred) == float(gt):
                        correct += 1
                except:
                    if pred.strip() == gt.strip():
                        correct += 1

            accuracy = correct / len(outputs) * 100
            print(f"\\n*** RESULT: {accuracy:.2f}% ({correct}/{len(outputs)}) ***\\n")

        except Exception as e:
            print(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("HuggingFace format not found - need to merge sharded weights first")
else:
    print(f"Checkpoint path not found: {ckpt_path}")

EOF
}

echo "=========================================="
echo "ROBUSTNESS TESTING FOR E5b CHECKPOINTS"
echo "=========================================="

# Test 1: Clean evaluation (no noise)
echo ""
echo "### TEST 1: Clean Evaluation (0% noise) ###"
run_eval "${CHECKPOINT_BASE}/global_step_58" "0" "Epoch1-Step58-Clean"
run_eval "${CHECKPOINT_BASE}/global_step_116" "0" "Epoch2-Step116-Clean"

# Test 2: Training noise level (5%)
echo ""
echo "### TEST 2: Training Noise Level (5% noise) ###"
run_eval "${CHECKPOINT_BASE}/global_step_58" "5e-2" "Epoch1-Step58-5pct"
run_eval "${CHECKPOINT_BASE}/global_step_116" "5e-2" "Epoch2-Step116-5pct"

# Test 3: Stress test (10% noise)
echo ""
echo "### TEST 3: Stress Test (10% noise) ###"
run_eval "${CHECKPOINT_BASE}/global_step_58" "1e-1" "Epoch1-Step58-10pct"
run_eval "${CHECKPOINT_BASE}/global_step_116" "1e-1" "Epoch2-Step116-10pct"

echo ""
echo "=========================================="
echo "ROBUSTNESS TESTING COMPLETE"
echo "=========================================="
