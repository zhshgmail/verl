#!/usr/bin/env python3
"""
Robustness evaluation for AQN-trained checkpoints.
Tests checkpoints at different noise levels (0%, 5%, 10%).

Usage for 1.5B (E5b):
    python scripts/robustness_eval.py \
        --checkpoint_base /path/to/noisy_ops_aqn_epoch_aware_ckpt_5e-2 \
        --tokenizer /data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/... \
        --steps 58 116

Usage for 7B (E7c):
    python scripts/robustness_eval.py \
        --checkpoint_base /data/z00637938/verl_checkpoints/noisy_ops_aqn_7b_test/noisy_ops_aqn_7b_5e-2 \
        --tokenizer /data/g30067331/Qwen2.5-7B-Instruct \
        --steps 58 232 \
        --tp_size 2
"""

import os
import sys
import torch
import re
import json
import argparse
import pandas as pd
from pathlib import Path

def run_evaluation(model_path, tokenizer_path, val_data_path, noise_scale, n_samples=200, tp_size=1):
    """Run evaluation on GSM8K with given noise level."""
    from vllm import LLM, SamplingParams

    # Set noise environment
    if noise_scale > 0:
        os.environ["VERL_NOISY_OPS_ENABLED"] = "1"
        os.environ["VERL_NOISY_OPS_SCALE"] = str(noise_scale)
        os.environ["VERL_NOISY_OPS_TYPE"] = "relative_gaussian"
    else:
        os.environ["VERL_NOISY_OPS_ENABLED"] = "0"

    print(f"\n=== Evaluation with noise={noise_scale*100:.0f}% ===")
    print(f"Model: {model_path}")
    print(f"Tensor Parallel Size: {tp_size}")

    # Load data
    df = pd.read_parquet(val_data_path)
    df = df.head(n_samples)
    print(f"Loaded {len(df)} samples")

    # Initialize vLLM
    llm = LLM(
        model=model_path,
        tokenizer=tokenizer_path,
        tensor_parallel_size=tp_size,
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
        # Handle the verl data format
        # prompt column is a numpy array of conversation dicts
        prompt_data = row['prompt']
        if hasattr(prompt_data, 'tolist'):
            prompt_data = prompt_data.tolist()

        if isinstance(prompt_data, list) and len(prompt_data) > 0:
            # Get user message content
            user_msg = prompt_data[0]
            if isinstance(user_msg, dict):
                prompt = user_msg.get('content', '')
            else:
                prompt = str(user_msg)
        else:
            prompt = str(prompt_data)

        # Format as chat template
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        prompts.append(formatted_prompt)

        # Get ground truth from reward_model
        reward_model = row.get('reward_model', {})
        if isinstance(reward_model, dict):
            gt = str(reward_model.get('ground_truth', ''))
        else:
            gt = ''
        answers.append(gt)

    # Generate
    print(f"Generating responses for {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)

    # Score
    correct = 0
    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        # Extract last number as answer
        numbers = re.findall(r'[-+]?\d+(?:\.\d+)?', text.replace(',', ''))
        pred = numbers[-1] if numbers else ''
        gt = answers[i]

        try:
            if abs(float(pred) - float(gt)) < 0.01:
                correct += 1
        except:
            if pred.strip() == gt.strip():
                correct += 1

    accuracy = correct / len(outputs) * 100

    print(f"\n{'='*50}")
    print(f"RESULT: accuracy={accuracy:.2f}% ({correct}/{len(outputs)})")
    print(f"{'='*50}\n")

    # Cleanup
    del llm
    torch.cuda.empty_cache()

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Robustness evaluation for AQN-trained checkpoints")
    parser.add_argument("--checkpoint_base", type=str, required=True,
                       help="Base path to checkpoint directory (e.g., /path/to/noisy_ops_aqn_epoch_aware_ckpt_5e-2)")
    parser.add_argument("--tokenizer", type=str, required=True,
                       help="Path to tokenizer/base model (e.g., /data/g30067331/Qwen2.5-7B-Instruct)")
    parser.add_argument("--val_data", type=str,
                       default="/data/z00637938/gsm8k/test.parquet",
                       help="Path to validation data parquet file")
    parser.add_argument("--n_samples", type=int, default=200,
                       help="Number of samples to evaluate (default: 200)")
    parser.add_argument("--steps", type=int, nargs="+", default=[58, 116],
                       help="Checkpoint steps to evaluate (default: 58 116, use '58 232' for 7B)")
    parser.add_argument("--tp_size", type=int, default=1,
                       help="Tensor parallel size (default: 1, use 2 for 7B)")
    args = parser.parse_args()

    results = {}
    steps = args.steps

    # Test all checkpoints at all noise levels
    for step in steps:
        model_path = f"{args.checkpoint_base}/global_step_{step}/merged_hf"
        results[step] = {}

        for noise in [0, 0.05, 0.1]:
            noise_label = f"{int(noise*100)}%"
            print(f"\n{'='*60}")
            print(f"Testing Step {step} @ {noise_label} noise")
            print(f"{'='*60}")

            try:
                acc = run_evaluation(
                    model_path=model_path,
                    tokenizer_path=args.tokenizer,
                    val_data_path=args.val_data,
                    noise_scale=noise,
                    n_samples=args.n_samples,
                    tp_size=args.tp_size,
                )
                results[step][noise_label] = acc
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                results[step][noise_label] = None

    # Print summary
    print("\n" + "=" * 70)
    print("ROBUSTNESS TEST SUMMARY")
    print("=" * 70)
    print(f"\n{'Checkpoint':<20} {'0% Noise':<15} {'5% Noise':<15} {'10% Noise':<15}")
    print("-" * 65)

    for i, step in enumerate(steps):
        epoch = i + 1
        row = f"Step {step} (Epoch {epoch})"
        row = f"{row:<20}"
        for noise in ["0%", "5%", "10%"]:
            acc = results.get(step, {}).get(noise)
            if acc is not None:
                row += f"{acc:.2f}%{'':<9}"
            else:
                row += f"{'N/A':<15}"
        print(row)

    print("-" * 65)

    # Calculate degradation
    print(f"\nDegradation Analysis:")
    for i, step in enumerate(steps):
        epoch = i + 1
        clean = results.get(step, {}).get("0%")
        noise5 = results.get(step, {}).get("5%")
        noise10 = results.get(step, {}).get("10%")

        if clean is not None:
            deg5 = clean - noise5 if noise5 is not None else "N/A"
            deg10 = clean - noise10 if noise10 is not None else "N/A"
            if isinstance(deg5, float):
                print(f"  Step {step} (Epoch {epoch}): 5% noise = -{deg5:.2f}%, 10% noise = -{deg10:.2f}%")

    print("\n" + "=" * 70)

    # Save results
    results_file = Path(args.checkpoint_base) / "robustness_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
