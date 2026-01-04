#!/usr/bin/env python3
"""
Robustness evaluation using native PyTorch/HuggingFace (NOT vLLM).
This ensures noisy_ops is properly applied during inference.

Usage for 7B (E7c):
    python scripts/robustness_eval_native.py \
        --checkpoint_base /data/z00637938/verl_checkpoints/noisy_ops_aqn_7b_test/noisy_ops_aqn_7b_5e-2 \
        --tokenizer /data/g30067331/Qwen2.5-7B-Instruct \
        --steps 232 \
        --n_samples 200
"""

import os
import sys
import torch
import re
import json
import argparse
import pandas as pd
from pathlib import Path

# Import noisy_ops BEFORE transformers to ensure monkey-patching works
from verl.utils.noisy_ops import enable_noisy_ops, disable_noisy_ops, get_injection_stats


def run_evaluation(model_path, tokenizer_path, val_data_path, noise_scale, n_samples=200):
    """Run evaluation on GSM8K with given noise level using native PyTorch."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*60}")
    print(f"Evaluation with noise={noise_scale*100:.0f}%")
    print(f"Model: {model_path}")
    print(f"{'='*60}")

    # Setup noise injection
    if noise_scale > 0:
        enable_noisy_ops(error_scale=noise_scale, error_type='relative_gaussian')
        print(f"[NoisyOps] Enabled with scale={noise_scale}")
    else:
        disable_noisy_ops()
        print(f"[NoisyOps] Disabled (clean inference)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        padding_side='left'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded. Device: {next(model.parameters()).device}")

    # Load data
    df = pd.read_parquet(val_data_path)
    df = df.head(n_samples)
    print(f"Loaded {len(df)} samples")

    # Prepare prompts and answers
    prompts = []
    answers = []

    for idx, row in df.iterrows():
        prompt_data = row['prompt']
        if hasattr(prompt_data, 'tolist'):
            prompt_data = prompt_data.tolist()

        if isinstance(prompt_data, list) and len(prompt_data) > 0:
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

        # Get ground truth
        reward_model = row.get('reward_model', {})
        if isinstance(reward_model, dict):
            gt = str(reward_model.get('ground_truth', ''))
        else:
            gt = ''
        answers.append(gt)

    # Generate responses
    print(f"Generating responses for {len(prompts)} prompts...", flush=True)
    correct = 0

    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,  # greedy decoding
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Decode
            generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            # Extract answer (last number)
            numbers = re.findall(r'[-+]?\d+(?:\.\d+)?', generated.replace(',', ''))
            pred = numbers[-1] if numbers else ''
            gt = answers[i]

            # Check correctness
            try:
                if abs(float(pred) - float(gt)) < 0.01:
                    correct += 1
            except:
                if pred.strip() == gt.strip():
                    correct += 1

            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(prompts)}, Current accuracy: {correct/(i+1)*100:.1f}%", flush=True)

    accuracy = correct / len(prompts) * 100

    # Get injection stats
    stats = get_injection_stats()
    print(f"\n[NoisyOps] Injection stats: {stats}")

    print(f"\n{'='*50}")
    print(f"RESULT: accuracy={accuracy:.2f}% ({correct}/{len(prompts)})")
    print(f"{'='*50}\n")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    # Disable noisy ops for next run
    if noise_scale > 0:
        disable_noisy_ops()

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Robustness evaluation using native PyTorch")
    parser.add_argument("--checkpoint_base", type=str, required=True,
                       help="Base path to checkpoint directory")
    parser.add_argument("--tokenizer", type=str, required=True,
                       help="Path to tokenizer/base model")
    parser.add_argument("--val_data", type=str,
                       default="/data/z00637938/gsm8k/test.parquet",
                       help="Path to validation data parquet file")
    parser.add_argument("--n_samples", type=int, default=200,
                       help="Number of samples to evaluate (default: 200)")
    parser.add_argument("--steps", type=int, nargs="+", default=[232],
                       help="Checkpoint steps to evaluate")
    args = parser.parse_args()

    results = {}

    for step in args.steps:
        model_path = f"{args.checkpoint_base}/global_step_{step}/merged_hf"

        if not Path(model_path).exists():
            print(f"Checkpoint not found: {model_path}")
            continue

        results[step] = {}

        for noise in [0, 0.05, 0.1]:
            noise_label = f"{int(noise*100)}%"
            print(f"\n{'#'*70}")
            print(f"# Testing Step {step} @ {noise_label} noise")
            print(f"{'#'*70}")

            try:
                acc = run_evaluation(
                    model_path=model_path,
                    tokenizer_path=args.tokenizer,
                    val_data_path=args.val_data,
                    noise_scale=noise,
                    n_samples=args.n_samples,
                )
                results[step][noise_label] = acc
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                results[step][noise_label] = None

    # Print summary
    print("\n" + "=" * 70)
    print("ROBUSTNESS TEST SUMMARY (Native PyTorch with NoisyOps)")
    print("=" * 70)
    print(f"\n{'Checkpoint':<20} {'0% Noise':<15} {'5% Noise':<15} {'10% Noise':<15}")
    print("-" * 65)

    for step in args.steps:
        if step not in results:
            continue
        row = f"Step {step}"
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
    for step in args.steps:
        if step not in results:
            continue
        clean = results.get(step, {}).get("0%")
        noise5 = results.get(step, {}).get("5%")
        noise10 = results.get(step, {}).get("10%")

        if clean is not None and noise5 is not None and noise10 is not None:
            deg5 = clean - noise5
            deg10 = clean - noise10
            print(f"  Step {step}: 5% noise = {-deg5:+.2f}%, 10% noise = {-deg10:+.2f}%")

    print("\n" + "=" * 70)

    # Save results
    results_file = Path(args.checkpoint_base) / "robustness_results_native.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
