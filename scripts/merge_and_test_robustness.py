#!/usr/bin/env python3
"""
Merge FSDP sharded checkpoints and run robustness testing.

This script:
1. Merges sharded FSDP weights into a single HuggingFace model
2. Runs evaluation at different noise levels (0%, 5%, 10%)
3. Compares robustness between checkpoints

Usage:
    python scripts/merge_and_test_robustness.py
"""

import os
import sys
import torch
import json
from pathlib import Path
from collections import OrderedDict
import re

# Configuration
CHECKPOINT_BASE = "/home/dpsk_a2a/DeepEP/checkpoints/noisy_ops_aqn_epoch_aware_test/noisy_ops_aqn_epoch_aware_ckpt_5e-2"
MODEL_BASE = "/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
VAL_DATA = "/data/z00637938/gsm8k/test.parquet"
CHECKPOINTS = ["global_step_58", "global_step_116"]
NOISE_LEVELS = [0, 0.05, 0.1]  # 0%, 5%, 10%


def merge_fsdp_checkpoint(checkpoint_path, output_path):
    """Merge FSDP sharded weights into a single state dict."""
    actor_path = Path(checkpoint_path) / "actor"

    # Find all model shard files
    shard_files = sorted(actor_path.glob("model_world_size_*_rank_*.pt"))
    if not shard_files:
        print(f"No shard files found in {actor_path}")
        return None

    print(f"Found {len(shard_files)} model shards")

    # Merge shards
    merged_state_dict = OrderedDict()

    for shard_file in shard_files:
        print(f"Loading {shard_file.name}...")
        shard = torch.load(shard_file, map_location="cpu")

        # FSDP shards have flat tensors, need to reconstruct
        for key, value in shard.items():
            if key in merged_state_dict:
                # Concatenate along the sharded dimension
                merged_state_dict[key] = torch.cat([merged_state_dict[key], value], dim=0)
            else:
                merged_state_dict[key] = value

    # Save merged weights
    output_file = Path(output_path) / "pytorch_model.bin"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_state_dict, output_file)
    print(f"Saved merged model to {output_file}")

    # Copy config files
    hf_config_path = actor_path / "huggingface"
    if hf_config_path.exists():
        import shutil
        for f in hf_config_path.glob("*"):
            shutil.copy(f, output_path)
        print(f"Copied config files from {hf_config_path}")

    return output_path


def evaluate_model(model_path, tokenizer_path, val_data_path, noise_scale, n_samples=200):
    """Evaluate model on GSM8K with given noise level."""
    import pandas as pd
    from vllm import LLM, SamplingParams

    # Set noise environment
    if noise_scale > 0:
        os.environ["VERL_NOISY_OPS_ENABLED"] = "1"
        os.environ["VERL_NOISY_OPS_SCALE"] = str(noise_scale)
        os.environ["VERL_NOISY_OPS_TYPE"] = "relative_gaussian"
    else:
        os.environ["VERL_NOISY_OPS_ENABLED"] = "0"

    print(f"\n=== Evaluating with noise={noise_scale*100:.0f}% ===")

    # Load data
    df = pd.read_parquet(val_data_path)
    df = df.head(n_samples)
    print(f"Loaded {len(df)} samples")

    # Initialize vLLM
    try:
        llm = LLM(
            model=model_path,
            tokenizer=tokenizer_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
            trust_remote_code=True,
            dtype="bfloat16",
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
    )

    # Prepare prompts and answers
    prompts = []
    answers = []

    for idx, row in df.iterrows():
        # Handle verl data format
        if 'conversations' in df.columns:
            convs = row['conversations']
            if isinstance(convs, list) and len(convs) > 0:
                prompt = convs[0].get('content', '')
            else:
                prompt = str(convs)
        else:
            prompt = row.get('question', row.get('prompt', ''))

        prompts.append(prompt)

        # Get ground truth answer
        if 'reward_model' in df.columns and isinstance(row.get('reward_model'), dict):
            gt = row['reward_model'].get('ground_truth', '')
        else:
            gt = row.get('answer', '')
        answers.append(str(gt))

    # Generate
    print(f"Generating responses...")
    outputs = llm.generate(prompts, sampling_params)

    # Score
    correct = 0
    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        # Extract last number
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
    print(f"Result: {accuracy:.2f}% ({correct}/{len(outputs)})")

    # Cleanup
    del llm
    torch.cuda.empty_cache()

    return accuracy


def main():
    print("=" * 60)
    print("ROBUSTNESS TESTING FOR E5b CHECKPOINTS")
    print("=" * 60)

    results = {}

    for ckpt_name in CHECKPOINTS:
        ckpt_path = Path(CHECKPOINT_BASE) / ckpt_name
        print(f"\n### Processing checkpoint: {ckpt_name} ###")

        # Check if merged model exists, if not merge
        merged_path = ckpt_path / "merged_hf"
        if not (merged_path / "pytorch_model.bin").exists():
            print("Merging FSDP shards...")
            merge_fsdp_checkpoint(ckpt_path, merged_path)
        else:
            print(f"Using existing merged model at {merged_path}")

        results[ckpt_name] = {}

        # Test at each noise level
        for noise in NOISE_LEVELS:
            label = f"{noise*100:.0f}%"
            print(f"\n--- Testing {ckpt_name} @ {label} noise ---")

            acc = evaluate_model(
                model_path=str(merged_path),
                tokenizer_path=MODEL_BASE,
                val_data_path=VAL_DATA,
                noise_scale=noise,
            )

            if acc is not None:
                results[ckpt_name][label] = acc

    # Print summary
    print("\n" + "=" * 60)
    print("ROBUSTNESS TEST SUMMARY")
    print("=" * 60)
    print(f"\n{'Checkpoint':<20} {'0% Noise':<12} {'5% Noise':<12} {'10% Noise':<12}")
    print("-" * 56)

    for ckpt_name, scores in results.items():
        row = f"{ckpt_name:<20}"
        for noise_label in ["0%", "5%", "10%"]:
            acc = scores.get(noise_label, "N/A")
            if isinstance(acc, float):
                row += f"{acc:.2f}%{'':<6}"
            else:
                row += f"{acc:<12}"
        print(row)

    print("\n" + "=" * 60)

    # Save results
    results_file = Path(CHECKPOINT_BASE) / "robustness_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
