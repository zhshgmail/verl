#!/usr/bin/env python3
"""
Simple robustness evaluation using verl's checkpoint loading.

This script evaluates checkpoints at different noise levels.
"""

import os
import sys
import torch
import json
import re
from pathlib import Path

def run_gsm8k_eval(model, tokenizer, val_data_path, n_samples=200):
    """Run GSM8K evaluation and return accuracy."""
    import pandas as pd

    df = pd.read_parquet(val_data_path)
    df = df.head(n_samples)

    correct = 0
    total = len(df)

    model.eval()

    for idx, row in df.iterrows():
        # Get prompt
        if 'conversations' in df.columns:
            convs = row['conversations']
            if isinstance(convs, list) and len(convs) > 0:
                prompt = convs[0].get('content', '')
            else:
                prompt = str(convs)
        else:
            prompt = row.get('question', row.get('prompt', ''))

        # Get ground truth
        if 'reward_model' in df.columns and isinstance(row.get('reward_model'), dict):
            gt = str(row['reward_model'].get('ground_truth', ''))
        else:
            gt = str(row.get('answer', ''))

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Extract answer
        numbers = re.findall(r'[-+]?\d+(?:\.\d+)?', response.replace(',', ''))
        pred = numbers[-1] if numbers else ''

        # Compare
        try:
            if abs(float(pred) - float(gt)) < 0.01:
                correct += 1
        except:
            if pred.strip() == gt.strip():
                correct += 1

        if (idx + 1) % 50 == 0:
            print(f"Progress: {idx+1}/{total}, Current acc: {correct/(idx+1)*100:.2f}%")

    return correct / total * 100


def main():
    import argparse
    from transformers import AutoTokenizer, AutoModelForCausalLM

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--noise_scale", type=float, default=0)
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--model_base", type=str,
                       default="/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    parser.add_argument("--val_data", type=str,
                       default="/data/z00637938/gsm8k/test.parquet")
    args = parser.parse_args()

    # Set noise environment
    if args.noise_scale > 0:
        os.environ["VERL_NOISY_OPS_ENABLED"] = "1"
        os.environ["VERL_NOISY_OPS_SCALE"] = str(args.noise_scale)
        os.environ["VERL_NOISY_OPS_TYPE"] = "relative_gaussian"
        print(f"Noise enabled: {args.noise_scale*100:.0f}%")
    else:
        os.environ["VERL_NOISY_OPS_ENABLED"] = "0"
        print("Noise disabled (clean evaluation)")

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model base: {args.model_base}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_base,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load checkpoint weights (FSDP sharded)
    ckpt_path = Path(args.checkpoint) / "actor"
    shard_files = sorted(ckpt_path.glob("model_world_size_*_rank_*.pt"))

    if shard_files:
        print(f"Loading {len(shard_files)} checkpoint shards...")

        # For single-GPU eval, we need to reconstruct the full state dict
        # This is simplified - proper FSDP requires distributed loading
        full_state_dict = {}

        for shard_file in shard_files:
            shard = torch.load(shard_file, map_location="cpu")
            for key, value in shard.items():
                # Clean up FSDP key names
                clean_key = key.replace("_fsdp_wrapped_module.", "").replace("_flat_param", "")
                if clean_key not in full_state_dict:
                    full_state_dict[clean_key] = value

        # Try to load compatible keys
        model_state = model.state_dict()
        loaded_keys = 0
        for key in model_state:
            if key in full_state_dict:
                try:
                    model_state[key] = full_state_dict[key]
                    loaded_keys += 1
                except:
                    pass

        print(f"Loaded {loaded_keys}/{len(model_state)} keys from checkpoint")
        model.load_state_dict(model_state, strict=False)
    else:
        print("Warning: No checkpoint shards found, using base model")

    model.eval()

    # Run evaluation
    print(f"\nRunning evaluation on {args.n_samples} samples...")
    accuracy = run_gsm8k_eval(model, tokenizer, args.val_data, args.n_samples)

    print(f"\n{'='*50}")
    print(f"RESULT: {accuracy:.2f}%")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Noise: {args.noise_scale*100:.0f}%")
    print(f"{'='*50}")

    return accuracy


if __name__ == "__main__":
    main()
