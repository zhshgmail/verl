#!/usr/bin/env python3
"""
E14c checkpoint evaluation - WITHOUT merging LoRA weights.

During training:
- Base model weights are quantized (NVFP4)
- LoRA weights (lora_A, lora_B) are EXCLUDED from quantization (BF16)
- LoRA is applied during forward pass on top of quantized base

This script replicates that behavior:
1. Load base model
2. Load LoRA adapter (NO merge!)
3. Apply NVFP4 quantization hooks (excluding lora_A, lora_B, base_layer)
4. Run inference

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 eval_e14c_unmerged.py [--nvfp4] [--n_samples 50]
"""

import argparse
import sys
from pathlib import Path

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from verl.utils.reward_score import default_compute_score

# Hardcoded paths for E14c
BASE_MODEL = "/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
LORA_ADAPTER = "/tmp/nvfp4_w4a4_e14c_sigma_decay/checkpoints/global_step_29/actor/lora_adapter"
DATA_PATH = "/data/z00637938/gsm8k/test.parquet"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvfp4", action="store_true", help="Apply NVFP4 W4A4 quantization")
    parser.add_argument("--n_samples", type=int, default=None, help="Number of samples (default: all)")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index")
    args = parser.parse_args()

    print("=" * 60)
    print("E14c EVALUATION (UNMERGED LoRA)")
    print("=" * 60)
    print(f"Base model: {BASE_MODEL}")
    print(f"LoRA adapter: {LORA_ADAPTER}")
    print(f"NVFP4: {args.nvfp4}")
    print()
    print("IMPORTANT: LoRA weights are NOT merged!")
    print("  - NVFP4 applied to base model (excludes lora_A, lora_B, base_layer)")
    print("  - LoRA applied during forward pass (BF16)")
    print()

    # Load data
    print("[1/5] Loading test data...")
    df = pd.read_parquet(DATA_PATH)
    if args.n_samples:
        df = df.iloc[args.start_idx:args.start_idx + args.n_samples]
    print(f"      Loaded {len(df)} samples")

    # Load tokenizer
    print("[2/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Load base model
    print("[3/5] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )

    # Load LoRA adapter WITHOUT merging
    print("[3/5] Loading LoRA adapter (NOT merging)...")
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
    model.eval()
    print("      LoRA loaded (weights separate from base)")

    # Apply NVFP4 to base model (excluding LoRA modules)
    injector = None
    if args.nvfp4:
        print("[4/5] Applying NVFP4 W4A4 (excluding LoRA)...")
        from verl.utils.hw_error_injection import HWErrorConfig, HWErrorInjector

        # Same exclude list as training
        config = HWErrorConfig(
            enabled=True,
            error_type="nvfp4",
            injection_point="both",  # W4A4
            target_modules=["linear"],
            exclude_modules=["lm_head", "embed_tokens", "lora_A", "lora_B", "layers.0", "layers.27", "base_layer"],
            use_ste=False,
        )
        injector = HWErrorInjector(config)
        num_hooks = injector.register_hooks(model, verbose=True)
        print(f"      Registered {num_hooks} NVFP4 hooks")
    else:
        print("[4/5] Skipping NVFP4 (BF16 mode)")

    # Evaluate
    print("[5/5] Starting evaluation...")
    print()

    correct = 0
    total = 0

    for idx, row in df.iterrows():
        prompt = row.get("prompt", row.get("question", ""))
        ground_truth = row.get("reward_model", {})

        if isinstance(ground_truth, dict):
            ground_truth = ground_truth.get("ground_truth", "")
        else:
            ground_truth = row.get("answer", "")

        # Handle prompt format
        if isinstance(prompt, (list, tuple)) or hasattr(prompt, 'tolist'):
            if hasattr(prompt, 'tolist'):
                prompt = prompt.tolist()
            messages = list(prompt)
        else:
            messages = [{"role": "user", "content": str(prompt)}]

        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Generate
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Score
        score = default_compute_score(
            data_source="openai/gsm8k",
            solution_str=response,
            ground_truth=ground_truth,
        )

        is_correct = score is not None and score > 0
        if is_correct:
            correct += 1
        total += 1

        # Progress every 50 samples
        if total % 50 == 0 or total == len(df):
            acc = correct / total * 100
            print(f"Progress: {total}/{len(df)} samples, {correct}/{total} correct ({acc:.2f}%)")

    # Final result
    accuracy = correct / total * 100 if total > 0 else 0

    print()
    print("=" * 60)
    mode_str = "NVFP4 W4A4 (unmerged LoRA)" if args.nvfp4 else "BF16 (unmerged LoRA)"
    print(f"FINAL RESULT ({mode_str}):")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Correct: {correct}/{total}")
    print("=" * 60)
    print()
    print("COMPARISON:")
    print("  E13j_v4 (MXFP4): Step 20=70.28%, Step 29=66.11% (-4.17%)")
    print("  E14c (NVFP4): Step 20=70.74%, Step 29=[SEE ABOVE]")
    print()
    if args.nvfp4:
        if accuracy < 70.74:
            print("VERDICT: E14c ALSO DEGRADES -> This is a BUG in our code")
        else:
            print("VERDICT: E14c does NOT degrade -> Issue is MXFP4-specific")


if __name__ == "__main__":
    main()
