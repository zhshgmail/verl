#!/usr/bin/env python3
"""
Evaluate a LoRA checkpoint on GSM8K test set.
This script loads a base model + LoRA adapter and evaluates on GSM8K.

Supports MXFP4 fake quantization for consistent evaluation with W4A4 training.
"""

import argparse
import sys
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from verl.utils.reward_score import default_compute_score
from verl.utils.hw_error_injection import HWErrorConfig, HWErrorInjector


def evaluate_gsm8k(model, tokenizer, data_path, n_samples=None, max_tokens=512):
    """Evaluate model on GSM8K."""
    df = pd.read_parquet(data_path)

    if n_samples and n_samples > 0:
        df = df.head(n_samples)

    correct = 0
    total = 0

    model.eval()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        prompt = row.get("prompt", row.get("question", ""))
        ground_truth = row.get("reward_model", {})
        if isinstance(ground_truth, dict):
            ground_truth = ground_truth.get("ground_truth", "")
        else:
            ground_truth = row.get("answer", "")

        # Handle prompt format - could be list of messages or plain string
        if isinstance(prompt, (list, tuple)) or (hasattr(prompt, 'tolist') and callable(prompt.tolist)):
            # Already in chat format
            if hasattr(prompt, 'tolist'):
                prompt = prompt.tolist()
            messages = list(prompt)
        else:
            # Plain string, wrap in chat format
            messages = [{"role": "user", "content": str(prompt)}]

        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Generate
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Score using verl's GSM8K scorer
        score = default_compute_score(
            data_source="openai/gsm8k",
            solution_str=response,
            ground_truth=ground_truth,
        )

        if score is not None and score > 0:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    return {"accuracy": accuracy, "correct": correct, "total": total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True,
                       help="Path to base model (e.g., E8c checkpoint)")
    parser.add_argument("--lora_adapter_path", type=str, required=True,
                       help="Path to LoRA adapter checkpoint")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to GSM8K test data (parquet file)")
    parser.add_argument("--n_samples", type=int, default=None,
                       help="Number of samples to evaluate (default: all)")
    parser.add_argument("--max_tokens", type=int, default=512,
                       help="Max tokens to generate per sample")
    # MXFP4 fake quantization options
    parser.add_argument("--mxfp4", action="store_true",
                       help="Apply MXFP4 W4A4 fake quantization during evaluation")
    parser.add_argument("--mxfp4_injection_point", type=str, default="both",
                       choices=["weight", "input", "both"],
                       help="MXFP4 injection point: weight (W4A16), input (W16A4), both (W4A4)")
    args = parser.parse_args()

    print(f"Loading base model from {args.base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from {args.lora_adapter_path}...")
    model = PeftModel.from_pretrained(base_model, args.lora_adapter_path)

    print("Merging LoRA adapter with base model...")
    model = model.merge_and_unload()

    # Apply MXFP4 fake quantization if requested
    injector = None
    if args.mxfp4:
        print(f"Applying MXFP4 fake quantization (injection_point={args.mxfp4_injection_point})...")
        config = HWErrorConfig(
            enabled=True,
            error_type="mxfp4",
            injection_point=args.mxfp4_injection_point,
            target_modules=["linear"],
            exclude_modules=["lm_head", "embed_tokens"],
            use_ste=False,  # No STE needed for inference-only
        )
        injector = HWErrorInjector(config)
        num_hooks = injector.register_hooks(model, verbose=True)
        print(f"Registered {num_hooks} MXFP4 quantization hooks")

    n_samples_str = f"{args.n_samples} samples" if args.n_samples else "all samples"
    quant_str = " (with MXFP4 W4A4)" if args.mxfp4 else " (BF16, no quantization)"
    print(f"Evaluating on {n_samples_str}{quant_str}...")
    results = evaluate_gsm8k(model, tokenizer, args.data_path, args.n_samples, args.max_tokens)

    # Clean up hooks
    if injector:
        injector.remove_hooks()

    print(f"\n{'='*50}")
    print(f"Results{quant_str}:")
    print(f"  Accuracy: {results['accuracy']*100:.2f}%")
    print(f"  Correct: {results['correct']}/{results['total']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
