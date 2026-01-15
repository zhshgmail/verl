#!/usr/bin/env python3
"""
Quick evaluation of original model with MXFP4 fake quantization.
This measures baseline accuracy before any training.
"""

import argparse
import sys
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from verl.utils.mxfp4_quant import mxfp4_quantize
from verl.utils.reward_score import default_compute_score


def apply_mxfp4_to_model(model, exclude_modules=None):
    """Apply MXFP4 fake quantization to all linear layers."""
    if exclude_modules is None:
        exclude_modules = ['lm_head', 'embed_tokens']

    count = 0
    for name, module in model.named_modules():
        if any(exc in name for exc in exclude_modules):
            continue
        if isinstance(module, torch.nn.Linear):
            with torch.no_grad():
                module.weight.data = mxfp4_quantize(module.weight.data)
            count += 1
    print(f"[MXFP4] Applied fake quantization to {count} linear layers")
    return model


def evaluate_gsm8k(model, tokenizer, data_path, n_samples=100, max_tokens=512):
    """Evaluate model on GSM8K."""
    df = pd.read_parquet(data_path)

    if n_samples > 0:
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
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--with_mxfp4", action="store_true", help="Apply MXFP4 fake quantization")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.with_mxfp4:
        print("Applying MXFP4 fake quantization...")
        model = apply_mxfp4_to_model(model)

    print(f"Evaluating on {args.n_samples} samples...")
    results = evaluate_gsm8k(model, tokenizer, args.data_path, args.n_samples)

    quant_str = "with MXFP4" if args.with_mxfp4 else "without MXFP4"
    print(f"\n{'='*50}")
    print(f"Results ({quant_str}):")
    print(f"  Accuracy: {results['accuracy']*100:.2f}%")
    print(f"  Correct: {results['correct']}/{results['total']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
