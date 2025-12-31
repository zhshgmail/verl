#!/usr/bin/env python3
"""
Clean Evaluation Script for Robustness Testing

This script evaluates a checkpoint WITHOUT noise injection to measure robustness.
Use this after training with noisy_ops to see if the model maintains accuracy
when deployed on clean hardware.

Usage:
    # Evaluate a checkpoint with clean inference (no noise)
    python scripts/clean_eval_checkpoint.py \
        --model_path /path/to/checkpoint \
        --data_path /path/to/gsm8k/test.parquet \
        --n_samples 5 \
        --output_file results.json

    # Compare with noisy inference
    VERL_NOISY_OPS_ENABLED=1 VERL_NOISY_OPS_SCALE=5e-2 python scripts/clean_eval_checkpoint.py ...

Key Features:
    - VERL_NOISY_OPS_ENABLED is explicitly set to 0 by default
    - Uses vLLM for efficient batch inference
    - Computes GSM8K accuracy using verl's reward function
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Explicitly disable noisy ops for clean evaluation (can be overridden by env var)
if "VERL_NOISY_OPS_ENABLED" not in os.environ:
    os.environ["VERL_NOISY_OPS_ENABLED"] = "0"

import pandas as pd
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Add verl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from verl.utils.reward_score import default_compute_score


def load_gsm8k_data(data_path: str) -> list[dict]:
    """Load GSM8K test data from parquet file."""
    df = pd.read_parquet(data_path)
    data = []
    for _, row in df.iterrows():
        item = {
            "prompt": row.get("prompt", row.get("question", "")),
            "ground_truth": row.get("reward_model", {}).get("ground_truth", row.get("answer", "")),
        }
        # Handle nested dict in parquet
        if isinstance(item["ground_truth"], dict):
            item["ground_truth"] = item["ground_truth"].get("ground_truth", "")
        data.append(item)
    return data


def build_prompt(question: str, tokenizer) -> str:
    """Build chat prompt for Qwen model."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Solve the math problem step by step."},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def evaluate_gsm8k(
    model_path: str,
    data_path: str,
    n_samples: int = 5,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.8,
) -> dict:
    """
    Evaluate a model on GSM8K with optional noise injection.

    Returns:
        dict with accuracy, individual scores, and metadata
    """
    # Check noise status
    noisy_enabled = os.environ.get("VERL_NOISY_OPS_ENABLED", "0") == "1"
    noisy_scale = os.environ.get("VERL_NOISY_OPS_SCALE", "0")

    print("=" * 60)
    print("Clean Evaluation Script")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"N samples per question: {n_samples}")
    print(f"Noisy ops enabled: {noisy_enabled}")
    if noisy_enabled:
        print(f"Noisy ops scale: {noisy_scale}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=True,  # Disable torch.compile for compatibility
    )
    tokenizer = llm.get_tokenizer()

    # Load data
    print("Loading data...")
    data = load_gsm8k_data(data_path)
    print(f"Loaded {len(data)} questions")

    # Prepare prompts
    prompts = []
    ground_truths = []
    for item in data:
        prompt = build_prompt(item["prompt"], tokenizer)
        for _ in range(n_samples):
            prompts.append(prompt)
            ground_truths.append(item["ground_truth"])

    # Generate
    print(f"\nGenerating {len(prompts)} responses...")
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
    )
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    # Score
    print("\nScoring responses...")
    scores = []
    for output, gt in tqdm(zip(outputs, ground_truths), total=len(outputs)):
        response = output.outputs[0].text
        try:
            score = default_compute_score(
                data_source="openai/gsm8k",
                solution_str=response,
                ground_truth=gt,
            )
            scores.append(float(score))
        except Exception as e:
            print(f"Scoring error: {e}")
            scores.append(0.0)

    # Aggregate by question (best-of-n)
    question_scores = []
    for i in range(0, len(scores), n_samples):
        question_scores.append(max(scores[i : i + n_samples]))

    accuracy = sum(question_scores) / len(question_scores) * 100
    mean_score = sum(scores) / len(scores) * 100

    results = {
        "model_path": model_path,
        "data_path": data_path,
        "n_questions": len(data),
        "n_samples": n_samples,
        "noisy_ops_enabled": noisy_enabled,
        "noisy_ops_scale": noisy_scale if noisy_enabled else None,
        "accuracy_best_of_n": accuracy,
        "accuracy_mean": mean_score,
        "individual_scores": question_scores,
    }

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Questions: {len(data)}")
    print(f"Samples per question: {n_samples}")
    print(f"Best-of-{n_samples} Accuracy: {accuracy:.2f}%")
    print(f"Mean Accuracy: {mean_score:.2f}%")
    print(f"Noisy ops: {'ENABLED' if noisy_enabled else 'DISABLED'}")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="Clean evaluation for robustness testing")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="Path to GSM8K test parquet")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of samples per question")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="TP size for vLLM")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="GPU memory utilization")
    parser.add_argument("--output_file", type=str, default=None, help="Output JSON file for results")

    args = parser.parse_args()

    results = evaluate_gsm8k(
        model_path=args.model_path,
        data_path=args.data_path,
        n_samples=args.n_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
