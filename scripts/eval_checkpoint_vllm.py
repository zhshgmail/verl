#!/usr/bin/env python3
"""
Fast evaluation using vLLM for batched inference.
This matches the speed of validation during training.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from verl.utils.reward_score import default_compute_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model (can be merged checkpoint or base+LoRA)")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size for vLLM inference")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--tensor_parallel_size", type=int, default=8,
                       help="Number of GPUs for tensor parallelism")
    args = parser.parse_args()

    print(f"Loading data from {args.data_path}...")
    df = pd.read_parquet(args.data_path)

    if args.n_samples and args.n_samples > 0:
        df = df.head(args.n_samples)

    print(f"Initializing vLLM with {args.tensor_parallel_size} GPUs...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        skip_special_tokens=True,
    )

    # Prepare prompts
    print("Preparing prompts...")
    prompts = []
    ground_truths = []

    for _, row in df.iterrows():
        prompt = row.get("prompt", row.get("question", ""))
        ground_truth = row.get("reward_model", {})

        if isinstance(ground_truth, dict):
            ground_truth = ground_truth.get("ground_truth", "")
        else:
            ground_truth = row.get("answer", "")

        # Handle prompt format
        if isinstance(prompt, (list, tuple)) or (hasattr(prompt, 'tolist') and callable(prompt.tolist)):
            if hasattr(prompt, 'tolist'):
                prompt = prompt.tolist()
            messages = list(prompt)
        else:
            messages = [{"role": "user", "content": str(prompt)}]

        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(chat_prompt)
        ground_truths.append(ground_truth)

    print(f"Running batched inference on {len(prompts)} samples...")

    # Process in batches
    all_outputs = []
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Inference"):
        batch_prompts = prompts[i:i+args.batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)
        all_outputs.extend(outputs)

    # Score results
    print("Scoring results...")
    correct = 0
    total = 0

    for output, ground_truth in tqdm(zip(all_outputs, ground_truths), total=len(all_outputs), desc="Scoring"):
        response = output.outputs[0].text

        score = default_compute_score(
            data_source="openai/gsm8k",
            solution_str=response,
            ground_truth=ground_truth,
        )

        if score is not None and score > 0:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0

    print(f"\n{'='*50}")
    print(f"Results:")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  Correct: {correct}/{total}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
