#!/usr/bin/env python3
"""
Evaluate a LoRA checkpoint on GSM8K test set using Ray for parallel inference across multiple GPUs.
This is much faster than sequential evaluation.
"""

import argparse
import sys
from pathlib import Path

import torch
import pandas as pd
import ray
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from verl.utils.reward_score import default_compute_score


@ray.remote(num_gpus=1)
class ModelWorker:
    def __init__(self, base_model_path, lora_adapter_path, worker_id):
        self.worker_id = worker_id
        print(f"[Worker {worker_id}] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

        print(f"[Worker {worker_id}] Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        )

        print(f"[Worker {worker_id}] Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, lora_adapter_path)

        print(f"[Worker {worker_id}] Merging LoRA adapter...")
        self.model = model.merge_and_unload()
        self.model.eval()

        print(f"[Worker {worker_id}] Ready!")

    def evaluate_sample(self, prompt, ground_truth, max_tokens=512):
        # Handle prompt format
        if isinstance(prompt, (list, tuple)) or (hasattr(prompt, 'tolist') and callable(prompt.tolist)):
            if hasattr(prompt, 'tolist'):
                prompt = prompt.tolist()
            messages = list(prompt)
        else:
            messages = [{"role": "user", "content": str(prompt)}]

        chat_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Generate
        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Score
        score = default_compute_score(
            data_source="openai/gsm8k",
            solution_str=response,
            ground_truth=ground_truth,
        )

        return 1 if (score is not None and score > 0) else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--lora_adapter_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8, help="Number of GPU workers")
    args = parser.parse_args()

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    print(f"Loading data from {args.data_path}...")
    df = pd.read_parquet(args.data_path)

    if args.n_samples and args.n_samples > 0:
        df = df.head(args.n_samples)

    print(f"Initializing {args.num_workers} GPU workers...")
    workers = [
        ModelWorker.remote(args.base_model_path, args.lora_adapter_path, i)
        for i in range(args.num_workers)
    ]

    # Wait for all workers to initialize
    ray.get([w.evaluate_sample.remote("test", "test", 10) for w in workers])
    print("All workers ready!")

    print(f"Evaluating on {len(df)} samples...")

    # Distribute work across workers
    futures = []
    for idx, row in df.iterrows():
        prompt = row.get("prompt", row.get("question", ""))
        ground_truth = row.get("reward_model", {})
        if isinstance(ground_truth, dict):
            ground_truth = ground_truth.get("ground_truth", "")
        else:
            ground_truth = row.get("answer", "")

        # Round-robin assignment to workers
        worker_idx = idx % args.num_workers
        future = workers[worker_idx].evaluate_sample.remote(prompt, ground_truth, args.max_tokens)
        futures.append(future)

    # Collect results with progress bar
    correct = 0
    total = 0
    with tqdm(total=len(futures), desc="Evaluating") as pbar:
        while futures:
            # Wait for at least one result
            done_ids, futures = ray.wait(futures, num_returns=min(len(futures), args.num_workers))
            for result_id in done_ids:
                result = ray.get(result_id)
                correct += result
                total += 1
                pbar.update(1)
                pbar.set_postfix({"acc": f"{correct/total*100:.2f}%"})

    accuracy = correct / total if total > 0 else 0

    print(f"\n{'='*50}")
    print(f"Results:")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  Correct: {correct}/{total}")
    print(f"{'='*50}")

    ray.shutdown()


if __name__ == "__main__":
    main()
