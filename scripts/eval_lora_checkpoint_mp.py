#!/usr/bin/env python3
"""
Evaluate a LoRA checkpoint on GSM8K using multiprocessing across 8 GPUs.
Each process handles 1/8 of the samples on its own GPU.
"""

import argparse
import os
import sys
from pathlib import Path
from multiprocessing import Process, Queue

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from verl.utils.reward_score import default_compute_score


def worker_process(gpu_id, base_model_path, lora_adapter_path, samples, result_queue, max_tokens=512):
    """Worker process that evaluates samples on a specific GPU."""
    # Set this process to use only its assigned GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    print(f"[GPU {gpu_id}] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    print(f"[GPU {gpu_id}] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"[GPU {gpu_id}] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    print(f"[GPU {gpu_id}] Merging LoRA adapter...")
    model = model.merge_and_unload()
    model.eval()

    print(f"[GPU {gpu_id}] Processing {len(samples)} samples...")

    correct = 0
    for idx, row in samples.iterrows():
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

        # Score
        score = default_compute_score(
            data_source="openai/gsm8k",
            solution_str=response,
            ground_truth=ground_truth,
        )

        if score is not None and score > 0:
            correct += 1

    # Send results back to main process
    result_queue.put((gpu_id, correct, len(samples)))
    print(f"[GPU {gpu_id}] Completed: {correct}/{len(samples)} correct")


def main():
    # CRITICAL: Set spawn method for CUDA compatibility with multiprocessing
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--lora_adapter_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--num_gpus", type=int, default=8)
    args = parser.parse_args()

    print(f"Loading data from {args.data_path}...")
    df = pd.read_parquet(args.data_path)

    if args.n_samples and args.n_samples > 0:
        df = df.head(args.n_samples)

    total_samples = len(df)
    samples_per_gpu = (total_samples + args.num_gpus - 1) // args.num_gpus

    print(f"Total samples: {total_samples}")
    print(f"Samples per GPU: ~{samples_per_gpu}")
    print(f"Using {args.num_gpus} GPUs")

    # Split data across GPUs
    processes = []
    result_queue = Queue()

    for gpu_id in range(args.num_gpus):
        start_idx = gpu_id * samples_per_gpu
        end_idx = min(start_idx + samples_per_gpu, total_samples)

        if start_idx >= total_samples:
            break

        gpu_samples = df.iloc[start_idx:end_idx]

        p = Process(
            target=worker_process,
            args=(gpu_id, args.base_model_path, args.lora_adapter_path, gpu_samples, result_queue, args.max_tokens)
        )
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Collect results
    total_correct = 0
    total_evaluated = 0

    while not result_queue.empty():
        gpu_id, correct, count = result_queue.get()
        total_correct += correct
        total_evaluated += count

    accuracy = total_correct / total_evaluated if total_evaluated > 0 else 0

    print(f"\n{'='*50}")
    print(f"Results:")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  Correct: {total_correct}/{total_evaluated}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
