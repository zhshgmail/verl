#!/usr/bin/env python3
"""
Evaluate a LoRA checkpoint on GSM8K test set with MXFP4 applied ONLY to base model.

This script does NOT merge the LoRA adapter, ensuring MXFP4 is applied only to
base model weights (as during training), while LoRA computation stays in BF16.

During training, the forward pass is:
  output = MXFP4(base_weight) @ MXFP4(input) + lora_B @ lora_A @ input

If we merge LoRA first and then quantize, we get a different result:
  merged = base_weight + scale * (lora_B @ lora_A)
  output = MXFP4(merged) @ MXFP4(input)  # WRONG - quantizes LoRA contribution!

This script keeps LoRA separate and applies MXFP4 only to base model layers.
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
from verl.utils.hw_error_injection import HWErrorConfig, HWErrorInjector


@ray.remote(num_gpus=1)
class ModelWorker:
    def __init__(self, base_model_path, lora_adapter_path, worker_id, mxfp4=False, mxfp4_injection_point="both", mxfp4_exclude_modules=None):
        self.worker_id = worker_id
        self.mxfp4 = mxfp4
        print(f"[Worker {worker_id}] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

        print(f"[Worker {worker_id}] Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        )

        print(f"[Worker {worker_id}] Loading LoRA adapter (NO MERGE - kept separate)...")
        self.model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        # DO NOT merge - keep LoRA computation in BF16
        self.model.eval()

        # Apply MXFP4 AFTER loading LoRA, excluding lora_A and lora_B
        # This ensures only base model weights are quantized, not LoRA
        self.injector = None
        if mxfp4:
            if mxfp4_exclude_modules is None:
                mxfp4_exclude_modules = ["lm_head", "embed_tokens", "layers.0", "layers.27", "lora_A", "lora_B"]
            else:
                # Always exclude LoRA layers
                if "lora_A" not in mxfp4_exclude_modules:
                    mxfp4_exclude_modules = list(mxfp4_exclude_modules) + ["lora_A", "lora_B"]
            print(f"[Worker {worker_id}] Applying MXFP4 (excluding LoRA layers: {mxfp4_exclude_modules})...")
            config = HWErrorConfig(
                enabled=True,
                error_type="mxfp4",
                injection_point=mxfp4_injection_point,
                target_modules=["linear"],
                exclude_modules=mxfp4_exclude_modules,
                use_ste=False,
            )
            self.injector = HWErrorInjector(config)
            num_hooks = self.injector.register_hooks(self.model, verbose=(worker_id == 0))
            print(f"[Worker {worker_id}] Registered {num_hooks} MXFP4 hooks (LoRA layers excluded)")

        quant_str = " (MXFP4 W4A4 on base only)" if mxfp4 else " (BF16)"
        print(f"[Worker {worker_id}] Ready!{quant_str}")

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
    # MXFP4 fake quantization options
    parser.add_argument("--mxfp4", action="store_true",
                       help="Apply MXFP4 W4A4 fake quantization to BASE MODEL ONLY")
    parser.add_argument("--mxfp4_injection_point", type=str, default="both",
                       choices=["weight", "input", "both"],
                       help="MXFP4 injection point: weight (W4A16), input (W16A4), both (W4A4)")
    parser.add_argument("--mxfp4_exclude_modules", type=str, nargs="+",
                       default=["lm_head", "embed_tokens", "layers.0", "layers.27"],
                       help="Modules to exclude from MXFP4 quantization")
    args = parser.parse_args()

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    print(f"Loading data from {args.data_path}...")
    df = pd.read_parquet(args.data_path)

    if args.n_samples and args.n_samples > 0:
        df = df.head(args.n_samples)

    quant_str = " (with MXFP4 W4A4 on BASE MODEL ONLY)" if args.mxfp4 else " (BF16, no quantization)"
    print(f"Initializing {args.num_workers} GPU workers{quant_str}...")
    print("NOTE: LoRA adapter is NOT merged - kept separate for proper evaluation")
    workers = [
        ModelWorker.remote(args.base_model_path, args.lora_adapter_path, i, args.mxfp4, args.mxfp4_injection_point, args.mxfp4_exclude_modules)
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
    print(f"Results{quant_str}:")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  Correct: {correct}/{total}")
    if args.mxfp4:
        print(f"  NOTE: MXFP4 applied to base model only, LoRA kept in BF16")
    print(f"{'='*50}")

    ray.shutdown()


if __name__ == "__main__":
    main()
