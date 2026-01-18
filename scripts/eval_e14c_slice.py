#!/usr/bin/env python3
"""
Single GPU slice evaluator for E14c checkpoint.
Called by eval_e14c_parallel.sh.
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
    parser.add_argument("--start_idx", type=int, required=True)
    parser.add_argument("--n_samples", type=int, required=True)
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--nvfp4", action="store_true", help="Apply NVFP4 W4A4 quantization")
    args = parser.parse_args()

    print(f"[GPU {args.gpu_id}] Starting...")
    print(f"[GPU {args.gpu_id}] Samples: {args.start_idx} to {args.start_idx + args.n_samples - 1}")
    print(f"[GPU {args.gpu_id}] NVFP4: {args.nvfp4}")

    # Load data slice
    print(f"[GPU {args.gpu_id}] Loading data...")
    df = pd.read_parquet(DATA_PATH)
    df_slice = df.iloc[args.start_idx:args.start_idx + args.n_samples]
    print(f"[GPU {args.gpu_id}] Loaded {len(df_slice)} samples")

    # Load tokenizer
    print(f"[GPU {args.gpu_id}] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Load model + LoRA
    print(f"[GPU {args.gpu_id}] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )

    print(f"[GPU {args.gpu_id}] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)

    print(f"[GPU {args.gpu_id}] Merging LoRA...")
    model = model.merge_and_unload()
    model.eval()

    # Apply NVFP4 if requested
    if args.nvfp4:
        print(f"[GPU {args.gpu_id}] Applying NVFP4 W4A4...")
        from verl.utils.hw_error_injection import HWErrorConfig, HWErrorInjector

        config = HWErrorConfig(
            enabled=True,
            error_type="nvfp4",
            injection_point="both",
            target_modules=["linear"],
            exclude_modules=["lm_head", "embed_tokens", "layers.0", "layers.27"],
            use_ste=False,
        )
        injector = HWErrorInjector(config)
        num_hooks = injector.register_hooks(model, verbose=False)
        print(f"[GPU {args.gpu_id}] Registered {num_hooks} NVFP4 hooks")

    # Evaluate
    print(f"[GPU {args.gpu_id}] Evaluating...")
    correct = 0
    total = 0

    for idx, row in df_slice.iterrows():
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

        if score is not None and score > 0:
            correct += 1
        total += 1

        # Progress every 25 samples
        if total % 25 == 0:
            print(f"[GPU {args.gpu_id}] Progress: {total}/{len(df_slice)}, acc so far: {correct/total*100:.1f}%")

    # Save result
    result_file = f"{args.output_dir}/gpu{args.gpu_id}_result.txt"
    with open(result_file, 'w') as f:
        f.write(f"{correct}/{total}")

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"[GPU {args.gpu_id}] DONE: {correct}/{total} = {accuracy:.2f}%")


if __name__ == "__main__":
    main()
