#!/usr/bin/env python3
"""
Single GPU worker for parallel evaluation.
Evaluates a slice of the dataset on one GPU.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--lora_adapter_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--start_idx", type=int, required=True)
    parser.add_argument("--n_samples", type=int, required=True)
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--max_tokens", type=int, default=512)
    args = parser.parse_args()

    print(f"[GPU {args.gpu_id}] Loading data...")
    df = pd.read_parquet(args.data_path)
    df_slice = df.iloc[args.start_idx:args.start_idx + args.n_samples]

    print(f"[GPU {args.gpu_id}] Processing {len(df_slice)} samples (idx {args.start_idx}-{args.start_idx+len(df_slice)-1})")

    print(f"[GPU {args.gpu_id}] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)

    print(f"[GPU {args.gpu_id}] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"[GPU {args.gpu_id}] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, args.lora_adapter_path)

    print(f"[GPU {args.gpu_id}] Merging LoRA...")
    model = model.merge_and_unload()
    model.eval()

    print(f"[GPU {args.gpu_id}] Starting evaluation...")

    correct = 0
    for idx, row in df_slice.iterrows():
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
                max_new_tokens=args.max_tokens,
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

    # Save result to file
    result_file = f"/tmp/mxfp4_w4a4_e13j_global_aqn/eval_gpu{args.gpu_id}_result.txt"
    with open(result_file, 'w') as f:
        f.write(f"{correct}/{len(df_slice)}")

    accuracy = correct / len(df_slice) * 100 if len(df_slice) > 0 else 0
    print(f"[GPU {args.gpu_id}] Completed: {correct}/{len(df_slice)} = {accuracy:.2f}%")


if __name__ == "__main__":
    main()
