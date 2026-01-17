#!/usr/bin/env python3
"""Quick test to verify evaluation is correct by testing base model (no LoRA)."""

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.insert(0, "/home/z00637938/workspace/verl")
from verl.utils.reward_score import default_compute_score

# Load base model (no LoRA)
base_path = "/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
print("Loading base model (NO LoRA)...")
tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)
model.eval()

# Test on first 50 samples
df = pd.read_parquet("/data/z00637938/gsm8k/test.parquet")
correct = 0
total = 50

print(f"Testing base model on {total} samples...")
for idx in range(total):
    row = df.iloc[idx]
    prompt = row["prompt"]
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    gt = row["reward_model"]["ground_truth"]

    chat = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    resp = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    score = default_compute_score("openai/gsm8k", resp, gt)
    if score and score > 0:
        correct += 1
    if (idx + 1) % 10 == 0:
        print(f"  {idx+1}/{total}: {correct}/{idx+1} = {correct/(idx+1)*100:.1f}%")

print(f"\n=== Base model (NO LoRA) accuracy: {correct}/{total} = {correct/total*100:.1f}% ===")
