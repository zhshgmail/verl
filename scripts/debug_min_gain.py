#!/usr/bin/env python3
"""Debug script to analyze min_gain baseline across layers."""

import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()

    inputs = tokenizer("Hello world", return_tensors="pt").to("cuda")
    bias = 50.0

    print(f"\nBias magnitude: {bias}")
    print(f"{'Layer':<6}{'Min':<10}{'Max':<10}{'Mean':<10}{'Spread':<10}")
    print("-" * 46)

    for layer_id in range(len(model.model.layers)):
        layer = model.model.layers[layer_id]

        # Baseline
        baseline_out = []
        def hook_base(m, i, o):
            baseline_out.append(o[0].detach().clone())
        h1 = layer.register_forward_hook(hook_base)
        with torch.no_grad():
            model(**inputs)
        h1.remove()

        # Biased
        biased_out = []
        def hook_pre(m, args):
            return (args[0] + bias,) + args[1:]
        def hook_post(m, i, o):
            biased_out.append(o[0].detach().clone())
        h2 = layer.register_forward_pre_hook(hook_pre)
        h3 = layer.register_forward_hook(hook_post)
        with torch.no_grad():
            model(**inputs)
        h2.remove()
        h3.remove()

        shift = biased_out[0] - baseline_out[0]
        gains = shift / bias
        flat_gains = gains.float().flatten()

        min_g = torch.quantile(flat_gains, 0.001).item()
        max_g = torch.quantile(flat_gains, 0.999).item()
        mean_g = flat_gains.mean().item()
        spread = max_g - min_g

        marker = " <-- ANOMALY" if min_g < 0 or max_g > 2.0 else ""
        print(f"L{layer_id:<4}{min_g:<10.4f}{max_g:<10.4f}{mean_g:<10.4f}{spread:<10.4f}{marker}")

if __name__ == "__main__":
    main()
