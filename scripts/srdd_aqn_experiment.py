#!/usr/bin/env python3
"""
SRDD-Guided AQN vs Global AQN Experiment

This experiment validates whether SRDD-guided targeted AQN outperforms global AQN
when dealing with MXFP4 deadzone errors in specific layers.

Experiment Design:
1. Inject deadzone error in layer N (simulating MXFP4 quantization loss)
2. Compare training approaches:
   - Baseline: No deadzone, no AQN (clean reference)
   - Deadzone only: Deadzone error without AQN (shows degradation)
   - Global AQN: Deadzone + AQN on ALL layers (current approach)
   - SRDD-guided AQN: Deadzone + AQN ONLY on faulty layer (proposed approach)

Expected Results:
- SRDD-guided AQN should achieve better metrics than Global AQN because:
  - Less noise in healthy layers = better gradient signal
  - Targeted noise where needed = same protection against deadzone

Usage:
    python scripts/srdd_aqn_experiment.py \
        --model_path /path/to/Qwen2.5-1.5B-Instruct \
        --faulty_layer 10 \
        --deadzone_threshold 0.01
"""

import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add verl to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class DeadzoneHook:
    """Hook that applies deadzone (zeros small values)."""

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
        self.call_count = 0
        self.total_zeroed = 0
        self.total_elements = 0

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        # Apply deadzone: zero values < threshold * max
        max_val = hidden.abs().max()
        deadzone_thresh = self.threshold * max_val
        mask = hidden.abs() < deadzone_thresh

        self.call_count += 1
        self.total_zeroed += mask.sum().item()
        self.total_elements += hidden.numel()

        result = hidden.masked_fill(mask, 0.0)

        if isinstance(output, tuple):
            return (result,) + output[1:]
        return result


class AQNHook:
    """Hook that applies AQN (Adaptive Quantization Noise)."""

    def __init__(self, gamma: float = 0.01):
        self.gamma = gamma
        self.call_count = 0

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        # Add AQN noise proportional to activation magnitude
        noise = torch.randn_like(hidden) * self.gamma * hidden.abs().mean()
        result = hidden + noise

        self.call_count += 1

        if isinstance(output, tuple):
            return (result,) + output[1:]
        return result


def compute_loss(model, texts, tokenizer, device):
    """Compute average cross-entropy loss on texts."""
    total_loss = 0
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs.input_ids)
            total_loss += outputs.loss.item()
    return total_loss / len(texts)


def run_experiment(
    model_path: str,
    faulty_layer: int,
    deadzone_threshold: float,
    aqn_gamma: float,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Run the full SRDD-guided AQN vs Global AQN experiment."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 60)
    print("SRDD-Guided AQN vs Global AQN Experiment")
    print("=" * 60)
    print(f"  Model: {model_path}")
    print(f"  Faulty layer: {faulty_layer}")
    print(f"  Deadzone threshold: {deadzone_threshold}")
    print(f"  AQN gamma: {aqn_gamma}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Test texts for loss computation
    test_texts = [
        "The answer to 2 + 2 is 4.",
        "Paris is the capital of France.",
        "Water freezes at 0 degrees Celsius.",
        "The sun rises in the east.",
        "Python is a programming language.",
        "Machine learning is a subset of artificial intelligence.",
        "The Earth orbits around the Sun.",
        "Oxygen is essential for human survival.",
    ]

    results = {}

    # Config 1: Clean (baseline)
    print("--- Config 1: Clean (no deadzone, no AQN) ---")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model.eval()
    clean_loss = compute_loss(model, test_texts, tokenizer, device)
    print(f"  Loss: {clean_loss:.4f}")
    results['clean'] = {'loss': clean_loss}
    del model
    torch.cuda.empty_cache()

    # Config 2: Deadzone only (shows degradation)
    print(f"\n--- Config 2: Deadzone on layer {faulty_layer} (no AQN) ---")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model.eval()

    deadzone_hook = DeadzoneHook(threshold=deadzone_threshold)
    handle = model.model.layers[faulty_layer].register_forward_hook(deadzone_hook)

    deadzone_loss = compute_loss(model, test_texts, tokenizer, device)
    zeroed_pct = deadzone_hook.total_zeroed / deadzone_hook.total_elements * 100 if deadzone_hook.total_elements > 0 else 0
    print(f"  Loss: {deadzone_loss:.4f}")
    print(f"  Deadzone effect: {zeroed_pct:.1f}% elements zeroed")
    print(f"  Degradation vs clean: {(deadzone_loss - clean_loss) / clean_loss * 100:+.1f}%")
    results['deadzone_only'] = {
        'loss': deadzone_loss,
        'zeroed_pct': zeroed_pct,
        'degradation': (deadzone_loss - clean_loss) / clean_loss,
    }

    handle.remove()
    del model
    torch.cuda.empty_cache()

    # Config 3: Deadzone + Global AQN (all layers)
    print(f"\n--- Config 3: Deadzone + Global AQN (all {28} layers) ---")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model.eval()
    num_layers = len(model.model.layers)

    # Add deadzone on faulty layer
    deadzone_hook = DeadzoneHook(threshold=deadzone_threshold)
    dz_handle = model.model.layers[faulty_layer].register_forward_hook(deadzone_hook)

    # Add AQN on ALL layers
    aqn_hooks = []
    aqn_handles = []
    for i in range(num_layers):
        hook = AQNHook(gamma=aqn_gamma)
        handle = model.model.layers[i].register_forward_hook(hook)
        aqn_hooks.append(hook)
        aqn_handles.append(handle)

    global_aqn_loss = compute_loss(model, test_texts, tokenizer, device)
    print(f"  Loss: {global_aqn_loss:.4f}")
    print(f"  vs clean: {(global_aqn_loss - clean_loss) / clean_loss * 100:+.1f}%")
    print(f"  vs deadzone: {(global_aqn_loss - deadzone_loss) / deadzone_loss * 100:+.1f}%")
    results['global_aqn'] = {
        'loss': global_aqn_loss,
        'vs_clean': (global_aqn_loss - clean_loss) / clean_loss,
        'vs_deadzone': (global_aqn_loss - deadzone_loss) / deadzone_loss,
        'num_aqn_layers': num_layers,
    }

    for handle in aqn_handles:
        handle.remove()
    dz_handle.remove()
    del model
    torch.cuda.empty_cache()

    # Config 4: Deadzone + Targeted AQN (SRDD-guided, only faulty layer)
    print(f"\n--- Config 4: Deadzone + Targeted AQN (layer {faulty_layer} only) ---")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model.eval()

    # Add deadzone on faulty layer
    deadzone_hook = DeadzoneHook(threshold=deadzone_threshold)
    dz_handle = model.model.layers[faulty_layer].register_forward_hook(deadzone_hook)

    # Add AQN ONLY on faulty layer (SRDD-guided)
    aqn_hook = AQNHook(gamma=aqn_gamma)
    aqn_handle = model.model.layers[faulty_layer].register_forward_hook(aqn_hook)

    targeted_aqn_loss = compute_loss(model, test_texts, tokenizer, device)
    print(f"  Loss: {targeted_aqn_loss:.4f}")
    print(f"  vs clean: {(targeted_aqn_loss - clean_loss) / clean_loss * 100:+.1f}%")
    print(f"  vs deadzone: {(targeted_aqn_loss - deadzone_loss) / deadzone_loss * 100:+.1f}%")
    results['targeted_aqn'] = {
        'loss': targeted_aqn_loss,
        'vs_clean': (targeted_aqn_loss - clean_loss) / clean_loss,
        'vs_deadzone': (targeted_aqn_loss - deadzone_loss) / deadzone_loss,
        'num_aqn_layers': 1,
    }

    aqn_handle.remove()
    dz_handle.remove()
    del model
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Clean loss:         {clean_loss:.4f}")
    print(f"  Deadzone loss:      {deadzone_loss:.4f} ({results['deadzone_only']['degradation']*100:+.1f}%)")
    print(f"  Global AQN loss:    {global_aqn_loss:.4f} ({results['global_aqn']['vs_clean']*100:+.1f}% vs clean)")
    print(f"  Targeted AQN loss:  {targeted_aqn_loss:.4f} ({results['targeted_aqn']['vs_clean']*100:+.1f}% vs clean)")
    print()

    # Determine winner
    if targeted_aqn_loss < global_aqn_loss:
        improvement = (global_aqn_loss - targeted_aqn_loss) / global_aqn_loss * 100
        print(f"  CONCLUSION: SRDD-guided AQN is {improvement:.2f}% BETTER than Global AQN")
        results['conclusion'] = 'srdd_guided_wins'
        results['improvement_pct'] = improvement
    elif targeted_aqn_loss > global_aqn_loss:
        diff = (targeted_aqn_loss - global_aqn_loss) / global_aqn_loss * 100
        print(f"  CONCLUSION: Global AQN is {diff:.2f}% better than SRDD-guided")
        results['conclusion'] = 'global_aqn_wins'
        results['improvement_pct'] = -diff
    else:
        print(f"  CONCLUSION: Both approaches perform equally")
        results['conclusion'] = 'tie'
        results['improvement_pct'] = 0

    # Additional analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    # Noise efficiency: loss improvement per AQN layer
    global_aqn_improvement = deadzone_loss - global_aqn_loss
    targeted_aqn_improvement = deadzone_loss - targeted_aqn_loss

    global_efficiency = global_aqn_improvement / num_layers if num_layers > 0 else 0
    targeted_efficiency = targeted_aqn_improvement / 1  # Only 1 layer

    print(f"  Global AQN: {global_aqn_improvement:.4f} improvement using {num_layers} layers")
    print(f"    -> Efficiency: {global_efficiency:.6f} per layer")
    print(f"  Targeted AQN: {targeted_aqn_improvement:.4f} improvement using 1 layer")
    print(f"    -> Efficiency: {targeted_efficiency:.6f} per layer")

    if targeted_efficiency > global_efficiency:
        efficiency_ratio = targeted_efficiency / global_efficiency if global_efficiency != 0 else float('inf')
        print(f"\n  SRDD-guided AQN is {efficiency_ratio:.1f}x more efficient per layer!")

    results['analysis'] = {
        'global_improvement': global_aqn_improvement,
        'targeted_improvement': targeted_aqn_improvement,
        'global_efficiency': global_efficiency,
        'targeted_efficiency': targeted_efficiency,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="SRDD-Guided AQN vs Global AQN Experiment")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--faulty_layer", type=int, default=10, help="Layer to inject deadzone")
    parser.add_argument("--deadzone_threshold", type=float, default=0.01, help="Deadzone threshold")
    parser.add_argument("--aqn_gamma", type=float, default=0.01, help="AQN noise gamma")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()

    results = run_experiment(
        model_path=args.model_path,
        faulty_layer=args.faulty_layer,
        deadzone_threshold=args.deadzone_threshold,
        aqn_gamma=args.aqn_gamma,
        device=args.device,
    )

    results['config'] = {
        'model_path': args.model_path,
        'faulty_layer': args.faulty_layer,
        'deadzone_threshold': args.deadzone_threshold,
        'aqn_gamma': args.aqn_gamma,
    }
    results['timestamp'] = datetime.now().isoformat()

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
