#!/usr/bin/env python3
"""
SRDD-Guided AQN vs Global AQN Experiment v2.0

Fixed issues from QA review:
1. Hook ordering: AQN applied BEFORE deadzone (combined hook)
2. AQN implementation: Per-element adaptive noise (not global mean)
3. Statistical rigor: Multiple runs with different seeds, significance tests
4. Control group: AQN on healthy layers only
5. Larger test set: 50+ samples

Usage:
    python scripts/srdd_aqn_experiment_v2.py \
        --model_path /path/to/Qwen2.5-1.5B-Instruct \
        --faulty_layer 10 \
        --num_runs 5
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))


class CombinedAQNDeadzoneHook:
    """
    Combined hook that applies AQN BEFORE deadzone.

    Order matters: AQN noise can help values "break through" the deadzone threshold.
    """
    def __init__(self, aqn_gamma: float = 0.0, deadzone_threshold: float = 0.0):
        self.aqn_gamma = aqn_gamma
        self.deadzone_threshold = deadzone_threshold
        self.call_count = 0
        self.total_zeroed = 0
        self.total_elements = 0
        self.aqn_enabled = aqn_gamma > 0
        self.deadzone_enabled = deadzone_threshold > 0

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        result = hidden

        # Step 1: Apply AQN FIRST (per-element adaptive noise)
        if self.aqn_enabled:
            # Fix: Use per-element scaling, not global mean
            noise = torch.randn_like(hidden) * self.aqn_gamma * hidden.abs()
            result = result + noise

        # Step 2: Apply deadzone AFTER AQN
        if self.deadzone_enabled:
            max_val = result.abs().max()
            if max_val > 0:
                deadzone_thresh = self.deadzone_threshold * max_val
                mask = result.abs() < deadzone_thresh
                self.total_zeroed += mask.sum().item()
                self.total_elements += result.numel()
                result = result.masked_fill(mask, 0.0)

        self.call_count += 1

        if isinstance(output, tuple):
            return (result,) + output[1:]
        return result

    def get_stats(self):
        zeroed_pct = self.total_zeroed / self.total_elements * 100 if self.total_elements > 0 else 0
        return {
            'calls': self.call_count,
            'zeroed_pct': zeroed_pct,
            'total_zeroed': self.total_zeroed,
            'total_elements': self.total_elements,
        }


class AQNOnlyHook:
    """Hook that applies only AQN (per-element adaptive noise)."""
    def __init__(self, gamma: float = 0.01):
        self.gamma = gamma
        self.call_count = 0

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        # Per-element adaptive noise (FIXED from v1)
        noise = torch.randn_like(hidden) * self.gamma * hidden.abs()
        result = hidden + noise
        self.call_count += 1

        if isinstance(output, tuple):
            return (result,) + output[1:]
        return result


def get_test_texts() -> List[str]:
    """Return a larger set of test texts (50+ samples)."""
    return [
        # Math/Logic
        "The answer to 2 + 2 is 4.",
        "If x = 5 and y = 3, then x + y equals 8.",
        "The square root of 16 is 4.",
        "10 divided by 2 equals 5.",
        "Three times four is twelve.",
        "Half of 100 is 50.",
        "The sum of 7 and 8 is 15.",
        "20 minus 12 equals 8.",
        "5 squared is 25.",
        "The cube of 3 is 27.",
        # Facts
        "Paris is the capital of France.",
        "Water freezes at 0 degrees Celsius.",
        "The sun rises in the east.",
        "The Earth orbits around the Sun.",
        "Oxygen is essential for human survival.",
        "The Moon orbits the Earth.",
        "Light travels faster than sound.",
        "Gold is a precious metal.",
        "The Atlantic Ocean separates Europe and America.",
        "Mount Everest is the tallest mountain.",
        # Programming
        "Python is a programming language.",
        "A function returns a value.",
        "Variables store data in memory.",
        "Loops repeat code multiple times.",
        "Arrays contain multiple elements.",
        "Classes define object templates.",
        "Recursion calls itself repeatedly.",
        "Strings are sequences of characters.",
        "Integers are whole numbers.",
        "Boolean values are true or false.",
        # Science
        "DNA contains genetic information.",
        "Atoms are the building blocks of matter.",
        "Gravity pulls objects toward Earth.",
        "Photosynthesis converts light to energy.",
        "Cells are the basic unit of life.",
        "Electrons orbit the nucleus.",
        "Chemical reactions change substances.",
        "Energy cannot be created or destroyed.",
        "Evolution explains species diversity.",
        "The speed of light is constant.",
        # Language
        "Verbs describe actions.",
        "Nouns name people and things.",
        "Adjectives modify nouns.",
        "Sentences express complete thoughts.",
        "Paragraphs group related sentences.",
        "Grammar defines language rules.",
        "Punctuation clarifies meaning.",
        "Vocabulary is word knowledge.",
        "Syntax is sentence structure.",
        "Semantics is word meaning.",
    ]


def compute_loss(model, texts: List[str], tokenizer, device: str) -> float:
    """Compute average cross-entropy loss on texts."""
    total_loss = 0
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs.input_ids)
            total_loss += outputs.loss.item()
    return total_loss / len(texts)


def run_single_experiment(
    model_path: str,
    faulty_layer: int,
    deadzone_threshold: float,
    aqn_gamma: float,
    device: str,
    seed: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run a single experiment with a specific seed."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_texts = get_test_texts()
    results = {'seed': seed}

    # Config 1: Clean (baseline)
    if verbose:
        print(f"  [Seed {seed}] Config 1: Clean")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model.eval()
    clean_loss = compute_loss(model, test_texts, tokenizer, device)
    results['clean'] = clean_loss
    del model
    torch.cuda.empty_cache()

    # Config 2: Deadzone only
    if verbose:
        print(f"  [Seed {seed}] Config 2: Deadzone only")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model.eval()

    hook = CombinedAQNDeadzoneHook(aqn_gamma=0, deadzone_threshold=deadzone_threshold)
    handle = model.model.layers[faulty_layer].register_forward_hook(hook)

    deadzone_loss = compute_loss(model, test_texts, tokenizer, device)
    results['deadzone_only'] = deadzone_loss
    results['deadzone_stats'] = hook.get_stats()

    handle.remove()
    del model
    torch.cuda.empty_cache()

    # Config 3: Global AQN (deadzone + AQN on ALL layers)
    if verbose:
        print(f"  [Seed {seed}] Config 3: Global AQN")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model.eval()
    num_layers = len(model.model.layers)

    handles = []
    # Faulty layer: AQN + Deadzone (combined, AQN first)
    faulty_hook = CombinedAQNDeadzoneHook(aqn_gamma=aqn_gamma, deadzone_threshold=deadzone_threshold)
    handles.append(model.model.layers[faulty_layer].register_forward_hook(faulty_hook))

    # Other layers: AQN only
    for i in range(num_layers):
        if i != faulty_layer:
            hook = AQNOnlyHook(gamma=aqn_gamma)
            handles.append(model.model.layers[i].register_forward_hook(hook))

    global_aqn_loss = compute_loss(model, test_texts, tokenizer, device)
    results['global_aqn'] = global_aqn_loss

    for h in handles:
        h.remove()
    del model
    torch.cuda.empty_cache()

    # Config 4: Targeted AQN (deadzone + AQN on faulty layer ONLY)
    if verbose:
        print(f"  [Seed {seed}] Config 4: Targeted AQN")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model.eval()

    # Faulty layer: AQN + Deadzone (combined, AQN first)
    hook = CombinedAQNDeadzoneHook(aqn_gamma=aqn_gamma, deadzone_threshold=deadzone_threshold)
    handle = model.model.layers[faulty_layer].register_forward_hook(hook)

    targeted_aqn_loss = compute_loss(model, test_texts, tokenizer, device)
    results['targeted_aqn'] = targeted_aqn_loss

    handle.remove()
    del model
    torch.cuda.empty_cache()

    # Config 5: Healthy AQN (deadzone + AQN on all layers EXCEPT faulty)
    if verbose:
        print(f"  [Seed {seed}] Config 5: Healthy AQN (control)")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model.eval()

    handles = []
    # Faulty layer: Deadzone only (NO AQN)
    faulty_hook = CombinedAQNDeadzoneHook(aqn_gamma=0, deadzone_threshold=deadzone_threshold)
    handles.append(model.model.layers[faulty_layer].register_forward_hook(faulty_hook))

    # Other layers: AQN only
    for i in range(num_layers):
        if i != faulty_layer:
            hook = AQNOnlyHook(gamma=aqn_gamma)
            handles.append(model.model.layers[i].register_forward_hook(hook))

    healthy_aqn_loss = compute_loss(model, test_texts, tokenizer, device)
    results['healthy_aqn'] = healthy_aqn_loss

    for h in handles:
        h.remove()
    del model
    torch.cuda.empty_cache()

    return results


def run_experiment(
    model_path: str,
    faulty_layer: int,
    deadzone_threshold: float,
    aqn_gamma: float,
    device: str,
    num_runs: int = 5,
) -> Dict[str, Any]:
    """Run experiment multiple times and compute statistics."""
    from transformers import AutoModelForCausalLM

    print("=" * 70)
    print("SRDD-Guided AQN vs Global AQN Experiment v2.0")
    print("=" * 70)
    print(f"  Model: {model_path}")
    print(f"  Faulty layer: {faulty_layer}")
    print(f"  Deadzone threshold: {deadzone_threshold}")
    print(f"  AQN gamma: {aqn_gamma}")
    print(f"  Number of runs: {num_runs}")
    print(f"  Test samples: {len(get_test_texts())}")
    print()

    # Get number of layers
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    num_layers = len(model.model.layers)
    del model
    torch.cuda.empty_cache()
    print(f"  Model layers: {num_layers}")
    print()

    # Run multiple experiments
    all_results = []
    for i in range(num_runs):
        seed = 42 + i * 1000
        print(f"Run {i+1}/{num_runs} (seed={seed})...")
        result = run_single_experiment(
            model_path=model_path,
            faulty_layer=faulty_layer,
            deadzone_threshold=deadzone_threshold,
            aqn_gamma=aqn_gamma,
            device=device,
            seed=seed,
            verbose=True,
        )
        all_results.append(result)
        print(f"  Clean: {result['clean']:.4f}, Deadzone: {result['deadzone_only']:.4f}, "
              f"Global: {result['global_aqn']:.4f}, Targeted: {result['targeted_aqn']:.4f}, "
              f"Healthy: {result['healthy_aqn']:.4f}")
        print()

    # Compute statistics
    configs = ['clean', 'deadzone_only', 'global_aqn', 'targeted_aqn', 'healthy_aqn']
    stats_results = {}

    for config in configs:
        values = [r[config] for r in all_results]
        stats_results[config] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values,
        }

    # Statistical significance tests
    print("=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    # T-test: Targeted AQN vs Global AQN
    targeted_values = [r['targeted_aqn'] for r in all_results]
    global_values = [r['global_aqn'] for r in all_results]
    t_stat, p_value_targeted_vs_global = stats.ttest_rel(targeted_values, global_values)

    print(f"\nTargeted AQN vs Global AQN:")
    print(f"  Targeted: {stats_results['targeted_aqn']['mean']:.4f} +/- {stats_results['targeted_aqn']['std']:.4f}")
    print(f"  Global:   {stats_results['global_aqn']['mean']:.4f} +/- {stats_results['global_aqn']['std']:.4f}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value_targeted_vs_global:.6f}")
    print(f"  Significant (p<0.05): {'YES' if p_value_targeted_vs_global < 0.05 else 'NO'}")

    # T-test: Healthy AQN vs Global AQN
    healthy_values = [r['healthy_aqn'] for r in all_results]
    t_stat_healthy, p_value_healthy_vs_global = stats.ttest_rel(healthy_values, global_values)

    print(f"\nHealthy AQN vs Global AQN:")
    print(f"  Healthy: {stats_results['healthy_aqn']['mean']:.4f} +/- {stats_results['healthy_aqn']['std']:.4f}")
    print(f"  Global:  {stats_results['global_aqn']['mean']:.4f} +/- {stats_results['global_aqn']['std']:.4f}")
    print(f"  t-statistic: {t_stat_healthy:.4f}")
    print(f"  p-value: {p_value_healthy_vs_global:.6f}")
    print(f"  Significant (p<0.05): {'YES' if p_value_healthy_vs_global < 0.05 else 'NO'}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Configuration':<25} {'Loss (mean±std)':<20} {'vs Clean':<15} {'vs Deadzone':<15}")
    print("-" * 70)

    clean_mean = stats_results['clean']['mean']
    deadzone_mean = stats_results['deadzone_only']['mean']

    for config in configs:
        mean = stats_results[config]['mean']
        std = stats_results[config]['std']
        vs_clean = (mean - clean_mean) / clean_mean * 100
        vs_deadzone = (mean - deadzone_mean) / deadzone_mean * 100 if config != 'clean' else '-'

        loss_str = f"{mean:.4f} ± {std:.4f}"
        vs_clean_str = f"{vs_clean:+.1f}%" if config != 'clean' else '-'
        vs_deadzone_str = f"{vs_deadzone:+.1f}%" if isinstance(vs_deadzone, float) else vs_deadzone

        print(f"{config:<25} {loss_str:<20} {vs_clean_str:<15} {vs_deadzone_str:<15}")

    # Conclusions
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)

    targeted_mean = stats_results['targeted_aqn']['mean']
    global_mean = stats_results['global_aqn']['mean']
    healthy_mean = stats_results['healthy_aqn']['mean']

    if targeted_mean < global_mean:
        improvement = (global_mean - targeted_mean) / global_mean * 100
        print(f"\n1. Targeted AQN is {improvement:.2f}% better than Global AQN")
        if p_value_targeted_vs_global < 0.05:
            print(f"   -> Statistically significant (p={p_value_targeted_vs_global:.6f})")
        else:
            print(f"   -> NOT statistically significant (p={p_value_targeted_vs_global:.6f})")
    else:
        diff = (targeted_mean - global_mean) / global_mean * 100
        print(f"\n1. Global AQN is {diff:.2f}% better than Targeted AQN")

    if healthy_mean < global_mean:
        improvement = (global_mean - healthy_mean) / global_mean * 100
        print(f"\n2. Healthy AQN (control) is {improvement:.2f}% better than Global AQN")
        print(f"   -> This suggests AQN on faulty layer may not be beneficial")
    else:
        diff = (healthy_mean - global_mean) / global_mean * 100
        print(f"\n2. Global AQN is {diff:.2f}% better than Healthy AQN (control)")
        print(f"   -> This suggests AQN on faulty layer IS beneficial")

    # Deadzone stats from first run
    print(f"\n3. Deadzone effect: {all_results[0]['deadzone_stats']['zeroed_pct']:.1f}% elements zeroed")

    # Package results
    final_results = {
        'config': {
            'model_path': model_path,
            'faulty_layer': faulty_layer,
            'deadzone_threshold': deadzone_threshold,
            'aqn_gamma': aqn_gamma,
            'num_runs': num_runs,
            'num_layers': num_layers,
            'num_test_samples': len(get_test_texts()),
        },
        'statistics': stats_results,
        'significance': {
            'targeted_vs_global': {
                't_statistic': t_stat,
                'p_value': p_value_targeted_vs_global,
                'significant': p_value_targeted_vs_global < 0.05,
            },
            'healthy_vs_global': {
                't_statistic': t_stat_healthy,
                'p_value': p_value_healthy_vs_global,
                'significant': p_value_healthy_vs_global < 0.05,
            },
        },
        'raw_results': all_results,
        'timestamp': datetime.now().isoformat(),
    }

    return final_results


def main():
    parser = argparse.ArgumentParser(description="SRDD-Guided AQN Experiment v2.0")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--faulty_layer", type=int, default=10)
    parser.add_argument("--deadzone_threshold", type=float, default=0.01)
    parser.add_argument("--aqn_gamma", type=float, default=0.01)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    results = run_experiment(
        model_path=args.model_path,
        faulty_layer=args.faulty_layer,
        deadzone_threshold=args.deadzone_threshold,
        aqn_gamma=args.aqn_gamma,
        device=args.device,
        num_runs=args.num_runs,
    )

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
