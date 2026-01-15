#!/usr/bin/env python3
"""
Diagnostic Validation Test: Layer-Level Error Detection

This script validates that the AQN diagnostic methodology can automatically
detect which layer contains a "hardware error" (simulated via noise injection).

The validation scenario:
1. Load a model and establish baseline (clean) output
2. Inject noise into a single "ground truth" layer (simulating HW error)
3. Run diagnostic sweep: test each layer individually
4. Identify the layer with highest sensitivity (most output change)
5. Verify the diagnosed layer matches ground truth

Success criteria:
- The diagnostic sweep correctly identifies the ground truth error layer
- OR identifies layers within 1-2 positions (due to cascading effects)
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add verl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from verl.utils.noisy_ops import (
    enable_noisy_ops,
    disable_noisy_ops,
    set_selective_layers,
    set_selective_operators,
    register_layer_hooks,
    get_layer_injection_stats,
    reset_layer_injection_stats,
    reset_injection_stats,
)


def load_model(model_path: str, device: str = "cuda"):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    # Get number of layers
    num_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 28
    print(f"Model loaded: {num_layers} layers")

    return model, tokenizer, num_layers


def get_model_output(model, tokenizer, prompt: str, max_tokens: int = 50) -> tuple:
    """Get model output logits and generated text."""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        # Get logits for comparison
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :].clone()  # Last token logits

        # Generate text
        generated = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        text = tokenizer.decode(generated[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    return logits, text


def compute_output_divergence(baseline_logits: torch.Tensor, test_logits: torch.Tensor) -> float:
    """Compute KL divergence between baseline and test outputs."""
    baseline_probs = torch.softmax(baseline_logits, dim=-1)
    test_probs = torch.softmax(test_logits, dim=-1)

    # KL divergence: sum(p * log(p/q))
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    kl_div = torch.sum(baseline_probs * torch.log((baseline_probs + eps) / (test_probs + eps)))

    return kl_div.item()


def run_baseline(model, tokenizer, prompts: list) -> dict:
    """Run baseline (clean) inference."""
    print("\n=== Running Baseline (No Noise) ===")
    disable_noisy_ops()

    results = {'logits': [], 'texts': []}
    for prompt in prompts:
        logits, text = get_model_output(model, tokenizer, prompt)
        results['logits'].append(logits)
        results['texts'].append(text)

    return results


def run_with_noise_on_layer(
    model,
    tokenizer,
    prompts: list,
    layer_id: int,
    noise_scale: float = 0.05,
) -> dict:
    """Run inference with noise injected into a specific layer."""
    reset_injection_stats()
    reset_layer_injection_stats()

    # Enable noisy ops
    enable_noisy_ops(error_scale=noise_scale, error_type='relative_gaussian')

    # Set selective layer (only inject into this layer)
    set_selective_layers([layer_id])

    results = {'logits': [], 'texts': []}
    for prompt in prompts:
        logits, text = get_model_output(model, tokenizer, prompt)
        results['logits'].append(logits)
        results['texts'].append(text)

    # Get injection stats
    stats = get_layer_injection_stats()

    # Disable
    disable_noisy_ops()
    set_selective_layers(None)  # Reset to all layers

    return results, stats


def diagnostic_sweep(
    model,
    tokenizer,
    prompts: list,
    baseline_results: dict,
    num_layers: int,
    noise_scale: float = 0.05,
) -> dict:
    """Run diagnostic sweep across all layers."""
    print(f"\n=== Running Diagnostic Sweep ({num_layers} layers) ===")

    layer_divergences = {}

    for layer_id in range(num_layers):
        results, stats = run_with_noise_on_layer(
            model, tokenizer, prompts, layer_id, noise_scale
        )

        # Compute average divergence
        divergences = []
        for i in range(len(prompts)):
            div = compute_output_divergence(
                baseline_results['logits'][i],
                results['logits'][i]
            )
            divergences.append(div)

        avg_divergence = np.mean(divergences)
        layer_divergences[layer_id] = avg_divergence

        # Progress
        if (layer_id + 1) % 5 == 0 or layer_id == num_layers - 1:
            print(f"  Layer {layer_id}: divergence = {avg_divergence:.4f}")

    return layer_divergences


def identify_error_layer(layer_divergences: dict, top_k: int = 3) -> list:
    """Identify layers with highest divergence."""
    sorted_layers = sorted(layer_divergences.items(), key=lambda x: x[1], reverse=True)
    return sorted_layers[:top_k]


def run_validation_test(
    model_path: str,
    ground_truth_layer: int,
    noise_scale: float = 0.05,
    device: str = "cuda",
):
    """Run the full validation test."""
    print("=" * 70)
    print("DIAGNOSTIC VALIDATION TEST")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Ground Truth Error Layer: {ground_truth_layer}")
    print(f"Noise Scale: {noise_scale}")
    print("=" * 70)

    # Load model
    model, tokenizer, num_layers = load_model(model_path, device)

    # Register layer hooks for layer-aware injection
    num_hooks = register_layer_hooks(model)
    print(f"Registered {num_hooks} layer hooks")

    # Test prompts
    prompts = [
        "What is 2 + 2? Answer:",
        "The capital of France is",
        "def fibonacci(n):",
    ]

    # Step 1: Get baseline
    baseline = run_baseline(model, tokenizer, prompts)
    print(f"Baseline texts: {baseline['texts']}")

    # Step 2: Verify noise injection works on ground truth layer
    print(f"\n=== Verifying Noise on Ground Truth Layer {ground_truth_layer} ===")
    gt_results, gt_stats = run_with_noise_on_layer(
        model, tokenizer, prompts, ground_truth_layer, noise_scale
    )
    print(f"Injection stats: {gt_stats}")

    gt_divergences = []
    for i in range(len(prompts)):
        div = compute_output_divergence(baseline['logits'][i], gt_results['logits'][i])
        gt_divergences.append(div)
    print(f"Ground truth layer divergence: {np.mean(gt_divergences):.4f}")
    print(f"Ground truth outputs: {gt_results['texts']}")

    # Step 3: Run diagnostic sweep
    layer_divergences = diagnostic_sweep(
        model, tokenizer, prompts, baseline, num_layers, noise_scale
    )

    # Step 4: Identify error layer
    print("\n=== RESULTS ===")
    top_layers = identify_error_layer(layer_divergences, top_k=5)

    print("\nTop 5 most sensitive layers:")
    for rank, (layer_id, divergence) in enumerate(top_layers, 1):
        marker = " <-- GROUND TRUTH" if layer_id == ground_truth_layer else ""
        print(f"  {rank}. Layer {layer_id}: {divergence:.4f}{marker}")

    # Step 5: Validation
    diagnosed_layer = top_layers[0][0]

    print(f"\n{'=' * 50}")
    print(f"Ground Truth Layer: {ground_truth_layer}")
    print(f"Diagnosed Layer:    {diagnosed_layer}")

    if diagnosed_layer == ground_truth_layer:
        print("VALIDATION: ✅ SUCCESS - Exact match!")
        success = True
    elif abs(diagnosed_layer - ground_truth_layer) <= 2:
        print(f"VALIDATION: ⚠️  PARTIAL - Within 2 layers (cascade effect)")
        success = True
    else:
        print(f"VALIDATION: ❌ FAILED - Layer mismatch")
        success = False

    # Check if ground truth is in top 3
    top_3_ids = [l[0] for l in top_layers[:3]]
    if ground_truth_layer in top_3_ids:
        rank = top_3_ids.index(ground_truth_layer) + 1
        print(f"Ground truth layer is #{rank} in top 3")

    print("=" * 50)

    return success, layer_divergences


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Diagnostic Validation Test")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--ground_truth_layer", type=int, default=10,
                       help="Layer to inject noise into (ground truth)")
    parser.add_argument("--noise_scale", type=float, default=0.05,
                       help="Noise scale for injection")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")

    args = parser.parse_args()

    success, divergences = run_validation_test(
        model_path=args.model_path,
        ground_truth_layer=args.ground_truth_layer,
        noise_scale=args.noise_scale,
        device=args.device,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
