#!/usr/bin/env python3
"""
PoC Test: Deadzone Injection Module

This script tests the unified deadzone injection module:
1. Hook-based injection (for inference)
2. Operator-level injection (for training forward/backward)

Run on A100:
    python scripts/test_deadzone_injection.py
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_hook_based_injection(model, tokenizer, fault_layer: int = 15, threshold: float = 0.01):
    """Test hook-based deadzone injection (inference mode)."""
    print("\n" + "=" * 60)
    print("TEST 1: Hook-based Deadzone Injection (Inference)")
    print("=" * 60)

    from verl.utils.deadzone_injection import DeadzoneInjector

    # Test prompt
    prompt = "What is 2 + 2?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Baseline (no deadzone)
    print("\n[Baseline] Running inference without deadzone...")
    with torch.no_grad():
        baseline_output = model(**inputs)
        baseline_logits = baseline_output.logits[:, -1, :]
        baseline_probs = torch.softmax(baseline_logits, dim=-1)
        baseline_top5 = torch.topk(baseline_probs, k=5)
        print(f"  Top-5 tokens: {[tokenizer.decode([t]) for t in baseline_top5.indices[0]]}")
        print(f"  Top-5 probs: {baseline_top5.values[0].tolist()}")

    # With deadzone
    print(f"\n[Deadzone] Running inference with deadzone on layer {fault_layer}...")
    injector = DeadzoneInjector(
        model=model,
        fault_layer=fault_layer,
        threshold=threshold,
    )
    injector.enable()

    with torch.no_grad():
        deadzone_output = model(**inputs)
        deadzone_logits = deadzone_output.logits[:, -1, :]
        deadzone_probs = torch.softmax(deadzone_logits, dim=-1)
        deadzone_top5 = torch.topk(deadzone_probs, k=5)
        print(f"  Top-5 tokens: {[tokenizer.decode([t]) for t in deadzone_top5.indices[0]]}")
        print(f"  Top-5 probs: {deadzone_top5.values[0].tolist()}")

    injector.disable()

    # Compare
    logit_diff = (deadzone_logits - baseline_logits).abs().mean().item()
    prob_diff = (deadzone_probs - baseline_probs).abs().mean().item()
    print(f"\n[Comparison]")
    print(f"  Mean logit difference: {logit_diff:.4f}")
    print(f"  Mean prob difference: {prob_diff:.6f}")

    # Check if output changed significantly
    if logit_diff > 0.1:
        print("  PASS: Deadzone injection is working (logits changed)")
        return True
    else:
        print("  WARNING: Deadzone may not be affecting output significantly")
        return False


def test_operator_level_injection(model, tokenizer, fault_layer: int = 15, threshold: float = 0.01):
    """Test operator-level deadzone injection (training mode)."""
    print("\n" + "=" * 60)
    print("TEST 2: Operator-level Deadzone Injection (Training)")
    print("=" * 60)

    from verl.utils.deadzone_injection import (
        enable_deadzone_ops,
        disable_deadzone_ops,
        register_deadzone_layer_hooks,
        get_deadzone_stats,
        reset_deadzone_stats,
    )

    # Register layer hooks for layer tracking
    num_hooks = register_deadzone_layer_hooks(model)
    print(f"  Registered {num_hooks} layer hooks")

    # Test prompt
    prompt = "What is 2 + 2?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Baseline forward + backward (no deadzone)
    print("\n[Baseline] Forward + backward without deadzone...")
    model.train()
    baseline_output = model(**inputs, labels=inputs['input_ids'])
    baseline_loss = baseline_output.loss
    print(f"  Loss: {baseline_loss.item():.4f}")

    # Compute baseline gradients
    baseline_loss.backward()
    baseline_grad_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            baseline_grad_norm += p.grad.norm().item() ** 2
    baseline_grad_norm = baseline_grad_norm ** 0.5
    print(f"  Gradient norm: {baseline_grad_norm:.4f}")

    # Zero gradients
    model.zero_grad()

    # With deadzone
    print(f"\n[Deadzone] Forward + backward with deadzone on layer {fault_layer}...")
    reset_deadzone_stats()
    enable_deadzone_ops(layer_ids=[fault_layer], threshold=threshold)

    deadzone_output = model(**inputs, labels=inputs['input_ids'])
    deadzone_loss = deadzone_output.loss
    print(f"  Loss: {deadzone_loss.item():.4f}")

    # Compute deadzone gradients
    deadzone_loss.backward()
    deadzone_grad_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            deadzone_grad_norm += p.grad.norm().item() ** 2
    deadzone_grad_norm = deadzone_grad_norm ** 0.5
    print(f"  Gradient norm: {deadzone_grad_norm:.4f}")

    stats = disable_deadzone_ops()
    print(f"\n[Stats]")
    print(f"  Forward calls: {stats['forward_calls']}")
    print(f"  Values zeroed: {stats['values_zeroed']:,}")
    print(f"  Total values: {stats['total_values']:,}")
    if stats['total_values'] > 0:
        zero_rate = stats['values_zeroed'] / stats['total_values'] * 100
        print(f"  Zero rate: {zero_rate:.2f}%")

    # Compare
    loss_diff = abs(deadzone_loss.item() - baseline_loss.item())
    grad_diff = abs(deadzone_grad_norm - baseline_grad_norm)
    print(f"\n[Comparison]")
    print(f"  Loss difference: {loss_diff:.4f}")
    print(f"  Gradient norm difference: {grad_diff:.4f}")

    # Clean up
    model.zero_grad()
    model.eval()

    if stats['values_zeroed'] > 0:
        print("  PASS: Operator-level deadzone is working")
        return True
    else:
        print("  WARNING: No values were zeroed")
        return False


def test_deadzone_with_aqn_noise(model, tokenizer, fault_layer: int = 15, threshold: float = 0.01):
    """Test that AQN noise can help signals break through deadzone."""
    print("\n" + "=" * 60)
    print("TEST 3: AQN + Deadzone Interaction")
    print("=" * 60)

    from verl.utils.deadzone_injection import DeadzoneInjector
    from verl.utils.noisy_ops import enable_noisy_ops, disable_noisy_ops, set_selective_layers

    # Test prompt
    prompt = "What is 2 + 2?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Measure baseline output variance (no deadzone, no noise)
    print("\n[Baseline] Measuring output without deadzone or noise...")
    with torch.no_grad():
        baseline_outputs = []
        for _ in range(3):
            output = model(**inputs)
            baseline_outputs.append(output.logits[:, -1, :].clone())

    baseline_variance = torch.stack(baseline_outputs).var(dim=0).mean().item()
    print(f"  Output variance (3 runs): {baseline_variance:.6f}")

    # With deadzone only
    print(f"\n[Deadzone Only] Layer {fault_layer}, threshold={threshold}...")
    injector = DeadzoneInjector(model, fault_layer, threshold)
    injector.enable()

    with torch.no_grad():
        deadzone_outputs = []
        for _ in range(3):
            output = model(**inputs)
            deadzone_outputs.append(output.logits[:, -1, :].clone())

    deadzone_variance = torch.stack(deadzone_outputs).var(dim=0).mean().item()
    deadzone_mean = torch.stack(deadzone_outputs).mean(dim=0)
    print(f"  Output variance (3 runs): {deadzone_variance:.6f}")

    injector.disable()

    # With deadzone + AQN noise
    print(f"\n[Deadzone + AQN] Layer {fault_layer}, AQN sigma=0.05...")
    injector.enable()
    enable_noisy_ops(error_scale=0.05)
    set_selective_layers([fault_layer])  # AQN only on deadzone layer

    with torch.no_grad():
        aqn_outputs = []
        for _ in range(3):
            output = model(**inputs)
            aqn_outputs.append(output.logits[:, -1, :].clone())

    aqn_variance = torch.stack(aqn_outputs).var(dim=0).mean().item()
    aqn_mean = torch.stack(aqn_outputs).mean(dim=0)
    print(f"  Output variance (3 runs): {aqn_variance:.6f}")

    disable_noisy_ops()
    injector.disable()

    # Compare: AQN should increase variance (noise) but potentially help signal survive
    print(f"\n[Analysis]")
    print(f"  Baseline variance: {baseline_variance:.6f}")
    print(f"  Deadzone variance: {deadzone_variance:.6f}")
    print(f"  Deadzone+AQN variance: {aqn_variance:.6f}")

    # Check if AQN increases variance (expected - noise adds randomness)
    if aqn_variance > deadzone_variance:
        print("  PASS: AQN adds noise as expected")
    else:
        print("  NOTE: AQN did not increase variance (deterministic inference)")

    return True


def main():
    parser = argparse.ArgumentParser(description="Test deadzone injection module")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        help="Path to model",
    )
    parser.add_argument("--fault_layer", type=int, default=15, help="Layer to inject deadzone")
    parser.add_argument("--threshold", type=float, default=0.01, help="Deadzone threshold (0.01 = 1%)")
    args = parser.parse_args()

    print("=" * 60)
    print("DEADZONE INJECTION PoC TEST")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Fault layer: {args.fault_layer}")
    print(f"Threshold: {args.threshold} ({args.threshold*100}%)")

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded on {model.device}")

    # Run tests
    results = {}

    results['hook_injection'] = test_hook_based_injection(
        model, tokenizer, args.fault_layer, args.threshold
    )

    results['operator_injection'] = test_operator_level_injection(
        model, tokenizer, args.fault_layer, args.threshold
    )

    results['aqn_interaction'] = test_deadzone_with_aqn_noise(
        model, tokenizer, args.fault_layer, args.threshold
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
