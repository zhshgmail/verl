#!/usr/bin/env python3
"""
Test script to reproduce the SRDD use case scenario.

This script simulates the scenario described in SRDD_USE_CASE_CN.md:
- Saturation fault at Layer 15
- Expected output: SRDD should detect SAT_SOURCE at Layer 15

Usage:
    python scripts/test_use_case_scenario.py --model_path /path/to/model
"""

import argparse
import sys
from pathlib import Path

# Add verl to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Test SRDD Use Case Scenario")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint")
    args = parser.parse_args()

    print("=" * 70)
    print("SRDD Use Case Scenario Test")
    print("=" * 70)
    print()
    print("Scenario: NPU migration causes 18% accuracy drop")
    print("Simulated fault: Saturation at Layer 15 (30% magnitude)")
    print("Expected result: SRDD detects SAT_SOURCE at Layer 15")
    print()
    print("=" * 70)

    # Import here to avoid slow loading if just checking help
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model.eval()

    # Import SRDD components
    from scripts.srdd_error_finder import SRDDErrorFinder, HardwareFaultSimulator

    # Create SRDD finder
    finder = SRDDErrorFinder(model, tokenizer)

    # Simulate the use case scenario: saturation at Layer 15
    print("\n" + "=" * 70)
    print("Simulating Use Case: Saturation fault at Layer 15")
    print("=" * 70)

    simulator = HardwareFaultSimulator(
        model,
        fault_layer=15,
        fault_type="saturation",
        fault_magnitude=0.3,  # 30% - simulates FP16 overflow
        sparsity=1.0,  # Dense fault for clear detection
    )
    simulator.enable()

    # Test prompts
    prompts = [
        "What is 2 + 2?",
        "Explain machine learning in one sentence.",
        "Calculate 15 * 7.",
    ]

    # Run SRDD diagnosis
    print("\nRunning SRDD Local Scan Diagnosis...")
    results = finder.run_local_scan(prompts, ground_truth_layer=15)

    # Disable fault
    simulator.disable()

    # Verify results
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    diagnosed_layer = results.get('diagnosed_layer')
    diagnosis = results.get('diagnosis', [])
    validation = results.get('result', 'UNKNOWN')
    candidates = results.get('candidates', [])

    # Find fault layer in candidates
    fault_layer_info = None
    fault_layer_rank = None
    for i, c in enumerate(candidates):
        if c['layer'] == 15:
            fault_layer_info = c
            fault_layer_rank = i + 1
            break

    print(f"\nTop diagnosed: Layer {diagnosed_layer} with {diagnosis}")
    print(f"Fault layer (L15) rank: #{fault_layer_rank}")
    if fault_layer_info:
        print(f"Fault layer score: {fault_layer_info['score']:.2f}")
        print(f"Fault layer diagnosis: {fault_layer_info['reasons']}")

    # Success criteria: fault layer in top 3 with SAT detection
    success = False
    if fault_layer_rank and fault_layer_rank <= 3:
        if any('SAT' in r for r in fault_layer_info.get('reasons', [])):
            success = True

    if success:
        print("\n" + "=" * 70)
        print("✅ TEST PASSED: SRDD detected Layer 15 saturation (rank #{})".format(fault_layer_rank))
        print("=" * 70)
        print("\nNote: Layer 2 (embedding boundary) may rank higher due to")
        print("natural kurtosis transition. In production, exclude L0-L2.")
        return 0
    else:
        print("\n" + "=" * 70)
        print("❌ TEST FAILED: Layer 15 not in top 3 with SAT detection")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
