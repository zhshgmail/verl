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

    print(f"\nExpected: Layer 15 with SAT_SOURCE")
    print(f"Actual:   Layer {diagnosed_layer} with {diagnosis}")
    print(f"Validation: {validation}")

    if diagnosed_layer == 15 and any('SAT' in d for d in diagnosis):
        print("\n" + "=" * 70)
        print("✅ TEST PASSED: SRDD correctly identified Layer 15 saturation")
        print("=" * 70)
        print("\nThis output format can be used in the PPT use case document.")
        return 0
    else:
        print("\n" + "=" * 70)
        print("❌ TEST FAILED: SRDD did not identify the expected fault")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
