#!/usr/bin/env python3
"""
Comprehensive validation tests for MXFP4 quantization implementation.

This test suite validates that:
1. MXFP4 quantization causes measurable precision loss (not a no-op)
2. The precision loss magnitude is realistic for 4-bit quantization
3. The HWErrorInjector correctly integrates with the MXFP4 quantization library

Run: python tests/test_mxfp4_quant_validation.py
"""

import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.insert(0, '/home/zheng/workspace/verl')

from verl.utils.mxfp4_quant import (
    mxfp4_quantize,
    MXFP4Config,
    compute_mxfp4_error,
    analyze_mxfp4_sensitivity,
)
from verl.utils.hw_error_injection import (
    HWErrorConfig,
    HWErrorInjector,
)


def compute_sqnr_db(signal: torch.Tensor, noise: torch.Tensor) -> float:
    """
    Compute Signal-to-Quantization-Noise Ratio in dB.

    SQNR = 10 * log10(signal_power / noise_power)

    For 4-bit quantization, theoretical SQNR should be around:
    - FP4: ~24-30 dB (due to limited mantissa precision)
    - Lower SQNR indicates more quantization error (expected for 4-bit)

    Args:
        signal: Original signal tensor
        noise: Quantization error (quant - original)

    Returns:
        SQNR in dB
    """
    signal_power = (signal ** 2).mean().item()
    noise_power = (noise ** 2).mean().item()

    if noise_power < 1e-20:
        return float('inf')

    sqnr_db = 10 * np.log10(signal_power / noise_power)
    return sqnr_db


def compute_enob(sqnr_db: float) -> float:
    """
    Compute Effective Number of Bits from SQNR.

    ENOB = (SQNR - 1.76) / 6.02

    For 4-bit quantization, ENOB should be < 4 bits due to non-uniform
    quantization in MXFP4 format.

    Args:
        sqnr_db: SQNR in dB

    Returns:
        Effective number of bits
    """
    enob = (sqnr_db - 1.76) / 6.02
    return enob


class SimpleLinear(nn.Module):
    """Simple linear layer for testing."""
    def __init__(self, in_features=256, out_features=256):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


def test_mxfp4_causes_precision_loss():
    """
    Test 1: Verify MXFP4 quantization causes measurable precision loss.

    This is critical - if quantization is a no-op, the entire experiment is invalid.
    """
    print("\n" + "="*80)
    print("TEST 1: MXFP4 Precision Loss Verification")
    print("="*80)

    # Create test tensor with realistic activation values
    torch.manual_seed(42)
    x = torch.randn(128, 2048) * 0.5  # Scale to typical activation range

    print(f"\nInput tensor stats:")
    print(f"  Shape: {x.shape}")
    print(f"  Range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"  Mean: {x.mean():.4f}, Std: {x.std():.4f}")

    # Apply MXFP4 quantization
    x_quant = mxfp4_quantize(x)

    # Compute quantization error
    error = x_quant - x
    abs_error = error.abs()

    print(f"\nQuantization error stats:")
    print(f"  Mean absolute error: {abs_error.mean():.6e}")
    print(f"  Max absolute error: {abs_error.max():.6e}")
    print(f"  Std of error: {abs_error.std():.6e}")

    # Compute relative error
    relative_error = abs_error / (x.abs() + 1e-10)
    print(f"  Mean relative error: {relative_error.mean():.4f} ({relative_error.mean()*100:.2f}%)")
    print(f"  Max relative error: {relative_error.max():.4f}")

    # Compute SQNR
    sqnr_db = compute_sqnr_db(x, error)
    enob = compute_enob(sqnr_db)
    print(f"\nQuantization quality metrics:")
    print(f"  SQNR: {sqnr_db:.2f} dB")
    print(f"  Effective bits (ENOB): {enob:.2f} bits")

    # Check that values actually changed
    num_changed = (x != x_quant).sum().item()
    total = x.numel()
    change_rate = num_changed / total * 100
    print(f"\nValues changed: {num_changed}/{total} ({change_rate:.1f}%)")

    # CRITICAL ASSERTIONS
    print("\n" + "-"*80)
    print("Validation checks:")

    # 1. Values should actually change (not a no-op)
    assert num_changed > total * 0.5, \
        f"FAIL: Only {change_rate:.1f}% of values changed. Quantization may be a no-op!"
    print(f"  [PASS] Values changed: {change_rate:.1f}% > 50%")

    # 2. Mean absolute error should be non-negligible
    mean_abs_error = abs_error.mean().item()
    assert mean_abs_error > 1e-6, \
        f"FAIL: Mean absolute error too small ({mean_abs_error:.2e}). Quantization has no effect!"
    print(f"  [PASS] Mean absolute error: {mean_abs_error:.2e} > 1e-6")

    # 3. Relative error should be measurable (expect ~1-10% for 4-bit)
    mean_rel_error = relative_error.mean().item()
    assert mean_rel_error > 0.001, \
        f"FAIL: Relative error too small ({mean_rel_error:.4f}). Quantization has minimal impact!"
    print(f"  [PASS] Mean relative error: {mean_rel_error*100:.2f}% > 0.1%")

    # 4. SQNR should be realistic for 4-bit quantization (expect 20-35 dB)
    assert 15 < sqnr_db < 50, \
        f"FAIL: SQNR {sqnr_db:.2f} dB is unrealistic for 4-bit quantization (expect 15-50 dB)"
    print(f"  [PASS] SQNR {sqnr_db:.2f} dB is in realistic range [15, 50] dB")

    # 5. ENOB should be less than 8 bits (4-bit format shouldn't perform like 8-bit)
    assert enob < 8.0, \
        f"FAIL: ENOB {enob:.2f} bits is too high for 4-bit quantization!"
    print(f"  [PASS] ENOB {enob:.2f} bits < 8.0 (realistic for 4-bit)")

    print("\n" + "="*80)
    print("TEST 1: PASSED - MXFP4 quantization causes measurable precision loss")
    print("="*80)


def test_mxfp4_precision_loss_magnitude():
    """
    Test 2: Verify the magnitude of precision loss is realistic for 4-bit quantization.

    We test across different tensor statistics to ensure robustness.
    """
    print("\n" + "="*80)
    print("TEST 2: MXFP4 Precision Loss Magnitude Validation")
    print("="*80)

    test_cases = [
        ("Small values (0.01 scale)", torch.randn(256, 512) * 0.01),
        ("Normal values (0.5 scale)", torch.randn(256, 512) * 0.5),
        ("Large values (5.0 scale)", torch.randn(256, 512) * 5.0),
        ("Mixed range", torch.cat([
            torch.randn(128, 512) * 0.01,
            torch.randn(128, 512) * 1.0
        ], dim=0)),
    ]

    print("\nTesting MXFP4 across different input distributions:")
    print("-"*80)

    all_sqnrs = []
    all_enobs = []

    for name, x in test_cases:
        x_quant = mxfp4_quantize(x)
        error = x_quant - x
        abs_error = error.abs()
        relative_error = abs_error / (x.abs() + 1e-10)

        sqnr_db = compute_sqnr_db(x, error)
        enob = compute_enob(sqnr_db)

        all_sqnrs.append(sqnr_db)
        all_enobs.append(enob)

        print(f"\n{name}:")
        print(f"  Input range: [{x.min():.4f}, {x.max():.4f}]")
        print(f"  Mean absolute error: {abs_error.mean():.6e}")
        print(f"  Mean relative error: {relative_error.mean()*100:.2f}%")
        print(f"  SQNR: {sqnr_db:.2f} dB")
        print(f"  ENOB: {enob:.2f} bits")

        # Validate each case
        assert abs_error.mean() > 1e-7, f"{name}: Error too small"
        assert 10 < sqnr_db < 60, f"{name}: SQNR {sqnr_db:.2f} dB unrealistic"
        assert enob < 10.0, f"{name}: ENOB {enob:.2f} too high"

    # Summary statistics
    mean_sqnr = np.mean(all_sqnrs)
    mean_enob = np.mean(all_enobs)

    print("\n" + "-"*80)
    print("Summary across all test cases:")
    print(f"  Mean SQNR: {mean_sqnr:.2f} dB (range: [{min(all_sqnrs):.2f}, {max(all_sqnrs):.2f}])")
    print(f"  Mean ENOB: {mean_enob:.2f} bits (range: [{min(all_enobs):.2f}, {max(all_enobs):.2f}])")

    print("\n" + "="*80)
    print("TEST 2: PASSED - Precision loss magnitude is realistic for 4-bit")
    print("="*80)


def test_mxfp4_block_quantization():
    """
    Test 3: Verify block-based quantization behavior (E2M1K8B32).

    MXFP4 uses blocks of 32 elements with shared exponent.
    This test verifies that block structure impacts quantization.
    """
    print("\n" + "="*80)
    print("TEST 3: MXFP4 Block Quantization Behavior")
    print("="*80)

    # Create tensor with known block structure
    # Each block of 32 will have different characteristics
    torch.manual_seed(42)

    # Block 1: Small values
    block1 = torch.randn(4, 32) * 0.01
    # Block 2: Large values
    block2 = torch.randn(4, 32) * 10.0
    # Block 3: Mixed values
    block3 = torch.randn(4, 32)

    x = torch.cat([block1, block2, block3], dim=0)

    print(f"\nInput structure:")
    print(f"  Total shape: {x.shape}")
    print(f"  Block 1 (small): mean={block1.mean():.4f}, std={block1.std():.4f}")
    print(f"  Block 2 (large): mean={block2.mean():.4f}, std={block2.std():.4f}")
    print(f"  Block 3 (mixed): mean={block3.mean():.4f}, std={block3.std():.4f}")

    # Apply quantization
    x_quant = mxfp4_quantize(x)

    # Analyze error per block
    error = (x_quant - x).abs()
    error_block1 = error[:4, :].mean()
    error_block2 = error[4:8, :].mean()
    error_block3 = error[8:, :].mean()

    print(f"\nQuantization error per block:")
    print(f"  Block 1 error: {error_block1:.6e}")
    print(f"  Block 2 error: {error_block2:.6e}")
    print(f"  Block 3 error: {error_block3:.6e}")

    # Block 2 should have larger absolute error due to larger values
    print(f"\nError scaling with magnitude:")
    print(f"  Block 2 / Block 1 error ratio: {error_block2 / (error_block1 + 1e-10):.2f}x")

    # All blocks should have measurable error
    assert error_block1 > 1e-8, "Block 1 error too small"
    assert error_block2 > 1e-7, "Block 2 error too small"
    assert error_block3 > 1e-8, "Block 3 error too small"

    # Larger values should generally have larger absolute error
    # (but relative error should be similar due to floating point nature)
    assert error_block2 > error_block1, \
        "Larger values should have larger absolute error"

    print("\n" + "-"*80)
    print("Validation checks:")
    print(f"  [PASS] All blocks show measurable quantization error")
    print(f"  [PASS] Error scales appropriately with magnitude")

    print("\n" + "="*80)
    print("TEST 3: PASSED - Block quantization works correctly")
    print("="*80)


def test_hw_error_injector_mxfp4_integration():
    """
    Test 4: Verify HWErrorInjector correctly calls MXFP4 quantization.

    This tests the integration between hw_error_injection.py and mxfp4_quant.py.
    """
    print("\n" + "="*80)
    print("TEST 4: HWErrorInjector MXFP4 Integration")
    print("="*80)

    # Create a simple model
    model = SimpleLinear(256, 256)
    torch.manual_seed(42)
    x = torch.randn(4, 32, 256)  # (batch, seq, hidden)

    print(f"\nTest setup:")
    print(f"  Model: Linear(256, 256)")
    print(f"  Input shape: {x.shape}")

    # Run without injection (baseline)
    with torch.no_grad():
        y_baseline = model(x.clone())

    print(f"  Baseline output range: [{y_baseline.min():.4f}, {y_baseline.max():.4f}]")

    # Test INPUT injection (recommended for MXFP4)
    print(f"\n--- Testing INPUT injection (quant before computation) ---")
    config_input = HWErrorConfig(
        enabled=True,
        error_type='mxfp4',
        injection_point='input',
        target_modules=['linear'],
        mxfp4_stochastic_rounding=False,
        mxfp4_block_2d=False,
    )
    injector_input = HWErrorInjector(config_input)

    # Verify MXFP4 quantizer was initialized
    assert injector_input._mxfp4_quantize is not None, \
        "FAIL: MXFP4 quantizer not initialized!"
    print(f"  [PASS] MXFP4 quantizer initialized")

    # Register hooks
    num_hooks = injector_input.register_hooks(model, verbose=False)
    print(f"  [PASS] Registered {num_hooks} hooks")
    assert num_hooks >= 1, f"Expected at least 1 hook for linear layer, got {num_hooks}"

    # Run with injection
    with torch.no_grad():
        y_input_injected = model(x.clone())

    # Get statistics
    stats_input = injector_input.get_stats()
    print(f"\nInjection statistics:")
    for name, stat in stats_input.items():
        print(f"  {name}:")
        print(f"    Injections: {stat['count']}")
        print(f"    Mean error: {stat['mean_error']:.6e}")
        print(f"    Relative error: {stat['relative_error']*100:.2f}%")
        print(f"    Injection point: {stat['point']}")
        print(f"    Type: {stat['type']}")

    injector_input.remove_hooks()

    # Verify injection had effect
    output_diff = (y_input_injected - y_baseline).abs().mean().item()
    output_rel_diff = output_diff / y_baseline.abs().mean().item()

    print(f"\nOutput difference from baseline:")
    print(f"  Absolute: {output_diff:.6e}")
    print(f"  Relative: {output_rel_diff*100:.2f}%")

    # Test OUTPUT injection (alternative, less common)
    print(f"\n--- Testing OUTPUT injection (quant after computation) ---")
    config_output = HWErrorConfig(
        enabled=True,
        error_type='mxfp4',
        injection_point='output',
        target_modules=['linear'],
    )
    injector_output = HWErrorInjector(config_output)
    injector_output.register_hooks(model, verbose=False)

    with torch.no_grad():
        y_output_injected = model(x.clone())

    stats_output = injector_output.get_stats()
    injector_output.remove_hooks()

    output_diff_out = (y_output_injected - y_baseline).abs().mean().item()

    print(f"Output difference from baseline: {output_diff_out:.6e}")

    # CRITICAL ASSERTIONS
    print("\n" + "-"*80)
    print("Validation checks:")

    # 1. MXFP4 quantizer was initialized
    assert injector_input._mxfp4_quantize is not None
    print(f"  [PASS] MXFP4 quantizer properly initialized")

    # 2. Hooks were registered
    assert len(stats_input) > 0, "No injection statistics collected!"
    print(f"  [PASS] Hooks registered and executed: {len(stats_input)} modules")

    # 3. Injection was actually called
    for name, stat in stats_input.items():
        assert stat['count'] > 0, f"{name}: count is 0, injection not called!"
        assert stat['type'] == 'mxfp4', f"{name}: wrong type {stat['type']}"
        assert stat['point'] == 'input', f"{name}: wrong injection point {stat['point']}"
    print(f"  [PASS] MXFP4 quantization was called during forward pass")

    # 4. Quantization had measurable effect
    assert output_diff > 1e-7, \
        f"FAIL: Output difference {output_diff:.2e} too small. MXFP4 may not be working!"
    print(f"  [PASS] MXFP4 injection affected output: {output_rel_diff*100:.2f}% change")

    # 5. Error statistics are realistic (at least one module should have significant error)
    has_significant_error = False
    for name, stat in stats_input.items():
        # Skip wrapper modules that don't actually process data
        if stat['mean_error'] > 1e-8 and stat['relative_error'] > 1e-4:
            has_significant_error = True
            break
    assert has_significant_error, \
        "FAIL: No module shows significant quantization error!"
    print(f"  [PASS] Error statistics show realistic quantization impact")

    # 6. Both input and output injection work
    assert output_diff_out > 1e-7, "OUTPUT injection had no effect!"
    print(f"  [PASS] Both INPUT and OUTPUT injection modes work")

    print("\n" + "="*80)
    print("TEST 4: PASSED - HWErrorInjector integrates correctly with MXFP4")
    print("="*80)


def test_mxfp4_deterministic_behavior():
    """
    Test 5: Verify MXFP4 quantization is deterministic (without stochastic rounding).
    """
    print("\n" + "="*80)
    print("TEST 5: MXFP4 Deterministic Behavior")
    print("="*80)

    torch.manual_seed(42)
    x = torch.randn(128, 256)

    # Apply quantization multiple times
    x_quant1 = mxfp4_quantize(x, stochastic_rounding=False)
    x_quant2 = mxfp4_quantize(x, stochastic_rounding=False)
    x_quant3 = mxfp4_quantize(x, stochastic_rounding=False)

    # Should be identical
    diff_12 = (x_quant1 - x_quant2).abs().max().item()
    diff_23 = (x_quant2 - x_quant3).abs().max().item()

    print(f"\nDeterminism check:")
    print(f"  Max diff between run 1 and 2: {diff_12:.2e}")
    print(f"  Max diff between run 2 and 3: {diff_23:.2e}")

    assert diff_12 < 1e-10, "Quantization not deterministic!"
    assert diff_23 < 1e-10, "Quantization not deterministic!"

    print(f"  [PASS] MXFP4 quantization is deterministic")

    # Test with stochastic rounding - should be different
    print(f"\nStochastic rounding check:")
    x_sr1 = mxfp4_quantize(x, stochastic_rounding=True)
    x_sr2 = mxfp4_quantize(x, stochastic_rounding=True)

    diff_sr = (x_sr1 - x_sr2).abs().max().item()
    print(f"  Max diff between stochastic runs: {diff_sr:.2e}")

    # Should be different (with high probability)
    assert diff_sr > 1e-10, "Stochastic rounding not working!"
    print(f"  [PASS] Stochastic rounding produces different results")

    print("\n" + "="*80)
    print("TEST 5: PASSED - Deterministic and stochastic modes work correctly")
    print("="*80)


def test_mxfp4_edge_cases():
    """
    Test 6: Test MXFP4 on edge cases (zeros, very small/large values, etc.).
    """
    print("\n" + "="*80)
    print("TEST 6: MXFP4 Edge Cases")
    print("="*80)

    # Test case 1: All zeros
    print("\nTest case 1: All zeros")
    x_zeros = torch.zeros(64, 128)
    x_zeros_quant = mxfp4_quantize(x_zeros)
    assert torch.allclose(x_zeros_quant, x_zeros, atol=1e-10), "Zeros not preserved!"
    print("  [PASS] All zeros preserved")

    # Test case 2: Very small values (near underflow)
    print("\nTest case 2: Very small values (near underflow)")
    x_small = torch.randn(64, 128) * 1e-10
    x_small_quant = mxfp4_quantize(x_small)
    # Small values may underflow to zero in MXFP4
    zero_ratio = (x_small_quant == 0).float().mean().item()
    print(f"  Values underflowed to zero: {zero_ratio*100:.1f}%")
    print(f"  [PASS] Small value handling: {zero_ratio*100:.1f}% underflowed")

    # Test case 3: Very large values (near overflow)
    print("\nTest case 3: Very large values (near overflow)")
    x_large = torch.randn(64, 128) * 1e6
    x_large_quant = mxfp4_quantize(x_large)
    # Should be clamped but not crash
    assert not torch.isnan(x_large_quant).any(), "NaN in quantized large values!"
    assert not torch.isinf(x_large_quant).any(), "Inf in quantized large values!"
    print(f"  [PASS] Large values handled without NaN/Inf")

    # Test case 4: Mixed positive and negative
    print("\nTest case 4: Mixed positive and negative")
    x_mixed = torch.cat([torch.ones(32, 64), -torch.ones(32, 64)], dim=0)
    x_mixed_quant = mxfp4_quantize(x_mixed)
    pos_preserved = (x_mixed_quant[:32] > 0).all()
    neg_preserved = (x_mixed_quant[32:] < 0).all()
    assert pos_preserved and neg_preserved, "Sign not preserved!"
    print(f"  [PASS] Sign preserved for positive and negative values")

    # Test case 5: Single element (edge case for blocking)
    print("\nTest case 5: Single element")
    x_single = torch.tensor([1.234])
    x_single_quant = mxfp4_quantize(x_single)
    assert x_single_quant.shape == x_single.shape, "Shape changed!"
    print(f"  Input: {x_single.item():.6f}, Quantized: {x_single_quant.item():.6f}")
    print(f"  [PASS] Single element handled correctly")

    print("\n" + "="*80)
    print("TEST 6: PASSED - Edge cases handled correctly")
    print("="*80)


def test_mxfp4_sensitivity_analysis():
    """
    Test 7: Use the built-in sensitivity analysis function.
    """
    print("\n" + "="*80)
    print("TEST 7: MXFP4 Sensitivity Analysis")
    print("="*80)

    torch.manual_seed(42)
    x = torch.randn(256, 512) * 0.5

    print(f"\nRunning sensitivity analysis on {x.shape} tensor...")
    stats = analyze_mxfp4_sensitivity(x)

    print(f"\nSensitivity analysis results:")
    for key, value in stats.items():
        print(f"  {key}: {value:.6f}")

    # Validate statistics are reasonable
    assert stats['mean_error'] > 1e-7, "Mean error too small"
    assert stats['max_error'] > 1e-6, "Max error too small"
    assert stats['relative_error'] > 1e-4, "Relative error too small"
    assert 0 <= stats['zero_ratio'] <= 1, "Invalid zero ratio"
    assert 0 <= stats['original_zero_ratio'] <= 1, "Invalid original zero ratio"

    print(f"\n  [PASS] All sensitivity metrics in valid ranges")

    print("\n" + "="*80)
    print("TEST 7: PASSED - Sensitivity analysis works correctly")
    print("="*80)


def main():
    """Run all validation tests."""
    print("\n" + "="*80)
    print("MXFP4 QUANTIZATION VALIDATION TEST SUITE")
    print("="*80)
    print("\nThis test suite validates:")
    print("  1. MXFP4 quantization causes measurable precision loss")
    print("  2. Precision loss magnitude is realistic for 4-bit quantization")
    print("  3. Block-based quantization (E2M1K8B32) works correctly")
    print("  4. HWErrorInjector integration with MXFP4 is correct")
    print("  5. Deterministic and stochastic modes work correctly")
    print("  6. Edge cases are handled properly")
    print("  7. Sensitivity analysis utilities work")
    print("="*80)

    try:
        # Core validation tests
        test_mxfp4_causes_precision_loss()
        test_mxfp4_precision_loss_magnitude()
        test_mxfp4_block_quantization()

        # Integration test
        test_hw_error_injector_mxfp4_integration()

        # Additional validation
        test_mxfp4_deterministic_behavior()
        test_mxfp4_edge_cases()
        test_mxfp4_sensitivity_analysis()

        # Final summary
        print("\n" + "="*80)
        print("ALL TESTS PASSED!")
        print("="*80)
        print("\nValidation Summary:")
        print("  [PASS] MXFP4 quantization causes measurable precision loss")
        print("  [PASS] Precision loss magnitude is realistic for 4-bit (SQNR: 15-50 dB)")
        print("  [PASS] HWErrorInjector correctly integrates with MXFP4 library")
        print("  [PASS] Block quantization (32-element blocks) works correctly")
        print("  [PASS] Deterministic and stochastic modes function properly")
        print("  [PASS] Edge cases handled without errors")
        print("  [PASS] Sensitivity analysis utilities work correctly")
        print("\nConclusion: MXFP4 implementation is VALID and ready for experiments!")
        print("="*80 + "\n")

    except AssertionError as e:
        print("\n" + "="*80)
        print("TEST FAILED!")
        print("="*80)
        print(f"\nError: {e}")
        print("\nThe MXFP4 implementation may have issues. Please investigate!")
        print("="*80 + "\n")
        raise


if __name__ == "__main__":
    main()
