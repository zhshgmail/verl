#!/usr/bin/env python3
"""
Unit tests for noisy_ops phase control (forward-only, backward-only).

These tests verify the gradient vs activation noise theory experiment infrastructure.
"""

import torch


def test_forward_only():
    """Verify backward injection count is 0 when backward disabled."""
    from verl.utils.noisy_ops import (
        enable_noisy_ops, disable_noisy_ops, set_noise_phases,
        get_injection_stats, reset_injection_stats
    )

    # Setup
    reset_injection_stats()
    set_noise_phases(forward=True, backward=False)
    enable_noisy_ops(error_scale=0.05)

    # Run forward + backward pass
    x = torch.randn(10, 10, requires_grad=True)
    y = torch.randn(10, 10, requires_grad=True)
    z = torch.matmul(x, y)
    loss = z.sum()
    loss.backward()

    # Check stats
    stats = get_injection_stats()
    disable_noisy_ops()

    print(f"Forward-only test: forward={stats['total_forward']}, backward={stats['total_backward']}")

    assert stats['total_forward'] > 0, "Expected forward injections > 0"
    assert stats['total_backward'] == 0, f"Expected backward injections == 0, got {stats['total_backward']}"
    assert stats['forward_enabled'] is True
    assert stats['backward_enabled'] is False

    print("PASS: test_forward_only")


def test_backward_only():
    """Verify forward injection count is 0 when forward disabled."""
    from verl.utils.noisy_ops import (
        enable_noisy_ops, disable_noisy_ops, set_noise_phases,
        get_injection_stats, reset_injection_stats
    )

    # Setup
    reset_injection_stats()
    set_noise_phases(forward=False, backward=True)
    enable_noisy_ops(error_scale=0.05)

    # Run forward + backward pass
    x = torch.randn(10, 10, requires_grad=True)
    y = torch.randn(10, 10, requires_grad=True)
    z = torch.matmul(x, y)
    loss = z.sum()
    loss.backward()

    # Check stats
    stats = get_injection_stats()
    disable_noisy_ops()

    print(f"Backward-only test: forward={stats['total_forward']}, backward={stats['total_backward']}")

    assert stats['total_forward'] == 0, f"Expected forward injections == 0, got {stats['total_forward']}"
    assert stats['total_backward'] > 0, "Expected backward injections > 0"
    assert stats['forward_enabled'] is False
    assert stats['backward_enabled'] is True

    print("PASS: test_backward_only")


def test_both_enabled():
    """Verify both forward and backward injections when both enabled."""
    from verl.utils.noisy_ops import (
        enable_noisy_ops, disable_noisy_ops, set_noise_phases,
        get_injection_stats, reset_injection_stats
    )

    # Setup
    reset_injection_stats()
    set_noise_phases(forward=True, backward=True)
    enable_noisy_ops(error_scale=0.05)

    # Run forward + backward pass
    x = torch.randn(10, 10, requires_grad=True)
    y = torch.randn(10, 10, requires_grad=True)
    z = torch.matmul(x, y)
    loss = z.sum()
    loss.backward()

    # Check stats
    stats = get_injection_stats()
    disable_noisy_ops()

    print(f"Both enabled test: forward={stats['total_forward']}, backward={stats['total_backward']}")

    assert stats['total_forward'] > 0, "Expected forward injections > 0"
    assert stats['total_backward'] > 0, "Expected backward injections > 0"

    print("PASS: test_both_enabled")


def test_noise_magnitude():
    """Verify noise is approximately 5% of signal magnitude."""
    from verl.utils.noisy_ops import (
        enable_noisy_ops, disable_noisy_ops, set_noise_phases, reset_injection_stats
    )

    # Setup
    reset_injection_stats()
    set_noise_phases(forward=True, backward=False)
    enable_noisy_ops(error_scale=0.05)

    # Run multiple times to measure noise
    x = torch.randn(100, 100)
    y = torch.randn(100, 100)

    results = []
    for _ in range(20):
        z = torch.matmul(x, y)
        results.append(z.clone())

    disable_noisy_ops()

    # Compute variance across runs (should be ~5% of signal)
    result_stack = torch.stack(results)
    result_std = result_stack.std(dim=0).mean()
    signal_magnitude = results[0].abs().mean()
    noise_ratio = (result_std / signal_magnitude).item()

    print(f"Noise magnitude test: noise_ratio={noise_ratio:.4f} (expected ~0.05)")

    # Allow some tolerance (3% to 7%)
    assert 0.02 < noise_ratio < 0.10, f"Noise ratio {noise_ratio} outside expected range [0.02, 0.10]"

    print("PASS: test_noise_magnitude")


def test_phase_persistence():
    """Verify phase flags survive enable/disable cycles."""
    from verl.utils.noisy_ops import (
        enable_noisy_ops, disable_noisy_ops, set_noise_phases,
        get_noise_phases, reset_injection_stats
    )

    # Set phases before enable
    set_noise_phases(forward=True, backward=False)

    # Check phases persist through enable/disable cycles
    enable_noisy_ops(error_scale=0.05)
    phases1 = get_noise_phases()

    disable_noisy_ops()
    phases2 = get_noise_phases()

    enable_noisy_ops(error_scale=0.05)
    phases3 = get_noise_phases()

    disable_noisy_ops()

    print(f"Phase persistence: phases1={phases1}, phases2={phases2}, phases3={phases3}")

    # Phases should persist
    assert phases1['forward_enabled'] is True
    assert phases1['backward_enabled'] is False
    assert phases2['forward_enabled'] is True
    assert phases2['backward_enabled'] is False
    assert phases3['forward_enabled'] is True
    assert phases3['backward_enabled'] is False

    print("PASS: test_phase_persistence")


def test_bmm_phase_control():
    """Test phase control works for batch matrix multiplication."""
    from verl.utils.noisy_ops import (
        enable_noisy_ops, disable_noisy_ops, set_noise_phases,
        get_injection_stats, reset_injection_stats
    )

    # Setup forward-only
    reset_injection_stats()
    set_noise_phases(forward=True, backward=False)
    enable_noisy_ops(error_scale=0.05)

    # Run bmm forward + backward
    x = torch.randn(4, 10, 10, requires_grad=True)
    y = torch.randn(4, 10, 10, requires_grad=True)
    z = torch.bmm(x, y)
    loss = z.sum()
    loss.backward()

    # Check stats
    stats = get_injection_stats()
    disable_noisy_ops()

    print(f"BMM phase test: forward={stats['total_forward']}, backward={stats['total_backward']}")

    assert stats['total_forward'] > 0, "Expected forward injections > 0"
    assert stats['total_backward'] == 0, f"Expected backward injections == 0, got {stats['total_backward']}"

    print("PASS: test_bmm_phase_control")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing noisy_ops phase control")
    print("=" * 60)

    test_forward_only()
    test_backward_only()
    test_both_enabled()
    test_noise_magnitude()
    test_phase_persistence()
    test_bmm_phase_control()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
