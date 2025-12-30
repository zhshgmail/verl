#!/usr/bin/env python3
"""
Test HW Error Injection module.

Run: python tests/test_hw_error_injection.py
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/home/zheng/workspace/verl')

from verl.utils.hw_error_injection import (
    HWErrorConfig,
    HWErrorInjector,
    create_hw_error_injector,
    inject_hw_error_once,
)


class SimpleRMSNorm(nn.Module):
    """Simple RMSNorm for testing."""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class SimpleTransformerBlock(nn.Module):
    """Simple transformer block for testing."""
    def __init__(self, hidden_size=256, intermediate_size=512):
        super().__init__()
        self.input_layernorm = SimpleRMSNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.post_attention_layernorm = SimpleRMSNorm(hidden_size)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        # Attention
        h = self.input_layernorm(x)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        # Simplified attention (no actual attention computation)
        attn_out = self.o_proj(q + k + v)
        x = x + attn_out

        # MLP
        h = self.post_attention_layernorm(x)
        gate = torch.nn.functional.silu(self.gate_proj(h))
        up = self.up_proj(h)
        mlp_out = self.down_proj(gate * up)
        x = x + mlp_out

        return x


def test_basic_error_injection():
    """Test basic error injection on a tensor."""
    print("\n=== Test 1: Basic Error Injection ===")

    x = torch.randn(32, 256)

    for error_type in ['relative_gaussian', 'absolute_gaussian', 'systematic_bias']:
        x_noisy = inject_hw_error_once(x, error_scale=1e-4, error_type=error_type)
        rel_error = (x_noisy - x).abs().mean() / x.abs().mean()
        print(f"  {error_type}: relative error = {rel_error:.2e}")

    print("  PASSED")


def test_hook_registration():
    """Test hook registration on model."""
    print("\n=== Test 2: Hook Registration ===")

    model = SimpleTransformerBlock()

    # Test with RMSNorm only
    config = HWErrorConfig(enabled=True, error_scale=1e-5, target_modules=['rmsnorm'])
    injector = HWErrorInjector(config)
    count = injector.register_hooks(model, verbose=False)
    print(f"  RMSNorm target: {count} hooks registered")
    assert count == 2, f"Expected 2 hooks (input_layernorm, post_attention_layernorm), got {count}"
    injector.remove_hooks()

    # Test with Linear (down_proj)
    config = HWErrorConfig(enabled=True, error_scale=1e-5, target_modules=['down_proj'])
    injector = HWErrorInjector(config)
    count = injector.register_hooks(model, verbose=False)
    print(f"  down_proj target: {count} hooks registered")
    assert count == 1, f"Expected 1 hook (down_proj), got {count}"
    injector.remove_hooks()

    # Test with multiple targets
    config = HWErrorConfig(enabled=True, error_scale=1e-5, target_modules=['rmsnorm', 'down_proj', 'o_proj'])
    injector = HWErrorInjector(config)
    count = injector.register_hooks(model, verbose=False)
    print(f"  Multiple targets: {count} hooks registered")
    assert count == 4, f"Expected 4 hooks, got {count}"
    injector.remove_hooks()

    print("  PASSED")


def test_forward_with_injection():
    """Test forward pass with error injection."""
    print("\n=== Test 3: Forward Pass with Injection ===")

    model = SimpleTransformerBlock()
    x = torch.randn(2, 16, 256)  # (batch, seq, hidden)

    # Run without injection
    with torch.no_grad():
        y_clean = model(x.clone())

    # Run with injection
    config = HWErrorConfig(enabled=True, error_scale=1e-4, target_modules=['rmsnorm'])
    injector = HWErrorInjector(config)
    injector.register_hooks(model, verbose=False)

    with torch.no_grad():
        y_noisy = model(x.clone())

    injector.remove_hooks()

    # Check outputs are different
    diff = (y_noisy - y_clean).abs().mean()
    rel_diff = diff / y_clean.abs().mean()
    print(f"  Output difference: {diff:.2e} (relative: {rel_diff:.2e})")
    assert diff > 0, "Outputs should be different with error injection"

    # Run again without injection - should match clean
    with torch.no_grad():
        y_clean2 = model(x.clone())
    diff2 = (y_clean2 - y_clean).abs().mean()
    print(f"  After removing hooks, diff: {diff2:.2e}")
    assert diff2 < 1e-10, "After removing hooks, outputs should match"

    print("  PASSED")


def test_phase_control():
    """Test phase-based injection control."""
    print("\n=== Test 4: Phase Control ===")

    model = SimpleTransformerBlock()
    x = torch.randn(2, 16, 256)

    # Create injector that only applies during 'training'
    config = HWErrorConfig(enabled=True, error_scale=1e-4, target_modules=['rmsnorm'], apply_during='training')
    injector = HWErrorInjector(config)
    injector.register_hooks(model, verbose=False)

    # During 'rollout' phase - should not inject
    injector.set_phase('rollout')
    with torch.no_grad():
        y_rollout = model(x.clone())

    # During 'training' phase - should inject
    injector.set_phase('training')
    with torch.no_grad():
        y_training = model(x.clone())

    injector.remove_hooks()

    # Compare
    diff = (y_training - y_rollout).abs().mean()
    print(f"  Rollout vs Training diff: {diff:.2e}")
    assert diff > 0, "Training phase should have injection, rollout should not"

    print("  PASSED")


def test_statistics():
    """Test injection statistics tracking."""
    print("\n=== Test 5: Statistics Tracking ===")

    model = SimpleTransformerBlock()
    x = torch.randn(2, 16, 256)

    config = HWErrorConfig(enabled=True, error_scale=1e-4, target_modules=['rmsnorm'])
    injector = HWErrorInjector(config)
    injector.register_hooks(model, verbose=False)

    # Run multiple forward passes
    for _ in range(5):
        with torch.no_grad():
            _ = model(x.clone())

    stats = injector.get_stats()
    print(f"  Stats keys: {list(stats.keys())}")
    for name, stat in stats.items():
        print(f"    {name}: count={stat['count']}, mean_error={stat['mean_error']:.2e}")

    injector.remove_hooks()
    print("  PASSED")


def test_error_scales():
    """Test different error scales and their impact."""
    print("\n=== Test 6: Error Scale Impact ===")

    model = SimpleTransformerBlock()
    x = torch.randn(2, 16, 256)

    with torch.no_grad():
        y_clean = model(x.clone())

    for scale in [1e-6, 1e-5, 1e-4, 1e-3]:
        config = HWErrorConfig(enabled=True, error_scale=scale, target_modules=['rmsnorm'])
        injector = HWErrorInjector(config)
        injector.register_hooks(model, verbose=False)

        with torch.no_grad():
            y_noisy = model(x.clone())

        rel_diff = (y_noisy - y_clean).abs().mean() / y_clean.abs().mean()
        print(f"  scale={scale:.0e}: relative output diff = {rel_diff:.2e}")

        injector.remove_hooks()

    print("  PASSED")


def main():
    print("=" * 60)
    print("HW Error Injection Module Tests")
    print("=" * 60)

    test_basic_error_injection()
    test_hook_registration()
    test_forward_with_injection()
    test_phase_control()
    test_statistics()
    test_error_scales()

    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
