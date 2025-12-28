# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for noise injection module."""

import pytest
import torch
import torch.nn as nn


class MockRMSNorm(nn.Module):
    """Mock RMSNorm layer for testing."""

    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        return x


class MockExpertMLP(nn.Module):
    """Mock MLP expert with RMSNorm."""

    def __init__(self, hidden_size):
        super().__init__()
        self.norm = MockRMSNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)


class MockMoELayer(nn.Module):
    """Mock MoE layer with router and experts."""

    def __init__(self, hidden_size, num_experts=8):
        super().__init__()
        self.router = nn.Linear(hidden_size, num_experts)
        self.router_norm = MockRMSNorm(hidden_size)  # Should NOT receive noise
        self.experts = nn.ModuleList([MockExpertMLP(hidden_size) for _ in range(num_experts)])


class MockMoEModel(nn.Module):
    """Mock MoE model for testing."""

    def __init__(self, hidden_size=768, num_layers=2, num_experts=8):
        super().__init__()
        self.embed_norm = MockRMSNorm(hidden_size)  # Should NOT receive noise
        self.layers = nn.ModuleList([MockMoELayer(hidden_size, num_experts) for _ in range(num_layers)])
        self.final_norm = MockRMSNorm(hidden_size)  # Should NOT receive noise


def test_sigma_schedule():
    """Test sigma schedule generation."""
    from verl.utils.noise_injection import get_sigma_schedule, get_sigma_by_step

    # Test exponential decay
    schedule = get_sigma_schedule(sigma_start=0.01, sigma_end=0.001, num_stages=10)
    assert len(schedule) == 9  # num_stages - 1
    assert schedule[0] == pytest.approx(0.01, abs=1e-6)
    assert schedule[-1] == pytest.approx(0.001, abs=1e-6)

    # Test monotonic decrease
    for i in range(len(schedule) - 1):
        assert schedule[i] > schedule[i + 1]

    # Test get_sigma_by_step
    total_steps = 1000
    sigma_id, sigma = get_sigma_by_step(0, total_steps, schedule)
    assert sigma_id == 0
    assert sigma == 0  # First interval has no noise

    sigma_id, sigma = get_sigma_by_step(500, total_steps, schedule)
    assert sigma > 0


def test_expert_noise_targeting():
    """Test that noise is applied only to expert RMSNorm layers."""
    from verl.utils.noise_injection import generate_expert_gaussian_noise, get_sigma_schedule

    # Create mock model
    model = MockMoEModel(hidden_size=64, num_layers=2, num_experts=4)

    # Store original weights
    original_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, MockRMSNorm):
            original_weights[name] = module.weight.data.clone()

    # Apply noise
    sigma_trend = get_sigma_schedule(0.01, 0.001, 10)
    generate_expert_gaussian_noise(
        model,
        step=500,  # Mid-training
        total_step=1000,
        sigma_trend=sigma_trend,
        target_modules=["experts"],
        verbose=False
    )

    # Check that expert RMSNorm layers received noise
    expert_norms_changed = 0
    non_expert_norms_unchanged = 0

    for name, module in model.named_modules():
        if isinstance(module, MockRMSNorm):
            weight_changed = not torch.allclose(module.weight.data, original_weights[name])

            if "experts" in name:
                # Expert RMSNorm should have changed
                assert weight_changed, f"Expert norm {name} should receive noise"
                expert_norms_changed += 1
            elif "router" in name or "embed" in name or "final" in name:
                # Router and non-expert norms should NOT have changed
                assert not weight_changed, f"Non-expert norm {name} should NOT receive noise"
                non_expert_norms_unchanged += 1

    assert expert_norms_changed > 0, "At least some expert norms should receive noise"
    assert non_expert_norms_unchanged > 0, "Non-expert norms should remain unchanged"


def test_router_exclusion():
    """Test that router parameters are never affected by noise."""
    from verl.utils.noise_injection import generate_expert_gaussian_noise, get_sigma_schedule

    model = MockMoEModel(hidden_size=64, num_layers=2, num_experts=4)

    # Store original router weights
    original_router_weights = {}
    for name, module in model.named_modules():
        if "router" in name and isinstance(module, (nn.Linear, MockRMSNorm)):
            if hasattr(module, 'weight'):
                original_router_weights[name] = module.weight.data.clone()

    # Apply noise with explicit router exclusion
    sigma_trend = get_sigma_schedule(0.01, 0.001, 10)
    generate_expert_gaussian_noise(
        model,
        step=500,
        total_step=1000,
        sigma_trend=sigma_trend,
        target_modules=["experts"],
        verbose=False
    )

    # Verify router weights unchanged
    for name, original_weight in original_router_weights.items():
        current_weight = None
        for mod_name, module in model.named_modules():
            if mod_name == name and hasattr(module, 'weight'):
                current_weight = module.weight.data
                break

        if current_weight is not None:
            assert torch.allclose(current_weight, original_weight), \
                f"Router parameter {name} should NOT be modified by noise injection"


def test_noise_zero_at_start():
    """Test that no noise is applied in the first interval (step 0)."""
    from verl.utils.noise_injection import generate_expert_gaussian_noise, get_sigma_schedule

    model = MockMoEModel(hidden_size=64, num_layers=1, num_experts=2)

    # Store original weights
    original_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, MockRMSNorm):
            original_weights[name] = module.weight.data.clone()

    # Apply noise at step 0
    sigma_trend = get_sigma_schedule(0.01, 0.001, 10)
    generate_expert_gaussian_noise(
        model,
        step=0,
        total_step=1000,
        sigma_trend=sigma_trend,
        verbose=False
    )

    # All weights should remain unchanged
    for name, module in model.named_modules():
        if isinstance(module, MockRMSNorm):
            assert torch.allclose(module.weight.data, original_weights[name]), \
                f"No noise should be applied at step 0, but {name} was modified"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
