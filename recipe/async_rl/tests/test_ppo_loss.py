# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
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
"""
Unit tests for PPO loss functions (standard and decoupled).
"""

import pytest
import torch

from recipe.async_rl.ppo_loss import ppo_loss, decoupled_ppo_loss


class TestPPOLoss:
    """Test PPO loss functions."""

    def test_standard_ppo_vs_decoupled_same_logprobs(self):
        """Test that ppo_loss and decoupled_ppo_loss give same result when proximal=old."""
        batch_size = 4
        seq_len = 10

        # Create dummy data
        logprobs = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        advantages = torch.randn(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len)

        # Standard PPO
        loss_standard, stats_standard = ppo_loss(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
        )

        # Decoupled PPO with proximal=old (should reduce to standard PPO)
        loss_decoupled, stats_decoupled = decoupled_ppo_loss(
            logprobs=logprobs,
            proximal_logprobs=old_logprobs,  # Same as old
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
        )

        # Losses should be identical when proximal=old
        # (behav_imp_weight = exp(proximal - old) = exp(0) = 1)
        assert torch.allclose(loss_standard, loss_decoupled, atol=1e-6)

    def test_importance_weighting(self):
        """Test that importance weighting is applied correctly in decoupled PPO."""
        batch_size = 2
        seq_len = 5

        logprobs = torch.zeros(batch_size, seq_len)  # Current policy
        proximal_logprobs = torch.zeros(batch_size, seq_len)  # Proximal policy
        old_logprobs = torch.full((batch_size, seq_len), -1.0)  # Behavior policy (different)
        advantages = torch.ones(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len)

        loss, stats = decoupled_ppo_loss(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
        )

        # Behavior importance weight should be exp(0 - (-1)) = exp(1) ≈ 2.718
        expected_behav_weight = torch.exp(proximal_logprobs - old_logprobs)
        assert torch.allclose(stats["behave_imp_weight"], expected_behav_weight, atol=1e-5)

    def test_importance_weight_capping(self):
        """Test that importance weight upper bound works in decoupled PPO."""
        batch_size = 2
        seq_len = 5

        logprobs = torch.zeros(batch_size, seq_len)
        proximal_logprobs = torch.zeros(batch_size, seq_len)
        # Old policy is much worse: exp(0 - (-5)) = exp(5) ≈ 148
        old_logprobs = torch.full((batch_size, seq_len), -5.0)
        advantages = torch.ones(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len)

        # Cap at 10.0
        loss, stats = decoupled_ppo_loss(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            behav_imp_weight_cap=10.0,
        )

        # Without cap, weight would be ~148, but should be zeroed out due to cap
        # behave_mask should exclude these tokens
        assert stats["behave_mask"].sum() == 0  # All tokens exceed cap

    def test_importance_weight_floor(self):
        """Test that importance weight lower bound works in decoupled PPO."""
        batch_size = 2
        seq_len = 5

        logprobs = torch.zeros(batch_size, seq_len)
        proximal_logprobs = torch.zeros(batch_size, seq_len)
        # Proximal policy is much worse: exp(0 - 5) = exp(-5) ≈ 0.0067
        old_logprobs = torch.full((batch_size, seq_len), 5.0)
        advantages = torch.ones(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len)

        # Floor at 0.1
        loss, stats = decoupled_ppo_loss(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            behav_imp_weight_floor=0.1,
        )

        # Without floor, weight would be ~0.0067, but should be zeroed out due to floor
        # behave_mask should exclude these tokens
        assert stats["behave_mask"].sum() == 0  # All tokens below floor

    def test_symmetric_importance_weight_bounds(self):
        """Test symmetric bounds (both cap and floor)."""
        batch_size = 4
        seq_len = 5

        logprobs = torch.zeros(batch_size, seq_len)
        proximal_logprobs = torch.zeros(batch_size, seq_len)

        # Create samples with different importance weights
        old_logprobs = torch.zeros(batch_size, seq_len)
        old_logprobs[0, :] = -1.5  # exp(0 - (-1.5)) = exp(1.5) ≈ 4.48 (within [0.2, 5.0])
        old_logprobs[1, :] = -3.0  # exp(0 - (-3)) = exp(3) ≈ 20.09 (exceeds cap 5.0)
        old_logprobs[2, :] = 2.0   # exp(0 - 2) = exp(-2) ≈ 0.135 (below floor 0.2)
        old_logprobs[3, :] = -1.0  # exp(0 - (-1)) = exp(1) ≈ 2.72 (within [0.2, 5.0])

        advantages = torch.ones(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len)

        # Symmetric bounds: [0.2, 5.0]
        loss, stats = decoupled_ppo_loss(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            behav_imp_weight_cap=5.0,
            behav_imp_weight_floor=0.2,
        )

        # Only samples 0 and 3 should pass the bounds
        expected_valid_tokens = 2 * seq_len  # 2 samples * 5 tokens each
        assert stats["behave_mask"].sum() == expected_valid_tokens

    def test_loss_mask_application(self):
        """Test that loss mask is applied correctly in standard PPO."""
        batch_size = 2
        seq_len = 5

        logprobs = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        advantages = torch.ones(batch_size, seq_len)

        # Only first 3 positions are valid
        loss_mask = torch.zeros(batch_size, seq_len)
        loss_mask[:, :3] = 1.0

        loss, stats = ppo_loss(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
        )

        # Check that importance weights are zero for masked positions
        assert (stats["importance_weight"][:, 3:] == 0).all()

    def test_staleness_metrics(self):
        """Test that staleness metrics are computed in decoupled PPO."""
        batch_size = 2
        seq_len = 5

        logprobs = torch.zeros(batch_size, seq_len)
        proximal_logprobs = torch.zeros(batch_size, seq_len)
        # Make half the samples stale (weight > 1.2)
        old_logprobs = torch.zeros(batch_size, seq_len)
        old_logprobs[0, :] = -0.5  # exp(0 - (-0.5)) = exp(0.5) ≈ 1.65 > 1.2 (stale)
        old_logprobs[1, :] = -0.1  # exp(0 - (-0.1)) = exp(0.1) ≈ 1.1 < 1.2 (fresh)

        advantages = torch.ones(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len)

        loss, stats = decoupled_ppo_loss(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
        )

        # Should have staleness metrics
        assert "staleness_fraction" in stats
        assert "max_behav_imp_weight" in stats
        assert "mean_behav_imp_weight" in stats

        # About half should be stale
        assert 0.4 < stats["staleness_fraction"] < 0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
