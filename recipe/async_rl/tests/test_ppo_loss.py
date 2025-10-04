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
Unit tests for decoupled PPO loss.
"""

import pytest
import torch

from recipe.async_rl.ppo_loss import decoupled_ppo_loss


class TestDecoupledPPOLoss:
    """Test decoupled PPO loss computation."""

    def test_standard_ppo_mode(self):
        """Test that decoupled loss reduces to standard PPO when proximal=old."""
        batch_size = 4
        seq_len = 10

        # Create dummy data
        logprobs = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        proximal_logprobs = old_logprobs.clone()  # Same as old
        advantages = torch.randn(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len)

        # Compute loss with decoupled=False (standard PPO)
        loss_standard, stats_standard = decoupled_ppo_loss(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            use_decoupled_loss=False,
        )

        # Compute loss with decoupled=True but proximal=old (should be same)
        loss_decoupled, stats_decoupled = decoupled_ppo_loss(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            use_decoupled_loss=True,
        )

        # Losses should be identical when proximal=old
        # (behav_imp_weight = exp(proximal - old) = exp(0) = 1)
        assert torch.allclose(loss_standard, loss_decoupled, atol=1e-6)

    def test_importance_weighting(self):
        """Test that importance weighting is applied correctly."""
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
            use_decoupled_loss=True,
        )

        # Behavior importance weight should be exp(0 - (-1)) = exp(1) ≈ 2.718
        expected_behav_weight = torch.exp(proximal_logprobs - old_logprobs)
        assert torch.allclose(stats["behave_imp_weight"], expected_behav_weight, atol=1e-5)

    def test_importance_weight_capping(self):
        """Test that importance weight capping works."""
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
            use_decoupled_loss=True,
        )

        # Without cap, weight would be ~148, but should be zeroed out due to cap
        # behave_mask should exclude these tokens
        assert stats["behave_mask"].sum() == 0  # All tokens exceed cap

    def test_loss_mask_application(self):
        """Test that loss mask is applied correctly."""
        batch_size = 2
        seq_len = 5

        logprobs = torch.randn(batch_size, seq_len)
        proximal_logprobs = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        advantages = torch.ones(batch_size, seq_len)

        # Only first 3 positions are valid
        loss_mask = torch.zeros(batch_size, seq_len)
        loss_mask[:, :3] = 1.0

        loss, stats = decoupled_ppo_loss(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            use_decoupled_loss=False,
        )

        # Check that importance weights are zero for masked positions
        assert (stats["importance_weight"][:, 3:] == 0).all()

    def test_staleness_metrics(self):
        """Test that staleness metrics are computed."""
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
            use_decoupled_loss=True,
        )

        # Should have staleness metrics
        assert "staleness_fraction" in stats
        assert "max_behav_imp_weight" in stats
        assert "mean_behav_imp_weight" in stats

        # About half should be stale
        assert 0.4 < stats["staleness_fraction"] < 0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
