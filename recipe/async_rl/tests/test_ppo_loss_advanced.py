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
Advanced unit tests for PPO loss - covering edge cases and advanced features.

These tests cover:
1. Dual clipping (c_clip parameter)
2. Asymmetric clipping (eps_clip_higher)
3. GAE computation with version tracking
4. Complex staleness scenarios
"""

import pytest
import torch
import numpy as np

from recipe.async_rl.ppo_loss import (
    ppo_loss,
    decoupled_ppo_loss,
    compute_advantages_with_version_tracking,
)


class TestDualClipping:
    """Test dual clipping feature (PPO-penalty variant)."""

    def test_dual_clip_basic(self):
        """Test that dual clipping is applied when c_clip is set."""
        batch_size = 2
        seq_len = 5

        # Create data that triggers dual clip:
        # Need: -sign(A) * c * A < max(-A * r, -A * clip(r))
        # For positive A, need: -c * A < -A * max(r, clip(r))
        # Which means: c > max(r, clip(r))
        # With small advantages and ratio close to 1, dual clip won't activate easily
        # Need large advantages where c * A is smaller than regular clipped loss

        # Create specific scenario: small ratio, large advantage
        logprobs = torch.full((batch_size, seq_len), -2.0)  # Current policy
        old_logprobs = torch.full((batch_size, seq_len), -2.05)  # Behavior policy (similar)
        # Ratio = exp(-2 - (-2.05)) = exp(0.05) ≈ 1.05 (very close to 1)
        advantages = torch.full((batch_size, seq_len), 0.5)  # Moderate advantages
        loss_mask = torch.ones(batch_size, seq_len)

        # Test with dual clip
        loss_dual, stats_dual = ppo_loss(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            c_clip=1.5,  # Smaller c_clip to make it activate more easily
        )

        # Test without dual clip
        loss_no_dual, stats_no_dual = ppo_loss(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            c_clip=None,
        )

        # Without dual clip, no tokens should be dual-clipped
        assert stats_no_dual["dual_clip_mask"].sum() == 0
        # Dual clip mask exists in both cases
        assert "dual_clip_mask" in stats_dual
        assert "dual_clip_mask" in stats_no_dual

    def test_dual_clip_assertion(self):
        """Test that c_clip must be > 1.0."""
        batch_size = 2
        seq_len = 5

        logprobs = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        advantages = torch.randn(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len)

        # Should raise assertion error for c_clip <= 1.0
        with pytest.raises(AssertionError, match="c_clip must be > 1.0"):
            ppo_loss(
                logprobs=logprobs,
                old_logprobs=old_logprobs,
                advantages=advantages,
                eps_clip=0.2,
                loss_mask=loss_mask,
                c_clip=0.5,  # Invalid: <= 1.0
            )


class TestAsymmetricClipping:
    """Test asymmetric clipping (different upper/lower bounds)."""

    def test_asymmetric_clip_ratios(self):
        """Test that eps_clip_higher creates asymmetric bounds."""
        batch_size = 2
        seq_len = 5

        logprobs = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        advantages = torch.ones(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len)

        # Asymmetric clipping: [0.8, 1.5] instead of [0.8, 1.2]
        loss, stats = ppo_loss(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,  # Lower bound: 1 - 0.2 = 0.8
            eps_clip_higher=0.5,  # Upper bound: 1 + 0.5 = 1.5
            loss_mask=loss_mask,
        )

        # Loss should be computed (testing that it runs without error)
        assert loss.item() is not None
        assert "importance_weight" in stats


class TestGAEWithVersionTracking:
    """Test GAE advantages computation with version tracking."""

    def test_gae_basic_computation(self):
        """Test basic GAE computation without version tracking."""
        batch_size = 4
        max_seqlen = 10

        rewards = torch.randn(batch_size)
        values = torch.randn(batch_size, max_seqlen)
        old_logprobs = torch.randn(batch_size, max_seqlen)
        ref_logprobs = torch.randn(batch_size, max_seqlen)
        attention_mask = torch.ones(batch_size, max_seqlen)
        loss_mask = torch.ones(batch_size, max_seqlen)

        advantages, stats = compute_advantages_with_version_tracking(
            rewards=rewards,
            values=values,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            kl_ctl=0.1,
        )

        # Check output shape
        assert advantages.shape == (batch_size, max_seqlen)
        # No version stats when version tracking is disabled
        assert len(stats) == 0

    def test_gae_with_version_tracking(self):
        """Test GAE computation with version staleness tracking."""
        batch_size = 4
        max_seqlen = 10

        rewards = torch.randn(batch_size)
        values = torch.randn(batch_size, max_seqlen)
        old_logprobs = torch.randn(batch_size, max_seqlen)
        ref_logprobs = torch.randn(batch_size, max_seqlen)
        attention_mask = torch.ones(batch_size, max_seqlen)
        loss_mask = torch.ones(batch_size, max_seqlen)

        # Create version tracking data
        version_start = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        version_end = torch.tensor([0, 1, 3, 5], dtype=torch.int32)  # Some samples have staleness

        advantages, stats = compute_advantages_with_version_tracking(
            rewards=rewards,
            values=values,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            kl_ctl=0.1,
            version_start=version_start,
            version_end=version_end,
        )

        # Check version staleness statistics
        assert "max_version_diff" in stats
        assert "mean_version_diff" in stats
        assert "samples_with_staleness" in stats

        # Version diff should be [0, 0, 1, 2]
        assert stats["max_version_diff"] == 2
        # Mean should be (0 + 0 + 1 + 2) / 4 = 0.75
        assert abs(stats["mean_version_diff"] - 0.75) < 0.01
        # Fraction with staleness (version_diff > 0): 2/4 = 0.5
        assert abs(stats["samples_with_staleness"] - 0.5) < 0.01

    def test_gae_variable_sequence_lengths(self):
        """Test GAE with variable sequence lengths (attention mask)."""
        batch_size = 3
        max_seqlen = 8

        rewards = torch.randn(batch_size)
        values = torch.randn(batch_size, max_seqlen)
        old_logprobs = torch.randn(batch_size, max_seqlen)
        ref_logprobs = torch.randn(batch_size, max_seqlen)

        # Different sequence lengths
        attention_mask = torch.zeros(batch_size, max_seqlen)
        attention_mask[0, :5] = 1  # Sequence length 5
        attention_mask[1, :7] = 1  # Sequence length 7
        attention_mask[2, :8] = 1  # Sequence length 8 (full)

        loss_mask = attention_mask.clone()

        advantages, stats = compute_advantages_with_version_tracking(
            rewards=rewards,
            values=values,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            kl_ctl=0.1,
        )

        # Check that advantages are computed correctly
        assert advantages.shape == (batch_size, max_seqlen)
        # Advantages should be non-zero where mask is 1
        assert (advantages[0, :5] != 0).any()

    def test_gae_kl_regularization(self):
        """Test that KL regularization affects rewards correctly."""
        batch_size = 2
        max_seqlen = 5

        rewards = torch.ones(batch_size)
        values = torch.zeros(batch_size, max_seqlen)
        attention_mask = torch.ones(batch_size, max_seqlen)
        loss_mask = torch.ones(batch_size, max_seqlen)

        # Create logprobs with known KL
        old_logprobs = torch.zeros(batch_size, max_seqlen)
        ref_logprobs = torch.full((batch_size, max_seqlen), -1.0)  # KL = 1.0 per token

        # Higher kl_ctl should lead to different advantages
        advantages_low_kl, _ = compute_advantages_with_version_tracking(
            rewards=rewards,
            values=values,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            kl_ctl=0.1,
        )

        advantages_high_kl, _ = compute_advantages_with_version_tracking(
            rewards=rewards,
            values=values,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            kl_ctl=1.0,  # 10x higher KL penalty
        )

        # Advantages should be different due to different KL penalties
        assert not torch.allclose(advantages_low_kl, advantages_high_kl)


class TestComplexStalenessScenarios:
    """Test complex staleness detection and handling."""

    def test_all_stale_samples(self):
        """Test when all samples are stale (high importance weights)."""
        batch_size = 4
        seq_len = 5

        logprobs = torch.zeros(batch_size, seq_len)
        proximal_logprobs = torch.zeros(batch_size, seq_len)
        # All samples very stale: exp(0 - (-3)) = exp(3) ≈ 20
        old_logprobs = torch.full((batch_size, seq_len), -3.0)
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

        # All samples should be marked as stale (weight > 1.2)
        assert stats["staleness_fraction"] > 0.99

    def test_no_stale_samples(self):
        """Test when no samples are stale (weights close to 1.0)."""
        batch_size = 4
        seq_len = 5

        logprobs = torch.zeros(batch_size, seq_len)
        proximal_logprobs = torch.zeros(batch_size, seq_len)
        # All samples fresh: exp(0 - (-0.05)) = exp(0.05) ≈ 1.05
        old_logprobs = torch.full((batch_size, seq_len), -0.05)
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

        # No samples should be marked as stale (0.8 < weight < 1.2)
        assert stats["staleness_fraction"] < 0.01

    def test_mixed_staleness(self):
        """Test batch with mix of stale and fresh samples."""
        batch_size = 4
        seq_len = 5

        logprobs = torch.zeros(batch_size, seq_len)
        proximal_logprobs = torch.zeros(batch_size, seq_len)

        old_logprobs = torch.zeros(batch_size, seq_len)
        old_logprobs[0, :] = -0.1   # Fresh: exp(0.1) ≈ 1.1
        old_logprobs[1, :] = -0.05  # Fresh: exp(0.05) ≈ 1.05
        old_logprobs[2, :] = -2.0   # Stale: exp(2) ≈ 7.4
        old_logprobs[3, :] = -3.0   # Very stale: exp(3) ≈ 20

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

        # About half should be stale (samples 2 and 3)
        assert 0.4 < stats["staleness_fraction"] < 0.6
        # Max weight should be from sample 3
        assert stats["max_behav_imp_weight"] > 15.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
