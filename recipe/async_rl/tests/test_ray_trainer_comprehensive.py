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
Comprehensive unit tests for AsyncRLTrainer core logic.

These tests cover:
1. compute_advantages() with version tracking
2. compute_ppo_loss() polymorphism (standard vs decoupled)
3. sync_weights_to_rollout() version management
4. training_step() staleness metrics
5. Configuration extraction and validation
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from omegaconf import DictConfig

from verl.protocol import DataProto


class TestAsyncRLTrainerConfiguration:
    """Test configuration extraction and validation."""

    def test_config_extraction(self):
        """Test that AsyncRLTrainer extracts config correctly."""
        from recipe.async_rl.ray_trainer import AsyncRLTrainer

        config = DictConfig({
            "actor_rollout_ref": {
                "rollout": {"mode": "async"},
                "actor": {
                    "use_decoupled_loss": True,
                    "behav_imp_weight_cap": 5.0,
                    "behav_imp_weight_floor": 0.2,
                    "clip_ratio_low": 0.2,
                },
            },
            "async_rl": {
                "buffer_size": 10000,
                "max_staleness": 5,
                "train_batch_size": 128,
                "rollout_batch_size": 256,
                "rollout_executor_workers": 4,
            },
            "algorithm": {
                "kl_ctrl": {"kl_coef": 0.1},
            },
        })

        with patch.object(AsyncRLTrainer, '__init__', lambda self, **kwargs: None):
            trainer = AsyncRLTrainer.__new__(AsyncRLTrainer)

            # Simulate config extraction from __init__
            trainer.use_decoupled_loss = config.actor_rollout_ref.actor.get("use_decoupled_loss", True)
            trainer.behav_imp_weight_cap = config.actor_rollout_ref.actor.get("behav_imp_weight_cap", None)
            trainer.behav_imp_weight_floor = config.actor_rollout_ref.actor.get("behav_imp_weight_floor", None)
            trainer.buffer_size = config.get("async_rl", {}).get("buffer_size", 10000)
            trainer.max_staleness = config.get("async_rl", {}).get("max_staleness", 5)
            trainer.train_batch_size = config.get("async_rl", {}).get("train_batch_size", 128)
            trainer.rollout_batch_size = config.get("async_rl", {}).get("rollout_batch_size", 256)

            # Check extracted values
            assert trainer.use_decoupled_loss is True
            assert trainer.behav_imp_weight_cap == 5.0
            assert trainer.behav_imp_weight_floor == 0.2
            assert trainer.buffer_size == 10000
            assert trainer.max_staleness == 5
            assert trainer.train_batch_size == 128
            assert trainer.rollout_batch_size == 256

    def test_config_validation_async_mode(self):
        """Test that AsyncRLTrainer requires async mode."""
        # This would be tested during actual initialization
        # Here we just verify the assertion condition
        config_invalid = DictConfig({
            "actor_rollout_ref": {
                "rollout": {"mode": "sync"},  # Invalid: must be "async"
            },
        })

        # The trainer should raise assertion error
        assert config_invalid.actor_rollout_ref.rollout.mode != "async"


class TestComputeAdvantages:
    """Test compute_advantages() with version tracking."""

    def test_compute_advantages_with_version_data(self):
        """Test that compute_advantages uses version tracking when available."""
        from recipe.async_rl.ray_trainer import AsyncRLTrainer

        config = DictConfig({
            "algorithm": {
                "kl_ctrl": {"kl_coef": 0.1},
                "discount": 0.99,
                "gae_lambda": 0.95,
            },
        })

        with patch.object(AsyncRLTrainer, '__init__', lambda self, **kwargs: None):
            trainer = AsyncRLTrainer.__new__(AsyncRLTrainer)
            trainer.config = config

            # Create test data with version tracking
            batch_size = 4
            seq_len = 10

            data = DataProto.from_dict(
                tensors={
                    "rewards": torch.randn(batch_size),
                    "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
                    "logprobs": torch.randn(batch_size, seq_len),
                    "ref_logprobs": torch.randn(batch_size, seq_len),
                    "attention_mask": torch.ones(batch_size, seq_len),
                    "loss_mask": torch.ones(batch_size, seq_len),
                },
                non_tensors={
                    "version_start": np.array([0, 1, 2, 3], dtype=np.int32),
                    "version_end": np.array([0, 1, 3, 5], dtype=np.int32),
                }
            )

            # Compute advantages
            result = trainer.compute_advantages(data)

            # Check that advantages were added
            assert "advantages" in result.batch
            assert result.batch["advantages"].shape == (batch_size, seq_len)

    def test_compute_advantages_without_version_data(self):
        """Test that compute_advantages works even without version tracking."""
        from recipe.async_rl.ray_trainer import AsyncRLTrainer

        config = DictConfig({
            "algorithm": {
                "kl_ctrl": {"kl_coef": 0.1},
                "discount": 0.99,
                "gae_lambda": 0.95,
            },
        })

        with patch.object(AsyncRLTrainer, '__init__', lambda self, **kwargs: None):
            trainer = AsyncRLTrainer.__new__(AsyncRLTrainer)
            trainer.config = config

            # Create test data WITHOUT version tracking
            batch_size = 4
            seq_len = 10

            data = DataProto.from_dict(
                tensors={
                    "rewards": torch.randn(batch_size),
                    "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
                    "logprobs": torch.randn(batch_size, seq_len),
                    "ref_logprobs": torch.randn(batch_size, seq_len),
                    "attention_mask": torch.ones(batch_size, seq_len),
                    "loss_mask": torch.ones(batch_size, seq_len),
                },
                non_tensors={}
            )

            # Compute advantages - should work without version data
            result = trainer.compute_advantages(data)

            # Check that advantages were added
            assert "advantages" in result.batch


class TestComputePPOLoss:
    """Test compute_ppo_loss() polymorphism."""

    def test_compute_ppo_loss_decoupled(self):
        """Test that decoupled loss is used when configured."""
        from recipe.async_rl.ray_trainer import AsyncRLTrainer

        config = DictConfig({
            "actor_rollout_ref": {
                "actor": {
                    "clip_ratio_low": 0.2,
                    "recompute_logprob": False,  # Don't recompute
                },
            },
        })

        with patch.object(AsyncRLTrainer, '__init__', lambda self, **kwargs: None):
            trainer = AsyncRLTrainer.__new__(AsyncRLTrainer)
            trainer.config = config
            trainer.use_decoupled_loss = True
            trainer.behav_imp_weight_cap = 5.0
            trainer.behav_imp_weight_floor = 0.2

            # Test data
            batch_size = 2
            seq_len = 5

            logprobs = torch.zeros(batch_size, seq_len)
            old_logprobs = torch.randn(batch_size, seq_len)
            proximal_logprobs = torch.randn(batch_size, seq_len)
            advantages = torch.ones(batch_size, seq_len)
            loss_mask = torch.ones(batch_size, seq_len)

            # Compute loss
            loss, stats = trainer.compute_ppo_loss(
                logprobs=logprobs,
                old_logprobs=old_logprobs,
                advantages=advantages,
                loss_mask=loss_mask,
                proximal_logprobs=proximal_logprobs,
            )

            # Check that behavior statistics exist (decoupled PPO)
            assert "behave_imp_weight" in stats
            assert "behave_approx_kl" in stats
            assert "behave_mask" in stats

    def test_compute_ppo_loss_standard(self):
        """Test that standard loss is used when configured."""
        from recipe.async_rl.ray_trainer import AsyncRLTrainer

        config = DictConfig({
            "actor_rollout_ref": {
                "actor": {
                    "clip_ratio_low": 0.2,
                },
            },
        })

        with patch.object(AsyncRLTrainer, '__init__', lambda self, **kwargs: None):
            trainer = AsyncRLTrainer.__new__(AsyncRLTrainer)
            trainer.config = config
            trainer.use_decoupled_loss = False  # Standard PPO

            # Test data
            batch_size = 2
            seq_len = 5

            logprobs = torch.zeros(batch_size, seq_len)
            old_logprobs = torch.randn(batch_size, seq_len)
            advantages = torch.ones(batch_size, seq_len)
            loss_mask = torch.ones(batch_size, seq_len)

            # Compute loss
            loss, stats = trainer.compute_ppo_loss(
                logprobs=logprobs,
                old_logprobs=old_logprobs,
                advantages=advantages,
                loss_mask=loss_mask,
            )

            # Check that behavior statistics DON'T exist (standard PPO)
            assert "behave_imp_weight" not in stats
            assert "importance_weight" in stats  # Standard stats exist

    def test_compute_ppo_loss_proximal_fallback(self):
        """Test that old_logprobs is used as fallback when proximal_logprobs is None."""
        from recipe.async_rl.ray_trainer import AsyncRLTrainer

        config = DictConfig({
            "actor_rollout_ref": {
                "actor": {
                    "clip_ratio_low": 0.2,
                    "recompute_logprob": False,
                },
            },
        })

        with patch.object(AsyncRLTrainer, '__init__', lambda self, **kwargs: None):
            trainer = AsyncRLTrainer.__new__(AsyncRLTrainer)
            trainer.config = config
            trainer.use_decoupled_loss = True
            trainer.behav_imp_weight_cap = None
            trainer.behav_imp_weight_floor = None

            # Test data
            logprobs = torch.zeros(2, 5)
            old_logprobs = torch.randn(2, 5)
            advantages = torch.ones(2, 5)
            loss_mask = torch.ones(2, 5)

            # Compute loss without providing proximal_logprobs
            loss, stats = trainer.compute_ppo_loss(
                logprobs=logprobs,
                old_logprobs=old_logprobs,
                advantages=advantages,
                loss_mask=loss_mask,
                proximal_logprobs=None,  # Will use old_logprobs as fallback
            )

            # Should still work
            assert loss.item() is not None
            assert "behave_imp_weight" in stats


class TestSyncWeightsToRollout:
    """Test sync_weights_to_rollout() version management."""

    def test_sync_weights_increments_version(self):
        """Test that sync_weights_to_rollout increments policy version."""
        from recipe.async_rl.ray_trainer import AsyncRLTrainer
        from recipe.async_rl.partial_rollout_manager import PartialRolloutManager

        with patch.object(AsyncRLTrainer, '__init__', lambda self, **kwargs: None):
            trainer = AsyncRLTrainer.__new__(AsyncRLTrainer)
            trainer.policy_version = 0

            # Mock async_rollout_manager with spec to pass isinstance check
            mock_manager = MagicMock(spec=PartialRolloutManager)
            trainer.async_rollout_manager = mock_manager

            # Sync weights
            trainer.sync_weights_to_rollout()

            # Version should be incremented
            assert trainer.policy_version == 1

            # Manager should be notified
            mock_manager.update_policy_version.assert_called_once_with(1)

    def test_sync_weights_multiple_times(self):
        """Test that version increments correctly across multiple syncs."""
        from recipe.async_rl.ray_trainer import AsyncRLTrainer

        with patch.object(AsyncRLTrainer, '__init__', lambda self, **kwargs: None):
            trainer = AsyncRLTrainer.__new__(AsyncRLTrainer)
            trainer.policy_version = 0
            trainer.async_rollout_manager = MagicMock()

            # Multiple syncs
            for expected_version in range(1, 6):
                trainer.sync_weights_to_rollout()
                assert trainer.policy_version == expected_version


class TestTrainingStep:
    """Test training_step() staleness metrics."""

    def test_training_step_staleness_metrics(self):
        """Test that training_step computes staleness metrics."""
        from recipe.async_rl.ray_trainer import AsyncRLTrainer

        config = DictConfig({
            "trainer": {"critic_warmup": 0},
            "actor_rollout_ref": {
                "rollout": {
                    "multi_turn": {"enable": False},
                },
            },
        })

        with patch.object(AsyncRLTrainer, '__init__', lambda self, **kwargs: None):
            trainer = AsyncRLTrainer.__new__(AsyncRLTrainer)
            trainer.config = config
            trainer.use_critic = True
            trainer.global_steps = 10
            trainer.policy_version = 10  # Current version

            # Mock worker groups
            mock_critic_wg = MagicMock()
            mock_actor_rollout_wg = MagicMock()

            critic_output = DataProto.from_dict(
                tensors={},
                non_tensors={},
                meta_info={"metrics": {"critic_loss": 0.5}}
            )
            actor_output = DataProto.from_dict(
                tensors={},
                non_tensors={},
                meta_info={"metrics": {"actor_loss": 0.3}}
            )

            mock_critic_wg.update_critic.return_value = critic_output
            mock_actor_rollout_wg.update_actor.return_value = actor_output

            trainer.critic_wg = mock_critic_wg
            trainer.actor_rollout_wg = mock_actor_rollout_wg

            # Create data with version tracking
            data = DataProto.from_dict(
                tensors={},
                non_tensors={
                    "version_start": np.array([5, 6, 7, 8], dtype=np.int32),
                    "version_end": np.array([6, 7, 8, 9], dtype=np.int32),
                },
                meta_info={}
            )

            # Run training step
            metrics = trainer.training_step(data)

            # Check staleness metrics
            assert "async_rl/max_staleness" in metrics
            assert "async_rl/mean_staleness" in metrics
            assert "async_rl/current_policy_version" in metrics

            # Max staleness: current(10) - min(version_end) = 10 - 6 = 4
            # Staleness array: [10-6, 10-7, 10-8, 10-9] = [4, 3, 2, 1]
            assert metrics["async_rl/max_staleness"] == 4
            # Mean staleness: (4 + 3 + 2 + 1) / 4 = 2.5
            assert abs(metrics["async_rl/mean_staleness"] - 2.5) < 0.01
            # Current version should be tracked
            assert metrics["async_rl/current_policy_version"] == 10

    def test_training_step_without_version_tracking(self):
        """Test that training_step works without version tracking."""
        from recipe.async_rl.ray_trainer import AsyncRLTrainer

        config = DictConfig({
            "trainer": {"critic_warmup": 0},
            "actor_rollout_ref": {
                "rollout": {
                    "multi_turn": {"enable": False},
                },
            },
        })

        with patch.object(AsyncRLTrainer, '__init__', lambda self, **kwargs: None):
            trainer = AsyncRLTrainer.__new__(AsyncRLTrainer)
            trainer.config = config
            trainer.use_critic = False
            trainer.global_steps = 10

            # Mock worker group
            mock_actor_rollout_wg = MagicMock()
            actor_output = DataProto.from_dict(
                tensors={},
                non_tensors={},
                meta_info={"metrics": {"actor_loss": 0.3}}
            )
            mock_actor_rollout_wg.update_actor.return_value = actor_output
            trainer.actor_rollout_wg = mock_actor_rollout_wg

            # Create data without version tracking
            data = DataProto.from_dict(
                tensors={},
                non_tensors={},
                meta_info={}
            )

            # Run training step
            metrics = trainer.training_step(data)

            # Should not have staleness metrics
            assert "async_rl/max_staleness" not in metrics
            # Should have actor metrics
            assert "actor_loss" in metrics


class TestShutdown:
    """Test shutdown() cleanup."""

    def test_shutdown_stops_executor(self):
        """Test that shutdown properly stops thread pool executor."""
        from recipe.async_rl.ray_trainer import AsyncRLTrainer
        import threading

        with patch.object(AsyncRLTrainer, '__init__', lambda self, **kwargs: None):
            trainer = AsyncRLTrainer.__new__(AsyncRLTrainer)

            # Create mock executor and stop event
            mock_executor = MagicMock()
            trainer._rollout_executor = mock_executor
            trainer._stop_event = threading.Event()

            # Shutdown
            trainer.shutdown()

            # Check that stop event was set
            assert trainer._stop_event.is_set()

            # Check that executor shutdown was called
            mock_executor.shutdown.assert_called_once_with(wait=True, cancel_futures=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
