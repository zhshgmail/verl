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
Unit tests for AsyncRLTrainer initialization and setup.

These tests cover:
1. __init__() with various configurations
2. init_workers() with mocked Ray objects
3. Configuration validation
4. ThreadPoolExecutor creation
"""

import pytest
import threading
from unittest.mock import MagicMock, patch, call
from omegaconf import DictConfig
from concurrent.futures import ThreadPoolExecutor


class TestAsyncRLTrainerInit:
    """Test AsyncRLTrainer.__init__() execution paths."""

    def test_init_with_minimal_config(self):
        """Test __init__() executes with minimal configuration."""
        from recipe.async_rl.ray_trainer import AsyncRLTrainer

        config = DictConfig({
            "actor_rollout_ref": {
                "rollout": {"mode": "async"},
                "actor": {},
            },
            "algorithm": {
                "kl_ctrl": {"kl_coef": 0.1},
            },
        })

        # Mock parent __init__ to prevent Ray initialization
        with patch('recipe.async_rl.ray_trainer.RayPPOTrainer.__init__', return_value=None):
            trainer = AsyncRLTrainer(
                config=config,
                tokenizer=MagicMock(),
                role_worker_mapping={},
                resource_pool_manager=MagicMock(),
            )

            # Check that __init__ executed and set attributes
            assert trainer.use_decoupled_loss is True  # Default value
            assert trainer.behav_imp_weight_cap is None
            assert trainer.behav_imp_weight_floor is None
            assert trainer.buffer_size == 10000  # Default
            assert trainer.max_staleness == 5  # Default
            assert trainer.train_batch_size == 128  # Default
            assert trainer.rollout_batch_size == 256  # Default
            assert trainer.policy_version == 0
            assert isinstance(trainer._rollout_executor, ThreadPoolExecutor)
            assert isinstance(trainer._stop_event, threading.Event)
            assert trainer._rollout_running is False
            assert trainer._training_running is False

    def test_init_with_custom_config(self):
        """Test __init__() with custom async_rl configuration."""
        from recipe.async_rl.ray_trainer import AsyncRLTrainer

        config = DictConfig({
            "actor_rollout_ref": {
                "rollout": {"mode": "async"},
                "actor": {
                    "use_decoupled_loss": False,
                    "behav_imp_weight_cap": 10.0,
                    "behav_imp_weight_floor": 0.1,
                },
            },
            "async_rl": {
                "buffer_size": 5000,
                "max_staleness": 10,
                "train_batch_size": 64,
                "rollout_batch_size": 128,
                "rollout_executor_workers": 8,
            },
            "algorithm": {
                "kl_ctrl": {"kl_coef": 0.1},
            },
        })

        with patch('recipe.async_rl.ray_trainer.RayPPOTrainer.__init__', return_value=None):
            trainer = AsyncRLTrainer(
                config=config,
                tokenizer=MagicMock(),
                role_worker_mapping={},
                resource_pool_manager=MagicMock(),
            )

            # Check custom config values were extracted
            assert trainer.use_decoupled_loss is False
            assert trainer.behav_imp_weight_cap == 10.0
            assert trainer.behav_imp_weight_floor == 0.1
            assert trainer.buffer_size == 5000
            assert trainer.max_staleness == 10
            assert trainer.train_batch_size == 64
            assert trainer.rollout_batch_size == 128
            assert trainer._rollout_executor._max_workers == 8

    def test_init_validates_async_mode(self):
        """Test that __init__() validates rollout.mode='async'."""
        from recipe.async_rl.ray_trainer import AsyncRLTrainer

        config_invalid = DictConfig({
            "actor_rollout_ref": {
                "rollout": {"mode": "sync"},  # Invalid
                "actor": {},
            },
        })

        with pytest.raises(AssertionError, match="requires rollout.mode='async'"):
            AsyncRLTrainer(
                config=config_invalid,
                tokenizer=MagicMock(),
                role_worker_mapping={},
                resource_pool_manager=MagicMock(),
            )

    def test_init_creates_thread_pool(self):
        """Test that __init__() creates dedicated ThreadPoolExecutor."""
        from recipe.async_rl.ray_trainer import AsyncRLTrainer

        config = DictConfig({
            "actor_rollout_ref": {
                "rollout": {"mode": "async"},
                "actor": {},
            },
            "async_rl": {
                "rollout_executor_workers": 6,
            },
            "algorithm": {
                "kl_ctrl": {"kl_coef": 0.1},
            },
        })

        with patch('recipe.async_rl.ray_trainer.RayPPOTrainer.__init__', return_value=None):
            trainer = AsyncRLTrainer(
                config=config,
                tokenizer=MagicMock(),
                role_worker_mapping={},
                resource_pool_manager=MagicMock(),
            )

            assert isinstance(trainer._rollout_executor, ThreadPoolExecutor)
            assert trainer._rollout_executor._max_workers == 6
            # Check thread name prefix for debugging
            assert trainer._rollout_executor._thread_name_prefix == "async_rl_rollout"

    def test_init_sets_tq_client_to_none(self):
        """Test that __init__() sets tq_client to None (initialized later)."""
        from recipe.async_rl.ray_trainer import AsyncRLTrainer

        config = DictConfig({
            "actor_rollout_ref": {
                "rollout": {"mode": "async"},
                "actor": {},
            },
            "algorithm": {
                "kl_ctrl": {"kl_coef": 0.1},
            },
        })

        with patch('recipe.async_rl.ray_trainer.RayPPOTrainer.__init__', return_value=None):
            trainer = AsyncRLTrainer(
                config=config,
                tokenizer=MagicMock(),
                role_worker_mapping={},
                resource_pool_manager=MagicMock(),
            )

            assert trainer.tq_client is None  # Initialized in init_workers()


class TestAsyncRLTrainerInitWorkers:
    """Test AsyncRLTrainer.init_workers() execution paths."""

    def test_init_workers_calls_parent(self):
        """Test that init_workers() calls parent implementation."""
        from recipe.async_rl.ray_trainer import AsyncRLTrainer

        config = DictConfig({
            "actor_rollout_ref": {
                "rollout": {
                    "mode": "async",
                    "tensor_model_parallel_size": 1,
                    "name": "vllm",
                },
                "actor": {},
            },
            "async_rl": {
                "transfer_queue": {
                    "num_storage_units": 2,
                    "num_controllers": 1,
                },
            },
            "algorithm": {
                "kl_ctrl": {"kl_coef": 0.1},
            },
        })

        with patch('recipe.async_rl.ray_trainer.RayPPOTrainer.__init__', return_value=None):
            trainer = AsyncRLTrainer.__new__(AsyncRLTrainer)
            trainer.config = config
            trainer.train_batch_size = 128
            trainer.async_rollout_mode = True
            trainer.use_rm = False

            # Mock parent's init_workers
            with patch('recipe.async_rl.ray_trainer.RayPPOTrainer.init_workers') as mock_parent_init:
                # Mock Ray objects
                with patch('recipe.async_rl.ray_trainer.get_placement_group') as mock_pg:
                    with patch('recipe.async_rl.ray_trainer.TransferQueueStorageSimpleUnit') as mock_storage:
                        with patch('recipe.async_rl.ray_trainer.TransferQueueController') as mock_controller:
                            with patch('recipe.async_rl.ray_trainer.process_zmq_server_info') as mock_zmq:
                                with patch('recipe.async_rl.ray_trainer.ray.get') as mock_ray_get:
                                    with patch('recipe.async_rl.ray_trainer.AsyncTransferQueueClient') as mock_client:
                                        # Setup mocks
                                        mock_pg.return_value = MagicMock()
                                        mock_storage.options.return_value.remote.return_value = MagicMock()
                                        mock_controller.options.return_value.remote.return_value = MagicMock()
                                        mock_zmq.return_value = [{"controller": "info"}]  # Must be list
                                        mock_ray_get.return_value = None

                                        # Setup worker group
                                        mock_wg = MagicMock()
                                        mock_wg.world_size = 4
                                        trainer.actor_rollout_wg = mock_wg

                                        # Mock PartialRolloutManager to prevent actual initialization
                                        with patch('recipe.async_rl.ray_trainer.PartialRolloutManager'):
                                            # Run init_workers
                                            trainer.init_workers()

                                            # Verify parent was called
                                            mock_parent_init.assert_called_once()

    def test_init_workers_creates_partial_rollout_manager_with_rollout_wg(self):
        """Test that init_workers() creates PartialRolloutManager with separate rollout_wg."""
        from recipe.async_rl.ray_trainer import AsyncRLTrainer

        config = DictConfig({
            "actor_rollout_ref": {
                "rollout": {"mode": "async"},
                "actor": {},
            },
            "async_rl": {
                "transfer_queue": {},
            },
            "algorithm": {
                "kl_ctrl": {"kl_coef": 0.1},
            },
        })

        with patch('recipe.async_rl.ray_trainer.RayPPOTrainer.__init__', return_value=None):
            trainer = AsyncRLTrainer.__new__(AsyncRLTrainer)
            trainer.config = config
            trainer.train_batch_size = 128
            trainer.async_rollout_mode = True
            trainer.use_rm = False

            # Mock separate rollout_wg
            mock_rollout_wg = MagicMock()
            mock_rollout_wg.world_size = 8
            trainer.rollout_wg = mock_rollout_wg

            with patch('recipe.async_rl.ray_trainer.RayPPOTrainer.init_workers'):
                with patch('recipe.async_rl.ray_trainer.get_placement_group'):
                    with patch('recipe.async_rl.ray_trainer.TransferQueueStorageSimpleUnit'):
                        with patch('recipe.async_rl.ray_trainer.TransferQueueController'):
                            with patch('recipe.async_rl.ray_trainer.process_zmq_server_info', return_value=[{}]):
                                with patch('recipe.async_rl.ray_trainer.ray.get'):
                                    with patch('recipe.async_rl.ray_trainer.AsyncTransferQueueClient'):
                                        with patch('recipe.async_rl.ray_trainer.PartialRolloutManager') as mock_prm:
                                            trainer.init_workers()

                                            # Verify PartialRolloutManager was created with rollout_wg
                                            mock_prm.assert_called_once()
                                            call_kwargs = mock_prm.call_args[1]
                                            assert call_kwargs['worker_group'] == mock_rollout_wg

    def test_init_workers_creates_partial_rollout_manager_with_actor_rollout_wg(self):
        """Test init_workers() uses actor_rollout_wg when no separate rollout_wg."""
        from recipe.async_rl.ray_trainer import AsyncRLTrainer

        config = DictConfig({
            "actor_rollout_ref": {
                "rollout": {"mode": "async"},
                "actor": {},
            },
            "async_rl": {
                "transfer_queue": {},
            },
            "algorithm": {
                "kl_ctrl": {"kl_coef": 0.1},
            },
        })

        with patch('recipe.async_rl.ray_trainer.RayPPOTrainer.__init__', return_value=None):
            trainer = AsyncRLTrainer.__new__(AsyncRLTrainer)
            trainer.config = config
            trainer.train_batch_size = 128
            trainer.async_rollout_mode = True
            trainer.use_rm = False

            # No rollout_wg, only actor_rollout_wg
            mock_actor_rollout_wg = MagicMock()
            mock_actor_rollout_wg.world_size = 4
            trainer.actor_rollout_wg = mock_actor_rollout_wg

            with patch('recipe.async_rl.ray_trainer.RayPPOTrainer.init_workers'):
                with patch('recipe.async_rl.ray_trainer.get_placement_group'):
                    with patch('recipe.async_rl.ray_trainer.TransferQueueStorageSimpleUnit'):
                        with patch('recipe.async_rl.ray_trainer.TransferQueueController'):
                            with patch('recipe.async_rl.ray_trainer.process_zmq_server_info', return_value=[{}]):
                                with patch('recipe.async_rl.ray_trainer.ray.get'):
                                    with patch('recipe.async_rl.ray_trainer.AsyncTransferQueueClient'):
                                        with patch('recipe.async_rl.ray_trainer.PartialRolloutManager') as mock_prm:
                                            trainer.init_workers()

                                            # Verify PartialRolloutManager was created with actor_rollout_wg
                                            mock_prm.assert_called_once()
                                            call_kwargs = mock_prm.call_args[1]
                                            assert call_kwargs['worker_group'] == mock_actor_rollout_wg

    def test_init_workers_creates_transfer_queue(self):
        """Test that init_workers() creates TransferQueue system."""
        from recipe.async_rl.ray_trainer import AsyncRLTrainer

        config = DictConfig({
            "actor_rollout_ref": {
                "rollout": {"mode": "async"},
                "actor": {},
            },
            "async_rl": {
                "transfer_queue": {
                    "num_storage_units": 3,
                    "num_controllers": 2,
                    "num_global_batch": 4,
                },
            },
            "algorithm": {
                "kl_ctrl": {"kl_coef": 0.1},
            },
        })

        with patch('recipe.async_rl.ray_trainer.RayPPOTrainer.__init__', return_value=None):
            trainer = AsyncRLTrainer.__new__(AsyncRLTrainer)
            trainer.config = config
            trainer.train_batch_size = 128
            trainer.async_rollout_mode = False  # Skip PartialRolloutManager creation

            with patch('recipe.async_rl.ray_trainer.RayPPOTrainer.init_workers'):
                with patch('recipe.async_rl.ray_trainer.get_placement_group') as mock_pg:
                    with patch('recipe.async_rl.ray_trainer.TransferQueueStorageSimpleUnit') as mock_storage_cls:
                        with patch('recipe.async_rl.ray_trainer.TransferQueueController') as mock_controller_cls:
                            with patch('recipe.async_rl.ray_trainer.process_zmq_server_info') as mock_zmq:
                                with patch('recipe.async_rl.ray_trainer.ray.get') as mock_ray_get:
                                    with patch('recipe.async_rl.ray_trainer.AsyncTransferQueueClient') as mock_client_cls:
                                        # Setup mocks
                                        mock_pg.return_value = MagicMock()
                                        mock_storage = MagicMock()
                                        mock_storage_cls.options.return_value.remote.return_value = mock_storage
                                        mock_controller = MagicMock()
                                        mock_controller_cls.options.return_value.remote.return_value = mock_controller
                                        mock_zmq.return_value = [{"addr": "localhost:5555"}]
                                        mock_ray_get.return_value = None

                                        trainer.init_workers()

                                        # Verify TransferQueue components were created
                                        assert mock_storage_cls.options.return_value.remote.call_count == 3  # num_storage_units
                                        assert mock_controller_cls.options.return_value.remote.call_count == 2  # num_controllers
                                        mock_client_cls.assert_called_once()

                                        # Verify client was assigned
                                        assert trainer.tq_client is not None


class TestComputePPOLossRecomputePath:
    """Test recompute_logprob path in compute_ppo_loss()."""

    def test_recompute_logprob_warning_path(self):
        """Test that recompute_logprob=True triggers warning."""
        from recipe.async_rl.ray_trainer import AsyncRLTrainer
        import torch

        config = DictConfig({
            "actor_rollout_ref": {
                "actor": {
                    "clip_ratio_low": 0.2,
                    "recompute_logprob": True,  # Enable recompute
                },
            },
        })

        with patch.object(AsyncRLTrainer, '__init__', lambda self, **kwargs: None):
            trainer = AsyncRLTrainer.__new__(AsyncRLTrainer)
            trainer.config = config
            trainer.use_decoupled_loss = True
            trainer.behav_imp_weight_cap = None
            trainer.behav_imp_weight_floor = None

            logprobs = torch.zeros(2, 5)
            old_logprobs = torch.randn(2, 5)
            advantages = torch.ones(2, 5)
            loss_mask = torch.ones(2, 5)

            # Call without proximal_logprobs to trigger recompute path
            with patch('recipe.async_rl.ray_trainer.logger') as mock_logger:
                loss, stats = trainer.compute_ppo_loss(
                    logprobs=logprobs,
                    old_logprobs=old_logprobs,
                    advantages=advantages,
                    loss_mask=loss_mask,
                    proximal_logprobs=None,  # Will trigger recompute path
                )

                # Verify warning was logged
                mock_logger.warning.assert_called_once()
                assert "recomputation not implemented" in str(mock_logger.warning.call_args)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
