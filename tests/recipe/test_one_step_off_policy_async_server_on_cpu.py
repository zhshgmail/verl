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
"""Unit tests for AsyncLLMServerManager in one-step off-policy recipe.

This tests the simplified AsyncLLMServerManager that serves as a thin wrapper
for compatibility with RayPPOTrainer's async rollout mode. The manager simply
delegates to worker_group.async_generate_sequences() without any mode switching
or RolloutReplica infrastructure since one-step off-policy uses separate GPU pools.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from omegaconf import OmegaConf

from recipe.one_step_off_policy.async_server import AsyncLLMServerManager
from verl.protocol import DataProto


@pytest.fixture
def base_config():
    """Create a base configuration for testing."""
    return OmegaConf.create({
        "actor_rollout_ref": {
            "rollout": {
                "name": "vllm",
                "tensor_model_parallel_size": 1,
                "mode": "async",
            },
            "model": {
                "path": "test/model/path",
            }
        },
        "trainer": {
            "n_gpus_per_node": 6,
        },
        "rollout": {
            "n_gpus_per_node": 2,
        }
    })


@pytest.fixture
def mock_worker_group():
    """Create a mock RayWorkerGroup with async_generate_sequences method."""
    mock_wg = MagicMock()
    mock_wg.world_size = 2

    # Mock async_generate_sequences to return a DataProto
    mock_output = DataProto(non_tensor_batch={"test": np.array([1, 2, 3])})
    mock_wg.async_generate_sequences = MagicMock(return_value=mock_output)

    return mock_wg


class TestAsyncLLMServerManagerInit:
    """Test AsyncLLMServerManager initialization."""

    def test_init_basic(self, base_config, mock_worker_group):
        """Test basic initialization stores config and worker_group."""
        manager = AsyncLLMServerManager(base_config, mock_worker_group)

        # Verify attributes are stored correctly
        assert manager.config == base_config
        assert manager.worker_group == mock_worker_group

    def test_init_no_replica_infrastructure(self, base_config, mock_worker_group):
        """Test that no RolloutReplica infrastructure is created."""
        manager = AsyncLLMServerManager(base_config, mock_worker_group)

        # Verify no replica-related attributes exist
        assert not hasattr(manager, 'rollout_replicas')
        assert not hasattr(manager, 'server_handles')
        assert not hasattr(manager, 'server_addresses')


class TestAsyncLLMServerManagerMethods:
    """Test AsyncLLMServerManager methods."""

    def test_generate_sequences_delegates_to_worker_group(self, base_config, mock_worker_group):
        """Test generate_sequences delegates to worker_group.async_generate_sequences()."""
        manager = AsyncLLMServerManager(base_config, mock_worker_group)

        # Test generation with sampling params
        prompts = DataProto(non_tensor_batch={"prompt": np.array(["test prompt"])})
        result = manager.generate_sequences(prompts, temperature=0.8, top_p=0.9)

        # Verify worker_group.async_generate_sequences was called with correct args
        mock_worker_group.async_generate_sequences.assert_called_once_with(
            prompts, temperature=0.8, top_p=0.9
        )

        # Verify result is passed through correctly
        assert result.non_tensor_batch["test"].tolist() == [1, 2, 3]

    def test_generate_sequences_without_sampling_params(self, base_config, mock_worker_group):
        """Test generate_sequences works without sampling params."""
        manager = AsyncLLMServerManager(base_config, mock_worker_group)

        prompts = DataProto(non_tensor_batch={"prompt": np.array(["test prompt"])})
        result = manager.generate_sequences(prompts)

        # Verify worker_group.async_generate_sequences was called
        mock_worker_group.async_generate_sequences.assert_called_once_with(prompts)

    def test_no_wake_up_method(self, base_config, mock_worker_group):
        """Test that wake_up method does not exist (no mode switching)."""
        manager = AsyncLLMServerManager(base_config, mock_worker_group)

        # Verify no wake_up method exists
        assert not hasattr(manager, 'wake_up')

    def test_no_sleep_method(self, base_config, mock_worker_group):
        """Test that sleep method does not exist (no mode switching)."""
        manager = AsyncLLMServerManager(base_config, mock_worker_group)

        # Verify no sleep method exists
        assert not hasattr(manager, 'sleep')


class TestAsyncLLMServerManagerIntegration:
    """Integration tests for AsyncLLMServerManager."""

    def test_multiple_generations(self, base_config, mock_worker_group):
        """Test multiple sequential generations work correctly."""
        manager = AsyncLLMServerManager(base_config, mock_worker_group)

        # Generate multiple times
        prompts1 = DataProto(non_tensor_batch={"prompt": np.array(["prompt 1"])})
        prompts2 = DataProto(non_tensor_batch={"prompt": np.array(["prompt 2"])})

        result1 = manager.generate_sequences(prompts1, temperature=0.7)
        result2 = manager.generate_sequences(prompts2, temperature=0.9)

        # Verify both calls went through
        assert mock_worker_group.async_generate_sequences.call_count == 2

        # Verify both results are correct
        assert result1.non_tensor_batch["test"].tolist() == [1, 2, 3]
        assert result2.non_tensor_batch["test"].tolist() == [1, 2, 3]

    def test_different_sampling_params(self, base_config, mock_worker_group):
        """Test that different sampling params are passed through correctly."""
        manager = AsyncLLMServerManager(base_config, mock_worker_group)

        prompts = DataProto(non_tensor_batch={"prompt": np.array(["test"])})

        # Call with various sampling params
        manager.generate_sequences(prompts, temperature=1.0, top_p=0.95, top_k=50)

        # Verify params were passed through
        call_args = mock_worker_group.async_generate_sequences.call_args
        assert call_args[0][0] == prompts
        assert call_args[1] == {"temperature": 1.0, "top_p": 0.95, "top_k": 50}


class TestOneStepOffPolicyArchitecture:
    """Tests verifying one-step off-policy specific architecture patterns."""

    def test_no_mode_switching_infrastructure(self, base_config, mock_worker_group):
        """Test that no mode switching infrastructure exists.

        One-step off-policy uses separate GPU pools for training and rollout,
        so no wake_up/sleep mode switching is needed.
        """
        manager = AsyncLLMServerManager(base_config, mock_worker_group)

        # Verify no mode switching methods
        assert not hasattr(manager, 'wake_up')
        assert not hasattr(manager, 'sleep')
        assert not hasattr(manager, 'rollout_mode')
        assert not hasattr(manager, 'trainer_mode')

    def test_no_replica_management(self, base_config, mock_worker_group):
        """Test that no RolloutReplica infrastructure exists.

        Rollout workers are already initialized by OneStepOffRayTrainer,
        so AsyncLLMServerManager doesn't need to manage replicas.
        """
        manager = AsyncLLMServerManager(base_config, mock_worker_group)

        # Verify no replica infrastructure
        assert not hasattr(manager, 'rollout_replicas')
        assert not hasattr(manager, 'server_handles')
        assert not hasattr(manager, 'server_addresses')
        assert not hasattr(manager, 'init_hybrid')

    def test_simple_delegation_pattern(self, base_config, mock_worker_group):
        """Test that AsyncLLMServerManager is just a simple delegation wrapper.

        The class exists purely for interface compatibility with RayPPOTrainer,
        which expects async_rollout_manager to have a generate_sequences() method.
        """
        manager = AsyncLLMServerManager(base_config, mock_worker_group)

        # Verify it's truly minimal - only config, worker_group, and generate_sequences
        public_attrs = [attr for attr in dir(manager) if not attr.startswith('_')]

        # Should have: config, worker_group, generate_sequences (and inherited object methods)
        essential_attrs = {'config', 'worker_group', 'generate_sequences'}
        assert essential_attrs.issubset(set(public_attrs))

        # Should NOT have: wake_up, sleep, rollout_replicas, init_hybrid, etc.
        unwanted_attrs = {'wake_up', 'sleep', 'rollout_replicas', 'init_hybrid',
                         'server_handles', 'server_addresses', 'rollout_mode', 'trainer_mode'}
        assert unwanted_attrs.isdisjoint(set(public_attrs))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
