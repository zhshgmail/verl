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
Advanced unit tests for PartialRolloutManager - async operations.

These tests cover:
1. request_weight_update_async() with NCCL and disk methods
2. Error handling in async operations
3. Weight coordinator integration
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from omegaconf import DictConfig

from recipe.async_rl.weight_update import WeightUpdateConfig, WeightUpdateCoordinator


class TestRequestWeightUpdateAsync:
    """Test async weight update request method."""

    @pytest.mark.asyncio
    async def test_request_weight_update_without_coordinator(self):
        """Test that weight update is skipped when coordinator is None."""
        from recipe.async_rl.partial_rollout_manager import PartialRolloutManager

        config = DictConfig({
            "actor_rollout_ref": {
                "rollout": {
                    "backend": "vllm"
                }
            }
        })

        with patch('recipe.async_rl.partial_rollout_manager.AgentLoopManager.__init__', return_value=None):
            manager = PartialRolloutManager.__new__(PartialRolloutManager)
            manager.current_policy_version = 5
            manager.weight_coordinator = None  # No coordinator

            # Should return current version without update
            new_version = await manager.request_weight_update_async()
            assert new_version == 5  # Version unchanged

    @pytest.mark.asyncio
    async def test_request_weight_update_nccl_method(self):
        """Test weight update via NCCL method."""
        from recipe.async_rl.partial_rollout_manager import PartialRolloutManager

        config = DictConfig({
            "actor_rollout_ref": {
                "rollout": {
                    "backend": "vllm"
                }
            }
        })

        with patch('recipe.async_rl.partial_rollout_manager.AgentLoopManager.__init__', return_value=None):
            manager = PartialRolloutManager.__new__(PartialRolloutManager)
            manager.current_policy_version = 3

            # Mock weight coordinator with NCCL method
            mock_coordinator = MagicMock()
            mock_coordinator.config = WeightUpdateConfig(update_method="nccl")
            mock_coordinator.update_weights_nccl = AsyncMock(return_value=4)
            manager.weight_coordinator = mock_coordinator

            # Request update
            new_version = await manager.request_weight_update_async()

            # Should call NCCL update
            mock_coordinator.update_weights_nccl.assert_called_once()
            # Version should be updated
            assert new_version == 4
            assert manager.current_policy_version == 4

    @pytest.mark.asyncio
    async def test_request_weight_update_disk_method_not_implemented(self):
        """Test that disk method currently returns error."""
        from recipe.async_rl.partial_rollout_manager import PartialRolloutManager

        config = DictConfig({
            "actor_rollout_ref": {
                "rollout": {
                    "backend": "vllm"
                }
            }
        })

        with patch('recipe.async_rl.partial_rollout_manager.AgentLoopManager.__init__', return_value=None):
            manager = PartialRolloutManager.__new__(PartialRolloutManager)
            manager.current_policy_version = 5

            # Mock weight coordinator with disk method
            mock_coordinator = MagicMock()
            mock_coordinator.config = WeightUpdateConfig(update_method="disk")
            manager.weight_coordinator = mock_coordinator

            # Request update - should return current version (not implemented)
            new_version = await manager.request_weight_update_async()

            # Version should remain unchanged (disk method not implemented)
            assert new_version == 5
            assert manager.current_policy_version == 5

    @pytest.mark.asyncio
    async def test_request_weight_update_increments_version(self):
        """Test that version is properly incremented through multiple updates."""
        from recipe.async_rl.partial_rollout_manager import PartialRolloutManager

        config = DictConfig({
            "actor_rollout_ref": {
                "rollout": {
                    "backend": "vllm"
                }
            }
        })

        with patch('recipe.async_rl.partial_rollout_manager.AgentLoopManager.__init__', return_value=None):
            manager = PartialRolloutManager.__new__(PartialRolloutManager)
            manager.current_policy_version = 0

            # Mock weight coordinator
            mock_coordinator = MagicMock()
            mock_coordinator.config = WeightUpdateConfig(update_method="nccl")

            # Simulate sequential version updates
            async def mock_update_nccl():
                manager.current_policy_version += 1
                return manager.current_policy_version

            mock_coordinator.update_weights_nccl = mock_update_nccl
            manager.weight_coordinator = mock_coordinator

            # First update: 0 -> 1
            v1 = await manager.request_weight_update_async()
            assert v1 == 1

            # Second update: 1 -> 2
            v2 = await manager.request_weight_update_async()
            assert v2 == 2

            # Third update: 2 -> 3
            v3 = await manager.request_weight_update_async()
            assert v3 == 3


class TestWeightCoordinatorIntegration:
    """Test integration with WeightUpdateCoordinator."""

    @pytest.mark.asyncio
    async def test_coordinator_error_propagation(self):
        """Test that coordinator errors are propagated correctly."""
        from recipe.async_rl.partial_rollout_manager import PartialRolloutManager

        config = DictConfig({
            "actor_rollout_ref": {
                "rollout": {
                    "backend": "vllm"
                }
            }
        })

        with patch('recipe.async_rl.partial_rollout_manager.AgentLoopManager.__init__', return_value=None):
            manager = PartialRolloutManager.__new__(PartialRolloutManager)
            manager.current_policy_version = 5

            # Mock coordinator that raises error
            mock_coordinator = MagicMock()
            mock_coordinator.config = WeightUpdateConfig(update_method="nccl")

            async def mock_update_error():
                raise RuntimeError("NCCL update failed")

            mock_coordinator.update_weights_nccl = mock_update_error
            manager.weight_coordinator = mock_coordinator

            # Should propagate the error
            with pytest.raises(RuntimeError, match="NCCL update failed"):
                await manager.request_weight_update_async()

    @pytest.mark.asyncio
    async def test_version_tracking_across_update_failure(self):
        """Test that version tracking is correct even when update fails."""
        from recipe.async_rl.partial_rollout_manager import PartialRolloutManager

        config = DictConfig({
            "actor_rollout_ref": {
                "rollout": {
                    "backend": "vllm"
                }
            }
        })

        with patch('recipe.async_rl.partial_rollout_manager.AgentLoopManager.__init__', return_value=None):
            manager = PartialRolloutManager.__new__(PartialRolloutManager)
            manager.current_policy_version = 10

            # Mock coordinator that fails
            mock_coordinator = MagicMock()
            mock_coordinator.config = WeightUpdateConfig(update_method="nccl")

            async def mock_update_fail():
                raise RuntimeError("Network error")

            mock_coordinator.update_weights_nccl = mock_update_fail
            manager.weight_coordinator = mock_coordinator

            # Try update and catch error
            try:
                await manager.request_weight_update_async()
            except RuntimeError:
                pass

            # Version should remain unchanged after failed update
            assert manager.current_policy_version == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
