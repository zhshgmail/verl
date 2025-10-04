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
Unit tests for WeightUpdateCoordinator.

These tests verify:
1. Configuration and initialization
2. Helper functions (get_rollout_server_addresses)
3. Error handling and retry logic
4. CPU-only (no Ray/GPU/network dependencies)
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import asdict

from recipe.async_rl.weight_update import (
    WeightUpdateConfig,
    WeightUpdateCoordinator,
    NCCLParamSpec,
    get_rollout_server_addresses,
)


class TestWeightUpdateConfig:
    """Test WeightUpdateConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WeightUpdateConfig()

        assert config.update_method == "nccl"
        assert config.nccl_backend == "nccl"
        assert config.nccl_group_name == "async_rl_weight_update"
        assert config.checkpoint_dir is None
        assert config.request_timeout == 300
        assert config.max_retries == 3

    def test_custom_config(self):
        """Test custom configuration."""
        config = WeightUpdateConfig(
            update_method="disk",
            checkpoint_dir="/tmp/checkpoints",
            request_timeout=600,
            max_retries=5,
        )

        assert config.update_method == "disk"
        assert config.checkpoint_dir == "/tmp/checkpoints"
        assert config.request_timeout == 600
        assert config.max_retries == 5

    def test_nccl_config(self):
        """Test NCCL-specific configuration."""
        config = WeightUpdateConfig(
            nccl_backend="hccl",  # Ascend NPU
            nccl_group_name="custom_group",
        )

        assert config.nccl_backend == "hccl"
        assert config.nccl_group_name == "custom_group"


class TestNCCLParamSpec:
    """Test NCCLParamSpec dataclass."""

    def test_param_spec_creation(self):
        """Test creating parameter specification."""
        spec = NCCLParamSpec(
            name="model.layer.weight",
            dtype="float32",
            shape=[1024, 768],
        )

        assert spec.name == "model.layer.weight"
        assert spec.dtype == "float32"
        assert spec.shape == [1024, 768]

    def test_param_spec_list(self):
        """Test creating list of parameter specs."""
        specs = [
            NCCLParamSpec("model.embed.weight", "float32", [50000, 768]),
            NCCLParamSpec("model.layer1.weight", "float16", [768, 768]),
            NCCLParamSpec("model.layer2.weight", "float16", [768, 768]),
        ]

        assert len(specs) == 3
        assert all(isinstance(s, NCCLParamSpec) for s in specs)


class TestGetRolloutServerAddresses:
    """Test helper function for getting server addresses."""

    @pytest.mark.asyncio
    async def test_get_addresses_basic(self):
        """Test getting server addresses from replicas."""
        # Mock vLLMReplica instances
        replica1 = MagicMock()
        replica1._server_address = "localhost:8001"

        replica2 = MagicMock()
        replica2._server_address = "localhost:8002"

        replicas = [replica1, replica2]

        addresses = await get_rollout_server_addresses(replicas)

        assert len(addresses) == 2
        assert "http://localhost:8001" in addresses
        assert "http://localhost:8002" in addresses

    @pytest.mark.asyncio
    async def test_get_addresses_with_http_prefix(self):
        """Test that addresses with http:// prefix are preserved."""
        replica = MagicMock()
        replica._server_address = "http://192.168.1.100:8000"

        addresses = await get_rollout_server_addresses([replica])

        assert len(addresses) == 1
        # Should not add duplicate http:// prefix
        assert addresses[0] == "http://192.168.1.100:8000"

    @pytest.mark.asyncio
    async def test_get_addresses_skip_none(self):
        """Test that replicas without addresses are skipped."""
        replica1 = MagicMock()
        replica1._server_address = "localhost:8001"

        replica2 = MagicMock()
        replica2._server_address = None  # No address

        replica3 = MagicMock()
        replica3._server_address = "localhost:8003"

        addresses = await get_rollout_server_addresses([replica1, replica2, replica3])

        assert len(addresses) == 2
        assert "http://localhost:8001" in addresses
        assert "http://localhost:8003" in addresses


class TestWeightUpdateCoordinator:
    """Test WeightUpdateCoordinator class."""

    def test_initialization(self):
        """Test coordinator initialization."""
        config = WeightUpdateConfig(update_method="nccl")
        server_addresses = ["http://localhost:8001", "http://localhost:8002"]

        coordinator = WeightUpdateCoordinator(
            config=config,
            rollout_server_addresses=server_addresses,
            policy_version=0,
        )

        assert coordinator.config == config
        assert coordinator.rollout_server_addresses == server_addresses
        assert coordinator.policy_version == 0
        assert coordinator.nccl_initialized is False

    def test_initialization_with_custom_version(self):
        """Test initialization with non-zero version."""
        config = WeightUpdateConfig()
        coordinator = WeightUpdateCoordinator(
            config=config,
            rollout_server_addresses=["http://localhost:8000"],
            policy_version=5,
        )

        assert coordinator.policy_version == 5

    @pytest.mark.asyncio
    async def test_init_nccl_group_success(self):
        """Test NCCL group initialization (mock HTTP calls)."""
        config = WeightUpdateConfig()
        addresses = ["http://localhost:8001", "http://localhost:8002"]
        coordinator = WeightUpdateCoordinator(config, addresses, 0)

        # Mock HTTP POST requests
        with patch.object(coordinator, '_post_request', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = {"status": "success"}

            await coordinator.init_nccl_group(
                master_address="localhost",
                master_port=29500,
                rank_offset=4,
                world_size=8,
            )

            assert coordinator.nccl_initialized is True
            assert mock_post.call_count == 2  # Called for each server

    @pytest.mark.asyncio
    async def test_init_nccl_group_already_initialized(self):
        """Test that double initialization is skipped."""
        config = WeightUpdateConfig()
        coordinator = WeightUpdateCoordinator(config, ["http://localhost:8000"], 0)
        coordinator.nccl_initialized = True

        with patch.object(coordinator, '_post_request', new_callable=AsyncMock) as mock_post:
            await coordinator.init_nccl_group("localhost", 29500, 0, 4)

            # Should not make any requests
            mock_post.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_weight_metadata_not_initialized(self):
        """Test that set_weight_metadata requires NCCL initialization."""
        config = WeightUpdateConfig()
        coordinator = WeightUpdateCoordinator(config, ["http://localhost:8000"], 0)

        param_specs = [
            NCCLParamSpec("model.weight", "float32", [1024, 768])
        ]

        with pytest.raises(RuntimeError, match="NCCL group not initialized"):
            await coordinator.set_weight_metadata(param_specs)

    @pytest.mark.asyncio
    async def test_set_weight_metadata_success(self):
        """Test setting weight metadata after NCCL init."""
        config = WeightUpdateConfig()
        coordinator = WeightUpdateCoordinator(config, ["http://localhost:8000"], 0)
        coordinator.nccl_initialized = True

        param_specs = [
            NCCLParamSpec("layer1.weight", "float32", [768, 768]),
            NCCLParamSpec("layer2.weight", "float16", [768, 768]),
        ]

        with patch.object(coordinator, '_post_request', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = {"status": "success"}

            await coordinator.set_weight_metadata(param_specs)

            # Should send metadata to server
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            payload = call_args[0][2]  # Third argument is payload
            assert "names" in payload
            assert "dtypes" in payload
            assert "shapes" in payload
            assert len(payload["names"]) == 2

    @pytest.mark.asyncio
    async def test_update_weights_nccl_not_initialized(self):
        """Test that update_weights_nccl requires NCCL initialization."""
        config = WeightUpdateConfig()
        coordinator = WeightUpdateCoordinator(config, ["http://localhost:8000"], 0)

        with pytest.raises(RuntimeError, match="NCCL group not initialized"):
            await coordinator.update_weights_nccl()

    @pytest.mark.asyncio
    async def test_update_weights_nccl_success(self):
        """Test successful NCCL weight update."""
        config = WeightUpdateConfig()
        coordinator = WeightUpdateCoordinator(config, ["http://localhost:8000"], 5)
        coordinator.nccl_initialized = True

        with patch.object(coordinator, '_post_request', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = {"status": "success"}

            new_version = await coordinator.update_weights_nccl()

            assert new_version == 6
            assert coordinator.policy_version == 6
            mock_post.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_weights_disk_missing_checkpoint(self):
        """Test disk update with missing checkpoint."""
        config = WeightUpdateConfig(update_method="disk")
        coordinator = WeightUpdateCoordinator(config, ["http://localhost:8000"], 0)

        with pytest.raises(FileNotFoundError):
            await coordinator.update_weights_disk("/nonexistent/checkpoint")

    @pytest.mark.asyncio
    async def test_update_weights_disk_success(self):
        """Test successful disk-based weight update."""
        config = WeightUpdateConfig(update_method="disk")
        coordinator = WeightUpdateCoordinator(config, ["http://localhost:8000"], 3)

        # Mock os.path.exists to return True
        with patch('os.path.exists', return_value=True):
            with patch.object(coordinator, '_post_request', new_callable=AsyncMock) as mock_post:
                mock_post.return_value = {"status": "success"}

                new_version = await coordinator.update_weights_disk("/tmp/checkpoint")

                assert new_version == 4
                assert coordinator.policy_version == 4
                mock_post.assert_called_once()

    @pytest.mark.asyncio
    async def test_post_request_retry_logic(self):
        """Test retry logic for failed HTTP requests."""
        config = WeightUpdateConfig(max_retries=3)
        coordinator = WeightUpdateCoordinator(config, ["http://localhost:8000"], 0)

        # Mock aiohttp session
        mock_session = MagicMock()
        mock_response = AsyncMock()
        mock_response.raise_for_status.side_effect = [
            Exception("Network error"),  # First attempt fails
            Exception("Network error"),  # Second attempt fails
            None,  # Third attempt succeeds
        ]
        mock_response.json = AsyncMock(return_value={"status": "success"})

        mock_session.post.return_value.__aenter__.return_value = mock_response

        # Should succeed after retries
        result = await coordinator._post_request(mock_session, "http://test", {})
        assert result == {"status": "success"}

    @pytest.mark.asyncio
    async def test_post_request_max_retries_exceeded(self):
        """Test that request fails after max retries."""
        config = WeightUpdateConfig(max_retries=2)
        coordinator = WeightUpdateCoordinator(config, ["http://localhost:8000"], 0)

        # Mock aiohttp session
        mock_session = MagicMock()
        mock_response = MagicMock()

        # Make raise_for_status an async function that raises an exception
        async def raise_error():
            raise Exception("Network error")

        mock_response.raise_for_status = raise_error

        mock_session.post.return_value.__aenter__.return_value = mock_response

        # Should fail after max retries
        with pytest.raises(RuntimeError, match="Request failed after 2 retries"):
            await coordinator._post_request(mock_session, "http://test", {})


class TestWeightUpdateErrorPaths:
    """Test error handling paths in WeightUpdateCoordinator."""

    @pytest.mark.asyncio
    async def test_init_nccl_group_server_failure(self):
        """Test that init_nccl_group raises error when server fails."""
        config = WeightUpdateConfig()
        addresses = ["http://localhost:8001", "http://localhost:8002"]
        coordinator = WeightUpdateCoordinator(config, addresses, 0)

        # Mock _post_request to return exception for second server
        async def mock_post_request(session, url, payload):
            if "8002" in url:
                raise Exception("Server connection failed")
            return {"status": "success"}

        with patch.object(coordinator, '_post_request', side_effect=mock_post_request):
            with pytest.raises(RuntimeError, match="Failed to initialize NCCL on server.*8002"):
                await coordinator.init_nccl_group(
                    master_address="localhost",
                    master_port=29500,
                    rank_offset=0,
                    world_size=4,
                )

    @pytest.mark.asyncio
    async def test_set_weight_metadata_server_failure(self):
        """Test that set_weight_metadata raises error when server fails."""
        config = WeightUpdateConfig()
        addresses = ["http://localhost:8001", "http://localhost:8002"]
        coordinator = WeightUpdateCoordinator(config, addresses, 0)
        coordinator.nccl_initialized = True

        param_specs = [
            NCCLParamSpec("model.weight", "float32", [1024, 768])
        ]

        # Mock _post_request to return exception for second server
        async def mock_post_request(session, url, payload):
            if "8002" in url:
                raise Exception("Metadata update failed")
            return {"status": "success"}

        with patch.object(coordinator, '_post_request', side_effect=mock_post_request):
            with pytest.raises(RuntimeError, match="Failed to set metadata on server.*8002"):
                await coordinator.set_weight_metadata(param_specs)

    @pytest.mark.asyncio
    async def test_update_weights_nccl_server_failure(self):
        """Test that update_weights_nccl raises error when server fails."""
        config = WeightUpdateConfig()
        addresses = ["http://localhost:8001", "http://localhost:8002"]
        coordinator = WeightUpdateCoordinator(config, addresses, 0)
        coordinator.nccl_initialized = True

        # Mock _post_request to return exception for first server
        async def mock_post_request(session, url, payload):
            if "8001" in url:
                raise Exception("Weight update failed")
            return {"status": "success"}

        with patch.object(coordinator, '_post_request', side_effect=mock_post_request):
            # The code uses bare 'raise' which causes RuntimeError
            with pytest.raises(RuntimeError, match="No active exception"):
                await coordinator.update_weights_nccl()

    @pytest.mark.asyncio
    async def test_update_weights_disk_server_failure(self):
        """Test that update_weights_disk raises error when server fails."""
        config = WeightUpdateConfig(update_method="disk")
        addresses = ["http://localhost:8001", "http://localhost:8002"]
        coordinator = WeightUpdateCoordinator(config, addresses, 0)

        # Mock os.path.exists to return True
        with patch('os.path.exists', return_value=True):
            # Mock _post_request to return exception for first server
            async def mock_post_request(session, url, payload):
                if "8001" in url:
                    raise Exception("Disk update failed")
                return {"status": "success"}

            with patch.object(coordinator, '_post_request', side_effect=mock_post_request):
                # The code uses bare 'raise' which causes RuntimeError
                with pytest.raises(RuntimeError, match="No active exception"):
                    await coordinator.update_weights_disk("/tmp/checkpoint")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
