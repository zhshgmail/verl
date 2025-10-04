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
Unit tests for PartialRolloutManager version tracking.

These tests verify:
1. Version tracking (version_start, version_end, token_policy_versions)
2. Logprob renaming (rollout_log_probs → old_logprobs)
3. Version update logic
4. CPU-only (no Ray/GPU dependencies)
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from omegaconf import DictConfig

from verl.protocol import DataProto


class TestPartialRolloutManagerVersionTracking:
    """Test version tracking logic without Ray/GPU dependencies."""

    def test_version_annotation_basic(self):
        """Test that generate_sequences adds version metadata."""
        # Mock config
        config = DictConfig({
            "actor_rollout_ref": {
                "rollout": {
                    "backend": "vllm"
                }
            }
        })

        # Mock parent class to avoid Ray initialization
        with patch('recipe.async_rl.partial_rollout_manager.AgentLoopManager.__init__', return_value=None):
            from recipe.async_rl.partial_rollout_manager import PartialRolloutManager

            manager = PartialRolloutManager.__new__(PartialRolloutManager)
            manager.current_policy_version = 5
            manager.weight_coordinator = None

            # Mock parent's generate_sequences to return dummy data
            batch_size = 4
            seq_len = 10
            mock_output = DataProto.from_dict(
                tensors={
                    "rollout_log_probs": torch.randn(batch_size, seq_len),
                    "responses": torch.randint(0, 1000, (batch_size, seq_len)),
                },
                non_tensors={
                    "some_metadata": np.array(["test"] * batch_size),
                }
            )

            with patch.object(PartialRolloutManager.__bases__[0], 'generate_sequences', return_value=mock_output):
                prompts = MagicMock()
                output = manager.generate_sequences(prompts)

                # Check version annotations
                assert "version_start" in output.non_tensor_batch
                assert "version_end" in output.non_tensor_batch
                assert len(output.non_tensor_batch["version_start"]) == batch_size
                assert len(output.non_tensor_batch["version_end"]) == batch_size
                assert all(output.non_tensor_batch["version_start"] == 5)
                assert all(output.non_tensor_batch["version_end"] == 5)

    def test_logprob_renaming(self):
        """Test that rollout_log_probs is renamed to old_logprobs."""
        config = DictConfig({
            "actor_rollout_ref": {
                "rollout": {
                    "backend": "vllm"
                }
            }
        })

        with patch('recipe.async_rl.partial_rollout_manager.AgentLoopManager.__init__', return_value=None):
            from recipe.async_rl.partial_rollout_manager import PartialRolloutManager

            manager = PartialRolloutManager.__new__(PartialRolloutManager)
            manager.current_policy_version = 0
            manager.weight_coordinator = None

            batch_size = 2
            seq_len = 5
            rollout_logprobs = torch.randn(batch_size, seq_len)

            mock_output = DataProto.from_dict(
                tensors={
                    "rollout_log_probs": rollout_logprobs,
                },
                non_tensors={}
            )

            with patch.object(PartialRolloutManager.__bases__[0], 'generate_sequences', return_value=mock_output):
                prompts = MagicMock()
                output = manager.generate_sequences(prompts)

                # Check renaming
                assert "rollout_log_probs" not in output.batch
                assert "old_logprobs" in output.batch
                assert torch.allclose(output.batch["old_logprobs"], rollout_logprobs)

    def test_token_policy_versions(self):
        """Test that per-token policy versions are added."""
        config = DictConfig({
            "actor_rollout_ref": {
                "rollout": {
                    "backend": "vllm"
                }
            }
        })

        with patch('recipe.async_rl.partial_rollout_manager.AgentLoopManager.__init__', return_value=None):
            from recipe.async_rl.partial_rollout_manager import PartialRolloutManager

            manager = PartialRolloutManager.__new__(PartialRolloutManager)
            manager.current_policy_version = 3
            manager.weight_coordinator = None

            batch_size = 4
            seq_len = 10

            mock_output = DataProto.from_dict(
                tensors={
                    "rollout_log_probs": torch.randn(batch_size, seq_len),
                },
                non_tensors={}
            )

            with patch.object(PartialRolloutManager.__bases__[0], 'generate_sequences', return_value=mock_output):
                prompts = MagicMock()
                output = manager.generate_sequences(prompts)

                # Check per-token versions
                assert "token_policy_versions" in output.batch
                token_versions = output.batch["token_policy_versions"]
                assert token_versions.shape == (batch_size, seq_len)
                assert (token_versions == 3).all()
                assert token_versions.dtype == torch.int32

    def test_version_update(self):
        """Test that update_policy_version updates the version correctly."""
        config = DictConfig({
            "actor_rollout_ref": {
                "rollout": {
                    "backend": "vllm"
                }
            }
        })

        with patch('recipe.async_rl.partial_rollout_manager.AgentLoopManager.__init__', return_value=None):
            from recipe.async_rl.partial_rollout_manager import PartialRolloutManager

            manager = PartialRolloutManager.__new__(PartialRolloutManager)
            manager.current_policy_version = 0
            manager.weight_coordinator = None

            # Update version
            manager.update_policy_version(5)
            assert manager.current_policy_version == 5

            # Update again
            manager.update_policy_version(10)
            assert manager.current_policy_version == 10

    def test_version_consistency_no_mid_generation_update(self):
        """Test version_start == version_end when no update happens during generation."""
        config = DictConfig({
            "actor_rollout_ref": {
                "rollout": {
                    "backend": "vllm"
                }
            }
        })

        with patch('recipe.async_rl.partial_rollout_manager.AgentLoopManager.__init__', return_value=None):
            from recipe.async_rl.partial_rollout_manager import PartialRolloutManager

            manager = PartialRolloutManager.__new__(PartialRolloutManager)
            manager.current_policy_version = 7
            manager.weight_coordinator = None

            batch_size = 3
            seq_len = 8

            mock_output = DataProto.from_dict(
                tensors={
                    "rollout_log_probs": torch.randn(batch_size, seq_len),
                },
                non_tensors={}
            )

            with patch.object(PartialRolloutManager.__bases__[0], 'generate_sequences', return_value=mock_output):
                prompts = MagicMock()
                output = manager.generate_sequences(prompts)

                # When no update happens, version_start should equal version_end
                assert all(output.non_tensor_batch["version_start"] == 7)
                assert all(output.non_tensor_batch["version_end"] == 7)
                assert all(output.non_tensor_batch["version_start"] == output.non_tensor_batch["version_end"])

    def test_handles_missing_rollout_log_probs(self):
        """Test that manager handles missing rollout_log_probs gracefully."""
        config = DictConfig({
            "actor_rollout_ref": {
                "rollout": {
                    "backend": "vllm"
                }
            }
        })

        with patch('recipe.async_rl.partial_rollout_manager.AgentLoopManager.__init__', return_value=None):
            from recipe.async_rl.partial_rollout_manager import PartialRolloutManager

            manager = PartialRolloutManager.__new__(PartialRolloutManager)
            manager.current_policy_version = 0
            manager.weight_coordinator = None

            # Mock output without rollout_log_probs
            mock_output = DataProto.from_dict(
                tensors={
                    "responses": torch.randint(0, 1000, (2, 5)),
                },
                non_tensors={}
            )

            with patch.object(PartialRolloutManager.__bases__[0], 'generate_sequences', return_value=mock_output):
                prompts = MagicMock()
                output = manager.generate_sequences(prompts)

                # Should not crash, just skip renaming
                assert "old_logprobs" not in output.batch
                assert "token_policy_versions" not in output.batch

                # But version annotations should still be added
                assert "version_start" in output.non_tensor_batch
                assert "version_end" in output.non_tensor_batch


class TestPartialRolloutManagerInit:
    """Test PartialRolloutManager initialization and logging."""

    def test_init_logs_correctly(self):
        """Test that __init__ executes logging statements."""
        from recipe.async_rl.partial_rollout_manager import PartialRolloutManager
        from omegaconf import DictConfig

        config = DictConfig({
            "actor_rollout_ref": {
                "rollout": {
                    "tensor_model_parallel_size": 1,
                    "name": "vllm",
                },
            },
        })

        mock_worker_group = MagicMock()
        mock_worker_group.world_size = 4

        # Mock parent init to prevent actual initialization
        with patch('recipe.async_rl.partial_rollout_manager.AgentLoopManager.__init__', return_value=None):
            with patch('recipe.async_rl.partial_rollout_manager.logger') as mock_logger:
                manager = PartialRolloutManager(
                    config=config,
                    worker_group=mock_worker_group,
                    rm_wg=None,
                    weight_coordinator=None,
                )

                # Verify __init__ executed (lines 99-106)
                assert manager.current_policy_version == 0
                assert manager.weight_coordinator is None

                # Verify logger.info was called
                assert mock_logger.info.called

    def test_initialize_llm_servers_logs_warning(self):
        """Test that _initialize_llm_servers logs warning about workaround."""
        from recipe.async_rl.partial_rollout_manager import PartialRolloutManager

        with patch('recipe.async_rl.partial_rollout_manager.AgentLoopManager.__init__', return_value=None):
            manager = PartialRolloutManager.__new__(PartialRolloutManager)

            # Mock parent's _initialize_llm_servers
            with patch('recipe.async_rl.partial_rollout_manager.AgentLoopManager._initialize_llm_servers'):
                with patch('recipe.async_rl.partial_rollout_manager.logger') as mock_logger:
                    manager._initialize_llm_servers()

                    # Verify logger.warning was called (lines 140-142)
                    mock_logger.warning.assert_called_once()
                    warning_msg = str(mock_logger.warning.call_args)
                    assert "init_hybrid" in warning_msg
                    assert "workaround" in warning_msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
