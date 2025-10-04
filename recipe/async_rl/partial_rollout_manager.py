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
Partial Rollout Manager for Asynchronous RL Training

Extends AgentLoopManager to support policy version tracking and weight updates.

Key features:
1. Track policy version for each sample (version_start, version_end)
2. Support weight updates during generation (aborts requests, resets cache)
3. Integrate with WeightUpdateCoordinator for HTTP-based updates

IMPORTANT: We do NOT implement chunked generation like AReaL.
         - AReaL's chunking is a workaround to reduce wasted work when aborting
         - Quote from AReaL code: "This is a hack usage. We don't need it if
           the server can pause requests, update weights, and recompute kv caches"
         - With our abort_all_requests() + reset_prefix_cache(), we just accept
           the wasted work when updating weights mid-generation
         - Simpler implementation, same version tracking benefits
"""

import asyncio
import logging
import os
from typing import List, Optional

import aiohttp
import numpy as np
import ray
import torch
from omegaconf import DictConfig

from verl.experimental.agent_loop import AgentLoopManager
from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup

from .weight_update import WeightUpdateCoordinator, get_rollout_server_addresses

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class PartialRolloutManager(AgentLoopManager):
    """
    Manager for partial rollout with chunked generation and version tracking.

    Extends AgentLoopManager to support:
    1. Chunked generation: Generate N tokens at a time instead of full completion
    2. Version tracking: Annotate samples with policy version info
    3. Weight updates: Coordinate with trainer for mid-generation updates

    Architecture Note:
        This manager is designed for separate GPU pools (actor_pool vs rollout_pool),
        which is different from AgentLoopManager's hybrid mode (colocated workers).

        FIXME: AgentLoopManager._initialize_llm_servers() only supports:
        - hybrid mode: worker_group has colocated actor+rollout
        - standalone mode: creates new workers

        For separate pools, we need worker_group to be rollout-only workers.
        This requires modifying how init_hybrid() is called or adding a new
        init_separate_pools() method. For now, we work around this limitation.
    """

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup,
        rm_wg: Optional[RayWorkerGroup] = None,
        weight_coordinator: Optional[WeightUpdateCoordinator] = None,
    ):
        """
        Initialize PartialRolloutManager.

        Args:
            config: Full trainer configuration
            worker_group: RayWorkerGroup for rollout workers (on separate GPU pool)
            rm_wg: Optional reward model worker group
            weight_coordinator: WeightUpdateCoordinator for managing weight updates.
                If None, weight updates are disabled.

        Note:
            FIXME: worker_group should be rollout-only workers on separate pool,
            but AgentLoopManager expects hybrid workers. This is a known limitation
            that should be fixed by adding init_separate_pools() to RolloutReplica.
        """
        self.current_policy_version = 0  # Track current policy version
        self.weight_coordinator = weight_coordinator  # Weight update coordinator

        # Initialize parent AgentLoopManager
        # This will call _initialize_llm_servers() which has the hybrid/standalone issue
        super().__init__(config, worker_group, rm_wg)

        logger.info(
            f"PartialRolloutManager initialized:\n"
            f"  - initial_policy_version={self.current_policy_version}\n"
            f"  - weight_updates_enabled={weight_coordinator is not None}"
        )

    def _initialize_llm_servers(self):
        """
        Override to handle separate GPU pools.

        FIXME: This is a workaround for AgentLoopManager's limitation.
        AgentLoopManager._initialize_llm_servers() assumes either:
        1. worker_group has hybrid workers (actor+rollout colocated)
        2. worker_group is None (creates standalone workers)

        But for separate pools, we have:
        - worker_group = rollout-only workers on dedicated GPUs
        - Should call init_hybrid(worker_group) but workers don't have actor

        Workaround:
        - For now, pass rollout worker_group to parent implementation
        - Parent will call server.init_hybrid(worker_group)
        - vLLMReplica.init_hybrid() only uses workers to extract Ray actors,
          doesn't actually check if they have actor module
        - So it works, but semantically misleading

        TODO: Upstream fix to verl engine:
        - Add RolloutReplica.init_separate_pools(rollout_worker_group) method
        - Explicitly designed for separate training/rollout pools
        - Then we can call that instead of init_hybrid()
        """
        # Call parent implementation
        # This will work because init_hybrid() only extracts worker actors,
        # but it's semantically wrong to call it "hybrid" mode
        super()._initialize_llm_servers()

        logger.warning(
            "Using init_hybrid() for separate rollout pool. "
            "This is a workaround - should use init_separate_pools() instead. "
            "See FIXME in partial_rollout_manager.py"
        )

    def generate_sequences(self, prompts: DataProto, **sampling_params) -> DataProto:
        """
        Generate sequences with per-token version tracking.

        This method wraps parent's generate_sequences() to add:
        1. Per-token policy versions (for staleness validation)
        2. Behavior logprobs (renamed from rollout_log_probs for clarity)
        3. Sample-level version_start/version_end

        Following AReaL design:
        - old_logprobs: Per-token behavior logprobs π_old(a_t|s_t) from generation
        - token_policy_versions: Per-token policy version when each token was generated
        - Samples with weight updates mid-generation will be rejected by TransferDock

        If weight update happens during generation:
        - Requests are aborted by the weight update process
        - Those samples are lost (not returned)
        - Next generate_sequences() call will use new weights

        Args:
            prompts: Input prompts
            **sampling_params: Sampling parameters (temperature, top_p, etc.)

        Returns:
            DataProto with generated sequences, annotated with per-token metadata
        """
        # Record version at start of generation
        version_start = self.current_policy_version

        # Call parent's generate_sequences
        output = super().generate_sequences(prompts, **sampling_params)

        # Version at end may have changed if update happened
        version_end = self.current_policy_version

        # Annotate with version information
        batch_size = len(output)
        version_start_arr = np.array([version_start] * batch_size, dtype=np.int32)
        version_end_arr = np.array([version_end] * batch_size, dtype=np.int32)

        # Add sample-level version tracking
        if "version_start" not in output.non_tensor_batch:
            output.non_tensor_batch["version_start"] = version_start_arr
        if "version_end" not in output.non_tensor_batch:
            output.non_tensor_batch["version_end"] = version_end_arr

        # Rename rollout_log_probs → old_logprobs for clarity
        # These are the behavior logprobs π_old from the policy that generated the data
        if "rollout_log_probs" in output.batch:
            output.batch["old_logprobs"] = output.batch.pop("rollout_log_probs")

        # Add per-token policy versions
        # For now, assume no weight update mid-generation (all tokens have same version)
        # TODO: For true chunked generation with mid-generation updates, this should
        # track the actual version when each token was generated
        if "old_logprobs" in output.batch:
            # Get sequence length from logprobs
            seqlen = output.batch["old_logprobs"].shape[1]
            # Create per-token versions (all tokens have version_start for now)
            # Shape: [batch_size, seqlen]
            token_versions = torch.full(
                (batch_size, seqlen),
                version_start,
                dtype=torch.int32,
                device=output.batch["old_logprobs"].device
            )
            output.batch["token_policy_versions"] = token_versions

        return output

    def update_policy_version(self, new_version: int):
        """
        Update the current policy version.

        Called by the trainer after syncing new weights to rollout workers.

        Args:
            new_version: New policy version number

        Note:
            In a full implementation, this would also:
            1. Trigger weight update on vLLM servers
            2. Flush/restart incomplete generations
            3. Rely on prefix caching to resume efficiently
        """
        old_version = self.current_policy_version
        self.current_policy_version = new_version

        logger.info(f"Policy version updated: {old_version} -> {new_version}")

        # TODO: Implement actual weight update coordination
        # This would involve:
        # 1. Calling update_weights API on vLLM HTTP servers
        # 2. Handling incomplete requests (abort or continue with new weights)
        # 3. Coordinating with AgentLoopWorkers to track version per request

    async def request_weight_update_async(self):
        """
        Request weight update on rollout servers via HTTP.

        This calls the weight update endpoints installed by vllm_server_extension.
        It will:
        1. Abort all in-flight requests on rollout servers
        2. Reset prefix cache
        3. Update weights via NCCL or disk

        IMPORTANT: This assumes vllm_server_extension is imported and hooks are installed.

        Returns:
            New policy version after update

        TODO for verl engine integration:
            - Add this to RolloutReplica.update_weights()
            - Make it part of standard rollout API
        """
        if self.weight_coordinator is None:
            logger.warning("Weight coordinator not set, skipping weight update")
            return self.current_policy_version

        logger.info(f"Requesting weight update (version {self.current_policy_version} -> {self.current_policy_version + 1})")

        # Use weight coordinator to trigger updates
        if self.weight_coordinator.config.update_method == "nccl":
            new_version = await self.weight_coordinator.update_weights_nccl()
        else:
            # Disk-based update requires checkpoint path
            # TODO: Get checkpoint path from trainer
            logger.error("Disk-based updates require checkpoint path - not implemented")
            return self.current_policy_version

        # Update local version
        self.current_policy_version = new_version

        return new_version
