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
Weight Update Coordination for Async RL

This module handles coordinating weight updates between training workers
and rollout servers, supporting both disk-based and NCCL-based synchronization.

IMPORTANT: This integrates with verl's vLLMReplica to get server addresses.
         No need for separate name resolution - uses verl's existing infrastructure.

TODO for verl engine integration:
    1. Move to verl/trainer/ppo/weight_sync.py
    2. Add to RayPPOTrainer as optional feature
    3. Support async weight updates as trainer configuration
    4. Add metrics tracking for update frequency/latency
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

import aiohttp
import torch

if TYPE_CHECKING:
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


# ============================================================================
# Helper Functions
# ============================================================================


async def get_rollout_server_addresses(rollout_replicas: List["vLLMReplica"]) -> List[str]:
    """
    Get HTTP server addresses from vLLMReplica instances.

    This uses verl's existing server address management instead of
    needing a separate name resolver like AReaL.

    Args:
        rollout_replicas: List of vLLMReplica instances

    Returns:
        List of HTTP server URLs (e.g., ["http://host1:port1", "http://host2:port2"])

    FIXME: Currently requires async call to get addresses
           Should be cached or synchronously accessible

    TODO for verl engine integration:
        - Add RolloutReplica.get_server_url() synchronous method
        - Cache addresses after launch_servers() completes
        - Support dynamic address discovery if servers restart
    """
    addresses = []
    for replica in rollout_replicas:
        # vLLMReplica stores server address as "host:port"
        server_addr = replica._server_address
        if server_addr:
            # Ensure http:// prefix
            if not server_addr.startswith("http://"):
                server_addr = f"http://{server_addr}"
            addresses.append(server_addr)
        else:
            logger.warning(f"Replica has no server address, skipping")

    logger.info(f"Retrieved {len(addresses)} rollout server addresses from vLLMReplica")
    return addresses


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class WeightUpdateConfig:
    """Configuration for weight updates."""

    # Which method to use: "nccl" or "disk"
    update_method: str = "nccl"

    # For NCCL updates
    nccl_backend: str = "nccl"  # or "hccl" for Ascend
    nccl_group_name: str = "async_rl_weight_update"

    # For disk updates
    checkpoint_dir: Optional[str] = None

    # HTTP timeout for API calls
    request_timeout: int = 300  # 5 minutes

    # Max retries for failed requests
    max_retries: int = 3


@dataclass
class NCCLParamSpec:
    """Specification for a parameter to be synced via NCCL."""
    name: str
    dtype: str
    shape: List[int]


# ============================================================================
# Weight Update Coordinator
#
# FIXME: This should be part of verl.trainer.ppo.ray_trainer
#        Currently in recipe to avoid engine changes
#
# TODO for engine integration:
#     - Add as optional feature to RayPPOTrainer via config flag
#     - Integrate with existing sync_rollout_weights() method
#     - Add metrics/logging for update frequency and latency
#     - Support both sync and async update modes
# ============================================================================

class WeightUpdateCoordinator:
    """
    Coordinates weight updates from training workers to rollout servers.

    Supports two update methods:
    1. NCCL: Efficient for cross-GPU updates via collective communication
    2. Disk: Save checkpoint to shared storage, rollout servers load it

    FIXME: Currently sends HTTP requests to custom endpoints
           Should use verl's worker group abstraction instead

    TODO for engine integration:
        - Use RayWorkerGroup.update_weights() instead of HTTP
        - Add to RayPPOTrainer as self.weight_update_coordinator
        - Make update policy configurable (every N steps, every epoch, etc.)
    """

    def __init__(
        self,
        config: WeightUpdateConfig,
        rollout_server_addresses: List[str],
        policy_version: int = 0,
    ):
        """
        Initialize weight update coordinator.

        Args:
            config: Weight update configuration
            rollout_server_addresses: List of HTTP server URLs (e.g., ["http://host:port"])
            policy_version: Initial policy version number
        """
        self.config = config
        self.rollout_server_addresses = rollout_server_addresses
        self.policy_version = policy_version

        self.nccl_initialized = False

        logger.info(
            f"WeightUpdateCoordinator initialized:\n"
            f"  - Update method: {config.update_method}\n"
            f"  - Rollout servers: {len(rollout_server_addresses)}\n"
            f"  - Initial version: {policy_version}"
        )

    async def init_nccl_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
    ):
        """
        Initialize NCCL process group for weight updates.

        This must be called once before using NCCL-based updates.

        Args:
            master_address: NCCL master address
            master_port: NCCL master port
            rank_offset: Rank offset for rollout workers
            world_size: Total world size (training + rollout workers)

        FIXME: Currently sends HTTP POST to custom endpoint
               Should use verl's distributed initialization instead

        TODO for engine integration:
            - Use verl's NCCL group management
            - Integrate with ResourcePoolManager
            - Auto-discover addresses from RayWorkerGroup
        """
        if self.nccl_initialized:
            logger.warning("NCCL group already initialized, skipping")
            return

        logger.info(f"Initializing NCCL group for {len(self.rollout_server_addresses)} servers")

        async with aiohttp.ClientSession() as session:
            tasks = []
            for addr in self.rollout_server_addresses:
                payload = {
                    "master_address": master_address,
                    "master_port": master_port,
                    "rank_offset": rank_offset,
                    "world_size": world_size,
                    "backend": self.config.nccl_backend,
                    "group_name": self.config.nccl_group_name,
                }
                task = self._post_request(
                    session,
                    f"{addr}/async_rl/init_nccl_group",
                    payload,
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for failures
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    raise RuntimeError(
                        f"Failed to initialize NCCL on server {self.rollout_server_addresses[i]}: {result}"
                    )

        self.nccl_initialized = True
        logger.info("NCCL group initialized successfully")

    async def set_weight_metadata(self, param_specs: List[NCCLParamSpec]):
        """
        Set metadata for parameters to be synced via NCCL.

        This must be called before update_weights_nccl().

        Args:
            param_specs: List of parameter specifications

        FIXME: Should be automatic from model inspection
               Currently requires manual specification
        """
        if not self.nccl_initialized:
            raise RuntimeError("NCCL group not initialized. Call init_nccl_group() first.")

        logger.info(f"Setting weight metadata for {len(param_specs)} parameters")

        async with aiohttp.ClientSession() as session:
            payload = {
                "names": [spec.name for spec in param_specs],
                "dtypes": [spec.dtype for spec in param_specs],
                "shapes": [spec.shape for spec in param_specs],
                "group_name": self.config.nccl_group_name,
            }

            tasks = [
                self._post_request(session, f"{addr}/async_rl/set_weight_meta", payload)
                for addr in self.rollout_server_addresses
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for failures
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    raise RuntimeError(
                        f"Failed to set metadata on server {self.rollout_server_addresses[i]}: {result}"
                    )

        logger.info("Weight metadata set successfully")

    async def update_weights_nccl(self):
        """
        Update weights via NCCL collective communication.

        Training workers should call torch.distributed.broadcast() or send()
        at the same time this is called, to send weights to rollout workers.

        Returns:
            New policy version number

        FIXME: Coordination between this call and training worker NCCL send
               is manual and error-prone. Should be atomic operation.

        TODO for engine integration:
            - Make this part of RayPPOTrainer.sync_rollout_weights()
            - Auto-coordinate NCCL send from training workers
            - Add proper error handling and retry logic
        """
        if not self.nccl_initialized:
            raise RuntimeError("NCCL group not initialized. Call init_nccl_group() first.")

        logger.info(f"Triggering NCCL weight update (version {self.policy_version} -> {self.policy_version + 1})")

        async with aiohttp.ClientSession() as session:
            tasks = [
                self._post_request(session, f"{addr}/async_rl/update_weights_nccl", {})
                for addr in self.rollout_server_addresses
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for failures
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Failed to update weights on server {self.rollout_server_addresses[i]}: {result}"
                    )
                    raise

        self.policy_version += 1
        logger.info(f"NCCL weight update completed. New version: {self.policy_version}")

        return self.policy_version

    async def update_weights_disk(self, checkpoint_path: str):
        """
        Update weights by loading from disk checkpoint.

        Args:
            checkpoint_path: Path to checkpoint (must be accessible from rollout servers)

        Returns:
            New policy version number

        TODO for engine integration:
            - Integrate with verl's checkpoint saving
            - Add validation that checkpoint exists before triggering update
            - Support async checkpoint saving to avoid blocking training
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Triggering disk weight update from {checkpoint_path}")

        async with aiohttp.ClientSession() as session:
            payload = {"model_path": checkpoint_path}
            tasks = [
                self._post_request(session, f"{addr}/async_rl/update_weights", payload)
                for addr in self.rollout_server_addresses
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for failures
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Failed to update weights on server {self.rollout_server_addresses[i]}: {result}"
                    )
                    raise

        self.policy_version += 1
        logger.info(f"Disk weight update completed. New version: {self.policy_version}")

        return self.policy_version

    async def _post_request(
        self,
        session: aiohttp.ClientSession,
        url: str,
        payload: dict,
    ):
        """
        Send POST request with retry logic.

        Args:
            session: aiohttp session
            url: Request URL
            payload: JSON payload

        Returns:
            Response JSON

        Raises:
            RuntimeError: If all retries fail
        """
        for attempt in range(self.config.max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
                async with session.post(url, json=payload, timeout=timeout) as response:
                    response.raise_for_status()
                    return await response.json()
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise RuntimeError(f"Request failed after {self.config.max_retries} retries: {e}")
                logger.warning(f"Request to {url} failed (attempt {attempt + 1}), retrying: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
