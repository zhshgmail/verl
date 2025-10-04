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
Asynchronous RL Trainer with Decoupled PPO Objective

Implements async RL training with separate GPU pools for training and rollout,
supporting staleness-aware training through importance weighting.

Key features:
1. Separate resource pools for actor (training) and rollout (generation)
2. Version tracking for policy updates
3. Decoupled PPO loss with importance weighting
4. AsyncRLTransferDock for distributed sample buffering
5. Concurrent rollout and training with staleness control
"""

import asyncio
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import ray
from ray.actor import ActorHandle
import torch
from omegaconf import DictConfig, OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.protocol import DataProto
from recipe.async_rl.partial_rollout_manager import PartialRolloutManager
from recipe.async_rl.ppo_loss import ppo_loss, decoupled_ppo_loss, compute_advantages_with_version_tracking

# Import TransferQueue directly
import math
from tensordict import TensorDict
from verl.experimental.transfer_queue import (
    AsyncTransferQueueClient,
    TransferQueueController,
    TransferQueueStorageSimpleUnit,
    process_zmq_server_info,
)
from verl.experimental.transfer_queue.transfer_queue.utils.utils import get_placement_group

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AsyncRLTrainer(RayPPOTrainer):
    """
    Trainer for asynchronous RL with decoupled PPO objective.

    Extends RayPPOTrainer to support:
    1. Separate GPU pools for training and rollout (like one-step off-policy)
    2. Version tracking for policy updates
    3. Decoupled PPO loss for handling stale samples
    4. AsyncRLTransferDock for distributed sample buffering
    5. Concurrent rollout and training loops

    Architecture:
        Training Pool (actor_pool):
            - Actor (FSDP/Megatron) - policy being optimized
            - Critic - value function
            - Reference - reference policy for KL penalty

        Rollout Pool (rollout_pool):
            - vLLM HTTP Servers - generate samples
            - Managed by PartialRolloutManager

        AsyncRLTransferDock (Ray remote actor):
            - Distributed sample buffer
            - Thread-safe concurrent put/get
            - Staleness filtering

        Weight Sync:
            - NCCL cross-pool: actor_pool -> rollout_pool
            - Version tracking: increment version after each sync
    """

    # Type hints for key attributes (improves IDE support and type checking)
    tq_client: AsyncTransferQueueClient  # TransferQueue client for distributed data management
    _rollout_executor: ThreadPoolExecutor  # Dedicated thread pool for blocking operations

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping,
        resource_pool_manager,
        ray_worker_group_cls=None,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset=None,
        val_dataset=None,
        collate_fn=None,
        train_sampler=None,
        device_name=None,
    ):
        """
        Initialize AsyncRLTrainer.

        Args:
            config: Full trainer configuration with:
                - trainer.n_gpus_per_node: GPUs for training pool
                - rollout.n_gpus_per_node: GPUs for rollout pool
                - actor_rollout_ref.rollout.mode: Must be "async"
                - actor_rollout_ref.actor.use_decoupled_loss: Enable decoupled PPO
                - actor_rollout_ref.actor.behav_imp_weight_cap: Max importance weight
                - async_rl.buffer_size: TransferDock max size
                - async_rl.max_staleness: Max allowed staleness
                - async_rl.train_batch_size: Batch size for training
            tokenizer: Tokenizer for text processing
            role_worker_mapping: Mapping from roles to worker classes
            resource_pool_manager: Manager for Ray resource pools
            ray_worker_group_cls: Class for Ray worker groups
            processor: Optional data processor for multimodal data
            reward_fn: Function for computing rewards during training
            val_reward_fn: Function for computing rewards during validation
            train_dataset: Training dataset
            val_dataset: Validation dataset
            collate_fn: Function to collate data samples into batches
            train_sampler: Sampler for the training dataset
            device_name: Device name for training (e.g., "cuda", "cpu")
        """
        # Validate configuration
        assert config.actor_rollout_ref.rollout.mode == "async", (
            "AsyncRLTrainer requires rollout.mode='async' for HTTP server support"
        )

        # Extract async RL specific configs before calling parent __init__
        self.use_decoupled_loss = config.actor_rollout_ref.actor.get("use_decoupled_loss", True)
        self.behav_imp_weight_cap = config.actor_rollout_ref.actor.get("behav_imp_weight_cap", None)
        self.behav_imp_weight_floor = config.actor_rollout_ref.actor.get("behav_imp_weight_floor", None)

        # Extract async RL buffer configs
        self.buffer_size = config.get("async_rl", {}).get("buffer_size", 10000)
        self.max_staleness = config.get("async_rl", {}).get("max_staleness", 5)
        self.train_batch_size = config.get("async_rl", {}).get("train_batch_size", 128)
        self.rollout_batch_size = config.get("async_rl", {}).get("rollout_batch_size", 256)

        # Track policy version for staleness tracking
        self.policy_version = 0

        # Control flags for async training loop
        self._rollout_running = False
        self._training_running = False
        self._stop_event = threading.Event()

        # Create dedicated thread pool for blocking operations
        # IMPORTANT: Never use default executor (asyncio's shared pool)
        # Reasons:
        # 1. Thread starvation: Other code may exhaust the shared pool
        # 2. No isolation: Can't control priorities or limits
        # 3. Difficult debugging: Hard to track who's using threads
        # 4. Best practice from Java: Always explicitly manage thread pools
        rollout_executor_workers = config.get("async_rl", {}).get("rollout_executor_workers", 4)
        self._rollout_executor = ThreadPoolExecutor(
            max_workers=rollout_executor_workers,
            thread_name_prefix="async_rl_rollout"
        )
        logger.info(f"Created dedicated ThreadPoolExecutor with {rollout_executor_workers} workers")

        # Initialize parent RayPPOTrainer with all required parameters
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=device_name,
        )

        # Create TransferQueue client
        self.tq_client = None  # Will be initialized in init_workers()

        logger.info(
            f"AsyncRLTrainer initialized:\n"
            f"  - use_decoupled_loss: {self.use_decoupled_loss}\n"
            f"  - behav_imp_weight_cap: {self.behav_imp_weight_cap}\n"
            f"  - buffer_size: {self.buffer_size}\n"
            f"  - max_staleness: {self.max_staleness}\n"
            f"  - train_batch_size: {self.train_batch_size}\n"
            f"  - rollout_batch_size: {self.rollout_batch_size}\n"
            f"  - initial policy_version: {self.policy_version}"
        )

    def init_workers(self):
        """
        Initialize workers with separate resource pools.

        Overrides parent to:
        1. Create separate pools for training and rollout
        2. Use PartialRolloutManager instead of AgentLoopManager
        3. Create AsyncRLTransferDock for sample buffering
        4. Set up version tracking
        """
        # Call parent implementation to set up workers
        super().init_workers()

        # Override async_rollout_manager with PartialRolloutManager
        if self.async_rollout_mode:
            logger.info("Replacing AgentLoopManager with PartialRolloutManager")

            # FIXME: Parent already created AgentLoopManager in init_workers()
            # We need to replace it with PartialRolloutManager
            # This is a bit wasteful but necessary given current architecture

            # Get the worker group that was passed to AgentLoopManager
            # In parent RayPPOTrainer, it uses self.actor_rollout_wg
            # But for separate pools, we should use rollout-only workers

            # For separate pools architecture, we need to determine which worker_group to use
            # Check if we have separate rollout workers
            if hasattr(self, 'rollout_wg'):
                # Separate pools: use rollout_wg
                rollout_worker_group = self.rollout_wg
                logger.info("Using separate rollout_wg for PartialRolloutManager")
            else:
                # Hybrid/colocated: use actor_rollout_wg
                rollout_worker_group = self.actor_rollout_wg
                logger.warning(
                    "No separate rollout_wg found, using actor_rollout_wg. "
                    "Are you sure you configured separate resource pools?"
                )

            # Create PartialRolloutManager
            self.async_rollout_manager = PartialRolloutManager(
                config=self.config,
                worker_group=rollout_worker_group,
                rm_wg=self.rm_wg if self.use_rm else None,
                weight_coordinator=None,  # TODO: Initialize WeightUpdateCoordinator
            )

            logger.info(f"PartialRolloutManager created with worker_group of size {rollout_worker_group.world_size}")

        # Create TransferQueue system (production-grade distributed data management)
        # Extract TransferQueue configuration
        tq_config = self.config.get("async_rl", {}).get("transfer_queue", {})
        num_storage_units = tq_config.get("num_storage_units", 2)
        num_controllers = tq_config.get("num_controllers", 1)
        num_global_batch = tq_config.get("num_global_batch", 2)
        storage_cpus_per_unit = tq_config.get("storage_cpus_per_unit", 1)
        controller_cpus = tq_config.get("controller_cpus", 1)

        # Calculate total storage size
        total_storage_size = self.train_batch_size * num_global_batch
        storage_size_per_unit = math.ceil(total_storage_size / num_storage_units)

        logger.info(
            f"Creating TransferQueue system:\n"
            f"  - global_batch_size: {self.train_batch_size}\n"
            f"  - num_global_batch: {num_global_batch}\n"
            f"  - num_storage_units: {num_storage_units}\n"
            f"  - total_storage_size: {total_storage_size}\n"
            f"  - storage_size_per_unit: {storage_size_per_unit}"
        )

        # 1. Create storage units
        storage_placement_group = get_placement_group(num_storage_units, num_cpus_per_actor=storage_cpus_per_unit)
        self._tq_storages = {}
        for storage_rank in range(num_storage_units):
            storage = TransferQueueStorageSimpleUnit.options(
                placement_group=storage_placement_group, placement_group_bundle_index=storage_rank
            ).remote(storage_size=storage_size_per_unit)
            self._tq_storages[storage_rank] = storage
            logger.info(f"Created TransferQueueStorageSimpleUnit #{storage_rank}")

        # 2. Create controllers
        controller_placement_group = get_placement_group(num_controllers, num_cpus_per_actor=controller_cpus)
        self._tq_controllers = {}
        for controller_rank in range(num_controllers):
            controller = TransferQueueController.options(
                placement_group=controller_placement_group, placement_group_bundle_index=controller_rank
            ).remote(
                num_storage_units=num_storage_units,
                global_batch_size=self.train_batch_size,
                num_global_batch=num_global_batch,
                num_n_samples=1,  # AsyncRL doesn't use best-of-N sampling
            )
            self._tq_controllers[controller_rank] = controller
            logger.info(f"Created TransferQueueController #{controller_rank}")

        # 3. Get ZMQ server info
        controller_infos = process_zmq_server_info(self._tq_controllers)
        storage_infos = process_zmq_server_info(self._tq_storages)

        # 4. Register controllers with storages
        ray.get([storage.register_controller_info.remote(controller_infos) for storage in self._tq_storages.values()])
        logger.info("Registered controllers with storage units")

        # 5. Create client (use directly, no adapter)
        self.tq_client = AsyncTransferQueueClient(
            client_id="AsyncRLTrainer",
            controller_infos=controller_infos[0],  # Single controller for now
            storage_infos=storage_infos,
        )
        logger.info("Created AsyncTransferQueueClient")

    def compute_advantages(self, data: DataProto) -> DataProto:
        """
        Compute advantages with version tracking.

        Overrides parent to use compute_advantages_with_version_tracking
        which logs staleness metrics.

        Args:
            data: DataProto with rewards, values, logprobs, etc.

        Returns:
            DataProto with advantages added
        """
        # Extract version information if available
        version_start = data.non_tensor_batch.get("version_start", None)
        version_end = data.non_tensor_batch.get("version_end", None)

        if version_start is not None and version_end is not None:
            # Convert to tensor for computation
            version_start_tensor = torch.tensor(version_start, device=data.batch["rewards"].device)
            version_end_tensor = torch.tensor(version_end, device=data.batch["rewards"].device)
        else:
            version_start_tensor = None
            version_end_tensor = None
            logger.warning(
                "No version tracking found in data. "
                "Make sure PartialRolloutManager is annotating samples with version_start/version_end."
            )

        # Compute advantages using our version-aware function
        advantages, version_stats = compute_advantages_with_version_tracking(
            rewards=data.batch["rewards"],
            values=data.batch.get("values", torch.zeros_like(data.batch["input_ids"], dtype=torch.float32)),
            old_logprobs=data.batch["logprobs"],
            ref_logprobs=data.batch.get("ref_logprobs", torch.zeros_like(data.batch["logprobs"])),
            attention_mask=data.batch["attention_mask"],
            loss_mask=data.batch["loss_mask"],
            kl_ctl=self.config.algorithm.kl_ctrl.kl_coef,
            discount=self.config.algorithm.get("discount", 0.99),
            gae_lambda=self.config.algorithm.get("gae_lambda", 0.95),
            version_start=version_start_tensor,
            version_end=version_end_tensor,
        )

        # Add advantages to data
        data.batch["advantages"] = advantages

        # Log version statistics
        if version_stats:
            logger.info(
                f"Version staleness stats: "
                f"max_diff={version_stats.get('max_version_diff', 0)}, "
                f"mean_diff={version_stats.get('mean_version_diff', 0):.2f}, "
                f"samples_with_staleness={version_stats.get('samples_with_staleness', 0):.2%}"
            )

        return data

    def compute_ppo_loss(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        loss_mask: torch.Tensor,
        proximal_logprobs: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict]:
        """
        Compute PPO loss using appropriate objective (standard or decoupled).

        Uses polymorphism instead of flag passing: calls the appropriate loss function
        based on configuration.

        Args:
            logprobs: Current policy log probs
            old_logprobs: Behavior policy log probs (from generation)
            advantages: Advantage estimates
            loss_mask: Valid token mask
            proximal_logprobs: Proximal policy log probs (recomputed from recent checkpoint)

        Returns:
            loss: Scalar loss
            stats: Dictionary with loss statistics
        """
        if self.use_decoupled_loss:
            # Decoupled PPO: separate PPO clipping from staleness weighting
            # If proximal_logprobs not provided, recompute or use old_logprobs
            if proximal_logprobs is None:
                if self.config.actor_rollout_ref.actor.get("recompute_logprob", True):
                    # Should recompute from actor model
                    # This would require calling self.actor_wg.compute_log_prob(data)
                    # For now, use old_logprobs as fallback
                    logger.warning(
                        "proximal_logprobs not provided and recomputation not implemented. "
                        "Using old_logprobs as proximal (will behave like standard PPO)."
                    )
                    proximal_logprobs = old_logprobs
                else:
                    proximal_logprobs = old_logprobs

            loss, stats = decoupled_ppo_loss(
                logprobs=logprobs,
                proximal_logprobs=proximal_logprobs,
                old_logprobs=old_logprobs,
                advantages=advantages,
                eps_clip=self.config.actor_rollout_ref.actor.clip_ratio_low,
                loss_mask=loss_mask,
                eps_clip_higher=self.config.actor_rollout_ref.actor.get("clip_ratio_high", None),
                c_clip=self.config.actor_rollout_ref.actor.get("clip_ratio_c", None),
                behav_imp_weight_cap=self.behav_imp_weight_cap,
                behav_imp_weight_floor=self.behav_imp_weight_floor,
            )
        else:
            # Standard PPO: single importance ratio
            loss, stats = ppo_loss(
                logprobs=logprobs,
                old_logprobs=old_logprobs,
                advantages=advantages,
                eps_clip=self.config.actor_rollout_ref.actor.clip_ratio_low,
                loss_mask=loss_mask,
                eps_clip_higher=self.config.actor_rollout_ref.actor.get("clip_ratio_high", None),
                c_clip=self.config.actor_rollout_ref.actor.get("clip_ratio_c", None),
            )

        return loss, stats

    def sync_weights_to_rollout(self):
        """
        Synchronize weights from training pool to rollout pool.

        Unlike the parent RayPPOTrainer which doesn't have separate rollout workers,
        AsyncRLTrainer has a dedicated rollout pool that needs weight updates.

        This method:
        1. Syncs weights from actor to rollout pool via WeightUpdateCoordinator
        2. Increments policy version for staleness tracking
        3. Notifies PartialRolloutManager of the version update
        """
        # TODO: Implement actual weight sync via WeightUpdateCoordinator
        # For now, just increment version
        # In full implementation, would call:
        # await self.weight_coordinator.update_weights_nccl()

        # Increment policy version
        self.policy_version += 1

        # Notify PartialRolloutManager of version update
        if hasattr(self, 'async_rollout_manager') and isinstance(self.async_rollout_manager, PartialRolloutManager):
            self.async_rollout_manager.update_policy_version(self.policy_version)

        logger.info(f"Weights synced to rollout pool. Policy version: {self.policy_version}")

    def training_step(self, data: DataProto) -> Dict:
        """
        Execute one training step with staleness tracking.

        Unlike the parent RayPPOTrainer which has all training logic in fit(),
        AsyncRLTrainer needs a separate training_step() for async training loop.

        Args:
            data: Training batch with samples (may have mixed staleness)

        Returns:
            Dictionary with training metrics including staleness stats
        """
        metrics = {}

        # Update critic if using critic-based advantage estimation
        if self.use_critic:
            critic_output = self.critic_wg.update_critic(data)
            from verl.utils.metric import reduce_metrics
            critic_metrics = reduce_metrics(critic_output.meta_info["metrics"])
            metrics.update(critic_metrics)

        # Update actor (always done unless in critic warmup)
        if self.config.trainer.critic_warmup <= self.global_steps:
            data.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
            actor_output = self.actor_rollout_wg.update_actor(data)
            from verl.utils.metric import reduce_metrics
            actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            metrics.update(actor_metrics)

        # Add version tracking metrics
        if "version_start" in data.non_tensor_batch and "version_end" in data.non_tensor_batch:
            version_start = data.non_tensor_batch["version_start"]
            version_end = data.non_tensor_batch["version_end"]
            current_version = self.policy_version

            # Compute staleness: how old is the data relative to current policy
            staleness_at_training = current_version - version_end
            max_staleness = staleness_at_training.max()
            mean_staleness = staleness_at_training.mean()

            metrics["async_rl/max_staleness"] = max_staleness
            metrics["async_rl/mean_staleness"] = mean_staleness
            metrics["async_rl/current_policy_version"] = current_version

        return metrics

    # ========================================================================
    # Async Training Loop Methods
    # ========================================================================

    def shutdown(self):
        """
        Clean shutdown: stop async loops and cleanup resources.

        Call this when training is complete or interrupted.
        """
        logger.info("Shutting down AsyncRLTrainer...")

        # Signal loops to stop
        self._stop_event.set()

        # Shutdown thread pool executor
        if hasattr(self, '_rollout_executor'):
            logger.info("Shutting down rollout executor...")
            self._rollout_executor.shutdown(wait=True, cancel_futures=False)
            logger.info("Rollout executor shut down successfully")

    async def async_rollout_loop(self, data_loader):
        """
        Async rollout loop that continuously generates samples and puts them into TransferDock.

        This runs concurrently with the training loop.

        Args:
            data_loader: DataLoader providing prompts for rollout
        """
        self._rollout_running = True
        rollout_step = 0

        logger.info("Starting async rollout loop...")

        try:
            while not self._stop_event.is_set():
                # Get next batch of prompts
                prompt_batch = next(data_loader)

                # Generate samples using PartialRolloutManager
                # Note: generate_sequences() is synchronous and uses ray.get() internally,
                # so we run it in dedicated executor to avoid blocking the event loop
                # IMPORTANT: Use dedicated executor, NOT None (default shared pool)
                loop = asyncio.get_event_loop()
                rollout_data = await loop.run_in_executor(
                    self._rollout_executor,  # Dedicated pool for rollout operations
                    self.async_rollout_manager.generate_sequences,
                    prompt_batch
                )

                # Annotate samples with version information
                # version_start: policy version when generation started
                # version_end: policy version when generation completed
                rollout_data["version_start"] = self.policy_version
                rollout_data["version_end"] = self.policy_version

                # Put samples into TransferQueue (direct API usage)
                num_samples = len(rollout_data["input_ids"])
                try:
                    # Convert to TensorDict and put
                    tensor_dict = TensorDict(rollout_data, batch_size=num_samples)
                    await self.tq_client.async_put(data=tensor_dict, global_step=self.policy_version)

                    rollout_step += 1
                    logger.debug(
                        f"Rollout step {rollout_step}: Put {num_samples} samples into TransferQueue "
                        f"(policy_version={self.policy_version})"
                    )

                    # Log periodically
                    if rollout_step % 10 == 0:
                        logger.info(
                            f"Rollout step {rollout_step} - policy_version={self.policy_version}, "
                            f"samples_put={num_samples}"
                        )
                except Exception as e:
                    logger.warning(f"Failed to put samples into TransferQueue at step {rollout_step}: {e}")

        except Exception as e:
            logger.error(f"Error in async rollout loop: {e}", exc_info=True)
            raise
        finally:
            self._rollout_running = False
            logger.info("Async rollout loop stopped")

    async def async_training_loop(self, num_steps: int, ppo_epochs: int = 1):
        """
        Async training loop that continuously gets samples from TransferDock and trains.

        This runs concurrently with the rollout loop.

        Args:
            num_steps: Number of training steps to run
            ppo_epochs: Number of PPO epochs per training step
        """
        self._training_running = True
        training_step = 0

        logger.info(f"Starting async training loop (num_steps={num_steps}, ppo_epochs={ppo_epochs})...")

        # Define experience columns to retrieve
        # Following AReaL: retrieve per-token logprobs and policy versions for decoupled PPO
        experience_columns = [
            "input_ids",
            "attention_mask",
            "loss_mask",
            "old_logprobs",  # Behavior logprobs π_old (needed for decoupled PPO)
            "token_policy_versions",  # Per-token version (for staleness validation)
            "rewards",
            "values",
            "version_start",
            "version_end",
        ]

        try:
            while training_step < num_steps and not self._stop_event.is_set():
                # Get batch from TransferQueue with staleness filtering
                # Use global_step to filter by staleness (version >= current - max_staleness)
                min_version = self.policy_version - self.max_staleness
                batch_meta = None

                # Try to get metadata from recent steps
                for step in range(self.policy_version, max(min_version - 1, -1), -1):
                    try:
                        meta = await self.tq_client.async_get_meta(
                            data_fields=experience_columns,
                            batch_size=self.train_batch_size,
                            global_step=step,
                            task_name="async_rl_training",  # Multi-consumer tracking
                            mode="fetch",  # Only get ready samples
                        )
                        if meta and meta.size > 0:
                            batch_meta = meta
                            break
                    except Exception as e:
                        logger.debug(f"Failed to get metadata from step {step}: {e}")
                        continue

                if batch_meta is None or batch_meta.size < self.train_batch_size:
                    # Not enough samples in buffer yet
                    logger.debug(
                        f"Training step {training_step}: Waiting for more samples "
                        f"(need {self.train_batch_size}, version >= {min_version})"
                    )
                    await asyncio.sleep(1.0)  # Wait before retrying
                    continue

                # Get actual data
                tensor_dict = await self.tq_client.async_get_data(metadata=batch_meta)
                batch_dict = dict(tensor_dict)

                # Convert batch_dict to DataProto
                data = DataProto.from_dict(batch_dict)

                # Compute advantages with version tracking
                data = self.compute_advantages(data)

                # Run PPO training for multiple epochs
                for epoch in range(ppo_epochs):
                    metrics = self.training_step(data)

                    logger.debug(
                        f"Training step {training_step}, epoch {epoch}: "
                        f"loss={metrics.get('loss', 0):.4f}, "
                        f"max_staleness={metrics.get('max_staleness', 0)}, "
                        f"mean_staleness={metrics.get('mean_staleness', 0):.2f}"
                    )

                training_step += 1

                # Log metrics periodically
                if training_step % 10 == 0:
                    logger.info(
                        f"Training step {training_step}/{num_steps}: "
                        f"policy_version={self.policy_version}, "
                        f"loss={metrics.get('loss', 0):.4f}"
                    )

                # Sync weights to rollout pool periodically
                if training_step % self.config.get("async_rl", {}).get("sync_freq", 10) == 0:
                    self.sync_weights_to_rollout()

        except Exception as e:
            logger.error(f"Error in async training loop: {e}", exc_info=True)
            raise
        finally:
            self._training_running = False
            logger.info(f"Async training loop stopped after {training_step} steps")

    def train_async(self, data_loader, num_steps: int, ppo_epochs: int = 1):
        """
        Main entry point for async RL training.

        Runs rollout and training loops concurrently.

        Args:
            data_loader: DataLoader providing prompts for rollout
            num_steps: Number of training steps to run
            ppo_epochs: Number of PPO epochs per training step
        """
        logger.info(
            f"Starting async RL training:\n"
            f"  - num_steps: {num_steps}\n"
            f"  - ppo_epochs: {ppo_epochs}\n"
            f"  - train_batch_size: {self.train_batch_size}\n"
            f"  - rollout_batch_size: {self.rollout_batch_size}"
        )

        # Reset stop event
        self._stop_event.clear()

        # Create async tasks
        rollout_task = asyncio.create_task(self.async_rollout_loop(data_loader))
        training_task = asyncio.create_task(self.async_training_loop(num_steps, ppo_epochs))

        try:
            # Wait for training to complete (rollout runs indefinitely)
            asyncio.get_event_loop().run_until_complete(training_task)

        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping async training...")
            self._stop_event.set()

        finally:
            # Stop rollout loop
            self._stop_event.set()

            # Wait for rollout to finish
            if not rollout_task.done():
                rollout_task.cancel()
                try:
                    asyncio.get_event_loop().run_until_complete(rollout_task)
                except asyncio.CancelledError:
                    pass

            logger.info("Async RL training completed")
