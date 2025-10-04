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
4. Support for chunked generation (via PartialRolloutManager)
"""

import logging
import os
from typing import Dict, Optional

import ray
import torch
from omegaconf import DictConfig, OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.protocol import DataProto
from recipe.async_rl.partial_rollout_manager import PartialRolloutManager
from recipe.async_rl.ppo_loss import decoupled_ppo_loss, compute_advantages_with_version_tracking

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AsyncRLTrainer(RayPPOTrainer):
    """
    Trainer for asynchronous RL with decoupled PPO objective.

    Extends RayPPOTrainer to support:
    1. Separate GPU pools for training and rollout (like one-step off-policy)
    2. Version tracking for policy updates
    3. Decoupled PPO loss for handling stale samples
    4. Chunked generation with weight updates

    Architecture:
        Training Pool (actor_pool):
            - Actor (FSDP/Megatron) - policy being optimized
            - Critic - value function
            - Reference - reference policy for KL penalty

        Rollout Pool (rollout_pool):
            - vLLM HTTP Servers - generate samples
            - Managed by PartialRolloutManager

        Weight Sync:
            - NCCL cross-pool: actor_pool -> rollout_pool
            - Version tracking: increment version after each sync
    """

    def __init__(self, config: DictConfig):
        """
        Initialize AsyncRLTrainer.

        Args:
            config: Full trainer configuration with:
                - trainer.n_gpus_per_node: GPUs for training pool
                - rollout.n_gpus_per_node: GPUs for rollout pool
                - actor_rollout_ref.rollout.mode: Must be "async"
                - actor_rollout_ref.rollout.new_tokens_per_chunk: Chunk size
                - actor_rollout_ref.actor.use_decoupled_loss: Enable decoupled PPO
                - actor_rollout_ref.actor.behav_imp_weight_cap: Max importance weight
        """
        # Validate configuration
        assert config.actor_rollout_ref.rollout.mode == "async", (
            "AsyncRLTrainer requires rollout.mode='async' for HTTP server support"
        )

        # Extract async RL specific configs
        self.new_tokens_per_chunk = config.actor_rollout_ref.rollout.get("new_tokens_per_chunk", 64)
        self.use_decoupled_loss = config.actor_rollout_ref.actor.get("use_decoupled_loss", True)
        self.behav_imp_weight_cap = config.actor_rollout_ref.actor.get("behav_imp_weight_cap", None)

        # Track policy version for staleness tracking
        self.policy_version = 0

        # Initialize parent RayPPOTrainer
        super().__init__(config)

        logger.info(
            f"AsyncRLTrainer initialized:\n"
            f"  - new_tokens_per_chunk: {self.new_tokens_per_chunk}\n"
            f"  - use_decoupled_loss: {self.use_decoupled_loss}\n"
            f"  - behav_imp_weight_cap: {self.behav_imp_weight_cap}\n"
            f"  - initial policy_version: {self.policy_version}"
        )

    def init_workers(self):
        """
        Initialize workers with separate resource pools.

        Overrides parent to:
        1. Create separate pools for training and rollout
        2. Use PartialRolloutManager instead of AgentLoopManager
        3. Set up version tracking
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
                new_tokens_per_chunk=self.new_tokens_per_chunk,
            )

            logger.info(f"PartialRolloutManager created with worker_group of size {rollout_worker_group.world_size}")

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
        Compute PPO loss using decoupled objective.

        Overrides parent to use decoupled_ppo_loss which handles staleness.

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
        # If proximal_logprobs not provided, recompute or use old_logprobs
        if proximal_logprobs is None:
            if self.use_decoupled_loss and self.config.actor_rollout_ref.actor.get("recompute_logprob", True):
                # Should recompute from actor model
                # This would require calling self.actor_wg.compute_log_prob(data)
                # For now, use old_logprobs as fallback
                logger.warning(
                    "proximal_logprobs not provided and recomputation not implemented. "
                    "Using old_logprobs as proximal (will behave like standard PPO)."
                )
                proximal_logprobs = old_logprobs
            else:
                # Standard PPO: use old_logprobs as proximal
                proximal_logprobs = old_logprobs

        # Compute decoupled PPO loss
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
            use_decoupled_loss=self.use_decoupled_loss,
        )

        return loss, stats

    def sync_weights_to_rollout(self):
        """
        Synchronize weights from training pool to rollout pool.

        Overrides parent to add version tracking.
        """
        # Call parent implementation (NCCL sync)
        super().sync_rollout_weights()

        # Increment policy version
        self.policy_version += 1

        # Notify PartialRolloutManager of version update
        if hasattr(self, 'async_rollout_manager') and isinstance(self.async_rollout_manager, PartialRolloutManager):
            self.async_rollout_manager.update_policy_version(self.policy_version)

        logger.info(f"Weights synced to rollout pool. Policy version: {self.policy_version}")

    def training_step(self, data: DataProto) -> Dict:
        """
        Execute one training step with staleness tracking.

        Args:
            data: Training batch with samples (may have mixed staleness)

        Returns:
            Dictionary with training metrics including staleness stats
        """
        # Call parent training_step
        metrics = super().training_step(data)

        # Add version tracking metrics
        if "version_start" in data.non_tensor_batch and "version_end" in data.non_tensor_batch:
            version_start = data.non_tensor_batch["version_start"]
            version_end = data.non_tensor_batch["version_end"]
            current_version = self.policy_version

            # Compute staleness: how old is the data relative to current policy
            staleness_at_training = current_version - version_end
            max_staleness = staleness_at_training.max()
            mean_staleness = staleness_at_training.mean()

            metrics["max_staleness"] = max_staleness
            metrics["mean_staleness"] = mean_staleness
            metrics["current_policy_version"] = current_version

        return metrics
