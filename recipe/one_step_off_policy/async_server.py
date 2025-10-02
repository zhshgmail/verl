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
AsyncLLMServerManager for one-step off-policy training recipe.

This is a thin compatibility wrapper that provides the interface expected by
RayPPOTrainer's async rollout mode while delegating to the rollout worker group.

Design Rationale:
- One-step off-policy uses SEPARATE GPU pools for training and rollout
- Rollout workers run continuously on dedicated GPUs (not alternating with training)
- No wake_up/sleep needed since rollout GPUs don't share with training
- No RolloutReplica infrastructure needed - workers already initialized by trainer
"""

from omegaconf import DictConfig

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup


class AsyncLLMServerManager:
    """
    Simple wrapper for async rollout in one-step off-policy training.

    This class exists purely for interface compatibility with RayPPOTrainer,
    which expects async_rollout_manager to have a generate_sequences() method.

    In one-step off-policy:
    - Rollout workers use dedicated GPUs (separate from training)
    - Workers are already initialized by the trainer
    - No mode switching or memory management needed
    - Just delegates to worker_group.generate_sequences()

    The actual async generation happens via rollout_wg.async_generate_sequences()
    called directly by OneStepOffRayTrainer.
    """

    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup):
        """Initialize AsyncLLMServerManager.

        Args:
            config (DictConfig): Full trainer config (kept for interface compatibility).
            worker_group (RayWorkerGroup): Rollout worker group with pre-initialized workers.
        """
        self.config = config
        self.worker_group = worker_group

    def generate_sequences(self, prompts: DataProto, **sampling_params) -> DataProto:
        """
        Generate sequences by delegating to the worker group.

        This method exists for interface compatibility with RayPPOTrainer's async mode.
        The parent class calls async_rollout_manager.generate_sequences() which delegates
        to worker_group.async_generate_sequences() where the actual vLLM generation happens.

        Args:
            prompts (DataProto): Input prompts.
            **sampling_params: Additional sampling parameters.

        Returns:
            DataProto: Generated sequences.
        """
        return self.worker_group.async_generate_sequences(prompts, **sampling_params)
