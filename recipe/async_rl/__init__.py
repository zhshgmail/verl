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
Asynchronous RL Recipe with Decoupled PPO Objective

Inspired by AReaL paper (arXiv:2505.24298).

See README.md for documentation.
"""

from .ray_trainer import AsyncRLTrainer
from .partial_rollout_manager import PartialRolloutManager
from .ppo_loss import ppo_loss, decoupled_ppo_loss, compute_advantages_with_version_tracking

__all__ = [
    "AsyncRLTrainer",
    "PartialRolloutManager",
    "ppo_loss",
    "decoupled_ppo_loss",
    "compute_advantages_with_version_tracking",
]
