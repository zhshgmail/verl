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
Main entry point for Asynchronous RL Training with Decoupled PPO.

This recipe implements async RL training inspired by AReaL (arXiv:2505.24298),
with separate GPU pools for training and rollout, supporting staleness-aware
training through importance weighting.

Usage:
    python -m recipe.async_rl.main_ppo \\
        data.train_files=path/to/data.parquet \\
        actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \\
        trainer.n_gpus_per_node=6 \\
        rollout.n_gpus_per_node=2 \\
        actor_rollout_ref.rollout.mode=async \\
        actor_rollout_ref.rollout.new_tokens_per_chunk=64 \\
        actor_rollout_ref.actor.use_decoupled_loss=True
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_reference_policy
from verl.utils.config import validate_config

from .ray_trainer import AsyncRLTrainer


def need_critic(config):
    """Check if critic is needed based on algorithm."""
    return config.algorithm.adv_estimator in ["gae", "vtrace"]


@hydra.main(config_path="config", config_name="async_rl_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    """Run asynchronous RL training with decoupled PPO objective."""
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"Ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    # Create remote task runner
    if (
        config.global_profiler.tool == "nsys"
        and OmegaConf.select(config.global_profiler, "steps") is not None
        and len(OmegaConf.select(config.global_profiler, "steps")) > 0
    ):
        nsight_options = OmegaConf.to_container(config.global_profiler.tool_config.nsys.controller_nsight_options)
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()

    ray.get(runner.run.remote(config))

    # Optional: save timeline trace for performance analysis
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)
class TaskRunner:
    """Remote task runner for async RL training."""

    def run(self, config):
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Validate async mode is enabled
        assert config.actor_rollout_ref.rollout.mode == "async", (
            "AsyncRL recipe requires rollout.mode='async'. "
            "This enables HTTP servers for chunked generation and weight updates."
        )

        # Import workers based on strategy
        if config.actor_rollout_ref.actor.strategy == "fsdp2":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

            # For async_rl, we use verl's workers since we leverage HTTP servers
            # No need for custom AsyncActorRolloutRefWorker like one-step off-policy
            # The async behavior is handled by PartialRolloutManager + HTTP servers
            from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMAsyncRollout

            # Define rollout worker class
            # We can use a simple wrapper or just use the base rollout
            # For now, use a minimal worker that just initializes vLLM
            class RolloutWorker:
                """Minimal rollout worker for separate GPU pool."""

                def __init__(self, config, role: str):
                    assert role == "rollout"
                    self.config = config
                    # Will be initialized by PartialRolloutManager via HTTP servers

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            class RolloutWorker:
                """Minimal rollout worker for separate GPU pool."""

                def __init__(self, config, role: str):
                    assert role == "rollout"
                    self.config = config

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError(f"Strategy {config.actor_rollout_ref.actor.strategy} not supported")

        # Define resource pools
        from recipe.one_step_off_policy.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.Actor: ray.remote(actor_rollout_cls),
            Role.Rollout: ray.remote(RolloutWorker),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "actor_pool"

        # Validate GPU configuration
        assert config.trainer.n_gpus_per_node > 0, "trainer.n_gpus_per_node must be > 0"
        assert config.trainer.nnodes > 0, "trainer.nnodes must be > 0"
        assert config.rollout.n_gpus_per_node > 0, "rollout.n_gpus_per_node must be > 0"
        assert config.rollout.nnodes > 0, "rollout.nnodes must be > 0"

        # Create separate resource pools for training and rollout
        actor_pool = [config.trainer.n_gpus_per_node] * config.trainer.nnodes
        rollout_pool = [config.rollout.n_gpus_per_node] * config.rollout.nnodes

        resource_pool_spec = {
            "actor_pool": actor_pool,
            "rollout_pool": rollout_pool,
        }
        mapping = {
            Role.Actor: "actor_pool",
            Role.Rollout: "rollout_pool",
            Role.Critic: "actor_pool",
        }

        print(f"Resource pool spec: {resource_pool_spec}")
        print(f"  - Training pool (actor_pool): {sum(actor_pool)} GPUs")
        print(f"  - Rollout pool (rollout_pool): {sum(rollout_pool)} GPUs")

        # Add reward model if enabled
        if config.reward_model.enable:
            if config.reward_model.strategy == "fsdp2":
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError(f"RM strategy {config.reward_model.strategy} not supported")
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # Add reference policy if needed
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # Validate configuration
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(role_worker_mapping),
            use_critic=need_critic(config),
        )

        # Download checkpoint to local
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Initialize tokenizer and processor
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        # Create datasets
        train_dataset, val_dataset = create_rl_dataset(config, tokenizer, processor)

        # Create data sampler
        train_sampler = create_rl_sampler(
            dataset=train_dataset,
            global_batch_size=config.data.train_batch_size,
            use_critic=need_critic(config),
        )

        # Initialize reward managers
        reward_fn = load_reward_manager(config, tokenizer, processor, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        val_reward_fn = load_reward_manager(config, tokenizer, processor, num_examine=1, **config.reward_model.get("reward_kwargs", {}))

        # Create resource pool manager
        resource_pool_manager = ResourcePoolManager(resource_pool_spec, mapping)

        # Import collate function
        from verl.utils.dataset.rl_dataset import collate_fn

        # Create trainer with all parameters
        # Note: RayPPOTrainer expects everything in __init__, not via setter methods
        trainer = AsyncRLTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.get("device", "cuda"),
        )

        # Initialize workers
        trainer.init_workers()

        # Create data loader for rollout
        # FIXME: This should use verl's data loader utilities
        from torch.utils.data import DataLoader

        rollout_loader = DataLoader(
            train_dataset,
            batch_size=config.get("async_rl", {}).get("rollout_batch_size", 256),
            shuffle=True,
            collate_fn=collate_fn,
        )

        # Run async training
        num_steps = config.trainer.get("total_training_steps", 1000)
        ppo_epochs = config.algorithm.get("ppo_epochs", 1)
        trainer.train_async(
            data_loader=rollout_loader,
            num_steps=num_steps,
            ppo_epochs=ppo_epochs,
        )


if __name__ == "__main__":
    main()
