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
vLLM Worker NCCL Hooks for Weight Updates

This module implements worker-side weight synchronization via NCCL/HCCL.
It provides methods to be called via vLLM's collective RPC mechanism.

IMPORTANT: This is a recipe-specific module that adds functionality to vLLM workers.
         For production use, these features should be upstreamed to verl engine.

TODO for verl engine integration:
    1. Move to verl/workers/rollout/vllm_rollout/weight_sync.py
    2. Add as plugin to vLLMRollout worker class
    3. Integrate with ResourcePoolManager for NCCL group management
    4. Support both sync and async weight update modes
"""

import logging
import os
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


# ============================================================================
# Worker-Side NCCL Weight Update Methods
#
# FIXME: These should be methods on vLLMRollout worker class
#        Currently injected via collective_rpc from vllm_server.py endpoints
#
# TODO for engine integration:
#     - Add these to verl/workers/rollout/vllm_rollout/vllm_rollout.py
#     - Make part of vLLMRollout's public API
#     - Support automatic parameter discovery (no manual spec needed)
#     - Add proper NCCL group lifecycle management
# ============================================================================


def init_nccl_weight_update_group(
    self,
    master_address: str,
    master_port: int,
    rank_offset: int,
    world_size: int,
    backend: str = "nccl",
    group_name: str = "async_rl_weight_update",
) -> Tuple[bool, str]:
    """
    Initialize NCCL process group for weight updates.

    This is called once before weight updates can be performed via NCCL.
    It creates a separate process group for weight synchronization.

    Args:
        master_address: NCCL master address (training worker address)
        master_port: NCCL master port
        rank_offset: Rank offset for this worker in the global group
        world_size: Total world size (training + rollout workers)
        backend: "nccl" or "hccl"
        group_name: Name for the process group

    Returns:
        (success: bool, message: str)

    FIXME: Currently creates process group manually
           Should use verl's NCCL group management instead

    TODO for engine integration:
        - Use ResourcePoolManager.create_nccl_group()
        - Auto-discover rank from worker's position in pool
        - Support multiple concurrent NCCL groups
        - Add health checks for group status
    """
    try:
        import torch.distributed as dist

        # Store group info for later use
        if not hasattr(self, "_nccl_weight_groups"):
            self._nccl_weight_groups = {}

        if group_name in self._nccl_weight_groups:
            return True, f"NCCL group {group_name} already initialized"

        # Get local rank from worker
        # FIXME: This assumes worker has local_rank attribute
        #        Should be part of worker's standard API
        local_rank = getattr(self, "local_rank", 0)
        global_rank = rank_offset + local_rank

        # Initialize process group
        os.environ["MASTER_ADDR"] = master_address
        os.environ["MASTER_PORT"] = str(master_port)

        # Create new group
        # FIXME: This uses global dist.init_process_group which may conflict
        #        Should use dist.new_group() for separate groups
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                rank=global_rank,
                world_size=world_size,
            )

        # Store group reference
        # For now we just store the config, actual group is the default one
        # TODO: Use dist.new_group() to create separate group
        self._nccl_weight_groups[group_name] = {
            "backend": backend,
            "rank": global_rank,
            "world_size": world_size,
            "master_address": master_address,
            "master_port": master_port,
        }

        logger.info(
            f"Initialized NCCL group {group_name}:\n"
            f"  - Backend: {backend}\n"
            f"  - Rank: {global_rank}\n"
            f"  - World size: {world_size}\n"
            f"  - Master: {master_address}:{master_port}"
        )

        return True, f"NCCL group {group_name} initialized successfully"

    except Exception as e:
        error_msg = f"Failed to initialize NCCL group: {e}"
        logger.error(error_msg)
        return False, error_msg


def set_nccl_weight_meta(
    self,
    names: List[str],
    dtypes: List[str],
    shapes: List[List[int]],
    group_name: str = "async_rl_weight_update",
) -> Tuple[bool, str]:
    """
    Set metadata for weights to be received via NCCL.

    This must be called before update_weights_nccl() to specify
    the structure of weights to receive.

    Args:
        names: List of parameter names
        dtypes: List of dtype strings (e.g., "torch.float32")
        shapes: List of tensor shapes
        group_name: NCCL group name

    Returns:
        (success: bool, message: str)

    FIXME: Requires manual specification of parameter metadata
           Should auto-discover from model instead

    TODO for engine integration:
        - Auto-extract metadata from model.named_parameters()
        - Support LoRA and other parameter-efficient methods
        - Add validation that metadata matches actual model
        - Support sharded parameters (for TP/PP)
    """
    try:
        if not hasattr(self, "_nccl_weight_groups"):
            return False, "NCCL group not initialized. Call init_nccl_weight_update_group first."

        if group_name not in self._nccl_weight_groups:
            return False, f"NCCL group {group_name} not found"

        # Validate inputs
        if not (len(names) == len(dtypes) == len(shapes)):
            return False, f"Mismatched lengths: names={len(names)}, dtypes={len(dtypes)}, shapes={len(shapes)}"

        # Store metadata
        self._nccl_weight_groups[group_name]["param_metadata"] = {
            "names": names,
            "dtypes": dtypes,
            "shapes": shapes,
        }

        logger.info(f"Set weight metadata for group {group_name}: {len(names)} parameters")

        return True, f"Weight metadata set for {len(names)} parameters"

    except Exception as e:
        error_msg = f"Failed to set weight metadata: {e}"
        logger.error(error_msg)
        return False, error_msg


def update_weights_nccl(
    self,
    group_name: str = "async_rl_weight_update",
) -> Tuple[bool, str]:
    """
    Update weights by receiving them via NCCL from training workers.

    This method:
    1. Receives new weights via NCCL broadcast/recv
    2. Updates model parameters
    3. Does NOT abort requests (that's done by EngineCore.abort_all_requests)

    Args:
        group_name: NCCL group name

    Returns:
        (success: bool, message: str)

    FIXME: Manual NCCL recv for each parameter
           Should use automatic parameter synchronization

    TODO for engine integration:
        - Integrate with verl's weight sync utilities
        - Support sharded parameters (TP/PP)
        - Add checksum validation
        - Support incremental updates (only changed params)
        - Add metrics tracking (update latency, bandwidth)
    """
    try:
        import torch.distributed as dist

        if not hasattr(self, "_nccl_weight_groups"):
            return False, "NCCL group not initialized"

        if group_name not in self._nccl_weight_groups:
            return False, f"NCCL group {group_name} not found"

        group_info = self._nccl_weight_groups[group_name]

        if "param_metadata" not in group_info:
            return False, "Weight metadata not set. Call set_nccl_weight_meta first."

        metadata = group_info["param_metadata"]
        names = metadata["names"]
        dtypes = metadata["dtypes"]
        shapes = metadata["shapes"]

        # Get model
        # FIXME: This assumes worker has model attribute
        #        Should be part of worker's standard API
        model = getattr(self, "model", None)
        if model is None:
            return False, "Worker has no model attribute"

        # Receive weights via NCCL
        # FIXME: This assumes sender uses broadcast from rank 0
        #        Should be configurable
        received_weights = {}
        for name, dtype_str, shape in zip(names, dtypes, shapes):
            # Parse dtype
            dtype = getattr(torch, dtype_str.replace("torch.", ""))

            # Create tensor buffer
            tensor = torch.empty(shape, dtype=dtype, device="cuda")

            # Receive from training workers
            # FIXME: Hardcoded src=0, should get from config
            # TODO: Support multiple training workers (average or latest?)
            dist.recv(tensor, src=0)

            received_weights[name] = tensor

        # Update model parameters
        # FIXME: Direct parameter assignment may not work for all models
        #        Should use model.load_state_dict() for safety
        state_dict = model.state_dict()
        for name, weight in received_weights.items():
            if name in state_dict:
                state_dict[name].copy_(weight)
            else:
                logger.warning(f"Parameter {name} not found in model, skipping")

        logger.info(f"Updated {len(received_weights)} parameters via NCCL")

        return True, f"Successfully updated {len(received_weights)} parameters"

    except Exception as e:
        error_msg = f"Failed to update weights via NCCL: {e}"
        logger.error(error_msg)
        return False, error_msg


def update_weights(
    self,
    model_path: str,
    load_format: str = "auto",
) -> Tuple[bool, str]:
    """
    Update weights by loading from disk checkpoint.

    This is the simpler alternative to NCCL-based updates.

    Args:
        model_path: Path to model checkpoint
        load_format: Format to load ("auto", "safetensors", "pt", etc.)

    Returns:
        (success: bool, message: str)

    FIXME: Direct model.load_weights() call
           Should integrate with verl's checkpoint loading

    TODO for engine integration:
        - Use verl's checkpoint manager
        - Support async loading (non-blocking)
        - Add validation before loading
        - Support partial checkpoint loading
    """
    try:
        # Get model
        # FIXME: This assumes worker has model attribute
        model = getattr(self, "model", None)
        if model is None:
            return False, "Worker has no model attribute"

        # Check if model has load_weights method (vLLM models do)
        if hasattr(model, "load_weights"):
            model.load_weights(model_path, load_format=load_format)
        else:
            # Fallback to torch load
            import torch
            state_dict = torch.load(model_path, map_location="cuda")
            model.load_state_dict(state_dict)

        logger.info(f"Updated weights from disk: {model_path}")

        return True, f"Successfully loaded weights from {model_path}"

    except Exception as e:
        error_msg = f"Failed to load weights: {e}"
        logger.error(error_msg)
        return False, error_msg


logger.info(
    "async_rl.vllm_worker_hooks module loaded\n"
    "NOTE: This module provides worker-side NCCL weight sync methods.\n"
    "      These are called by AsyncRLvLLMAsyncRollout._execute_method().\n"
    "      See module docstring for engine integration TODO items."
)
