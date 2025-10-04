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
AsyncRL-extended vLLMAsyncRollout Worker (vLLM Only)

This extends vLLMAsyncRollout to support custom RPC methods for weight updates.

IMPORTANT: This is vLLM-specific. SGLang is NOT supported (yet) because:
    - SGLang does have request abortion support (similar to vLLM)
    - But SGLang's HTTP server architecture is different, requires adaptation
    - Would need SGLang-specific replica extension (AsyncRLSGLangAsyncRollout)
    - Current focus: vLLM only. SGLang can be added later if needed.

CLEAN DESIGN: No monkey-patching! Just clean subclassing and method override.

Usage:
    from recipe.async_rl.async_rl_vllm_rollout import AsyncRLvLLMAsyncRollout

    # Use in AsyncRLvLLMReplica
    _async_rl_worker_actor_cls = ray.remote(AsyncRLvLLMAsyncRollout)

TODO for verl engine integration:
    - Add to verl/workers/rollout/vllm_rollout/async_rl_extensions.py
    - Make configurable via RolloutConfig.enable_async_weight_updates
    - Register as plugin in worker registry
"""

import logging
import os

from verl.workers.rollout.vllm_rollout import vLLMAsyncRollout

# Import our worker-side methods
from recipe.async_rl.vllm_worker_hooks import (
    init_nccl_weight_update_group,
    set_nccl_weight_meta,
    update_weights_nccl,
    update_weights,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class AsyncRLvLLMAsyncRollout(vLLMAsyncRollout):
    """
    Extended vLLMAsyncRollout with AsyncRL weight update methods.

    This is a CLEAN extension - we simply override _execute_method to
    add our custom RPC methods before delegating to the parent.

    What we add:
    - init_nccl_weight_update_group: Initialize NCCL group for weight sync
    - set_nccl_weight_meta: Set weight metadata for NCCL
    - update_weights_nccl: Update weights via NCCL broadcast
    - async_rl_update_weights: Update weights from disk checkpoint

    What we keep from parent:
    - All vLLM worker functionality
    - ZeroMQ RPC handling
    - Standard execute_method delegation
    """

    async def _execute_method(self, method: str | bytes, *args, **kwargs):
        """
        Override to handle AsyncRL custom methods.

        This extends the parent's _execute_method to support our
        custom weight update methods before delegating to vLLM.

        CLEAN DESIGN: No monkey-patching, just clean method override!
        """
        # Get the actual vLLM worker (where the model lives)
        worker = self.inference_engine.worker if self.inference_engine else None

        # Handle AsyncRL custom methods
        if method == "init_nccl_weight_update_group":
            if worker is None:
                return False, "Worker not initialized"
            return init_nccl_weight_update_group(worker, *args, **kwargs)

        elif method == "set_nccl_weight_meta":
            if worker is None:
                return False, "Worker not initialized"
            return set_nccl_weight_meta(worker, *args, **kwargs)

        elif method == "update_weights_nccl":
            if worker is None:
                return False, "Worker not initialized"
            return update_weights_nccl(worker, *args, **kwargs)

        elif method == "async_rl_update_weights":
            if worker is None:
                return False, "Worker not initialized"
            return update_weights(worker, *args, **kwargs)

        else:
            # Delegate to parent for standard vLLM methods
            return await super()._execute_method(method, *args, **kwargs)


logger.info(
    "async_rl.async_rl_vllm_rollout module loaded\n"
    "NOTE: This provides CLEAN extension of vLLMAsyncRollout.\n"
    "      Use AsyncRLvLLMAsyncRollout as worker class in AsyncRLvLLMReplica."
)
