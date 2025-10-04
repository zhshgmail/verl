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
vLLM EngineCore Patches for Weight Updates

This module ONLY patches vLLM's EngineCore class (unavoidable monkey-patch).

**What this patches**:
- vLLM's EngineCore.abort_all_requests()
- vLLM's EngineCore.async_rl_update_weights_*()

**What this does NOT patch**:
- FastAPI endpoints (that's in extended_vllm_server.py via clean subclassing)
- verl's vLLMHttpServer (that's in extended_vllm_server.py via subclassing)
- verl's vLLMReplica (that's in extended_vllm_server.py via subclassing)

**Why monkey-patching EngineCore**:
vLLM's EngineCore is tightly coupled with no extension points.
We have NO CHOICE but to monkey-patch it.

**Usage**:
    # Import this FIRST, before creating any vLLM engines
    import recipe.async_rl.vllm_engine_patches

    # Then use extended_vllm_server for clean verl extensions
    from recipe.async_rl.extended_vllm_server import AsyncRLvLLMReplica

**Scope of patching**:
- MINIMAL: Only EngineCore methods
- NO FastAPI router patching (that's handled cleanly in extended_vllm_server)
- NO verl class patching (use subclassing instead)

TODO for upstreaming to vLLM:
    1. Add EngineCore.abort_all_requests() as official API
    2. Add weight update hooks to vLLM's extension system
"""

import logging

# Import vLLM components
try:
    from vllm.logger import init_logger
    from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.request import RequestStatus
except ImportError as e:
    raise ImportError(
        f"Failed to import vLLM internals: {e}\n"
        "This recipe requires vLLM v1 API. Please ensure vLLM is installed."
    )

logger = init_logger("async_rl_vllm_engine_patches")
logger.setLevel(logging.INFO)


# ============================================================================
# EngineCore Patches (Unavoidable Monkey-Patching)
# ============================================================================


def abort_all_requests(self):
    """
    Abort all running and waiting requests and reset KV cache.

    This is called before weight updates to prevent KV cache pollution.

    IMPORTANT: Resets prefix cache after aborting to ensure cache coherency
             with new weights.
    """
    scheduler = self.scheduler
    abort_lists = list(scheduler.running) + list(scheduler.waiting)

    if not abort_lists:
        # No requests, but still reset cache
        success = scheduler.reset_prefix_cache()
        if not success:
            logger.warning("Failed to reset prefix cache (no running requests)")
        return

    # Create abort outputs
    client_outputs = {}
    for req in abort_lists:
        engine_output = EngineCoreOutput(
            request_id=req.request_id,
            new_token_ids=[],
            finish_reason=FinishReason.ABORT,
            new_logprobs=None,
            new_prompt_logprobs_tensors=None,
            stop_reason=None,
        )
        if req.client_index not in client_outputs:
            client_outputs[req.client_index] = []
        client_outputs[req.client_index].append(engine_output)

    # Mark as finished/aborted
    request_ids = [req.request_id for req in abort_lists]
    scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    # Send abort outputs
    for client_index, outputs in client_outputs.items():
        engine_core_outputs = EngineCoreOutputs(outputs=outputs)
        self.output_queue.put_nowait((client_index, engine_core_outputs))

    # CRITICAL: Reset prefix cache
    success = scheduler.reset_prefix_cache()
    if not success:
        raise RuntimeError(
            "Failed to reset prefix cache! KV cache will be polluted with old weights."
        )

    logger.info(f"Aborted {len(abort_lists)} requests and reset prefix cache")


def async_rl_update_weights_disk(self, model_path: str):
    """
    Update weights from disk.

    Calls:
    1. abort_all_requests()
    2. collective_rpc("update_weights")
    """
    self.abort_all_requests()
    return self.collective_rpc("update_weights", args=(model_path,))


def async_rl_update_weights_nccl(self):
    """
    Update weights via NCCL.

    Calls:
    1. abort_all_requests()
    2. collective_rpc("update_weights_nccl")
    """
    self.abort_all_requests()
    return self.collective_rpc("update_weights_nccl")


# ============================================================================
# Installation
# ============================================================================


def install_engine_patches():
    """
    Install patches on vLLM's EngineCore.

    This is the ONLY monkey-patching we do. Everything else uses clean OOP.

    Idempotent - safe to call multiple times.
    """
    try:
        if hasattr(EngineCore, "abort_all_requests"):
            logger.info("vLLM EngineCore patches already installed")
            return

        # Install methods
        setattr(EngineCore, "abort_all_requests", abort_all_requests)
        setattr(EngineCore, "async_rl_update_weights_disk", async_rl_update_weights_disk)
        setattr(EngineCore, "async_rl_update_weights_nccl", async_rl_update_weights_nccl)

        logger.info("Successfully patched vLLM EngineCore with weight update methods")

    except Exception as e:
        raise RuntimeError(
            f"Failed to patch vLLM EngineCore: {e}\n"
            "vLLM's internal API may have changed."
        )


# Auto-install on import
install_engine_patches()

logger.info(
    "async_rl.vllm_engine_patches loaded\n"
    "Patched: vLLM's EngineCore only\n"
    "For verl extensions, see: extended_vllm_server.py"
)
