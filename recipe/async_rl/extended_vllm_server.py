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
Extended vLLM HTTP Server for Async RL

This module provides CLEAN EXTENSION of verl's vLLMHttpServer via subclassing.
NO monkey-patching of verl classes - only proper OOP.

**What we extend**:
- vLLMHttpServer → AsyncRLvLLMHttpServer (add custom endpoints)
- vLLMReplica → AsyncRLvLLMReplica (factory to create our server)

**What we still patch** (unavoidable):
- vLLM's EngineCore (no extension points)
- Import vllm_engine_patches for this

Architecture:
```
vLLMReplica.launch_servers()
  └─> Creates vLLMHttpServer instances
      └─> Calls build_app() to create FastAPI app

AsyncRLvLLMReplica.launch_servers()  # Our extension
  └─> Creates AsyncRLvLLMHttpServer instances  # Our subclass
      └─> Calls build_app() + adds custom endpoints  # Extended
```

Usage:
```python
# Import engine patches FIRST (unavoidable monkey-patch)
import recipe.async_rl.vllm_engine_patches

# Use our extended classes (clean OOP)
from recipe.async_rl.extended_vllm_server import AsyncRLvLLMReplica

replica = AsyncRLvLLMReplica(config, model_config, RolloutMode.STANDALONE)
await replica.launch_servers()  # Creates AsyncRLvLLMHttpServer with custom endpoints
```
"""

import argparse
import logging
from typing import Any, List, Optional

import ray
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from vllm.entrypoints.openai.protocol import OpenAIBaseModel

from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    vLLMHttpServer,
    vLLMReplica,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================================
# Request Models for Custom Endpoints
# ============================================================================


class UpdateWeightsRequest(OpenAIBaseModel):
    """Request to update model weights from disk."""

    model_path: str
    load_format: str = "auto"


class UpdateWeightsFromNCCLRequest(OpenAIBaseModel):
    """Request to update model weights via NCCL."""

    pass


class InitNCCLGroupRequest(OpenAIBaseModel):
    """Request to initialize NCCL group for weight updates."""

    master_address: str
    master_port: int
    rank_offset: int
    world_size: int
    backend: str = "nccl"
    group_name: str = "async_rl_weight_update"


class SetWeightMetaRequest(OpenAIBaseModel):
    """Request to set metadata for NCCL weight sync."""

    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]
    group_name: str = "async_rl_weight_update"


# ============================================================================
# Helper Functions
# ============================================================================


def build_response(ret_list):
    """Build JSON response from collective RPC results."""
    success = True
    message = ""
    for rank, ret_value in enumerate(ret_list):
        if_success, msg = ret_value
        success = success and if_success
        status = "success" if if_success else f"failed: {msg}"
        message += f"TP rank {rank}: {status}\n"

    content = {"success": success, "message": message}
    status_code = 200 if success else 400
    return JSONResponse(content, status_code=status_code)


# ============================================================================
# Extended vLLMHttpServer - CLEAN SUBCLASS
# ============================================================================


class AsyncRLvLLMHttpServer(vLLMHttpServer):
    """
    Extended vLLMHttpServer with custom weight update endpoints.

    This is a CLEAN extension via subclassing - no monkey-patching of verl code.

    What we override:
    - run_server(): Add custom endpoints after build_app()

    What we keep from parent:
    - All initialization logic
    - Server launch logic
    - Everything else

    IMPORTANT: This assumes vllm_engine_patches is imported to patch EngineCore.
             We only extend verl classes here, not vLLM classes.
    """

    async def run_server(self, args: argparse.Namespace):
        """
        Override to add custom endpoints after building the FastAPI app.

        This is the CLEAN way to extend - we call parent's logic,
        then add our custom endpoints to the app instance.
        """
        # Import here to avoid circular dependency
        from vllm import AsyncEngineArgs
        from vllm.entrypoints.openai.api_server import build_app, init_app_state
        from vllm.usage.usage_lib import UsageContext
        from vllm.v1.engine.async_llm import AsyncLLM

        from verl.workers.rollout.utils import run_unvicorn

        # Same as parent implementation
        engine_args = AsyncEngineArgs.from_cli_args(args)
        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = engine_args.create_engine_config(usage_context=usage_context)
        vllm_config.parallel_config.data_parallel_master_port = self._dp_master_port

        engine_client = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            disable_log_requests=engine_args.disable_log_requests,
            disable_log_stats=engine_args.disable_log_stats,
        )

        await engine_client.reset_mm_cache()

        # Build the FastAPI app
        app = build_app(args)
        await init_app_state(engine_client, vllm_config, app.state, args)

        # HERE'S THE EXTENSION: Add our custom endpoints
        self._add_custom_endpoints(app)

        if self.replica_rank == 0 and self.node_rank == 0:
            logger.info(f"Initializing V1 LLM engine with AsyncRL extensions")

        self.engine = engine_client
        self._server_port, self._server_task = await run_unvicorn(app, args, self._server_address)

    def _add_custom_endpoints(self, app: FastAPI):
        """
        Add custom weight update endpoints to the FastAPI app.

        This is called after build_app() creates the app, so we add
        endpoints to this specific app instance.

        IMPORTANT: This assumes vllm_engine_patches has patched EngineCore
                 to add abort_all_requests() and async_rl_update_weights_*()
        """

        @app.post("/async_rl/update_weights")
        async def update_weights_from_disk(request: UpdateWeightsRequest, raw_request: Request):
            """Update weights from disk checkpoint."""
            logger.info(f"Updating weights from disk: {request.model_path}")

            llm = raw_request.app.state.engine_client

            # Call the patched method on EngineCore
            ret_list = await llm.engine_core.call_utility_async(
                "async_rl_update_weights_disk",
                request.model_path,
            )

            return build_response(ret_list)

        @app.post("/async_rl/update_weights_nccl")
        async def update_weights_from_nccl(request: UpdateWeightsFromNCCLRequest, raw_request: Request):
            """Update weights via NCCL."""
            logger.info("Updating weights via NCCL")

            llm = raw_request.app.state.engine_client

            ret_list = await llm.engine_core.call_utility_async(
                "async_rl_update_weights_nccl",
            )

            return build_response(ret_list)

        @app.post("/async_rl/init_nccl_group")
        async def init_nccl_group(request: InitNCCLGroupRequest, raw_request: Request):
            """Initialize NCCL group for weight updates."""
            logger.info(f"Initializing NCCL group: {request.group_name}")

            llm = raw_request.app.state.engine_client

            ret_list = await llm.collective_rpc(
                "init_nccl_weight_update_group",
                args=(
                    request.master_address,
                    request.master_port,
                    request.rank_offset,
                    request.world_size,
                    request.backend,
                    request.group_name,
                ),
            )

            return build_response(ret_list)

        @app.post("/async_rl/set_weight_meta")
        async def set_weight_meta(request: SetWeightMetaRequest, raw_request: Request):
            """Set metadata for NCCL weight sync."""
            logger.info(f"Setting weight metadata for group: {request.group_name}")

            llm = raw_request.app.state.engine_client

            ret_list = await llm.collective_rpc(
                "set_nccl_weight_meta",
                args=(
                    request.names,
                    request.dtypes,
                    request.shapes,
                    request.group_name,
                ),
            )

            return build_response(ret_list)

        logger.info(
            "Added custom AsyncRL endpoints:\n"
            "  - POST /async_rl/update_weights\n"
            "  - POST /async_rl/update_weights_nccl\n"
            "  - POST /async_rl/init_nccl_group\n"
            "  - POST /async_rl/set_weight_meta"
        )


# ============================================================================
# Extended vLLMReplica - Clean Injection Pattern
# ============================================================================


class AsyncRLvLLMReplica(vLLMReplica):
    """
    Extended vLLMReplica that uses AsyncRLvLLMAsyncRollout workers.

    This is a CLEAN injection pattern - we override the worker class
    and server class to inject our custom implementations.

    What we override:
    - get_ray_class_with_init_args(): Inject AsyncRLvLLMAsyncRollout worker class
    - launch_servers(): Create AsyncRLvLLMHttpServer instead of vLLMHttpServer

    What we keep from parent:
    - All worker management
    - All replica logic
    - Everything else

    CLEAN DESIGN: No monkey-patching, no remote hook installation!
                  Just clean subclassing and dependency injection.
    """

    def get_ray_class_with_init_args(self):
        """
        Override to inject AsyncRLvLLMAsyncRollout worker class.

        This is the CLEAN way to extend workers - just return our
        extended worker class instead of the base vLLMAsyncRollout.
        """
        import ray
        from verl.single_controller.ray import RayClassWithInitArgs
        from recipe.async_rl.async_rl_vllm_rollout import AsyncRLvLLMAsyncRollout

        # Use our extended worker class
        _async_rl_worker_actor_cls = ray.remote(AsyncRLvLLMAsyncRollout)

        worker_dict_cls = RayClassWithInitArgs(
            cls=_async_rl_worker_actor_cls,
            config=self.config,
            model_config=self.model_config,
            device_mesh=None,
        )
        return worker_dict_cls

    async def launch_servers(self):
        """
        Override to create AsyncRLvLLMHttpServer instances.

        This uses our extended server class instead of the base vLLMHttpServer.

        NOTE: Workers are already AsyncRLvLLMAsyncRollout instances (injected
              via get_ray_class_with_init_args), so they already support our
              custom RPC methods. No hook installation needed!
        """
        import asyncio

        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        # Get node_id of all workers
        worker_node_ids = await asyncio.gather(
            *[
                worker.__ray_call__.remote(lambda self: ray.get_runtime_context().get_node_id())
                for worker in self.workers
            ]
        )

        # For non-data parallel case
        nnodes, gpus_per_node = self.nnodes, self.gpus_per_node
        if self.config.data_parallel_size == 1:
            nnodes = 1
            gpus_per_node = self.world_size

        # Create AsyncRLvLLMHttpServer instead of vLLMHttpServer
        for node_rank in range(nnodes):
            workers = self.workers[node_rank * gpus_per_node : (node_rank + 1) * gpus_per_node]
            node_id = worker_node_ids[node_rank * gpus_per_node]

            # Use AsyncRLvLLMHttpServer
            server = AsyncRLvLLMHttpServer.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                name=f"async_rl_vllm_server_{self.replica_rank}_{node_rank}",
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                gpus_per_node=gpus_per_node,
                nnodes=nnodes,
            )
            self.servers.append(server)

        # Launch servers
        master_address, master_port = await self.servers[0].get_master_address.remote()
        await asyncio.gather(
            *[
                server.launch_server.remote(master_address=master_address, master_port=master_port)
                for server in self.servers
            ]
        )

        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = f"{server_address}:{server_port}"

        logger.info(f"AsyncRLvLLMReplica launched with custom endpoints at {self._server_address}")


logger.info(
    "async_rl.extended_vllm_server module loaded\n"
    "NOTE: This provides CLEAN extension of verl's vLLMHttpServer.\n"
    "      Use AsyncRLvLLMReplica instead of vLLMReplica.\n"
    "      Still requires importing vllm_engine_patches for EngineCore patching."
)
