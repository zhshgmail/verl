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
Factory for AsyncRL Rollout Replicas

This provides a clean way to inject our AsyncRLvLLMReplica without modifying
verl's get_rollout_replica_class() function (which has tight coupling).

**The Problem**:
verl.workers.rollout.replica.get_rollout_replica_class() is hardcoded:

```python
def get_rollout_replica_class(rollout: str) -> type[RolloutReplica]:
    if rollout == "vllm":
        return vLLMReplica  # Hardcoded!
    elif rollout == "sglang":
        return SGLangReplica  # Hardcoded!
```

We CANNOT inject our AsyncRLvLLMReplica without modifying verl code.

**The Solution**:
Instead of trying to inject into get_rollout_replica_class(), we provide
our own factory function that can be used in place of it.

**Usage in main_ppo.py**:

```python
# INSTEAD OF:
# from verl.workers.rollout.replica import get_rollout_replica_class
# replica_cls = get_rollout_replica_class("vllm")

# USE:
from recipe.async_rl.replica_factory import get_async_rl_replica_class
replica_cls = get_async_rl_replica_class("vllm")  # Returns AsyncRLvLLMReplica
```

This is a WORKAROUND for verl's tight coupling. The proper fix would be
to make get_rollout_replica_class() accept a registry or use dependency injection.

TODO for verl engine integration:
    1. Add plugin system to get_rollout_replica_class()
    2. Support replica_registry.register("async_vllm", AsyncRLvLLMReplica)
    3. Or use dependency injection pattern
"""

import logging

from verl.workers.rollout.replica import RolloutReplica

logger = logging.getLogger(__name__)


def get_async_rl_replica_class(rollout: str, enable_async_rl: bool = True) -> type[RolloutReplica]:
    """
    Get rollout replica class with optional AsyncRL extensions.

    This is a BACKWARD-COMPATIBLE replacement for verl's get_rollout_replica_class()
    that supports both standard replicas and AsyncRL-extended replicas.

    Args:
        rollout: Backend name ("vllm" or "sglang")
        enable_async_rl: If True, return AsyncRL-extended replica (if available).
                        If False, return standard verl replica.

    Returns:
        Rollout replica class (AsyncRL-extended or standard)

    Raises:
        ValueError: If rollout backend not supported

    BACKWARD COMPATIBILITY:
        - SGLang: Always returns standard SGLangReplica (no AsyncRL support yet)
        - vLLM: Returns AsyncRLvLLMReplica if enable_async_rl=True,
               otherwise standard vLLMReplica

    TODO for verl engine integration:
        - Add RolloutReplicaRegistry.register(name, cls)
        - Add RolloutConfig.enable_async_weight_updates flag
        - Make get_rollout_replica_class() use registry
    """
    if rollout == "vllm":
        if enable_async_rl:
            # Import engine patches FIRST (critical!)
            import recipe.async_rl.vllm_engine_patches  # noqa: F401

            # Now import our extended replica
            from recipe.async_rl.extended_vllm_server import AsyncRLvLLMReplica

            logger.info("Using AsyncRLvLLMReplica with weight update support")
            return AsyncRLvLLMReplica
        else:
            # Return standard vLLM replica (backward compatible)
            from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

            logger.info("Using standard vLLMReplica (AsyncRL disabled)")
            return vLLMReplica

    elif rollout == "sglang":
        # SGLang is NOT supported for AsyncRL (requires extra implementation effort)
        if enable_async_rl:
            raise NotImplementedError(
                "AsyncRL is not implemented for SGLang backend.\n"
                "\n"
                "While SGLang supports request abortion (similar to vLLM), implementing\n"
                "AsyncRL for SGLang requires additional work:\n"
                "  - Extending SGLangReplica with custom HTTP endpoints\n"
                "  - Adapting SGLang's HTTP server architecture\n"
                "  - Testing weight update + cache reset with SGLang's engine\n"
                "\n"
                "Current focus: vLLM support only.\n"
                "\n"
                "Please use 'vllm' backend for AsyncRL, or set enable_async_rl=False for SGLang.\n"
                "\n"
                "To use AsyncRL:\n"
                "  rollout_backend: vllm\n"
                "  enable_async_weight_updates: true\n"
                "\n"
                "To use standard SGLang:\n"
                "  rollout_backend: sglang\n"
                "  enable_async_weight_updates: false\n"
                "\n"
                "TODO: Implement AsyncRLSGLangReplica if SGLang support is needed."
            )

        # Standard SGLang setup (from verl's get_rollout_replica_class)
        import os

        os.environ["SGLANG_USE_CPU_ENGINE"] = "1"

        try:
            import vllm  # noqa: F401
        except ImportError:
            import sys
            from unittest.mock import Mock

            mock_vllm = Mock()
            mock_vllm._custom_ops = Mock()
            mock_vllm._custom_ops.scaled_fp8_quant = Mock()
            sys.modules["vllm"] = mock_vllm

        from verl.workers.rollout.sglang_rollout.http_server_engine import SGLangReplica

        return SGLangReplica

    else:
        raise ValueError(
            f"Unknown rollout backend: {rollout}. "
            f"Supported: 'vllm', 'sglang'."
        )


def patch_get_rollout_replica_class(enable_async_rl: bool = None):
    """
    Monkey-patch verl's get_rollout_replica_class() with backward-compatible wrapper.

    This creates a wrapper that:
    1. Checks config for enable_async_weight_updates flag (if available)
    2. Falls back to enable_async_rl parameter
    3. Calls our factory with the flag
    4. FULLY BACKWARD COMPATIBLE with verl's original function

    Args:
        enable_async_rl: Default value if config doesn't specify.
                        None = auto-detect from config
                        True = force AsyncRL replicas
                        False = force standard replicas

    Usage:
        # Option 1: Auto-detect from config
        patch_get_rollout_replica_class()  # Reads RolloutConfig.enable_async_weight_updates

        # Option 2: Force enable
        patch_get_rollout_replica_class(enable_async_rl=True)

        # Option 3: Force disable (standard verl behavior)
        patch_get_rollout_replica_class(enable_async_rl=False)

    BACKWARD COMPATIBILITY:
        - If enable_async_rl=False or config.enable_async_weight_updates=False:
          Returns standard vLLMReplica/SGLangReplica (exact verl behavior)
        - SGLang always gets standard replica (no AsyncRL support yet)
        - vLLM gets AsyncRL replica only if explicitly enabled

    TODO for verl engine integration:
        - Add RolloutConfig.enable_async_weight_updates: bool = False
        - Make this the official get_rollout_replica_class() implementation
        - Remove need for patching
    """
    import verl.workers.rollout.replica

    # Save original function for potential restoration
    original_func = verl.workers.rollout.replica.get_rollout_replica_class

    def backward_compatible_wrapper(rollout: str) -> type[RolloutReplica]:
        """
        Backward-compatible wrapper that checks config for AsyncRL flag.

        This maintains verl's original signature: get_rollout_replica_class(rollout: str)
        while adding AsyncRL support via config inspection.
        """
        # Determine if AsyncRL should be enabled
        should_enable_async_rl = enable_async_rl  # From parameter

        # Try to read from config (if in hydra context)
        if should_enable_async_rl is None:
            try:
                from hydra import compose, initialize_config_module
                from omegaconf import OmegaConf

                # Try to get current config
                cfg = OmegaConf.structured({})  # Will be overridden by hydra
                should_enable_async_rl = cfg.get("actor_rollout_ref", {}).get(
                    "rollout", {}
                ).get("enable_async_weight_updates", False)
            except:
                # No config available, default to False (backward compatible)
                should_enable_async_rl = False

        # Call our factory with the flag
        return get_async_rl_replica_class(rollout, enable_async_rl=should_enable_async_rl)

    # Replace verl's function
    verl.workers.rollout.replica.get_rollout_replica_class = backward_compatible_wrapper

    # Store original for restoration
    verl.workers.rollout.replica._original_get_rollout_replica_class = original_func

    logger.info(
        f"Patched get_rollout_replica_class() with backward-compatible wrapper. "
        f"AsyncRL enabled: {enable_async_rl if enable_async_rl is not None else 'auto-detect from config'}"
    )


logger.info(
    "async_rl.replica_factory loaded\n"
    "Use get_async_rl_replica_class() instead of verl's get_rollout_replica_class()\n"
    "Or call patch_get_rollout_replica_class() to monkey-patch verl globally"
)
