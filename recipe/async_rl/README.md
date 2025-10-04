# AsyncRL Recipe - Asynchronous RL with Weight Updates

Asynchronous RL training with decoupled PPO objective, inspired by [AReaL](https://arxiv.org/abs/2505.24298).

**⚠️ IMPORTANT: vLLM backend only. SGLang is not supported (yet).**

## Features

- ✅ **Weight updates during generation** - Update model weights while generation is in progress
- ✅ **Request abortion + cache reset** - Abort in-flight requests before weight updates
- ✅ **Version tracking** - Track policy version (start/end) for staleness-aware training
- ✅ **Decoupled PPO** - Three logprobs (π_θ, π_prox, π_old) with importance weighting
- ✅ **NCCL or disk-based weight sync** - Flexible weight synchronization
- ✅ **Clean OOP design** - Minimal patching, clean subclassing

## Quick Start

### Option 1: Direct Factory (Recommended)

```python
from recipe.async_rl.replica_factory import get_async_rl_replica_class

# Get AsyncRL replica class
replica_cls = get_async_rl_replica_class("vllm", enable_async_rl=True)

# Create and initialize replica
replica = replica_cls(
    replica_rank=0,
    config=rollout_config,
    model_config=model_config,
    gpus_per_node=8,
)
await replica.init_standalone()

# Server address for HTTP requests
server_addr = replica.server_address  # e.g., "192.168.1.1:8000"
```

### Option 2: Monkey-Patch (For Integration)

```python
# At the top of main_ppo.py
from recipe.async_rl.replica_factory import patch_get_rollout_replica_class

# Patch verl's factory function
patch_get_rollout_replica_class(enable_async_rl=True)

# Now use verl's standard API
from verl.workers.rollout.replica import get_rollout_replica_class

replica_cls = get_rollout_replica_class("vllm")  # Returns AsyncRLvLLMReplica
```

### Configuration-Driven Approach

```yaml
# config/async_rl_trainer.yaml
actor_rollout_ref:
  rollout:
    enable_async_weight_updates: true  # Enable AsyncRL
    # ... other rollout config
```

```python
# Auto-detect from config
from recipe.async_rl.replica_factory import patch_get_rollout_replica_class
patch_get_rollout_replica_class()  # Reads enable_async_weight_updates from config
```

## Custom HTTP Endpoints

AsyncRL adds custom endpoints to the vLLM HTTP server:

```python
import requests

# 1. Update weights from disk
requests.post(
    f"http://{server_addr}/async_rl/update_weights",
    json={"model_path": "/path/to/checkpoint"}
)

# 2. Initialize NCCL group for weight sync
requests.post(
    f"http://{server_addr}/async_rl/init_nccl_group",
    json={
        "master_address": "192.168.1.100",
        "master_port": 29500,
        "rank_offset": 8,
        "world_size": 16,
        "backend": "nccl",
        "group_name": "async_rl_weight_update"
    }
)

# 3. Set weight metadata for NCCL
requests.post(
    f"http://{server_addr}/async_rl/set_weight_meta",
    json={
        "names": ["model.layers.0.weight", ...],
        "dtypes": ["torch.float32", ...],
        "shapes": [[4096, 4096], ...],
        "group_name": "async_rl_weight_update"
    }
)

# 4. Update weights via NCCL
requests.post(
    f"http://{server_addr}/async_rl/update_weights_nccl",
    json={"group_name": "async_rl_weight_update"}
)
```

## Architecture Overview

```
AsyncRLvLLMReplica (extends vLLMReplica)
├── Injects AsyncRLvLLMAsyncRollout worker class
│   └── Overrides _execute_method() for custom RPC methods
└── Creates AsyncRLvLLMHttpServer instances
    └── Adds custom /async_rl/* endpoints

vLLM EngineCore (patched)
└── abort_all_requests() + reset_prefix_cache()
```

**Clean design principles**:
- Only vLLM's EngineCore is patched (unavoidable - no extension points)
- All verl classes extended via clean subclassing
- Worker methods injected via dependency injection (no monkey-patching)

## Files

```
recipe/async_rl/
├── vllm_engine_patches.py       # Patches vLLM EngineCore
├── async_rl_vllm_rollout.py     # Extends vLLMAsyncRollout
├── extended_vllm_server.py      # Extends vLLMHttpServer + vLLMReplica
├── vllm_worker_hooks.py         # Worker RPC methods
├── replica_factory.py           # Factory for injection
├── weight_update.py             # HTTP coordination
├── partial_rollout_manager.py   # Version tracking
├── ray_trainer.py               # AsyncRLTrainer
├── ppo_loss.py                  # Decoupled PPO loss
└── main_ppo.py                  # Entry point
```

## Backward Compatibility

| Backend | enable_async_rl | Result |
|---------|----------------|--------|
| vLLM | True | AsyncRLvLLMReplica ✅ |
| vLLM | False | vLLMReplica (standard) ✅ |
| SGLang | True | NotImplementedError ❌ |
| SGLang | False | SGLangReplica (standard) ✅ |

**Why vLLM only?**
- SGLang does support request abortion (similar to vLLM)
- However, extending for SGLang requires extra work:
  - Adapting different HTTP server architecture
  - Extending SGLangReplica with custom endpoints
  - Testing weight update + cache reset flow
- Current focus: vLLM only
- Future: SGLang support can be added if needed

## Key Differences from Standard PPO

### 1. Decoupled PPO Objective

**Standard PPO**:
```python
ratio = exp(logprobs - old_logprobs)  # π_θ / π_old
clipped_ratio = clip(ratio, 1-ε, 1+ε)
loss = min(ratio * advantages, clipped_ratio * advantages)
```

**AsyncRL Decoupled PPO**:
```python
# Three logprobs instead of two
ratio = exp(logprobs - proximal_logprobs)       # π_θ / π_prox (for clipping)
behav_weight = exp(proximal_logprobs - old_logprobs)  # π_prox / π_old (staleness)

# Apply staleness weighting
clipped_ratio = clip(ratio, 1-ε, 1+ε)
loss = min(ratio * advantages, clipped_ratio * advantages) * behav_weight
```

### 2. Version Tracking

```python
# Track which policy version generated each sample
output.non_tensor_batch["version_start"] = version_at_generation_start
output.non_tensor_batch["version_end"] = version_at_generation_end

# Use for staleness-aware training
if version_end - version_start > staleness_threshold:
    # Sample crossed weight update boundary - handle specially
    pass
```

### 3. No Chunked Generation

AReaL's code uses chunked generation but calls it a "hack":
> "This is a hack usage. We don't need it if the server can pause requests, update weights, and recompute kv caches"

With `abort_all_requests()` + `reset_prefix_cache()`, we accept wasted work for simpler code.

## TODOs for Upstream Integration

### To vLLM
- Contribute `abort_all_requests()` as official API
- Add `reset_prefix_cache()` as public method

### To verl
- Add plugin/registry system for rollout replicas
- Move AsyncRLvLLMAsyncRollout → `verl/workers/rollout/vllm_rollout/async_extensions.py`
- Move AsyncRLvLLMHttpServer → `verl/workers/rollout/vllm_rollout/async_server.py`
- Add `RolloutConfig.enable_async_weight_updates` flag
- Support configuration-driven replica selection

## See Also

- **[DESIGN.md](./DESIGN.md)** - Architecture, design decisions, implementation details
