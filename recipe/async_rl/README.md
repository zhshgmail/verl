# AsyncRL Recipe - Asynchronous RL with Weight Updates

Asynchronous RL training with decoupled PPO objective, inspired by [AReaL](https://arxiv.org/abs/2505.24298).

**⚠️ IMPORTANT: vLLM backend only. SGLang is not supported (yet).**

## Features

- ✅ **Production-grade data management** - [TransferQueue](https://github.com/TransferQueue/TransferQueue) for distributed asynchronous streaming (based on [AsyncFlow paper](https://arxiv.org/abs/2507.01663))
- ✅ **Concurrent rollout + training** - True async RL with separate GPU pools
- ✅ **Per-token metadata** - Store behavior logprobs and policy versions per token (following AReaL)
- ✅ **Staleness validation** - Reject samples with weight updates mid-generation
- ✅ **Staleness control** - Filter samples by max_staleness (policy version)
- ✅ **Weight updates during generation** - Update model weights while generation is in progress
- ✅ **Request abortion + cache reset** - Abort in-flight requests before weight updates
- ✅ **Version tracking** - Track policy version (start/end) for staleness-aware training
- ✅ **Decoupled PPO** - Three logprobs (π_θ, π_prox, π_old) with importance weighting
- ✅ **NCCL or disk-based weight sync** - Flexible weight synchronization
- ✅ **Clean OOP design** - Minimal patching, clean subclassing
- ✅ **Multi-node ready** - Distributed storage units across nodes with ZMQ communication
- ✅ **Extensible schema** - Add new sample fields via TransferQueue's field metadata
- ✅ **Type-safe** - Proper type hints for IDE support and runtime safety
- ✅ **Explicit resource management** - Dedicated thread pools (never use default executor)

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

## TransferQueue Configuration

### Production-Grade Distributed Data Management

AsyncRL now uses **[TransferQueue](https://github.com/TransferQueue/TransferQueue)**, a high-performance distributed data management system based on the [AsyncFlow paper](https://arxiv.org/abs/2507.01663). TransferQueue replaces the simple TransferDock with:

- **Distributed storage units** across nodes (horizontal scaling)
- **ZMQ-based communication** (high-performance, low-latency)
- **Per-field metadata tracking** (fine-grained production/consumption status)
- **Multi-consumer support** (independent consumption tracking per task)

Configuration in `config/async_rl_trainer.yaml`:

```yaml
async_rl:
  max_staleness: 5  # Maximum staleness (policy version difference)
  train_batch_size: 128  # Training batch size

  transfer_queue:
    num_storage_units: 2  # Distribute data across 2 storage units
    num_controllers: 1  # Single controller (supports load balancing)
    num_global_batch: 2  # Buffer 2 batches (total capacity: 256 samples)
    storage_cpus_per_unit: 1  # CPU cores per storage unit
    controller_cpus: 1  # CPU cores for controller
```

**Key benefits over simple TransferDock**:
- ✅ True multi-node distributed storage
- ✅ ZMQ instead of Ray ObjectRef (avoids bottleneck)
- ✅ Panoramic metadata tracking (per-sample per-field)
- ✅ Built-in multi-consumer support
- ✅ Future RDMA support via MoonCakeStore backend

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
┌──────────────────────────────────────────────────────────┐
│              AsyncRLTrainer                              │
│  - Concurrent rollout & training loops                   │
│  - Version tracking (policy_version)                     │
└──────┬───────────────┬────────────────────┬──────────────┘
       │               │                    │
       ▼               ▼                    ▼
┌──────────┐    ┌───────────────────┐  ┌──────────────┐
│ Training │    │ TransferQueue     │  │ Rollout Pool │
│ Pool     │    │ (Distributed)     │  │ (vLLM HTTP)  │
│ (FSDP)   │    │                   │  │              │
│          │    │ - Controller(s)   │  │ - Continuous │
│ - Actor  │    │   (Metadata)      │  │   generation │
│ - Critic │    │ - Storage Units   │  │ - Version    │
│ - Ref    │    │   (Data, ZMQ)     │  │   tracking   │
│          │    │ - Per-field       │  │ - Weight     │
│          │    │   tracking        │  │   updates    │
└──────────┘    └───────────────────┘  └──────────────┘
      │                ▲                       │
      │   async_get    │    async_put          │
      └────────────────┴──────────────────┘
              Async Training Loop
```

**Clean design principles**:
- Only vLLM's EngineCore is patched (unavoidable - no extension points)
- All verl classes extended via clean subclassing
- Worker methods injected via dependency injection (no monkey-patching)
- TransferDock is Ray remote actor (multi-node from day 1)

## Files

```
recipe/async_rl/
├── transfer_dock.py             # Ray-based distributed sample buffer
├── ray_trainer.py               # AsyncRLTrainer with async loops
├── vllm_engine_patches.py       # Patches vLLM EngineCore
├── async_rl_vllm_rollout.py     # Extends vLLMAsyncRollout
├── extended_vllm_server.py      # Extends vLLMHttpServer + vLLMReplica
├── vllm_worker_hooks.py         # Worker RPC methods
├── replica_factory.py           # Factory for injection
├── weight_update.py             # HTTP coordination
├── partial_rollout_manager.py   # Version tracking
├── ppo_loss.py                  # Decoupled PPO loss
├── main_ppo.py                  # Entry point
└── tests/
    ├── test_transfer_dock.py    # TransferDock unit tests
    └── test_ppo_loss.py         # PPO loss unit tests
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

### 2. Per-Token Metadata (Following AReaL)

```python
# Store per-token behavior logprobs and policy versions
output.batch["old_logprobs"] = rollout_log_probs  # π_old(a_t|s_t) per token
output.batch["token_policy_versions"] = token_versions  # Policy version per token

# Sample-level version tracking
output.non_tensor_batch["version_start"] = version_at_generation_start
output.non_tensor_batch["version_end"] = version_at_generation_end

# Staleness validation: reject if weight update happened mid-generation
if any(token_versions != token_versions[0]):
    # Sample crossed weight update boundary - rejected by TransferDock!
    pass
```

### 3. No Chunked Generation

AReaL's code uses chunked generation but calls it a "hack":
> "This is a hack usage. We don't need it if the server can pause requests, update weights, and recompute kv caches"

With `abort_all_requests()` + `reset_prefix_cache()`, we accept wasted work for simpler code.

## Extending the Sample Schema

**Adding new fields is easy - update only 1-2 places!**

Example: Add per-token `entropy` for exploration tracking:

### Step 1: Update DEFAULT_EXPERIENCE_COLUMNS (1 place)

```python
# In recipe/async_rl/transfer_dock.py
DEFAULT_EXPERIENCE_COLUMNS = [
    "input_ids",
    "attention_mask",
    "loss_mask",
    "old_logprobs",
    "token_policy_versions",
    "rewards",
    "entropy",  # ← Add here!
    "version_start",
    "version_end",
]
```

### Step 2: Produce the data (1 place)

```python
# In your data producer (PartialRolloutManager or training loop)
output.batch["entropy"] = computed_entropy  # Per-token entropy values
```

**That's it!** TransferDock automatically handles:
- Storage (any columns in experience_columns)
- Batching/padding (via _batch_to_samples and _samples_to_batch)
- Staleness validation (via StalenessValidator)

No need to update:
- ❌ Ray actor methods
- ❌ put_experience/get_experience logic
- ❌ Batching/padding code
- ❌ Multiple files scattered across the codebase

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
