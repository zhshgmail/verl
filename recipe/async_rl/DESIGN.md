# AsyncRL Recipe - Design Document

Complete architecture, design decisions, and implementation details for the AsyncRL recipe.

## Table of Contents

1. [Design Principles](#design-principles)
2. [Architecture](#architecture)
3. [Component Details](#component-details)
4. [Design Decisions](#design-decisions)
5. [Implementation Journey](#implementation-journey)
6. [Upstream Integration](#upstream-integration)

---

## Design Principles

### 1. Minimize Patching, Maximize Clean OOP

**Goal**: Extend verl/vLLM with minimal monkey-patching.

**What we patch**:
- ✅ **Only vLLM EngineCore** (unavoidable - no extension points)

**What we extend cleanly**:
- ✅ **vLLMAsyncRollout** → `AsyncRLvLLMAsyncRollout` (override `_execute_method`)
- ✅ **vLLMHttpServer** → `AsyncRLvLLMHttpServer` (override `run_server`)
- ✅ **vLLMReplica** → `AsyncRLvLLMReplica` (inject worker class)
- ✅ **AgentLoopManager** → `PartialRolloutManager` (add version tracking)
- ✅ **RayPPOTrainer** → `AsyncRLTrainer` (add async RL logic)

### 2. Dependency Injection over Monkey-Patching

**Pattern**: Use verl's built-in extension points.

```python
# ✅ GOOD: Dependency injection via get_ray_class_with_init_args()
class AsyncRLvLLMReplica(vLLMReplica):
    def get_ray_class_with_init_args(self):
        return RayClassWithInitArgs(cls=AsyncRLvLLMAsyncRollout, ...)

# ❌ BAD: Monkey-patching worker instances
def install_hooks(worker):
    worker.execute_method = wrapper(worker.execute_method)
```

### 3. Fail Early and Clearly

**SGLang support**: Not implemented yet. Fail with clear error instead of silent fallback.

```python
# ✅ GOOD: Explicit error with helpful message
if rollout == "sglang" and enable_async_rl:
    raise NotImplementedError("AsyncRL not implemented for SGLang yet...")

# ❌ BAD: Warning + silent fallback
if rollout == "sglang" and enable_async_rl:
    logger.warning("AsyncRL not implemented, falling back...")
    return SGLangReplica  # User thinks AsyncRL is working!
```

---

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     AsyncRLvLLMReplica                      │
│  - Injects AsyncRLvLLMAsyncRollout worker class             │
│  - Creates AsyncRLvLLMHttpServer instances                  │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌──────────────────┐    ┌──────────────────────┐
│ AsyncRLvLLM      │    │ AsyncRLvLLM          │
│ AsyncRollout     │    │ HttpServer           │
│ (Worker)         │    │ (Server)             │
│                  │    │                      │
│ - Override       │    │ - Override           │
│   _execute_      │    │   run_server()       │
│   method()       │    │ - Add /async_rl/*    │
│ - Handle custom  │    │   endpoints          │
│   RPC methods    │    │                      │
└──────────────────┘    └──────────────────────┘
         │                       │
         │                       ▼
         │              ┌──────────────────┐
         │              │ vLLM AsyncLLM    │
         │              │ EngineCore       │
         │              │ (Patched)        │
         │              │                  │
         │              │ + abort_all_     │
         │              │   requests()     │
         └──────────────► + reset_prefix_  │
                        │   cache()        │
                        └──────────────────┘
```

### Flow: Weight Update via NCCL

```
1. HTTP POST /async_rl/init_nccl_group
   ↓
2. AsyncRLvLLMHttpServer endpoint
   ↓
3. llm.collective_rpc("init_nccl_weight_update_group", ...)
   ↓
4. ZeroMQ RPC to all workers
   ↓
5. AsyncRLvLLMAsyncRollout._execute_method()
   ↓
6. init_nccl_weight_update_group(worker, ...)
   ↓
7. Initialize NCCL group on worker

---

1. HTTP POST /async_rl/update_weights_nccl
   ↓
2. AsyncRLvLLMHttpServer endpoint
   ↓
3. llm.engine_core.call_utility_async("async_rl_update_weights_nccl")
   ↓
4. EngineCore.abort_all_requests() + reset_prefix_cache()
   ↓
5. Workers receive weights via NCCL broadcast
   ↓
6. Workers update model parameters
```

---

## Component Details

### 1. vllm_engine_patches.py

**Purpose**: Patch vLLM EngineCore to add weight update + abort capabilities.

**Why patching is needed**: vLLM doesn't expose APIs for:
- Aborting all in-flight requests
- Resetting prefix cache
- Updating weights on the fly

**Methods added**:
```python
def abort_all_requests(self):
    """Abort all running/waiting requests and reset prefix cache."""
    # Get all requests
    abort_lists = list(scheduler.running) + list(scheduler.waiting)

    # Create abort outputs
    for req in abort_lists:
        outputs.append(RequestOutput(..., finished=True))

    # Mark as aborted
    scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    # CRITICAL: Reset prefix cache to prevent KV pollution
    scheduler.reset_prefix_cache()

    return outputs

def async_rl_update_weights_nccl(self):
    """Update weights via NCCL then abort requests."""
    # 1. Abort all requests FIRST
    abort_outputs = self.abort_all_requests()

    # 2. Update weights via NCCL
    ret_list = await self.collective_rpc("update_weights_nccl")

    return ret_list

def async_rl_update_weights_disk(self, model_path: str):
    """Update weights from disk then abort requests."""
    # 1. Abort all requests FIRST
    abort_outputs = self.abort_all_requests()

    # 2. Update weights from disk
    ret_list = await self.collective_rpc("async_rl_update_weights", args=(model_path,))

    return ret_list
```

**Auto-install on import**:
```python
setattr(EngineCore, "abort_all_requests", abort_all_requests)
setattr(EngineCore, "async_rl_update_weights_nccl", async_rl_update_weights_nccl)
setattr(EngineCore, "async_rl_update_weights_disk", async_rl_update_weights_disk)
```

### 2. async_rl_vllm_rollout.py

**Purpose**: Extend vLLMAsyncRollout to handle custom RPC methods.

**Key insight**: verl already provides the perfect extension point!
- `vLLMAsyncRollout._execute_method()` is designed for extension
- `vLLMReplica.get_ray_class_with_init_args()` allows worker class injection

**Implementation**:
```python
class AsyncRLvLLMAsyncRollout(vLLMAsyncRollout):
    async def _execute_method(self, method: str | bytes, *args, **kwargs):
        worker = self.inference_engine.worker

        # Handle AsyncRL custom methods
        if method == "init_nccl_weight_update_group":
            return init_nccl_weight_update_group(worker, *args, **kwargs)
        elif method == "set_nccl_weight_meta":
            return set_nccl_weight_meta(worker, *args, **kwargs)
        elif method == "update_weights_nccl":
            return update_weights_nccl(worker, *args, **kwargs)
        elif method == "async_rl_update_weights":
            return update_weights(worker, *args, **kwargs)
        else:
            # Delegate to parent for standard vLLM methods
            return await super()._execute_method(method, *args, **kwargs)
```

**Why this is clean**:
- ✅ No monkey-patching - just method override
- ✅ Clear delegation to parent for standard methods
- ✅ Workers created with right class from the start

### 3. extended_vllm_server.py

**Purpose**: Extend vLLMHttpServer and vLLMReplica for AsyncRL.

**AsyncRLvLLMHttpServer** (extends vLLMHttpServer):
```python
class AsyncRLvLLMHttpServer(vLLMHttpServer):
    async def run_server(self, args):
        # 1. Call parent to build app
        app = build_app(args)
        await init_app_state(engine_client, vllm_config, app.state, args)

        # 2. Add custom endpoints to THIS app instance
        self._add_custom_endpoints(app)

        # 3. Continue with parent logic
        self._server_port, self._server_task = await run_unvicorn(app, args, ...)

    def _add_custom_endpoints(self, app: FastAPI):
        @app.post("/async_rl/update_weights")
        async def update_weights_from_disk(...):
            # Call patched EngineCore method
            ret_list = await llm.engine_core.call_utility_async(
                "async_rl_update_weights_disk", model_path
            )
            return build_response(ret_list)

        @app.post("/async_rl/init_nccl_group")
        async def init_nccl_group(...):
            # Call worker RPC method
            ret_list = await llm.collective_rpc(
                "init_nccl_weight_update_group", args=(...)
            )
            return build_response(ret_list)

        # ... more endpoints
```

**AsyncRLvLLMReplica** (extends vLLMReplica):
```python
class AsyncRLvLLMReplica(vLLMReplica):
    def get_ray_class_with_init_args(self):
        """Inject AsyncRLvLLMAsyncRollout worker class."""
        _async_rl_worker_actor_cls = ray.remote(AsyncRLvLLMAsyncRollout)

        return RayClassWithInitArgs(
            cls=_async_rl_worker_actor_cls,
            config=self.config,
            model_config=self.model_config,
            device_mesh=None,
        )

    async def launch_servers(self):
        """Create AsyncRLvLLMHttpServer instead of vLLMHttpServer."""
        # ... create servers using AsyncRLvLLMHttpServer
        server = AsyncRLvLLMHttpServer.options(...).remote(...)
```

**Why this is clean**:
- ✅ Dependency injection via `get_ray_class_with_init_args()`
- ✅ Endpoints added to app instance (not global patching)
- ✅ Workers automatically have custom RPC methods

### 4. vllm_worker_hooks.py

**Purpose**: Worker-side NCCL weight update methods.

**Methods** (called by `AsyncRLvLLMAsyncRollout._execute_method`):
- `init_nccl_weight_update_group()` - Initialize NCCL group
- `set_nccl_weight_meta()` - Set parameter metadata
- `update_weights_nccl()` - Receive weights via NCCL
- `update_weights()` - Load weights from disk

**Example**:
```python
def update_weights_nccl(self, group_name="async_rl_weight_update"):
    """Update weights via NCCL broadcast."""
    metadata = self._nccl_weight_groups[group_name]["param_metadata"]

    # Receive weights
    received_weights = {}
    for name, dtype_str, shape in zip(names, dtypes, shapes):
        tensor = torch.empty(shape, dtype=dtype, device="cuda")
        dist.recv(tensor, src=0)  # From training workers
        received_weights[name] = tensor

    # Update model
    state_dict = self.model.state_dict()
    for name, weight in received_weights.items():
        state_dict[name].copy_(weight)

    return True, f"Updated {len(received_weights)} parameters"
```

### 5. replica_factory.py

**Purpose**: Factory to inject AsyncRLvLLMReplica without modifying verl.

**Problem**: verl's `get_rollout_replica_class()` is hardcoded:
```python
# verl/workers/rollout/replica.py
def get_rollout_replica_class(rollout: str):
    if rollout == "vllm":
        return vLLMReplica  # Hardcoded!
```

**Solution**: Provide alternative factory with backward compatibility:
```python
def get_async_rl_replica_class(rollout: str, enable_async_rl: bool = True):
    if rollout == "vllm":
        if enable_async_rl:
            from recipe.async_rl.extended_vllm_server import AsyncRLvLLMReplica
            return AsyncRLvLLMReplica
        else:
            from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica
            return vLLMReplica  # Backward compatible!

    elif rollout == "sglang":
        if enable_async_rl:
            raise NotImplementedError("AsyncRL not implemented for SGLang yet...")
        # Return standard SGLang
        from verl.workers.rollout.sglang_rollout.http_server_engine import SGLangReplica
        return SGLangReplica
```

**Optional monkey-patch for integration**:
```python
def patch_get_rollout_replica_class(enable_async_rl: bool = None):
    """Patch verl's factory with backward-compatible wrapper."""
    def wrapper(rollout: str):
        # Read from config if not specified
        should_enable = enable_async_rl or read_from_config()
        return get_async_rl_replica_class(rollout, should_enable)

    verl.workers.rollout.replica.get_rollout_replica_class = wrapper
```

---

## Design Decisions

### 1. Why No Chunked Generation?

**AReaL uses chunked generation** but calls it a "hack":
> "This is a hack usage. We don't need it if the server can pause requests, update weights, and recompute kv caches"

**Our approach**: Accept wasted work for simpler code.

**Tradeoff**:
- ❌ Some generation work is wasted when requests are aborted
- ✅ Much simpler code - no chunking logic
- ✅ Easier to maintain and debug
- ✅ `abort_all_requests()` + `reset_prefix_cache()` is sufficient

### 2. Why Dependency Injection Instead of Monkey-Patching Workers?

**Initial approach (wrong)**:
```python
# ❌ BAD: Remote hook installation
def install_worker_hooks(inference_engine):
    inference_engine.execute_method = wrapper(...)

# Call remotely on each worker
await worker.__ray_call__.remote(install_worker_hooks)
```

**Final approach (correct)**:
```python
# ✅ GOOD: Dependency injection
class AsyncRLvLLMReplica(vLLMReplica):
    def get_ray_class_with_init_args(self):
        return RayClassWithInitArgs(cls=AsyncRLvLLMAsyncRollout, ...)
```

**Why better**:
- ✅ Follows verl's design patterns
- ✅ Workers created with right class from the start
- ✅ No remote installation needed
- ✅ Easier to upstream

### 3. Why Fail on SGLang + AsyncRL?

**Wrong approach**:
```python
# ❌ BAD: Silent fallback
if rollout == "sglang" and enable_async_rl:
    logger.warning("AsyncRL not implemented, falling back...")
    return SGLangReplica  # User thinks AsyncRL is working!
```

**Correct approach**:
```python
# ✅ GOOD: Fail early with clear error
if rollout == "sglang" and enable_async_rl:
    raise NotImplementedError("AsyncRL not implemented for SGLang yet...")
```

**Why**:
- ✅ User immediately knows what's wrong
- ✅ Clear error message explains why and how to fix
- ✅ Prevents silent failures in production
- ✅ SGLang + standard mode still works fine

---

## Implementation Journey

### Phase 1: Initial Attempt (Too Much Patching)

**Problems**:
- ❌ Global router patching with `@router.post()`
- ❌ Duplicate files (vllm_server.py, vllm_server_extension.py)
- ❌ Not backward-compatible (broke SGLang)
- ❌ Confusing module names

### Phase 2: Architecture Clarification

**Improvements**:
- ✅ Separated vLLM patches from verl extensions
- ✅ Clean subclassing of vLLMHttpServer
- ✅ Deleted duplicate files

### Phase 3: Worker Hook Installation (Wrong)

**Approach**:
- Remote installation via `worker.__ray_call__.remote()`
- Wrapping `execute_method` on each worker

**Why wrong**:
- Not using verl's extension points
- Complicated remote installation logic
- Monkey-patching workers instead of clean subclassing

### Phase 4: Clean Dependency Injection (Final)

**Discovery**: verl already has the perfect extension point!
- `vLLMAsyncRollout._execute_method()` is designed for extension
- `get_ray_class_with_init_args()` allows worker class injection

**Solution**:
- ✅ Subclass `vLLMAsyncRollout` → `AsyncRLvLLMAsyncRollout`
- ✅ Override `_execute_method()` to handle custom RPC methods
- ✅ Inject via `get_ray_class_with_init_args()`

**Result**: Zero monkey-patching for workers!

---

## Upstream Integration

### To vLLM

**File**: `vllm_engine_patches.py` → Contribute as official API

**Methods to add**:
```python
# vllm/v1/engine/core.py
class EngineCore:
    def abort_all_requests(self) -> list[RequestOutput]:
        """Abort all in-flight requests and reset prefix cache."""
        # ... implementation

    async def update_weights_async(self, model_path: str):
        """Update model weights from checkpoint."""
        # ... implementation
```

### To verl

**1. Add AsyncRL worker class**:
```
verl/workers/rollout/vllm_rollout/
└── async_extensions.py  # AsyncRLvLLMAsyncRollout
```

**2. Add AsyncRL server class**:
```
verl/workers/rollout/vllm_rollout/
└── async_server.py  # AsyncRLvLLMHttpServer
```

**3. Add AsyncRL replica class**:
```
verl/workers/rollout/vllm_rollout/
└── async_replica.py  # AsyncRLvLLMReplica
```

**4. Add configuration flag**:
```python
# verl/workers/config.py
@dataclass
class RolloutConfig:
    # ... existing fields
    enable_async_weight_updates: bool = False  # Default: disabled
```

**5. Add plugin/registry system**:
```python
# verl/workers/rollout/replica.py
class RolloutReplicaRegistry:
    _registry = {
        "vllm": vLLMReplica,
        "sglang": SGLangReplica,
        "async_vllm": AsyncRLvLLMReplica,  # Register AsyncRL
    }

    @classmethod
    def register(cls, name: str, replica_cls: type):
        cls._registry[name] = replica_cls

    @classmethod
    def get(cls, name: str):
        return cls._registry[name]

def get_rollout_replica_class(rollout: str, config: RolloutConfig = None):
    """Get replica class with AsyncRL support."""
    if config and config.enable_async_weight_updates:
        return RolloutReplicaRegistry.get(f"async_{rollout}")
    return RolloutReplicaRegistry.get(rollout)
```

**Benefits**:
- ✅ Backward compatible (enable_async_weight_updates=False by default)
- ✅ Configuration-driven
- ✅ Easy to extend with more backends
- ✅ Already follows clean OOP patterns

---

## Summary

**What we built**:
- Complete asynchronous RL recipe with decoupled PPO
- Weight updates during generation
- Request abortion + cache reset
- Version tracking for staleness-aware training

**How we built it**:
- ✅ Minimal patching (only vLLM EngineCore)
- ✅ Clean OOP (subclassing + dependency injection)
- ✅ Backward compatible (SGLang still works)
- ✅ Fail early (clear errors for unsupported cases)

**Ready for**:
- ✅ Immediate use (via factory pattern)
- ✅ Future upstream (clean integration paths)
- ✅ Easy maintenance (well-organized, documented)
