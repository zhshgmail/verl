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
┌────────────────────────────────────────────────────────────────┐
│                      AsyncRLTrainer                            │
│  - Manages async rollout & training loops                      │
│  - Version tracking (policy_version)                           │
│  - Staleness-aware training                                    │
└────┬──────────────────────┬──────────────────────┬─────────────┘
     │                      │                      │
     ▼                      ▼                      ▼
┌─────────────┐   ┌──────────────────┐   ┌──────────────────────┐
│ Training    │   │ TransferQueue    │   │ Rollout Pool         │
│ Pool        │   │ (Distributed)    │   │ (vLLM HTTP)          │
│ (Actor)     │   │                  │   │                      │
│             │   │ - Controllers    │   │ - Continuous         │
│ - FSDP      │   │   (Metadata)     │   │   generation         │
│ - Critic    │   │ - Storage Units  │   │ - Version tracking   │
│ - Reference │   │   (Data, ZMQ)    │   │ - Weight updates     │
│             │   │ - Per-field      │   │   via NCCL           │
│             │   │   tracking       │   │                      │
└─────────────┘   └──────────────────┘   └──────────────────────┘
     │                      ▲                      │
     │                      │                      │
     │      async_get()     │   async_put()        │
     └──────────────────────┴──────────────────────┘
                   Async Training Loop

                           ▼
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

### 4. transfer_queue_adapter.py

**Purpose**: Production-grade distributed data management for multi-node async RL.

**Design**: Wraps [TransferQueue](https://github.com/TransferQueue/TransferQueue) (from [AsyncFlow paper](https://arxiv.org/abs/2507.01663)) for AsyncRL.

**Why TransferQueue?**
- ✅ Production-grade (published paper, active development)
- ✅ True distributed storage (multiple storage units across nodes)
- ✅ ZMQ communication (high-performance, avoids Ray ObjectRef bottleneck)
- ✅ Panoramic metadata tracking (per-sample per-field production/consumption status)
- ✅ Built-in multi-consumer support (independent task consumption tracking)
- ✅ Future RDMA support (MoonCakeStore backend)

**AsyncRLTransferQueueAdapter** (wraps AsyncTransferQueueClient):
```python
class AsyncRLTransferQueueAdapter:
    """
    Adapter that wraps TransferQueue for AsyncRL training.

    Features:
    - Automatic metadata management
    - Version tracking via global_step
    - Per-token metadata support (old_logprobs, token_policy_versions)
    - Staleness validation via filtering
    """

    def __init__(self, client: AsyncTransferQueueClient, experience_columns: List[str], ...):
        self.client = client  # AsyncTransferQueueClient
        self.experience_columns = experience_columns
        self.task_name = "async_rl_training"  # Multi-consumer tracking

    async def async_put_experience(self, data: Dict[str, Tensor], current_version: int):
        """Put experience using current_version as global_step"""
        tensor_dict = TensorDict(data, batch_size=...)
        await self.client.async_put(data=tensor_dict, global_step=current_version)

    async def async_get_experience(self, experience_count: int, current_version: int):
        """Get experience with staleness filtering (version >= current - max_staleness)"""
        min_version = current_version - self.max_staleness
        all_metas = []

        # Try to get metadata from recent steps
        for step in range(current_version, min_version - 1, -1):
            meta = await self.client.async_get_meta(
                data_fields=self.experience_columns,
                batch_size=needed_count,
                global_step=step,
                task_name=self.task_name,
                mode="fetch"
            )
            if meta: all_metas.append(meta)

        # Get actual data
        combined_meta = BatchMeta.concat(all_metas)
        data = await self.client.async_get_data(metadata=combined_meta)
        return dict(data)
```

**TransferQueue System Architecture**:
```
┌─────────────────────────────────────────────┐
│     TransferQueue (Distributed System)     │
├─────────────────────────────────────────────┤
│                                             │
│  ┌─────────────┐      ┌─────────────┐     │
│  │ Controller  │      │ Storage #0  │     │
│  │ (Metadata)  │──────│ (ZMQ)       │     │
│  │             │      └─────────────┘     │
│  │ - 2D status │      ┌─────────────┐     │
│  │   [samples  │──────│ Storage #1  │     │
│  │    × fields]│      │ (ZMQ)       │     │
│  └─────────────┘      └─────────────┘     │
│                                             │
│  - Per-field production status             │
│  - Per-task consumption status             │
│  - Dynamic field registration              │
└─────────────────────────────────────────────┘
```

**Configuration via YAML**:
```yaml
async_rl:
  max_staleness: 5
  train_batch_size: 128

  transfer_queue:
    num_storage_units: 2  # Distribute across 2 nodes
    num_controllers: 1  # Single controller
    num_global_batch: 2  # Buffer 2 batches (256 samples total)
    storage_cpus_per_unit: 1
    controller_cpus: 1
```

**Key Advantages over simple TransferDock**:
- ✅ **Scalability**: Distributed storage units (horizontal scaling)
- ✅ **Performance**: ZMQ communication (no Ray ObjectRef bottleneck)
- ✅ **Metadata**: Fine-grained per-field tracking
- ✅ **Multi-consumer**: Built-in task consumption tracking
- ✅ **Future-proof**: Active development with RDMA support planned

### 5. vllm_worker_hooks.py

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

### 5. ray_trainer.py

**Purpose**: AsyncRLTrainer orchestrates async RL training with TransferDock.

**AsyncRLTrainer** (extends RayPPOTrainer):
```python
class AsyncRLTrainer(RayPPOTrainer):
    """
    Trainer for asynchronous RL with decoupled PPO objective.

    Architecture:
    - Training Pool (actor_pool): Actor, Critic, Reference
    - Rollout Pool (rollout_pool): vLLM HTTP servers
    - AsyncRLTransferDock: Distributed sample buffer
    - Concurrent rollout and training loops
    """

    def __init__(self, config: DictConfig):
        # Extract async RL configs
        self.buffer_size = config.async_rl.buffer_size
        self.max_staleness = config.async_rl.max_staleness
        self.train_batch_size = config.async_rl.train_batch_size

        # Policy version tracking
        self.policy_version = 0

        # Create TransferDock (Ray remote actor)
        self.transfer_dock = create_transfer_dock(
            max_size=self.buffer_size,
            max_staleness=self.max_staleness,
        )

    async def async_rollout_loop(self, data_loader):
        """Continuous rollout: generate → annotate → put_experience()"""
        while not self._stop_event.is_set():
            # Generate samples
            rollout_data = await self.async_rollout_manager.generate(prompt_batch)

            # Annotate with version
            rollout_data["version_start"] = self.policy_version
            rollout_data["version_end"] = self.policy_version

            # Put into buffer
            await self.transfer_dock.put_experience.remote(rollout_data)

    async def async_training_loop(self, num_steps: int, ppo_epochs: int):
        """Continuous training: get_experience() → compute_advantages() → ppo_update()"""
        while training_step < num_steps:
            # Get batch with staleness filtering
            batch_dict, indices = await self.transfer_dock.get_experience.remote(
                consumer="trainer",
                experience_columns=["input_ids", "logprobs", ...],
                experience_count=self.train_batch_size,
                current_version=self.policy_version,  # Filter stale samples
            )

            if batch_dict is None:
                await asyncio.sleep(1.0)  # Wait for more samples
                continue

            # Compute advantages with version tracking
            data = self.compute_advantages(batch_dict)

            # Train for multiple epochs
            for epoch in range(ppo_epochs):
                metrics = self.training_step(data)

            # Sync weights periodically
            if training_step % sync_freq == 0:
                self.sync_weights_to_rollout()

    def train_async(self, data_loader, num_steps: int, ppo_epochs: int):
        """Run rollout and training loops concurrently."""
        rollout_task = asyncio.create_task(self.async_rollout_loop(data_loader))
        training_task = asyncio.create_task(self.async_training_loop(num_steps, ppo_epochs))

        asyncio.get_event_loop().run_until_complete(training_task)
```

**Key features**:
- ✅ Concurrent rollout + training (true async RL)
- ✅ TransferDock handles thread-safe sample storage
- ✅ Staleness filtering at sampling time
- ✅ Version tracking for off-policy correction

### 6. replica_factory.py

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

### 3. Why train_async() Instead of Overriding fit()?

**Question**: Why create a new `train_async()` method instead of overriding `fit()`?

**Answer**: **Fundamentally different execution models**:

**RayPPOTrainer.fit()** (~380 lines):
- Sequential pipeline: generate → reward → train (one batch at a time)
- Dataloader-driven (for loop over epochs/batches)
- Synchronous blocking (each step waits for completion)
- Tight coupling (all steps in one method)

**AsyncRLTrainer.train_async()**:
- Concurrent loops: continuous rollout + continuous training
- Buffer-driven (TransferDock intermediary)
- Asynchronous non-blocking (await, parallel execution)
- Loose coupling (separate rollout/training loops)

**Why not override fit()?**

1. **Different execution model**: Cannot express concurrency in fit()'s for-loop
   ```python
   # fit(): Sequential
   for batch in dataloader:
       generate(batch)  # Wait
       train(batch)     # Wait

   # train_async(): Concurrent
   rollout_loop:  generate → generate → generate → ...
   training_loop: train → train → train → ...
   # Both run at same time!
   ```

2. **Different data flow**: fit() has direct flow (batch → generate → train), train_async() has buffered flow (generate → TransferDock → sample → train)

3. **Different control flow**: fit() is epoch/batch-driven, train_async() is step-driven with staleness

4. **Would require complete rewrite**: 380+ lines changed, breaks all parent assumptions

**Benefits of separate method**:
- ✅ Clear API (user knows what to expect)
- ✅ Preserve parent's fit() (can still use standard PPO)
- ✅ Explicit execution model (no confusion)
- ✅ Follows precedent (MindSpeed-RL also uses separate `fit_with_partial_rollout()`)

### 4. Per-Token Metadata Storage (Following AReaL)

**Why:** Decoupled PPO requires accurate behavior logprobs and staleness tracking.

**What we store per token**:
```python
# Per-token data (shape: [batch_size, seqlen])
old_logprobs           # π_old(a_t|s_t) - behavior logprobs from rollout
token_policy_versions  # Policy version when each token was generated

# Per-sample data (shape: [batch_size])
version_start          # Policy version at generation start
version_end            # Policy version at generation end
```

**Data flow**:
1. **Rollout** (PartialRolloutManager):
   - vLLM returns `rollout_log_probs` (behavior policy logprobs)
   - Rename to `old_logprobs` (semantic clarity)
   - Add `token_policy_versions` (same version for all tokens currently)
   - Add `version_start`, `version_end` (sample-level tracking)

2. **TransferDock** (validation + storage):
   - Validate staleness: reject if `token_policy_versions` has multiple unique values
   - This means weight update happened mid-generation (staleness broken)
   - Store all per-token metadata

3. **Training** (AsyncRLTrainer):
   - Retrieve `old_logprobs` from TransferDock
   - Recompute `proximal_logprobs` (π_prox from recent checkpoint)
   - Compute `logprobs` (π_θ from current policy)
   - Pass all 3 to decoupled PPO loss

4. **Decoupled PPO Loss**:
   ```python
   ratio = exp(logprobs - proximal_logprobs)       # π_θ / π_prox (for clipping)
   behav_weight = exp(proximal_logprobs - old_logprobs)  # π_prox / π_old (staleness)
   loss = clip(ratio) * advantages * behav_weight
   ```

**Why per-token instead of per-sample?**
- ✅ Detect weight updates mid-generation (reject samples with broken staleness)
- ✅ More accurate importance weighting (each token has correct behavior logprob)
- ✅ Matches AReaL's design

**Staleness validation**:
```python
class StalenessValidator:
    @staticmethod
    def validate_sample(sample):
        # Per-token validation (preferred)
        if "token_policy_versions" in sample:
            unique_versions = sample["token_policy_versions"].unique()
            if len(unique_versions) > 1:
                return False  # Weight update mid-generation!

        # Fallback: sample-level validation
        if sample["version_start"] != sample["version_end"]:
            return False

        return True
```

**Extensibility design**:

To add new fields (e.g., `entropy`), update **only 2 places**:

1. **DEFAULT_EXPERIENCE_COLUMNS** (1 line in `transfer_dock.py`):
   ```python
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

2. **Data producer** (1 line in `PartialRolloutManager` or training loop):
   ```python
   output.batch["entropy"] = computed_entropy
   ```

**What you DON'T need to update**:
- ❌ Ray actor methods
- ❌ put_experience/get_experience logic
- ❌ Batching/padding code
- ❌ Multiple scattered files

**Why this works**:
- `DEFAULT_EXPERIENCE_COLUMNS` is single source of truth
- `_batch_to_samples()` and `_samples_to_batch()` are generic (loop over columns)
- Validation logic is separate (StalenessValidator)

---

## Architecture Clarifications

### Single Controller Pattern

**Yes, this recipe follows the single controller design pattern**, inherited from RayPPOTrainer:

```
┌────────────────────────────────────────────────────┐
│           Single Controller (Driver)               │
│  - Runs on head node / designated controller       │
│  - Orchestrates all worker groups via RPC          │
│  - Manages TransferDock (Ray remote actor)         │
│  - Runs async_rollout_loop() and                   │
│    async_training_loop() as controller tasks       │
└────────────────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Training     │ │ Rollout      │ │ TransferDock │
│ WorkerGroup  │ │ WorkerGroup  │ │ (Ray Actor)  │
│              │ │              │ │              │
│ world_size=N │ │ world_size=M │ │ Single actor │
│              │ │              │ │              │
│ Workers:     │ │ Workers:     │ │              │
│ - Actor[0]   │ │ - vLLM[0]    │ │              │
│ - Actor[1]   │ │ - vLLM[1]    │ │              │
│ - ...        │ │ - ...        │ │              │
│ - Critic[0]  │ │ - vLLM[M-1]  │ │              │
│ - Critic[1]  │ │              │ │              │
└──────────────┘ └──────────────┘ └──────────────┘
```

### Multiple Workers: YES ✅

**Training WorkerGroup** (inherited from RayPPOTrainer):
- `self.actor_rollout_wg.world_size` = N (e.g., 8 workers)
- `self.critic_wg.world_size` = N
- Uses FSDP/Megatron data parallel + tensor parallel

**Rollout WorkerGroup**:
- `self.rollout_wg.world_size` or `self.actor_rollout_wg.world_size` = M (e.g., 4 workers)
- Each worker runs vLLM HTTP server
- Managed by PartialRolloutManager

### CRITICAL ISSUE: Single Task vs Multiple Workers ❌

**Current Implementation (INCORRECT)**:
```python
def train_async(self, data_loader, num_steps, ppo_epochs):
    # Creates ONLY 2 tasks on controller
    rollout_task = asyncio.create_task(self.async_rollout_loop(data_loader))
    training_task = asyncio.create_task(self.async_training_loop(num_steps, ppo_epochs))

    # Problem: These are controller tasks, not worker tasks!
    # - rollout_task calls self.async_rollout_manager (which dispatches to M workers)
    # - training_task calls self.training_step() (which dispatches to N workers)
```

**What actually happens**:
1. **Controller runs ONE rollout loop**: Calls `async_rollout_manager.generate()` which internally dispatches to M rollout workers
2. **Controller runs ONE training loop**: Calls `training_step()` which internally dispatches to N training workers
3. **WorkerGroups handle parallelism**: The dispatch happens inside generate() and training_step()

**This is CORRECT for single controller pattern!** ✅

### Clarification: Controller Tasks ≠ Worker Tasks

**Common confusion**:
- ❌ "We need M rollout tasks for M rollout workers"
- ❌ "We need N training tasks for N training workers"

**Reality**:
- ✅ **ONE controller rollout task** → dispatches to M rollout workers via `async_rollout_manager.generate()`
- ✅ **ONE controller training task** → dispatches to N training workers via `self.actor_rollout_wg.update_actor()`

**How WorkerGroups handle multiple workers**:
```python
# In async_rollout_loop (controller task)
rollout_data = await self.async_rollout_manager.generate(prompt_batch)
# ↑ This internally:
#   1. Splits prompt_batch across M rollout workers
#   2. Each worker generates in parallel
#   3. Collects results from all workers
#   4. Returns combined rollout_data

# In async_training_loop (controller task)
self.training_step(data)
# ↑ This internally:
#   1. Splits data across N training workers (DP sharding)
#   2. Each worker trains in parallel
#   3. Synchronizes gradients via NCCL
#   4. Returns aggregated metrics
```

### Diagram: Controller Tasks vs Worker Parallelism

```
Controller (Single Node):
┌─────────────────────────────────────────────┐
│  rollout_task (1 task)                      │
│    │                                         │
│    └─> async_rollout_manager.generate()     │
│           │                                  │
│           └─> Dispatches to M workers       │
│                                              │
│  training_task (1 task)                     │
│    │                                         │
│    └─> training_step()                      │
│           │                                  │
│           └─> Dispatches to N workers       │
└─────────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
Rollout Workers   Training Workers
┌────────────┐    ┌────────────┐
│ vLLM[0]    │    │ Actor[0]   │
│ vLLM[1]    │    │ Actor[1]   │
│ ...        │    │ ...        │
│ vLLM[M-1]  │    │ Critic[0]  │
└────────────┘    │ Critic[1]  │
                  └────────────┘
   M workers         N workers
   (parallel)        (parallel)
```

### Correct Design Pattern ✅

The current implementation is **correct for single controller pattern**:

1. **Controller orchestrates** (train_async creates 2 tasks)
2. **WorkerGroups parallelize** (generate() and training_step() dispatch to workers)
3. **TransferDock buffers** (decouples controller's rollout and training tasks)

**This matches verl's design**: RayPPOTrainer.fit() also runs on controller and dispatches to worker groups!

### What Would Multi-Controller Look Like? (Not This Recipe)

```python
# NOT what we're doing - this would be multi-controller
for worker_id in range(num_rollout_workers):
    rollout_tasks.append(
        asyncio.create_task(rollout_worker_loop(worker_id))
    )
# Each task runs on separate controller node
```

**This recipe uses SINGLE controller** (correct).

### 4. Why Fail on SGLang + AsyncRL?

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
