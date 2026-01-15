# R3 (Router Replay) vLLM NPU Compatibility Analysis

**Date**: 2025-12-29
**Branch**: `feature/npu-aqn-test`
**Author**: Claude Code

## Executive Summary

R3 (Rollout Router Replay) is implemented in verl main branch but **cannot work on NPU** due to:
1. Missing vLLM dependency (PR #28284 not merged)
2. vllm-ascend bypasses the routing capture mechanism

This document analyzes what's needed to enable R3 on Ascend NPU.

## Background: What is R3?

### R2 vs R3

| Mode | Full Name | Source of Routing | Works Today |
|------|-----------|-------------------|-------------|
| **R2** | Router Record & Replay | Megatron (`compute_log_prob`) | ✅ Yes |
| **R3** | Rollout Router Replay | vLLM (during generation) | ❌ No |

### Why R3 Matters for RL

In reinforcement learning (PPO/GRPO) with MoE models:
- **Rollout phase**: vLLM generates responses (uses routing decision A)
- **Training phase**: Megatron computes gradients (would use routing decision B)

Without R3, the same token goes to **different experts** in inference vs training, causing instability.

R3 captures vLLM's routing decisions and replays them in Megatron training.

## Current Status

### verl Main Branch

| Component | Status | Location |
|-----------|--------|----------|
| RouterReplayConfig | ✅ Implemented | `verl/workers/config/actor.py` |
| Megatron replay patch | ✅ Implemented | `verl/utils/megatron/router_replay_patch.py` |
| Actor integration | ✅ Implemented | `verl/workers/actor/megatron_actor.py` |
| Rollout config | ✅ Implemented | `enable_rollout_routing_replay` |
| vLLM capture | ⚠️ Expects vLLM support | `vllm_async_server.py` |
| Example script | ✅ Available | `examples/router_replay/` |

**Note**: Example script defaults to `ROUTING_REPLAY_MODE="R2"` because R3 dependency is not ready.

### vLLM Dependency

verl's R3 implementation expects:
```python
# vllm_async_server.py
args.update({"enable_return_routed_experts": True})
...
routed_experts = final_res.outputs[0].routed_experts
```

This requires vLLM PR: https://github.com/vllm-project/vllm/pull/28284

**PR Status** (as of 2025-12-29):
```json
{
  "state": "OPEN",
  "title": "[Feature] Support recording expert indices for rollout router replay",
  "mergedAt": null
}
```

**The PR is NOT merged.**

### NPU Limitation in verl

In `verl/utils/megatron/router_replay_patch.py`:
```python
try:
    from megatron.core.transformer.moe.moe_utils import (...)
except ImportError:
    warnings.warn("NPU not support router replay for now.", stacklevel=2)
```

NPU's Megatron implementation lacks required MoE utilities.

## PR #28284 Analysis

### Files Modified

| File | Changes | vLLM 0.11.x Compatible |
|------|---------|------------------------|
| `vllm/config/model.py` | Add `enable_return_routed_experts` | ✅ Yes |
| `vllm/engine/arg_utils.py` | Add CLI arg | ✅ Yes |
| `vllm/entrypoints/llm.py` | Add parameter | ✅ Yes |
| `vllm/model_executor/layers/fused_moe/layer.py` | Add capture hook | ✅ Yes |
| `vllm/model_executor/layers/fused_moe/routed_experts_capturer.py` | **NEW FILE** | ✅ Yes |
| `vllm/outputs.py` | Add `routed_experts` field | ✅ Yes |
| `vllm/v1/core/sched/scheduler.py` | Propagate routing | ✅ Yes |
| `vllm/v1/engine/output_processor.py` | Process routing | ✅ Yes |
| `vllm/v1/worker/gpu_model_runner.py` | Capture in worker | ⚠️ Needs NPU version |

### Key Addition: RoutedExpertsCapturer

PR adds a new file `routed_experts_capturer.py` with:
- `RoutedExpertsCapturer` class - captures routing during forward pass
- Uses shared memory for multi-process communication
- Captures `topk_ids` (which experts are selected) per layer

Capture happens in `FusedMoE.select_experts()`:
```python
capturer = RoutedExpertsCapturer.get_instance()
if capturer is not None:
    capturer.capture(layer_id=self.layer_id, topk_ids=topk_ids)
```

## vllm-ascend Compatibility Issue

### Architecture Difference

```
GPU vLLM (PR #28284):
  FusedMoE.select_experts()
    └── RoutedExpertsCapturer.capture(topk_ids)  ← CAPTURE HERE

NPU vllm-ascend:
  AscendUnquantizedFusedMoEMethod.apply()
    └── vllm_ascend.ops.moe.experts_selector.select_experts()  ← DIFFERENT PATH!
        └── NO CAPTURE
```

### Why vllm-ascend Bypasses Capture

In `/home/zheng/workspace/vllm-ascend/vllm_ascend/ops/common_fused_moe.py`:
```python
from vllm_ascend.ops.moe.experts_selector import select_experts  # NPU-specific!

class AscendUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    def apply(self, ...):
        topk_weights, topk_ids = select_experts(...)  # Uses NPU version
```

The NPU `select_experts()` in `vllm_ascend/ops/moe/experts_selector.py` does NOT have routing capture.

### Files Needing NPU Updates

| File | Change Needed |
|------|---------------|
| `vllm_ascend/ops/moe/experts_selector.py` | Add `RoutedExpertsCapturer` integration |
| `vllm_ascend/worker/model_runner_v1.py` | Propagate `routed_experts` through pipeline |
| Possibly `vllm_ascend/ops/common_fused_moe.py` | Ensure capture is called |

## Implementation Plan for NPU R3 Support

### Phase 1: Port PR #28284 to vLLM 0.11.x

**Effort**: Medium (1-2 days)

1. Cherry-pick or manually port PR #28284 changes
2. Test on GPU first to verify base functionality
3. Create local vLLM fork with changes

### Phase 2: Update vllm-ascend

**Effort**: Medium-High (2-3 days)

1. **Add RoutedExpertsCapturer to NPU path**:
   ```python
   # In vllm_ascend/ops/moe/experts_selector.py
   from vllm.model_executor.layers.fused_moe.routed_experts_capturer import RoutedExpertsCapturer

   def select_experts(...):
       topk_weights, topk_ids = _select_experts_with_fusion_ops(...)

       # Add capture
       capturer = RoutedExpertsCapturer.get_instance()
       if capturer is not None:
           capturer.capture(layer_id=layer_id, topk_ids=topk_ids)

       return topk_weights, topk_ids
   ```

2. **Update model_runner_v1.py**:
   - Add routing collection in execute_model
   - Pass through output pipeline

3. **Test with MoE model on NPU**

### Phase 3: Integration Testing

**Effort**: 1-2 days

1. Test vLLM routing capture on NPU
2. Test verl R3 mode end-to-end
3. Verify routing consistency between vLLM and Megatron

## Alternative: Use R2 Mode

If R3 porting effort is too high, R2 mode works today:

```yaml
actor_rollout_ref:
  actor:
    router_replay:
      mode: R2  # Works without vLLM changes
```

**Limitation**: R2 only ensures consistency within Megatron (compute_log_prob → update_policy), not vLLM → Megatron.

## Local Source Code Locations

| Component | Path | Version |
|-----------|------|---------|
| vLLM | `/home/zheng/workspace/vllm` | v0.11.1rc6-229-g80b6080dd |
| vllm-ascend | `/home/zheng/workspace/vllm-ascend` | v0.11.0-50-g81c358a2 |
| verl | `/home/zheng/workspace/verl` | feature/npu-aqn-test |

## Conclusion

### Summary Table

| Component | GPU | NPU |
|-----------|-----|-----|
| verl R3 code | ✅ Ready | ✅ Ready |
| vLLM routing capture | ❌ PR not merged | ❌ PR not merged |
| vllm-ascend support | N/A | ❌ Needs implementation |
| **R3 Working** | ❌ No | ❌ No |
| **R2 Working** | ✅ Yes | ⚠️ Megatron import issues |

### Recommended Next Steps

1. **Short-term**: Use R2 mode for MoE training on GPU
2. **Medium-term**:
   - Monitor PR #28284 merge status
   - Prepare vllm-ascend patches for when PR merges
3. **Long-term**: Full R3 support on NPU requires coordinated vLLM + vllm-ascend updates

## References

- vLLM PR #28284: https://github.com/vllm-project/vllm/pull/28284
- verl Router Replay: `examples/router_replay/README.md`
- verl R3 Implementation: `verl/utils/megatron/router_replay_patch.py`
- Previous Assessment: `docs/qerl/R3_AQN_NPU_REALISTIC_ASSESSMENT.md`
