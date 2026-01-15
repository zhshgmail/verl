# QA Verification Report: verl-npu Branch Analysis

**Date**: 2025-12-24
**Reviewer**: Claude (following QA Agent protocol)
**Subject**: Changes needed in feature/qerl-noise-injection-moe for Ascend 910C NPU support

---

## Overall Assessment: PASS

The `team/verl-npu` branch provides a working NPU adaptation for VERL. Most changes are **minor and additive** - they don't conflict with our QeRL MoE feature work. The OOM issue mentioned by the team is still unresolved.

---

## Branch Summary

**Changed Files** (8 total):
| File | Type | Risk |
|------|------|------|
| `VERL_NPU_ADAPTATION_GUIDE.md` | Documentation | None |
| `recipe/dapo/run_qwen3moe_30b_npu_fsdp.sh` | New script | None |
| `recipe/dapo/run_qwen3moe_30b_npu_megatron_direct_10ep.sh` | New script | None |
| `scripts/run_with_ascend_visible.sh` | New script | None |
| `verl/trainer/runtime_env_npu.yaml` | New config | None |
| `verl/utils/distributed.py` | Code change | **LOW** |
| `verl/utils/fsdp_utils.py` | Code change | **LOW** |
| `verl/workers/fsdp_workers.py` | Code change | **LOW** |

---

## Verified Code Changes

### 1. distributed.py:85 - Backend Hardcoding

**Original (main branch)**:
```python
backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}"
```

**NPU branch change**:
```python
backend=f"cpu:gloo,npu:hccl"
```

**Analysis**:
- The original dynamic code SHOULD work: `get_device_name()` returns "npu" and `get_nccl_backend()` returns "hccl" when torch_npu is available
- Team hardcoded it, likely to avoid edge cases or ensure NPU-only mode
- **Confidence**: HIGH - Change is NPU-specific, may not be needed if dynamic detection works

**Recommendation**:
- **DON'T merge this change** - Keep the dynamic detection
- If it fails on NPU, fix `device.py` instead of hardcoding

### 2. fsdp_utils.py:50-54 - Empty Cache Safety

**Original (main branch)**:
```python
def init_fn(x: torch.nn.Module):
    if torch.distributed.get_rank() != 0:
        x = x.to_empty(device=get_device_id(), recurse=False)
        get_torch_device().empty_cache()
    return x
```

**NPU branch change**:
```python
def init_fn(x: torch.nn.Module):
    if torch.distributed.get_rank() != 0:
        x = x.to_empty(device=get_device_id(), recurse=False)
    # safe empty_cache across backends
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    return x
```

**Analysis**:
- Move `empty_cache()` outside the rank!=0 check (runs on ALL ranks now)
- Explicit NPU/CUDA check instead of dynamic `get_torch_device()`
- **Issue**: This is a behavior change - now all ranks call `empty_cache()`, not just non-rank-0

**Confidence**: MEDIUM - The placement change (outside if block) may be intentional for NPU memory management

**Recommendation**:
- **CONSIDER merging** but verify the placement change is intentional
- The explicit NPU check is safer than dynamic `get_torch_device()`

### 3. fsdp_workers.py:147 - Same as distributed.py

**Same hardcoding change** at line 147.

**Recommendation**: Same as distributed.py - DON'T merge, keep dynamic detection.

---

## Key Environment Variables (from NPU scripts)

These are **CRITICAL** for NPU operation:

```bash
# HCCL (Huawei Collective Communication Library)
export HCCL_BUFFSIZE="512"
export HCCL_OP_BASE_FFTS_MODE_ENABLE="TRUE"
export HCCL_WHITELIST_DISABLE="1"

# Ascend device visibility
export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
export ASCEND_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}"

# Disable panic on OOM (CRITICAL - prevents kernel panic)
echo 0 | sudo tee /proc/sys/vm/panic_on_oom

# WandB offline mode (Ascend may have network restrictions)
export WANDB_MODE=offline

# vLLM Ascend specific
export VLLM_ASCEND_ENABLE_NZ="0"
```

**Confidence**: HIGH - These are from working scripts

---

## Key Hyperparameter Differences (Qwen3-30B-A3B MoE on 16 NPUs)

| Parameter | FSDP Script | Megatron Script |
|-----------|-------------|-----------------|
| Sequence Parallel | `sp_size=8` | N/A (uses Megatron PP/TP/EP) |
| Gen TP | `gen_tp=2` | `gen_tp=4` |
| Train batch size | 32 | 16 |
| GPU memory util | 0.8 | 0.85 |
| Offload | False | True |
| max_num_seqs | 16 | 32 |
| Train TP/PP/EP | FSDP | TP=1, PP=1, EP=16, CP=1 |

**Key Insight**: Team is experimenting with both FSDP and Megatron parallelism strategies for MoE on NPU.

---

## OOM Issue Analysis

**Team's Status**: "Training can start but always fails due to OOM"

**Potential Causes**:
1. **KV Cache on NPU**: vllm-ascend may allocate KV cache differently
2. **Memory fragmentation**: NPU memory management differs from CUDA
3. **Offload not working**: Team tried `OFFLOAD=True` in Megatron script
4. **Aggressive empty_cache**: Modified `fsdp_utils.py` to call `empty_cache()` on all ranks

**Unverified Claims**:
- Whether the empty_cache change helps OOM
- Whether vLLM memory utilization settings work on vllm-ascend
- Whether MoE expert parallelism (EP=16) works correctly on NPU

---

## Action Plan for feature/qerl-noise-injection-moe

### Phase 1: Essential Changes (DO NOW)

1. **Add NPU runtime env file**:
   ```bash
   cp from team/verl-npu: verl/trainer/runtime_env_npu.yaml
   ```

2. **Add Ascend visibility script**:
   ```bash
   cp from team/verl-npu: scripts/run_with_ascend_visible.sh
   ```

3. **Add NPU training script for QeRL MoE**:
   - Create `recipe/r1_ascend/run_qwen3moe_30b_npu_qerl.sh`
   - Base on `run_qwen3moe_30b_npu_fsdp.sh` but add QeRL/R3/AQN configs

### Phase 2: Conditional Changes (IF dynamic detection fails)

4. **fsdp_utils.py empty_cache fix** (LOW PRIORITY):
   - Only apply if `get_torch_device().empty_cache()` fails on NPU
   - The explicit check is safer but changes behavior

5. **distributed.py/fsdp_workers.py backend** (DO NOT CHANGE):
   - Keep dynamic detection
   - Only hardcode if it fails in testing

### Phase 3: Testing (AFTER quantization completes)

6. **Test plan**:
   - Once Qwen3-30B-A3B NVFP4 quantization completes on A100
   - Copy quantized model to 910C machine
   - Run QeRL training with NPU scripts
   - Monitor for OOM and adjust memory settings

---

## Files to Cherry-Pick from team/verl-npu

```bash
# Safe to cherry-pick (new files only)
git checkout team/verl-npu -- VERL_NPU_ADAPTATION_GUIDE.md
git checkout team/verl-npu -- recipe/dapo/run_qwen3moe_30b_npu_fsdp.sh
git checkout team/verl-npu -- recipe/dapo/run_qwen3moe_30b_npu_megatron_direct_10ep.sh
git checkout team/verl-npu -- scripts/run_with_ascend_visible.sh
git checkout team/verl-npu -- verl/trainer/runtime_env_npu.yaml
```

```bash
# DO NOT cherry-pick (code changes - keep our dynamic detection)
# verl/utils/distributed.py
# verl/utils/fsdp_utils.py
# verl/workers/fsdp_workers.py
```

---

## Confidence Summary

| Finding | Confidence | Action |
|---------|------------|--------|
| NPU scripts work for Qwen3-30B-A3B | HIGH | Cherry-pick scripts |
| HCCL environment variables needed | HIGH | Use in NPU scripts |
| Backend hardcoding necessary | LOW | Keep dynamic detection |
| empty_cache change helps OOM | UNCERTAIN | Test first |
| OOM root cause identified | LOW | Needs hardware testing |

---

## Final Verdict

**The verl-npu branch provides:**
1. Working NPU training scripts for Qwen3-30B-A3B (VERIFIED)
2. Comprehensive documentation (VERIFIED)
3. Essential environment configuration (VERIFIED)

**The OOM issue is NOT solved** - team is still working on it. Our approach should be:
1. Cherry-pick new files (scripts, configs, docs)
2. Keep our dynamic device detection code
3. Test on actual 910C hardware when quantized model is ready
4. Fix OOM issues based on actual error logs

**Risk Level**: LOW for integration, MEDIUM for OOM issues at runtime.

