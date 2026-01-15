# SRDD Co-Library Validation Session Handoff

**Date**: 2026-01-11
**Status**: IN PROGRESS
**Branch**: `feature/npu-aqn-test`

---

## Current Task

Validating our SRDD quantization implementations against llm-compressor reference library.

## What Was Done

### 1. llm-compressor Setup
- Installed llm-compressor from `/home/z00637938/workspace/llm-compressor` on A100
- Version: 0.9.1.dev34
- Available schemes: MXFP4, NVFP4A16

### 2. Created Comparison Script
- `scripts/srdd_colibrary_comparison.py` - compares verl vs llm-compressor
- Committed and pushed to personal remote

### 3. Initial verl-only Results (CRITICAL FINDING!)
```
verl MXFP4: 21.77% relative error
verl NVFP4: 26.51% relative error (HIGHER than MXFP4!)
```

This explains why NVFP4+LoRA (63.84%) performed WORSE than MXFP4+LoRA (65.88%).

## What Needs To Be Done

### NEXT STEP: Run llm-compressor comparison on A100

```bash
# SSH to A100
ssh root@90.90.102.18
docker exec -it verl-r3-test bash

# Pull latest code
source /home/z00637938/setup_proxy.sh
cd /home/z00637938/workspace/verl
git pull personal feature/npu-aqn-test

# Run full comparison (with llm-compressor)
python scripts/srdd_colibrary_comparison.py \
    --model_path /data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
    --device cuda \
    --output_dir /tmp/srdd_colibrary_results
```

### Expected Analysis

1. If llm-compressor MXFP4 ≈ verl MXFP4 (~21%): Our implementation is correct
2. If llm-compressor NVFP4 << verl NVFP4: Our NVFP4 implementation has bugs
3. If llm-compressor NVFP4 ≈ verl NVFP4 (~26%): The "~1% error" claim is wrong

### After Comparison

1. Document findings in `docs/qerl/SRDD_COLIBRARY_VALIDATION.md`
2. If NVFP4 implementation is buggy, fix it
3. Update SRDD scanner to use validated implementations
4. Re-run experiments if needed

## Key Files

| File | Purpose |
|------|---------|
| `scripts/srdd_colibrary_comparison.py` | Comparison test script |
| `verl/utils/mxfp4_quant.py` | verl MXFP4 implementation |
| `verl/utils/nvfp4_quant.py` | verl NVFP4 implementation (may have bugs) |
| `scripts/srdd_quant_scanner.py` | SRDD scanner |
| `docs/qerl/LORA_EXPERIMENT_RESULTS_20260111.md` | Experiment results |
| `docs/qerl/A100_QUICK_REFERENCE.md` | A100 connection/proxy info |

## Context for User's Questions

User asked about:
1. Using SRDD scan results to guide AQN (apply more/less noise to high/low error layers)
2. Testing if NVFP4 and MXFP4 show different SRDD results
3. The fundamental validation of our quantization simulation

The SRDD co-library validation is FOUNDATIONAL work - all our experiments depend on accurate quantization simulation.

---

## Prompt for New Agent

```
Continue the SRDD co-library validation work from the previous session.

CONTEXT:
- Branch: feature/npu-aqn-test
- Previous session created scripts/srdd_colibrary_comparison.py
- verl-only test showed NVFP4 (26.51% error) is WORSE than MXFP4 (21.77%)
- Need to validate against llm-compressor reference implementation

IMMEDIATE TASK:
1. SSH to A100: ssh root@90.90.102.18, docker exec -it verl-r3-test bash
2. Source proxy: source /home/z00637938/setup_proxy.sh
3. Pull code: cd /home/z00637938/workspace/verl && git pull personal feature/npu-aqn-test
4. Run full test: python scripts/srdd_colibrary_comparison.py --model_path /data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --device cuda

ANALYZE:
- Compare llm-compressor MXFP4/NVFP4 vs verl implementations
- If verl NVFP4 is buggy, investigate/fix verl/utils/nvfp4_quant.py
- Document findings in docs/qerl/SRDD_COLIBRARY_VALIDATION.md

KEY DOCS:
- docs/qerl/SRDD_COLIBRARY_SESSION_HANDOFF.md (this file)
- docs/qerl/A100_QUICK_REFERENCE.md (connection/proxy info)
- docs/qerl/LORA_EXPERIMENT_RESULTS_20260111.md (experiment context)
```
