# E12-Aggressive Experiment Plan

**Date**: 2026-01-13
**Branch**: `feature/npu-aqn-test`

## Background

E12 experiments use high sigma AQN (0.05 start) with MXFP4 + LoRA. Current results:

| Experiment | Sigma Config | Epochs | Result | Notes |
|------------|--------------|--------|--------|-------|
| E12 (1ep) | 0.05 → 0.0005 | 1 | 72.48% | Best LoRA result |
| E12-2ep | 0.05 → 0.0005 | 2 | ~70% | Declined from peak (epoch_aware warmup issue?) |
| E12-2ep-v2 | 0.05 → 0.0005 | 2 | TBD | Testing epoch_aware=False |

## Hypothesis

Current E12 uses 100x decay ratio (0.05 → 0.0005). The ending sigma of 0.0005 still injects noticeable noise during final training steps.

**Question**: Should we keep ending sigma as small as possible for cleaner convergence?

**Proposed**: Use aggressive 5000x decay (0.05 → 0.00001) to allow:
- Strong exploration at start (σ=0.05)
- Clean convergence at end (σ≈0, negligible noise)

## Experiment: E12-aggressive-1ep

### Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base | E12 config | MXFP4 + LoRA + AQN |
| sigma_start | 0.05 | Same as E12 (high) |
| sigma_end | 0.00001 | 50x smaller than E12 |
| Decay ratio | 5000x | vs 100x in E12 |
| Epochs | 1 | Quick test first |
| epoch_aware | False | Step-based decay |
| num_stages | 10 | K-stage exponential |
| Steps | ~29 | 7473 / 256 batch |

### Expected Sigma Timeline (1 epoch, 29 steps)

With 10 stages + warmup over 29 steps:
- Steps 0-2: σ=0 (warmup, ~10%)
- Steps 3-5: σ≈0.05 (stage 1)
- Steps 6-8: σ≈0.017 (stage 2)
- Steps 9-11: σ≈0.006 (stage 3)
- Steps 12-14: σ≈0.002 (stage 4)
- Steps 15-17: σ≈0.0007 (stage 5)
- Steps 18-20: σ≈0.0002 (stage 6)
- Steps 21-23: σ≈0.00008 (stage 7)
- Steps 24-26: σ≈0.00003 (stage 8)
- Steps 27-29: σ≈0.00001 (stage 9)

By step ~18, sigma is already below E12's ending value (0.0005), giving more steps for clean fine-tuning.

### Success Criteria

- If E12-aggressive-1ep > 72.48%: Aggressive decay helps
- If E12-aggressive-1ep ≈ 72.48%: No significant difference
- If E12-aggressive-1ep < 72.48%: Aggressive decay hurts (unlikely)

## Execution Order

1. **E12-2ep-v2** (currently running): Test epoch_aware=False with 2 epochs
2. **E12-aggressive-1ep** (next): Test aggressive sigma decay with 1 epoch
3. **E12-aggressive-2ep** (optional): If 1ep shows promise, try 2 epochs

## Files

- Script: `scripts/test_mxfp4_e12_aggressive_1ep.sh`
- Log: `/tmp/mxfp4_e12_aggressive_1ep/training.log`
- Checkpoint: `/tmp/mxfp4_e12_aggressive_1ep/checkpoints/`
