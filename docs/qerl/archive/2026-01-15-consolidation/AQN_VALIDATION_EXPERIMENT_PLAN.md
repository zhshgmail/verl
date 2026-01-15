# AQN Validation Experiment Plan

**Version**: 1.0
**Date**: 2026-01-12
**Branch**: `feature/npu-aqn-test`
**Status**: Planned

---

## Executive Summary

This document outlines critical validation experiments identified through expert review and Gemini critique. The experiments address methodology gaps and missing tests needed to support our AQN conclusions.

---

## 1. Critical Validation Gaps

### 1.1 Confounded Comparison (E5b vs E9a)

**Problem**: E5b (high σ=0.05, all layers) vs E9a (low σ=0.01, targeted) differs in TWO variables:
- Sigma magnitude
- Layer targeting

**Solution**: Run E9a with HIGH sigma to isolate layer targeting effect.

### 1.2 Missing MXFP4 + LoRA + AQN Test

**Problem**: We recommend "AQN CRITICAL for LoRA + any quantization" but haven't tested:
- MXFP4 + LoRA + AQN (NPU deployment target)
- Only tested NVFP4 + LoRA + AQN (+2.27%)

**Solution**: Run E12 with MXFP4 + LoRA + high-sigma AQN.

### 1.3 Statistical Significance

**Problem**: +2.42% improvement is within single-run variance (95% CI ≈ ±2.47%).

**Solution**: Replication runs are needed but lower priority than filling test matrix gaps.

---

## 2. Experiment Matrix (Current vs Needed)

### Current Coverage:
| Config | Full FT | LoRA |
|--------|---------|------|
| BF16 | - | E7a: 71.27% |
| NVFP4 + AQN (low σ) | E4b: 72.63% | E5b-LoRA: 66.11% (+2.27%) |
| MXFP4 + AQN (low σ) | E3b: 74.37% | **MISSING** |
| 5% HW + AQN (high σ) | E5b: 70.58% | **MISSING** |
| 5% HW + SRDD (low σ) | E9a: 68.54% | - |

### Needed Experiments:
| ID | Config | Purpose | Priority |
|----|--------|---------|----------|
| **E12** | MXFP4 + LoRA + AQN (high σ) | Validate NPU deployment recommendation | **P0** |
| **E9a-high-σ** | 5% HW + SRDD-targeted (high σ) | Isolate layer targeting effect | **P0** |
| E6b-high-σ | MXFP4 + LoRA + AQN (high σ) | Compare with E12 baseline | P1 |

---

## 3. Experiment Specifications

### 3.1 E12: MXFP4 + LoRA + High-Sigma AQN (NPU Target)

**Objective**: Validate that LoRA + AQN benefit transfers to MXFP4 (NPU deployment target)

**Configuration**:
```yaml
quantization:
  type: mxfp4
  target: weight
  exclude: [lm_head, embed_tokens, lora_A, lora_B]

lora:
  rank: 32
  alpha: 16

aqn:
  enabled: true
  sigma_start: 0.05      # HIGH sigma (E5b finding)
  sigma_end: 0.0005
  num_stages: 10
  layer_types: [rmsnorm]
  epoch_aware: true

algorithm: dapo
epochs: 1
```

**Hypothesis**:
- E12 should show +2-3% AQN benefit (similar to NVFP4 LoRA +2.27%)
- If MXFP4 has higher error, AQN benefit may be even larger

**Script**: `scripts/test_mxfp4_v6.2_dapo_lora_aqn_high_sigma.sh`

---

### 3.2 E9a-high-σ: SRDD-Targeted with High Sigma

**Objective**: Isolate layer targeting effect by matching E5b's sigma magnitude

**Configuration**:
```yaml
noisy_ops:
  enabled: true
  error_scale: 0.05
  error_type: relative_gaussian

aqn:
  enabled: true
  epoch_aware: true
  sigma_start: 0.05      # Match E5b
  sigma_end: 0.0005      # Match E5b
  stages_per_epoch: 5
  layer_sigma_config:
    enabled: true
    default_multiplier: 0.0   # No AQN to other layers
    layer_multipliers:
      14: 1.0
      15: 1.0
      16: 1.0
      17: 1.0

epochs: 2
```

**Hypothesis**:
- If E9a-high-σ ≈ E5b (70.58%): Layer targeting provides no additional benefit
- If E9a-high-σ > E5b: Layer targeting IS beneficial
- If E9a-high-σ < E5b: All-layer AQN is better than targeted

**Script**: `scripts/test_noisy_ops_srdd_targeted_high_sigma.sh`

---

## 4. Expected Outcomes

### Success Criteria:

| Experiment | Expected Result | Success Definition |
|------------|-----------------|-------------------|
| E12 | ~68-70% (vs ~65% baseline) | AQN benefit +2-3% for MXFP4 LoRA |
| E9a-high-σ | Compare with E5b (70.58%) | Isolates layer targeting effect |

### Decision Matrix:

| E12 Result | E9a-high-σ Result | Conclusion |
|------------|-------------------|------------|
| +2-3% benefit | ≈ E5b | Recommendations validated: high σ, all layers |
| +2-3% benefit | > E5b | Update: targeting + high σ is optimal |
| +2-3% benefit | < E5b | Confirmed: all-layer AQN preferred |
| No benefit | Any | Revise MXFP4 LoRA recommendation |

---

## 5. Execution Plan

### Phase 1: Immediate (Today)
1. ✅ Create experiment plan document
2. ✅ Create E12 script (MXFP4 + LoRA + high-σ AQN)
3. ✅ Create E9a-high-σ script (SRDD-targeted + high-σ)
4. Wait for E9b to complete (~40 min remaining)
5. Start E12 on A100

### Phase 2: Sequential
6. After E12: Start E9a-high-σ
7. Document results in consolidated findings

### Phase 3: Follow-up (If Needed)
8. Replication runs for statistical validation
9. LoRA rank ablation (E10c) to validate gradient hypothesis

---

## 6. Resource Requirements

| Experiment | GPUs | Est. Time | Node |
|------------|------|-----------|------|
| E12 | 8x A100 | ~2h | 90.90.102.18 |
| E9a-high-σ | 8x A100 | ~2h | 90.90.102.18 |

**Total**: ~4h sequential on single A100 node

---

## 7. Questions This Plan Addresses

From Gemini critique:

| Question | Experiment | Answer Strategy |
|----------|------------|-----------------|
| Q1: Is E5b vs E9a comparison fair? | E9a-high-σ | Isolate layer targeting by matching sigma |
| Q8: Should we test MXFP4 + LoRA + AQN? | E12 | Direct test of NPU deployment config |
| Q11: Is recommendation too aggressive? | E12 | Validate with actual test data |

---

## 8. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| E12 shows no AQN benefit | High - invalidates recommendation | Re-examine sigma parameters |
| E9a-high-σ < E5b | Medium - layer targeting not useful | Still validates high sigma preference |
| A100 node unavailable | Medium - delays validation | Can queue for later execution |

---

*Document created: 2026-01-12*
*Author: Claude Code + Expert Review*
