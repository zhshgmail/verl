# MXFP4 + AQN Next Steps Plan

**Date**: 2026-01-09
**Last Updated**: 2026-01-09 (Exp 1D & 1E Complete)
**Status**: MXFP4 experiments complete, proceeding to NVFP4
**Reference**: Gemini Expert Review

---

## 0. Quick Start - Execution Commands

```bash
# SSH to A100 server
ssh root@90.90.102.18
docker exec -it verl-r3-test bash
cd /home/z00637938/workspace/verl

# Pull latest code (includes NVFP4 support!)
git pull personal feature/npu-aqn-test

# Clean up
pkill -f "ray|vllm" || true

# Run experiments in order:

# 1. MXFP4 Exp 1D (ultra-small sigma on Linear)
bash scripts/test_mxfp4_exp1d_tiny_sigma.sh 8

# 2. MXFP4 Exp 1E (RMSNorm targeting - QeRL default)
bash scripts/test_mxfp4_exp1e_rmsnorm.sh 8

# 3. NVFP4 Experiment (21x lower error, should work!)
bash scripts/test_nvfp4_w4a16_training.sh 8
```

---

## 1. Gemini Review Summary

### 1.1 Key Validations

| Aspect | Status | Notes |
|--------|--------|-------|
| **SRDD Validity** | Validated | Mathematically valid for diagnosing layer-wise quantization impact |
| **W4A16 Implementation** | Correct | `injection_point='weight'` correctly quantizes weights, not activations |
| **AQN on Linear** | Problematic | Causes model collapse - Linear layers too sensitive |
| **AQN on RMSNorm** | Recommended | QeRL's "safe" approach - indirect robustness via normalization |

### 1.2 Technical Insights

1. **Gain Scan** = local gradient measurement
2. **Deadzone** = gain near 0 (small values lost)
3. **Saturation** = min gain drop (large values clipped)
4. **W4A16 vs W16A4**: W4A16 (weights quantized) is correct for QeRL-style training

---

## 2. Experiment Status

> **⚠️ KNOWN ISSUE**: All MXFP4/NVFP4 experiments below have `lm_head` quantized (should be excluded).
> This bug was identified by Gemini review on 2026-01-09 and fixed in commit `d8c25492`.
> Production PTQ recipes exclude `lm_head` and `embed_tokens` from quantization.
> Results may improve when re-run with the fix (`exclude_modules=['lm_head', 'embed_tokens']`).

| Experiment | Config | Result | Status | lm_head Bug |
|------------|--------|--------|--------|-------------|
| Baseline | No MXFP4, No AQN | 75.97% | Completed | N/A |
| MXFP4-only | W4A16, No AQN | 70.05% | Completed | **YES** |
| MXFP4+AQN (original) | W4A16, AQN on RMSNorm | 67.48% | Completed (AQN hurt!) | **YES** |
| **1A** | W4A16, AQN on Linear (sigma=0.05) | **Collapsed at step 19** | Failed | **YES** |
| **1C** | W4A16, AQN on Linear (sigma=0.005) | **Partial collapse at step 19** | Failed | **YES** |
| **1D** | W4A16, AQN on Linear (sigma=0.001) | **66.49% (STABLE!)** | Completed | **YES** |
| **1E** | W4A16, AQN on RMSNorm (QeRL default) | **62.32% (WORST!)** | Completed | **YES** |

### 2.1 Experiment 1D Results (2026-01-09)

**BREAKTHROUGH: sigma=0.001 is the stability threshold for Linear layer AQN!**

| Metric | Value | Notes |
|--------|-------|-------|
| Final Accuracy | **66.49%** | Lower than MXFP4-only (70.05%) |
| Training Stability | **STABLE** | No collapse (entropy 0.05-0.3 throughout) |
| Runtime | 3 hours | 174 steps, 3 epochs |
| Sigma Range | 0.001 → 0.00001 | 10 stages |
| Checkpoint | global_step_174 | Saved successfully |

**Key Findings:**
1. **Stability threshold found**: sigma=0.001 is the minimum for stable Linear layer AQN
2. **No accuracy improvement**: 66.49% < 70.05% (MXFP4-only) < 75.97% (baseline)
3. **Ultra-small sigma = effectively no AQN benefit**: The noise is too weak to improve robustness
4. **Confirms Gemini's analysis**: Linear layers are too sensitive; RMSNorm is the safer target

**Validation Progression:**
- Step 0: 75.97% (pre-training baseline)
- Step 20: 66.21%
- Step 40: 73.16%
- Step 60: 66.51%
- Step 80: 69.82%
- Step 100: 66.67%
- Step 120: 66.06%
- Step 140: 66.59%
- Step 160: 67.02%
- Step 174: **66.49%** (final)

### 2.2 Experiment 1E Results (2026-01-09)

**CONCLUSION: QeRL's RMSNorm AQN does NOT help with MXFP4's high error rate!**

| Metric | Value | Notes |
|--------|-------|-------|
| Final Accuracy | **62.32%** | WORST of all experiments! |
| Training Stability | STABLE | No collapse (entropy 0.073 at end) |
| Runtime | 2h 41m | 174 steps, 3 epochs |
| Sigma Range | 0.05 → 0.0005 | QeRL default schedule |
| Checkpoint | global_step_174 | Saved successfully |

**Key Findings:**
1. **RMSNorm AQN hurts MXFP4 training**: 62.32% < 66.49% (Exp 1D) < 70.05% (MXFP4-only)
2. **~21% MXFP4 error is too high**: AQN cannot compensate for such large quantization error
3. **QeRL's approach needs lower error format**: NVFP4 (~1% error) is the logical next step
4. **All MXFP4+AQN variants underperform MXFP4-only**: AQN adds training noise without benefit

**Accuracy Ranking (worst to best):**
1. Exp 1E (RMSNorm AQN): **62.32%** ← Worst
2. Exp 1D (Linear AQN sigma=0.001): **66.49%**
3. Original MXFP4+AQN: **67.48%**
4. MXFP4-only: **70.05%** ← Best with MXFP4
5. Baseline (no quant): **75.97%**

**Next Action**: Try NVFP4 (21x lower error than MXFP4). With ~1% relative error, AQN should be effective.

---

## 3. Next Steps Plan

### Phase 1: Final Linear Layer Test (Experiment 1D)

**Goal**: Determine if any direct noise on Linear layers is tolerable.

**Configuration**:
```yaml
trainer.noise_injection:
  enabled: true
  sigma_start: 0.001      # 50x smaller than original
  sigma_end: 0.00001
  num_stages: 10
  layer_types: ['linear']
```

**Success Criteria**:
- No collapse (entropy < 1.0 at step 19+)
- Score maintains > 60%

**Expected Outcome**: Likely still problematic based on 1A/1C trends.

### Phase 2: RMSNorm Targeting (Experiment 1E)

**Goal**: Validate QeRL's RMSNorm approach with MXFP4 W4A16.

**Configuration**:
```yaml
trainer.noise_injection:
  enabled: true
  sigma_start: 0.05       # QeRL default
  sigma_end: 0.0005
  num_stages: 10
  layer_types: ['rmsnorm']  # Safe target
```

**Success Criteria**:
- Stable training (no collapse)
- Final accuracy >= 70% (better than MXFP4-only 70.05%)

**Expected Outcome**: Should be stable, question is whether it improves robustness.

### Phase 3: Analysis and Comparison

**Goal**: Compare SRDD metrics between checkpoints.

**Tasks**:
1. Run SRDD scan on 1E checkpoint (if successful)
2. Compare with MXFP4-only checkpoint
3. Measure actual quantization robustness improvement

---

## 4. Execution Commands

```bash
# SSH to A100 server
ssh root@90.90.102.18

# Enter docker container
docker exec -it verl-r3-test bash
cd /home/z00637938/workspace/verl

# Pull latest code
git pull personal feature/npu-aqn-test

# Clean up
pkill -f "ray|vllm" || true

# Phase 1: Experiment 1D (ultra-small sigma on Linear)
bash scripts/test_mxfp4_exp1d_tiny_sigma.sh 8

# Wait for completion, check results
# If collapsed: proceed to Phase 2
# If stable: analyze and document

# Phase 2: Experiment 1E (RMSNorm targeting)
bash scripts/test_mxfp4_exp1e_rmsnorm.sh 8

# Phase 3: SRDD comparison (after 1E completes)
python scripts/srdd_quant_scanner.py \
    --model_path /tmp/mxfp4_exp1e_rmsnorm/checkpoints/global_step_XXX \
    --quant_type mxfp4 \
    --output results_exp1e_srdd.json
```

---

## 5. Decision Tree

```
Start
  │
  ▼
Run Experiment 1D (sigma=0.001 on Linear)
  │
  ├─ Collapsed? ─────────────────────────────────────────┐
  │      │                                                │
  │      ▼                                                │
  │   CONCLUSION: Linear layers cannot tolerate          │
  │   ANY noise injection. AQN must target RMSNorm.      │
  │      │                                                │
  │      ▼                                                │
  │   Run Experiment 1E (RMSNorm)                        │
  │      │                                                │
  │      ├─ Success (≥70%)? ─────────────┐               │
  │      │      │                         │               │
  │      │      ▼                         │               │
  │      │   QeRL approach validated     │               │
  │      │   Document findings            │               │
  │      │                                │               │
  │      └─ Failed (<70%)? ──────────────┤               │
  │             │                         │               │
  │             ▼                         │               │
  │          Investigate:                 │               │
  │          - Sigma tuning               │               │
  │          - More epochs                │               │
  │          - MXFP4 error too high?     │               │
  │                                       │               │
  └─ Stable? ─────────────────────────────┘               │
         │                                                │
         ▼                                                │
      BREAKTHROUGH: Small noise on Linear               │
      is tolerable. Document sigma threshold.           │
         │                                                │
         ▼                                                │
      Test higher sigma (0.005, 0.01) to find            │
      optimal trade-off                                   │
```

---

## 6. Key Insights from Experiments

### 6.1 Why Linear Layer Noise Causes Collapse

Linear layer weights are the model's **learned knowledge**:
- `q_proj, k_proj, v_proj, o_proj`: Attention patterns
- `gate_proj, up_proj, down_proj`: FFN transformations
- `lm_head`: Token embeddings

Adding noise directly corrupts these representations, causing:
1. **Attention breakdown**: Wrong attention patterns
2. **FFN corruption**: Garbled intermediate representations
3. **Output collapse**: Random token predictions (high entropy)

### 6.2 Why RMSNorm Noise is Safe

RMSNorm only affects **scale/variance**, not learned representations:
- Noise in RMSNorm creates "perturbation field"
- Model learns to be robust to scale variations
- Indirectly improves tolerance to Linear layer quantization
- Similar to Dropout (noise in intermediate layers → generalization)

### 6.3 The Indirect Robustness Mechanism

```
Training with RMSNorm noise:
  Input → [Linear + MXFP4] → [RMSNorm + noise] → [Linear + MXFP4] → ...
                                    ↑
                            Model learns to handle
                            scale variations here
                                    ↓
                            Becomes more tolerant to
                            quantization errors everywhere
```

---

## 7. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| 1D still collapses | High (80%) | Low | Already have 1E ready |
| 1E doesn't improve | Medium (40%) | Medium | Try sigma tuning, more epochs |
| MXFP4 error too high | Low (20%) | High | **Try NVFP4 instead** |

---

## 7.1 NVFP4 Fallback Plan

If MXFP4 experiments don't yield good results, **NVFP4 is the recommended fallback**:

### Why NVFP4?

| Metric | MXFP4 | NVFP4 | Improvement |
|--------|-------|-------|-------------|
| **Relative Error** | ~21% | ~1% | **21x better** |
| **Block Size** | 32 elements | 16 elements | Finer granularity |
| **Scale Format** | E8M0 | E4M3 | Higher precision |
| **QeRL Compatibility** | Untested | Proven | QeRL's original format |

### NVFP4 Implementation Options

1. **Use quant_compute library** (if available on A100):
   ```python
   from quant_cy_npu import QType
   qtype = QType('nvfp4')  # NVIDIA FP4 format
   ```

2. **Implement standalone NVFP4** (like mxfp4_quant.py):
   - E4M3 format for scales (vs E8M0)
   - 16-element blocks (vs 32)
   - Should be straightforward adaptation

### When to Switch to NVFP4

| Condition | Action |
|-----------|--------|
| Exp 1E accuracy < 70% | Consider NVFP4 |
| SRDD shows no improvement | Switch to NVFP4 |
| User explicitly requests | Implement NVFP4 |

### Expected NVFP4 Results

With 21x lower quantization error:
- AQN sigma=0.05 should be appropriate (not 21x too weak)
- Model should be able to adapt during training
- Final accuracy target: **74-76%** (near baseline)

---

## 8. Timeline (Estimated)

| Phase | Task | Duration |
|-------|------|----------|
| 1 | Run Experiment 1D | ~2 hours |
| 2 | Analyze 1D results | 30 min |
| 3 | Run Experiment 1E | ~2 hours |
| 4 | Analyze 1E results | 30 min |
| 5 | SRDD comparison | 1 hour |
| 6 | Document findings | 30 min |

**Total**: ~6-7 hours

---

## 9. Success Metrics

| Metric | Baseline | Target | Stretch |
|--------|----------|--------|---------|
| Final Accuracy | 70.05% (MXFP4-only) | ≥70% | ≥72% |
| Training Stability | N/A | No collapse | Smooth curve |
| Entropy (step 19+) | N/A | <1.0 | <0.5 |
| SRDD Improvement | N/A | Measurable | >5% SQNR |

---

## 10. References

- [QeRL Paper](https://arxiv.org/abs/XXXX) - Original AQN on LayerNorm approach
- [SRDD Documentation](docs/qerl/SRDD_MXFP4_QUANT_EXPERIMENT.md) - Full experiment log
- [Gemini Review](conversation) - Expert validation of approach
