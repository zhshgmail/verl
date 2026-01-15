# RIN Experiment Tracker: Live Status and Analysis

**Last Updated**: 2026-01-15
**Current Phase**: Phase 1 - Foundation
**Next Experiment**: E13i-baseline

---

## Quick Status Dashboard

| Experiment | Status | Accuracy | vs E13h | Hypothesis | Outcome |
|------------|--------|----------|---------|------------|---------|
| **E13h** (baseline) | ✅ Complete | 56.41% | - | No RIN | Baseline |
| **E13i-baseline** | 📋 Planned | TBD | TBD | H1a Global | - |
| **E13i-targeted** | 📋 Planned | TBD | TBD | H1b Targeted | - |
| E13i-high | ⏸️ Pending | - | - | H2a Escape | - |
| E13i-low | ⏸️ Pending | - | - | H2b Sufficient | - |
| E13j-variable | ⏸️ Pending | - | - | H3a Variable | - |

**Legend**: ✅ Complete | 🏃 Running | 📋 Planned | ⏸️ Pending | ❌ Cancelled

---

## Hypothesis Status Tracker

### H1: Global vs Targeted RIN

| Hypothesis | Confidence | Status | Evidence |
|------------|------------|--------|----------|
| H1a: Global RIN better | 70% | 🔬 Testing | Awaiting E13i-baseline |
| H1b: Targeted RIN better | 60% | 🔬 Testing | Awaiting E13i-targeted |

**Decision Point**: After E13i-baseline and E13i-targeted complete

---

### H2: Sigma Magnitude for High-Error Zones

| Hypothesis | Confidence | Status | Evidence |
|------------|------------|--------|----------|
| H2a: Higher sigma (escape) | 55% | ⏸️ Pending | Need Phase 1 complete |
| H2b: Lower sigma (sufficient) | 45% | ⏸️ Pending | Need Phase 1 complete |

**Decision Point**: After Phase 1 winner identified

---

### H3: Variable vs Constant Sigma

| Hypothesis | Confidence | Status | Evidence |
|------------|------------|--------|----------|
| H3a: Variable sigma better | 65% | ⏸️ Pending | Need Phase 2 insights |
| H3b: Constant sigma better | 35% | ⏸️ Pending | Need Phase 2 insights |

**Decision Point**: After optimal sigma range identified

---

## SRDD Correlation Analysis

### Baseline SRDD Data (E13h)

**Mean Error by Zone**:
- First (0-3): 32.5% error, 18.9% deadzone
- Early-Mid (4-9): 35.6% error, 22.4% deadzone
- Middle (10-19): 40.4% error, 26.4% deadzone ← HIGHEST
- Late-Mid (20-25): 35.6% error, 21.9% deadzone
- Last (26-27): 30.3% error, 17.1% deadzone

### Per-Experiment Correlation (To Be Filled)

**E13i-baseline**: TBD
**E13i-targeted**: TBD

*Analysis questions*:
1. Do high-error layers improve more with RIN?
2. Does deadzone ratio predict RIN effectiveness?
3. Is there a threshold effect (error >40% needs different treatment)?

---

## Best Configuration Tracker

### Current Best

**Experiment**: E13h (baseline, no RIN)
**Accuracy**: 56.41%
**Configuration**: MXFP4 W4A4 + STE, no RIN
**Gap vs NVFP4**: -4.47%

### History

| Date | Experiment | Accuracy | Key Factors | Notes |
|------|------------|----------|-------------|-------|
| 2026-01-15 | E13h | 56.41% | MXFP4 W4A4 STE | Baseline |

---

## Priority Queue (Dynamic)

### Current Priority Order

1. **E13i-baseline** (Conf 70%) - Global RIN
2. **E13i-targeted** (Conf 60%) - Targeted RIN
3. *Decision point after Phase 1*
4. E13i-high (Conf 55%) - Higher sigma
5. E13i-low (Conf 45%) - Lower sigma
6. *Decision point after Phase 2*
7. E13j-variable (Conf 65%) - Variable sigma

### Re-Prioritization Log

*To be updated after each experiment*

**Example**:
```
[Date] After E13i-baseline (58.2%):
- Success! Global RIN helps (+1.8%)
- Proceeding to Phase 2 sigma optimization
- Boosting priority of E13i-high (escape hypothesis looks promising)
```

---

## Detailed Experiment Logs

### E13h: MXFP4 W4A4 + STE (Baseline)

**Status**: ✅ Complete
**Date**: 2026-01-15
**Script**: `scripts/test_mxfp4_w4a4_ste_fix_e13h.sh`

**Results**:
- Step 0: 7.66%
- Step 20: **56.41%**
- Training score: 41.89%

**Configuration**:
- MXFP4 W4A4 mode (injection_point=both)
- STE enabled
- No RIN

**Analysis**:
- Establishes baseline for RIN experiments
- 4.47% gap vs E13g NVFP4 (60.88%)
- All layers have high SRDD error (36.4% mean)

**Next**: Test if RIN can close the gap

---

### E13i-baseline: Global RIN

**Status**: 📋 Planned - NEXT TO RUN
**Hypothesis**: H1a - Global RIN better (all layers need noise)
**Pre-Confidence**: 70%

**Configuration**:
```yaml
trainer.noise_injection.enabled = True
trainer.noise_injection.sigma_start = 0.05
trainer.noise_injection.sigma_end = 0.0005
trainer.noise_injection.num_stages = 10
trainer.noise_injection.target_layers = null  # ALL layers
trainer.noise_injection.exclude_patterns = ["lm_head", "embed_tokens", "lora_"]
```

**Expected Result**: 58-60% (+1.6 to +3.6%)

**Analysis Plan**:
1. Compare step-by-step accuracy vs E13h
2. Check if improvement is uniform across training
3. Analyze gradient norms for stability
4. Measure per-layer improvement correlation with SRDD error

**Success Criteria**:
- Minimum: ≥57% (+0.6%)
- Good: ≥58% (+1.6%)
- Excellent: ≥59% (+2.6%)

**If Successful**: Proceed to Phase 2 sigma optimization
**If Failed**: Test E13i-targeted and E13m-rmsnorm

---

### E13i-targeted: Targeted RIN (Middle Zone)

**Status**: 📋 Planned
**Hypothesis**: H1b - Targeted RIN better (focus on worst layers)
**Pre-Confidence**: 60%

**Configuration**:
```yaml
trainer.noise_injection.enabled = True
trainer.noise_injection.sigma_start = 0.05
trainer.noise_injection.sigma_end = 0.0005
trainer.noise_injection.num_stages = 10
trainer.noise_injection.target_layers = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
trainer.noise_injection.exclude_patterns = ["lm_head", "embed_tokens", "lora_"]
```

**Expected Result**: 57-59% (+0.6 to +2.6%)

**Comparison Focus**:
- vs E13i-baseline: Which is better?
- Zone analysis: Do layers 0-9 and 20-27 behave differently?

**Decision Logic**:
- If E13i-targeted > E13i-baseline + 0.5%: Targeted wins → optimize middle zone
- If E13i-baseline > E13i-targeted + 0.5%: Global wins → optimize globally
- If similar (±0.5%): Targeted sufficient → efficiency win

---

## Analysis After Each Experiment

*Template to fill after each experiment completes*

### Post-E13i-baseline Analysis

**[TO BE FILLED]**

#### Quantitative Results

```
Step 0: X.XX%
Step 5: X.XX%
Step 10: X.XX%
Step 15: X.XX%
Step 20: X.XX%
Improvement vs E13h: +X.XX%
```

#### Hypothesis Validation

```
H1a (Global RIN): ✓ Confirmed / ✗ Rejected / ~ Inconclusive
Reasoning: [...]
Confidence Update: 70% → XX%
```

#### SRDD Correlation

```
High-error layers (10-19) improvement: X.XX%
Low-error layers (0-9, 20-27) improvement: X.XX%
Correlation with deadzone: [positive/negative/none]
Correlation with relative error: [positive/negative/none]
```

#### Training Dynamics

```
Gradient stability: [stable/unstable]
Loss convergence: [smooth/oscillating]
Late-stage behavior: [improving/plateauing/degrading]
```

#### Decision for Next Experiment

```
Next: E13i-targeted [or other]
Reason: [Based on findings]
Priority adjustments: [Any changes to queue]
```

---

## Cross-Experiment Comparisons

### Accuracy Comparison

*To be filled as experiments complete*

```
E13h (baseline):     56.41% ████████████████████████████░░░░
E13i-baseline:       XX.XX% ████████████████████████████████
E13i-targeted:       XX.XX% ██████████████████████████████░░
E13i-high:           XX.XX% ████████████████████████████████
E13i-low:            XX.XX% ███████████████████████████████░
E13j-variable:       XX.XX% ████████████████████████████████░
E13g (NVFP4):        60.88% ████████████████████████████████░

Goal: ≥60.88%
```

### Configuration Summary

| Exp | RIN Mode | Sigma Range | Target Layers | Result | Rank |
|-----|----------|-------------|---------------|--------|------|
| E13h | None | - | - | 56.41% | Baseline |
| E13i-baseline | Global | 0.05→0.0005 | All | TBD | - |
| E13i-targeted | Targeted | 0.05→0.0005 | 10-19 | TBD | - |

---

## Insights and Learnings

### Key Findings (Updated After Each Experiment)

**Finding 1**: [TBD after E13i-baseline]

**Finding 2**: [TBD after E13i-targeted]

**Finding 3**: [TBD after Phase 2]

---

### Surprising Observations

*Document unexpected results here*

---

### Failed Hypotheses

*Document rejected hypotheses with reasoning*

---

## Next Steps and Open Questions

### Immediate Next Steps

1. ✅ SRDD analysis complete
2. ✅ Systematic experiment plan created
3. ⏭️ **Run E13i-baseline** (global RIN)
4. ⏭️ Analyze E13i-baseline results
5. ⏭️ Run E13i-targeted based on findings
6. ⏭️ Phase 1 decision point

### Open Questions

1. **Why does MXFP4 have 4.47% gap vs NVFP4?**
   - Is it purely quantization error (36.4% vs ~20%)?
   - Or are there other factors?

2. **Can RIN fully close the MXFP4-NVFP4 gap?**
   - Theoretical limit: ~60-62% for W4A4?
   - Or can we exceed NVFP4 with good RIN?

3. **Does deadzone ratio matter more than relative error?**
   - Layer 15: 42.65% error, 28.71% deadzone
   - Should we target by deadzone or error?

4. **Is there an interaction between layers?**
   - If layer 15 improves, does layer 14 benefit?
   - Or independent optimization?

5. **How does RIN compare to mixed precision?**
   - Would FP16 for layers 10-19 work better?
   - Or is RIN more practical?

---

## Resource Tracking

### Compute Time Used

| Experiment | Duration | GPU-hours | Status |
|------------|----------|-----------|--------|
| E13h | ~90 min | 12 | Complete |
| E13i-baseline | TBD | TBD | Pending |

**Total So Far**: 12 GPU-hours
**Estimated Remaining**: 42-60 GPU-hours (Phase 1-3)

---

## References and Links

- **Systematic Plan**: `docs/qerl/RIN_EXPERIMENT_PLAN_SYSTEMATIC.md`
- **SRDD Results**: `logs/srdd_analysis/mxfp4_activation_scan_20260115.json`
- **E13h Log**: `logs/w4a4_experiments/e13h_mxfp4_w4a4_ste_fix_56.41.log`
- **RIN Config Guide**: `docs/qerl/MXFP4_RIN_CONFIGURATION_GUIDE.md`
- **W4A4 Experiment Log**: `docs/qerl/E13_W4A4_EXPERIMENT_LOG.md`

---

## Maintenance Notes

**Update Frequency**: After each experiment completion
**Update Checklist**:
- [ ] Add experiment to status dashboard
- [ ] Fill detailed experiment log section
- [ ] Update hypothesis status tracker
- [ ] Update SRDD correlation analysis
- [ ] Update best configuration tracker
- [ ] Re-prioritize remaining experiments
- [ ] Document key findings and insights
- [ ] Update open questions
- [ ] Commit and push changes

**File Conventions**:
- Keep this file as master tracker
- Create separate `E13X_EXPERIMENT_LOG.md` for each experiment's details
- Update `RIN_EXPERIMENT_PLAN_SYSTEMATIC.md` if major hypothesis changes
