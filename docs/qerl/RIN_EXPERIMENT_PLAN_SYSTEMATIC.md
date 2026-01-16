# RIN (Resilient-Improving Noise) Experiment Plan: Systematic Study

**Date**: 2026-01-15 (Updated: 2026-01-16)
**Goal**: Study correlation between SRDD quantization error analysis and RIN configuration for MXFP4 W4A4
**Baseline**: E13h MXFP4 W4A4 + STE = **71.42%** (step 29 final)
**Context**: MXFP4 W4A4 now OUTPERFORMS NVFP4 W4A4 (70.89%) by +0.53%
**Target**: Explore if RIN can improve beyond 71.42% baseline toward BF16 Full FT (74.75%)

---

## ⚠️ CRITICAL FINDING: E13i-baseline FAILED (2026-01-16)

**Experiment**: E13i-baseline (Global RIN, σ=0.05→0.0005)
**Result**: ❌ **FAILED at step 3** - Generation quality collapse
**Error**: `num_gen_batches=5 >= max_num_gen_batches=5` (filter rejection loop)

**Root Cause Analysis**:
- Global RIN activated at step 3 with σ=0.05
- **Compounded precision loss**: W4A4 (4-bit weights + 4-bit activations) + σ=0.05 noise
- Generation quality degraded so severely that filter system rejected all batches
- Attempted 5 regenerations, never got enough quality samples to proceed

**Critical Insight**:
> **σ_start=0.05 that works for W4A16 is TOO AGGRESSIVE for W4A4!**
>
> W4A16 has 16-bit activations providing precision buffer. W4A4 has NO buffer - activations already degraded to 4-bit. Adding σ=0.05 noise on top of 4-bit activations crosses the "usability threshold" where model output becomes gibberish.

**Revised Strategy**:
1. **Lower sigma dramatically**: Try σ_start=0.01 or 0.005 (10x lower)
2. **Start with targeted RIN**: Fewer layers = less aggregate noise impact
3. **Consider adaptive sigma**: Scale down further for W4A4 vs W4A16

**Evidence**:
- Steps 0-2: Training normal (σ=0.0)
- Step 3: RIN activated → immediate collapse
- Log archived: `e13i_baseline_global_rin_FAILED_step3_sigma0.05.log`

---

## ⚠️ CRITICAL CONCERN: SRDD-RIN Correlation Validity (2026-01-16)

### The Fundamental Problem

**Observation**: W4A16 tolerates σ=0.05, but W4A4 requires σ=0.01 (5x lower). Yet SRDD scans would show **identical results** for both configurations if we only scan activation quantization error.

**The Gap**:
- ✅ **SRDD measures**: Static quantization error at activation level
- ❌ **SRDD cannot predict**: Training dynamics, sigma tolerance, stability boundaries

### What SRDD Misses

SRDD provides spatial error distribution but cannot capture:

1. **Weight precision impact**: W4 vs W16 for weights (not measured by activation scan)
2. **Compounded noise tolerance**: How model handles quantization error + training noise
3. **Gradient flow quality**: Dynamic behavior through noisy activations during training
4. **Training stability boundaries**: Empirical thresholds where model collapses

**Critical Insight**:
> SRDD tells us **WHERE** layers have high error (spatial), but not **HOW MUCH** additional noise they can tolerate (dynamic).

### Scope Reassessment

**What SRDD CAN Still Guide**:
1. ✅ **Layer targeting**: Which layers need noise injection?
   - All layers similar error → Global RIN
   - Specific layers much worse → Targeted RIN (layers 10-19)
2. ✅ **Relative scaling**: Layer-specific sigma multipliers
   - High-error layers: 1.5x multiplier
   - Low-error layers: 0.5x multiplier
3. ✅ **Spatial patterns**: First/middle/last layer error distribution

**What SRDD CANNOT Predict**:
1. ❌ **Absolute sigma values**: Need empirical search per quantization mode
2. ❌ **Quantization mode tolerance**: W4A16 vs W4A4 sigma difference unpredictable
3. ❌ **Training stability**: When generation quality collapses

### Implications for This Study

**If SRDD only guides layer selection**, then:
- **Targeted RIN** (SRDD picks layers) has value over blind uniform noise
- **Variable RIN** (SRDD-weighted multipliers) may have value
- **Sigma tuning** requires per-mode empirical search anyway

**Alternative to pure SRDD-RIN correlation**:
1. Use SRDD for **layer targeting only** (WHERE question)
2. Build **sigma lookup table** per quantization mode (HOW MUCH question):
   - W4A16: σ ∈ {0.01, 0.03, 0.05, 0.08}
   - W4A4: σ ∈ {0.002, 0.005, 0.01, 0.02}
3. Combine: SRDD-targeted layers + empirically-tuned sigma per mode

### Decision Criteria: When to Stop

**Continue SRDD-RIN work if**:
- Targeted RIN (SRDD-selected layers) outperforms global RIN
- Variable RIN (SRDD-weighted multipliers) outperforms uniform scaling
- Can establish **relative correlation** (high-error layers need different sigma than low-error)

**Pivot away from SRDD-RIN if**:
- Targeted RIN shows no benefit vs global RIN
- Variable RIN shows no benefit vs uniform scaling
- Simple AQN (uniform noise, no SRDD) achieves similar results

**Scientific value even if SRDD-RIN fails**:
- ✅ Document what **doesn't work** (helps community avoid dead ends)
- ✅ Establish **sigma tuning guidelines** per quantization mode
- ✅ Show **training dynamics matter** beyond static error analysis

### E13i-v2 Result: FAILED at Step 4 (σ=0.01)

**Experiment**: E13i-v2 with σ=0.01 global RIN (5x lower than E13i-baseline)
**Result**: ❌ **FAILED at step 4** - Filter rejection loop, generation quality collapse

**Progression**:
- Steps 0-2: Normal (σ=0.0, no noise)
- Step 3: ✅ **Passed** (σ=0.01 activated, training continued)
- Step 4: ❌ **Failed** (5 regeneration attempts, all rejected)

**Comparison**:

| Experiment | Sigma | Failure Point | Survival |
|------------|-------|---------------|----------|
| E13i-baseline | σ=0.05 | Step 3 | 2 steps |
| E13i-v2 | σ=0.01 | Step 4 | 3 steps |

**Critical Finding**:
> **5x sigma reduction only delayed failure by 1 step.**
>
> Even σ=0.01 causes progressive generation quality degradation. The compounded precision loss (W4A4 quantization + training noise) accumulates over steps, eventually crossing the "usability threshold" where filter rejects all output.

**Root Cause - Progressive Degradation**:
1. W4A4 baseline: 4-bit weights + 4-bit activations (high intrinsic error)
2. Training noise: σ=0.01 added on top (even "small" noise compounds)
3. Accumulation: Quality degrades progressively over training steps
4. Threshold crossed: By step 4, output quality below filter acceptance
5. Rejection loop: All 5 regeneration batches rejected → training halted

**Evidence archived**: `e13i_v2_lower_sigma_FAILED_step4_sigma0.01.log`

---

## ⚠️ CRITICAL CONCLUSION: W4A4 + RIN Fundamental Incompatibility?

### The Paradox

**Observation**: Even minimal training noise (σ=0.01) fails within 3-4 steps for W4A4.

**Two interpretations**:

#### 1. Sigma Too High (Optimistic)
- σ=0.01 still too aggressive
- Try even lower: σ ∈ {0.005, 0.002, 0.001}
- Hypothesis: There exists a "Goldilocks sigma" for W4A4

#### 2. Fundamental Incompatibility (Realistic)
- **W4A4 has NO precision buffer** (unlike W4A16's 16-bit activations)
- **ANY additional noise** compounds with already-high quantization error
- **Training dynamics require quality generation** - can't learn from gibberish
- Hypothesis: W4A4 + training noise = fundamentally broken

### Evidence for Incompatibility

1. **Rapid failure progression**: σ=0.05→step 3, σ=0.01→step 4
   - 5x reduction bought only 1 step
   - Extrapolate: σ=0.002 might reach step 5-6 before failing

2. **No precision buffer**: W4A16 works because 16-bit acts absorb noise
   - W4A4 acts already degraded to 4-bit
   - No room for additional perturbation

3. **Progressive degradation**: Not immediate collapse, but accumulation
   - Suggests approaching an asymptote, not finding sweet spot
   - Like trying to add noise to already-noisy signal

### Decision Point: Continue or Pivot?

**Option A: Try σ=0.005 (ultra-low)**
- **Pros**: Might work, validates "Goldilocks" hypothesis
- **Cons**: If fails at step 5-6, confirms asymptotic behavior
- **Cost**: ~90 minutes, another failed experiment

**Option B: Pivot away from RIN for W4A4**
- **Accept**: W4A4 too constrained for robust training methods
- **Focus**: Better quantization formats (NF4, GPTQ, AWQ) or W4A8
- **Value**: Document negative result for community

**Option C: Test targeted RIN (layers 10-19, σ=0.005)**
- **Rationale**: Fewer layers = less aggregate noise
- **Test**: SRDD "WHERE" hypothesis with ultra-conservative sigma
- **Risk**: Still likely to fail if issue is fundamental

### Recommendation

**Try ONE more experiment** (E13i-v3: σ=0.005 global) as scientific due diligence:
- If **passes step 10**: Validates Goldilocks hypothesis → continue tuning
- If **fails by step 6-7**: Confirms asymptotic behavior → **PIVOT AWAY**

**Then regardless of result**:
1. ✅ **Document W4A4 RIN limitations** thoroughly

---

## 🔍 BREAKTHROUGH: Root Cause Identified - Implementation Bugs (2026-01-16)

### Critical Context: QeRL Proves W4A4 + AQN Works!

**User insight**: "before we give up RIN, you should know that the QeRL (../QeRL) has already proved W4A4 works with AQN"

This changes everything! If QeRL succeeds with W4A4 + noise injection, our failures suggest **implementation bugs**, not fundamental incompatibility.

### Bug Investigation: Comparing QeRL vs verl

#### Bug #1: Incompatible Filter Mechanism (PRIMARY BUG)

**The Problem**: verl's `filter_groups` mechanism is **fundamentally incompatible** with aggressive noise injection.

**How filter_groups Works** (dapo_ray_trainer.py:230-285):
```python
# Step 1: Filter prompts by trajectory variance
prompt_uid2metric_std[uid] = np.std(trajectory_rewards)
kept_prompts = [uid for uid, std in ... if std > 0 or len(trajectories) == 1]

# Step 2: If not enough prompts, keep generating
if num_prompt_in_batch < train_batch_size:
    if num_gen_batches < max_num_gen_batches:  # max=5
        continue  # Generate more batches
    else:
        raise ValueError("Generated too many batches")  # ❌ HALT!
```

**Normal Training (no noise)**:
- Some trajectories good (reward=1.0), some bad (reward=0.0)
- Variance exists → prompts kept
- Training proceeds

**With Aggressive Noise (σ=0.05 or 0.01)**:
- ALL trajectories uniformly degraded
- ALL rewards → 0.0 (generation gibberish)
- Variance ≈ 0 → **ALL prompts rejected!**
- System keeps regenerating → hits max_num_gen_batches=5 → **TRAINING HALTS**

**Why QeRL Doesn't Have This Problem**:
- QeRL has **NO such filter mechanism**
- Bad noise → low scores, but training continues
- Model learns from low-quality samples (still contains signal)
- This is **NORMAL and EXPECTED** per user's insight:
  > "How's a gating can fail the whole testing? Normally it just lead to long response, large entropy and lower score."

**Evidence in Our Code**:
```bash
# All our experiments have this filter enabled:
$ grep filter_groups scripts/test_mxfp4_w4a4_e13*.sh
enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=5
```

#### Bug #2: Wrong Target Layers (SECONDARY BUG)

**QeRL Implementation** (QeRL/trl_trainer/noise_scheduler.py:24-25):
```python
# Targets RMSNorm layers ONLY
if isinstance(module, Qwen2RMSNorm) or isinstance(module, LlamaRMSNorm):
    weight_tensor = module.weight
    noise = torch.normal(mean=0, std=sigma, ...)
    module.weight.add_(noise)
```

**Our Implementation** (verl/utils/noise_injection.py:175-176):
```python
# Default: RMSNorm (QeRL compatible)
if layer_types is None:
    layer_types = ['rmsnorm']
```

**But Our E13i Scripts Override**:
```bash
# E13i-baseline and E13i-v2
'++trainer.noise_injection.layer_types=["linear"]'  # ❌ WRONG!
```

**Impact**:
- **RMSNorm**: Normalization layers, fewer parameters, less critical for generation
- **Linear**: Weight matrices (attention, MLP), core computation, MUCH larger impact
- Targeting Linear → **more aggressive perturbation** → faster quality degradation

### Complete Bug Summary

| Aspect | QeRL (Working) | verl E13i (Failing) |
|--------|----------------|---------------------|
| **Filter mechanism** | ❌ No filter | ✅ filter_groups with max=5 |
| **Failure behavior** | Low scores, training continues | Training HALTS (rejection loop) |
| **Target layers** | RMSNorm only | Linear layers |
| **Noise impact** | Localized (norm layers) | Global (weight matrices) |
| **Result** | ✅ W4A4 works | ❌ Training crashes |

### Proposed Fixes

#### Fix #1: Disable filter_groups for RIN Experiments (REQUIRED)

**Change**:
```bash
# Current (E13i-baseline/v2)
enable_filter_groups=True
max_num_gen_batches=5

# Fixed (E13i-v3)
enable_filter_groups=False  # Or set max_num_gen_batches=0 for unlimited
```

**Rationale**:
- Filter was designed for data quality control, NOT for handling training noise
- RIN/AQN expects some bad samples during training (part of the method)
- Rejecting bad samples defeats the purpose of robustness training

#### Fix #2: Target RMSNorm Layers (Match QeRL)

**Change**:
```bash
# Current (E13i-baseline/v2)
'++trainer.noise_injection.layer_types=["linear"]'

# Fixed (E13i-v3)
'++trainer.noise_injection.layer_types=["rmsnorm"]'  # Match QeRL
```

**Rationale**:
- QeRL's proven approach targets RMSNorm
- Less aggressive than perturbing Linear weight matrices
- May allow higher sigma values while maintaining stability

### Validation Plan

**E13i-v3: Bug Fixes Applied**
- ✅ Disable filter_groups (or set max_num_gen_batches=0)
- ✅ Target RMSNorm layers (match QeRL)
- ✅ Test with σ=0.05 (original baseline sigma)
- **Expected**: Training proceeds past step 4, possibly to completion

**E13i-v4: Iterative Debugging** (if E13i-v3 still fails)
- Try ONLY Fix #1 (disable filter, keep Linear targeting)
- Try ONLY Fix #2 (keep filter, use RMSNorm targeting)
- Isolate which bug is primary vs secondary

**E14: W4A16 MXFP4 + RIN Validation** (user's suggested backup)
- Verify our implementation works for W4A16
- If W4A16 also fails → confirms recent STE changes broke something
- If W4A16 succeeds → isolates issue to W4A4 specifics

---
2. ✅ **Establish sigma tolerance boundaries** empirically
3. ✅ **Guide future quantization research** (what doesn't work matters)
4. Consider **alternative robustness methods** for W4A4 (e.g., distillation, better initialization)

---

## Scientific Methodology

This document follows a systematic approach:

1. **Formulate hypotheses** with clear reasoning and confidence levels
2. **Design experiments** to test each hypothesis
3. **Execute in priority order** (highest confidence first)
4. **Analyze results** after each experiment
5. **Reflect and update** hypotheses based on findings
6. **Re-prioritize** remaining experiments
7. **Document everything** for future SRDD-RIN correlation analysis

---

## SRDD Analysis Summary (Baseline Data)

**Source**: `/home/z00637938/workspace/verl/logs/srdd_analysis/mxfp4_activation_scan_20260115.json`

### Overall Statistics

| Metric | Min | Max | Mean±Std | All Layers Status |
|--------|-----|-----|----------|-------------------|
| Relative Error % | 28.6 | 42.7 | 36.4±3.3 | 100% problematic |
| Deadzone % | 15.7 | 28.7 | 22.9±3.4 | 100% high |
| SQNR (dB) | 16.0 | 18.1 | 17.0±0.4 | 100% low |

### Error Distribution by Zone

| Zone | Layers | Mean Error | Mean Deadzone | Range |
|------|--------|------------|---------------|-------|
| First (0-3) | 4 | 32.5% | 18.9% | Lower |
| Early-Mid (4-9) | 6 | 35.6% | 22.4% | Medium |
| **Middle (10-19)** | 10 | **40.4%** | **26.4%** | **HIGHEST** |
| Late-Mid (20-25) | 6 | 35.6% | 21.9% | Medium |
| Last (26-27) | 2 | 30.3% | 17.1% | Lower |

**Key Pattern**: Clear gradient with middle layers (10-19) having significantly worse error.

### Top 10 Worst Layers

| Layer | Relative Error | Deadzone % | SQNR (dB) |
|-------|----------------|------------|-----------|
| 15 | 42.65% | 28.71% | 17.2 |
| 14 | 41.94% | 27.98% | 17.2 |
| 16 | 41.85% | 27.98% | 17.2 |
| 17 | 40.77% | 27.07% | 17.2 |
| 12 | 40.60% | 26.53% | 17.2 |
| 13 | 40.36% | 26.44% | 17.2 |
| 11 | 39.94% | 26.04% | 17.2 |
| 18 | 39.50% | 25.78% | 17.2 |
| 19 | 38.96% | 25.36% | 17.2 |
| 10 | 38.57% | 24.71% | 17.1 |

---

## Hypotheses and Reasoning

### Hypothesis 1: Global vs Targeted RIN

**Question**: Should we apply RIN to ALL layers (global) or only high-error layers (targeted)?

#### H1a: Global RIN is better (ALL layers need noise)

**Reasoning**:
- SRDD shows 100% of layers have high error (36.4% mean)
- Even "low-error" layers (28-33%) are still far above acceptable threshold (10%)
- W4A4 quantizes BOTH weights and activations everywhere
- Gradient flow needs smooth error landscape across all layers
- Targeted noise might create discontinuities in error distribution

**Supporting Evidence**:
- E12 (W4A16 + global RIN) achieved 72.48%
- All previous AQN experiments used global noise
- Layer 0 still has 30% error (3x threshold)

**Confidence**: **70%** - Strong theoretical basis, but middle layers clearly worse

**Expected Outcome**: 58-60% accuracy (+1.6 to +3.6% vs E13h)

#### H1b: Targeted RIN is better (focus on worst layers)

**Reasoning**:
- Middle layers (10-19) have 40.4% error vs 32.5% for first layers
- This 8% differential (24% relative difference) is significant
- Injecting noise in low-error layers might hurt more than help
- Want to match noise intensity to actual quantization error distribution
- Targeted approach is more efficient

**Supporting Evidence**:
- Clear SRDD gradient shows middle layers need more help
- E12's high σ (0.1) might have been overkill for W4A16
- Principle of least intervention

**Confidence**: **60%** - Logical but contradicts "global error" observation

**Expected Outcome**: 57-59% accuracy (+0.6 to +2.6% vs E13h)

**Experiment Design**: Test both and compare directly.

---

### Hypothesis 2: Sigma Level for High-Error Zones

**Question**: For layers with HIGH deadzone (>25%), should we use HIGHER or LOWER sigma?

#### H2a: High-error zones need HIGHER sigma (escape hypothesis)

**Reasoning**:
- High deadzone (25-29%) means many values falling to zero
- Model is "stuck" in a bad local optimum
- Need strong perturbation to escape the deadzone trap
- More turbulence = more exploration = find better gradients
- Analogy: Simulated annealing needs high temperature to escape local minima

**Supporting Evidence**:
- E12 used high σ=0.1 and succeeded (72.48%)
- Deadzone is fundamentally a "stuck" problem
- Physics: need activation energy to cross barrier

**Confidence**: **55%** - Intuitive but untested for W4A4

**Expected Outcome**: Layers 10-19 with σ=0.08→0.001 perform better than σ=0.05→0.0005

#### H2b: High-error zones need LOWER sigma (sufficient error hypothesis)

**Reasoning**:
- High deadzone means layer ALREADY has 25-29% values corrupted
- This is inherent "noise" from quantization error
- Adding MORE noise on top = double penalty
- Model needs stable gradients to learn, not chaos
- Exploration already provided by quantization error itself

**Supporting Evidence**:
- 40% relative error is already massive perturbation
- Too much noise → gradient explosion or vanishing
- W4A4 harder than W4A16 (E12 context different)

**Confidence**: **45%** - Counter-intuitive but possible

**Expected Outcome**: Layers 10-19 with σ=0.03→0.0003 perform better than σ=0.05→0.0005

**Experiment Design**:
1. Baseline global: σ=0.05→0.0005 (E13i-baseline)
2. High zone elevated: Layers 10-19 use σ=0.08→0.001 (E13i-high)
3. High zone reduced: Layers 10-19 use σ=0.03→0.0003 (E13i-low)

---

### Hypothesis 3: Variable vs Constant Sigma

**Question**: Should sigma scale proportionally with error (variable) or be constant (binary)?

#### H3a: Variable sigma is better (proportional scaling)

**Reasoning**:
- SRDD shows continuous error gradient (28% → 43%)
- Noise should match error intensity for optimal adaptation
- Layer 15 (43% error) needs more help than Layer 0 (30% error)
- Fine-grained control = better optimization
- Matches intuition: bigger problem → bigger intervention

**Mathematical Model**:
```
multiplier = layer_error / mean_error
sigma_layer = base_sigma * multiplier

Layer 15: 42.65/36.4 = 1.17x base sigma
Layer 0: 30.08/36.4 = 0.83x base sigma
```

**Supporting Evidence**:
- E12 used variable σ based on SRDD (RIN-variable)
- Continuous optimization generally better than discrete

**Confidence**: **65%** - Strong theoretical basis

**Expected Outcome**: 59-61% accuracy (+2.6 to +4.6% vs E13h)

#### H3b: Constant sigma is better (binary selection)

**Reasoning**:
- All layers above threshold (>28% error vs 10% threshold)
- Difference between 30% and 43% error might not matter for training
- Simpler = more robust, less hyperparameter sensitivity
- Binary selection easier to interpret and debug
- Noise injection is qualitative, not quantitative signal

**Supporting Evidence**:
- Many successful regularization techniques use constant dropout
- W4A4 might be too noisy for fine-grained tuning
- Occam's Razor: simplicity wins

**Confidence**: **35%** - Simpler but less sophisticated

**Expected Outcome**: 58-59% accuracy (+1.6 to +2.6% vs E13h)

**Experiment Design**: Test both with same layer selection.

---

### Hypothesis 4: Noise Schedule (Decay Rate)

**Question**: How fast should sigma decay during training?

#### H4a: Aggressive decay (E12 style: 10 stages)

**Config**: σ = 0.05 → 0.0005 over 10 stages (10x reduction per stage)

**Reasoning**:
- Early training: need exploration (high σ)
- Late training: need exploitation (low σ)
- 1 epoch = 29 steps, decay every 3 steps
- E12 (72.48%) used this schedule successfully

**Confidence**: **70%** - Proven in E12

#### H4b: Conservative decay (fewer stages)

**Config**: σ = 0.05 → 0.005 over 5 stages (2x reduction per stage)

**Reasoning**:
- W4A4 needs continuous noise to combat quantization
- Too fast decay = model forgets robustness
- Maintain moderate noise throughout training

**Confidence**: **40%** - Speculative

#### H4c: Constant sigma (no decay)

**Config**: σ = 0.03 throughout training

**Reasoning**:
- Quantization error is constant (doesn't decrease)
- Model should learn WITH noise, not without it
- Deployment will have quantization error, so train with it

**Confidence**: **30%** - Radical departure from prior work

**Experiment Design**: Start with H4a (proven), test others if needed.

---

### Hypothesis 5: Target Module Selection

**Question**: Which modules should have noise injection?

#### H5a: RMSNorm only (QeRL original)

**Reasoning**:
- QeRL paper used RMSNorm for AQN
- Normalization layers have multiplicative effect
- Small noise → large impact on downstream activations

**Confidence**: **50%** - Worked for W4A16, but W4A4 different

#### H5b: All linear layers (comprehensive)

**Reasoning**:
- W4A4 quantizes ALL linear layer activations
- SRDD measured error at decoder layer output (post-all-linears)
- Need noise where quantization happens

**Confidence**: **60%** - More directly matches W4A4 mode

**Experiment Design**: Start with H5b (linear), fallback to H5a if poor results.

---

## Experiment Priority Ranking

### Priority 1: Baseline and Core Hypotheses

These experiments establish the foundation and test the most confident hypotheses.

#### E13i-baseline: Global RIN with Standard Config

**Priority**: **HIGHEST** (must run first)
**Confidence**: 70%
**Tests**: H1a (global), H4a (aggressive decay), H5b (linear)

**Configuration**:
```yaml
trainer.noise_injection.enabled = True
trainer.noise_injection.sigma_start = 0.05
trainer.noise_injection.sigma_end = 0.0005
trainer.noise_injection.num_stages = 10
trainer.noise_injection.target_modules = []  # Empty = ALL decoder layers
trainer.noise_injection.target_layers = null  # ALL layers
trainer.noise_injection.exclude_patterns = ["lm_head", "embed_tokens", "lora_"]
```

**Expected Result**: 58-60% (+1.6 to +3.6% vs E13h 56.41%)
**Interpretation**:
- If ≥58%: Global RIN helps W4A4 ✓ → Continue with zone experiments
- If <58%: RIN might not help W4A4 ✗ → Investigate why

**Analysis Questions**:
1. Does accuracy improve compared to E13h (56.41%)?
2. How does training score progression compare to E13h?
3. Do we see improvement throughout training or plateau?

---

#### E13i-targeted: Targeted RIN (Middle Zone Only)

**Priority**: **HIGH** (test H1b)
**Confidence**: 60%
**Tests**: H1b (targeted), H4a (aggressive decay), H5b (linear)

**Configuration**:
```yaml
trainer.noise_injection.enabled = True
trainer.noise_injection.sigma_start = 0.05
trainer.noise_injection.sigma_end = 0.0005
trainer.noise_injection.num_stages = 10
trainer.noise_injection.target_modules = []
trainer.noise_injection.target_layers = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
trainer.noise_injection.exclude_patterns = ["lm_head", "embed_tokens", "lora_"]
```

**Expected Result**: 57-59% (+0.6 to +2.6% vs E13h)
**Interpretation**:
- If E13i-targeted > E13i-baseline: Targeted better (H1b wins)
- If E13i-baseline > E13i-targeted: Global better (H1a wins)
- If both similar (~±0.5%): Targeted sufficient (efficiency win)

**Analysis Questions**:
1. Does limiting to high-error layers hurt or help?
2. Do low-error layers (0-9, 20-27) show different behavior?
3. Is there interaction between zones?

---

### Priority 2: Sigma Magnitude Experiments

These test H2a vs H2b for high-error zones.

#### E13i-high: Elevated Sigma for High-Error Zone

**Priority**: **MEDIUM-HIGH** (test H2a escape hypothesis)
**Confidence**: 55%
**Tests**: H2a (higher sigma for deadzone)

**Configuration**:
```yaml
# Implement zone-based sigma (requires code changes or multi-config)
# Zone 1 (layers 10-19): σ = 0.08 → 0.001 (60% higher start, 100% higher end)
# Zone 2 (other layers): σ = 0.05 → 0.0005 (baseline)
```

**Pre-Execution Check**: Verify if current codebase supports per-zone sigma. If not, may need to:
1. Run separate experiments with different target_layers, OR
2. Implement multi-zone config support

**Expected Result**: 58-60% if H2a correct, 56-58% if H2b correct
**Interpretation**:
- If E13i-high > E13i-baseline: High error needs MORE noise ✓ (H2a)
- If E13i-high < E13i-baseline: High error needs LESS noise ✓ (H2b)

**Analysis Questions**:
1. Do layers 10-19 show better gradient updates?
2. Is there instability (gradient explosion) with σ=0.08?
3. How does deadzone ratio change during training?

---

#### E13i-low: Reduced Sigma for High-Error Zone

**Priority**: **MEDIUM** (test H2b sufficient error hypothesis)
**Confidence**: 45%
**Tests**: H2b (lower sigma for high deadzone)

**Configuration**:
```yaml
# Zone 1 (layers 10-19): σ = 0.03 → 0.0003 (40% reduction)
# Zone 2 (other layers): σ = 0.05 → 0.0005 (baseline)
```

**Expected Result**: 58-60% if H2b correct, 57-58% if H2a correct
**Interpretation**:
- If E13i-low > E13i-baseline: Less noise better for high error ✓ (H2b)
- If E13i-low < E13i-baseline: More noise needed ✓ (H2a)

**Analysis Questions**:
1. Does reduced noise improve gradient stability?
2. Do high-error layers still improve with less noise?
3. Is there an optimal noise level between 0.03 and 0.08?

---

### Priority 3: Variable Sigma Experiments

These test H3a (proportional scaling) vs H3b (constant).

#### E13j-variable: Error-Proportional Sigma

**Priority**: **MEDIUM** (test H3a)
**Confidence**: 65%
**Tests**: H3a (variable sigma proportional to error)

**Configuration**:
```yaml
trainer.noise_injection.enabled = True
trainer.noise_injection.use_variable_sigma = True  # Requires code implementation
trainer.noise_injection.sigma_start = 0.05  # Base value
trainer.noise_injection.sigma_end = 0.0005
trainer.noise_injection.num_stages = 10
trainer.noise_injection.layer_multipliers = {
    0: 0.83, 1: 0.89, 2: 0.93, 3: 0.93, 4: 0.93,
    5: 0.98, 6: 0.99, 7: 1.01, 8: 1.00, 9: 1.02,
    10: 1.06, 11: 1.10, 12: 1.12, 13: 1.11, 14: 1.15,
    15: 1.17, 16: 1.15, 17: 1.12, 18: 1.09, 19: 1.07,
    20: 1.02, 21: 0.99, 22: 0.96, 23: 0.94, 24: 0.91,
    25: 0.90, 26: 0.79, 27: 0.88,
}
```

**Pre-Execution Check**:
1. Does codebase support `use_variable_sigma` and `layer_multipliers`?
2. If not, need to implement in noise_injector.py

**Expected Result**: 59-61% (+2.6 to +4.6% vs E13h)
**Interpretation**:
- If E13j-variable > E13i-baseline by >1%: Variable scaling helps ✓
- If similar (±0.5%): Complexity not worth it
- If worse: Over-optimization, stick with simpler approach

**Analysis Questions**:
1. Do layer-specific multipliers create smoother convergence?
2. Is the improvement distributed across all layers or concentrated?
3. Are the multipliers correctly calibrated (0.83-1.17x)?

---

### Priority 4: Ablation Studies

These experiments test edge cases and alternative hypotheses.

#### E13k-constant: Constant Sigma (No Decay)

**Priority**: **LOW** (test H4c radical hypothesis)
**Confidence**: 30%
**Tests**: H4c (no decay)

**Configuration**:
```yaml
trainer.noise_injection.sigma_start = 0.03
trainer.noise_injection.sigma_end = 0.03  # No decay
trainer.noise_injection.num_stages = 1
```

**Expected Result**: 54-57% (likely worse than decay)
**Interpretation**:
- If surprisingly good: Quantization needs continuous noise
- If poor as expected: Decay is necessary for convergence

**When to Run**: Only if E13i-baseline succeeds and time permits.

---

#### E13l-conservative: Conservative Decay

**Priority**: **LOW** (test H4b)
**Confidence**: 40%
**Tests**: H4b (slower decay)

**Configuration**:
```yaml
trainer.noise_injection.sigma_start = 0.05
trainer.noise_injection.sigma_end = 0.005  # 10x reduction (not 100x)
trainer.noise_injection.num_stages = 5
```

**Expected Result**: 57-59%
**Interpretation**:
- If better than aggressive: W4A4 needs sustained noise
- If worse: Standard decay is optimal

**When to Run**: If E13i-baseline shows late-training degradation.

---

#### E13m-rmsnorm: RMSNorm Target (QeRL Original)

**Priority**: **LOW** (test H5a)
**Confidence**: 50%
**Tests**: H5a (RMSNorm vs linear)

**Configuration**:
```yaml
trainer.noise_injection.target_modules = ["rmsnorm"]  # QeRL original
# Keep other params same as E13i-baseline
```

**Expected Result**: 57-59%
**Interpretation**:
- If better than E13i-baseline: RMSNorm is the right target
- If worse: W4A4 needs noise in linear layers where quantization occurs

**When to Run**: If E13i-baseline fails or performs poorly.

---

## Execution Order by Confidence

### Phase 1: Foundation (Must Run)

**Goal**: Establish baseline RIN effect and global vs targeted

1. **E13i-baseline** (Priority 1, Conf 70%)
2. **E13i-targeted** (Priority 1, Conf 60%)

**Decision Point**:
- If BOTH fail (<57%): RIN might not help W4A4, investigate why
- If ONE succeeds (≥58%): Continue with winner's config to Phase 2
- If BOTH succeed: Use better performer for Phase 2

---

### Phase 2: Sigma Optimization (Conditional on Phase 1 success)

**Goal**: Find optimal sigma for high-error zones

3. **E13i-high** (Priority 2, Conf 55%) - Test "escape" hypothesis
4. **E13i-low** (Priority 2, Conf 45%) - Test "sufficient error" hypothesis

**Decision Point**:
- If E13i-high wins: H2a confirmed (high error needs MORE noise)
- If E13i-low wins: H2b confirmed (high error needs LESS noise)
- If neither beats Phase 1 baseline: Uniform sigma is optimal

**Reflection Questions**:
1. Does deadzone ratio decrease more with high or low sigma?
2. Is there gradient instability with σ=0.08?
3. Should we test intermediate values (σ=0.06)?

---

### Phase 3: Variable Sigma (Conditional on Phase 2 insights)

**Goal**: Test if fine-grained control improves results

5. **E13j-variable** (Priority 3, Conf 65%)

**Pre-Execution**:
- Calibrate multipliers based on Phase 2 findings
- If Phase 2 showed high sigma better, boost middle layer multipliers
- If Phase 2 showed low sigma better, reduce middle layer multipliers

**Decision Point**:
- If improvement >1%: Variable sigma worth the complexity
- If improvement <0.5%: Stick with simpler zone-based approach

---

### Phase 4: Ablations (Optional, Time Permitting)

**Goal**: Test alternative hypotheses and edge cases

6. **E13m-rmsnorm** (Priority 4, Conf 50%) - If Phase 1 failed
7. **E13l-conservative** (Priority 4, Conf 40%) - If late-training degradation seen
8. **E13k-constant** (Priority 4, Conf 30%) - Academic curiosity

---

## Analysis Framework

### Per-Experiment Analysis Checklist

After EACH experiment, document:

#### 1. Quantitative Results

```markdown
## E13x Results

**Configuration**: [Link to script]
**Hypothesis Tested**: H1a/H2b/etc.
**Pre-Experiment Confidence**: X%

**Results**:
- Step 0: X.XX%
- Step 10: X.XX%
- Step 20: X.XX%
- Final accuracy: X.XX%
- vs E13h baseline: +/- X.XX%
- vs best previous: +/- X.XX%

**Training Dynamics**:
- Critic score progression: [values]
- Gradient norm: [stable/unstable]
- Loss convergence: [smooth/oscillating]
```

#### 2. Hypothesis Validation

```markdown
**Hypothesis Status**: ✓ Confirmed / ✗ Rejected / ~ Inconclusive

**Reasoning**:
- [Why hypothesis was confirmed/rejected]
- [What evidence supports this conclusion]
- [Any surprising findings]

**Confidence Update**: X% → Y%
```

#### 3. SRDD Correlation Analysis

```markdown
**SRDD Correlation**:
- Did high-error layers (10-19) improve more than low-error layers?
- Does improvement correlate with deadzone ratio?
- Does improvement correlate with relative error?
- Correlation coefficient (if calculable): r = X.XX
```

#### 4. Comparative Analysis

```markdown
**vs E13h (no RIN)**: [Better/Worse/Similar]
**vs E13i-baseline (if applicable)**: [Better/Worse/Similar]
**vs E13g (NVFP4)**: [Gap closed by X.XX%]

**Best Configuration So Far**:
- Experiment: E13x
- Accuracy: X.XX%
- Key factors: [Global/Targeted, High/Low sigma, etc.]
```

#### 5. Next Steps

```markdown
**Immediate Next Experiment**: E13y
**Reason**: [Based on findings from this experiment]
**Priority Change**: [Any re-prioritization needed]

**New Hypotheses Generated**:
- H6: [If new pattern observed]

**Questions for Future Work**:
- [Open questions]
- [Anomalies to investigate]
```

---

## Expected Outcomes by Scenario

### Scenario A: Global RIN Works Well (Most Likely)

**If E13i-baseline achieves 58-60%**:

✓ **Confirmed**: H1a (global RIN)
→ Next: Test H2a vs H2b (sigma magnitude)
→ Then: Test H3a (variable sigma)
→ Goal: Push toward 60-61%

**Interpretation**: All layers benefit from noise, consistent with global high error.

---

### Scenario B: Targeted RIN Works Better (Possible)

**If E13i-targeted > E13i-baseline by >0.5%**:

✓ **Confirmed**: H1b (targeted RIN)
→ Next: Focus on optimizing middle zone (10-19) sigma
→ Then: Test variable sigma for middle zone only
→ Goal: Maximize efficiency

**Interpretation**: Noise in low-error layers hurts, focus intervention.

---

### Scenario C: RIN Doesn't Help Much (Unexpected)

**If ALL experiments show <57% (<0.6% improvement)**:

✗ **Rejected**: RIN beneficial for W4A4
→ Investigate: Why did E12 (W4A16 + RIN) work but not W4A4?
→ Hypothesis: Activation quantization adds too much noise already
→ Alternative: Focus on better quantization methods (NVFP4-like improvements)

**Interpretation**: W4A4 fundamentally harder than W4A16, RIN not the solution.

---

### Scenario D: High Sigma Works (Escape Hypothesis Wins)

**If E13i-high beats E13i-baseline and E13i-low**:

✓ **Confirmed**: H2a (escape hypothesis)
→ Next: Test even higher sigma (σ=0.10?) for layers 10-19
→ Then: Variable sigma with elevated middle zone
→ Goal: Aggressive noise for deadzone layers

**Interpretation**: Deadzone is a trap, need strong perturbation to escape.

---

### Scenario E: Low Sigma Works (Sufficient Error Hypothesis Wins)

**If E13i-low beats E13i-baseline and E13i-high**:

✓ **Confirmed**: H2b (sufficient error hypothesis)
→ Next: Test even lower sigma (σ=0.02?) for layers 10-19
→ Then: Variable sigma with reduced middle zone
→ Goal: Minimize additional noise where error already high

**Interpretation**: Quantization provides natural noise, don't add more.

---

## Implementation Notes

### Code Modifications Needed

#### For Variable Sigma (E13j)

**Files to modify**:
1. `verl/trainer/config/ppo_trainer.yaml`:
```yaml
noise_injection:
  use_variable_sigma: false  # Add this flag
  layer_multipliers: {}  # Add this dict
```

2. `verl/utils/noise_injector.py` (or equivalent):
```python
def get_sigma(self, layer_id: int, base_sigma: float) -> float:
    if self.config.use_variable_sigma:
        multiplier = self.config.layer_multipliers.get(layer_id, 1.0)
        return base_sigma * multiplier
    return base_sigma
```

#### For Zone-Based Sigma (E13i-high/low)

**Option 1**: Run separate experiments with different configs
**Option 2**: Implement multi-zone config (more complex)

**Recommendation**: Start with Option 1 (separate experiments), implement Option 2 if pattern emerges.

---

## Success Criteria

### Experiment-Level Success

| Experiment | Minimum Success | Good Success | Excellent Success |
|------------|-----------------|--------------|-------------------|
| E13i-baseline | ≥57% | ≥58% | ≥59% |
| E13i-targeted | ≥57% | ≥58% | ≥59% |
| E13i-high | ≥58% | ≥59% | ≥60% |
| E13i-low | ≥58% | ≥59% | ≥60% |
| E13j-variable | ≥59% | ≥60% | ≥61% |

### Overall Program Success

**Goal**: Close the MXFP4-NVFP4 gap

| Gap Reduction | Status | Interpretation |
|---------------|--------|----------------|
| 0-1% (56.4→57.4%) | ❌ Fail | RIN doesn't help W4A4 |
| 1-2% (57.4→58.4%) | ~ Marginal | Small benefit, not game-changer |
| 2-3% (58.4→59.4%) | ✓ Success | RIN helps, practical improvement |
| 3-4% (59.4→60.4%) | ✓✓ Strong Success | RIN closes most of gap |
| >4% (>60.4%) | ✓✓✓ Exceptional | RIN fully closes gap or exceeds NVFP4 |

**Final Target**: ≥60% to match E13g NVFP4 performance

---

## Documentation Templates

### Experiment Execution Log Template

Create file: `docs/qerl/E13X_EXPERIMENT_LOG.md`

```markdown
# E13x: [Experiment Name]

**Date**: 2026-01-XX
**Script**: scripts/test_mxfp4_w4a4_rin_e13x.sh
**Hypothesis**: HXy - [Hypothesis name]
**Pre-Experiment Confidence**: X%

## Configuration

[Full config details]

## Pre-Experiment Analysis

**Expected Outcome**: X.X% to Y.Y%
**Reasoning**: [Why we expect this]
**Alternative Outcomes**: [What if different?]

## Execution

**Started**: [Timestamp]
**PID**: [Process ID]
**Log**: [Path]

## Results

[Filled after completion]

## Analysis

[Filled after results]

## Next Steps

[Filled after analysis]
```

### Cross-Experiment Analysis Template

Create file: `docs/qerl/RIN_EXPERIMENT_ANALYSIS_SUMMARY.md`

Update after EACH experiment with:
- Comparison table
- Hypothesis status tracker
- Correlation analysis
- Priority re-ranking

---

## Timeline Estimate

Assuming each experiment takes ~90 minutes:

- **Phase 1** (E13i-baseline, E13i-targeted): ~3 hours
- **Phase 2** (E13i-high, E13i-low): ~3 hours
- **Phase 3** (E13j-variable): ~1.5 hours
- **Phase 4** (Ablations): ~3 hours (optional)

**Total**: 7-10 hours of compute time over 1-2 days

---

## Risk Management

### Risk 1: Code doesn't support variable sigma

**Mitigation**: Test E13i-baseline and E13i-targeted first (no code changes needed)
**Contingency**: If needed for E13j, implement feature or skip to simpler configs

### Risk 2: All experiments fail (<57%)

**Mitigation**: Document learnings, investigate why W4A4 different from W4A16
**Contingency**: Focus on alternative approaches (better quantization, mixed precision)

### Risk 3: Results inconsistent

**Mitigation**: Run multiple seeds, check for stability
**Contingency**: Use median of 3 runs if high variance

### Risk 4: Time constraints

**Mitigation**: Prioritize Phase 1 and 2, skip Phase 4 if needed
**Contingency**: Document remaining experiments for future work

---

## References

- **E13h Baseline**: 56.41% (MXFP4 W4A4 + STE, no RIN)
- **E13g Comparison**: 60.88% (NVFP4 W4A4 + STE, no RIN)
- **E12 Reference**: 72.48% (MXFP4 W4A16 + RIN-variable)
- **SRDD Analysis**: `/home/z00637938/workspace/verl/logs/srdd_analysis/mxfp4_activation_scan_20260115.json`
- **RIN Config Guide**: `docs/qerl/MXFP4_RIN_CONFIGURATION_GUIDE.md`
