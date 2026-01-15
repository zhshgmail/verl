# RIN (Resilient-Improving Noise) Experiment Plan: Systematic Study

**Date**: 2026-01-15
**Goal**: Study correlation between SRDD quantization error analysis and RIN configuration for MXFP4 W4A4
**Baseline**: E13h MXFP4 W4A4 + STE = 56.41% (step 20)
**Target**: Approach E13g NVFP4 W4A4 = 60.88% (close the 4.47% gap)

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
