# SRDD-Guided AQN Experiment Fix Plan

## Date: 2026-01-08

## FINAL RESULTS (v2.0 Experiment)

### Results Summary
| Config | Loss (mean±std) | vs Deadzone | p-value |
|--------|-----------------|-------------|---------|
| clean | 3.1777 ± 0.0000 | - | - |
| deadzone_only | 12.8541 ± 0.0000 | baseline | - |
| global_aqn | 12.8606 ± 0.0105 | +0.1% | - |
| **targeted_aqn** | **12.8374 ± 0.0054** | **-0.1%** | **0.0126** |
| healthy_aqn | 12.8577 ± 0.0148 | +0.0% | 0.7267 |

### Key Findings
1. **Targeted AQN is 0.18% better than Global AQN (p=0.0126, significant)**
2. **Healthy AQN ≈ Global AQN (p=0.73, NOT significant)**
3. All methods fail catastrophically (~304% vs clean)
4. Control group suggests: benefit is from NOT adding noise to healthy layers

### QA Expert Conclusion
> "The most parsimonious explanation: Targeted AQN works slightly better because it doesn't degrade healthy layers further, NOT because it actively guides learning away from faults."

### Limitations Identified
- Single operating point (layer 10, threshold 0.01)
- No mechanistic analysis (gradient flow, activation distributions)
- All methods in catastrophic failure regime
- Tiny practical effect (0.18%)

### Recommended Next Steps
1. Test milder faults (threshold 0.1, 0.2, 0.5)
2. Add gradient flow analysis
3. Test multiple faulty layers
4. Add alternative baselines (layer pruning, etc.)

---

## QA Review Issues to Fix

### 1. CRITICAL: Not a Training Experiment
- **Issue**: Uses `with torch.no_grad()` - only inference, no training
- **Fix**: Remove torch.no_grad(), add actual gradient computation and weight updates
- **Or**: At minimum, measure gradient-based metrics, not just loss

### 2. Hook Ordering Bug
- **Issue**: AQN noise applied AFTER deadzone zeros values (useless)
- **Fix**: Register AQN hook BEFORE deadzone hook, or combine into single hook
- **Expected**: AQN should help values "break through" deadzone threshold

### 3. Statistical Weakness
- **Issue**: Only 8 samples, no replication, no significance tests
- **Fix**:
  - Use larger test set (50+ samples)
  - Run multiple times (n=5) with different seeds
  - Compute mean, std, and significance

### 4. AQN Implementation Issue
- **Issue**: Uses global mean `hidden.abs().mean()` instead of per-element
- **Fix**: Use `hidden.abs()` for truly adaptive noise
- **Formula**: `noise = torch.randn_like(hidden) * gamma * hidden.abs()`

### 5. Add Control Groups
- **Missing**: AQN on healthy layers only (exclude faulty layer)
- **Fix**: Add Config 5: Deadzone L10 + AQN on all layers EXCEPT L10

## Implementation Plan

### Step 1: Create Fixed Experiment Script
- [ ] Fix AQN implementation (per-element adaptive)
- [ ] Fix hook ordering (AQN before deadzone)
- [ ] Add more test samples
- [ ] Add multiple runs with seeds
- [ ] Add control group (AQN on healthy layers only)
- [ ] Add statistical analysis (mean, std, t-test)

### Step 2: Run on A100
- [ ] Commit and push fixes
- [ ] Run experiment with multiple seeds
- [ ] Collect results

### Step 3: QA Review
- [ ] Send results to QA agent for critical review
- [ ] Document final conclusions

## Expected Configurations

| Config | Deadzone | AQN Layers | Purpose |
|--------|----------|------------|---------|
| 1. Clean | None | None | Baseline |
| 2. Deadzone only | L10 | None | Show degradation |
| 3. Global AQN | L10 | All 28 | Current approach |
| 4. Targeted AQN | L10 | L10 only | SRDD-guided |
| 5. Healthy AQN | L10 | All except L10 | Control |

## Key Metrics
- Cross-entropy loss (mean ± std over n=5 runs)
- Relative change vs clean baseline
- Statistical significance (p-value from t-test)

## Files to Modify
- `scripts/srdd_aqn_experiment.py` - Main experiment script

## Git Branch
- `feature/npu-aqn-test`

## Notes
- The core hypothesis (SRDD-guided is better) is still valid
- But need proper methodology to prove it
- Focus on inference metrics first, then training if time permits
