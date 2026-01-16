# SRDD-Guided RIN: Competing Hypotheses

**Date**: 2026-01-16
**Context**: E13l experiment design - how should SRDD error guide sigma multipliers?

## The Core Question

**SRDD Analysis Shows**:
- Middle layers (10-19): ~40% quantization error
- Edge layers (0-9, 20-27): ~32% quantization error

**Question**: Should high-error layers get MORE or LESS noise?

---

## Hypothesis A: High Error → MORE Noise (Robustness Training)

**Strategy**:
```
Layer 15 (42% error): 1.17x multiplier → σ_eff = 0.0585
Layer 26 (29% error): 0.79x multiplier → σ_eff = 0.0395
```

**Rationale**:
- High-error layers struggle with quantization → need robustness training
- MORE noise helps model learn to function despite quantization artifacts
- Trains the network to be resilient to the errors in these layers

**Implementation**: E13l

---

## Hypothesis B: High Error → LESS Noise (Don't Overwhelm)

**Strategy**:
```
Layer 15 (42% error): 0.79x multiplier → σ_eff = 0.0395
Layer 26 (29% error): 1.17x multiplier → σ_eff = 0.0585
```

**Rationale**:
- High-error layers already have "natural noise" from quantization
- Adding MORE noise on top might overwhelm them → training collapse
- Low-error layers are stable → can handle more noise for exploration
- Use exploration noise where the model can actually learn from it

**Implementation**: E13m (if E13l fails)

---

## QeRL's Approach (Uniform)

QeRL uses **uniform noise across all layers** (Global AQN):
- All layers: same sigma (e.g., 0.05 → 0.0005)
- No layer-specific tuning
- E13j achieved 73.31% with this approach

**Implication**: If NEITHER hypothesis A nor B beats 73.31%, then:
→ Layer-specific tuning doesn't help
→ Uniform Global AQN is sufficient
→ SRDDcomplexity not justified

---

## Experimental Plan

1. **E13l** (Hypothesis A): High error = MORE noise
   - Target: Beat 73.31%
   - If successful: Validates Hypothesis A, SRDD-RIN has value

2. **E13m** (Hypothesis B, if needed): High error = LESS noise
   - Only run if E13l ≤ 73.31%
   - Target: Beat 73.31%
   - If successful: Validates Hypothesis B, opposite intuition correct

3. **Conclusion**:
   - If BOTH fail: Uniform AQN (E13j) is optimal, Variable RIN not worth it
   - If ONE succeeds: SRDD provides value, document correct strategy
   - If BOTH succeed: Need ablation to determine which is better

---

## Why This Matters

This is NOT just academic - it determines whether SRDD analysis provides actionable guidance:

**If Hypothesis A wins**:
- SRDD → Direct mapping to RIN configuration
- High SRDD error = high sigma multiplier
- Clear correlation, easy to apply

**If Hypothesis B wins**:
- SRDD → Inverse mapping (counterintuitive)
- High SRDD error = low sigma multiplier
- Requires documenting the inversion

**If neither wins**:
- SRDD doesn't help for noise scheduling
- Use uniform Global AQN (simpler, works)
- SRDD may still help for other purposes (layer targeting, etc.)

---

## Current Status

- [x] E13j (uniform σ=0.05): **73.31%** ← baseline
- [x] E13k (uniform σ=0.01): **65.96%** ← worse
- [ ] E13l (Hyp A: high error = more noise): Running...
- [ ] E13m (Hyp B: high error = less noise): Pending (if needed)
