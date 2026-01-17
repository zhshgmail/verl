# E14: Zone-Based SRDD-Guided Noise Injection

## Context

After E13l/m/n failures, we learned that:
1. Per-layer sigma creates **gradient flow inconsistency** (expert diagnosis)
2. Direction is correct: resilient layers (low error) → more noise, vulnerable layers (high error) → less noise
3. Problem is mechanism, not strategy

## SRDD Zone Analysis

```
EDGE zones (layers 0-5, 23-27): 32.72% avg error - RESILIENT
MIDDLE zone (layers 6-22):      38.78% avg error - VULNERABLE

Gap: ~6% relative error difference between zones
```

## E14a: Schedule-Based Zone Noise

**Key Insight**: Maintain gradient consistency early, leverage SRDD late.

### Configuration
- Base: MXFP4 W4A4 + LoRA (rank=32, alpha=16)
- Target: RMSNorm layers
- Sigma schedule: 0.05 → 0.0005 (10 stages)

### Zone Noise Schedule

**Phase 1 (Steps 0-10): Uniform Noise**
- ALL zones: σ_base (standard schedule)
- Rationale: Maintain gradient flow consistency during rapid learning

**Phase 2 (Steps 11-20): SRDD-Guided Differential**
- EDGE zones (0-5, 23-27): σ_base × 1.0 (full noise)
- MIDDLE zone (6-22): σ_base × 0.5 (half noise)
- Rationale: Protect vulnerable layers during convergence

### Expected Outcome
- Expert assessment: ~35% success probability
- Target: Match or exceed E13j (73.31%)
- Risk: Zone boundaries may still cause discontinuities

---

## E14b: Learning Rate Scaling (Expert Recommendation)

**Alternative Approach**: Use SRDD to guide learning rate, not noise.

### Configuration
- Base: MXFP4 W4A4 + LoRA (rank=32, alpha=16)
- AQN: Uniform Global (like E13j, σ=0.05 → 0.0005)
- SRDD guides LoRA learning rate per zone

### Zone LR Multipliers
- MIDDLE zone (6-22): LR × 1.2 (help catch up for information loss)
- EDGE zones (0-5, 23-27): LR × 1.0 (standard)

### Rationale
- Expert: "SRDD should guide what to protect, not how much noise"
- High error = more information lost = needs higher LR to compensate
- Noise stays uniform = maintains gradient consistency

---

## Experiment Priority

1. **E14a** (Zone Noise Schedule): Tests if temporal separation solves gradient inconsistency
2. **E14b** (Zone LR Scaling): Tests expert's recommended pivot

## Success Criteria

- E14a success: ≥73.31% (match E13j)
- E14b success: >73.31% (beat E13j)

If both fail: SRDD-guided approaches may not be effective for noise injection; consider SRDD for other purposes (pruning, mixed precision allocation).
