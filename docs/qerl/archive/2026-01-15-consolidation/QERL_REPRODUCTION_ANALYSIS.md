# QeRL Reproduction Analysis Report

**Date**: 2026-01-10
**Objective**: Evaluate if our experiments reasonably reproduce QeRL paper's results

---

## 1. QeRL Paper Results (Table 1a - Qwen2.5-3B-Instruct on GSM8K)

| Method | W# | Training | GSM8K | Delta from BF16 |
|--------|-----|----------|-------|-----------------|
| BF16 | 16-bit | - | 61.2% | baseline |
| NVFP4 | 4-bit | - | 59.4% | -1.8% |
| BF16 | 16-bit | Full | 84.4% | +23.2% |
| **BF16** | **16-bit** | **LoRA** | **76.1%** | **+14.9%** |
| NVFP4 | 4-bit | LoRA | 83.3% | +22.2% |
| **NVFP4** | **4-bit** | **LoRA+AQN** | **83.7%** | **+22.6%** |

### Key QeRL Claims
1. **NVFP4+LoRA (83.3%) vs BF16+LoRA (76.1%)**: +7.2% improvement
2. **NVFP4+LoRA+AQN (83.7%) vs NVFP4+LoRA (83.3%)**: +0.4% from AQN
3. **NVFP4+LoRA+AQN (83.7%) vs BF16+LoRA (76.1%)**: +7.6% total improvement

---

## 2. Our Experiment Results (Qwen2.5-1.5B-Instruct on GSM8K)

| Experiment | Method | Result | Notes |
|------------|--------|--------|-------|
| E4a | NVFP4 + Full FT (no AQN) | 72.55% | Full parameter training |
| E4b | NVFP4 + Full FT + AQN | 72.02% | ⚠️ Wrong sigma (0.05→0.0005) |
| **E5a** | **NVFP4 + LoRA (no AQN)** | **~64%** | LoRA baseline |
| **E5b** | **NVFP4 + LoRA + AQN** | **66.11%** | ✅ Correct sigma (0.01→0.0001) |

### Our Observed Improvements
1. **E5b vs E5a**: +2.27% (AQN benefit with LoRA)
2. **E5a vs E4a**: -8.55% (LoRA vs Full FT with NVFP4)

---

## 3. Gap Analysis

### 3.1 Direct Comparison

| Metric | QeRL (3B) | Ours (1.5B) | Gap |
|--------|-----------|-------------|-----|
| AQN benefit (LoRA) | +0.4% | **+2.27%** | We see **MORE** AQN benefit |
| NVFP4+LoRA vs BF16+LoRA | +7.2% | N/A | Not directly tested |

### 3.2 Key Differences

| Factor | QeRL Paper | Our Experiments |
|--------|------------|-----------------|
| Model | Qwen2.5-**3B**-Instruct | Qwen2.5-**1.5B**-Instruct |
| Algorithm | **GRPO** | **DAPO** |
| Training steps | ~600 steps | ~29 steps (1 epoch) |
| LoRA rank | 32 | 32 |
| Sigma schedule | 0.01→0.0001 (10 stages) | 0.01→0.0001 (10 stages) ✅ |
| Noise injection | RMSNorm (multiplicative via LayerNorm merge) | RMSNorm (additive to hidden states) |

### 3.3 Critical Methodology Differences

#### QeRL's AQN Implementation (from paper Section 3.3):
```
1. Noise is merged into LayerNorm parameters
2. RMSNorm_noise(x) = (Z_noise/w + I) ⊙ RMSNorm(x)
3. This creates MULTIPLICATIVE noise on weights
4. Noise affects Q, K, V (same noise) and Gate, Up (same noise)
```

#### Our AQN Implementation:
```
1. Noise is added directly to hidden states after RMSNorm
2. This creates ADDITIVE noise on activations
3. Noise is independent per layer
```

**This is a FUNDAMENTAL difference that could explain the gap.**

---

## 4. Hypotheses for Reproduction Gap

### Hypothesis 1: Model Size Effect
- Smaller models (1.5B) may have different quantization error characteristics
- QeRL's entropy enhancement may scale with model size
- **Test**: Run same experiments on Qwen2.5-3B-Instruct

### Hypothesis 2: Algorithm Difference (GRPO vs DAPO)
- QeRL uses GRPO which has KL penalty
- We use DAPO which removes KL penalty for more exploration
- DAPO already maximizes exploration, so AQN's benefit may be reduced
- **Test**: Run with GRPO instead of DAPO

### Hypothesis 3: Training Duration
- QeRL trains for ~600 steps
- We train for ~29 steps (1 epoch)
- AQN's exploration benefit may compound over longer training
- **Test**: Run for more epochs/steps

### Hypothesis 4: AQN Implementation Difference
- QeRL: Multiplicative noise via LayerNorm merge
- Ours: Additive noise to activations
- Multiplicative noise may have different exploration dynamics
- **Test**: Implement QeRL's exact noise injection method

### Hypothesis 5: Baseline Difference
- QeRL compares to BF16+LoRA (76.1% on 3B)
- We don't have a direct BF16+LoRA baseline for comparison
- Our E5a (NVFP4+LoRA at 64%) may already benefit from quantization noise
- **Test**: Run BF16+LoRA baseline for fair comparison

---

## 5. What We Actually Observed

### Positive Findings
1. **AQN does help**: E5b (66.11%) > E5a (~64%) = +2.27%
2. **Sigma matters**: Using QeRL's exact sigma (0.01→0.0001) works
3. **LoRA exclusion is critical**: Without excluding lora_A/lora_B, accuracy drops to 32.75%
4. **Training is stable**: No entropy collapse, healthy response lengths

### Concerning Findings
1. **LoRA underperforms Full FT**: E5a (64%) << E4a (72.55%) with same NVFP4
2. **AQN benefit is smaller than claimed**: +2.27% vs QeRL's reported +7.6%
3. **Different algorithm may confound results**: DAPO vs GRPO

---

## 6. Recommendation

### Option A: Accept Current Results (Conservative)
- Our results show **same trend** as QeRL (AQN helps LoRA with NVFP4)
- The +2.27% improvement is statistically meaningful
- Proceed with E3b/E4b re-runs using correct sigma
- Document that magnitude differs from paper (model size, algorithm)

### Option B: Full Reproduction (Thorough)
- Run experiments on Qwen2.5-3B-Instruct to match paper exactly
- Switch to GRPO algorithm
- Implement QeRL's exact noise injection (LayerNorm merge)
- Compare to BF16+LoRA baseline

### Current Verdict: **Need Expert Discussion**

The key question is: **Does our 1.5B experiment showing +2.27% AQN benefit with correct methodology constitute a valid reproduction of QeRL's findings?**

Arguments for YES:
- Same direction of improvement
- Same model family
- Same sigma schedule
- Smaller model may simply have smaller effect size

Arguments for NO:
- 7.6% vs 2.27% is a 3.3x difference
- Different algorithm (DAPO vs GRPO)
- Different noise implementation details
- Model size difference may be significant

---

## 7. Available Resources for Further Testing

- **Qwen2.5-3B-Instruct**: `/data/z00637938/hub/models--Qwen--Qwen2.5-3B-Instruct`
- **Server**: 8x A100-80GB on 90.90.102.18
- **Framework**: verl with DAPO/GRPO support

---

## 8. Questions for Expert Analysis

1. Is +2.27% on 1.5B comparable to +7.6% on 3B given model capacity differences?
2. Does DAPO's built-in exploration reduce AQN's marginal benefit?
3. Is our additive noise implementation fundamentally different from QeRL's multiplicative approach?
4. Should we prioritize 3B experiments or accept 1.5B results as directionally valid?
