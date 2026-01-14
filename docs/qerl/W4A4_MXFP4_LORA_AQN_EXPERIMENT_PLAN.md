# W4A4 MXFP4 + LoRA + AQN Experiment Plan

**Date**: 2026-01-14
**Status**: In Progress (E13a running with fixed implementation)
**Branch**: `feature/npu-aqn-test`
**Critical Fix**: Commit `df828442` - W4A4 activation quantization must use POST-hook

---

## Executive Summary

This experiment tests **W4A4 MXFP4 quantization** (both weights AND activations quantized to 4-bit) combined with **16-bit LoRA adapters** and **SRDD-guided AQN** for training robustness.

### Key Question

**Will SRDD-guided AQN provide larger gains with W4A4 compared to W4A16?**

### Hypothesis

W4A4 creates **non-uniform error distribution** across layers because:
1. **Activation quantization error accumulates** through depth (deeper layers have more error)
2. **LoRA provides non-uniform compensation** (different ranks per layer)
3. **Combined effect**: Middle layers (L6-L15) likely to be error-dense

Expected: **SRDD-guided AQN gains 2-3%** vs global AQN.

---

## 1. Motivation

### 1.1 Previous Results

| Experiment | Config | Result | Gap to BF16 |
|------------|--------|--------|-------------|
| E7a | BF16 + LoRA | 71.27% | BASELINE |
| E6a | W4A16 MXFP4 + LoRA | 65.88% | -5.39% |
| E6b | W4A16 MXFP4 + LoRA + AQN | 67.48% | -3.79% |
| E12 | W4A16 MXFP4 + LoRA + SRDD-AQN | 72.48% | +1.21% |

**Key insight**: SRDD-guided AQN (E12) achieved **+5.00%** over uniform AQN (E6b).

### 1.2 Why W4A4?

| Aspect | W4A16 | W4A4 |
|--------|-------|------|
| **Quantization scope** | Weights only | Weights + Activations |
| **Error accumulation** | Linear with depth | Exponential with depth |
| **NPU relevance** | Partial | Full (W4A4 is real NPU mode) |
| **Expected error** | ~21% per layer | ~35-50% per layer |

---

## 2. Experiment Design

### 2.1 Experiment Matrix

| ID | Config | AQN Strategy | Purpose |
|----|--------|--------------|---------|
| **E13a** | W4A4 + LoRA (no AQN) | None | Baseline degradation |
| **E13b** | W4A4 + LoRA + Global AQN | All 28 layers, σ=0.01→0.00001 | Standard AQN |
| **E13c** | W4A4 + LoRA + SRDD-targeted AQN | Target layers only, σ=0.01→0.00001 | Binary targeting |
| **E13d** | W4A4 + LoRA + SRDD-variable AQN | Variable σ per layer | Best expected |

### 2.2 Training Configuration

```yaml
# Model
model: Qwen2.5-1.5B-Instruct
algorithm: DAPO
epochs: 1
total_steps: 29

# LoRA
lora_rank: 32
lora_alpha: 16
target_modules: all-linear

# Batch size (same as E6a/E6b)
train_batch_size: 128
gen_batch_size: 256
ppo_mini_batch_size: 32
n_resp_per_prompt: 8

# Learning rate
lr: 1e-5
warmup_steps: 10
```

### 2.3 W4A4 Quantization Configuration

```yaml
hw_error_injection:
  enabled: true
  error_type: mxfp4
  injection_point: both  # NEW: quantize BOTH weights and activations
  apply_during: both
  target_modules: ["linear"]
  exclude_modules: ["lm_head", "embed_tokens", "lora_A", "lora_B"]
  use_ste: true
```

### 2.4 AQN Configuration

```yaml
# E13b: Global AQN
noise_injection:
  enabled: true
  sigma_start: 0.01
  sigma_end: 0.00001
  num_stages: 10
  layer_types: ["rmsnorm"]

# E13c: SRDD-targeted AQN
noise_injection:
  enabled: true
  sigma_start: 0.01
  sigma_end: 0.00001
  num_stages: 10
  layer_types: ["rmsnorm"]
  layer_sigma_config:
    default_multiplier: 0.0    # No noise on non-targeted layers
    layer_multipliers:
      6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0,
      11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0

# E13d: SRDD-variable AQN
noise_injection:
  enabled: true
  sigma_start: 0.01
  sigma_end: 0.00001
  num_stages: 10
  layer_types: ["rmsnorm"]
  layer_sigma_config:
    default_multiplier: 1.0
    layer_multipliers:
      6: 1.5, 7: 1.5, 8: 1.5, 9: 1.5, 10: 1.5,
      11: 1.5, 12: 1.5, 13: 1.5, 14: 1.5, 15: 1.5
```

---

## 3. Expected Results

### 3.1 Predicted Error Distribution

Based on W4A4 error accumulation:

```
Layer Group      Weight Error    Activation Error    Total Error    Status
------------------------------------------------------------------------------
L0-L5 (early)         21%              15%              36%        MODERATE
L6-L15 (middle)       21%              25%              46%        ERROR DENSE
L16-L25 (deep)        21%              30%              51%        HIGH
L26-L27 (output)      21%              20%              41%        MODERATE
```

### 3.2 Predicted Accuracy

| Experiment | Predicted | Reasoning |
|------------|-----------|-----------|
| E13a (no AQN) | 60-62% | W4A4 more severe than W4A16 (65.88% - 4%) |
| E13b (global AQN) | 63-65% | AQN helps, similar to E6b gain (+1.6%) |
| E13c (targeted AQN) | 64-66% | Target error-dense layers (+1-2% vs global) |
| E13d (variable AQN) | 65-67% | Best - scale sigma by error level |

**Target**: E13d achieves **65-67%** (within 5% of BF16 baseline 71.27%)

---

## 4. Implementation Plan

### 4.1 Framework Changes

**File**: `verl/utils/hw_error_injection.py`

Add support for `injection_point='both'` (W4A4 mode):

```python
# In register_hooks() method:
elif self.config.injection_point == 'both':
    # W4A4 mode: quantize BOTH weights AND activations
    # Weight quantization (pre-hook + backward hook)
    pre_hook = module.register_forward_pre_hook(self._create_weight_quant_pre_hook(name))
    self.hooks.append(pre_hook)

    backward_hook = module.register_full_backward_hook(
        self._create_weight_restore_backward_hook(name)
    )
    self.hooks.append(backward_hook)

    # Activation quantization (POST-hook for output - see Section 4.2 for why!)
    act_hook = module.register_forward_hook(self._create_forward_hook(name))
    self.hooks.append(act_hook)
```

---

## 4.2 Critical Bug Discovery and Fix (2026-01-14)

### The Bug: Pre-hook Activation Quantization Destroys Model

**Symptom**: E13a validation accuracy was only **10.61%** at step 20, far below expected 60-62%.

**Root Cause**: The initial W4A4 implementation used **pre-hook** for activation quantization, which quantizes the **INPUT** to linear layers. In transformer architecture, this means quantizing **RMSNorm outputs**:

```
Transformer Data Flow:
┌─────────────────────────────────────────────────────────────────┐
│ hidden_states                                                    │
│     ↓                                                            │
│ input_layernorm (RMSNorm) → normalized output (carefully scaled) │
│     ↓                                                            │
│ q_proj, k_proj, v_proj (Linear)  ← INPUT IS RMSNORM OUTPUT!     │
│     ↓                                                            │
│ attention computation                                            │
│     ↓                                                            │
│ o_proj (Linear)                                                  │
│     ↓                                                            │
│ residual + output                                                │
│     ↓                                                            │
│ post_attention_layernorm (RMSNorm)                               │
│     ↓                                                            │
│ gate_proj, up_proj (Linear)  ← INPUT IS RMSNORM OUTPUT!         │
│     ↓                                                            │
│ SiLU activation + down_proj                                      │
└─────────────────────────────────────────────────────────────────┘
```

**Why This Breaks the Model**:
1. RMSNorm carefully normalizes activations to specific statistical distribution
2. 4-bit quantization (only 16 discrete values!) severely distorts this distribution
3. The normalized values are critical for attention softmax and gradient flow
4. Quantizing norm outputs effectively destroys the normalization benefit

### The Fix: Use POST-hook for Activation Quantization

**Correct W4A4 Flow**:
```
RMSNorm output (FP16) → Linear with W4 weights → Output (FP16) → Quantize to A4
```

**Wrong (Broken) Flow**:
```
RMSNorm output (FP16) → Quantize to A4 ❌ → Linear with W4 weights
```

**Code Change** (commit `df828442`):
```python
# BEFORE (broken):
# 3. Activation quantization pre-hook (applied before weight-quantized forward)
act_hook = module.register_forward_pre_hook(self._create_activation_quant_pre_hook(name))

# AFTER (fixed):
# 3. Activation quantization POST-hook (applied to LINEAR OUTPUT, not input!)
#    This uses _create_forward_hook which quantizes the output after computation
#    Key: This avoids quantizing RMSNorm outputs which are extremely sensitive
act_hook = module.register_forward_hook(self._create_forward_hook(name))
```

### Why W4A16 Is Not Affected

W4A16 uses `injection_point='weight'`, which:
- Only quantizes **weights** (not activations)
- Uses pre-hook correctly (weight quantization before forward)
- Does NOT touch RMSNorm outputs at all

The bug only affects `injection_point='both'` (W4A4 mode).

### Verification

| Implementation | Step 20 Val Accuracy | Status |
|----------------|---------------------|--------|
| Pre-hook (broken) | 10.61% | ❌ Model destroyed |
| Post-hook (fixed) | TBD (running) | Expected 60-62% |

### Key Learnings

1. **Never quantize normalization layer outputs to low precision** - they carry critical statistical information
2. **W4A4 "activation quantization" should target Linear OUTPUT**, not input
3. **Pre-hook vs Post-hook matters** - the hook placement determines what gets quantized
4. **QeRL uses `output_activations`** for this reason (see `compressed-tensors_replacement/forward.py`)

### 4.3 Scripts to Create

1. **`scripts/test_mxfp4_w4a4_lora_baseline.sh`** - E13a (no AQN)
2. **`scripts/test_mxfp4_w4a4_lora_global_aqn.sh`** - E13b (global AQN)
3. **`scripts/test_mxfp4_w4a4_lora_srdd_targeted.sh`** - E13c (targeted)
4. **`scripts/test_mxfp4_w4a4_lora_srdd_variable.sh`** - E13d (variable)

### 4.4 SRDD Scan (Optional Pre-step)

```bash
# Run SRDD scan on W4A4 to identify error-dense layers
python scripts/srdd_quant_scanner.py \
    --model_path /data/z00637938/hub/Qwen2.5-1.5B-Instruct \
    --quant_type mxfp4 \
    --injection_point both \
    --output w4a4_srdd_results.json
```

---

## 5. Execution Plan

### Phase 1: Framework Implementation (15 min)
- [x] Add `injection_point='both'` support to hw_error_injection.py
- [x] Test with simple forward pass

### Phase 2: Script Creation (10 min)
- [ ] Create E13a baseline script
- [ ] Create E13b global AQN script
- [ ] Create E13c/d SRDD-guided scripts

### Phase 3: Execute Experiments (2-3 hours)
- [ ] Run E13a (no AQN) - 1 hour
- [ ] Run E13b (global AQN) - 1 hour
- [ ] Run E13c or E13d (SRDD-guided) - 1 hour

### Phase 4: Analysis (30 min)
- [ ] Compare results
- [ ] Update experiment summary
- [ ] Document findings

---

## 6. Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| **E13a runs successfully** | Completes without crash | Training log |
| **W4A4 error higher than W4A16** | E13a < 65.88% (E6a) | Final accuracy |
| **Global AQN helps** | E13b > E13a by 2-4% | Accuracy gain |
| **SRDD-guided AQN helps** | E13d > E13b by 1-3% | Accuracy gain |
| **Final gap to baseline** | E13d within 5% of 71.27% | 66-67% target |

---

## 7. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| W4A4 too severe (all fail) | Start with E13a to validate degradation level |
| Framework changes break existing | Test W4A16 still works after changes |
| Out of memory | Use same batch size as E6a/E6b |
| Training crashes | Monitor first 5 steps, kill if unstable |

---

## 8. Timeline

| Phase | Duration | ETA |
|-------|----------|-----|
| Implementation | 30 min | 2026-01-14 14:00 |
| E13a run | 1 hour | 2026-01-14 15:00 |
| E13b run | 1 hour | 2026-01-14 16:00 |
| E13d run | 1 hour | 2026-01-14 17:00 |
| Analysis | 30 min | 2026-01-14 17:30 |

**Total**: ~4 hours

---

## 9. Follow-up Questions

After this experiment, we can answer:

1. **Is W4A4 viable for training?** (E13a result)
2. **Does AQN help with extreme quantization?** (E13b vs E13a)
3. **Is SRDD-guided AQN worth the complexity?** (E13d vs E13b)
4. **What's the gap to full precision?** (E13d vs E7a 71.27%)

---

## References

- Previous W4A16 results: `docs/qerl/LORA_EXPERIMENT_RESULTS_20260111.md`
- SRDD methodology: `docs/qerl/SRDD_TECH_REPORT_CN.md`
- SRDD-guided AQN: `docs/qerl/SRDD_GUIDED_AQN_PROPOSAL_CN.md`
