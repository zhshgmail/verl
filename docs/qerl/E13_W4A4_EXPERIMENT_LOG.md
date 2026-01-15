# E13 Series: W4A4 Quantization Experiment Log

**Date**: 2026-01-14 ~ 2026-01-15
**Goal**: Achieve ~60% accuracy on GSM8K with W4A4 (4-bit weights + 4-bit activations)
**Status**: ✅ **RESOLVED** - STE fix enables W4A4 training for both NVFP4 and MXFP4

## Executive Summary

After 7 failed experiments (E13a-E13f, all ~7-10% accuracy), we discovered that **activation quantization requires STE (Straight-Through Estimator)** for gradient flow. With STE enabled:
- **E13g (NVFP4 W4A4)**: 60.88% accuracy at step 20 ✅
- **E13h (MXFP4 W4A4)**: 56.41% accuracy at step 20 ✅

**Root cause**: Without STE, gradients cannot flow through quantized activations, preventing learning. Weight-only quantization doesn't have this issue because weights are dequantized before computing gradients.

---

## Complete Results Summary (Step 0 and Step 20)

| ID | Description | Hook Type | Blocking | Step 0 Acc | Step 20 Acc | Status |
|----|-------------|-----------|----------|------------|-------------|--------|
| E13a-mxfp4 | MXFP4 POST-hook | POST | row-wise | 8.11% | 7.43% | FAILED |
| E13a-nvfp4 | NVFP4 POST-hook large batch | POST | row-wise | 7.58% | 9.02% | FAILED |
| E13b-nvfp4 | NVFP4 POST-hook small batch | POST | row-wise | 7.81% | 8.34% | FAILED |
| E13c-nvfp4 | NVFP4 POST-hook training-only | POST | row-wise | 7.58% | 9.02% | FAILED |
| E13d-nvfp4 | NVFP4 PRE-hook | PRE | row-wise | 8.49% | N/A | FAILED |
| E13e-nvfp4 | NVFP4 PRE-hook + exclude base_layer | PRE | row-wise | 7.66% | N/A | FAILED |
| E13f-nvfp4 | NVFP4 PRE-hook + column-wise | PRE | column-wise | 8.19% | 10.39% | FAILED |
| **E13g-nvfp4** | **NVFP4 W4A4 + STE FIX** | PRE | column-wise | 8.11% | **60.88%** | **SUCCESS** ✅ |
| **E13h-mxfp4** | **MXFP4 W4A4 + STE FIX** | PRE | column-wise | 7.66% | **56.41%** | **SUCCESS** ✅ |

**Expected accuracy**: ~60% (based on QeRL colleague's reproduction)
- **E13g (NVFP4)**: 60.88% - STE fix successfully resolved W4A4 training! 🎉
- **E13h (MXFP4)**: 56.41% - STE fix works for MXFP4 too! Ready for RIN experiments.

---

## Detailed Experiment Records

### E13a-mxfp4: MXFP4 W4A4 Large Batch

**Script**: `scripts/test_mxfp4_w4a4_lora_baseline.sh`
**Key Config**:
- error_type: mxfp4
- injection_point: both
- apply_during: both
- train_batch_size: 128
- Hook type: POST-hook (register_forward_hook on _create_forward_hook)

**Results**:
- Step 0: val-core/openai/gsm8k/acc/mean@1 = 8.11%
- Step 20: val-core/openai/gsm8k/acc/mean@1 = **7.43%** (DEGRADED)
- Status: FAILED (expected 60%)

---

### E13a-nvfp4: NVFP4 W4A4 Large Batch

**Script**: `scripts/test_nvfp4_w4a4_lora_baseline_fixed.sh`
**Key Config**:
- error_type: nvfp4
- injection_point: both
- apply_during: both
- train_batch_size: 128
- Hook type: POST-hook

**Results**:
- Step 0: val-core/openai/gsm8k/acc/mean@1 = 7.58%
- Step 20: val-core/openai/gsm8k/acc/mean@1 = **9.02%**
- Status: FAILED

---

### E13b-nvfp4: NVFP4 W4A4 Small Batch

**Script**: `scripts/test_nvfp4_w4a4_lora_small_batch.sh`
**Key Config**:
- error_type: nvfp4
- injection_point: both
- apply_during: both
- train_batch_size: 16 (2 per GPU, matches QeRL's single GPU setup)
- gen_batch_size: 32
- Hook type: POST-hook

**Hypothesis**: QeRL uses batch_size=2 on 1 GPU. Our batch_size=128 on 8 GPUs is 512x larger. Small batch might help.

**Results**:
- Step 0: val-core/openai/gsm8k/acc/mean@1 = 7.81%
- Step 20: val-core/openai/gsm8k/acc/mean@1 = **8.34%**
- Status: FAILED

**Conclusion**: Batch size is NOT the root cause.

---

### E13c-nvfp4: NVFP4 apply_during=training

**Script**: `scripts/test_nvfp4_w4a4_training_only.sh`
**Key Config**:
- error_type: nvfp4
- injection_point: both
- apply_during: training (NOT both)
- train_batch_size: 128
- Hook type: POST-hook

**Hypothesis**: QeRL might only apply quantization during training forward pass, not during vLLM inference/rollout.

**Results**:
- Step 0: val-core/openai/gsm8k/acc/mean@1 = 7.58%
- Step 20: val-core/openai/gsm8k/acc/mean@1 = **9.02%**
- Status: FAILED

**Conclusion**: apply_during setting is NOT the root cause.

---

### E13d-nvfp4: TRUE W4A4 with PRE-hook

**Script**: `scripts/test_nvfp4_w4a4_true_prehook.sh`
**Key Config**:
- error_type: nvfp4
- injection_point: both
- apply_during: both
- train_batch_size: 128
- Hook type: **PRE-hook** (register_forward_pre_hook on _create_activation_quant_pre_hook)

**Code Change** (commit b087831e):
```python
# Changed from:
act_hook = module.register_forward_hook(self._create_forward_hook(name))

# To:
act_hook = module.register_forward_pre_hook(self._create_activation_quant_pre_hook(name))
```

**Hypothesis**:
Our POST-hook was WRONG:
- POST-hook: `y_quant = quant(W_quant @ x_fp16)` - INPUT x remains FP16!
- PRE-hook: `y = W_quant @ x_quant` - TRUE W4A4

**Results**:
- Step 0: val-core/openai/gsm8k/acc/mean@1 = **8.49%**
- Status: FAILED

**Conclusion**: Both PRE-hook and POST-hook give ~8% accuracy. Hook placement is NOT the root cause.

---

### E13e-nvfp4: Exclude base_layer from quantization

**Script**: `scripts/test_nvfp4_w4a4_exclude_baselayer.sh`

**Bug found**: We were quantizing both LoRA wrapper (q_proj) and underlying base_layer (q_proj.base_layer), causing duplicate hooks (364 instead of 182).

**Fix**: Added 'base_layer' to exclusion list. Hook count reduced from 364 to 182.

**Results**:
- Step 0: val-core/openai/gsm8k/acc/mean@1 = **7.66%**
- Status: FAILED (actually worse than E13d!)

---

### E13f-nvfp4: Column-wise blocking (matching quant_compute reference)

**Script**: `scripts/test_nvfp4_w4a4_columnwise_e13f.sh`
**Key Config**:
- error_type: nvfp4
- injection_point: both
- apply_during: both
- train_batch_size: 128
- **Blocking direction: COLUMN-WISE** (matching quant_compute reference)

**Code Changes**:
1. Added `nvfp4_quantize_columnwise()` function to `verl/utils/nvfp4_quant.py`
2. Modified `_apply_nvfp4()` in `hw_error_injection.py` to use column-wise for 2D weight tensors
3. **Performance fix**: Vectorized the column-wise implementation (commit ea9930aa) - original Python loop version was 1000x slower

**Blocking Direction Comparison**:
```
Our row-wise:    Flatten tensor → reshape to (N, 16) → quantize rows
Reference:       For each column → take 16 rows → quantize column slice

Numerical test: Column-wise diff from reference = 0.005 (16x better than row-wise 0.085)
```

**Hypothesis**: The blocking direction mismatch was causing incorrect quantization patterns.

**Results**:
- Step 0: val-core/openai/gsm8k/acc/mean@1 = **8.19%**
- Step 20: val-core/openai/gsm8k/acc/mean@1 = **10.39%**
- Status: **FAILED**

**Conclusion**: Column-wise blocking shows slight improvement (10.39% vs 7-9% for others), but still far below expected 60%. The model IS learning (8.19% → 10.39%), but the learning is severely impaired.

---

### E13g-nvfp4: STE Fix for Activation Quantization

**Script**: `scripts/test_nvfp4_w4a4_ste_fix_e13g.sh`
**Date**: 2026-01-15
**Key Config**:
- error_type: nvfp4
- injection_point: both
- apply_during: both
- train_batch_size: 128
- Hook type: PRE-hook
- **STE enabled**: use_ste=True

**CRITICAL FIX APPLIED**:
After deep analysis, discovered that `STEQuantizeActivation` class was DEFINED but NEVER USED in activation quantization. This blocked gradient flow through quantized activations, preventing LoRA adapters from learning.

**Root Cause**:
1. W4A16 (E3-E7) succeeded because only weights were quantized - gradients flowed through FP16 activations
2. W4A4 (E13a-f) failed because activation quantization returned tensors directly without using STE
3. LoRA adapters received zero/corrupted gradients due to blocked backward pass

**Code Changes** (commit a04eacda):
1. Modified `_create_activation_quant_pre_hook` to use `STEQuantizeActivation.apply()`
2. Removed `@torch.no_grad()` decorators from `nvfp4_quantize`, `nvfp4_quantize_columnwise`, and `mxfp4_quantize`
3. This allows gradient computation through quantization operations

**Results**:
- Step 0: val-core/openai/gsm8k/acc/mean@1 = **8.11%** (expected baseline)
- Step 20: val-core/openai/gsm8k/acc/mean@1 = **60.88%** ✓
- Training scores progression:
  - Steps 1-3: 21-22%
  - Steps 12-16: 26-32%
  - Step 20: 51.76% (critic/score/mean)
- **Status**: **SUCCESS** - Achieved expected 60% accuracy!

**Comparison with Previous E13 Experiments**:
| Experiment | Step 20 Accuracy | vs E13g |
|------------|------------------|---------|
| E13a-mxfp4 | 7.43% | **8.2x worse** |
| E13a-nvfp4 | 9.02% | **6.7x worse** |
| E13b-nvfp4 | 8.34% | **7.3x worse** |
| E13f-nvfp4 | 10.39% | **5.9x worse** |
| **E13g-nvfp4** | **60.88%** | **✓ SUCCESS** |

**Key Insight**: The STE fix allows gradients to flow backward through quantized activations. Even in LoRA training, earlier layers need ∂L/∂y from upper layers, which was blocked without STE.

**Log Files**:
- Original path: `/tmp/nvfp4_w4a4_ste_fix_e13g/training.log` (deleted due to /tmp cleanup)
- Recovered to: `/home/z00637938/workspace/verl/logs/e13g_training_20260115_171158.log`

---

### E13h-mxfp4: MXFP4 W4A4 with STE Fix

**Script**: `scripts/test_mxfp4_w4a4_ste_fix_e13h.sh`
**Date**: 2026-01-15
**Key Config**:
- error_type: mxfp4
- injection_point: both (W4A4 mode)
- apply_during: both
- train_batch_size: 128
- Hook type: PRE-hook
- **STE enabled**: use_ste=True
- Blocking: column-wise (32-group size for MXFP4)

**Purpose**: Validate that the STE fix (E13g) also works for MXFP4 quantization format. MXFP4 is the target format for Ascend NPU deployment, with higher quantization error (~21% rel) compared to NVFP4 (~15% rel).

**Results**:
- Step 0: val-core/openai/gsm8k/acc/mean@1 = **7.66%** (baseline with MXFP4 W4A4)
- Step 20: val-core/openai/gsm8k/acc/mean@1 = **56.41%** ✓
- Training scores progression:
  - Step 1: 19.92% (critic/score/mean)
  - Step 20: 41.89% (critic/score/mean)
- **Status**: **SUCCESS** - STE fix works for MXFP4!

**Comparison with E13g (NVFP4)**:
| Metric | E13g (NVFP4) | E13h (MXFP4) | Difference |
|--------|--------------|--------------|------------|
| Step 20 accuracy | 60.88% | 56.41% | -4.47% |
| Quantization error | ~15% rel | ~21% rel | +6% |
| Step 0 baseline | 8.11% | 7.66% | -0.45% |
| Training score @step20 | 51.76% | 41.89% | -9.87% |

**Analysis**:
- MXFP4 achieves 56.41% vs NVFP4's 60.88% = **92.7% of NVFP4 accuracy**
- The 4.47% accuracy gap is expected given MXFP4's higher quantization error
- Both formats successfully train with STE enabled
- Ready to proceed with RIN (Resilient-Improving Noise) experiments on MXFP4

**Configuration Issues Resolved**:
1. **Logger configuration**: Changed `trainer.logger=null` to `trainer.logger='["console"]'` to avoid TypeError
2. **PyTorch distributed**: Container restart resolved duplicate backend string error

**Log Files**:
- Permanent path: `/home/z00637938/workspace/verl/logs/w4a4_experiments/e13h_mxfp4_w4a4_ste_fix_56.41.log`

**Next Steps**: E13i/j/k will add RIN (SRDD-guided noise injection) to MXFP4 W4A4 to further improve accuracy.

---

## Ruled Out Causes

After 7 experiments (E13a-E13f), the following have been ruled out as root cause:

| Hypothesis | Tested | Result |
|------------|--------|--------|
| MXFP4 vs NVFP4 format | E13a-mxfp4 vs E13a-nvfp4 | Both fail ~7-8% |
| Batch size (128 vs 16) | E13a vs E13b | Both fail ~7-8% |
| apply_during (both vs training) | E13a vs E13c | Both fail ~7-8% |
| Hook type (POST vs PRE) | E13c vs E13d | Both fail ~7-8% |
| Duplicate hooks on base_layer | E13d vs E13e | Both fail ~7-8% |
| Blocking direction (row vs column) | E13e vs E13f | Both fail ~7-10% |

**All experiments give ~7-10% accuracy at step 20, far below expected 60%.**

---

## Key Insight from Colleague

**Colleague successfully reproduced W4A4 results using `../quant_compute` directly.**

This suggests:
1. Our NVFP4 implementation may still differ from quant_compute in subtle ways
2. The integration of quantization into the training loop may have issues
3. The reference quant_compute implementation works correctly

---

## Remaining Investigation Directions

1. **Direct quant_compute integration**: Instead of reimplementing, directly call quant_compute functions
2. **Scale computation**: Compare our E4M3 scale computation with quant_compute reference in detail
3. **FP4 value mapping**: Verify our FP4 LUT (0, 0.5, 1, 1.5, 2, 3, 4, 6) matches exactly
4. **Inference vs Training mismatch**: Our rollout uses vLLM (no quant hooks), but training uses FSDP with hooks
5. **Get exact configuration from colleague**: What exact configuration/code did they use?
6. **Try W4A16**: Disable activation quant to isolate weight vs activation issue

---

## QeRL Code Analysis

**File**: `/home/zheng/workspace/QeRL/replacement/compressed-tensors_replacement/forward.py`

```python
def wrapped_forward(self, *args, **kwargs):
    input_ = args[0]

    # INPUT activation quantization (PRE-forward)
    if scheme.input_activations is not None:
        input_ = forward_quantize(module, input_, "input", scheme.input_activations)

    # Forward call with quantized input
    output = forward_func_orig.__get__(module, module.__class__)(
        input_, *args[1:], **kwargs
    )

    # OUTPUT activation quantization (POST-forward, optional)
    if scheme.output_activations is not None:
        output = forward_quantize(module, output, "output", scheme.output_activations)

    return output
```

QeRL supports BOTH input_activations (PRE) and output_activations (POST).
For W4A4, they use input_activations.

---

## Files Modified/Created

| File | Description |
|------|-------------|
| `verl/utils/hw_error_injection.py` | PRE-hook for activation quant, STE for gradient flow, column-wise for weight quant |
| `verl/utils/nvfp4_quant.py` | Added vectorized `nvfp4_quantize_columnwise()`, removed `@torch.no_grad()` for STE |
| `verl/utils/mxfp4_quant.py` | Removed `@torch.no_grad()` from `mxfp4_quantize()` for STE |
| `scripts/test_nvfp4_w4a4_training_only.sh` | E13c test script |
| `scripts/test_nvfp4_w4a4_true_prehook.sh` | E13d test script |
| `scripts/test_nvfp4_w4a4_exclude_baselayer.sh` | E13e test script |
| `scripts/test_nvfp4_w4a4_columnwise_e13f.sh` | E13f test script |
| **`scripts/test_nvfp4_w4a4_ste_fix_e13g.sh`** | **E13g test script with STE fix** |

---

## Git Commits

- `df828442` - "fix: W4A4 activation quantization should use POST-hook, not PRE-hook"
- `b087831e` - "fix: W4A4 must use PRE-hook for INPUT activation quantization (QeRL-style)"
- `edbf21e6` - "feat: add column-wise NVFP4 blocking to match quant_compute reference"
- `d8995d0d` - "docs: update E13 log with E13f failure and ruled out causes"
- `ea9930aa` - "perf: vectorize column-wise NVFP4 blocking for 1000x speedup"
- **`a04eacda`** - **"fix: apply STE to activation quantization for proper W4A4 gradient flow"** (E13g)

---

## Conclusion

**FINAL RESULT 2026-01-15**: ✅ **W4A4 PROBLEM SOLVED**

After 7 failed experiments (E13a-f), the root cause was identified and fixed in E13g:

### Root Cause
`STEQuantizeActivation` class was DEFINED but NEVER USED in activation quantization code. This blocked gradient flow through quantized activations between layers.

### Why This Matters Even for LoRA
While LoRA parameters don't need gradients through frozen weights, they DO need gradients flowing backward from the loss:
- In multi-layer W4A4, each layer's output is quantized before feeding to the next layer
- Without STE: quant(y) has no defined gradient → ∂L/∂y = 0 for earlier layers
- With STE: gradients flow through → all layers receive proper ∂L/∂y from the loss

### Results Summary

**W4A16 (weight-only quantization)**:
- E3-E7: 60-74% accuracy
- Gradients flow through FP16 activations naturally

**W4A4 without STE (E13a-f)**:
- All failed: 7-10% accuracy
- Activation quantization blocked gradient flow

**W4A4 with STE (E13g)**:
- **SUCCESS: 60.88% accuracy** ✓
- Matches expected performance
- Proves STE fix resolves the gradient flow issue

### Key Commits
- `a04eacda` - Applied STE to activation quantization
- Fixed by using `STEQuantizeActivation.apply()` and removing `@torch.no_grad()`

**Investigation complete. W4A4 + LoRA training now works correctly with NVFP4 quantization.**
