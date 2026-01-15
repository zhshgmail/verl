# E13 Series: W4A4 Quantization Experiment Log

**Date**: 2026-01-14 ~ 2026-01-15
**Goal**: Achieve ~60% accuracy on GSM8K with W4A4 (4-bit weights + 4-bit activations)
**Status**: ALL EXPERIMENTS FAILED (~7-10% accuracy vs expected 60%)

## Executive Summary

After 7 experiments (E13a-E13f), we have systematically ruled out multiple hypotheses but still cannot achieve the expected 60% accuracy. All experiments show ~7-10% accuracy at step 20, indicating the model is not learning effectively under W4A4 quantization.

**Key insight from colleague**: They successfully reproduced W4A4 results using `../quant_compute` directly, suggesting our NVFP4 implementation may still have subtle differences from the reference.

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
| **E13f-nvfp4** | **NVFP4 PRE-hook + column-wise** | PRE | **column-wise** | 8.19% | **10.39%** | FAILED |

**Expected accuracy**: ~60% (based on QeRL colleague's reproduction)

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
| `verl/utils/hw_error_injection.py` | PRE-hook for activation quant, column-wise for weight quant |
| `verl/utils/nvfp4_quant.py` | Added vectorized `nvfp4_quantize_columnwise()` function |
| `scripts/test_nvfp4_w4a4_training_only.sh` | E13c test script |
| `scripts/test_nvfp4_w4a4_true_prehook.sh` | E13d test script |
| `scripts/test_nvfp4_w4a4_exclude_baselayer.sh` | E13e test script |
| `scripts/test_nvfp4_w4a4_columnwise_e13f.sh` | E13f test script |

---

## Git Commits

- `df828442` - "fix: W4A4 activation quantization should use POST-hook, not PRE-hook"
- `b087831e` - "fix: W4A4 must use PRE-hook for INPUT activation quantization (QeRL-style)"
- `edbf21e6` - "feat: add column-wise NVFP4 blocking to match quant_compute reference"
- `d8995d0d` - "docs: update E13 log with E13f failure and ruled out causes"
- `ea9930aa` - "perf: vectorize column-wise NVFP4 blocking for 1000x speedup"

---

## Conclusion

Despite extensive investigation and 7 experiments:
- All W4A4 experiments fail with ~7-10% accuracy
- Expected accuracy is ~60%
- Multiple hypotheses have been ruled out
- Colleague's success with quant_compute suggests our implementation differs in some critical way
- Next step: Get exact details from colleague or directly integrate quant_compute
