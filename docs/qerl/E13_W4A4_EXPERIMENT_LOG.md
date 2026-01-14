# E13 Series: W4A4 Quantization Experiment Log

**Date**: 2026-01-14
**Goal**: Achieve ~60% accuracy on GSM8K with W4A4 (4-bit weights + 4-bit activations)

## Experiment Summary Table

| ID | Description | Hook Type | Val Acc Step 0 | Val Acc Step 20 | Status |
|----|-------------|-----------|----------------|-----------------|--------|
| E13a-mxfp4 | MXFP4 W4A4 large batch (POST-hook) | POST | 8.11% | 7.43% | FAILED |
| E13a-nvfp4 | NVFP4 W4A4 large batch (POST-hook) | POST | 7.58% | 9.02% | FAILED |
| E13b-nvfp4 | NVFP4 W4A4 small batch (POST-hook) | POST | 7.81% | 8.34% | FAILED |
| E13c-nvfp4 | NVFP4 apply_during=training (POST-hook) | POST | 7.58% | 9.02% | FAILED |
| E13d-nvfp4 | NVFP4 TRUE W4A4 (PRE-hook) | PRE | TBD | TBD | RUNNING |

**Expected**: ~60-62% (based on QeRL)

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
- Step 20: val-core/openai/gsm8k/acc/mean@1 = 7.43%
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
- Step 20: val-core/openai/gsm8k/acc/mean@1 = 9.02%
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
- Step 20: val-core/openai/gsm8k/acc/mean@1 = 8.34%
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
- Step 20: val-core/openai/gsm8k/acc/mean@1 = 9.02%
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

QeRL's forward.py line 374-376 shows they use `input_activations` (PRE-hook style).

**Results**: TBD (running)

---

## Investigation Timeline

1. **Initial observation**: W4A4 experiments give ~8% accuracy instead of expected 60%
2. **Hypothesis 1 - Format**: MXFP4 vs NVFP4. Both fail → ruled out
3. **Hypothesis 2 - Batch size**: 512x larger than QeRL. Small batch also fails → ruled out
4. **Hypothesis 3 - apply_during**: training vs both. No difference → ruled out
5. **Hypothesis 4 - Hook type**: POST-hook vs PRE-hook. **TESTING NOW**

---

## Key Finding: POST-hook vs PRE-hook

**Previous "fix" (commit df828442)**: Changed from PRE-hook to POST-hook
- Reasoning: PRE-hook destroys RMSNorm outputs (only 16 discrete values)
- Result: Still fails with ~8% accuracy

**Current hypothesis**: The POST-hook "fix" was actually a BUG!
- POST-hook quantizes OUTPUT: `y_quant = quant(W_quant @ x)`
- But true W4A4 needs INPUT quantization: `y = W_quant @ x_quant`

QeRL's code confirms they use `input_activations` (line 374-376 in forward.py).

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
For W4A4, they likely use input_activations.

---

### E13d-nvfp4: TRUE W4A4 with PRE-hook

**Results**:
- Step 0: val-core/openai/gsm8k/acc/mean@1 = **8.49%**
- Status: FAILED

**Conclusion**: Both PRE-hook and POST-hook give ~8% accuracy. Hook placement is NOT the root cause.

---

## CRITICAL FINDING: QeRL Uses Pre-Quantized Models!

After analyzing QeRL's source code, discovered a fundamental difference:

### QeRL's Approach (qerl.py)

```python
# Line 71: Detect NVFP4 model by name
noise_scheduler = True if 'nvfp4' in model_args.model_name.lower() else False

# Line 73-77: Load PRE-QUANTIZED model
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_args.model_name,  # Already NVFP4!
    ...
)

# Line 88-98: Enable AQN noise injection
trainer = GRPOTrainer(
    ...
    sigma_start = model_args.sigma_start,   # 0.01
    sigma_end = model_args.sigma_end,       # 0.0001
    num_stages = model_args.num_stages,     # 10
    noise_scheduler = noise_scheduler       # True for NVFP4
)
```

**QeRL W4A4 Training**:
1. Pre-quantize model to NVFP4 using llm-compressor (one-time)
2. Load pre-quantized model (weights already W4)
3. Apply AQN noise injection during training
4. Only LoRA weights trained (in BF16)

**Our Approach (WRONG)**:
1. Load BF16 model
2. Fake-quantize weights AND activations every forward pass
3. No AQN or AQN without pre-quantized weights

### Why Our Approach Fails

The "fake quantization every forward pass" approach has fundamental issues:

1. **Error Accumulation**: Every forward pass injects quantization error (1-21%)
2. **Gradient Confusion**: Model receives different quantized values each forward
3. **No Stable Reference**: Model can't learn to compensate for consistent quantization patterns

### What QeRL Actually Does

1. **Pre-quantized Weights**: Model weights are already in W4 format (stable)
2. **AQN Noise**: Adds learnable noise to help model adapt to quantization
3. **LoRA Only**: Only LoRA weights (BF16) are trained, base weights frozen

This is NOT traditional QAT (fake quantization). It's:
- **Inference in W4A4** (using pre-quantized model)
- **Training only LoRA** (in BF16)
- **AQN noise injection** (to help adaptation)

---

### E13e-nvfp4: Exclude base_layer from quantization

**Bug found**: We were quantizing both LoRA wrapper (q_proj) and underlying base_layer (q_proj.base_layer), causing duplicate hooks.

**Fix**: Added 'base_layer' to exclusion list. Hook count reduced from 364 to 182.

**Results**:
- Step 0: val-core/openai/gsm8k/acc/mean@1 = **7.66%**
- Status: FAILED (actually worse than E13d!)

---

## Summary of All E13 Experiments

| ID | Description | Hook Count | Step 0 Acc | Status |
|----|-------------|------------|------------|--------|
| E13a-mxfp4 | MXFP4 POST-hook | ~364 | 8.11% | FAILED |
| E13a-nvfp4 | NVFP4 POST-hook large batch | ~364 | 7.58% | FAILED |
| E13b-nvfp4 | NVFP4 POST-hook small batch | ~364 | 7.81% | FAILED |
| E13c-nvfp4 | NVFP4 POST-hook training-only | ~364 | 7.58% | FAILED |
| E13d-nvfp4 | NVFP4 PRE-hook | ~364 | 8.49% | FAILED |
| E13e-nvfp4 | NVFP4 PRE-hook + exclude base_layer | 182 | 7.66% | FAILED |

All experiments give ~7-8.5% accuracy, far below expected 60%.

---

### E13f-nvfp4: Column-wise blocking (matching quant_compute)

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

**Blocking Direction Comparison**:
```
Our row-wise:    Flatten tensor → reshape to (N, 16) → quantize rows
Reference:       For each column → take 16 rows → quantize column slice

Test result: Column-wise diff from reference = 0.005 (16x better than row-wise 0.085)
```

**Hypothesis**: The blocking direction mismatch was causing incorrect quantization patterns.

**Results**:
- Step 0: val-core/openai/gsm8k/acc/mean@1 = **7.51%**
- Status: **FAILED**

**Conclusion**: Column-wise blocking did NOT fix the issue. Both row-wise and column-wise give ~7-8% accuracy.

---

## Summary of All E13 Experiments (Updated)

| ID | Description | Hook Type | Blocking | Step 0 Acc | Status |
|----|-------------|-----------|----------|------------|--------|
| E13a-mxfp4 | MXFP4 POST-hook | POST | row-wise | 8.11% | FAILED |
| E13a-nvfp4 | NVFP4 POST-hook large batch | POST | row-wise | 7.58% | FAILED |
| E13b-nvfp4 | NVFP4 POST-hook small batch | POST | row-wise | 7.81% | FAILED |
| E13c-nvfp4 | NVFP4 POST-hook training-only | POST | row-wise | 7.58% | FAILED |
| E13d-nvfp4 | NVFP4 PRE-hook | PRE | row-wise | 8.49% | FAILED |
| E13e-nvfp4 | NVFP4 PRE-hook + exclude base_layer | PRE | row-wise | 7.66% | FAILED |
| E13f-nvfp4 | NVFP4 PRE-hook + column-wise blocking | PRE | **column-wise** | 7.51% | FAILED |

---

## Files Modified/Created

| File | Description |
|------|-------------|
| `verl/utils/hw_error_injection.py` | Changed W4A4 activation quantization to PRE-hook |
| `verl/utils/nvfp4_quant.py` | Added `nvfp4_quantize_columnwise()` function |
| `scripts/test_nvfp4_w4a4_training_only.sh` | E13c test script |
| `scripts/test_nvfp4_w4a4_true_prehook.sh` | E13d test script |
| `scripts/test_nvfp4_w4a4_exclude_baselayer.sh` | E13e test script |
| `scripts/test_nvfp4_w4a4_columnwise_e13f.sh` | E13f test script |

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
| Blocking direction (row vs column) | E13e vs E13f | Both fail ~7-8% |

**All experiments give ~7.5-8.5% accuracy, far below expected 60%.**

---

## Remaining Investigation Directions

1. **Scale computation**: Compare our E4M3 scale computation with quant_compute reference
2. **FP4 value mapping**: Verify our FP4 LUT matches the reference exactly
3. **Inference vs Training mismatch**: Our rollout uses vLLM (no quant hooks), but training uses FSDP with hooks
4. **Get more details from colleague**: What exact configuration did they use to reproduce 60%?
5. **Try completely disabling activation quant**: Test W4A16 only to isolate weight vs activation issue

---

## Git Commits

- `df828442` - "fix: W4A4 activation quantization should use POST-hook, not PRE-hook" (THE BUG!)
- `b087831e` - "fix: W4A4 must use PRE-hook for INPUT activation quantization (QeRL-style)" (REVERT)
- `edbf21e6` - "feat: add column-wise NVFP4 blocking to match quant_compute reference"
