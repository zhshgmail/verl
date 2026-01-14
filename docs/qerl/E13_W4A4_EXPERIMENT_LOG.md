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

## Next Steps

1. **Option A**: Pre-quantize model using llm-compressor, then train with AQN
2. **Option B**: Modify our fake quantization to be more like QeRL's approach
3. **Option C**: Abandon W4A4 and focus on W4A16 (proven to work)

---

## Files Modified/Created

| File | Description |
|------|-------------|
| `verl/utils/hw_error_injection.py` | Changed W4A4 activation quantization to PRE-hook |
| `scripts/test_nvfp4_w4a4_training_only.sh` | E13c test script |
| `scripts/test_nvfp4_w4a4_true_prehook.sh` | E13d test script |

---

## Git Commits

- `df828442` - "fix: W4A4 activation quantization should use POST-hook, not PRE-hook" (THE BUG!)
- `b087831e` - "fix: W4A4 must use PRE-hook for INPUT activation quantization (QeRL-style)" (REVERT)
