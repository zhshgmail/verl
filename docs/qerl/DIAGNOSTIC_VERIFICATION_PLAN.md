# AQN Diagnostic Verification Plan (A100)

**Date**: 2026-01-06
**Target**: Verify diagnostic feasibility on A100 GPU
**Model**: Qwen2.5-1.5B-Instruct (E8c checkpoint)

---

## 1. Implementation Status Analysis

### 1.1 What's IMPLEMENTED in `noisy_ops.py`

| Feature | API | Status |
|---------|-----|--------|
| Selective layer injection | `set_selective_layers([0, 5, 10])` | ✅ Ready |
| Layer hook registration | `register_layer_hooks(model)` | ✅ Ready |
| Per-layer statistics | `get_layer_injection_stats()` | ✅ Ready |
| Forward/backward phase control | `set_noise_phases(forward, backward)` | ✅ Ready |
| Core operators | matmul, bmm, linear | ✅ Always active |
| Extended operators | softmax, silu, gelu, layer_norm | ✅ Via `all_ops_mode=True` |

### 1.2 What's NOT IMPLEMENTED (documented but missing)

| Feature | Documented API | Status |
|---------|---------------|--------|
| Operator type filtering | `set_selective_operators(['softmax'])` | ❌ Missing |
| Integrated diagnostic class | `IntegratedDiagnostic()` | ❌ Missing |
| Activation capture | `ActivationCapture()` | ❌ Missing |
| Sliding window utility | `set_layer_window(start, end)` | ❌ Missing |
| Diagnostic scripts | `scripts/diagnose_model.py` | ❌ Missing |
| Channel-level analysis | Channel sensitivity profiling | ❌ Missing |

---

## 2. Verification Plan: Phase 1 (What We Can Test NOW)

### 2.1 Test 1: Layer Hook Registration

**Goal**: Verify `register_layer_hooks()` correctly tracks layer IDs during forward pass.

```python
# Test script: test_layer_hooks.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from verl.utils.noisy_ops import (
    register_layer_hooks,
    get_current_layer,
    set_selective_layers,
    enable_noisy_ops,
    get_layer_injection_stats,
    reset_layer_injection_stats
)

# Load model
model_path = "/home/z00637938/workspace/verl/checkpoints/noisy_ops_e8c_forward_only/e8c_forward_only_5e-2/global_step_116/merged_hf"
tokenizer_path = "/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Register hooks
num_hooks = register_layer_hooks(model)
print(f"Registered {num_hooks} layer hooks")

# Test: Run forward pass and check layer tracking
enable_noisy_ops(error_scale=0.05)
reset_layer_injection_stats()

inputs = tokenizer("What is 2+2?", return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)

# Check per-layer stats
stats = get_layer_injection_stats()
print(f"Per-layer injection stats: {stats}")

# Expected: Should see injections distributed across layers 0-27
```

**Success Criteria**:
- `register_layer_hooks()` returns 28 (number of layers)
- `get_layer_injection_stats()` shows injections in multiple layers

**Estimated Time**: 10 minutes

---

### 2.2 Test 2: Selective Layer Injection

**Goal**: Verify `set_selective_layers()` correctly limits noise to specific layers.

```python
# Test: Inject noise only in layers [0, 10, 20]
set_selective_layers([0, 10, 20])
enable_noisy_ops(error_scale=0.05)
reset_layer_injection_stats()

# Run inference
with torch.no_grad():
    outputs = model(**inputs)

stats = get_layer_injection_stats()
print(f"Selective layer stats: {stats}")

# Verify: Only layers 0, 10, 20 should have injections
for layer_id, counts in stats.items():
    if layer_id not in [0, 10, 20, 'unknown']:
        assert counts['forward'] == 0, f"Layer {layer_id} should not have injections"
```

**Success Criteria**:
- Only specified layers receive noise injection
- Non-specified layers have zero injection count

**Estimated Time**: 10 minutes

---

### 2.3 Test 3: Layer Sensitivity Profiling (Manual)

**Goal**: Profile sensitivity of early, middle, and late layers.

```python
# Test: Measure accuracy drop for different layer groups
import pandas as pd
import re

def evaluate_accuracy(model, tokenizer, test_data, n_samples=50):
    """Simple GSM8K evaluation."""
    correct = 0
    for idx, row in test_data.head(n_samples).iterrows():
        # ... evaluation logic ...
        pass
    return correct / n_samples

# Load test data
test_data = pd.read_parquet("/data/z00637938/gsm8k/test.parquet")

# Baseline (no noise)
disable_noisy_ops()
baseline_acc = evaluate_accuracy(model, tokenizer, test_data)
print(f"Baseline: {baseline_acc:.2%}")

# Test different layer groups
layer_groups = {
    "early": [0, 1, 2, 3, 4],
    "middle": [12, 13, 14, 15, 16],
    "late": [23, 24, 25, 26, 27],
}

results = {}
for group_name, layers in layer_groups.items():
    set_selective_layers(layers)
    enable_noisy_ops(error_scale=0.05)

    acc = evaluate_accuracy(model, tokenizer, test_data)
    degradation = (baseline_acc - acc) / baseline_acc

    results[group_name] = {
        "accuracy": acc,
        "degradation": degradation,
        "layers": layers
    }
    print(f"{group_name}: {acc:.2%} (degradation: {degradation:.1%})")

    disable_noisy_ops()
```

**Success Criteria**:
- Different layer groups show different sensitivity
- Results align with theory (late layers typically more sensitive)

**Estimated Time**: 30 minutes (50 samples x 3 groups)

---

### 2.4 Test 4: Sliding Window Diagnosis (Manual)

**Goal**: Find "avalanche points" where errors compound.

```python
# Sliding window: 3-layer groups
window_size = 3
num_layers = 28
results = {}

for start in range(0, num_layers - window_size + 1, 3):  # Step by 3 for speed
    end = start + window_size
    layers = list(range(start, end))

    set_selective_layers(layers)
    enable_noisy_ops(error_scale=0.05)

    acc = evaluate_accuracy(model, tokenizer, test_data, n_samples=30)
    degradation = (baseline_acc - acc) / baseline_acc

    results[f"L{start}-{end-1}"] = {
        "accuracy": acc,
        "degradation": degradation,
    }
    print(f"L{start}-{end-1}: {acc:.2%} (degradation: {degradation:.1%})")

    disable_noisy_ops()

# Find avalanche point (max degradation)
avalanche_point = max(results.items(), key=lambda x: x[1]["degradation"])
print(f"\nAvalanche Point: {avalanche_point[0]} with {avalanche_point[1]['degradation']:.1%} degradation")
```

**Success Criteria**:
- Identify layer range with highest sensitivity (avalanche point)
- Generate sensitivity profile

**Estimated Time**: 1 hour (10 windows x 30 samples each)

---

### 2.5 Test 5: Operator-Level Diagnosis (Partial)

**Goal**: Test `all_ops_mode` to see which operators are most sensitive.

**Note**: We don't have `set_selective_operators()`, but we can test with/without `all_ops_mode`.

```python
# Test 1: Core ops only (matmul, bmm, linear)
set_selective_layers(None)  # All layers
enable_noisy_ops(error_scale=0.05, all_ops_mode=False)
acc_core = evaluate_accuracy(model, tokenizer, test_data, n_samples=50)
print(f"Core ops only: {acc_core:.2%}")
disable_noisy_ops()

# Test 2: All ops (+ softmax, silu, gelu, layer_norm)
enable_noisy_ops(error_scale=0.05, all_ops_mode=True)
acc_all = evaluate_accuracy(model, tokenizer, test_data, n_samples=50)
print(f"All ops: {acc_all:.2%}")
disable_noisy_ops()

# Compare
print(f"\nCore ops degradation: {(baseline_acc - acc_core) / baseline_acc:.1%}")
print(f"All ops degradation: {(baseline_acc - acc_all) / baseline_acc:.1%}")
print(f"Additional degradation from non-linear ops: {(acc_core - acc_all) / baseline_acc:.1%}")
```

**Success Criteria**:
- Understand contribution of non-linear ops to overall sensitivity
- Determine if `all_ops_mode` significantly increases degradation

**Estimated Time**: 20 minutes

---

## 3. Verification Plan: Phase 2 (Implementation Required)

### 3.1 Implement `set_selective_operators()`

**Goal**: Enable filtering by operator type (matmul-only, softmax-only, etc.)

```python
# Proposed API addition to noisy_ops.py
_SELECTIVE_OPERATORS = None  # None = all, {'matmul', 'softmax'} = specific

def set_selective_operators(op_types: list = None) -> None:
    """
    Enable noise injection only for specific operator types.

    Args:
        op_types: List of operator names. Options:
                  - 'matmul', 'bmm', 'linear' (core ops)
                  - 'softmax', 'silu', 'gelu', 'layer_norm' (extended ops)
                  - None = all operators
    """
    global _SELECTIVE_OPERATORS
    if op_types is None:
        _SELECTIVE_OPERATORS = None
    else:
        _SELECTIVE_OPERATORS = set(op_types)
```

**Estimated Implementation**: 2 hours

---

### 3.2 Create Diagnostic Script

**Goal**: One command to run full layer sensitivity analysis.

```bash
# Usage
python scripts/layer_sensitivity_diagnosis.py \
    --checkpoint /path/to/checkpoint \
    --tokenizer /path/to/tokenizer \
    --test-data /path/to/test.parquet \
    --noise-scale 0.05 \
    --window-size 3 \
    --n-samples 50 \
    --output sensitivity_report.json
```

**Estimated Implementation**: 4 hours

---

### 3.3 Implement Activation Capture (Optional)

**Goal**: Capture activations for channel-level analysis.

```python
class ActivationCapture:
    def __init__(self, model, layer_patterns=['self_attn.o_proj', 'mlp.down_proj']):
        self.activations = {}
        self.hooks = []
        self._register(model, layer_patterns)

    def _register(self, model, patterns):
        for name, module in model.named_modules():
            if any(p in name for p in patterns):
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

    def _make_hook(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook

    def detect_outliers(self, threshold=10.0):
        """Detect channels with magnitude > threshold * median."""
        outliers = {}
        for name, act in self.activations.items():
            channel_magnitudes = act.abs().mean(dim=(0, 1))  # [hidden_dim]
            median = channel_magnitudes.median()
            ratios = channel_magnitudes / (median + 1e-8)
            outlier_mask = ratios > threshold
            if outlier_mask.any():
                outliers[name] = {
                    'num_outliers': outlier_mask.sum().item(),
                    'max_ratio': ratios.max().item(),
                    'outlier_channels': outlier_mask.nonzero().squeeze().tolist()
                }
        return outliers
```

**Estimated Implementation**: 4 hours

---

## 4. Execution Commands (A100)

### 4.1 Quick Verification (30 min)

```bash
# SSH to A100 server
ssh root@90.90.102.18

# Enter container
docker exec -it verl-r3-test bash

# Navigate to workspace
cd /home/z00637938/workspace/verl

# Run quick verification
python -c "
from verl.utils.noisy_ops import *
import torch
from transformers import AutoModelForCausalLM

# Test 1: Import works
print('Import OK')

# Test 2: Enable/disable works
enable_noisy_ops(error_scale=0.05)
print(f'Enabled: {get_injection_stats()}')
disable_noisy_ops()
print('Disable OK')

# Test 3: Selective layers works
set_selective_layers([0, 5, 10])
print(f'Selective layers: {get_selective_layers()}')
set_selective_layers(None)
print('Selective layers reset OK')

print('\\nAll basic tests PASSED')
"
```

### 4.2 Full Layer Sensitivity Test (2 hours)

```bash
# Create test script
cat > /tmp/layer_sensitivity_test.py << 'EOF'
#!/usr/bin/env python3
"""Layer sensitivity diagnosis for E8c checkpoint."""

import torch
import pandas as pd
import re
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from verl.utils.noisy_ops import (
    enable_noisy_ops, disable_noisy_ops,
    set_selective_layers, register_layer_hooks,
    get_layer_injection_stats, reset_layer_injection_stats
)

# Config
MODEL_PATH = "/home/z00637938/workspace/verl/checkpoints/noisy_ops_e8c_forward_only/e8c_forward_only_5e-2/global_step_116/merged_hf"
TOKENIZER_PATH = "/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
TEST_DATA_PATH = "/data/z00637938/gsm8k/test.parquet"
N_SAMPLES = 50
NOISE_SCALE = 0.05
WINDOW_SIZE = 3

def evaluate(model, tokenizer, test_data, n_samples):
    """Simple GSM8K evaluation."""
    correct = 0
    for idx, row in test_data.head(n_samples).iterrows():
        prompt_data = row['prompt']
        if hasattr(prompt_data, 'tolist'):
            prompt_data = prompt_data.tolist()
        if isinstance(prompt_data, list) and len(prompt_data) > 0:
            user_msg = prompt_data[0]
            prompt = user_msg.get('content', '') if isinstance(user_msg, dict) else str(user_msg)
        else:
            prompt = str(prompt_data)

        formatted_prompt = f'<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n'

        reward_model = row.get('reward_model', {})
        gt = str(reward_model.get('ground_truth', '')) if isinstance(reward_model, dict) else ''

        inputs = tokenizer(formatted_prompt, return_tensors='pt', padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id)
        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        numbers = re.findall(r'[-+]?\d+(?:\.\d+)?', generated.replace(',', ''))
        pred = numbers[-1] if numbers else ''

        try:
            if abs(float(pred) - float(gt)) < 0.01:
                correct += 1
        except:
            if pred.strip() == gt.strip():
                correct += 1

    return correct / n_samples

def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Register hooks
    num_hooks = register_layer_hooks(model)
    print(f"Registered {num_hooks} layer hooks")

    # Load test data
    test_data = pd.read_parquet(TEST_DATA_PATH)

    # Baseline
    print("\n=== Baseline (no noise) ===")
    disable_noisy_ops()
    baseline_acc = evaluate(model, tokenizer, test_data, N_SAMPLES)
    print(f"Baseline accuracy: {baseline_acc:.2%}")

    # Sliding window analysis
    print(f"\n=== Sliding Window Analysis (window={WINDOW_SIZE}) ===")
    num_layers = 28
    results = {}

    for start in range(0, num_layers - WINDOW_SIZE + 1, WINDOW_SIZE):
        end = start + WINDOW_SIZE
        layers = list(range(start, end))

        set_selective_layers(layers)
        enable_noisy_ops(error_scale=NOISE_SCALE)
        reset_layer_injection_stats()

        acc = evaluate(model, tokenizer, test_data, N_SAMPLES)
        degradation = (baseline_acc - acc) / baseline_acc if baseline_acc > 0 else 0

        results[f"L{start}-{end-1}"] = {
            "layers": layers,
            "accuracy": acc,
            "degradation": degradation,
            "injection_stats": get_layer_injection_stats()
        }
        print(f"L{start:02d}-{end-1:02d}: {acc:.2%} (degradation: {degradation:+.1%})")

        disable_noisy_ops()

    # Find avalanche point
    avalanche = max(results.items(), key=lambda x: x[1]["degradation"])
    print(f"\n=== Results ===")
    print(f"Avalanche Point: {avalanche[0]} with {avalanche[1]['degradation']:.1%} degradation")

    # Save results
    output = {
        "config": {
            "model": MODEL_PATH,
            "noise_scale": NOISE_SCALE,
            "window_size": WINDOW_SIZE,
            "n_samples": N_SAMPLES
        },
        "baseline_accuracy": baseline_acc,
        "layer_sensitivity": results,
        "avalanche_point": avalanche[0]
    }

    with open("/tmp/layer_sensitivity_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("\nResults saved to /tmp/layer_sensitivity_results.json")

if __name__ == "__main__":
    main()
EOF

# Copy and run
docker cp /tmp/layer_sensitivity_test.py verl-r3-test:/tmp/
docker exec verl-r3-test python /tmp/layer_sensitivity_test.py
```

---

## 5. Expected Results

### 5.1 Layer Sensitivity Pattern (Hypothesis)

Based on transformer architecture theory:

| Layer Range | Expected Sensitivity | Reasoning |
|-------------|---------------------|-----------|
| L0-2 (Early) | Medium (5-8%) | Feature extraction, some redundancy |
| L3-8 | Low (3-5%) | Building abstractions |
| L9-14 (Middle) | **HIGH (10-15%)** | Potential bottleneck |
| L15-20 | Low-Medium (4-7%) | High redundancy |
| L21-27 (Late) | Medium-High (6-10%) | Output formation |

### 5.2 Success Metrics

| Test | Success Criteria |
|------|------------------|
| Hook registration | Returns 28 hooks |
| Selective layers | Only specified layers get injections |
| Layer sensitivity | Different groups show different degradation |
| Sliding window | Identifies avalanche point |
| Operator comparison | All-ops shows more degradation than core-ops |

---

## 6. Timeline

| Phase | Task | Time |
|-------|------|------|
| Phase 1.1 | Hook registration test | 10 min |
| Phase 1.2 | Selective layer test | 10 min |
| Phase 1.3 | Layer group sensitivity | 30 min |
| Phase 1.4 | Sliding window diagnosis | 60 min |
| Phase 1.5 | Operator comparison | 20 min |
| **Total Phase 1** | | **~2 hours** |
| Phase 2.1 | Implement `set_selective_operators()` | 2 hours |
| Phase 2.2 | Create diagnostic script | 4 hours |
| Phase 2.3 | Implement ActivationCapture | 4 hours |
| **Total Phase 2** | | **~10 hours** |

---

## 7. Next Steps After Verification

1. **If Phase 1 succeeds**: Proceed to Phase 2 implementation
2. **Generate sensitivity heatmap**: Visualize layer sensitivity
3. **Cross-reference with E5b/E8c results**: Validate theory
4. **Publish findings**: Update AQN_DIAGNOSTIC_PROBE.md with real results

---

**Document Status**: Ready for execution
**Last Updated**: 2026-01-06
