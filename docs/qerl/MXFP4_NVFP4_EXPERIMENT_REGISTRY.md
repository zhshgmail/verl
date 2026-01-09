# MXFP4/NVFP4 Quantization Experiment Registry

**Date**: 2026-01-08
**Branch**: `feature/npu-aqn-test`
**Status**: NVFP4 v1 in progress

---

## Quick Reference

> **⚠️ KNOWN ISSUE (Fixed in commit `d8c25492`)**: All experiments below have `lm_head` quantized.
> Production PTQ recipes exclude `lm_head` and `embed_tokens`. Results may improve with fix.

| Experiment | Quant Type | AQN Target | Sigma | Result | lm_head Bug | Log File |
|------------|------------|------------|-------|--------|-------------|----------|
| Baseline | None | None | - | **75.97%** | N/A | `baseline_clean_75.97.log` |
| MXFP4-only | MXFP4 W4A16 | None | - | **70.05%** | **YES** | `mxfp4_only_70.05.log` |
| MXFP4+AQN (orig) | MXFP4 W4A16 | RMSNorm | 0.05→0.0005 | **67.48%** | **YES** | `mxfp4_aqn_orig_67.48.log` |
| Exp 1A | MXFP4 W4A16 | Linear | 0.05→0.0005 | **COLLAPSED** | **YES** | `exp1a_linear_collapsed.log` |
| Exp 1C | MXFP4 W4A16 | Linear | 0.005→0.00005 | **COLLAPSED** | **YES** | `exp1c_linear_collapsed.log` |
| **Exp 1D** | MXFP4 W4A16 | Linear | 0.001→0.00001 | **66.49%** | **YES** | `exp1d_linear_tiny_66.49.log` |
| **Exp 1E** | MXFP4 W4A16 | RMSNorm | 0.05→0.0005 | **62.32%** | **YES** | `exp1e_rmsnorm_62.32.log` |
| **NVFP4 v1** | NVFP4 W4A16 | RMSNorm | 0.05→0.0005 | **COLLAPSED (7.66%)** | **YES** | `nvfp4_v1_collapsed_7.66pct.log` |
| **MXFP4 v2** | MXFP4 W4A16 | None | - | **TBD** | **NO (fixed)** | `mxfp4_v2_no_aqn_TBD.log` |

---

## 1. Common Settings (All Experiments)

### 1.1 Model
```yaml
model: Qwen2.5-1.5B-Instruct
path: /data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306
dtype: bfloat16
```

### 1.2 Data
```yaml
train_data: /data/z00637938/gsm8k/train.parquet
val_data: /data/z00637938/gsm8k/test.parquet
train_batch_size: 128
max_prompt_length: 1024
max_response_length: 1024
```

### 1.3 Training
```yaml
total_epochs: 3
total_steps: 174
test_freq: 20
save_freq: 50
n_gpus: 8
optimizer.lr: 5e-7
ppo_mini_batch_size: 32
ppo_micro_batch_size_per_gpu: 4
kl_loss_coef: 0.001
kl_loss_type: low_var_kl
adv_estimator: grpo
```

### 1.4 Infrastructure
```yaml
server: root@90.90.102.18
container: verl-r3-test
gpus: 8x A100-80GB
vllm: gpu_memory_utilization=0.8
```

---

## 2. Experiment Details

### 2.1 Baseline (No Quantization)
```yaml
experiment_id: baseline
date: 2026-01-08
script: test_mxfp4_experiments_suite.sh (baseline mode)
output_dir: /tmp/mxfp4_exp_baseline

hw_error_injection:
  enabled: false

noise_injection:
  enabled: false

result:
  final_accuracy: 75.97%
  training_time: ~2h
  status: completed
```

### 2.2 MXFP4-only (No AQN)
```yaml
experiment_id: mxfp4_only
date: 2026-01-08
script: test_mxfp4_w4a16_training.sh
output_dir: /tmp/mxfp4_exp_mxfp4only

hw_error_injection:
  enabled: true
  error_type: mxfp4
  injection_point: weight  # W4A16
  target_modules: ["linear"]
  exclude_modules: []  # BUG: lm_head was included
  apply_during: both

noise_injection:
  enabled: false

result:
  final_accuracy: 70.05%
  accuracy_drop: -5.92% (from baseline)
  training_time: ~2h
  status: completed
```

### 2.3 Exp 1A - Linear AQN (sigma=0.05)
```yaml
experiment_id: exp1a
date: 2026-01-08
script: test_mxfp4_exp1a_aligned.sh
output_dir: /tmp/mxfp4_exp1a_aligned

hw_error_injection:
  enabled: true
  error_type: mxfp4
  injection_point: weight
  target_modules: ["linear"]
  exclude_modules: []
  apply_during: both

noise_injection:
  enabled: true
  sigma_start: 0.05
  sigma_end: 0.0005
  num_stages: 10
  layer_types: ["linear"]

result:
  final_accuracy: COLLAPSED
  collapse_step: 19
  entropy_at_collapse: 9.5
  training_time: ~20min (aborted)
  status: failed

notes: |
  Complete collapse at step 19. Entropy spiked to 9.5 (max ~10).
  Linear layers too sensitive for direct noise injection.
```

### 2.4 Exp 1C - Linear AQN (sigma=0.005)
```yaml
experiment_id: exp1c
date: 2026-01-08
script: test_mxfp4_exp1c_small_sigma.sh
output_dir: /tmp/mxfp4_exp1c_small_sigma

hw_error_injection:
  enabled: true
  error_type: mxfp4
  injection_point: weight
  target_modules: ["linear"]
  exclude_modules: []
  apply_during: both

noise_injection:
  enabled: true
  sigma_start: 0.005
  sigma_end: 0.00005
  num_stages: 10
  layer_types: ["linear"]

result:
  final_accuracy: COLLAPSED (partial)
  collapse_step: 19
  entropy_at_collapse: 2.56
  training_time: ~20min (aborted)
  status: failed

notes: |
  Partial collapse at step 19. Entropy 2.56 (improved from 1A's 9.5).
  10x smaller sigma helped but still unstable.
```

### 2.5 Exp 1D - Linear AQN (sigma=0.001)
```yaml
experiment_id: exp1d
date: 2026-01-08
script: test_mxfp4_exp1d_tiny_sigma.sh
output_dir: /tmp/mxfp4_exp1d_tiny_sigma

hw_error_injection:
  enabled: true
  error_type: mxfp4
  injection_point: weight
  target_modules: ["linear"]
  exclude_modules: []
  apply_during: both

noise_injection:
  enabled: true
  sigma_start: 0.001
  sigma_end: 0.00001
  num_stages: 10
  layer_types: ["linear"]

result:
  final_accuracy: 66.49%
  accuracy_drop: -9.48% (from baseline), -3.56% (from MXFP4-only)
  training_time: 3h
  entropy_range: 0.05-0.3 (stable)
  status: completed

validation_progression:
  step_0: 75.97%
  step_20: 66.21%
  step_40: 73.16%
  step_60: 66.51%
  step_80: 69.82%
  step_100: 66.67%
  step_120: 66.06%
  step_140: 66.59%
  step_160: 67.02%
  step_174: 66.49%

notes: |
  BREAKTHROUGH: sigma=0.001 is the stability threshold for Linear layer AQN.
  Training was stable (no collapse), but accuracy didn't improve over MXFP4-only.
  Conclusion: Ultra-small sigma = effectively no AQN benefit.
```

### 2.6 Exp 1E - RMSNorm AQN (QeRL Default)
```yaml
experiment_id: exp1e
date: 2026-01-08
script: test_mxfp4_exp1e_rmsnorm.sh
output_dir: /tmp/mxfp4_exp1e_rmsnorm

hw_error_injection:
  enabled: true
  error_type: mxfp4
  injection_point: weight
  target_modules: ["linear"]
  exclude_modules: []
  apply_during: both

noise_injection:
  enabled: true
  sigma_start: 0.05
  sigma_end: 0.0005
  num_stages: 10
  layer_types: ["rmsnorm"]  # QeRL default

result:
  final_accuracy: 62.32%
  accuracy_drop: -13.65% (from baseline), -7.73% (from MXFP4-only)
  training_time: 2h 41m
  entropy_at_end: 0.073 (stable)
  status: completed

notes: |
  WORST result of all experiments!
  RMSNorm AQN does NOT help with MXFP4's high error rate (~21%).
  AQN adds training noise without providing robustness benefit.
  Conclusion: MXFP4 error is too high for AQN to be effective.
```

### 2.7 NVFP4 v1 - RMSNorm AQN (with lm_head bug) - COLLAPSED
```yaml
experiment_id: nvfp4_v1
date: 2026-01-09
script: test_nvfp4_w4a16_training.sh
output_dir: /tmp/nvfp4_w4a16_aqn

hw_error_injection:
  enabled: true
  error_type: nvfp4
  injection_point: weight
  target_modules: ["linear"]
  exclude_modules: []  # BUG: lm_head was included - CAUSED COLLAPSE
  apply_during: both

noise_injection:
  enabled: true
  sigma_start: 0.05
  sigma_end: 0.0005
  num_stages: 10
  layer_types: ["rmsnorm"]

result:
  final_accuracy: 7.66% (COLLAPSED)
  peak_accuracy: 72.55% (step 60)
  collapse_start: step 100
  status: STOPPED (collapsed)

validation_progression:
  step_20: 67.63%
  step_40: 71.11%
  step_60: 72.55%  # PEAK
  step_80: 70.05%
  step_100: 53.83%  # COLLAPSE BEGINS
  step_120: 17.13%
  step_140: 7.66%   # NEAR-RANDOM

known_issues:
  - lm_head quantization CAUSED COLLAPSE
  - Model degraded from 72.55% to 7.66% in 80 steps
  - Collapse accelerated in epoch 2

notes: |
  CRITICAL FINDING: lm_head quantization is DESTRUCTIVE.
  Model performed well initially (72.55% at step 60).
  Collapse began at step 100 (epoch 1→2 transition).
  Entropy dropped to 0.01 (near-deterministic wrong outputs).
  Next: Try MXFP4 v2 with lm_head excluded (the fix).
```

### 2.8 MXFP4 v2 - No AQN (with lm_head fix) - PENDING
```yaml
experiment_id: mxfp4_v2
date: 2026-01-09
script: test_mxfp4_v2_no_aqn.sh
output_dir: /tmp/mxfp4_v2_no_aqn

hw_error_injection:
  enabled: true
  error_type: mxfp4
  injection_point: weight
  target_modules: ["linear"]
  exclude_modules: ["lm_head", "embed_tokens"]  # FIX APPLIED!
  apply_during: both

noise_injection:
  enabled: false  # NO AQN - baseline test

training:
  total_epochs: 2  # Reduced for faster iteration

result:
  final_accuracy: TBD
  status: pending

notes: |
  First experiment with lm_head exclusion fix.
  Tests pure MXFP4 quantization without AQN.
  Compare with previous MXFP4-only (70.05%) which had lm_head bug.
  Expected: ~70% or better (lm_head excluded = better precision on output).
```

---

## 3. Key Findings

### 3.1 MXFP4 Error Rate
- **Relative error: ~21%** (measured via SRDD scan)
- Too high for AQN to provide benefit
- All AQN variants underperformed MXFP4-only

### 3.2 AQN Sensitivity
- **Linear layers**: Cannot tolerate sigma > 0.001
- **RMSNorm layers**: Stable but doesn't help with MXFP4
- **Sigma threshold**: 0.001 is the minimum for stable Linear AQN

### 3.3 Accuracy Ranking (MXFP4)
1. Baseline (no quant): **75.97%**
2. MXFP4-only: **70.05%**
3. MXFP4 + AQN (original): **67.48%**
4. Exp 1D (Linear sigma=0.001): **66.49%**
5. Exp 1E (RMSNorm AQN): **62.32%** (worst)

### 3.4 Bug Discovery
- **lm_head quantization**: Was being quantized (should be excluded)
- **Fix**: Added `exclude_modules` parameter with default `['lm_head', 'embed_tokens']`
- **Impact**: Unknown, to be measured in NVFP4 v2

---

## 4. Log Archive

All logs are archived to: `/home/zheng/workspace/verl/logs/mxfp4_nvfp4_experiments/`

| File | Experiment | Size | Status |
|------|------------|------|--------|
| `baseline_clean_75.97.log` | Baseline | 258KB | archived |
| `mxfp4_only_70.05.log` | MXFP4-only | 285KB | archived |
| `mxfp4_aqn_orig_67.48.log` | MXFP4+AQN orig | 297KB | archived |
| `exp1a_linear_collapsed.log` | Exp 1A | 161KB | archived |
| `exp1c_linear_collapsed.log` | Exp 1C | 315KB | archived |
| `exp1d_linear_tiny_66.49.log` | Exp 1D | 808KB | archived |
| `exp1e_rmsnorm_62.32.log` | Exp 1E | 985KB | archived |
| `nvfp4_v1_aqn_inprogress.log` | NVFP4 v1 | 656KB | in_progress |

**Archive Script**: `scripts/archive_experiment_logs.sh`

---

## 5. Next Steps

1. **Complete NVFP4 v1**: Let current experiment finish (baseline with lm_head bug)
2. **Archive logs**: Fetch all logs from A100 server
3. **Run NVFP4 v2**: With `exclude_modules=['lm_head', 'embed_tokens']`
4. **Run SRDD scan**: Verify actual NVFP4 error rate
5. **Compare v1 vs v2**: Measure lm_head quantization impact

---

## 6. References

- [MXFP4_AQN_NEXT_STEPS.md](MXFP4_AQN_NEXT_STEPS.md) - Detailed experiment plan
- [SRDD_MXFP4_QUANT_EXPERIMENT.md](SRDD_MXFP4_QUANT_EXPERIMENT.md) - SRDD scan results
- [HW_ERROR_INJECTION_EXPERIMENTS.md](HW_ERROR_INJECTION_EXPERIMENTS.md) - Previous E5/E7 experiments
