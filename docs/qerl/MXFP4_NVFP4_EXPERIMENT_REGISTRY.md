# MXFP4/NVFP4 Quantization Experiment Registry

**Date**: 2026-01-09
**Branch**: `feature/npu-aqn-test`
**Status**: E2b completed (68.84%) - Infrastructure fixes needed before further experiments

---

## ⚠️ CRITICAL PROJECT CONTEXT

### Final Goal
**Find proper MXFP4 W4A16 (or W4Axx) training recipe for Ascend NPU**

### Key Constraints
1. **GPU/NVFP4 is NOT a solution** - only validation/comparison tool
2. **Ascend NPU uses MXFP4** - we cannot switch formats in production
3. **HW Error Noise rationale**: Hardware heterogeneous differences (GPU vs NPU, different chip batches) are treated as error noise
4. **NVFP4 experiments**: Only to validate AQN approach works, NOT as alternative to MXFP4

### Current Blocker: Reward Hacking
All experiments show **response length explosion** in epoch 2:
- Step 40: 215 tokens, 73.24% accuracy
- Step 100: 410 tokens, 66.34% accuracy (+91% length, -7% accuracy)

**Entropy collapse** (0.27 → 0.11) indicates model exploits verbosity, not reasoning.

### Required Infrastructure Fixes (Before More Experiments)
1. **Use DAPO instead of PPO/GRPO** - Built-in overlong penalty + dynamic sampling
2. **Keep 1 epoch** - DAPO doesn't solve entropy collapse, 21% MXFP4 error compounds
3. **Response length penalty** - DAPO's overlong_buffer handles this
4. **Quantize Reference model** - Actor/Ref consistency for KL divergence (future)

### DAPO Analysis (2026-01-09)

**Key Finding**: DAPO prevents **length explosion** but NOT **entropy collapse**.

| DAPO Feature | Length Hack | Entropy Collapse |
|--------------|-------------|------------------|
| Overlong Buffer Penalty | ✅ Fixes | ❌ No effect |
| Asymmetric Clipping | ⚠️ Partial | ❌ No effect |
| Token-level Loss | ✅ Fixes | ❌ No effect |
| Dynamic Sampling | ✅ Helps | ❌ No effect |

**Recommendation**: Use DAPO with **1 epoch** (not 2). DAPO paper uses 1 epoch for all experiments.
MXFP4's 21% error compounds over epochs regardless of DAPO protections.

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
| **MXFP4 v2** | MXFP4 W4A16 | None | - | **65.96%** | **NO (fixed)** | `mxfp4_v2_no_aqn_65.96.log` |

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

### 2.8 MXFP4 v2 - No AQN (with lm_head fix) - COMPLETED
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
  exclude_modules: ["lm_head", "embed_tokens"]  # FIX APPLIED AND VERIFIED!
  apply_during: both

noise_injection:
  enabled: false  # NO AQN - baseline test

training:
  total_epochs: 2
  total_steps: 116

result:
  final_accuracy: 65.96%
  peak_accuracy: 73.16% (step 40)
  accuracy_drop: -10.01% (from baseline), -4.09% (from MXFP4-only with bug)
  training_time: 1h 40m
  entropy_range: 0.09-0.37 (stable)
  status: completed

validation_progression:
  step_0: 8.57%   # VERY LOW - MXFP4 at inference degrades model significantly
  step_20: 71.49%
  step_40: 73.16%  # PEAK
  step_60: 69.98%
  step_80: 64.14%
  step_100: 66.26%
  step_116: 65.96%  # FINAL

known_issues:
  - step_0 = 8.57% is concerning (MXFP4 applied during initial validation)
  - Model recovers to 73.16% (step 40) but then degrades
  - Final accuracy (65.96%) is LOWER than MXFP4-only with lm_head bug (70.05%)

notes: |
  UNEXPECTED RESULT: lm_head exclusion did NOT improve accuracy!
  - MXFP4-only (with lm_head bug): 70.05%
  - MXFP4 v2 (lm_head excluded): 65.96%

  Possible explanations:
  1. 2 epochs vs 3 epochs (less training time)
  2. MXFP4 at inference (step 0 = 8.57%) indicates severe degradation
  3. The lm_head exclusion may help inference but not training

  The step_0 = 8.57% is the key finding: MXFP4 quantization during
  validation severely degrades model performance before any training.
  Training recovers but doesn't reach baseline levels.
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
2. MXFP4-only (lm_head bug, 3ep): **70.05%**
3. MXFP4 + AQN (original): **67.48%**
4. Exp 1D (Linear sigma=0.001): **66.49%**
5. **MXFP4 v2 (lm_head fixed, 2ep): 65.96%**
6. Exp 1E (RMSNorm AQN): **62.32%** (worst)

### 3.5 lm_head Fix Impact
- **Result**: lm_head exclusion did NOT improve accuracy
- **MXFP4-only (with bug)**: 70.05% (3 epochs)
- **MXFP4 v2 (fixed)**: 65.96% (2 epochs)
- **Caveat**: Different epoch counts make direct comparison difficult
- **Key observation**: Step 0 = 8.57% shows MXFP4 at inference is destructive

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
| `nvfp4_v1_collapsed_7.66pct.log` | NVFP4 v1 | 656KB | archived |
| `mxfp4_v2_no_aqn_65.96.log` | MXFP4 v2 | ~600KB | archived |

**Archive Script**: `scripts/archive_experiment_logs.sh`

---

## 5. v2.x Series Experiment Plan (lm_head fix applied)

All experiments use 2 epochs and have `exclude_modules=['lm_head', 'embed_tokens']`.

**⚠️ PAUSED**: Reward hacking confounds results. Need infrastructure fixes first.

| ID | Quant | AQN Type | AQN Sigma | Script | Status | Notes |
|----|-------|----------|-----------|--------|--------|-------|
| **E2a (v2.0)** | MXFP4 | None | - | `test_mxfp4_v2_no_aqn.sh` | **65.96%** | Peak 73.16% @ step 40 |
| **E2b (v2.1)** | MXFP4 | RMSNorm (QeRL) | 0.05→0.0005 | `test_mxfp4_v2.1_rmsnorm_aqn.sh` | **68.84%** | Peak 73.24% @ step 40 |
| **E2c (v2.2)** | MXFP4 | Linear | 0.001→0.00001 | `test_mxfp4_v2.2_linear_aqn.sh` | PAUSED | Wait for infra fix |
| **E2d (v2.3)** | NVFP4 | None | - | `test_nvfp4_v2.3_baseline.sh` | PAUSED | **VALIDATION ONLY** |
| **E2e (v2.4)** | NVFP4 | RMSNorm (QeRL) | 0.05→0.0005 | `test_nvfp4_v2.4_rmsnorm_aqn.sh` | PAUSED | **VALIDATION ONLY** |

### E2a vs E2b Comparison (Confounded by Reward Hack)

| Metric | E2a (no AQN) | E2b (RMSNorm AQN) |
|--------|--------------|-------------------|
| Peak (step 40) | 73.16% | 73.24% |
| Final (step 116) | 65.96% | 68.84% |
| Decline from peak | -7.20% | -4.40% |
| Response length @ 100 | ~400 tokens | ~410 tokens |
| Entropy @ 100 | ~0.12 | ~0.11 |

**Observation**: Both experiments show same reward hacking pattern. AQN may help slightly (2.88% better final), but confounded by epoch-2 verbosity exploitation.

### Execution Order (REVISED):
1. ~~v2.1~~ ✓ E2b completed (68.84%)
2. **v3.0 (E3a)** - DAPO + MXFP4, 1 epoch (IN PROGRESS)
3. If v3.0 successful, continue with v3.1 (DAPO + MXFP4 + AQN)
4. NVFP4 experiments only for validation (NOT solution)

---

## 6. v3.x Series: DAPO-based Experiments

All v3.x experiments use DAPO algorithm with 1 epoch.

| ID | Quant | Algorithm | Overlong Penalty | Script | Status |
|----|-------|-----------|------------------|--------|--------|
| **E3a (v3.0)** | MXFP4 | DAPO | buffer=256, penalty=0.5 | `test_mxfp4_v3.0_dapo.sh` | RUNNING (started 2026-01-09 02:27 UTC) |

### v3.0 Verified Configuration (from training log):
```
[RayPPOTrainer] HW error injection initialized: scale=1e-05, type=mxfp4, point=weight, targets=['linear']
[HW Error] Registered 196 hooks on actor model: exclude=['lm_head', 'embed_tokens']
clip_ratio_high: 0.25
overlong_buffer: {'enable': True, 'len': 256, 'penalty_factor': 0.5}
filter_groups: enabled
```

### v3.0 Key Settings:
```yaml
# DAPO overlong penalty (prevents length explosion)
reward_model.overlong_buffer.enable: True
reward_model.overlong_buffer.len: 256
reward_model.overlong_buffer.penalty_factor: 0.5

# Asymmetric clipping (reduced for MXFP4 stability)
actor.clip_ratio_low: 0.2
actor.clip_ratio_high: 0.25  # DAPO default is 0.28

# Token-level loss
actor.loss_agg_mode: "token-mean"

# Dynamic sampling
algorithm.filter_groups.enable: True
algorithm.filter_groups.metric: acc

# Training
trainer.total_epochs: 1  # DAPO standard
```

### Expected Outcomes:
- No response length explosion (overlong penalty)
- Final accuracy closer to peak (~73%) vs E2a/E2b decline
- Stable entropy (less collapse)

### Monitoring Commands:
```bash
# Check training progress
ssh root@90.90.102.18 "docker exec verl-r3-test bash -c 'grep -E \"Training Progress|step:.*val-core\" /tmp/mxfp4_v3.0_dapo/training.log | tail -10'"

# Check response length (should stay ~200-250, not 400+)
ssh root@90.90.102.18 "docker exec verl-r3-test bash -c 'grep \"response_length/mean\" /tmp/mxfp4_v3.0_dapo/training.log | tail -5'"

# Check entropy (should stay ~0.25+, not collapse to 0.11)
ssh root@90.90.102.18 "docker exec verl-r3-test bash -c 'grep \"actor/entropy\" /tmp/mxfp4_v3.0_dapo/training.log | tail -5'"

# Check overlong penalty activation
ssh root@90.90.102.18 "docker exec verl-r3-test bash -c 'grep \"overlong\" /tmp/mxfp4_v3.0_dapo/training.log | tail -5'"
```

### Quick Start Commands:
```bash
ssh root@90.90.102.18
docker exec -it verl-r3-test bash
cd /home/z00637938/workspace/verl
git pull personal feature/npu-aqn-test

# Run experiments in order:
bash scripts/test_mxfp4_v2.1_rmsnorm_aqn.sh 8  # First
bash scripts/test_mxfp4_v2.2_linear_aqn.sh 8
bash scripts/test_nvfp4_v2.3_baseline.sh 8
bash scripts/test_nvfp4_v2.4_rmsnorm_aqn.sh 8
```

---

## 6. References

- [MXFP4_AQN_NEXT_STEPS.md](MXFP4_AQN_NEXT_STEPS.md) - Detailed experiment plan
- [SRDD_MXFP4_QUANT_EXPERIMENT.md](SRDD_MXFP4_QUANT_EXPERIMENT.md) - SRDD scan results
- [HW_ERROR_INJECTION_EXPERIMENTS.md](HW_ERROR_INJECTION_EXPERIMENTS.md) - Previous E5/E7 experiments
