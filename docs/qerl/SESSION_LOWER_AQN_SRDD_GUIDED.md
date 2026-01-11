# Session: Lower AQN + SRDD-Guided Experiments

**Date**: 2026-01-11
**Branch**: `feature/npu-aqn-test`
**Status**: IN PROGRESS

---

## Session Objectives

1. **Validate SRDD tool and fake quant** - COMPLETED
2. **Apply lower AQN** to HW error (5% matmul) experiments
3. **Plan/execute SRDD-guided AQN** experiments

---

## Part 1: SRDD/Fake Quant Validation (COMPLETED)

### Key Findings

1. **verl's NVFP4 implementation is CORRECT** - matches manual reference implementation
2. **SRDD's relative error metric is MISLEADING** for MXFP4 vs NVFP4 comparison
3. **By standard metrics (SQNR, MSE), NVFP4 is slightly BETTER than MXFP4**

| Metric | NVFP4 | MXFP4 | Winner |
|--------|-------|-------|--------|
| SQNR (dB) | **18.83** | 18.59 | NVFP4 |
| MSE | **0.000011** | 0.000012 | NVFP4 |
| Relative Error (misleading) | 26.51% | 21.77% | MXFP4 |
| Deadzone | 13.23% | 9.43% | MXFP4 |

**Conclusion**: SRDD relative error is inflated by NVFP4's higher deadzone. Use SQNR for accurate comparison.

**Documentation**: `docs/qerl/SRDD_COLIBRARY_VALIDATION.md`

---

## Part 2: Lower AQN Experiments

### Current E5b Settings (0.05→0.0005)

From `scripts/test_noisy_ops_aqn_epoch_aware.sh`:
```yaml
noise_injection:
  enabled: True
  epoch_aware: True
  sigma_start: 0.05      # Current (high)
  sigma_end: 0.0005      # Current (high)
  stages_per_epoch: 5

noisy_ops:
  enabled: True
  error_scale: 5e-2      # 5% matmul error
  error_type: relative_gaussian
```

**E5b Result**: 70.58% accuracy

### Proposed E5c: Lower AQN (0.01→0.00001)

```yaml
noise_injection:
  enabled: True
  epoch_aware: True
  sigma_start: 0.01      # 5x lower than E5b
  sigma_end: 0.00001     # 50x lower than E5b
  stages_per_epoch: 5

noisy_ops:
  enabled: True
  error_scale: 5e-2      # Same 5% matmul error
  error_type: relative_gaussian
```

### Rationale for Lower AQN

1. **QeRL paper uses σ=0.01→0.0001** for NVFP4 fake quant experiments
2. **E5b uses 0.05→0.0005** which is 5x higher than QeRL
3. **Lower noise may improve convergence** while maintaining robustness
4. For 5% HW error, lower AQN may be more appropriate than for 21% MXFP4 error

### Experiment Plan

| ID | Config | sigma_start | sigma_end | HW Error | Status |
|----|--------|-------------|-----------|----------|--------|
| E5b | Epoch-aware AQN (current) | 0.05 | 0.0005 | 5% matmul | **70.58%** (done) |
| **E5c** | Lower AQN | **0.01** | **0.00001** | 5% matmul | PLANNED |
| E5d | QeRL-exact AQN | 0.01 | 0.0001 | 5% matmul | PLANNED |

### Implementation

Create new script: `scripts/test_noisy_ops_aqn_lower.sh`

---

## Part 3: SRDD-Guided AQN Experiments

### Concept

Use SRDD scan results to guide WHERE and HOW STRONG to inject AQN:
1. **High SRDD error layers** → Higher AQN noise
2. **Low SRDD error layers** → Lower/no AQN noise

### SRDD Layer Analysis (from previous scans)

| Layer Range | Relative Error | Priority | Proposed AQN |
|-------------|----------------|----------|--------------|
| **Layer 14-17** | 40.8-42.7% | HIGH | σ × 1.5 |
| **Layer 10-13, 18-21** | 37-40% | MEDIUM | σ × 1.2 |
| **Layer 0-9, 22-27** | 28-37% | LOW | σ × 1.0 (baseline) |

### Proposed Experiments

| ID | Config | Description | Expected |
|----|--------|-------------|----------|
| E9a | Targeted AQN | Only inject to layers 14-17 | Faster training |
| E9b | Variable sigma | σ scaled by SRDD error | Better accuracy |
| E9c | Combined | Targeted + variable sigma | Best of both |

### Implementation Approach

#### Step 1: Create SRDD-to-Sigma Mapping

Based on SRDD scan results, create a mapping of layer → sigma multiplier:

```python
# Example: srdd_sigma_config.json
{
    "base_sigma": 0.01,  # Base sigma value
    "layer_multipliers": {
        # High error layers (14-17): 1.5x
        "14": 1.5, "15": 1.5, "16": 1.5, "17": 1.5,
        # Medium error layers (10-13, 18-21): 1.2x
        "10": 1.2, "11": 1.2, "12": 1.2, "13": 1.2,
        "18": 1.2, "19": 1.2, "20": 1.2, "21": 1.2,
        # Low error layers: 1.0x (default)
    }
}
```

#### Step 2: Modify `noise_injection.py`

Add layer-specific sigma support to `generate_expert_gaussian_noise`:

```python
def generate_expert_gaussian_noise(
    model, step, total_step, sigma_trend,
    ...,
    layer_sigma_config=None,  # NEW: SRDD-guided config
):
    """
    If layer_sigma_config provided:
      - base_sigma: base sigma value
      - layer_multipliers: dict mapping layer_id -> sigma multiplier
    """
    # For each layer, compute: sigma = base_sigma * multiplier
```

#### Step 3: Create Experiment Scripts

| Script | Config | Description |
|--------|--------|-------------|
| `test_noisy_ops_srdd_targeted.sh` | Only layers 14-17 | E9a: Targeted AQN |
| `test_noisy_ops_srdd_variable.sh` | Variable sigma per layer | E9b: SRDD-guided sigma |
| `test_noisy_ops_srdd_combined.sh` | Both targeted + variable | E9c: Combined |

---

## Execution Checklist

### Phase 1: Lower AQN (Today)
- [x] Create `scripts/test_noisy_ops_aqn_lower.sh` (E5c)
- [x] Git commit and push to personal remote
- [x] SSH to A100 and pull code
- [x] Run E5c experiment (RUNNING)
- [ ] Record results

### Phase 2: SRDD-Guided AQN (IMPLEMENTED)
- [x] Implement layer-specific sigma in `noise_injection.py`
- [x] Create SRDD-based sigma configs (full and targeted)
- [x] Create E9a/E9b experiment scripts
- [x] Git commit and push to personal remote
- [ ] Run E9a/E9b experiments after E5c completes
- [ ] Analyze results

---

## Quick Commands

### Code Sync (Git instead of SCP)
```bash
# Local: commit and push
cd /home/zheng/workspace/verl
git add -A && git commit -m "feat: add lower AQN experiment script"
git push personal feature/npu-aqn-test

# A100: pull
ssh root@90.90.102.18 "docker exec verl-r3-test bash -c 'cd /home/z00637938/workspace/verl && git pull personal feature/npu-aqn-test'"
```

### Run Experiment
```bash
ssh root@90.90.102.18
docker exec -it verl-r3-test bash
cd /home/z00637938/workspace/verl
nohup bash scripts/test_noisy_ops_aqn_lower.sh 5e-2 8 > /tmp/noisy_ops_aqn_lower.log 2>&1 &
```

### Monitor Progress
```bash
ssh root@90.90.102.18 "docker exec verl-r3-test tail -50 /tmp/noisy_ops_aqn_lower.log"
ssh root@90.90.102.18 "docker exec verl-r3-test grep -E 'step:|val-core' /tmp/noisy_ops_aqn_lower.log | tail -20"
```

---

## Status Updates

### 2026-01-11 23:10 UTC - SRDD-Guided AQN Implemented
- Implemented layer-specific sigma support in `noise_injection.py`
- Added `layer_sigma_config` parameter with per-layer multipliers
- Created sigma config files: `configs/srdd_sigma_qwen2.5_1.5b*.json`
- Created experiment scripts: `scripts/test_noisy_ops_srdd_*.sh`
- Git commit: `e6ff94f0` pushed to personal remote

### 2026-01-11 22:51 UTC - E5c Re-Started (NumPy Fix)
- Fixed NumPy/Numba incompatibility: downgraded NumPy 2.4.1 → 2.2.6
- E5c experiment re-launched successfully
- Log file: `/tmp/noisy_ops_aqn_lower.log`
- Ray started, vLLM initializing, AQN enabled
- Config confirmed: sigma_start=0.01, sigma_end=0.00001

### 2026-01-11 22:00 UTC - E5c Started (Failed)
- Initial attempt failed due to NumPy version incompatibility
- Error: "Numba needs NumPy 2.2 or less. Got NumPy 2.4"

### 2026-01-11 Initial Setup
- SRDD/fake quant validation completed
- Session document created
- Planning lower AQN experiments

---

## Monitoring E5c

```bash
# Check progress
ssh root@90.90.102.18 "docker exec verl-r3-test grep -E 'step:|val-core|Training Progress' /tmp/noisy_ops_aqn_lower.log | tail -20"

# Check AQN sigma decay
ssh root@90.90.102.18 "docker exec verl-r3-test grep -i 'sigma\|noise' /tmp/noisy_ops_aqn_lower.log | tail -10"

# Full tail
ssh root@90.90.102.18 "docker exec verl-r3-test tail -50 /tmp/noisy_ops_aqn_lower.log"
```

---

*Session started: 2026-01-11*
