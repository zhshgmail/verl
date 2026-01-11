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

1. **Modify `noise_injection.py`** to accept layer-specific sigma
2. **Create SRDD config file** mapping layer → sigma multiplier
3. **Run experiments** on HW error setup (5% matmul)

---

## Execution Checklist

### Phase 1: Lower AQN (Today)
- [ ] Create `scripts/test_noisy_ops_aqn_lower.sh` (E5c)
- [ ] Git commit and push to personal remote
- [ ] SSH to A100 and pull code
- [ ] Run E5c experiment
- [ ] Record results

### Phase 2: SRDD-Guided AQN (Next)
- [ ] Implement layer-specific sigma in `noise_injection.py`
- [ ] Create SRDD-based sigma config
- [ ] Run E9a/E9b/E9c experiments
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

### 2026-01-11 Initial Setup
- SRDD/fake quant validation completed
- Session document created
- Planning lower AQN experiments

---

*Session started: 2026-01-11*
