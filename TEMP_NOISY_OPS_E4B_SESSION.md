# Noisy Ops Test Session - 2025-12-30

## E4b Test (1e-4 scale) - COMPLETED

### Configuration
- **Model**: Qwen2.5-1.5B-Instruct
- **Dataset**: GSM8K (7473 train, 1319 test)
- **GPUs**: 8x A100-SXM4-80GB
- **Error Scale**: 1e-4 (relative_gaussian)
- **Implementation**: `verl/utils/noisy_ops.py` (operator-level)
- **Noise Coverage**: ALL phases (rollout + training, forward + backward)
- **Key Config**: `enforce_eager=True` to disable torch.compile in vLLM

### Final Results
| Step | OOD Accuracy | ID Reward |
|------|--------------|-----------|
| 0 | 7.88% | - |
| 20 | 73.01% | 75.0% |
| 40 | 75.82% | 78.6% |
| 60 | 77.41% | 81.9% |
| 80 | 77.33% | 79.4% |
| 100 | 77.18% | 81.7% |
| 116 | **77.33%** | **82.5%** |

### Comparison with GPU Baseline
| Metric | GPU Baseline | E4b (1e-4) | Delta |
|--------|--------------|------------|-------|
| Final OOD | 76.88% | 77.33% | **+0.45%** |

**Conclusion**: 1e-4 noise scale does NOT cause observable degradation. May act as regularization.

---

## E4c Test (1e-3 scale) - COMPLETED

### Configuration
- Same as E4b, but with **10x larger error scale (1e-3)**

### Final Results
| Step | OOD Accuracy | ID Reward |
|------|--------------|-----------|
| 0 | 8.49% | - |
| 20 | 73.84% | 74.8% |
| 40 | 74.22% | 77.2% |
| 60 | 75.74% | 79.8% |
| 80 | 75.06% | 80.3% |
| 100 | 76.42% | 83.0% |
| 116 | **77.18%** | **84.5%** |

### Comparison with Baseline
| Metric | GPU Baseline | E4b (1e-4) | E4c (1e-3) |
|--------|--------------|------------|------------|
| Final OOD | 76.88% | 77.33% | **77.18%** |
| vs Baseline | - | +0.45% | +0.30% |

### Key Finding
**Even 10x noise (1e-3) does NOT cause significant degradation!**
- Shows ~2% degradation mid-training (steps 40-80)
- Recovers by end of training (final delta only -0.15% vs E4b)
- Suggests noise acts as regularization (like dropout)

---

## Key Findings

### torch.compile Conflict Resolution
- vLLM V1 uses `torch.compile` by default
- Custom `autograd.Function` with `@torch.compiler.disable` causes conflict
- **Solution**: Set `enforce_eager=True` in rollout config

### Noise Injection Coverage
| Phase | Module-level (hw_error_injection.py) | Operator-level (noisy_ops.py) |
|-------|--------------------------------------|-------------------------------|
| Rollout Forward | Yes | Yes |
| Rollout Backward | No | **Yes** |
| Training Forward | Yes | Yes |
| Training Backward | No | **Yes** |

## Next Steps
1. ~~Wait for E4c (1e-3) to complete~~ **DONE - No significant degradation!**
2. Options to get observable degradation:
   - **E4d**: Test 1e-2 scale (100x original)
   - **E5**: Add noise to more ops: `torch.bmm`, `F.softmax`, `F.silu`, `F.gelu`
   - **E6**: Test systematic bias instead of Gaussian noise
3. Once degradation observed, test AQN + noisy_ops

## Commands

### SSH to A100 Machine
```bash
ssh root@90.90.102.18
docker exec -it verl-r3-test bash
```

### Check Test Progress
```bash
ssh root@90.90.102.18 "docker exec verl-r3-test tail -50 /tmp/noisy_ops_1e-4.log"
```

### Environment Variables for Noisy Ops
```bash
export VERL_NOISY_OPS_ENABLED=1
export VERL_NOISY_OPS_SCALE=1e-4
export VERL_NOISY_OPS_TYPE=relative_gaussian
```

### Key Files
- Operator-level implementation: `verl/utils/noisy_ops.py`
- Test script: `scripts/test_noisy_ops_a100.sh`
- Documentation: `docs/qerl/HW_ERROR_INJECTION_EXPERIMENTS.md`

## A100 Machine Info
- SSH: `root@90.90.102.18`
- Container: `verl-r3-test`
- Model path: `/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Data path: `/data/z00637938/gsm8k/`
