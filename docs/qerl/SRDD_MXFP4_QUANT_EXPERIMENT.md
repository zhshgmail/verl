# SRDD + MXFP4 Fake Quantization Experiment Plan

**Date**: 2026-01-08
**Status**: In Progress

---

## 1. Objective

Use SRDD diagnostic tool to identify quantization-sensitive layers in MXFP4 quantized Qwen2.5-1.5B model, then validate whether SRDD-guided AQN can improve robustness compared to baseline (global) AQN.

---

## 2. Background

### 2.1 Why This Experiment?

| Problem | SRDD Solution |
|---------|---------------|
| MXFP4 quantization causes accuracy degradation | SRDD can identify which layers are most affected |
| Don't know where quantization error is worst | Gain scan detects deadzone, Kurtosis scan detects saturation |
| Uniform AQN may not be optimal | SRDD-guided AQN targets problematic layers |

### 2.2 Fake Quantization Validity

Fake quantization (quant_compute) is mathematically equivalent to real quantization:
```
error_fake = dequant(quant(w)) - w
error_real = dequant(quant(w)) - w  # Same!
```

This allows us to:
- Simulate MXFP4 effects without actual FP4 storage
- Run SRDD scans on "quantized" model
- Train with AQN using simulated quantization error

---

## 3. Environment

### 3.1 Hardware
- **GPU**: A100-SXM4-80GB
- **Server**: 90.90.102.18
- **Container**: verl-r3-test

### 3.2 Model
- **Path**: `/data/z00637938/hub/` (Qwen2.5-1.5B-Instruct)
- **Precision**: BF16 → MXFP4 fake quantization

### 3.3 Libraries
- **mxfp4_quant**: `verl/utils/mxfp4_quant.py` (standalone MXFP4 fake quantization, no external deps)
- **SRDD**: `scripts/srdd_error_finder.py` (diagnostic tool)

### 3.4 Dependencies (all standard, should be pre-installed)
- torch
- scipy
- numpy
- transformers

---

## 4. Experiment Design

### 4.1 Phase 1: SRDD Scan Comparison

Compare BF16 baseline vs MXFP4 quantized model:

| Scan | BF16 (baseline) | MXFP4 | What to look for |
|------|-----------------|-------|------------------|
| Gain scan | ~1.0 (healthy) | Drop indicates deadzone | Layers with gain << 1.0 |
| Kurtosis scan | ~3500 (normal) | Drop indicates saturation | Layers with kurtosis drop |
| Instability scan | 0 (deterministic) | May increase with SR | Layers with variance |

### 4.2 Phase 2: Identify Problematic Layers

Criteria for "problematic" layers:
- **Deadzone**: `gain_mxfp4 / gain_bf16 < 0.9` (>10% gain loss)
- **Saturation**: `kurtosis_mxfp4 / kurtosis_bf16 < 0.9` (>10% kurtosis drop)
- **High error**: Layers contributing most to output divergence

### 4.3 Phase 3: AQN Strategy Comparison (if issues found)

| Config | Description | AQN Layers |
|--------|-------------|------------|
| Baseline (no AQN) | MXFP4 only | None |
| Global AQN | MXFP4 + AQN all layers | All 28 layers |
| SRDD-guided AQN | MXFP4 + AQN problematic layers only | SRDD-identified layers |
| Healthy AQN (control) | MXFP4 + AQN healthy layers only | All except problematic |

---

## 5. Implementation Plan

### 5.1 Script: `scripts/srdd_mxfp4_experiment.py`

```python
# Pseudo-code structure

class MXFP4SRDDExperiment:
    def __init__(self, model_path, quant_type='mxfp4'):
        self.model_path = model_path
        self.quant_type = quant_type

    def load_bf16_model(self):
        """Load original BF16 model"""
        pass

    def apply_fake_quantization(self, model):
        """Apply MXFP4 fake quantization using quant_compute"""
        pass

    def run_srdd_scan(self, model, name):
        """Run full SRDD diagnostic scan"""
        # - Gain scan (deadzone detection)
        # - Kurtosis scan (saturation detection)
        # - Instability scan (if using stochastic rounding)
        pass

    def compare_bf16_vs_mxfp4(self):
        """Compare SRDD results between BF16 and MXFP4"""
        pass

    def identify_problematic_layers(self):
        """Find layers most affected by quantization"""
        pass

    def run_aqn_comparison(self, problematic_layers):
        """Compare Global AQN vs SRDD-guided AQN"""
        pass
```

### 5.2 Key Integration Points

1. **quant_compute integration**:
```python
import sys
sys.path.insert(0, '/home/zheng/workspace/quant_compute')
from quant_cy_npu import QType, quant_dequant_float
from quant_cy_npu.utils import replace_linear
```

2. **SRDD integration**:
```python
from scripts.srdd_error_finder import SRDDErrorFinder
# Or implement inline SRDD scans
```

3. **Fake quant hook for SRDD**:
```python
class FakeQuantHook:
    """Hook that applies fake MXFP4 quantization to layer output"""
    def __init__(self, qtype='mxfp4'):
        self.Q = QType(qtype)

    def __call__(self, module, input, output):
        # Apply fake quantization to output
        if isinstance(output, tuple):
            hidden = output[0]
            hidden_quant = quant_dequant_float(hidden, self.Q)
            return (hidden_quant,) + output[1:]
        return quant_dequant_float(output, self.Q)
```

---

## 6. Expected Results

### 6.1 SRDD Scan Findings

| Layer | BF16 Gain | MXFP4 Gain | Status |
|-------|-----------|------------|--------|
| L0-L2 | ~1.0 | TBD | Embedding layers (may be anomalous) |
| L3-L26 | ~1.0 | TBD | Main transformer layers |
| L27 | ~1.0 | TBD | Final layer |

### 6.2 Quantization Impact Hypothesis

- **Early layers**: May be more sensitive (small activations → deadzone)
- **Attention layers**: Softmax may cause saturation
- **FFN layers**: Large intermediate values may clip

### 6.3 AQN Comparison Hypothesis

If SRDD finds problematic layers:
- SRDD-guided AQN should outperform Global AQN
- Improvement comes from not degrading healthy layers
- Similar to previous deadzone experiment results

---

## 7. Success Criteria

### 7.1 Phase 1 Success
- [ ] SRDD scan completes on both BF16 and MXFP4 models
- [ ] Clear difference visible between BF16 and MXFP4 metrics
- [ ] At least 1 layer identified as "problematic"

### 7.2 Phase 2 Success (if applicable)
- [ ] SRDD-guided AQN achieves lower loss than Global AQN
- [ ] Statistical significance (p < 0.05)
- [ ] Results align with previous deadzone experiment

---

## 8. Files

| File | Purpose |
|------|---------|
| `scripts/srdd_mxfp4_experiment.py` | Main experiment script |
| `TEMP_SRDD_MXFP4_QUANT_EXPERIMENT.md` | This plan document |
| `docs/qerl/SRDD_GUIDED_AQN_PROPOSAL_CN.md` | Final results (to update) |

---

## 9. Execution Commands

```bash
# SSH to A100 server
ssh root@90.90.102.18
docker exec -it verl-r3-test bash
cd /home/z00637938/workspace/verl

# Pull latest code
git pull personal feature/npu-aqn-test

# Run experiment (uses standalone mxfp4_quant, no external deps)
python scripts/srdd_mxfp4_experiment.py \
    --model_path /data/z00637938/hub/Qwen2.5-1.5B-Instruct \
    --quant_type mxfp4 \
    --output results_mxfp4_srdd.json

# Try different quantization modes:
# --quant_type mxfp4      # Standard MXFP4
# --quant_type mxfp4_sr   # With stochastic rounding
# --quant_type mxfp4_2d   # 2D block quantization (32x32)
```

---

## 10. Notes

- Start with W4A16 (weights only), then try W4A4 if needed
- Use `force_py=True` if NPU kernels unavailable on A100
- Compare `mxfp4` vs `mxfp4-sr` vs `mxfp4_2d` variants
- Document any unexpected findings
