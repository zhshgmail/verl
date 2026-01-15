# QeRL & MoE Quantization Research

**Generated**: 2025-12-24
**Source**: Subagent research on QeRL quantization approach and MoE best practices
**Confidence Level**: HIGH (85%+ overall)

---

## Table of Contents

1. [QeRL Quantization Approach](#qerl-quantization-approach)
2. [MoE Quantization Best Practices](#moe-quantization-best-practices)
3. [NVFP4 Format Details](#nvfp4-format-details)
4. [Layer-Specific Recommendations](#layer-specific-recommendations)
5. [Tools and Frameworks](#tools-and-frameworks)
6. [Implementation Guide](#implementation-guide)

---

## QeRL Quantization Approach

### Primary Format: NVFP4

QeRL uses **NVFP4 (NVIDIA's 4-bit Floating Point)** as the main quantization format:

- **Scheme**: NVFP4A16 (4-bit weights, 16-bit activations)
- **Tool**: llm-compressor with `QuantizationModifier` and `oneshot()` API
- **Output**: Compressed-tensors format with mixed-precision support

### Quantization Configuration

```python
# From QeRL's quantize_nvfp4.py
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(
    targets="Linear",           # Quantize all Linear layers
    scheme="NVFP4A16",         # FP4 weights, BF16 activations
    ignore=["lm_head"]         # Keep output layer in full precision
)

oneshot(model=model, recipe=recipe)
```

### MoE-Specific Exclusions

For MoE models, QeRL excludes router/gate layers from quantization:

```python
# Mixtral-8x7B
ignore=["lm_head", "re:.*block_sparse_moe.gate"]

# Qwen MoE models
ignore=["lm_head", "re:.*mlp.gate$", "re:.*mlp.shared_expert_gate$"]

# DeepSeek-R1
ignore=["lm_head", "re:.*mlp.gate$"]
```

**Rationale**: Gate/router layers determine which experts process each token and are highly sensitive to quantization noise. Keeping them at full precision preserves routing decisions.

### Calibration Settings

| Parameter | Value |
|-----------|-------|
| Dataset | HuggingFaceH4/ultrachat_200k (train_sft split) |
| Samples | 512 samples |
| Max sequence length | 2048 tokens |
| Calibration type | Static |

### Quantization Commands

```bash
cd llm-compressor
conda create -n llmcompressor python=3.12 -y
conda activate llmcompressor
pip install -e .

# Quantize model
python quantize_nvfp4.py --model Qwen/Qwen2.5-7B-Instruct
python quantize_nvfp4.py --model Qwen/Qwen2.5-14B-Instruct
python quantize_nvfp4.py --model Qwen/Qwen2.5-32B-Instruct
```

---

## MoE Quantization Best Practices

### Primary Challenges

1. **Inter-Expert Imbalance**: Uneven distribution of samples across experts leads to insufficient and biased calibration for less frequently utilized experts

2. **Intra-Expert Imbalance**: MoE's unique aggregation mechanism causes varying degrees of correlation between different samples and their assigned experts

3. **Router Sensitivity**: The router/gating network is the most critical component; even small quantization errors can cascade into misrouted tokens

4. **Memory Overhead**: MoE layers account for approximately 80-97% of the parameter footprint

### Precision Requirements

#### Must Keep in BF16/FP16 (HIGH CONFIDENCE: 95%)

| Component | Recommended Precision | Rationale |
|-----------|----------------------|-----------|
| **Router/Gating Networks** | **BF16/FP16/FP32** | Critical for expert selection; softmax requires high precision |
| **Embedding Module** | **BF16/FP16** | Input representations sensitive to precision loss |
| **Output Head (lm_head)** | **BF16/FP16** | Final projection layer affects output quality |
| **Layer Normalization** | **FP32** (training) / **BF16** (inference) | Numerical stability |
| **Attention Operators** | **BF16/FP16** | Softmax computations are precision-sensitive |

#### Can Quantize (HIGH CONFIDENCE: 90%)

| Component | Can Quantize To | Notes |
|-----------|----------------|-------|
| **Expert Weights** | **INT4/FP8/NVFP4** | MoE experts are more robust to quantization than dense FFN |
| **Linear Layers (non-expert)** | **INT8/FP8** | Standard linear transformations tolerate lower precision |
| **Shared Experts** | **Same as routed experts** | Can use same quantization strategy |

### Format Comparison

| Format | Bits | Memory vs FP16 | Accuracy (MoE) | Recommendation |
|--------|------|----------------|----------------|----------------|
| **FP8** | 8 | 2x reduction | **Best** (lossless) | **First choice** |
| **NVFP4** | 4.5 | 3.5x reduction | **Good** (1-2% loss) | **Second choice** |
| **INT8** | 8 | 2x reduction | **Good** (1-3% loss) | Alternative to FP8 |
| **INT4** | 4 | 4x reduction | **Moderate** | With GPTQ/AWQ |

---

## NVFP4 Format Details

### Specifications

- **Block size**: 16 elements
- **Scaling factors**: E4M3 FP8
- **Bits per element**: ~4.5 bits

### Performance Benefits

- **3.5x** memory reduction vs FP16
- **1.8x** memory reduction vs FP8
- **Example**: Llama 3.1 405B: 140GB (FP32) -> 17.5GB (FP4) = **8x reduction**

### Accuracy

- **1% or less degradation** for DeepSeek-R1 (quantized from FP8 to NVFP4)
- **5% higher accuracy** vs MXFP4 for KV cache
- In some cases, NVFP4 shows **2% better accuracy** than original

### Performance

- Up to **3.6x layer-wise speedup** on NVIDIA B200
- Up to **2.2x end-to-end speedup** on NVIDIA B200
- Up to **6x layer-wise** and **4x end-to-end speedup** on RTX5090

---

## Layer-Specific Recommendations

### Quantized Layers (Standard Models)

- All Linear layers except explicitly excluded
- Self-attention projections (q_proj, k_proj, v_proj, o_proj)
- Feed-forward layers (gate_proj, up_proj, down_proj)

### Excluded from Quantization

```python
ignore_list = [
    "lm_head",                    # Output projection
    "re:.*mlp.gate$",            # MoE router (Qwen/DeepSeek)
    "re:.*mlp.shared_expert_gate$",  # Shared expert gate (Qwen1.5-MoE)
    "re:.*block_sparse_moe.gate",    # MoE router (Mixtral)
]
```

### Model-Specific Notes

| Model | Router Pattern | Has Shared Expert Gate |
|-------|---------------|------------------------|
| Qwen1.5-MoE | `mlp.gate` | Yes (`shared_expert_gate`) |
| Qwen3-MoE | `mlp.gate` | No |
| Mixtral | `block_sparse_moe.gate` | No |
| DeepSeek-V3 | `mlp.gate` | No |

---

## Tools and Frameworks

### TensorRT-LLM (Production)

**MoE Support**:
- FP16/BF16/FP8/NVFP4 GEMM operations
- Fused Mixture-of-Experts (MoE) support
- Autotuner for fused MoE and NVFP4 linear operators

**Quantization Methods** (Priority Order):
1. **FP8**: Best performance + accuracy
2. **INT8 SmoothQuant**: Large-batch scenarios
3. **AWQ** (INT4): Weight-only, good for small-batch (<=4)
4. **GPTQ** (INT4): Weight-only alternative
5. **NVFP4**: Extreme memory constraints

### vLLM (Development)

**MoE Quantization Support**:
- FP8 (W8A8) for MoE models
- FlashInfer integration for FP8 MoE on Hopper GPUs
- INT4 weight-only via GPTQ
- Marlin kernels for W8A16 on Ampere GPUs

**Performance** (from verl testing):
- **Qwen3-8B Dense**: ~12% rollout speedup (CUDA 12.6), ~18% (CUDA 12.9)
- **Qwen3-30B-A3B MoE**: **>35% rollout speedup** with FP8

### llm-compressor (QeRL)

**Usage**:
```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=["lm_head", "re:.*mlp.gate$"]
)

oneshot(
    model=model,
    dataset=calibration_ds,
    recipe=recipe,
    calibrate_moe_context=True,  # Critical for MoE
)
```

---

## Implementation Guide

### Recommended Workflow

```
Step 1: Start with FP8 for Expert Weights
├── Tool: TensorRT-LLM or vLLM
├── Format: E4M3 for forward, E5M2 for backward
└── Expected: ~2x memory reduction, <1% accuracy loss

Step 2: Keep Critical Components at High Precision
├── Router/Gating: BF16/FP16
├── Embeddings: BF16/FP16
├── Layer Norm: FP32 (training) / BF16 (inference)
└── Attention: BF16/FP16

Step 3: If More Compression Needed
├── Option A: NVFP4 with calibrate_moe_context=True
│   └── Expected: 3.5x memory reduction, 1-2% accuracy loss
├── Option B: INT4 AWQ/GPTQ
│   └── Expected: 4x memory reduction, 3-5% accuracy loss
└── Option C: Mixed-precision per expert
    └── Expected: Optimized memory/accuracy trade-off

Step 4: Validation
├── Check router decision alignment
├── Monitor expert activation distribution
├── Measure task-specific accuracy
└── Profile inference latency
```

### Quick Start Commands

**TensorRT-LLM**:
```bash
python3 examples/quantization/quantize.py \
    --model_dir <model_path> \
    --dtype float16 \
    --qformat fp8 \
    --calib_size 512

trtllm-build --checkpoint_dir <ckpt> \
    --output_dir <engine> \
    --use_fp8
```

**vLLM**:
```python
from vllm import LLM

llm = LLM(
    model="<moe_model_path>",
    quantization="fp8",
    trust_remote_code=True
)
```

---

## Key Takeaways

### DO (HIGH CONFIDENCE)

1. Keep routers/gates in BF16/FP16 - **CRITICAL**
2. Start with FP8 for experts (best accuracy/performance balance)
3. Use `calibrate_moe_context=True` for MoE models
4. Keep embeddings, layer norm, attention in higher precision
5. Leverage hardware-accelerated kernels (Hopper/Ada for FP8)

### CONSIDER (MEDIUM CONFIDENCE)

1. NVFP4 for extreme memory constraints
2. Mixed-precision allocation based on expert sensitivity
3. INT4 AWQ/GPTQ for small-batch workloads

### AVOID (HIGH CONFIDENCE)

1. Quantizing router/gating networks below BF16
2. Uniform quantization without expert-aware calibration
3. Using INT4/INT2 without proper outlier handling
4. Quantizing layer normalization below FP32 (training)

---

## Sources

### Research Papers (2024-2025)
- MoEQuant Paper (May 2025) - Expert-balanced quantization
- EAQuant Paper (June 2025) - Affinity-guided quantization
- DeepSeek-V3 Technical Report - Production MoE quantization
- MXFP8 Training Recipes - Mixed-precision training

### Official Documentation
- NVIDIA NVFP4 Blog
- TensorRT-LLM Release Notes
- vLLM FP8 Documentation

### Local Implementations
- QeRL llm-compressor examples
- verl FP8 rollout implementation
- vllm-ascend quantization modules
