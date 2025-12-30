# HW Heterogeneous Error Injection - Research Plan

**Created**: 2025-12-30
**Objective**: Simulate HW heterogeneous errors (GPU vs NPU) on A100 GPU to test AQN robustness hypothesis
**Status**: Planning

---

## Background

### Problem Statement
- NPU baseline and GPU baseline show no significant difference in GSM8K accuracy
- AQN (Adaptive Quantization Noise) didn't show meaningful impact in our tests
- Hypothesis: The natural HW differences between GPU/NPU may be too small to observe

### Proposed Solution
Inject **synthetic HW heterogeneous errors** on GPU during training to:
1. Create a controlled environment to study error robustness
2. Test if AQN can help models become robust to these errors
3. Avoid needing NPU hardware for initial experiments

### Analogy
Similar to **quantization simulation** (fake quantization):
- Real: GPU doesn't support NVFP4 natively
- Simulation: Add quant/dequant error to BF16 computation
- Our case: Add "NPU-like errors" to GPU computation

---

## Phase 1: Operator Scope Identification

### Goal
Identify operators involved in Qwen2.5-1.5B-Instruct + GSM8K training on A100 (verl + FSDP + vLLM)

### Components to Analyze

| Component | Framework | Purpose | Operators |
|-----------|-----------|---------|-----------|
| Rollout | vLLM | Token generation | TBD |
| Actor Forward | FSDP | Log prob computation | TBD |
| Actor Backward | FSDP | Gradient computation | TBD |
| Ref Forward | FSDP | Reference log prob | TBD |

### Key Questions
1. What operators are used in Qwen2.5-1.5B forward pass?
2. Which operators differ most between GPU and NPU implementations?
3. Do we need eager mode for vLLM to ensure our hooks work?

### Qwen2.5-1.5B Architecture (Expected Operators)
```
Embedding: nn.Embedding
├── Transformer Layers (28 layers):
│   ├── input_layernorm: RMSNorm
│   ├── self_attn:
│   │   ├── q_proj: nn.Linear (BF16 matmul)
│   │   ├── k_proj: nn.Linear (BF16 matmul)
│   │   ├── v_proj: nn.Linear (BF16 matmul)
│   │   ├── attention: scaled_dot_product_attention / flash_attention
│   │   └── o_proj: nn.Linear (BF16 matmul)
│   ├── post_attention_layernorm: RMSNorm
│   └── mlp:
│       ├── gate_proj: nn.Linear (BF16 matmul)
│       ├── up_proj: nn.Linear (BF16 matmul)
│       ├── act_fn: SiLU
│       └── down_proj: nn.Linear (BF16 matmul)
└── lm_head: nn.Linear
```

### Operators to Investigate
| Operator | PyTorch Op | Potential HW Difference |
|----------|-----------|------------------------|
| Linear (MatMul) | torch.mm / F.linear | Accumulation order |
| RMSNorm | Custom kernel | Variance computation |
| Softmax | F.softmax | Numerical stability |
| SiLU | F.silu | Approximation |
| FlashAttention | flash_attn kernel | Tiling differences |
| ReduceSum | torch.sum | Reduction tree |

---

## Phase 2: vLLM/FSDP Mode Requirements

### vLLM Considerations
- **Eager mode**: May be required to ensure hooks work (no graph compilation)
- **CUDA graphs**: May need to disable for dynamic hook injection
- Current config: `enforce_eager=True` (already enabled in our tests)

### FSDP Considerations
- **Activation checkpointing**: May affect where hooks fire
- **Sharding**: Hooks should work on sharded parameters
- **Mixed precision**: BF16 autocast affects operator selection

### Investigation Needed
1. Does vLLM eager mode guarantee our forward hooks work?
2. Can we hook into FSDP-wrapped modules?
3. Are there any graph optimizations that bypass our hooks?

---

## Phase 3: quant_compute Library Analysis

### Location
`~/workspace/quant_compute`

### Purpose
Understand fake quantization implementation for reuse

### Expected Patterns
```python
# Fake quantization pattern
def fake_quant(x, scale, zero_point, num_bits):
    # Quantize
    x_int = round(x / scale) + zero_point
    x_int = clamp(x_int, 0, 2^num_bits - 1)
    # Dequantize
    x_dequant = (x_int - zero_point) * scale
    # Error = x_dequant - x (quantization error injected)
    return x_dequant
```

### Adaptation for HW Error
```python
# Fake HW heterogeneous error pattern
def fake_hw_error(x, error_scale, error_type):
    if error_type == 'gaussian':
        error = torch.randn_like(x) * x.abs() * error_scale
    elif error_type == 'systematic':
        error = error_scale * torch.sign(x)
    return x + error
```

---

## Phase 4: Implementation Plan

### Step 1: Create Error Injection Module
- Location: `verl/utils/hw_error_injection.py`
- Reuse patterns from quant_compute

### Step 2: Integration Points
- vLLM: Hook into model forward (rollout)
- FSDP: Hook into actor model (training)

### Step 3: Configuration
```yaml
trainer:
  hw_error_injection:
    enabled: true
    target_operators: ['linear', 'rmsnorm']
    error_scale: 1e-5
    error_type: 'gaussian'
    apply_during: 'both'  # 'rollout', 'training', 'both'
```

### Step 4: Experiments
| Exp | Error | AQN | Expected |
|-----|-------|-----|----------|
| E1 | None | None | Baseline |
| E2 | 1e-5 | None | Accuracy drop? |
| E3 | 1e-5 | 0.05 | Recovery? |
| E4 | 1e-4 | None | Larger drop |
| E5 | 1e-4 | 0.05 | Can AQN help? |

---

## Progress Log

### 2025-12-30 - Initial Planning
- Created this document
- Next: Investigate operator scope in Qwen2.5-1.5B
- Next: Analyze quant_compute library

### 2025-12-30 - Operator Analysis Complete

#### Qwen2.5-1.5B-Instruct Operators Identified
From `verl/models/transformers/qwen2.py` and transformers library:

**Attention Block** (per layer):
- `self.q_proj(hidden_states)` → nn.Linear (2048 → num_heads * head_dim)
- `self.k_proj(hidden_states)` → nn.Linear (2048 → num_kv_heads * head_dim)
- `self.v_proj(hidden_states)` → nn.Linear (2048 → num_kv_heads * head_dim)
- `apply_rotary_pos_emb()` → cos/sin multiplication
- `_flash_attention_forward()` → FlashAttention kernel
- `self.o_proj(attn_output)` → nn.Linear

**MLP Block** (per layer):
- `gate_proj` → nn.Linear (2048 → intermediate_size)
- `up_proj` → nn.Linear (2048 → intermediate_size)
- `F.silu(gate) * up` → SiLU activation + elementwise multiply
- `down_proj` → nn.Linear (intermediate_size → 2048)

**Normalization**:
- `input_layernorm` → Qwen2RMSNorm
- `post_attention_layernorm` → Qwen2RMSNorm

#### quant_compute Library Analysis Complete

**Key Files:**
- `quant_cy_npu/base/QFuncs/quant_basic.py` - Generic quantization with shared exponent
- `quant_cy_npu/base/QFuncs/nvf4.py` - NVFP4 (E4M3 scale + E2M1 values)
- `HiF4_NVFP4_v14f16.py` - NumPy reference implementations

**Fake Quantization Pattern:**
```python
@torch.no_grad()
def quant_nvf4(x: Tensor, Q: QType, qdim: int):
    # 1. Reshape to blocks of 16
    x = x.unflatten(qdim, (-1, 16))
    x_unsigned = torch.abs(x)
    sign = torch.sign(x)

    # 2. Compute E4M3 scale factor from group max
    grp_max = torch.amax(x_unsigned, dim=qdim, keepdim=True)
    sf = grp_max / 6
    sf = torch.clip_(sf, 0, 448)
    sf_exp = torch.floor(torch.log2(sf + eps))
    E4M3 = torch.round(sf * 2**(-sf_exp + 3)) * 2**(-3 + sf_exp)

    # 3. Quantize values to E2M1
    igv = x_unsigned / E4M3
    E2 = torch.floor(torch.log2(igv + eps))
    E2.clamp_(0)
    M1 = torch.round(igv * 2**(-E2 + 1)) * 2**(-1)
    E2M1 = 2**E2 * M1
    E2M1.clamp_(0, 6)

    # 4. Dequantize (result contains quantization error)
    res = sign * E4M3 * E2M1
    return res.flatten(qdim-1, qdim)
```

**Key Insight for HW Error Injection:**
- Quantization error is `x_quantized - x_original`
- We can directly inject error: `x_with_error = x + error_function(x)`
- Error magnitude typically relative to value: `error ~ x * scale`

#### vLLM/FSDP Mode Requirements

**vLLM:**
- `enforce_eager=True` **REQUIRED** - disables CUDA graph capture
- This ensures forward hooks are called on every forward pass
- Without eager mode, CUDA graphs would cache the computation graph

**FSDP:**
- Forward hooks work on FSDP-wrapped modules
- Hooks fire after each module's forward pass
- Mixed precision (BF16) is handled automatically

**Recommended Target Operators (Start Small):**
1. **RMSNorm** - Simple, well-defined output, affects all subsequent computation
2. **Linear (down_proj only)** - MLP output, directly affects residual stream

### 2025-12-30 - HW Error Injection Module Designed

#### Implementation Complete
Created `verl/utils/hw_error_injection.py` with:

**Key Features:**
1. **Two injection points** (like quant_compute's fake quantization):
   - `injection_point='input'`: Inject error to operator INPUT (default)
     - Like fake quant: `y = operator(x + error)`
     - Uses `register_forward_pre_hook`
   - `injection_point='output'`: Inject error to operator OUTPUT
     - Like: `y = operator(x) + error`
     - Uses `register_forward_hook`

2. **Three error types:**
   - `relative_gaussian`: `error = randn() * |x| * scale` (HW relative error model)
   - `absolute_gaussian`: `error = randn() * scale` (fixed-point noise)
   - `systematic_bias`: `error = sign(x) * scale` (rounding bias)

3. **Phase control:**
   - Apply during rollout only, training only, or both

**Usage:**
```python
from verl.utils.hw_error_injection import create_hw_error_injector

# Create injector (input injection like quant_compute)
injector = create_hw_error_injector(
    enabled=True,
    error_scale=1e-5,
    injection_point='input',  # Like fake quantization
    target_modules=['rmsnorm'],
)

# Register hooks on model
injector.register_hooks(model)

# Run forward pass - errors are automatically injected
output = model(input_ids)

# Check statistics
injector.print_stats()

# Remove hooks when done
injector.remove_hooks()
```

#### Files Created
- `verl/utils/hw_error_injection.py` - Main module
- `tests/test_hw_error_injection.py` - Unit tests

#### Next Steps
1. Test on A100 with real Qwen2.5-1.5B model
2. Integrate with verl training pipeline
3. Run experiments with different error scales

### 2025-12-30 - Integration Complete

#### Integration into verl
1. **ppo_trainer.yaml**: Added `trainer.hw_error_injection` config section
2. **ray_trainer.py**: Initialize and pass config to rollout workers
3. **vllm_rollout.py**: Register hooks on vLLM model after weight sync

#### Command Line Control
```bash
# Enable minimal (RMSNorm only)
trainer.hw_error_injection.enabled=True \
trainer.hw_error_injection.error_scale=1e-5 \
"trainer.hw_error_injection.target_modules=['rmsnorm']"

# Enable MLP-focused (RMSNorm + down_proj)
trainer.hw_error_injection.enabled=True \
trainer.hw_error_injection.error_scale=1e-5 \
"trainer.hw_error_injection.target_modules=['rmsnorm','down_proj']"
```

#### Test Script Created
`scripts/test_hw_error_injection_a100.sh`:
- `minimal`: RMSNorm only
- `mlp`: RMSNorm + down_proj
- `both`: Run both in parallel (4 GPUs each)

Usage:
```bash
bash scripts/test_hw_error_injection_a100.sh minimal  # Single test
bash scripts/test_hw_error_injection_a100.sh both     # Parallel tests
```

#### Logging Added
- `[HW Error] Registered X hooks on vLLM model after weight sync`
- `[HW Error] First injection on <module_name>: input_shape=..., mean_error=..., rel_error=...`
- Statistics tracking: count, mean_error, max_error per module

---

## References
- QeRL paper: Quantization error robustness
- vLLM source: Model execution path
- FSDP documentation: Hook integration
- quant_compute: Fake quantization implementation

