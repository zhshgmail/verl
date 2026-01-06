# Noise Injection as Diagnostic Probe: Comprehensive Methodology for Neural Network Error Localization

**Date**: 2026-01-05
**Authors**: Research Team
**Status**: Definitive Methodology v1.0
**Branch**: `feature/npu-aqn-test`

---

## Executive Summary

This document presents a comprehensive three-level diagnostic methodology using noise injection to locate sources of numerical errors in large language models (LLMs), particularly for NPU/FP4 deployment. The approach combines:

1. **Your existing implementation**: Layer-wise sensitivity profiling via `noisy_ops.py` with selective injection APIs
2. **Chinese methodology insights**: Three-level localization (operator → layer → channel) with sliding window analysis
3. **Academic foundations**: Information bottleneck theory, Fisher Information Matrix, and Hessian-based sensitivity analysis

**Key Innovation**: Transform noise injection from a "vaccine" (robustness training) into an "oscilloscope" (diagnostic tool) for hardware-aware neural network debugging.

---

## Table of Contents

1. [Theoretical Foundations](#1-theoretical-foundations)
2. [Three-Level Diagnostic Protocol](#2-three-level-diagnostic-protocol)
3. [Implementation Architecture](#3-implementation-architecture)
4. [Metrics and Interpretation Guide](#4-metrics-and-interpretation-guide)
5. [Practical Applications](#5-practical-applications)
6. [Literature Review and Related Work](#6-literature-review-and-related-work)
7. [Implementation Gaps and Roadmap](#7-implementation-gaps-and-roadmap)

---

## 1. Theoretical Foundations

### 1.1 Information Bottleneck Theory

**Core Principle**: Neural networks optimize for the best tradeoff between accuracy and compression when summarizing information. Layers go through two distinct phases:

1. **Fitting Phase**: Captures relevant information from input
2. **Compression Phase**: Discards redundant information

**Relevance to Noise Injection**:
- Noise injection reveals which layers have sufficient information redundancy to tolerate errors
- Layers in compression phase are more robust (information is already compressed/redundant)
- Layers in fitting phase are more sensitive (still extracting critical features)

**Quantized Networks Analysis**: Research on quantized neural networks shows that:
- Mutual information (MI) between layers can be computed exactly in quantized settings
- Compression phase may not occur in all layers (depends on activation function)
- Networks with ReLU activation often don't exhibit compression in hidden layers
- This explains why different layers show vastly different noise sensitivity

**Key Finding**: Generalization bounds scale with the degree of information bottleneck, unlike traditional bounds (VC dimension, Rademacher complexity). This provides theoretical justification for why noise-robust models generalize better.

**References**:
- [Information Bottleneck: Exact Analysis of (Quantized) Neural Networks](https://arxiv.org/abs/2106.12912) (ICLR 2022)
- [Information Bottleneck Theory and Applications in Deep Learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC7764901/)

### 1.2 Fisher Information Matrix (FIM)

**Definition**: The FIM quantifies parameter sensitivity in statistical models, measuring how much information a parameter carries about the data distribution.

**Mathematical Formulation**:
```
FIM[i,j] = E[∂log p(x|θ)/∂θᵢ · ∂log p(x|θ)/∂θⱼ]
```

**Applications in Neural Networks**:
1. **Parameter Importance Ranking**: Higher FIM values indicate parameters that significantly affect model output
2. **Quantization Loss Estimation**: Replace expensive Hessian computation with FIM approximation
3. **Natural Gradient Optimization**: FIM defines the Riemann metric over parameter space
4. **Pruning and Compression**: Remove parameters with low FIM values

**Relevance to Noise Injection**:
- FIM provides theoretical basis for which parameters are most sensitive to perturbations
- Diagonal FIM approximation enables scalable computation even for large models
- FIM-guided quantization (e.g., FIMA-Q for Vision Transformers) shows superior results

**Practical Implementation**:
```python
# Approximate FIM using squared gradients (BRECQ approach)
# 1. Replace Hessian with FIM
# 2. Diagonal approximation: diag(FIM) ≈ E[grad²]
# 3. Use task loss gradients for efficiency

def estimate_fim_diagonal(model, data_loader):
    fim_diag = {name: torch.zeros_like(param)
                for name, param in model.named_parameters()}

    for batch in data_loader:
        loss = compute_loss(model, batch)
        grads = torch.autograd.grad(loss, model.parameters())

        for (name, param), grad in zip(model.named_parameters(), grads):
            fim_diag[name] += grad ** 2

    return {k: v / len(data_loader) for k, v in fim_diag.items()}
```

**References**:
- [FIMA-Q: Post-Training Quantization for Vision Transformers](https://arxiv.org/html/2506.11543)
- [Ranking the parameters of deep neural networks using Fisher information](https://www.researchgate.net/publication/304372335_Ranking_the_parameters_of_deep_neural_networks_using_the_fisher_information)

### 1.3 Hessian Trace as Sensitivity Metric

**Definition**: The Hessian matrix is the second-order derivative of loss with respect to weights, capturing the curvature of the loss landscape.

**Why Hessian Trace?**
- **Trace = Sum of Eigenvalues**: Provides a scalar sensitivity measure for the entire layer
- **Large Eigenvalues**: Indicate high sensitivity to weight perturbations (steep loss landscape)
- **Flat Minima Correlation**: Low Hessian trace correlates with good generalization
- **Computational Efficiency**: Trace estimation is faster than full eigenvalue decomposition

**HAWQ-V2 Methodology**:
```
Layer Sensitivity = Trace(Hessian) = Σᵢ λᵢ

For mixed-precision quantization:
- High trace → Keep at higher precision (8-bit or FP16)
- Low trace → Aggressive quantization (4-bit or INT8)
```

**Hessian Regularization**:
- Penalizing Hessian trace during training encourages flat minima
- Stochastic trace estimation enables practical implementation
- Connection to Lyapunov stability in dynamical systems

**Comparison with Other Metrics**:

| Metric | Computational Cost | Captures | Best For |
|--------|-------------------|----------|----------|
| **Hessian Trace** | O(n²) with approx | Loss curvature | Layer-wise sensitivity |
| **Fisher Information** | O(n) diagonal approx | Parameter importance | Weight-level pruning |
| **Gradient Magnitude** | O(n) | First-order info | Quick screening |
| **Activation Statistics** | O(n) | Data-dependent | Runtime monitoring |

**Practical Trace Estimation**:
```python
# Hutchinson's stochastic trace estimator
def estimate_hessian_trace(model, loss_fn, data, num_samples=10):
    """
    Estimate Tr(H) = E[v^T H v] where v ~ N(0, I)

    Uses Pearlmutter's trick: Hv = ∇_w (∇_w loss · v)
    """
    trace_estimate = 0.0

    for _ in range(num_samples):
        # Random Gaussian vector
        v = [torch.randn_like(p) for p in model.parameters()]

        # Compute Hessian-vector product
        loss = loss_fn(model, data)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        # v^T · grad (scalar)
        grad_v_product = sum(torch.sum(g * vec) for g, vec in zip(grads, v))

        # Compute ∇(grad^T v)
        hv = torch.autograd.grad(grad_v_product, model.parameters())

        # v^T H v = v^T (Hv)
        trace_estimate += sum(torch.sum(vec * hvec) for vec, hvec in zip(v, hv))

    return trace_estimate / num_samples
```

**References**:
- [HAWQ-V2: Hessian Aware trace-Weighted Quantization](https://papers.neurips.cc/paper/2020/file/d77c703536718b95308130ff2e5cf9ee-Paper.pdf) (NeurIPS 2020)
- [Hessian-based mixed-precision quantization](https://www.sciencedirect.com/science/article/abs/pii/S0893608024008396)

### 1.4 Gradient Noise vs Activation Noise

**Critical Distinction**: Your E8c experiment revealed that forward and backward noise serve different purposes:

| Noise Type | When Applied | What It Affects | Skill Learned | Training Benefit | Inference Benefit |
|------------|--------------|-----------------|---------------|------------------|-------------------|
| **Activation Noise** | Forward pass | Layer outputs | Handle noisy computations | Low | High |
| **Gradient Noise** | Backward pass | Weight updates | Optimize despite noise | High | Low |

**Theoretical Basis**:

1. **Training with noise = Tikhonov regularization** (Bishop, 1995)
   - Noise on inputs equivalent to L2 regularization on weights
   - Prevents overfitting by adding implicit regularization term

2. **Gradient noise as optimizer regularization**:
   - Annealed Gaussian gradient noise improves deep network training
   - Adaptive optimizers (Adam, AdaGrad) interact differently with gradient noise vs weight noise
   - Noise effectively adapts to curvature of optimization landscape

3. **Per-example gradient regularization**:
   - Suppresses noise memorization in over-parameterized networks
   - Prioritizes signal learning over noise fitting
   - More effective when combined with batch normalization

**Your Experimental Validation (E8c)**:

```
E5b (Forward + Backward noise):
- Clean accuracy: 78%
- @ 5% noise: 64% (-14% degradation)
- Training benefit: +2.42% vs baseline

E8c (Forward-only noise):
- Clean accuracy: 69.4% (-8.6% vs E5b)
- @ 5% noise: PENDING EVALUATION
- Hypothesis: Better % retention than E5b
```

**Interpretation**:
- Removing backward (gradient) noise → -8.6% clean accuracy confirms gradient noise provides training regularization
- Forward-only noise should provide better inference robustness if theory holds
- The 7B model's robustness (0% degradation) suggests architectural redundancy overcomes noise type

**References**:
- [Training with Noise is Equivalent to Tikhonov Regularization](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-tikhonov-nc-95.pdf) (Bishop, 1995)
- [Adding Gradient Noise Improves Learning](https://openreview.net/pdf?id=rkjZ2Pcxe) (ICLR 2016)
- [Per-example gradient regularization](https://link.springer.com/article/10.1007/s10994-024-06661-5) (Machine Learning, 2024)

### 1.5 Why Sliding Window Beats Single-Layer Injection

**Single-Layer Problem**: Injecting noise into layer N in isolation doesn't capture error propagation effects.

**Sliding Window Advantage**:

1. **Captures Error Accumulation**:
   ```
   Single-layer: [Clean] → [L₃+noise] → [Clean]
   Sliding:      [Clean] → [L₃+noise] → [L₄+noise] → [L₅+noise] → [Clean]
   ```

2. **Identifies Avalanche Points**:
   - Some layer combinations amplify errors exponentially
   - Example: Attention layer errors compound with subsequent MLP errors
   - Window size k=3 captures most critical interactions

3. **Mimics Real Hardware Errors**:
   - Real hardware errors don't occur in single isolated layer
   - Quantization/approximation errors persist across consecutive operations
   - Sliding window reflects actual deployment scenario

**Optimal Window Size**:
- **k=1**: Single layer (baseline, misses interactions)
- **k=3**: Sweet spot (captures local interactions, manageable computation)
- **k=5**: Captures longer-range effects but expensive
- **k=all**: Full model (loses localization benefit)

**Implementation Strategy**:
```python
def sliding_window_diagnosis(model, window_size=3, noise_scale=0.05):
    """
    Perform sliding window noise injection diagnosis.

    Returns sensitivity map: sensitivity[start_layer] = degradation
    """
    num_layers = get_num_layers(model)
    sensitivity_map = {}

    baseline_acc = evaluate_clean(model)

    for start_layer in range(num_layers - window_size + 1):
        end_layer = start_layer + window_size

        # Enable noise only for layers [start_layer, end_layer)
        set_selective_layers(list(range(start_layer, end_layer)))

        noisy_acc = evaluate_with_noise(model, noise_scale)
        degradation = (baseline_acc - noisy_acc) / baseline_acc

        sensitivity_map[start_layer] = degradation

        # Reset
        set_selective_layers(None)

    return sensitivity_map
```

**Expected Pattern**:
```
Sensitivity Heatmap (window_size=3):

Layer Range    Degradation
[0-2]          5%      █████
[3-5]          12%     ████████████
[6-8]          8%      ████████
[9-11]         15%     ███████████████  ← AVALANCHE POINT
[12-14]        6%      ██████
...

Interpretation: Layers 9-11 form a critical bottleneck
```

**References**:
- [Probabilistic fault localization with sliding windows](https://link.springer.com/article/10.1007/s11432-012-4567-x)
- [Online fault monitoring with sliding window](https://www.sciencedirect.com/science/article/abs/pii/S0149197019303427)

### 1.6 Outlier Channels: Theoretical Basis

**Definition**: Outlier channels are activation channels with magnitude 10-100× larger than typical channels, making quantization difficult.

**Root Causes**:

1. **Residual Connection Accumulation**:
   ```
   x₀ = input
   x₁ = x₀ + f(x₀)      # First residual
   x₂ = x₁ + f(x₁)      # Second residual
   ...
   xₙ = xₙ₋₁ + f(xₙ₋₁)  # Accumulated magnitude grows
   ```

2. **Layer Normalization Interaction**:
   - LayerNorm normalizes across features but preserves relative channel magnitudes
   - Channels with consistently high pre-norm values become outliers post-norm

3. **Positional Encoding Artifacts**:
   - Certain positions (e.g., [CLS], [SEP] tokens) accumulate more information
   - Creates systematic outliers at specific sequence positions

4. **Attention Pattern Concentration**:
   - Some heads focus on few tokens → those tokens get large gradient updates
   - Over training, these channels develop extreme values

**SmoothQuant Insight**:
```
Problem: Per-tensor quantization wastes bits on outliers
         scale = max(|activation|) / 127

         If max = 100, typical = 1:
         effective_bits = log₂(100/1) ≈ 6.6 bits (out of 8)

Solution: Migrate difficulty from activations to weights
         X_int8 · W_int8 = (s_x · X̃) · (s_w · W̃)
                        = (s_x · s_w) · (X̃ · W̃)

         Smooth X by scaling: X' = s⁻ᵅ · X, W' = sᵅ · W
         where s = per-channel smoothing factor
```

**AWQ Strategy**:
- Protect channels with large activation magnitudes
- Use per-channel scaling: preserve "salient weights"
- 1-2% of weights contribute disproportionately to accuracy

**Detection Methods**:

1. **Statistical Threshold**:
   ```python
   def detect_outliers(activations, threshold=10.0):
       channel_max = activations.abs().max(dim=0).values  # [C]
       median_max = channel_max.median()
       outlier_ratio = channel_max / median_max
       return outlier_ratio > threshold
   ```

2. **Quantile-Based**:
   ```python
   def detect_outliers_quantile(activations, quantile=0.99):
       q99 = activations.abs().quantile(quantile)
       q50 = activations.abs().quantile(0.50)
       return (q99 / q50) > 5.0  # 5× spread indicates outliers
   ```

3. **Fisher Information Ranking**:
   ```python
   def detect_outliers_fisher(model, calibration_data):
       fim = compute_fisher_information(model, calibration_data)
       # Channels with high FIM are critical (potential outliers)
       return fim > fim.quantile(0.95)
   ```

**Mitigation Strategies**:

| Method | Approach | Pros | Cons |
|--------|----------|------|------|
| **SmoothQuant** | Migrate difficulty to weights | Training-free, general | Requires calibration data |
| **AWQ** | Protect salient weights | Hardware-efficient | Weight-only quantization |
| **OmniQuant** | Learnable equivalent transformation | State-of-the-art accuracy | Requires fine-tuning |
| **Outlier Suppression+** | Normalize outliers during training | Prevents outlier formation | Training-time overhead |

**References**:
- [SmoothQuant: Accurate and Efficient Post-Training Quantization](https://arxiv.org/abs/2211.10438) (ICML 2023)
- [AWQ: Activation-aware Weight Quantization](https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf) (MLSys 2024)
- [OmniQuant](https://arxiv.org/pdf/2308.13137) (ICLR 2024)

### 1.7 Operator-Level Differences: MatMul vs Softmax

**Numerical Stability Characteristics**:

| Operation | Primary Error Source | Typical Error Magnitude | Mitigation |
|-----------|---------------------|------------------------|------------|
| **MatMul** | Accumulation errors | O(n·ε) for n accumulations | Higher-precision accumulators |
| **Softmax** | Overflow/underflow | Exp(x) → ∞ or 0 | Max subtraction trick |
| **LayerNorm** | Division by near-zero | Variance ≈ 0 → NaN | Epsilon clamping (1e-5) |
| **Attention** | Compounding effects | MatMul + Softmax combined | Separate quantization scales |

**MatMul Accumulation Errors**:

```python
# Naive implementation (unstable)
result = 0.0
for i in range(n):
    result += a[i] * b[i]  # Each addition introduces error ε
# Total error: O(n·ε)

# In FP16 with n=4096: error ≈ 4096 × 2⁻¹¹ ≈ 0.2%
```

**Key Problem**: Matrix multiplication requires 32-bit accumulators even with 8-bit inputs.
- NPUs optimize for INT8 × INT8 → INT32 MAC operations
- Final result rescaled to INT8 with rounding
- Errors accumulate in high-dimensional spaces (d=4096 for LLaMA)

**Softmax Overflow Issue**:

```python
# Naive implementation (FAILS)
exp_x = np.exp(x)  # Overflow for x > 88 in FP32
softmax = exp_x / exp_x.sum()  # → NaN

# Numerically stable version
x_shifted = x - x.max()  # Max trick
exp_x = np.exp(x_shifted)  # No overflow
softmax = exp_x / exp_x.sum()
```

**Why This Matters for Hardware**:
- NPUs may use different approximations for exp() than GPUs
- Look-up tables (LUTs) have different granularity
- Error in exp() propagates to entire probability distribution

**Empirical Findings**:

From fault injection studies ([FKeras](https://dl.acm.org/doi/10.1145/3665334)):
- **Most Sensitive Bits (MSB)** are most critical for MatMul
- **Least Sensitive Bits (LSB)** have minimal impact
- **Softmax is particularly fragile**: Single-bit flip can cause NaN propagation

**Your Implementation Strategy**:

```python
# Current noisy_ops.py injects noise into:
# 1. MatMul (always) - via NoisyMatMul
# 2. Softmax (only in all_ops_mode) - via NoisySoftmax

# Recommendation: Separate error scales
_ERROR_SCALE_MATMUL = 1e-5    # Conservative for accumulation
_ERROR_SCALE_SOFTMAX = 1e-6   # Very conservative (exp sensitivity)
_ERROR_SCALE_LAYERNORM = 1e-5 # Moderate (epsilon protection)
```

**Diagnostic Protocol**:
```python
def operator_sensitivity_analysis(model, data):
    results = {}

    # Test MatMul only
    set_selective_ops(['matmul'])
    results['matmul'] = evaluate_with_noise(model, data, scale=1e-5)

    # Test Softmax only
    set_selective_ops(['softmax'])
    results['softmax'] = evaluate_with_noise(model, data, scale=1e-6)

    # Test LayerNorm only
    set_selective_ops(['layernorm'])
    results['layernorm'] = evaluate_with_noise(model, data, scale=1e-5)

    # Rank by sensitivity
    return sorted(results.items(), key=lambda x: x[1])
```

**References**:
- [Accurately computing log-sum-exp and softmax](https://academic.oup.com/imajna/article/41/4/2311/5893596)
- [Numerically Stable Softmax](https://jaykmody.com/blog/stable-softmax/)
- [Improving Numerical Stability of Fast Matrix Multiplication](https://www.cs.cornell.edu/~arb/papers/fast-matmul-simax2016.pdf)
- [FKeras: Sensitivity Analysis Tool](https://dl.acm.org/doi/10.1145/3665334)

---

## 2. Three-Level Diagnostic Protocol

### 2.1 Level 1: Operator-Level Diagnosis

**Goal**: Identify which operation types (MatMul, Softmax, LayerNorm) are most sensitive to noise.

**Protocol**:
```
For each operation_type in [matmul, bmm, softmax, silu, gelu, layernorm]:
    1. Enable noise ONLY for this operation type
    2. Run inference on test set (n=100 samples)
    3. Record accuracy degradation
    4. Compute sensitivity score

Output: Operator sensitivity ranking
```

**Implementation**:
```python
from verl.utils.noisy_ops import enable_noisy_ops, set_selective_ops

def operator_level_diagnosis(model, test_data, noise_scale=0.05):
    """
    Diagnose which operators are most sensitive to noise.

    Returns:
        dict: {operator_name: {'degradation': float, 'sensitivity_score': float}}
    """
    operators = ['matmul', 'softmax', 'silu', 'layernorm']
    baseline_acc = evaluate_model(model, test_data, noise_scale=0.0)

    results = {}
    for op in operators:
        # Enable noise only for this operator
        enable_noisy_ops(error_scale=noise_scale, all_ops_mode=True)
        set_selective_ops([op])

        noisy_acc = evaluate_model(model, test_data, noise_scale=noise_scale)
        degradation = (baseline_acc - noisy_acc) / baseline_acc

        # Sensitivity score: higher = more sensitive
        sensitivity_score = degradation / noise_scale

        results[op] = {
            'degradation': degradation,
            'sensitivity_score': sensitivity_score,
            'baseline_acc': baseline_acc,
            'noisy_acc': noisy_acc
        }

        set_selective_ops(None)  # Reset

    return results
```

**Expected Output**:
```
Operator Sensitivity Report:
┌─────────────┬─────────────┬──────────────────┐
│ Operator    │ Degradation │ Sensitivity Score│
├─────────────┼─────────────┼──────────────────┤
│ softmax     │ 22%         │ 4.4              │  ← CRITICAL
│ layernorm   │ 18%         │ 3.6              │  ← CRITICAL
│ matmul      │ 8%          │ 1.6              │  ← MODERATE
│ silu        │ 3%          │ 0.6              │  ← ROBUST
└─────────────┴─────────────┴──────────────────┘

Recommendation:
- Priority 1: Validate Softmax hardware implementation
- Priority 2: Check LayerNorm epsilon handling
- Priority 3: Monitor MatMul accumulator precision
```

**Interpretation Guide**:

| Sensitivity Score | Interpretation | Action |
|-------------------|----------------|--------|
| > 3.0 | CRITICAL - Major error source | Requires hardware validation |
| 1.5 - 3.0 | MODERATE - Needs attention | Monitor during deployment |
| < 1.5 | ROBUST - Low priority | Standard quantization OK |

### 2.2 Level 2: Layer-Level Diagnosis (Sliding Window)

**Goal**: Identify which layer ranges are most sensitive, revealing "avalanche points" where errors compound.

**Protocol**:
```
window_size = 3  # Captures local error interactions
baseline_acc = evaluate_clean(model)

For start_layer in range(0, num_layers - window_size):
    end_layer = start_layer + window_size

    1. Enable noise ONLY for layers [start_layer, end_layer)
    2. Run inference (n=100 samples)
    3. Record accuracy
    4. Compute degradation = (baseline - accuracy) / baseline

Output: Sensitivity heatmap [layer_range → degradation]
```

**Implementation**:
```python
def sliding_window_diagnosis(
    model,
    test_data,
    window_size=3,
    noise_scale=0.05,
    step_size=1  # 1 = overlapping windows, 3 = non-overlapping
):
    """
    Perform sliding window layer-wise sensitivity analysis.

    Args:
        window_size: Number of consecutive layers to inject noise into
        step_size: Stride for sliding window

    Returns:
        dict: {layer_range: sensitivity_metrics}
    """
    num_layers = get_num_transformer_layers(model)
    baseline_acc = evaluate_model(model, test_data, noise_scale=0.0)

    # Register layer hooks for tracking
    from verl.utils.noisy_ops import register_layer_hooks
    register_layer_hooks(model)

    sensitivity_map = {}

    for start_layer in range(0, num_layers - window_size + 1, step_size):
        end_layer = start_layer + window_size
        layer_range = f"L{start_layer}-{end_layer-1}"

        # Enable noise only for this window
        set_selective_layers(list(range(start_layer, end_layer)))
        enable_noisy_ops(error_scale=noise_scale)

        # Evaluate
        noisy_acc = evaluate_model(model, test_data, noise_scale=noise_scale)
        degradation = (baseline_acc - noisy_acc) / baseline_acc

        # Get injection statistics
        layer_stats = get_layer_injection_stats()
        total_injections = sum(
            stats['forward'] + stats['backward']
            for stats in layer_stats.values()
        )

        sensitivity_map[layer_range] = {
            'start_layer': start_layer,
            'end_layer': end_layer - 1,
            'degradation': degradation,
            'baseline_acc': baseline_acc,
            'noisy_acc': noisy_acc,
            'total_injections': total_injections,
            'sensitivity_score': degradation / noise_scale
        }

        # Reset
        set_selective_layers(None)
        reset_layer_injection_stats()

    return sensitivity_map
```

**Visualization**:
```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_sensitivity_heatmap(sensitivity_map, window_size=3):
    """Create visual sensitivity heatmap."""
    layer_ranges = list(sensitivity_map.keys())
    degradations = [sensitivity_map[lr]['degradation'] * 100
                   for lr in layer_ranges]

    fig, ax = plt.subplots(figsize=(15, 6))

    # Bar chart
    bars = ax.bar(range(len(layer_ranges)), degradations)

    # Color by severity
    for i, bar in enumerate(bars):
        if degradations[i] > 15:
            bar.set_color('red')  # Critical
        elif degradations[i] > 8:
            bar.set_color('orange')  # Moderate
        else:
            bar.set_color('green')  # Robust

    ax.set_xlabel('Layer Range', fontsize=12)
    ax.set_ylabel('Accuracy Degradation (%)', fontsize=12)
    ax.set_title(f'Layer-wise Sensitivity Analysis (Window Size={window_size})',
                 fontsize=14)
    ax.set_xticks(range(len(layer_ranges)))
    ax.set_xticklabels(layer_ranges, rotation=45, ha='right')
    ax.axhline(y=10, color='r', linestyle='--', alpha=0.3,
               label='Critical Threshold (10%)')
    ax.legend()

    plt.tight_layout()
    return fig
```

**Expected Pattern for 28-Layer Transformer**:
```
Sensitivity Heatmap (3-layer windows):

Degradation (%)
20│                              ╱╲
  │                             ╱  ╲
15│                            ╱    ╲        ← AVALANCHE POINT
  │                           ╱      ╲
10│              ╱╲          ╱        ╲╱╲
  │             ╱  ╲        ╱              ╲
 5│   ╱╲       ╱    ╲      ╱                ╲╱╲
  │  ╱  ╲     ╱      ╲    ╱                      ╲
 0│─╱────╲───╱────────╲──╱────────────────────────╲──
  └──┬────┬────┬────┬────┬────┬────┬────┬────┬────┬──
    L0-2 L3-5 L6-8 L9-11 L12-14 L15-17 L18-20 L21-23 L24-26

    Early   Middle        CRITICAL       Middle    Late
   (robust) (medium)     (AVALANCHE)    (medium) (robust)

Interpretation:
- Layers 9-14: Critical bottleneck (attention + MLP interaction)
- Layers 0-5: Feature extraction (some redundancy)
- Layers 21-26: Output formation (surprisingly robust due to residual paths)
```

**Identifying Avalanche Points**:
```python
def identify_avalanche_points(sensitivity_map, threshold=0.15):
    """
    Find layer ranges where errors compound dramatically.

    Avalanche point: degradation > threshold AND
                     degradation > 1.5× adjacent windows
    """
    avalanche_points = []

    sorted_ranges = sorted(sensitivity_map.items(),
                          key=lambda x: x[1]['start_layer'])

    for i, (layer_range, metrics) in enumerate(sorted_ranges):
        degradation = metrics['degradation']

        if degradation > threshold:
            # Check if significantly higher than neighbors
            is_peak = True

            if i > 0:
                prev_deg = sorted_ranges[i-1][1]['degradation']
                if degradation < prev_deg * 1.5:
                    is_peak = False

            if i < len(sorted_ranges) - 1:
                next_deg = sorted_ranges[i+1][1]['degradation']
                if degradation < next_deg * 1.5:
                    is_peak = False

            if is_peak:
                avalanche_points.append({
                    'layer_range': layer_range,
                    'degradation': degradation,
                    'severity': 'CRITICAL' if degradation > 0.20 else 'HIGH'
                })

    return avalanche_points
```

### 2.3 Level 3: Channel-Level Diagnosis (Outlier Detection)

**Goal**: Identify specific activation channels that cause quantization difficulty.

**Protocol**:
```
For each layer L:
    1. Capture activations on calibration set (n=1000 samples)
    2. Compute per-channel statistics:
       - max_magnitude = max(|activation|) per channel
       - mean_magnitude = mean(|activation|) per channel
       - outlier_ratio = max / median(max)
    3. Flag channels where outlier_ratio > 10.0
    4. Cross-reference with Fisher Information

Output: Per-layer outlier channel map
```

**Implementation**:
```python
import torch
from collections import defaultdict

class ChannelOutlierDetector:
    def __init__(self, model, calibration_data, quantile_threshold=10.0):
        """
        Detect outlier channels that cause quantization difficulty.

        Args:
            quantile_threshold: Channels with max/median ratio > this are outliers
        """
        self.model = model
        self.calibration_data = calibration_data
        self.quantile_threshold = quantile_threshold
        self.activation_stats = defaultdict(lambda: {
            'max': [],
            'mean': [],
            'std': [],
            'q99': []
        })
        self.outlier_channels = {}

    def register_hooks(self):
        """Register forward hooks to capture activations."""
        self.hooks = []

        def make_hook(name):
            def hook(module, input, output):
                # output shape: [batch, seq_len, hidden_dim] for transformers
                act = output.detach()

                # Compute per-channel statistics
                channel_max = act.abs().flatten(0, -2).max(dim=0).values  # [C]
                channel_mean = act.abs().flatten(0, -2).mean(dim=0)
                channel_std = act.abs().flatten(0, -2).std(dim=0)
                channel_q99 = act.abs().flatten(0, -2).quantile(0.99, dim=0)

                self.activation_stats[name]['max'].append(channel_max.cpu())
                self.activation_stats[name]['mean'].append(channel_mean.cpu())
                self.activation_stats[name]['std'].append(channel_std.cpu())
                self.activation_stats[name]['q99'].append(channel_q99.cpu())

            return hook

        # Register for key layers (attention output, MLP output)
        for name, module in self.model.named_modules():
            if any(pattern in name for pattern in
                   ['self_attn.o_proj', 'mlp.down_proj', 'mlp.gate_proj']):
                handle = module.register_forward_hook(make_hook(name))
                self.hooks.append(handle)

    def collect_statistics(self):
        """Run calibration data through model to collect stats."""
        self.register_hooks()

        self.model.eval()
        with torch.no_grad():
            for batch in self.calibration_data:
                _ = self.model(**batch)

        # Remove hooks
        for hook in self.hooks:
            hook.remove()

        # Aggregate statistics
        for layer_name, stats in self.activation_stats.items():
            # Stack all batches
            all_max = torch.stack(stats['max'])  # [num_batches, channels]
            all_mean = torch.stack(stats['mean'])

            # Global statistics per channel
            global_max = all_max.max(dim=0).values  # [channels]
            global_mean = all_mean.mean(dim=0)

            # Detect outliers
            median_max = global_max.median()
            outlier_ratio = global_max / (median_max + 1e-8)

            # Flag outlier channels
            outlier_mask = outlier_ratio > self.quantile_threshold
            outlier_indices = torch.where(outlier_mask)[0].tolist()

            if len(outlier_indices) > 0:
                self.outlier_channels[layer_name] = {
                    'outlier_indices': outlier_indices,
                    'num_outliers': len(outlier_indices),
                    'total_channels': len(global_max),
                    'outlier_percentage': len(outlier_indices) / len(global_max) * 100,
                    'outlier_ratios': outlier_ratio[outlier_indices].tolist(),
                    'max_ratio': outlier_ratio.max().item(),
                    'median_max': median_max.item(),
                    'global_max': global_max.max().item()
                }

    def get_report(self):
        """Generate outlier detection report."""
        report = {
            'summary': {
                'layers_with_outliers': len(self.outlier_channels),
                'total_outlier_channels': sum(
                    info['num_outliers']
                    for info in self.outlier_channels.values()
                )
            },
            'per_layer': self.outlier_channels
        }
        return report

    def apply_smoothquant_scaling(self, alpha=0.5):
        """
        Apply SmoothQuant-style scaling to mitigate outliers.

        Args:
            alpha: Migration strength (0=all to activations, 1=all to weights)
        """
        for layer_name, outlier_info in self.outlier_channels.items():
            # Compute per-channel scaling factors
            outlier_indices = outlier_info['outlier_indices']

            # s = (max_act / mean_act)^alpha
            # X' = X / s^alpha, W' = W * s^alpha

            # TODO: Implement actual weight/activation scaling
            print(f"Would scale {len(outlier_indices)} channels in {layer_name}")
```

**Usage**:
```python
# Step 1: Detect outliers
detector = ChannelOutlierDetector(model, calibration_loader)
detector.collect_statistics()
report = detector.get_report()

print(f"Found outliers in {report['summary']['layers_with_outliers']} layers")
print(f"Total outlier channels: {report['summary']['total_outlier_channels']}")

# Step 2: Analyze worst layers
sorted_layers = sorted(
    report['per_layer'].items(),
    key=lambda x: x[1]['outlier_percentage'],
    reverse=True
)

print("\nTop 5 layers with most outliers:")
for layer_name, info in sorted_layers[:5]:
    print(f"{layer_name}: {info['num_outliers']}/{info['total_channels']} "
          f"({info['outlier_percentage']:.1f}%), "
          f"max_ratio={info['max_ratio']:.1f}×")
```

**Expected Output**:
```
Found outliers in 18 layers
Total outlier channels: 342

Top 5 layers with most outliers:
model.layers.15.self_attn.o_proj: 24/4096 (0.59%), max_ratio=47.2×
model.layers.9.mlp.down_proj: 31/14336 (0.22%), max_ratio=38.1×
model.layers.22.self_attn.o_proj: 18/4096 (0.44%), max_ratio=35.6×
model.layers.11.mlp.gate_proj: 12/14336 (0.08%), max_ratio=29.3×
model.layers.8.self_attn.o_proj: 15/4096 (0.37%), max_ratio=28.7×

Recommendation:
- Layer 15 attention output: Apply SmoothQuant with α=0.75
- Layer 9 MLP: Use per-channel quantization for weights
- Cross-reference with Layer-Level diagnosis: Layer 9-11 is avalanche point!
```

**Cross-Level Correlation Analysis**:
```python
def correlate_layer_and_channel_sensitivity(
    sliding_window_results,
    outlier_report
):
    """
    Find correlation between layer-level sensitivity and outlier channels.

    Hypothesis: Layers with more outliers show higher noise sensitivity.
    """
    correlations = []

    for layer_range, sensitivity in sliding_window_results.items():
        start_layer = sensitivity['start_layer']
        end_layer = sensitivity['end_layer']

        # Count outliers in this range
        total_outliers = 0
        for layer_id in range(start_layer, end_layer + 1):
            layer_key = f"model.layers.{layer_id}"
            for name, info in outlier_report['per_layer'].items():
                if layer_key in name:
                    total_outliers += info['num_outliers']

        correlations.append({
            'layer_range': layer_range,
            'sensitivity_score': sensitivity['sensitivity_score'],
            'total_outliers': total_outliers,
            'degradation': sensitivity['degradation']
        })

    # Compute correlation coefficient
    import numpy as np
    outlier_counts = np.array([c['total_outliers'] for c in correlations])
    sensitivity_scores = np.array([c['sensitivity_score'] for c in correlations])

    correlation = np.corrcoef(outlier_counts, sensitivity_scores)[0, 1]

    print(f"Correlation between outliers and sensitivity: {correlation:.3f}")

    return correlations, correlation
```

### 2.4 Integrated Diagnostic Report

**Combined Analysis**:
```python
class IntegratedDiagnostic:
    """
    Combines all three levels of diagnosis into actionable report.
    """
    def __init__(self, model, test_data, calibration_data):
        self.model = model
        self.test_data = test_data
        self.calibration_data = calibration_data

        self.operator_results = None
        self.layer_results = None
        self.channel_results = None

    def run_full_diagnosis(
        self,
        noise_scale=0.05,
        window_size=3,
        outlier_threshold=10.0
    ):
        """Execute all three diagnostic levels."""
        print("=" * 70)
        print("INTEGRATED NEURAL NETWORK DIAGNOSTIC SUITE")
        print("=" * 70)

        # Level 1: Operator-level
        print("\n[1/3] Running Operator-Level Diagnosis...")
        self.operator_results = operator_level_diagnosis(
            self.model, self.test_data, noise_scale
        )

        # Level 2: Layer-level (sliding window)
        print("\n[2/3] Running Layer-Level Diagnosis (Sliding Window)...")
        self.layer_results = sliding_window_diagnosis(
            self.model, self.test_data, window_size, noise_scale
        )

        # Level 3: Channel-level (outlier detection)
        print("\n[3/3] Running Channel-Level Diagnosis (Outlier Detection)...")
        detector = ChannelOutlierDetector(
            self.model, self.calibration_data, outlier_threshold
        )
        detector.collect_statistics()
        self.channel_results = detector.get_report()

        print("\n" + "=" * 70)
        print("DIAGNOSIS COMPLETE")
        print("=" * 70)

    def generate_report(self):
        """Generate comprehensive diagnostic report."""
        report = {
            'executive_summary': self._generate_executive_summary(),
            'operator_analysis': self.operator_results,
            'layer_analysis': self.layer_results,
            'channel_analysis': self.channel_results,
            'recommendations': self._generate_recommendations()
        }
        return report

    def _generate_executive_summary(self):
        """High-level findings for stakeholders."""
        # Find top issues
        critical_operators = [
            op for op, metrics in self.operator_results.items()
            if metrics['sensitivity_score'] > 3.0
        ]

        avalanche_points = identify_avalanche_points(self.layer_results)

        total_outliers = self.channel_results['summary']['total_outlier_channels']

        return {
            'critical_operators': critical_operators,
            'num_critical_operators': len(critical_operators),
            'avalanche_points': avalanche_points,
            'num_avalanche_points': len(avalanche_points),
            'total_outlier_channels': total_outliers,
            'overall_risk_level': self._compute_risk_level()
        }

    def _compute_risk_level(self):
        """Compute overall deployment risk."""
        score = 0

        # Operator risk
        critical_ops = sum(
            1 for metrics in self.operator_results.values()
            if metrics['sensitivity_score'] > 3.0
        )
        score += critical_ops * 30

        # Layer risk (avalanche points)
        avalanche_points = identify_avalanche_points(self.layer_results)
        score += len(avalanche_points) * 25

        # Channel risk (outliers)
        outlier_percentage = (
            self.channel_results['summary']['total_outlier_channels'] /
            (4096 * 28)  # Assuming typical architecture
        ) * 100
        score += outlier_percentage * 5

        if score > 100:
            return "HIGH - Requires significant hardware validation"
        elif score > 50:
            return "MODERATE - Targeted mitigation recommended"
        else:
            return "LOW - Standard deployment practices sufficient"

    def _generate_recommendations(self):
        """Actionable recommendations based on findings."""
        recommendations = []

        # Operator recommendations
        for op, metrics in self.operator_results.items():
            if metrics['sensitivity_score'] > 3.0:
                recommendations.append({
                    'priority': 'HIGH',
                    'component': f'Operator: {op}',
                    'issue': f"{metrics['degradation']*100:.1f}% accuracy loss",
                    'action': self._get_operator_mitigation(op)
                })

        # Layer recommendations
        avalanche_points = identify_avalanche_points(self.layer_results)
        for ap in avalanche_points:
            recommendations.append({
                'priority': ap['severity'],
                'component': f"Layers: {ap['layer_range']}",
                'issue': f"{ap['degradation']*100:.1f}% degradation (avalanche point)",
                'action': "Increase precision for this layer range or apply targeted AQN training"
            })

        # Channel recommendations
        sorted_layers = sorted(
            self.channel_results['per_layer'].items(),
            key=lambda x: x[1]['outlier_percentage'],
            reverse=True
        )

        for layer_name, info in sorted_layers[:3]:  # Top 3 worst
            if info['outlier_percentage'] > 0.5:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'component': f"Layer: {layer_name}",
                    'issue': f"{info['num_outliers']} outlier channels ({info['outlier_percentage']:.1f}%)",
                    'action': f"Apply SmoothQuant with α=0.75 or use per-channel quantization"
                })

        # Sort by priority
        priority_order = {'HIGH': 0, 'CRITICAL': 0, 'MEDIUM': 1, 'LOW': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 2))

        return recommendations

    def _get_operator_mitigation(self, operator):
        """Get specific mitigation strategy for operator type."""
        strategies = {
            'softmax': "Verify exp() LUT accuracy; consider higher precision for attention scores",
            'layernorm': "Check epsilon value (recommend 1e-5 minimum); validate variance computation",
            'matmul': "Ensure 32-bit accumulators; monitor high-dimensional (d>4096) operations",
            'silu': "Validate sigmoid approximation; consider tanh-based fallback",
            'gelu': "Check polynomial approximation accuracy"
        }
        return strategies.get(operator, "Validate hardware implementation")

    def save_report(self, output_path):
        """Save report to JSON and markdown."""
        import json
        from pathlib import Path

        report = self.generate_report()

        # Save JSON
        json_path = Path(output_path).with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Save Markdown
        md_path = Path(output_path).with_suffix('.md')
        with open(md_path, 'w') as f:
            self._write_markdown_report(f, report)

        print(f"\nReport saved to:")
        print(f"  JSON: {json_path}")
        print(f"  Markdown: {md_path}")

    def _write_markdown_report(self, f, report):
        """Write formatted markdown report."""
        f.write("# Neural Network Diagnostic Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        summary = report['executive_summary']
        f.write(f"**Overall Risk Level**: {summary['overall_risk_level']}\n\n")
        f.write(f"- Critical Operators: {summary['num_critical_operators']}\n")
        f.write(f"- Avalanche Points: {summary['num_avalanche_points']}\n")
        f.write(f"- Outlier Channels: {summary['total_outlier_channels']}\n\n")

        # Recommendations
        f.write("## Priority Recommendations\n\n")
        for i, rec in enumerate(report['recommendations'], 1):
            f.write(f"### {i}. [{rec['priority']}] {rec['component']}\n\n")
            f.write(f"**Issue**: {rec['issue']}\n\n")
            f.write(f"**Action**: {rec['action']}\n\n")

        # Detailed Results
        f.write("## Detailed Analysis\n\n")
        f.write("### Operator-Level Results\n\n")
        f.write("| Operator | Degradation | Sensitivity Score |\n")
        f.write("|----------|-------------|-------------------|\n")
        for op, metrics in sorted(
            report['operator_analysis'].items(),
            key=lambda x: x[1]['sensitivity_score'],
            reverse=True
        ):
            f.write(f"| {op} | {metrics['degradation']*100:.1f}% | "
                   f"{metrics['sensitivity_score']:.2f} |\n")

        # Add more sections as needed...
```

**Usage**:
```python
# Run full diagnostic
diagnostic = IntegratedDiagnostic(model, test_loader, calibration_loader)
diagnostic.run_full_diagnosis(
    noise_scale=0.05,
    window_size=3,
    outlier_threshold=10.0
)

# Get report
report = diagnostic.generate_report()

# Print executive summary
print("\n" + "=" * 70)
print("EXECUTIVE SUMMARY")
print("=" * 70)
print(f"Overall Risk: {report['executive_summary']['overall_risk_level']}")
print(f"\nTop 3 Recommendations:")
for i, rec in enumerate(report['recommendations'][:3], 1):
    print(f"{i}. [{rec['priority']}] {rec['component']}: {rec['action']}")

# Save full report
diagnostic.save_report("diagnostic_report_2026-01-05")
```

---

## 3. Implementation Architecture

### 3.1 Current Implementation Analysis

**Your `noisy_ops.py` provides**:
- ✅ Operator-level injection (MatMul, BMM, Linear, Softmax, SiLU, GeLU, LayerNorm)
- ✅ Forward/backward phase separation (`set_noise_phases()`)
- ✅ Selective layer injection (`set_selective_layers()`)
- ✅ Per-layer injection statistics tracking
- ✅ Environment variable configuration
- ✅ Thread-safe global state management
- ✅ Torch.compile compatibility

**Existing APIs**:
```python
# Core control
enable_noisy_ops(error_scale=0.05, error_type='relative_gaussian', all_ops_mode=False)
disable_noisy_ops()
set_noise_phases(forward=True, backward=False)  # Your E8c experiment

# Layer-wise control
set_selective_layers([0, 5, 10])  # Only inject in these layers
register_layer_hooks(model)  # Enables layer tracking
get_layer_injection_stats()  # Per-layer injection counts

# Statistics
get_injection_stats()  # Global counts by phase
print_injection_summary()
reset_injection_stats()
```

### 3.2 Missing Components for Full Diagnostic Suite

**Need to add**:

#### A. Operator-Type Selection
```python
# NEW API in noisy_ops.py

_SELECTIVE_OPERATORS = None  # None = all, set() = specific operators

def set_selective_operators(operators=None):
    """
    Enable noise injection only for specific operator types.

    Args:
        operators: List of operator names ['matmul', 'softmax', 'layernorm']
                   None = all operators (default)
                   [] = no operators (effectively disables noise)

    Example:
        # Test softmax sensitivity only
        set_selective_operators(['softmax'])
        enable_noisy_ops(error_scale=0.05, all_ops_mode=True)
    """
    global _SELECTIVE_OPERATORS
    if operators is None:
        _SELECTIVE_OPERATORS = None
    else:
        _SELECTIVE_OPERATORS = set(operators)

    print(f"[NoisyOps] Selective operators: {operators}")

def should_inject_for_operator(operator_name):
    """Check if noise should be injected for this operator."""
    if _SELECTIVE_OPERATORS is None:
        return True  # All operators
    return operator_name in _SELECTIVE_OPERATORS

# Modify each NoisyXXX class:
class NoisyMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        result = _ORIGINAL_MATMUL(a, b)
        if (_NOISY_OPS_ENABLED and _NOISY_OPS_FORWARD_ENABLED and
            should_inject_for_layer() and should_inject_for_operator('matmul')):
            error = _compute_error(result)
            result = result + error
            _update_operator_stats('matmul', 'forward')
        return result
```

#### B. Operator-Level Statistics
```python
# NEW tracking in noisy_ops.py

_OPERATOR_INJECTION_STATS = {}  # {operator: {'forward': N, 'backward': M}}

def _update_operator_stats(operator_name, direction):
    """Track injections per operator type."""
    if operator_name not in _OPERATOR_INJECTION_STATS:
        _OPERATOR_INJECTION_STATS[operator_name] = {'forward': 0, 'backward': 0}
    _OPERATOR_INJECTION_STATS[operator_name][direction] += 1

def get_operator_injection_stats():
    """Get per-operator injection statistics."""
    return _OPERATOR_INJECTION_STATS.copy()

def reset_operator_injection_stats():
    """Reset operator statistics."""
    global _OPERATOR_INJECTION_STATS
    _OPERATOR_INJECTION_STATS = {}
```

#### C. Per-Operator Noise Scales
```python
# NEW fine-grained control in noisy_ops.py

_OPERATOR_NOISE_SCALES = {
    'matmul': 1.0,     # Multiplier on base error scale
    'softmax': 0.2,    # 5× more conservative for softmax
    'layernorm': 1.0,
    'silu': 1.0,
    'gelu': 1.0,
}

def set_operator_noise_scales(scale_dict):
    """
    Set per-operator noise scale multipliers.

    Args:
        scale_dict: {operator_name: multiplier}

    Example:
        # Make softmax 5× more conservative
        set_operator_noise_scales({
            'matmul': 1.0,
            'softmax': 0.2,
            'layernorm': 0.8
        })
    """
    global _OPERATOR_NOISE_SCALES
    _OPERATOR_NOISE_SCALES.update(scale_dict)

def _compute_error(tensor, operator_name='matmul'):
    """Compute error with operator-specific scaling."""
    base_scale = _ERROR_SCALE
    operator_scale = _OPERATOR_NOISE_SCALES.get(operator_name, 1.0)
    effective_scale = base_scale * operator_scale

    if _ERROR_TYPE == 'relative_gaussian':
        noise = torch.randn_like(tensor)
        return noise * tensor.abs() * effective_scale
    # ...
```

#### D. Activation Capture for Outlier Detection
```python
# NEW module in verl/utils/activation_capture.py

import torch
from collections import defaultdict

class ActivationCapture:
    """
    Capture and analyze layer activations for outlier detection.

    Usage:
        capture = ActivationCapture(model, layer_patterns=['mlp', 'self_attn'])
        capture.register_hooks()

        for batch in calibration_loader:
            model(batch)

        outliers = capture.detect_outliers(threshold=10.0)
    """
    def __init__(self, model, layer_patterns=None):
        self.model = model
        self.layer_patterns = layer_patterns or ['self_attn.o_proj', 'mlp.down_proj']
        self.activations = defaultdict(list)
        self.hooks = []

    def register_hooks(self):
        """Register forward hooks to capture activations."""
        def make_hook(name):
            def hook(module, input, output):
                # Detach and move to CPU to save memory
                self.activations[name].append(output.detach().cpu())
            return hook

        for name, module in self.model.named_modules():
            if any(pattern in name for pattern in self.layer_patterns):
                handle = module.register_forward_hook(make_hook(name))
                self.hooks.append(handle)

    def remove_hooks(self):
        """Clean up hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def detect_outliers(self, threshold=10.0):
        """
        Detect outlier channels.

        Returns:
            dict: {layer_name: outlier_info}
        """
        outlier_report = {}

        for layer_name, act_list in self.activations.items():
            # Concatenate all captured activations
            activations = torch.cat(act_list, dim=0)  # [total_samples, seq_len, hidden]

            # Per-channel max across all positions and samples
            channel_max = activations.abs().flatten(0, -2).max(dim=0).values  # [C]
            median_max = channel_max.median()

            # Outlier ratio
            outlier_ratio = channel_max / (median_max + 1e-8)
            outlier_mask = outlier_ratio > threshold

            if outlier_mask.any():
                outlier_indices = torch.where(outlier_mask)[0].tolist()
                outlier_report[layer_name] = {
                    'outlier_indices': outlier_indices,
                    'num_outliers': len(outlier_indices),
                    'outlier_ratios': outlier_ratio[outlier_indices].tolist(),
                    'max_ratio': outlier_ratio.max().item()
                }

        return outlier_report
```

### 3.3 Proposed Directory Structure

```
verl/utils/
├── noisy_ops.py                 # Existing operator-level injection
├── activation_capture.py        # NEW: Activation collection & outlier detection
├── diagnostic/                  # NEW: Diagnostic suite
│   ├── __init__.py
│   ├── operator_diagnosis.py   # Level 1: Operator sensitivity
│   ├── layer_diagnosis.py      # Level 2: Sliding window
│   ├── channel_diagnosis.py    # Level 3: Outlier detection
│   ├── integrated_report.py    # Combined analysis
│   └── visualizations.py       # Plotting utilities

scripts/
├── diagnose_model.py           # NEW: CLI for diagnostic suite
└── generate_diagnostic_report.py  # NEW: Report generation

docs/qerl/
└── NOISE_INJECTION_DIAGNOSTIC_METHODOLOGY.md  # This document
```

### 3.4 Integration with Existing Training Pipeline

**Diagnostic Mode vs Training Mode**:

```python
# config/diagnostic_config.yaml
diagnostic:
  enabled: true
  mode: "operator"  # "operator", "layer", "channel", "full"

  operator:
    noise_scale: 0.05
    operators: ["matmul", "softmax", "layernorm"]

  layer:
    window_size: 3
    step_size: 1
    noise_scale: 0.05

  channel:
    calibration_samples: 1000
    outlier_threshold: 10.0
    save_activations: false

# In trainer code
if config.diagnostic.enabled:
    from verl.utils.diagnostic import IntegratedDiagnostic

    diagnostic = IntegratedDiagnostic(model, test_data, calibration_data)

    if config.diagnostic.mode == "full":
        diagnostic.run_full_diagnosis()
        diagnostic.save_report("diagnostic_report")
    elif config.diagnostic.mode == "operator":
        results = operator_level_diagnosis(model, test_data)
        print(results)
    # ... etc
```

### 3.5 Minimal Implementation Roadmap

**Phase 1: Operator Selection (1-2 days)**
- Add `set_selective_operators()` API
- Update all 7 Noisy operator classes
- Add operator-level statistics tracking
- Unit tests for operator selection

**Phase 2: Diagnostic Scripts (2-3 days)**
- Implement `operator_level_diagnosis()`
- Implement `sliding_window_diagnosis()` (reuses existing `set_selective_layers()`)
- Create visualization utilities
- Integration tests

**Phase 3: Activation Capture (2-3 days)**
- Implement `ActivationCapture` class
- Channel outlier detection
- Memory-efficient activation storage
- Integration with diagnostic suite

**Phase 4: Integrated Report (1-2 days)**
- Combine all three levels
- Generate actionable recommendations
- Markdown/JSON report generation
- Documentation

**Total Estimate**: 6-10 days for full implementation

---

## 4. Metrics and Interpretation Guide

### 4.1 Primary Metrics

#### A. Relative Degradation (Layer/Operator Sensitivity)
```python
relative_degradation = (accuracy_clean - accuracy_noisy) / accuracy_clean

# Example: 78% → 64% with 5% noise
# relative_degradation = (78 - 64) / 78 = 0.179 = 17.9%
```

**Interpretation**:
| Degradation | Severity | Action |
|-------------|----------|--------|
| < 5% | LOW | Standard deployment OK |
| 5-10% | MODERATE | Monitor, consider targeted mitigation |
| 10-20% | HIGH | Requires intervention (precision increase or AQN) |
| > 20% | CRITICAL | Major issue, deployment not recommended |

#### B. Noise Tolerance Index
```python
noise_tolerance_index = 1 / relative_degradation

# Higher = more robust
# Example: 17.9% degradation → NTI = 1/0.179 = 5.6
```

**Interpretation**:
| NTI | Robustness | Meaning |
|-----|------------|---------|
| > 20 | EXCELLENT | Tolerates noise well |
| 10-20 | GOOD | Acceptable robustness |
| 5-10 | FAIR | Needs attention |
| < 5 | POOR | Critical sensitivity |

#### C. Critical Threshold
```python
# Binary search to find minimum noise causing >10% degradation
def find_critical_threshold(model, test_data, target_degradation=0.10):
    low, high = 0.0, 0.20
    baseline = evaluate_clean(model, test_data)

    while high - low > 0.001:
        mid = (low + high) / 2
        noisy_acc = evaluate_with_noise(model, test_data, noise_scale=mid)
        degradation = (baseline - noisy_acc) / baseline

        if degradation < target_degradation:
            low = mid
        else:
            high = mid

    return high
```

**Interpretation**:
| Critical Threshold | Brittleness | Deployment Risk |
|--------------------|-------------|-----------------|
| > 0.10 (10%) | LOW | Safe for aggressive quantization |
| 0.05 - 0.10 | MODERATE | Use conservative quantization (INT8) |
| 0.02 - 0.05 | HIGH | Requires mixed precision |
| < 0.02 (2%) | CRITICAL | Avoid quantization for this layer |

### 4.2 Secondary Metrics

#### A. Signal-to-Noise Ratio (SNR)
```python
def compute_snr(activations, noise_scale):
    """
    SNR in dB = 20 * log10(signal_magnitude / noise_magnitude)
    """
    signal_power = activations.abs().mean()
    noise_power = signal_power * noise_scale  # For relative noise
    snr_db = 20 * torch.log10(signal_power / (noise_power + 1e-10))
    return snr_db.item()
```

**Interpretation**:
| SNR (dB) | Quality | Layer Characteristics |
|----------|---------|----------------------|
| > 40 dB | EXCELLENT | Strong signal, minimal noise impact |
| 30-40 dB | GOOD | Normal operating range |
| 20-30 dB | FAIR | Noticeable noise, monitor |
| < 20 dB | POOR | Noise dominates, critical issue |

#### B. Gradient Magnitude Ratio
```python
def compute_gradient_ratio(model, clean_loss, noisy_loss):
    """
    Compare gradient magnitudes: clean vs noisy training.
    """
    clean_grads = torch.autograd.grad(clean_loss, model.parameters())
    noisy_grads = torch.autograd.grad(noisy_loss, model.parameters())

    clean_norm = torch.norm(torch.cat([g.flatten() for g in clean_grads]))
    noisy_norm = torch.norm(torch.cat([g.flatten() for g in noisy_grads]))

    return noisy_norm / clean_norm
```

**Interpretation**:
| Ratio | Meaning | Training Stability |
|-------|---------|-------------------|
| 0.9-1.1 | STABLE | Noise has minimal impact on gradients |
| 0.7-0.9 or 1.1-1.3 | MODERATE | Noticeable but manageable |
| 0.5-0.7 or 1.3-1.5 | CONCERNING | Optimization path significantly altered |
| < 0.5 or > 1.5 | CRITICAL | Gradient noise dominates |

#### C. Activation Sparsity
```python
def compute_sparsity(activations, threshold=1e-6):
    """
    Fraction of near-zero activations.
    """
    return (activations.abs() < threshold).float().mean().item()
```

**Relevance**:
- High sparsity (>70%) indicates many neurons contribute little
- Sparse layers typically MORE robust to noise (redundant pathways)
- Dense layers (<30% sparsity) more sensitive

#### D. Layer-wise Hessian Trace (Advanced)
```python
def compute_layer_hessian_trace(model, loss_fn, data, num_samples=10):
    """
    Estimate Hessian trace per layer using Hutchinson's estimator.
    """
    layer_traces = {}

    for name, param in model.named_parameters():
        if 'weight' not in name:
            continue

        trace_estimate = 0.0
        for _ in range(num_samples):
            v = torch.randn_like(param)

            loss = loss_fn(model, data)
            grads = torch.autograd.grad(loss, param, create_graph=True)
            grad_v = (grads[0] * v).sum()

            hv = torch.autograd.grad(grad_v, param)[0]
            trace_estimate += (v * hv).sum().item()

        layer_traces[name] = trace_estimate / num_samples

    return layer_traces
```

**Interpretation**:
- High trace → High curvature → High sensitivity
- Use for mixed-precision assignment
- Correlates with quantization-induced accuracy loss

### 4.3 Hardware Divergence Metrics

#### A. Mean Absolute Error (MAE)
```python
def compute_mae(gpu_activations, npu_activations):
    return torch.abs(gpu_activations - npu_activations).mean().item()
```

**Thresholds**:
- MAE < 1e-4: Negligible difference
- MAE 1e-4 to 1e-3: Acceptable for most applications
- MAE 1e-3 to 1e-2: Noticeable, investigate
- MAE > 1e-2: Critical divergence

#### B. Mean Absolute Percentage Error (MAPE)
```python
def compute_mape(gpu_activations, npu_activations):
    return (torch.abs(gpu_activations - npu_activations) /
            (gpu_activations.abs() + 1e-8)).mean().item()
```

**Thresholds**:
- MAPE < 1%: Excellent agreement
- MAPE 1-5%: Good, typical for FP16 vs BF16
- MAPE 5-10%: Moderate divergence
- MAPE > 10%: Significant, likely accuracy impact

#### C. Pearson Correlation
```python
def compute_correlation(gpu_activations, npu_activations):
    flat_gpu = gpu_activations.flatten()
    flat_npu = npu_activations.flatten()
    return torch.corrcoef(torch.stack([flat_gpu, flat_npu]))[0, 1].item()
```

**Interpretation**:
- Correlation > 0.99: Excellent match
- Correlation 0.95-0.99: Good agreement
- Correlation 0.90-0.95: Moderate divergence
- Correlation < 0.90: Poor match, investigate

#### D. Maximum Absolute Error
```python
def compute_max_error(gpu_activations, npu_activations):
    return torch.abs(gpu_activations - npu_activations).max().item()
```

**Use case**: Identify worst-case scenarios
- Check if max error occurs in outlier channels
- Cross-reference with channel-level diagnosis

### 4.4 Comprehensive Metric Dashboard

```python
class MetricsDashboard:
    """
    Collect and display all diagnostic metrics in organized format.
    """
    def __init__(self):
        self.metrics = {}

    def compute_all_metrics(
        self,
        model,
        test_data,
        gpu_activations=None,
        npu_activations=None
    ):
        """Compute all available metrics."""

        # Primary metrics
        self.metrics['primary'] = {
            'relative_degradation': self._compute_degradation(model, test_data),
            'noise_tolerance_index': None,  # Computed from degradation
            'critical_threshold': find_critical_threshold(model, test_data)
        }

        self.metrics['primary']['noise_tolerance_index'] = (
            1 / self.metrics['primary']['relative_degradation']
            if self.metrics['primary']['relative_degradation'] > 0
            else float('inf')
        )

        # Secondary metrics
        self.metrics['secondary'] = {
            'snr_db': compute_snr(gpu_activations, noise_scale=0.05),
            'gradient_ratio': None,  # Requires loss computation
            'activation_sparsity': compute_sparsity(gpu_activations)
        }

        # Hardware divergence (if available)
        if gpu_activations is not None and npu_activations is not None:
            self.metrics['hardware'] = {
                'mae': compute_mae(gpu_activations, npu_activations),
                'mape': compute_mape(gpu_activations, npu_activations),
                'correlation': compute_correlation(gpu_activations, npu_activations),
                'max_error': compute_max_error(gpu_activations, npu_activations)
            }

        return self.metrics

    def print_dashboard(self):
        """Pretty-print metrics dashboard."""
        print("\n" + "=" * 70)
        print("METRICS DASHBOARD")
        print("=" * 70)

        # Primary Metrics
        print("\nPrimary Metrics:")
        print("-" * 70)
        primary = self.metrics['primary']
        print(f"  Relative Degradation:   {primary['relative_degradation']*100:6.2f}%")
        print(f"  Noise Tolerance Index:  {primary['noise_tolerance_index']:6.2f}")
        print(f"  Critical Threshold:     {primary['critical_threshold']*100:6.2f}%")

        # Secondary Metrics
        print("\nSecondary Metrics:")
        print("-" * 70)
        secondary = self.metrics['secondary']
        print(f"  SNR:                    {secondary['snr_db']:6.2f} dB")
        print(f"  Activation Sparsity:    {secondary['activation_sparsity']*100:6.2f}%")

        # Hardware Divergence (if available)
        if 'hardware' in self.metrics:
            print("\nHardware Divergence (GPU vs NPU):")
            print("-" * 70)
            hw = self.metrics['hardware']
            print(f"  MAE:                    {hw['mae']:.6f}")
            print(f"  MAPE:                   {hw['mape']*100:6.2f}%")
            print(f"  Correlation:            {hw['correlation']:.6f}")
            print(f"  Max Error:              {hw['max_error']:.6f}")

        print("=" * 70)

    def get_risk_assessment(self):
        """Overall risk assessment based on metrics."""
        risk_score = 0

        # Primary metrics contribute most
        deg = self.metrics['primary']['relative_degradation']
        if deg > 0.20:
            risk_score += 50
        elif deg > 0.10:
            risk_score += 30
        elif deg > 0.05:
            risk_score += 10

        # Hardware divergence
        if 'hardware' in self.metrics:
            mape = self.metrics['hardware']['mape']
            if mape > 0.10:
                risk_score += 30
            elif mape > 0.05:
                risk_score += 15

        # Convert to category
        if risk_score > 60:
            return "HIGH RISK - Not recommended for deployment"
        elif risk_score > 30:
            return "MODERATE RISK - Requires mitigation"
        else:
            return "LOW RISK - Safe for deployment"
```

### 4.5 Cross-Metric Correlation Analysis

**Key Correlations to Validate**:

1. **Hessian Trace vs Noise Sensitivity**
   - Hypothesis: High trace → High degradation
   - Expected correlation: r > 0.7

2. **Outlier Channels vs Layer Sensitivity**
   - Hypothesis: More outliers → Higher degradation
   - Expected correlation: r > 0.6

3. **Activation Sparsity vs Robustness**
   - Hypothesis: High sparsity → Lower degradation
   - Expected correlation: r < -0.5 (negative)

4. **Hardware Divergence vs Accuracy Loss**
   - Hypothesis: MAPE > 5% → Degradation > 10%
   - Critical threshold for intervention

```python
def validate_metric_correlations(diagnostic_results):
    """
    Validate expected correlations between different metric types.
    """
    import numpy as np
    from scipy.stats import pearsonr

    correlations = {}

    # Extract data
    layer_ranges = list(diagnostic_results['layer_results'].keys())

    # 1. Hessian trace vs Sensitivity
    if 'hessian_traces' in diagnostic_results:
        hessian_vals = [diagnostic_results['hessian_traces'][lr]
                       for lr in layer_ranges]
        sensitivity_vals = [diagnostic_results['layer_results'][lr]['sensitivity_score']
                           for lr in layer_ranges]

        r, p = pearsonr(hessian_vals, sensitivity_vals)
        correlations['hessian_sensitivity'] = {
            'r': r, 'p': p,
            'interpretation': 'Strong positive correlation expected (r > 0.7)'
        }

    # 2. Outliers vs Sensitivity
    outlier_counts = [
        sum(1 for name in diagnostic_results['channel_results']['per_layer']
            if f".{lr.split('-')[0]}." in name)
        for lr in layer_ranges
    ]
    sensitivity_vals = [diagnostic_results['layer_results'][lr]['sensitivity_score']
                       for lr in layer_ranges]

    r, p = pearsonr(outlier_counts, sensitivity_vals)
    correlations['outliers_sensitivity'] = {
        'r': r, 'p': p,
        'interpretation': 'Moderate positive correlation expected (r > 0.6)'
    }

    # 3. Sparsity vs Robustness
    if 'sparsity' in diagnostic_results:
        sparsity_vals = [diagnostic_results['sparsity'][lr] for lr in layer_ranges]
        robustness_vals = [1 / (s + 0.01) for s in sensitivity_vals]  # Inverse of sensitivity

        r, p = pearsonr(sparsity_vals, robustness_vals)
        correlations['sparsity_robustness'] = {
            'r': r, 'p': p,
            'interpretation': 'Negative correlation expected (r < -0.5)'
        }

    return correlations
```

---

## 5. Practical Applications

### 5.1 Hardware Migration Planning (GPU → NPU)

**Use Case**: You're migrating a trained model from GPU to NPU and need to validate which layers will cause problems.

**Protocol**:
```python
def plan_hardware_migration(model, test_data, calibration_data):
    """
    Identify critical layers for GPU→NPU migration validation.
    """
    # Step 1: Run diagnostic suite
    diagnostic = IntegratedDiagnostic(model, test_data, calibration_data)
    diagnostic.run_full_diagnosis(noise_scale=0.05)

    # Step 2: Identify critical layers
    report = diagnostic.generate_report()
    avalanche_points = report['executive_summary']['avalanche_points']

    # Step 3: Prioritize validation layers
    critical_layers = []
    for ap in avalanche_points:
        layer_range = ap['layer_range']
        start = int(layer_range.split('-')[0].replace('L', ''))
        end = int(layer_range.split('-')[1])
        critical_layers.extend(range(start, end + 1))

    # Step 4: Add layers with outliers
    outlier_layers = [
        int(name.split('.layers.')[1].split('.')[0])
        for name in report['channel_analysis']['per_layer'].keys()
        if '.layers.' in name
    ]

    critical_layers = sorted(set(critical_layers + outlier_layers))

    # Step 5: Create validation plan
    validation_plan = {
        'priority_1_layers': critical_layers[:len(critical_layers)//3],  # Top 33%
        'priority_2_layers': critical_layers[len(critical_layers)//3:2*len(critical_layers)//3],
        'priority_3_layers': critical_layers[2*len(critical_layers)//3:],
        'recommendations': {
            'precision': 'Use FP16 for priority_1_layers, BF16 for others',
            'validation': 'Compare activations GPU vs NPU for ALL priority_1_layers',
            'fallback': 'If MAPE > 5%, increase precision or apply SmoothQuant'
        }
    }

    return validation_plan
```

**Example Output**:
```
Hardware Migration Validation Plan:
=====================================

Priority 1 Layers (High Risk - Validate First):
  [9, 10, 11, 15, 22]

  → Use FP16 precision
  → Capture activations on GPU and NPU
  → Ensure MAPE < 5% for these layers

Priority 2 Layers (Medium Risk):
  [3, 5, 8, 14, 18]

  → Use BF16 precision
  → Spot-check activations
  → Acceptable if MAPE < 10%

Priority 3 Layers (Low Risk):
  [All others]

  → Standard BF16 or INT8 quantization OK
  → No special validation needed

Validation Script:
  python scripts/validate_hardware_migration.py \
    --gpu-checkpoint checkpoints/model_gpu.pt \
    --npu-checkpoint checkpoints/model_npu.pt \
    --layers 9,10,11,15,22 \
    --max-mape 0.05
```

### 5.2 Adaptive AQN Training

**Use Case**: Apply stronger noise to robust layers, weaker to brittle layers.

**Implementation**:
```python
class AdaptiveAQN:
    """
    Adaptive Quantization Noise: Layer-specific noise scaling based on
    pre-computed sensitivity profile.
    """
    def __init__(self, model, sensitivity_profile):
        """
        Args:
            sensitivity_profile: dict mapping layer_id to sensitivity_score
                                 (from sliding window diagnosis)
        """
        self.model = model
        self.sensitivity_profile = sensitivity_profile
        self.base_noise_scale = 0.05

    def get_layer_noise_scale(self, layer_id):
        """
        Compute adaptive noise scale for given layer.

        Strategy:
        - High sensitivity (>0.15) → 0.5× base scale (protect)
        - Medium sensitivity (0.05-0.15) → 1.0× base scale (normal)
        - Low sensitivity (<0.05) → 2.0× base scale (aggressive)
        """
        sensitivity = self.sensitivity_profile.get(layer_id, 0.10)

        if sensitivity > 0.15:
            multiplier = 0.5  # Protect sensitive layers
        elif sensitivity > 0.05:
            multiplier = 1.0  # Normal
        else:
            multiplier = 2.0  # Aggressive for robust layers

        return self.base_noise_scale * multiplier

    def enable_adaptive_noise(self):
        """
        Enable layer-specific adaptive noise injection.

        This requires modifying noisy_ops.py to support per-layer scales.
        """
        # Store original scale
        from verl.utils import noisy_ops
        original_scale = noisy_ops._ERROR_SCALE

        # Create per-layer scale dictionary
        layer_scales = {
            layer_id: self.get_layer_noise_scale(layer_id)
            for layer_id in range(get_num_layers(self.model))
        }

        # Apply (requires new API in noisy_ops.py)
        noisy_ops.set_per_layer_scales(layer_scales)
        noisy_ops.enable_noisy_ops(error_scale=self.base_noise_scale)

        print(f"[AdaptiveAQN] Enabled layer-specific noise scaling:")
        print(f"  Protected layers (0.5×): {[l for l, s in layer_scales.items() if s < original_scale]}")
        print(f"  Aggressive layers (2.0×): {[l for l, s in layer_scales.items() if s > original_scale]}")
```

**Training Script**:
```python
# Phase 1: Profile sensitivity (one-time)
sensitivity_profile = sliding_window_diagnosis(model, test_data, window_size=3)
sensitivity_dict = {
    int(lr.split('-')[0].replace('L', '')): metrics['sensitivity_score']
    for lr, metrics in sensitivity_profile.items()
}

# Phase 2: Train with adaptive AQN
adaptive_aqn = AdaptiveAQN(model, sensitivity_dict)
adaptive_aqn.enable_adaptive_noise()

# Standard training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        loss = train_step(model, batch)
        loss.backward()
        optimizer.step()
```

**Expected Benefits**:
- 7B model: Improved training stability (+1-2% accuracy)
- 1.5B model: Better robustness retention (target: -7% instead of -14%)

### 5.3 Mixed Precision Quantization Strategy

**Use Case**: Assign per-layer bit precision based on sensitivity.

**Implementation**:
```python
def generate_mixed_precision_config(sensitivity_profile, target_avg_bits=6):
    """
    Assign bit precision to each layer based on sensitivity.

    Strategy:
    - Critical layers (>15% degradation): 16-bit
    - Moderate layers (5-15%): 8-bit
    - Robust layers (<5%): 4-bit

    Adjust to meet target average bit count.
    """
    num_layers = len(sensitivity_profile)

    # Initial assignment
    precision_map = {}
    for layer_id, sensitivity in sensitivity_profile.items():
        if sensitivity > 0.15:
            precision_map[layer_id] = 16
        elif sensitivity > 0.05:
            precision_map[layer_id] = 8
        else:
            precision_map[layer_id] = 4

    # Check average
    current_avg = sum(precision_map.values()) / len(precision_map)

    # Adjust if needed
    if current_avg > target_avg_bits:
        # Too high, need to reduce some layers
        # Sort by sensitivity (ascending)
        sorted_layers = sorted(precision_map.items(), key=lambda x: sensitivity_profile[x[0]])

        for layer_id, current_bits in sorted_layers:
            if current_avg <= target_avg_bits:
                break

            if current_bits == 8 and sensitivity_profile[layer_id] < 0.10:
                precision_map[layer_id] = 4
                current_avg = sum(precision_map.values()) / len(precision_map)

    elif current_avg < target_avg_bits:
        # Too low, can increase some layers
        sorted_layers = sorted(precision_map.items(), key=lambda x: sensitivity_profile[x[0]], reverse=True)

        for layer_id, current_bits in sorted_layers:
            if current_avg >= target_avg_bits:
                break

            if current_bits == 4:
                precision_map[layer_id] = 8
                current_avg = sum(precision_map.values()) / len(precision_map)

    return precision_map

# Example usage
sensitivity_dict = {i: sensitivity_profile[f'L{i}-{i+2}']['sensitivity_score']
                    for i in range(28)}

mixed_precision_config = generate_mixed_precision_config(sensitivity_dict, target_avg_bits=6)

print("Mixed Precision Configuration:")
print(f"Average bits: {sum(mixed_precision_config.values()) / len(mixed_precision_config):.1f}")
print("\n16-bit layers:", [l for l, b in mixed_precision_config.items() if b == 16])
print("8-bit layers:", [l for l, b in mixed_precision_config.items() if b == 8])
print("4-bit layers:", [l for l, b in mixed_precision_config.items() if b == 4])
```

### 5.4 Fault Injection Test Generation

**Use Case**: Generate targeted hardware fault injection tests based on diagnostic findings.

**Implementation**:
```python
class FaultInjectionTestGenerator:
    """
    Generate hardware fault injection tests targeting critical components.
    """
    def __init__(self, diagnostic_report):
        self.report = diagnostic_report

    def generate_test_cases(self):
        """
        Create test cases for hardware validation team.
        """
        test_cases = []

        # Test Case 1: Critical operators
        for op in self.report['executive_summary']['critical_operators']:
            test_cases.append({
                'test_id': f"OP_{op.upper()}",
                'component': f"Operator: {op}",
                'test_type': 'operator_fault',
                'parameters': {
                    'operator': op,
                    'fault_rate': 0.01,  # 1% of operations
                    'fault_type': 'bit_flip'
                },
                'acceptance_criteria': {
                    'max_degradation': 0.05,  # 5% accuracy loss
                    'max_nan_rate': 0.001     # 0.1% NaN outputs
                }
            })

        # Test Case 2: Avalanche points
        for ap in self.report['executive_summary']['avalanche_points']:
            test_cases.append({
                'test_id': f"LAYER_{ap['layer_range']}",
                'component': f"Layers: {ap['layer_range']}",
                'test_type': 'layer_fault',
                'parameters': {
                    'layer_range': ap['layer_range'],
                    'fault_model': 'stuck_at_zero',
                    'fault_duration': '1_step'
                },
                'acceptance_criteria': {
                    'max_degradation': 0.10,
                    'recovery_time': '< 5 steps'
                }
            })

        # Test Case 3: Outlier channels
        for layer, info in list(self.report['channel_analysis']['per_layer'].items())[:5]:
            test_cases.append({
                'test_id': f"CHANNEL_{layer.split('.')[-2]}",
                'component': f"Outlier channels in {layer}",
                'test_type': 'channel_fault',
                'parameters': {
                    'layer': layer,
                    'channel_indices': info['outlier_indices'],
                    'fault_type': 'value_corruption'
                },
                'acceptance_criteria': {
                    'max_degradation': 0.15,
                    'detection_required': True
                }
            })

        return test_cases

    def export_to_csv(self, test_cases, output_path):
        """Export test cases to CSV for hardware team."""
        import csv

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'test_id', 'component', 'test_type', 'parameters', 'acceptance_criteria'
            ])
            writer.writeheader()

            for tc in test_cases:
                writer.writerow({
                    'test_id': tc['test_id'],
                    'component': tc['component'],
                    'test_type': tc['test_type'],
                    'parameters': str(tc['parameters']),
                    'acceptance_criteria': str(tc['acceptance_criteria'])
                })
```

### 5.5 Continuous Integration Monitoring

**Use Case**: Track model robustness regression across training checkpoints.

**Implementation**:
```python
class RobustnessMonitor:
    """
    Monitor robustness metrics during training for CI/CD.
    """
    def __init__(self, baseline_report):
        self.baseline = baseline_report
        self.history = []

    def check_checkpoint(self, model, test_data, checkpoint_step):
        """
        Evaluate current checkpoint against baseline.

        Returns:
            dict: {'pass': bool, 'degradation': float, 'warnings': list}
        """
        # Quick sensitivity check (operator level only for speed)
        current_results = operator_level_diagnosis(model, test_data, noise_scale=0.05)

        warnings = []
        max_degradation = 0

        for op, metrics in current_results.items():
            baseline_deg = self.baseline['operator_analysis'][op]['degradation']
            current_deg = metrics['degradation']

            # Check for regression
            if current_deg > baseline_deg * 1.2:  # 20% worse
                warnings.append({
                    'operator': op,
                    'baseline': baseline_deg,
                    'current': current_deg,
                    'severity': 'WARNING'
                })

            max_degradation = max(max_degradation, current_deg)

        # Overall pass/fail
        passed = len(warnings) == 0 and max_degradation < 0.15

        result = {
            'checkpoint_step': checkpoint_step,
            'pass': passed,
            'max_degradation': max_degradation,
            'warnings': warnings,
            'timestamp': datetime.now().isoformat()
        }

        self.history.append(result)

        return result

    def plot_robustness_trend(self):
        """Plot robustness over training."""
        import matplotlib.pyplot as plt

        steps = [h['checkpoint_step'] for h in self.history]
        degradations = [h['max_degradation'] for h in self.history]

        plt.figure(figsize=(10, 5))
        plt.plot(steps, degradations, marker='o')
        plt.axhline(y=0.15, color='r', linestyle='--', label='Failure Threshold')
        plt.xlabel('Training Step')
        plt.ylabel('Max Degradation')
        plt.title('Model Robustness During Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return plt.gcf()
```

**Integration with Training Pipeline**:
```yaml
# .gitlab-ci.yml or similar
robustness_check:
  stage: validate
  script:
    - python scripts/check_robustness.py \
        --checkpoint checkpoints/latest.pt \
        --baseline-report baseline_robustness.json \
        --max-degradation 0.15
  rules:
    - if: '$CI_COMMIT_BRANCH == "main"'
      when: always
```

---

## 6. Literature Review and Related Work

### 6.1 Fault Injection for Neural Networks

**Key Papers**:

1. **[FKeras: A Sensitivity Analysis Tool for Edge Neural Networks](https://dl.acm.org/doi/10.1145/3665334)** (ACM JATS, 2024)
   - Hessian-based sensitivity analysis for fault injection
   - MSBs most sensitive, LSBs fairly insensitive
   - Bit-level understanding crucial for edge deployment

2. **[Fault Injection and Safe-Error Attack for Extraction](https://arxiv.org/html/2308.16703)** (ESORICS 2023)
   - Full 8-bit quantized models for embedded applications
   - Fault injection can extract model architecture

3. **[Resilience of Deep Learning: A Systematic Literature Review](https://arxiv.org/html/2309.16733v2)** (2024)
   - Survey of 220 papers (Jan 2019 - Mar 2024)
   - Pruning doesn't impact resilience
   - Quantization largely increases resilience (27.4× improvement)

4. **[SpikeFI: A Fault Injection Framework for SNNs](https://arxiv.org/abs/2412.06795)** (Dec 2024)
   - Spiking neural networks fault injection
   - Neuromorphic hardware reliability

**Key Finding**: Quantization can bring a 27.4× reliability increase relative to 32-bit floating-point baseline, validating our noise injection approach as conservative.

### 6.2 Quantization and Outlier Detection

**SmoothQuant and Related Methods**:

1. **[SmoothQuant: Accurate and Efficient Post-Training Quantization](https://arxiv.org/abs/2211.10438)** (ICML 2023)
   - Outliers in activations ~100× larger than typical values
   - Migrate quantization difficulty from activations to weights
   - Mathematically equivalent transformation

2. **[AWQ: Activation-aware Weight Quantization](https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf)** (MLSys 2024)
   - Preserve salient weights based on activation magnitudes
   - Per-channel scaling for important weights

3. **[OmniQuant](https://arxiv.org/pdf/2308.13137)** (ICLR 2024)
   - Learnable Equivalent Transformation (LET)
   - Addresses systematic outliers in specific channels
   - State-of-the-art W4A4 quantization

**Quantitative Benchmarks**:
- SmoothQuant: Enables W8A8 quantization with <1% accuracy loss
- AWQ: 3-bit weight-only quantization outperforms GPTQ
- OmniQuant: 4-bit weight+activation with minimal degradation

### 6.3 Hessian and Fisher Information

**Theoretical Foundations**:

1. **[HAWQ-V2: Hessian Aware trace-Weighted Quantization](https://papers.neurips.cc/paper/2020/file/d77c703536718b95308130ff2e5cf9ee-Paper.pdf)** (NeurIPS 2020)
   - Layer-wise Hessian trace measures sensitivity
   - Average of eigenvalues better than maximum
   - Enables mixed-precision quantization

2. **[FIMA-Q: Post-Training Quantization via FIM Approximation](https://arxiv.org/html/2506.11543)** (2024)
   - Fisher Information Matrix approximation
   - Diagonal FIM replaces expensive Hessian computation
   - Vision transformer quantization

3. **[Universal Statistics of Fisher Information in DNNs](https://proceedings.mlr.press/v89/karakida19a/karakida19a.pdf)** (AISTATS 2019)
   - Fisher Information Matrix evolution during training
   - Spectrum analysis reveals parameter importance

**Key Insight**: Hessian trace and FIM provide complementary views:
- Hessian: Loss landscape curvature (optimization perspective)
- FIM: Parameter information content (statistical perspective)

### 6.4 Information Bottleneck Theory

**Core Theory**:

1. **[Information Bottleneck: Exact Analysis of (Quantized) Neural Networks](https://arxiv.org/abs/2106.12912)** (ICLR 2022)
   - Exact mutual information computation in quantized networks
   - Compression phase depends on activation function
   - Resolves controversy about IB in deep learning

2. **[Information Bottleneck Analysis via Lossy Compression](https://www.researchgate.net/publication/370775416_Information_Bottleneck_Analysis_of_Deep_Neural_Networks_via_Lossy_Compression)** (2023)
   - Lossy compression framework for IB analysis
   - Connects to rate-distortion theory

**Relevance to Noise Injection**:
- Layers with high information redundancy tolerate noise better
- Noise injection acts as information bottleneck test
- Robust layers have completed compression phase

### 6.5 Gradient vs Activation Noise

**Training with Noise**:

1. **[Training with Noise is Equivalent to Tikhonov Regularization](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-tikhonov-nc-95.pdf)** (Bishop, 1995)
   - Foundational result connecting noise to regularization
   - Input noise equivalent to weight decay

2. **[Adding Gradient Noise Improves Learning](https://openreview.net/pdf?id=rkjZ2Pcxe)** (ICLR 2016)
   - Annealed Gaussian gradient noise
   - Adaptive optimizers break equivalence with weight noise
   - Noise adapts to optimization landscape curvature

3. **[Per-example Gradient Regularization](https://link.springer.com/article/10.1007/s10994-024-06661-5)** (Machine Learning, 2024)
   - Suppresses noise memorization
   - Prioritizes signal learning over noise fitting
   - Effective in over-parameterized networks

**Your E8c Experiment Validates**:
- Forward-only noise: -8.6% clean accuracy vs forward+backward
- Confirms gradient noise provides training regularization
- Pending: Forward-only should improve inference robustness

### 6.6 Hardware Error Robustness

**Noise Injection for Hardware Reliability**:

1. **[Bayes-Optimized Noise Injection](https://www.nature.com/articles/s44172-023-00074-3)** (Nature Communications, 2023)
   - 10-100× improvement over state-of-the-art
   - No hardware modifications required
   - Universally deployable across analog hardware

2. **[Hardware-Aware Training for Analog In-Memory Computing](https://www.nature.com/articles/s41467-023-40770-4)** (Nature Communications, 2023)
   - Using expected noise distribution from hardware measurements
   - Superior to simple Gaussian noise
   - Backward pass injection crucial

3. **[Online Quantization Adaptation for Fault Tolerance](https://link.springer.com/chapter/10.1007/978-3-031-40923-3_18)** (2023)
   - Runtime adaptation to counteract faults
   - Combines algorithmic properties with HW features

**Meta's and Google's Experience**:
- Meta: 6 unplanned interruptions from Silent Data Corruption in 54-day pre-training
- Google: SDC event every 1-2 weeks during Gemini training
- Hardware robustness is critical for large-scale training

### 6.7 Layer-wise Sensitivity and Sliding Window

**Diagnostic Methodologies**:

1. **[SAfER: Layer-Level Sensitivity Assessment](https://arxiv.org/html/2308.04753v2)** (2023)
   - Fine-grained weight relevance estimation
   - Attribution methods for layer importance
   - Benchmarking layer-wise sensitivity

2. **[Probabilistic Fault Localization with Sliding Windows](https://link.springer.com/article/10.1007/s11432-012-4567-x)**
   - Sliding window mechanism for fault localization
   - Incremental coverage using weighted bipartite graph
   - Addresses time window inaccuracy

3. **[Online Fault Monitoring with Sliding Window](https://www.sciencedirect.com/science/article/abs/pii/S0149197019303427)**
   - Sliding window for temporal analysis
   - Deep neural network-based monitoring

**Key Insight**: Sliding window captures error propagation and accumulation effects that single-layer injection misses.

### 6.8 Numerical Stability Analysis

**Operator-Level Stability**:

1. **[Accurately Computing Log-Sum-Exp and Softmax](https://academic.oup.com/imajna/article/41/4/2311/5893596)** (IMA Journal of Numerical Analysis, 2021)
   - Forward stable softmax algorithms
   - Relative error bounded by conditioning number
   - Max subtraction trick essential

2. **[Improving Numerical Stability of Fast Matrix Multiplication](https://www.cs.cornell.edu/~arb/papers/fast-matmul-simax2016.pdf)** (SIAM Journal, 2016)
   - Fast matrix multiplication error bounds
   - Numerical sacrifice not prohibitive
   - Column/row scaling improves stability

3. **[DeepStability: Study of Unstable Numerical Methods](https://arxiv.org/pdf/2202.03493)** (2022)
   - Systematic analysis of numerical instability
   - Three rewriting approaches: operations, ordering, epsilon

**Relevance**: Explains why softmax shows higher sensitivity than matmul in operator-level diagnosis.

### 6.9 QeRL and Noise-Enhanced RL

**Revolutionary Finding from NVIDIA**:

1. **[QeRL: Quantization-enhanced Reinforcement Learning](https://arxiv.org/abs/2510.11696)** (2024)
   - Quantization noise IMPROVES RL performance
   - Increases policy entropy and exploration
   - 90.8% on GSM8K vs 88.1% for FP16

**Key Insights**:
- Adaptive Quantization Noise (AQN) provides exploration benefit
- Channel-wise Gaussian perturbations in LayerNorm
- High noise early → broad exploration
- Decreasing noise later → exploit strategies

**Your Work Extends QeRL**:
- QeRL: Robustness to quantization errors
- Your Work: Robustness to HW heterogeneous errors (GPU↔NPU)
- Novel contribution beyond QeRL's original scope

### 6.10 Comparative Summary Table

| Method | Year | Key Contribution | Relevance Score |
|--------|------|------------------|-----------------|
| **QeRL (NVIDIA)** | 2024 | Noise improves RL | ⭐⭐⭐⭐⭐ |
| **SmoothQuant** | 2023 | Outlier mitigation | ⭐⭐⭐⭐⭐ |
| **HAWQ-V2** | 2020 | Hessian trace for mixed-precision | ⭐⭐⭐⭐⭐ |
| **FKeras** | 2024 | Sensitivity analysis tool | ⭐⭐⭐⭐ |
| **Information Bottleneck** | 2022 | Theoretical foundation | ⭐⭐⭐⭐ |
| **Bayes-Optimized Noise** | 2023 | Hardware robustness | ⭐⭐⭐⭐ |
| **Gradient Noise (ICLR)** | 2016 | Training dynamics | ⭐⭐⭐⭐ |
| **AWQ** | 2024 | Activation-aware quantization | ⭐⭐⭐ |
| **SAfER** | 2023 | Layer sensitivity assessment | ⭐⭐⭐ |

---

## 7. Implementation Gaps and Roadmap

### 7.1 Current Capabilities (Your `noisy_ops.py`)

✅ **Implemented**:
- Operator-level injection (7 operators)
- Forward/backward phase separation
- Selective layer injection
- Per-layer statistics tracking
- Environment variable configuration
- Thread-safe state management
- Torch.compile compatibility

### 7.2 Missing Components

❌ **Critical Gaps**:

1. **Operator-Type Selection**
   - Cannot inject noise only in softmax while keeping matmul clean
   - Need: `set_selective_operators(['softmax'])`

2. **Per-Operator Noise Scales**
   - All operators use same error scale
   - Need: Different scales for softmax (conservative) vs matmul

3. **Activation Capture Infrastructure**
   - No built-in activation collection for outlier detection
   - Need: Hook-based activation capture system

4. **Sliding Window Automation**
   - Manual loop required for sliding window analysis
   - Need: `sliding_window_diagnosis()` utility

5. **Integrated Reporting**
   - Results scattered across multiple calls
   - Need: Unified diagnostic report generator

### 7.3 Implementation Roadmap

#### Phase 1: Core Diagnostic APIs (Week 1)

**Goal**: Enable operator-level and sliding window diagnosis.

**Tasks**:
```
Day 1-2: Operator Selection API
- Add _SELECTIVE_OPERATORS global state
- Implement set_selective_operators()
- Update all 7 NoisyXXX classes
- Add operator-level statistics tracking
- Unit tests

Day 3-4: Sliding Window Utilities
- Implement sliding_window_diagnosis() function
- Visualization utilities (heatmap plots)
- Avalanche point detection
- Integration tests

Day 5: Documentation and Examples
- Update API documentation
- Create example scripts
- Add to existing tutorials
```

**Deliverables**:
- ✅ `verl/utils/noisy_ops.py` updated with operator selection
- ✅ `verl/utils/diagnostic/layer_diagnosis.py` with sliding window
- ✅ Example script: `scripts/diagnose_layer_sensitivity.py`
- ✅ Unit tests with >90% coverage

#### Phase 2: Activation Capture (Week 2)

**Goal**: Channel-level outlier detection.

**Tasks**:
```
Day 1-2: Activation Capture Module
- Create verl/utils/activation_capture.py
- Hook registration system
- Memory-efficient storage (CPU offload)
- Batch processing

Day 3-4: Outlier Detection
- Statistical methods (quantile, ratio)
- Fisher Information integration
- SmoothQuant scaling factor computation
- Visualization (per-layer bar charts)

Day 5: Integration with Diagnostic Suite
- Add to IntegratedDiagnostic class
- Cross-level correlation analysis
- Documentation
```

**Deliverables**:
- ✅ `verl/utils/activation_capture.py`
- ✅ `verl/utils/diagnostic/channel_diagnosis.py`
- ✅ Example: `scripts/detect_outlier_channels.py`
- ✅ Integration tests

#### Phase 3: Integrated Reporting (Week 3)

**Goal**: End-to-end diagnostic pipeline with actionable reports.

**Tasks**:
```
Day 1-2: Report Generation
- IntegratedDiagnostic class
- JSON/Markdown export
- Metrics dashboard
- Risk assessment scoring

Day 3-4: Visualization Suite
- Matplotlib/Seaborn plots
- Interactive dashboards (optional: Plotly)
- PDF report generation
- CI/CD integration

Day 5: End-to-end testing
- Full diagnostic runs on sample models
- Performance optimization
- Documentation finalization
```

**Deliverables**:
- ✅ `verl/utils/diagnostic/integrated_report.py`
- ✅ `verl/utils/diagnostic/visualizations.py`
- ✅ CLI tool: `scripts/diagnose_model.py`
- ✅ Example reports in docs/

#### Phase 4: Advanced Features (Week 4)

**Goal**: Practical applications and optimizations.

**Tasks**:
```
Day 1-2: Adaptive AQN
- Per-layer noise scaling
- Sensitivity-guided training
- Example training scripts

Day 3: Mixed Precision Config Generator
- Bit precision assignment
- Hardware migration planner
- Validation script

Day 4: CI/CD Integration
- Robustness regression tests
- GitHub Actions / GitLab CI templates
- Automated reporting

Day 5: Performance Optimization
- Caching/memoization
- Parallel evaluation
- Memory profiling
```

**Deliverables**:
- ✅ Adaptive AQN training examples
- ✅ Mixed-precision config generator
- ✅ CI/CD templates
- ✅ Performance benchmarks

### 7.4 Estimated Resource Requirements

**Development Time**:
- Phase 1: 40 hours (1 week)
- Phase 2: 40 hours (1 week)
- Phase 3: 40 hours (1 week)
- Phase 4: 40 hours (1 week)
- **Total**: 160 hours (~1 month for 1 developer)

**Compute Resources**:
- Diagnostic runs: ~2-4 GPU hours per model
- Sensitivity profiling: ~8-16 GPU hours (full sweep)
- Activation capture: Minimal (CPU-bound)

**Storage**:
- Activation data: ~5-10 GB per model (1000 calibration samples)
- Reports: Negligible (<1 MB per report)

### 7.5 Minimum Viable Implementation

**If time-constrained, prioritize**:

1. **Operator selection** (enables Level 1 diagnosis)
2. **Sliding window utilities** (enables Level 2 diagnosis)
3. **Basic reporting** (JSON export)

**Can defer**:
- Advanced visualizations
- CI/CD integration
- Adaptive AQN training (research)

**MVP Estimated Time**: 2 weeks (80 hours)

### 7.6 Integration Checklist

Before considering implementation complete:

- [ ] All APIs have docstrings with examples
- [ ] Unit tests achieve >85% coverage
- [ ] Integration tests for end-to-end workflows
- [ ] Performance benchmarks documented
- [ ] Example scripts in `scripts/` directory
- [ ] Documentation in `docs/qerl/`
- [ ] README updated with new features
- [ ] Changelog entry added
- [ ] Backward compatibility maintained
- [ ] Thread safety verified
- [ ] Memory leaks checked (activation capture)
- [ ] Works with torch.compile (noisy_ops compatibility)

---

## Conclusion

This comprehensive methodology transforms noise injection from a simple "vaccine" (robustness training) into a powerful "oscilloscope" (diagnostic tool) for neural network error localization. The three-level approach:

1. **Operator-Level**: Identifies which operations (MatMul, Softmax, LayerNorm) are most sensitive
2. **Layer-Level**: Reveals "avalanche points" where errors compound via sliding window analysis
3. **Channel-Level**: Detects outlier channels that cause quantization difficulty

**Theoretical Foundations**:
- Information Bottleneck Theory explains why different layers have different robustness
- Fisher Information Matrix and Hessian Trace provide mathematical sensitivity measures
- Gradient vs Activation Noise distinction clarifies training vs inference effects

**Practical Impact**:
- Hardware Migration Planning: Prioritize validation efforts
- Adaptive AQN Training: Apply noise intelligently based on sensitivity
- Mixed Precision Strategy: Assign bit precision optimally
- CI/CD Monitoring: Track robustness regression during development

**Novel Contribution**:
Your work extends QeRL (NVIDIA 2024) from quantization robustness to HW heterogeneous error robustness (GPU↔NPU), addressing a critical gap in large-scale LLM deployment.

**Implementation Status**:
- ✅ Core operator-level injection (complete)
- ✅ Layer-wise selective injection (complete)
- ✅ Forward/backward phase separation (complete)
- ⏳ Operator-type selection (pending)
- ⏳ Activation capture system (pending)
- ⏳ Integrated diagnostic suite (pending)

**Estimated Timeline**: 1 month for full implementation, 2 weeks for MVP.

---

## References

### Theory
1. [Information Bottleneck: Exact Analysis of (Quantized) Neural Networks](https://arxiv.org/abs/2106.12912) (ICLR 2022)
2. [HAWQ-V2: Hessian Aware trace-Weighted Quantization](https://papers.neurips.cc/paper/2020/file/d77c703536718b95308130ff2e5cf9ee-Paper.pdf) (NeurIPS 2020)
3. [FIMA-Q: Post-Training Quantization via FIM](https://arxiv.org/html/2506.11543) (2024)
4. [Training with Noise = Tikhonov Regularization](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-tikhonov-nc-95.pdf) (Bishop, 1995)
5. [Adding Gradient Noise Improves Learning](https://openreview.net/pdf?id=rkjZ2Pcxe) (ICLR 2016)

### Quantization
6. [SmoothQuant](https://arxiv.org/abs/2211.10438) (ICML 2023)
7. [AWQ: Activation-aware Weight Quantization](https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf) (MLSys 2024)
8. [OmniQuant](https://arxiv.org/pdf/2308.13137) (ICLR 2024)

### Fault Injection
9. [FKeras: Sensitivity Analysis Tool](https://dl.acm.org/doi/10.1145/3665334) (ACM JATS, 2024)
10. [Resilience of Deep Learning: Systematic Review](https://arxiv.org/html/2309.16733v2) (2024)
11. [Fault Injection and Safe-Error Attack](https://arxiv.org/html/2308.16703) (ESORICS 2023)

### Hardware Robustness
12. [Bayes-Optimized Noise Injection](https://www.nature.com/articles/s44172-023-00074-3) (Nature Comm., 2023)
13. [Hardware-Aware Training for Analog Computing](https://www.nature.com/articles/s41467-023-40770-4) (Nature Comm., 2023)
14. [Online Quantization Adaptation for Fault Tolerance](https://link.springer.com/chapter/10.1007/978-3-031-40923-3_18) (2023)

### RL and Noise
15. [QeRL: Quantization-enhanced RL](https://arxiv.org/abs/2510.11696) (NVIDIA, 2024)
16. [Per-example Gradient Regularization](https://link.springer.com/article/10.1007/s10994-024-06661-5) (Machine Learning, 2024)

### Numerical Stability
17. [Accurately Computing Softmax](https://academic.oup.com/imajna/article/41/4/2311/5893596) (IMA JNA, 2021)
18. [Improving Stability of Fast MatMul](https://www.cs.cornell.edu/~arb/papers/fast-matmul-simax2016.pdf) (SIAM, 2016)
19. [DeepStability: Unstable Numerical Methods](https://arxiv.org/pdf/2202.03493) (2022)

### Diagnosis
20. [SAfER: Layer-Level Sensitivity Assessment](https://arxiv.org/html/2308.04753v2) (2023)
21. [Probabilistic Fault Localization with Sliding Windows](https://link.springer.com/article/10.1007/s11432-012-4567-x)
22. [Online Fault Monitoring with Sliding Window](https://www.sciencedirect.com/science/article/abs/pii/S0149197019303427)

---

**Document Version**: 1.0
**Last Updated**: 2026-01-05
**Status**: Definitive Methodology - Ready for Implementation
