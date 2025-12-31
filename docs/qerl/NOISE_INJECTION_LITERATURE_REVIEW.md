# Noise Injection for Robustness: Literature Review

**Date**: 2025-12-31
**Purpose**: Survey of related work on noise injection for robustness in RL, LLM, Embodied AI, and Hardware domains

---

## Executive Summary

This literature review surveys research on noise injection during training to improve model robustness. The key finding is that **noise injection is a well-established technique across multiple domains**, and our approach using Gaussian noise for HW heterogeneous error simulation is methodologically sound and conservative.

**Key Takeaway**: QeRL (NVIDIA 2024) already demonstrated that quantization noise **improves** RL performance via exploration benefits. Our work extends this finding to HW heterogeneous errors.

---

## 1. Reinforcement Learning Domain

### 1.1 Parameter Noise vs Action Noise

**OpenAI - "Parameter Space Noise for Exploration" (ICLR 2018)**
- Adding adaptive noise to neural network parameters (not action space) leads to more consistent exploration
- Parameter noise is superior to action noise for both off-policy (DQN, DDPG) and on-policy (TRPO) methods
- Achieves 2x score improvement on HalfCheetah after 20 episodes

**Action Noise Study (2022)**
- Neither Gaussian nor Ornstein-Uhlenbeck noise is universally superior
- **Noise scale σ is critical** - optimal values vary dramatically by environment
- Using schedulers to reduce noise over time decreases variance and increases robustness

### 1.2 Noise as Regularization

**Theoretical Framework:**
- Noise prevents networks from memorizing training samples
- Results in smaller weights and lower generalization error
- Training on perturbed data forces models to find robust solutions

**TD3 (Twin Delayed DDPG):**
- Adds clipped noise to target actions as regularization
- Smooths Q-function, avoiding overfitting to narrow peaks
- More robust policies, especially in noisy environments

### 1.3 Domain Randomization

**Robust Domain Randomization (ICLR 2020)**
- Minimizing policy's Lipschitz constant with respect to randomization parameters
- More efficient than standard domain randomization

**Understanding Domain Randomization (ICLR 2022)**
- Domain randomization acts as regularization preventing overfitting to individual simulation instances
- Physical dynamics randomization includes: observation noise, motor efficiency, actuation delays, friction

### 1.4 Key Findings

| Finding | Implication for Our Work |
|---------|-------------------------|
| Parameter noise > action noise | Supports operator-level injection |
| Noise scale is environment-specific | Need to tune scale (we use 5e-2) |
| Progressive scheduling helps | AQN already does this |
| SAC/TD3 more robust than DDPG | Entropy/noise regularization works |

### 1.5 Noise Type Comparison

**Alpha-Stable Noise Study (2024)**
- Alpha-stable (non-Gaussian impulsive) noise more effective than Gaussian
- Training with alpha-stable noise outperforms Gaussian when test data has impulsive noise
- **Conclusion**: Noise type should match expected real-world corruption

**Gaussian vs Uniform:**
- No universal winner; choice depends on task characteristics
- Gaussian is more common due to Central Limit Theorem

---

## 2. LLM Quantization Domain

### 2.1 Quantization-Aware Training (QAT)

**LLM-QAT (ACL 2024)**
- Data-free QAT using distillation from pre-trained model
- Quantizes weights, activations, and KV cache to 4-bits
- Tested on LLaMA models (7B, 13B, 30B)

**EfficientQAT (ACL 2025)**
- 2-bit Llama-2-70B on single A100-80GB in 41 hours
- Less than 3 points accuracy degradation vs full precision

**PyTorch QAT (2024)**
- Recovers 96% of accuracy degradation on HellaSwag
- **Key finding**: Disabling fake quantization for first 1000 steps yields better results

### 2.2 Noise Injection Methods

**NICE - Noise Injection and Clamping Estimation**
- Uniform additive noise emulates quantization noise at inference
- Verified for weight quantization as coarse as 5 bits
- Advantage: Updates immediately influence forward pass

**UNIQ - Uniform Noise Injection for Quantization**
- Achieves SOTA with no accuracy degradation for MobileNet/ResNet-18
- 2-bit activation quantization on ImageNet

**DiffQ - Differentiable Model Compression (Facebook 2021)**
- Pseudo quantization noise that is differentiable
- Does NOT require Straight-Through Estimator (STE)
- **8x compression with only 0.5 perplexity loss** on Wikitext-103

**Quant-Noise (Facebook AI 2020)**
- Quantizes random subset of weights during training (not entire network)
- More stable for high compression schemes
- Training with 0.05-0.2 Quant-Noise recommended

### 2.3 QeRL: Quantization-Enhanced RL (NVIDIA 2024)

**Revolutionary Finding:**
> Quantization noise, traditionally viewed as detrimental, actually **improves RL performance** by increasing policy entropy and enhancing exploration.

**Mechanism:**
- NVFP4 (4-bit) + LoRA
- Deterministic FP4 quantization raises policy entropy
- Flattens token distributions early in training

**Adaptive Quantization Noise (AQN):**
- High noise early → broad exploration
- Decreasing noise later → exploit discovered strategies
- Channel-wise Gaussian perturbations in LayerNorm

**Results:**
| Benchmark | QeRL | 16-bit LoRA | QLoRA |
|-----------|------|-------------|-------|
| GSM8K (Qwen2.5-7B) | **90.8%** | 88.1% | 85.0% |
| AMC 23 (14B) | **57.5%** | 55.0% | - |

**This directly validates our hypothesis that noise injection helps RL!**

### 2.4 Stochastic vs Deterministic Quantization

| Property | Stochastic | Deterministic |
|----------|------------|---------------|
| Regularization | ✅ Yes | ❌ Limited |
| Exploration | ✅ Better | ❌ Less |
| Adversarial robustness | ✅ 43% better | ❌ Baseline |
| Inference consistency | ❌ Non-deterministic | ✅ Predictable |
| Production use | ❌ Impractical | ✅ Suitable |

---

## 3. Embodied AI / Sim-to-Real Transfer

### 3.1 Foundational Work

**Tobin et al. (2017) - Domain Randomization**
- First successful sim-to-real transfer using only simulated RGB images
- Object localization accurate to 1.5cm
- "With enough variability in simulation, real world appears as just another variation"

**Peng et al. (2018) - Dynamics Randomization**
- Randomized friction, mass, contact models during training
- Policies adapt to varying dynamics without real-world training
- Successfully demonstrated on 7-DOF Fetch robotic arm

### 3.2 Key Randomization Parameters

| Category | Parameters | Typical Range |
|----------|------------|---------------|
| Physical | Friction, mass, inertia | Wide variation |
| Sensor | Gaussian observation noise | σ = 0.01-0.20 |
| Actuator | Motor efficiency, latency, stiffness | 10-1000 |
| Environment | Lighting, textures, camera poses | Domain-specific |

### 3.3 Major Success Cases

**OpenAI Rubik's Cube (2019)**
- Automatic Domain Randomization (ADR)
- 13,000 years of simulated experience
- **60% success rate** on real robot
- Robust to perturbations never seen during training

**DART - Disturbances for Augmenting Robot Trajectories (Berkeley 2017)**
- **62% performance increase** over standard Behavior Cloning
- Injects noise into supervisor demonstrations

**Grasping with Chopsticks (UW)**
- **28% improvement** in physical robot success with noise injection
- Reduces covariate shift in behavior cloning

### 3.4 Key Conclusions

1. **Diversity over accuracy**: Wide randomization ranges beat precise modeling
2. **Combined approach**: Physical + visual + temporal randomization together
3. **Architecture matters**: LSTM policies achieve 0.89 real-world success vs lower for feedforward
4. **Progressive scheduling**: Start conservative, gradually add randomization

---

## 4. Hardware Error Robustness

### 4.1 Scale of the Problem

**Meta's Report:**
- 6 unplanned job interruptions from Silent Data Corruption (SDC) during 54-day pre-training

**Google's Estimate:**
- SDC event occurs every week or two during Gemini training

### 4.2 Noise Injection for Hardware Robustness

**Bayes-Optimized Noise Injection (Nature Communications 2023)**
- **10-100x improvement** over state-of-the-art
- No hardware modifications or accuracy sacrifice
- Universally deployable across different analog hardware
- Works for image classification, object detection, autonomous driving

**Hardware-Aware Training (Nature Communications 2023)**
- Various DNNs (CNNs, RNNs, transformers) can be robustly deployed on analog in-memory computing
- **Key finding**: Using expected noise distribution from hardware measurements is **superior to simple Gaussian noise**
- Injection of noise on weights crucial during backward pass

**Tiki-Taka Algorithm (TTv2, Frontiers 2021)**
- **100x improvement** in noise tolerance for analog hardware
- Reduces device conductance states requirement from 1000s to only 10s

**Knowledge Distillation + Noise Injection**
- **2x greater noise tolerance** vs previous best attempts
- Significant step toward practical analog hardware for deep learning

### 4.3 Fault-Aware Training (FAT)

- Improves DNN robustness to faults by **factor of 3**
- Injects faults during training (especially in convolutional layers)
- Particularly valuable for harsh environments with high radiation

### 4.4 GPU vs NPU Numerical Differences

**Key Challenges:**
- GPU results never exactly reproducible due to:
  - Rounding errors of floating-point arithmetic
  - Non-deterministic thread scheduling
  - Non-associativity of floating-point operations
- Each run produces unique model weights even with identical inputs

**Precision Differences:**
| Hardware | Common Formats | Notes |
|----------|---------------|-------|
| GPU (NVIDIA) | FP32, FP16, BF16, INT8 | Flexible precision |
| TPU | BF16, INT8 | Optimized for BF16 |
| NPU | BF16, INT8 | Architecture-specific |

**Gap in Literature:** Limited research specifically on numerical reproducibility differences between GPU and NPU during training.

---

## 5. Comparison: Random Noise vs Deterministic Errors

### 5.1 Key Differences

| Property | Deterministic (Quantization) | Stochastic (Our Approach) |
|----------|------------------------------|---------------------------|
| Same input → same error | ✅ Yes | ❌ No (random each time) |
| Bounded | ✅ Yes (by quant step) | ❌ No (Gaussian tail) |
| Systematic bias | ✅ Often present | ❌ Zero mean |
| Model can learn pattern | ✅ Yes | ❌ No |
| Train-inference match | ✅ Exact | ❌ Different noise |

### 5.2 Implications

**Stochastic noise is HARDER than deterministic:**
1. Model cannot compensate for consistent patterns
2. Gradients randomly perturbed (optimizer can't work around it)
3. Each forward pass sees different errors

**Conservative Interpretation:**
- Success with Gaussian noise → **over-robust** for real quantization
- Our approach is an **upper bound** on difficulty
- Results should transfer (or exceed) to real NVFP4 scenarios

### 5.3 When Stochastic Approximation Works

- **Effective for ≥5-bit quantization** (NICE, UNIQ studies)
- **Uniform noise better than Gaussian** for quantization (theoretically aligned)
- **At <4 bits**: interactions become non-linear, harder to approximate

---

## 6. Summary: Relevance to Our Experiments

### 6.1 Validation of Methodology

| Aspect | Literature Support |
|--------|-------------------|
| Noise injection for robustness | ✅ Well-established across all domains |
| Gaussian noise approximation | ✅ Valid, conservative (harder than real) |
| Noise in backward pass | ✅ Recommended (hardware-aware training) |
| Progressive scheduling (AQN) | ✅ Best practice |
| Operator-level injection | ✅ Supported by parameter noise studies |

### 6.2 Expected Results Based on Literature

| Experiment | Expected Outcome | Basis |
|------------|------------------|-------|
| E5 (noise only) | Degradation | Standard finding: noise without mitigation hurts |
| E5a (noise + AQN) | Better than E5 | QeRL: AQN helps via exploration |
| E5a robustness | Clean ≈ Noisy | Sim-to-real: noise training → robustness |

### 6.3 Novel Contribution

If E5a succeeds, our contribution extends QeRL:

> **QeRL**: AQN provides robustness to **quantization errors**
>
> **Our Work**: AQN provides robustness to **HW heterogeneous errors** (GPU↔NPU)

This is a novel application beyond QeRL's original scope.

---

## 7. Key Papers by Domain

### Reinforcement Learning
1. Plappert et al. (2018) - Parameter Space Noise for Exploration (OpenAI)
2. Robust Domain Randomization (ICLR 2020)
3. Understanding Domain Randomization (ICLR 2022)

### LLM Quantization
1. **QeRL (NVIDIA 2024)** - Quantization-enhanced RL ⭐ Most relevant
2. DiffQ (Facebook 2021) - Differentiable pseudo quantization noise
3. Quant-Noise (Facebook 2020) - Random subset quantization
4. NICE/UNIQ - Uniform noise injection

### Embodied AI / Sim-to-Real
1. Tobin et al. (2017) - Domain Randomization (OpenAI/Berkeley)
2. OpenAI Rubik's Cube (2019) - Automatic Domain Randomization
3. DART (Berkeley 2017) - Noise injection for imitation learning

### Hardware Robustness
1. Bayes-Optimized Noise Injection (Nature Comm. 2023)
2. Hardware-Aware Training (Nature Comm. 2023)
3. FAT - Fault-Aware Training
4. Understanding Hardware Failures in DL Training (ISCA 2023)

---

## References

### RL Domain
- [Parameter Space Noise for Exploration](https://arxiv.org/abs/1706.01905)
- [Robust Domain Randomization](https://openreview.net/forum?id=H1xSOTVtvH)
- [Understanding Domain Randomization](https://openreview.net/pdf?id=T8vZHIRTrY)
- [Alpha-Stable Training Noise](https://www.sciencedirect.com/science/article/abs/pii/S1051200424004032)

### LLM Quantization
- [QeRL: Quantization-enhanced RL](https://arxiv.org/abs/2510.11696)
- [DiffQ: Differentiable Model Compression](https://arxiv.org/abs/2104.09987)
- [Training with Quantization Noise](https://arxiv.org/abs/2004.07320)
- [NICE: Noise Injection and Clamping](https://www.mdpi.com/2227-7390/9/17/2144)
- [UNIQ: Uniform Noise Injection](https://arxiv.org/abs/1804.10969v1)

### Embodied AI
- [Domain Randomization for Sim-to-Real](https://arxiv.org/abs/1703.06907)
- [Solving Rubik's Cube with Robot Hand](https://arxiv.org/abs/1910.07113)
- [DART: Noise Injection for Imitation](https://arxiv.org/abs/1703.09327)
- [Sim-to-Real with Dynamics Randomization](https://arxiv.org/abs/1710.06537)

### Hardware Robustness
- [Bayes-Optimized Noise Injection](https://www.nature.com/articles/s44172-023-00074-3)
- [Hardware-Aware Training for Diverse DL](https://www.nature.com/articles/s41467-023-40770-4)
- [Understanding Hardware Failures in DL Training](https://dl.acm.org/doi/10.1145/3579371.3589105)
- [Enabling Training on Noisy Hardware](https://www.frontiersin.org/articles/10.3389/frai.2021.699148)
