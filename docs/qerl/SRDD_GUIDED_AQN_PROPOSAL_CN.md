# SRDD引导的AQN训练策略研究提案

**版本**: 0.2 (实验完成)
**日期**: 2026-01-08
**状态**: 初步验证完成

---

## 摘要

本提案探索将 **SRDD诊断结果** 用于指导 **AQN噪声注入策略** 的可能性。核心思想是：利用SRDD识别硬件敏感层，然后在训练时对这些层注入更多噪声，使模型在关键位置获得更强的鲁棒性。

---

## 1. 问题背景

### 1.1 当前AQN的局限性

当前AQN采用**均匀注入**策略：

```python
# 当前实现：所有层使用相同的σ
for layer in model.layers:
    output = layer(input) + σ * randn_like(output) * |output|
```

| 问题 | 影响 |
|-----|------|
| 所有层相同σ | 敏感层可能训练不足，稳定层浪费计算 |
| 无硬件针对性 | 无法针对特定NPU/GPU的问题层优化 |
| 盲目注入 | 不知道哪些层真正需要鲁棒性训练 |

### 1.2 SRDD提供的信息

SRDD诊断可以提供：

| SRDD输出 | 含义 | AQN启示 |
|---------|------|--------|
| 高峰度层 | 激活值分布尖锐，易饱和 | 需要更多噪声训练 |
| 低增益层 | 信号传递弱，易丢失 | 需要更多噪声训练 |
| 高不稳定层 | 已经存在硬件噪声 | 可能需要降低注入 |
| SAT_PROP层 | 受上游饱和影响 | 需要在SOURCE层注入 |

---

## 2. 提案方案

### 2.1 核心思想

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: SRDD诊断目标硬件 (NPU)                             │
│          → 识别敏感层 (e.g., L10, L15, L22)                 │
│          → 量化敏感度 (drop_z, gain, kurtosis)              │
├─────────────────────────────────────────────────────────────┤
│  Step 2: 生成层级AQN配置                                    │
│          → 敏感层: σ × 2.0                                  │
│          → 稳定层: σ × 1.0                                  │
│          → 已噪声层: σ × 0.5                                │
├─────────────────────────────────────────────────────────────┤
│  Step 3: 针对性AQN训练                                      │
│          → 模型在关键位置学习鲁棒性                          │
│          → 减少无效层的计算开销                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 SRDD指标 → AQN策略映射

| SRDD发现 | AQN策略 | 理由 |
|---------|--------|------|
| 高峰度层 (kurtosis > 4000) | σ × 1.5~2.0 | 易受饱和影响 |
| 低增益层 (gain < 0.8) | σ × 1.5 | 信号丢失风险 |
| 高基线不稳定层 | σ × 0.5 | 已有硬件噪声 |
| SAT_PROP传播层 | 在SOURCE层↑σ | 根源治理 |
| 正常层 | σ × 1.0 | 基础鲁棒性 |

---

## 3. 实现设计

### 3.1 `srdd_guided_aqn.py` 模块

```python
#!/usr/bin/env python3
"""
SRDD-Guided AQN: Use SRDD diagnosis to configure layer-specific noise injection.
"""

import numpy as np
from typing import Dict, List
from scripts.srdd_error_finder import SRDDErrorFinder


class SRDDGuidedAQN:
    """Use SRDD diagnosis to configure layer-specific AQN."""

    def __init__(self, model, tokenizer):
        self.finder = SRDDErrorFinder(model, tokenizer)
        self.num_layers = self.finder.num_layers

    def analyze_hardware_sensitivity(
        self,
        calibration_prompts: List[str] = None,
    ) -> Dict:
        """
        Run SRDD scans to identify hardware-sensitive layers.

        Returns:
            Dictionary with gain, kurtosis, and instability per layer
        """
        if calibration_prompts is None:
            calibration_prompts = [
                "What is 2 + 2?",
                "Explain machine learning briefly.",
                "Write a Python hello world.",
            ]

        # Run diagnostic scans
        gain_results = self.finder.local_gain_scan(
            calibration_prompts, noise_scale=0.1
        )
        kurtosis_results = self.finder.local_kurtosis_scan(calibration_prompts)
        ambient_results = self.finder.local_ambient_scan(
            calibration_prompts, num_trials=3
        )

        return {
            'gain': gain_results,
            'kurtosis': kurtosis_results,
            'instability': ambient_results,
        }

    def generate_layer_sigmas(
        self,
        sensitivity_results: Dict,
        base_sigma: float = 0.05,
    ) -> Dict[int, float]:
        """
        Generate layer-specific sigma values based on SRDD results.

        Args:
            sensitivity_results: Output from analyze_hardware_sensitivity()
            base_sigma: Default noise level (e.g., 0.05 = 5%)

        Returns:
            {layer_id: sigma} mapping
        """
        gain_results = sensitivity_results['gain']
        kurtosis_results = sensitivity_results['kurtosis']
        instability_results = sensitivity_results['instability']

        # Compute reference values
        gains = [gain_results.get(i, 1.0) for i in range(self.num_layers)]
        kurts = [kurtosis_results.get(i, 3000) for i in range(self.num_layers)]

        gain_median = np.median(gains)
        kurt_median = np.median(kurts)

        layer_sigmas = {}

        for layer_id in range(self.num_layers):
            gain = gain_results.get(layer_id, 1.0)
            kurt = kurtosis_results.get(layer_id, 3000)
            instab = instability_results.get(layer_id, 0.0)

            # Start with base sigma
            sigma = base_sigma

            # Factor 1: Low gain = signal loss risk → increase sigma
            if gain < gain_median * 0.8:
                sigma *= 1.5

            # Factor 2: High kurtosis = saturation risk → increase sigma
            if kurt > kurt_median * 1.2:
                sigma *= 1.5

            # Factor 3: High instability = already noisy → decrease sigma
            if instab > 0.1:
                sigma *= 0.5

            # Cap sigma at reasonable bounds
            sigma = max(0.01, min(sigma, 0.15))

            layer_sigmas[layer_id] = sigma

        return layer_sigmas

    def generate_training_config(
        self,
        sensitivity_results: Dict,
        base_sigma: float = 0.05,
        schedule: str = 'epoch_aware',
    ) -> Dict:
        """
        Generate complete AQN training configuration.

        Returns:
            Configuration dict for verl training
        """
        layer_sigmas = self.generate_layer_sigmas(
            sensitivity_results, base_sigma
        )

        # Identify high-priority layers
        high_sigma_layers = [
            lid for lid, sigma in layer_sigmas.items()
            if sigma > base_sigma * 1.2
        ]

        return {
            'aqn': {
                'enabled': True,
                'layer_specific': True,
                'base_sigma': base_sigma,
                'layer_sigmas': layer_sigmas,
                'schedule': schedule,
                'high_priority_layers': high_sigma_layers,
            },
            'srdd_metadata': {
                'num_layers': self.num_layers,
                'analysis_date': '2026-01-07',
            }
        }

    def print_config_summary(self, config: Dict):
        """Print human-readable summary of AQN config."""
        layer_sigmas = config['aqn']['layer_sigmas']
        base_sigma = config['aqn']['base_sigma']

        print("\n" + "=" * 60)
        print("SRDD-Guided AQN Configuration")
        print("=" * 60)
        print(f"Base sigma: {base_sigma}")
        print(f"Schedule: {config['aqn']['schedule']}")
        print(f"\nLayer-specific adjustments:")
        print("-" * 40)

        for lid, sigma in sorted(layer_sigmas.items()):
            ratio = sigma / base_sigma
            if ratio > 1.2:
                marker = "↑ HIGH"
            elif ratio < 0.8:
                marker = "↓ LOW"
            else:
                marker = "  BASE"
            print(f"  Layer {lid:2d}: σ={sigma:.4f} ({ratio:.1f}x) {marker}")

        print("-" * 40)
        high_layers = config['aqn']['high_priority_layers']
        print(f"High-priority layers: {high_layers}")
        print("=" * 60)
```

### 3.2 集成到verl训练

```python
# 在 verl/trainer/ppo_trainer.py 中

class PPOTrainer:
    def __init__(self, config, ...):
        # 加载SRDD引导的AQN配置
        if config.aqn.layer_specific:
            self.layer_sigmas = config.aqn.layer_sigmas
        else:
            self.layer_sigmas = None

    def _inject_noise(self, hidden_states, layer_id):
        """Layer-specific noise injection."""
        if self.layer_sigmas:
            sigma = self.layer_sigmas.get(layer_id, self.base_sigma)
        else:
            sigma = self.base_sigma

        noise = torch.randn_like(hidden_states) * hidden_states.abs() * sigma
        return hidden_states + noise
```

---

## 4. 使用流程

### 4.1 诊断阶段（在目标NPU上运行）

```bash
# Step 1: 在NPU上运行SRDD诊断
python scripts/srdd_guided_aqn.py \
    --model_path /path/to/model \
    --output aqn_config.yaml \
    --base_sigma 0.05

# 输出示例:
# ============================================================
# SRDD-Guided AQN Configuration
# ============================================================
# Base sigma: 0.05
#
# Layer-specific adjustments:
# ----------------------------------------
#   Layer  0: σ=0.0500 (1.0x)   BASE
#   Layer  1: σ=0.0500 (1.0x)   BASE
#   ...
#   Layer 10: σ=0.0750 (1.5x) ↑ HIGH   ← 高峰度，易饱和
#   Layer 15: σ=0.1000 (2.0x) ↑ HIGH   ← 低增益+高峰度
#   ...
# ----------------------------------------
# High-priority layers: [10, 15, 22]
# ============================================================
```

### 4.2 训练阶段（在GPU上运行）

```bash
# Step 2: 使用SRDD引导的AQN配置进行训练
python train_ppo.py \
    --model_path /path/to/model \
    --aqn_config aqn_config.yaml \
    --aqn_schedule epoch_aware
```

---

## 5. 预期收益

### 5.1 与均匀AQN对比

| 指标 | 均匀AQN | SRDD引导AQN | 改进 |
|-----|--------|------------|------|
| 训练稳定性提升 | +2.42% | +3~4% (预估) | +50% |
| 噪声注入层数 | 28层全部 | ~10层重点 | -64% |
| 计算开销 | 100% | ~50% | -50% |
| NPU迁移效果 | 通用 | **针对性** | 更好 |

### 5.2 理论基础

```
均匀AQN:  所有层等权重 → 无差别训练 → 效率低
          [σ][σ][σ][σ][σ][σ][σ][σ][σ][σ]

SRDD引导: 敏感层高权重 → 针对性训练 → 效率高
          [σ][σ][σ][2σ][σ][σ][2σ][σ][σ][σ]
                    ↑         ↑
                  L10敏感    L15敏感
```

---

## 6. 实验计划

### 6.1 验证实验

| 实验 | 目的 | 预期工作量 |
|-----|------|-----------|
| E1: SRDD配置生成 | 验证配置生成流程 | 1天 |
| E2: 对比训练 | 均匀AQN vs SRDD引导AQN | 3天 |
| E3: NPU迁移测试 | 验证针对性改进效果 | 2天 |

### 6.2 成功标准

- [x] SRDD引导AQN训练准确率 ≥ 均匀AQN (已验证，p=0.0126)
- [ ] 高优先级层在NPU上的误差显著降低
- [ ] 训练计算开销降低30%+

### 6.3 实验结果 (v2.0, 2026-01-08)

#### 实验设置
- **模型**: Qwen2.5-1.5B-Instruct
- **故障注入**: Layer 10, Deadzone threshold=0.01
- **AQN gamma**: 0.01
- **测试样本**: 50个文本
- **重复次数**: 5次 (不同随机种子)
- **统计方法**: t检验

#### 实验配置

| 配置 | Deadzone | AQN层 | 目的 |
|-----|----------|------|------|
| Clean | 无 | 无 | 基线 |
| Deadzone Only | L10 | 无 | 展示退化 |
| Global AQN | L10 | 全部28层 | 当前方法 |
| Targeted AQN | L10 | 仅L10 | SRDD引导 |
| Healthy AQN | L10 | 除L10外全部 | 控制组 |

#### 结果汇总

| 配置 | Loss (mean±std) | vs Deadzone | p-value |
|-----|-----------------|-------------|---------|
| Clean | 3.1777 ± 0.0000 | - | - |
| Deadzone Only | 12.8541 ± 0.0000 | baseline | - |
| Global AQN | 12.8606 ± 0.0105 | +0.1% | - |
| **Targeted AQN** | **12.8374 ± 0.0054** | **-0.1%** | **0.0126** |
| Healthy AQN | 12.8577 ± 0.0148 | +0.0% | 0.7267 |

#### 关键发现

1. **Targeted AQN比Global AQN好0.18% (p=0.0126, 显著)**
2. **Healthy AQN ≈ Global AQN (p=0.73, 不显著)**
3. 所有方法都出现灾难性退化 (~304% vs clean)
4. 控制组表明：收益来自于**不对健康层添加噪声**

#### QA专家结论

> "最简洁的解释：Targeted AQN表现略好是因为它没有进一步降低健康层的性能，而**不是**因为它主动引导学习远离故障。"

#### 实验局限性

- 单一操作点 (layer 10, threshold 0.01)
- 无机制分析 (梯度流、激活分布)
- 所有方法都在灾难性失败区域
- 实际效果微小 (0.18%)

#### 后续建议

1. 测试较温和的故障 (threshold 0.1, 0.2, 0.5)
2. 添加梯度流分析
3. 测试多个故障层
4. 添加替代基线 (层剪枝等)

---

## 7. 局限性与风险

| 风险 | 影响 | 缓解措施 |
|-----|------|---------|
| SRDD诊断可能不准确 | 错误的层优先级 | 多次诊断取平均 |
| 过高σ导致训练不稳定 | 训练发散 | 设置σ上限(0.15) |
| 硬件差异导致配置失效 | NPU A与NPU B结果不同 | 每种硬件单独诊断 |

---

## 8. 结论

### 8.1 实验验证结论

初步实验表明SRDD引导的Targeted AQN**在统计上显著优于**Global AQN (p=0.0126)，但需注意：

1. **效果微小**: 仅0.18%的改进
2. **机制存疑**: 改进可能来自"不伤害健康层"而非"主动修复故障层"
3. **灾难性失败**: 当前实验条件下所有方法都无法有效应对严重故障

### 8.2 研究价值

尽管实验效果有限，SRDD引导的AQN仍是有潜力的研究方向：

1. **诊断驱动**: 用数据指导训练，而非盲目注入
2. **针对性强**: 关注真正敏感的层，减少无效计算
3. **硬件适配**: 可针对不同NPU/GPU生成不同配置

### 8.3 下一步

1. 在较温和的故障条件下重新验证
2. 添加机制分析以理解真正的改进来源
3. 探索更有效的故障缓解策略

---

## 参考文档

- SRDD技术报告: `docs/qerl/SRDD_TECH_REPORT_CN.md`
- AQN实验总结: `docs/qerl/AQN_EXPERIMENT_SUMMARY_CN.md`
- SRDD使用场景: `docs/qerl/SRDD_USE_CASE_CN.md`
