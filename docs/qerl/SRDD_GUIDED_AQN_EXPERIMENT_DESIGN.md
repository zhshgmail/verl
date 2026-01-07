# SRDD引导的AQN训练实验设计

**版本**: 1.0
**日期**: 2026-01-07
**状态**: 设计阶段

---

## 1. 实验目标

验证 **SRDD定位 + 针对性AQN** 是否优于 **全局AQN** 在特定层存在硬件故障时的表现。

### 核心假设

```
假设: 当模型的某一层存在MXFP4量化导致的死区(deadzone)问题时，
      SRDD可以精确定位该层，然后仅对该层注入AQN噪声，
      训练效果将优于对所有层均匀注入AQN噪声。
```

### 实验对比

| 模型 | 故障注入 | AQN策略 | 预期结果 |
|-----|---------|--------|---------|
| Model A | Layer 15 deadzone | SRDD定位 → Layer 15 AQN | 更高OOD分数 |
| Model B | Layer 15 deadzone | 全局AQN (所有层) | 基线OOD分数 |

---

## 2. MXFP4死区模拟机制

### 2.1 MXFP4死区特性

MXFP4 (4-bit MicroScaling Floating Point) 的死区特性：

```
MXFP4死区:
  - 量化范围有限，小于阈值的值被量化为0
  - 典型死区阈值: |x| < 0.01 * max(|x|)
  - 这导致信号丢失，特别是在激活值分布接近0的层

与AQN的交互:
  - 信号: s (可能在死区内)
  - AQN噪声: n ~ N(0, σ²|s|²)
  - 输出:
    if |s + n| > threshold: s + n (信号存活)
    else: 0 (被死区吞噬)
```

### 2.2 死区模拟实现

```python
class MXFP4DeadzoneFaultSimulator:
    """
    模拟MXFP4量化导致的死区故障。

    关键特性:
    1. 小于阈值的激活值被置零 (死区)
    2. 当AQN噪声足够强时，可以"打破"死区
    3. 同时影响前向和反向传播

    Args:
        model: 目标模型
        fault_layer: 故障层索引
        deadzone_threshold: 死区阈值 (相对于层输出最大值的比例)
        sparsity: 故障稀疏度 (1.0 = 整层，0.1 = 10%权重)
    """

    def __init__(
        self,
        model,
        fault_layer: int,
        deadzone_threshold: float = 0.01,  # 1% of max
        sparsity: float = 1.0,
    ):
        self.fault_layer = fault_layer
        self.deadzone_threshold = deadzone_threshold
        self.sparsity = sparsity

        # 获取目标层
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            self.target_module = model.model.layers[fault_layer]
        else:
            raise ValueError(f"Cannot find layer {fault_layer}")

        # 生成固定的稀疏掩码 (哪些位置有故障)
        self.sparse_mask = None  # 在第一次forward时初始化

        self.hook_handle = None

    def _deadzone_hook(self, module, input, output):
        """应用MXFP4死区效果到层输出."""
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        # 计算动态阈值 (基于当前batch的最大值)
        max_val = hidden_states.abs().max()
        threshold = self.deadzone_threshold * max_val

        # 创建死区掩码
        deadzone_mask = hidden_states.abs() < threshold

        # 应用稀疏性 (仅部分位置有故障)
        if self.sparsity < 1.0:
            if self.sparse_mask is None or self.sparse_mask.shape != hidden_states.shape:
                self.sparse_mask = torch.rand_like(hidden_states) < self.sparsity
            deadzone_mask = deadzone_mask & self.sparse_mask

        # 应用死区: 将死区内的值置零
        hidden_states = hidden_states.masked_fill(deadzone_mask, 0.0)

        if rest is not None:
            return (hidden_states,) + rest
        return hidden_states

    def enable(self):
        """启用死区故障注入."""
        if self.hook_handle is None:
            self.hook_handle = self.target_module.register_forward_hook(
                self._deadzone_hook
            )
            print(f"[MXFP4 DEADZONE] Layer {self.fault_layer}: "
                  f"threshold={self.deadzone_threshold}, sparsity={self.sparsity}")

    def disable(self):
        """禁用死区故障注入."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
```

### 2.3 关键设计决策

| 决策点 | 选择 | 理由 |
|-------|------|------|
| 阈值类型 | 相对阈值 (% of max) | 适应不同层的激活范围 |
| 稀疏性 | 可配置 (0.1~1.0) | 模拟真实硬件部分故障 |
| 影响范围 | 前向输出 | MXFP4量化主要影响激活值 |
| 梯度影响 | 通过前向传播间接影响 | 死区零值→梯度消失 |

---

## 3. 注入点分析

verl的架构需要在多个位置注入死区故障：

### 3.1 代码路径分析

```
┌─────────────────────────────────────────────────────────────┐
│  verl RL Training Pipeline                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  vLLM        │    │  Actor       │    │  Critic      │  │
│  │  Rollout     │    │  Training    │    │  Training    │  │
│  │              │    │  Forward     │    │              │  │
│  │  (独立进程)   │    │  Backward    │    │              │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│        ↑                    ↑                    ↑          │
│        │                    │                    │          │
│  需要注入死区          需要注入死区          无需注入        │
│  (vLLM hooks)        (noisy_ops)           (Critic独立)     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 各注入点实现方案

#### 3.2.1 SRDD诊断阶段

```
位置: scripts/srdd_error_finder.py
方法: 已有的 HardwareFaultSimulator 类
状态: ✓ 已完成 (支持 dead_zone fault type)

注入方式:
  - 使用 register_forward_hook 在层输出上注入
  - SRDD扫描会检测到增益异常 (低增益 = 信号丢失)
```

#### 3.2.2 vLLM Rollout阶段

```
位置: verl/workers/rollout/vllm_rollout/vllm_rollout.py
挑战: vLLM在独立进程中运行，与训练模型分离

方案A: vLLM模型hook注入 (推荐)
  - 在 vLLM 初始化后，为目标层注册 forward hook
  - 需要在 vllm_rollout.py 的 init_model() 后添加 hook
  - 优点: 与训练保持一致的行为

方案B: 权重修改 (不推荐)
  - 修改模型权重使输出趋近于零
  - 缺点: 死区是动态的，依赖于输入

代码修改位置:
  verl/workers/rollout/vllm_rollout/vllm_rollout.py
  - 在 self.inference_engine 初始化后
  - 获取内部模型: self.inference_engine.model_runner.model
  - 注册 forward hook 到目标层
```

#### 3.2.3 Actor Training阶段

```
位置: verl/utils/noisy_ops.py
挑战: 需要同时处理前向和反向传播

方案: 扩展 noisy_ops 模块支持死区注入

新增类: DeadzoneMatMul(torch.autograd.Function)
  - forward: 计算 result = matmul(a, b)，然后应用死区
  - backward: 梯度在死区位置为0 (梯度消失)

API设计:
  enable_deadzone_injection(
      layer_id=15,           # 目标层
      threshold=0.01,        # 死区阈值
      sparsity=1.0,          # 稀疏度
  )

与AQN的交互:
  - 死区在 matmul 输出后应用
  - AQN噪声在死区前/后注入 (可配置)
  - 当 |output + noise| > threshold 时，信号存活
```

### 3.3 统一死区注入接口

```python
# verl/utils/deadzone_injection.py (新文件)

class DeadzoneInjector:
    """
    统一的死区注入接口，用于 SRDD、vLLM、Training。

    使用方式:
        injector = DeadzoneInjector(
            fault_layer=15,
            threshold=0.01,
            sparsity=1.0,
        )

        # 注入到 SRDD 诊断
        injector.inject_srdd(model)

        # 注入到 vLLM rollout
        injector.inject_vllm(vllm_engine)

        # 注入到 training (via noisy_ops)
        injector.inject_training()
    """

    def __init__(
        self,
        fault_layer: int,
        threshold: float = 0.01,
        sparsity: float = 1.0,
    ):
        self.fault_layer = fault_layer
        self.threshold = threshold
        self.sparsity = sparsity

    def inject_srdd(self, model):
        """为 SRDD 诊断注入死区."""
        from scripts.srdd_error_finder import HardwareFaultSimulator
        simulator = HardwareFaultSimulator(
            model=model,
            fault_layer=self.fault_layer,
            fault_type='dead_zone',
            fault_magnitude=self.threshold,
            sparsity=self.sparsity,
        )
        simulator.enable()
        return simulator

    def inject_vllm(self, vllm_engine):
        """为 vLLM rollout 注入死区."""
        # 获取 vLLM 内部模型
        model = vllm_engine.model_runner.model

        # 找到目标层
        target_layer = model.model.layers[self.fault_layer]

        # 注册死区 hook
        def deadzone_hook(module, input, output):
            # 实现与 SRDD 相同的死区逻辑
            ...

        target_layer.register_forward_hook(deadzone_hook)

    def inject_training(self):
        """为 training forward/backward 注入死区."""
        from verl.utils.noisy_ops import (
            set_selective_layers,
            enable_noisy_ops,
        )

        # 仅在目标层启用
        set_selective_layers([self.fault_layer])

        # 启用带死区的 noisy ops
        enable_noisy_ops(
            error_scale=0,  # 无随机噪声
            error_type='deadzone',  # 新增类型
            deadzone_threshold=self.threshold,
        )
```

---

## 4. 实验设计

### 4.1 实验流程

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: SRDD 诊断                                         │
├─────────────────────────────────────────────────────────────┤
│  1. 加载 Qwen2.5-1.5B 模型                                  │
│  2. 注入 Layer 15 死区故障 (threshold=0.01)                 │
│  3. 运行 SRDD 诊断                                          │
│  4. 验证 SRDD 能否正确定位 Layer 15                          │
│  5. 输出: detected_layer = ?                                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 2: Model A 训练 (SRDD引导AQN)                        │
├─────────────────────────────────────────────────────────────┤
│  1. 在 vLLM + Training 中注入 Layer 15 死区                 │
│  2. 仅在 Layer 15 注入 AQN 噪声                              │
│     - gamma = 0.05 (5% 相对噪声)                            │
│     - schedule: epoch-aware decay                           │
│  3. 训练 2 epochs                                           │
│  4. 评估 OOD 准确率                                          │
│  5. 输出: OOD_A = ?                                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 3: Model B 训练 (全局AQN)                            │
├─────────────────────────────────────────────────────────────┤
│  1. 在 vLLM + Training 中注入 Layer 15 死区                 │
│  2. 在所有层注入 AQN 噪声                                    │
│     - gamma = 0.05 (5% 相对噪声)                            │
│     - schedule: epoch-aware decay                           │
│  3. 训练 2 epochs                                           │
│  4. 评估 OOD 准确率                                          │
│  5. 输出: OOD_B = ?                                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 4: 对比分析                                          │
├─────────────────────────────────────────────────────────────┤
│  对比: OOD_A vs OOD_B                                       │
│  预期: OOD_A > OOD_B (SRDD引导更有效)                        │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 控制变量

| 变量 | Model A (SRDD引导) | Model B (全局AQN) | 说明 |
|-----|-------------------|------------------|------|
| 基础模型 | Qwen2.5-1.5B | Qwen2.5-1.5B | 相同 |
| 死区故障 | Layer 15, θ=0.01 | Layer 15, θ=0.01 | 相同 |
| AQN σ | 0.05 (5%) | 0.05 (5%) | 相同 |
| AQN层 | **仅 Layer 15** | **所有28层** | **关键差异** |
| 训练轮数 | 2 epochs | 2 epochs | 相同 |
| 数据集 | GSM8K | GSM8K | 相同 |

### 4.3 评估指标

| 指标 | 定义 | 期望 |
|-----|------|------|
| OOD准确率 | 在带死区故障的模型上评估 | A > B |
| 训练损失曲线 | loss vs step | A更平滑 |
| 死区存活率 | 通过死区的信号比例 | A ≈ B |
| 计算开销 | AQN注入时间 | A < B (仅1层) |

---

## 5. 实现计划

### 5.1 代码修改清单

| 文件 | 修改内容 | 优先级 |
|-----|---------|-------|
| `scripts/srdd_error_finder.py` | 添加 `MXFP4DeadzoneFaultSimulator` 类 | P0 |
| `verl/utils/deadzone_injection.py` | 新建统一死区注入接口 | P0 |
| `verl/utils/noisy_ops.py` | 扩展支持死区注入 + AQN交互 | P1 |
| `verl/workers/rollout/vllm_rollout/vllm_rollout.py` | 添加死区hook注入点 | P1 |
| `scripts/run_srdd_aqn_experiment.py` | 实验脚本 | P2 |

### 5.2 分阶段实施

#### 阶段1: PoC验证 (1-2天)

目标: 验证死区注入机制在单独inference上工作

```bash
# 1. 在 SRDD 中测试死区检测
python scripts/srdd_error_finder.py \
    --model_path /data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/... \
    --ground_truth_layer 15 \
    --fault_type dead_zone \
    --fault_magnitude 0.01

# 验证输出: Layer 15 被正确识别
```

#### 阶段2: vLLM注入 (1-2天)

目标: 在vLLM rollout中实现死区注入

```python
# 修改 vllm_rollout.py
class vLLMAsyncRollout(BaseRollout):
    def __init__(self, ...):
        ...
        # 死区配置
        self.deadzone_config = {
            'enabled': getattr(config, 'deadzone_enabled', False),
            'layer': getattr(config, 'deadzone_layer', None),
            'threshold': getattr(config, 'deadzone_threshold', 0.01),
        }

    def _setup_deadzone_hooks(self):
        """在vLLM模型上设置死区hooks."""
        if not self.deadzone_config['enabled']:
            return

        # 获取vLLM内部模型并注册hook
        model = self.inference_engine.model_runner.model
        ...
```

#### 阶段3: Training注入 (2-3天)

目标: 在noisy_ops中实现死区+AQN联合注入

```python
# 扩展 noisy_ops.py

class DeadzoneMatMul(torch.autograd.Function):
    """带死区效果的matmul."""

    @staticmethod
    def forward(ctx, a, b):
        result = _ORIGINAL_MATMUL(a, b)

        # 应用死区
        if _DEADZONE_ENABLED:
            threshold = _DEADZONE_THRESHOLD * result.abs().max()
            mask = result.abs() < threshold
            result = result.masked_fill(mask, 0.0)
            ctx.deadzone_mask = mask

        # 应用AQN噪声 (可以帮助信号突破死区)
        if _NOISY_OPS_ENABLED and _NOISY_OPS_FORWARD_ENABLED:
            noise = _compute_error(result)
            result = result + noise

            # 重新评估死区 (噪声可能让信号存活)
            if _DEADZONE_ENABLED:
                mask = result.abs() < threshold
                result = result.masked_fill(mask, 0.0)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        # 死区位置梯度为0
        if hasattr(ctx, 'deadzone_mask'):
            grad_output = grad_output.masked_fill(ctx.deadzone_mask, 0.0)

        # 正常计算梯度
        ...
```

#### 阶段4: 完整实验 (2-3天)

目标: 运行完整的对比实验

```bash
# Model A: SRDD引导AQN
python train_ppo.py \
    --model_path /data/.../Qwen2.5-1.5B-Instruct \
    --deadzone_enabled true \
    --deadzone_layer 15 \
    --deadzone_threshold 0.01 \
    --aqn_enabled true \
    --aqn_target_layers "15" \
    --aqn_sigma 0.05

# Model B: 全局AQN
python train_ppo.py \
    --model_path /data/.../Qwen2.5-1.5B-Instruct \
    --deadzone_enabled true \
    --deadzone_layer 15 \
    --deadzone_threshold 0.01 \
    --aqn_enabled true \
    --aqn_target_layers "all" \
    --aqn_sigma 0.05
```

---

## 6. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|-----|------|---------|
| vLLM hook注入困难 | 无法在rollout中注入死区 | 方案B: 使用HF native rollout |
| 死区阈值过大导致训练崩溃 | 信号完全丢失 | 从小阈值(0.001)开始 |
| AQN无法打破死区 | 实验假设不成立 | 增加AQN强度或调整死区阈值 |
| SRDD误诊断 | 错误定位故障层 | 多次诊断取平均 |

---

## 7. 预期成果

### 7.1 技术验证

- [ ] SRDD能准确检测MXFP4死区故障
- [ ] 死区注入在vLLM和Training中一致工作
- [ ] AQN能帮助信号突破死区

### 7.2 实验结论

- [ ] SRDD引导AQN的OOD准确率 > 全局AQN的OOD准确率
- [ ] 量化改进幅度 (预期: +1~3%)
- [ ] 计算效率提升 (预期: -50%+ AQN开销)

---

## 附录

### A. 相关文档

- SRDD技术报告: `docs/qerl/SRDD_TECH_REPORT_CN.md`
- AQN实验总结: `docs/qerl/AQN_EXPERIMENT_SUMMARY_CN.md`
- SRDD引导AQN提案: `docs/qerl/SRDD_GUIDED_AQN_PROPOSAL_CN.md`

### B. A100测试环境

```bash
# SSH连接
ssh root@90.90.102.18
docker exec -it verl-r3-test bash
cd /home/z00637938/workspace/verl

# 模型路径
MODEL=/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/...
TEST_DATA=/data/z00637938/gsm8k/test.parquet
```

### C. 配置参数参考

```yaml
# verl配置示例
deadzone:
  enabled: true
  layer: 15
  threshold: 0.01
  sparsity: 1.0

aqn:
  enabled: true
  target_layers: [15]  # SRDD引导
  # target_layers: "all"  # 全局
  sigma: 0.05
  schedule: "epoch_aware"
```
