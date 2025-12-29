# VERL NPU 训练指南 - QeRL/AQN 配置

本文档详细介绍如何在华为昇腾 NPU 上配置和运行 VERL 的 GRPO 训练，包括 AQN（自适应量化噪声）功能的配置。

## 目录

1. [环境准备](#环境准备)
2. [数据准备](#数据准备)
3. [配置说明](#配置说明)
4. [AQN 噪声注入配置](#aqn-噪声注入配置)
5. [启动训练](#启动训练)
6. [常见问题](#常见问题)

---

## 环境准备

### 1. 基础环境要求

- 华为昇腾 910B/910C NPU
- CANN 8.0+
- Python 3.11+
- PyTorch 2.1+ (with NPU support)
- torch_npu
- Ray 2.0+

### 2. 容器环境

推荐使用预构建的 VERL NPU 容器镜像。容器启动后需要确保以下环境变量已设置：

```bash
# NPU 相关环境变量
export HCCL_BUFFSIZE="512"
export HCCL_OP_BASE_FFTS_MODE_ENABLE="TRUE"
export HCCL_WHITELIST_DISABLE="1"

# vLLM 相关环境变量
export VLLM_USE_V1="1"
export VLLM_ASCEND_ENABLE_NZ="0"
export VERL_DISABLE_DYNAMO="1"

# 其他
export WANDB_MODE="offline"  # 离线模式，或设置为 "online" 并配置 WANDB_API_KEY
```

### 3. 检查 NPU 设备

```bash
# 查看 NPU 设备信息
npu-smi info -l

# 预期输出应显示可用的 NPU 设备
# Total Count: 8 (或您的设备数量)
```

---

## 数据准备

### 1. 数据格式

VERL 使用 Parquet 格式存储训练数据。数据文件应包含以下字段：

- `prompt`: 输入提示文本
- `answer`: 参考答案（可选，用于计算奖励）

### 2. 数据路径

将数据文件放置在容器可访问的路径，例如：

```bash
# 训练数据
/tmp/train_gsm8k_v4.parquet

# 验证数据
/tmp/test_gsm8k_v4.parquet
```

---

## 配置说明

### 1. 基础配置参数

以下是 GRPO 训练的关键配置参数：

```yaml
# 算法配置
algorithm:
  adv_estimator: grpo  # 使用 GRPO 算法

# 数据配置
data:
  train_files: /path/to/train.parquet
  val_files: /path/to/val.parquet
  train_batch_size: 64
  max_prompt_length: 512
  max_response_length: 2048
  filter_overlong_prompts: True

# Actor 模型配置
actor_rollout_ref:
  model:
    path: /path/to/model  # 模型路径

  actor:
    optim:
      lr: 5e-7  # 学习率
    ppo_mini_batch_size: 16
    ppo_micro_batch_size_per_gpu: 2  # 每 GPU 微批次大小
    use_kl_loss: True
    kl_loss_coef: 0.001

  rollout:
    name: vllm
    tensor_model_parallel_size: 1
    gpu_memory_utilization: 0.4
    enforce_eager: true
    n: 4  # 每个 prompt 生成的样本数
    log_prob_micro_batch_size_per_gpu: 2

  ref:
    log_prob_micro_batch_size_per_gpu: 2

# 训练器配置
trainer:
  n_gpus_per_node: 8  # 每节点 NPU 数量
  nnodes: 1
  total_epochs: 2
  test_freq: 20  # 验证频率
  val_before_train: True
  save_freq: -1  # 检查点保存频率，-1 表示不保存
  logger: console  # 日志输出方式
```

### 2. 微批次大小调整

对于 NPU 训练，可以根据显存情况调整微批次大小：

| 配置参数 | 保守设置 | 4x MBS 设置 |
|---------|---------|------------|
| `ppo_micro_batch_size_per_gpu` | 2 | 8 |
| `log_prob_micro_batch_size_per_gpu` | 2 | 8 |

---

## AQN 噪声注入配置

AQN (Adaptive Quantization Noise) 是一种在训练过程中向模型权重注入噪声的正则化技术，特别适用于量化模型的训练。

### 1. AQN 配置参数

```yaml
trainer:
  noise_injection:
    # 是否启用噪声注入
    enabled: true

    # 初始 sigma 值（噪声标准差）
    sigma_start: 0.05

    # 最终 sigma 值
    sigma_end: 0.0005

    # 衰减阶段数
    num_stages: 10

    # 目标模块（接收噪声注入的层）
    target_modules: ["post_attention_layernorm"]

    # 排除模式（不注入噪声的层）
    exclude_patterns: ["input_layernorm"]
```

### 2. Sigma 参数说明

`sigma` 是注入噪声的标准差，控制噪声的强度：

- **sigma_start**: 训练开始时的噪声强度（较大值）
- **sigma_end**: 训练结束时的噪声强度（较小值）
- **num_stages**: 噪声从 sigma_start 衰减到 sigma_end 的阶段数

推荐的 QeRL 设置：
- `sigma_start: 0.05` (5% 噪声)
- `sigma_end: 0.0005` (0.05% 噪声)
- `num_stages: 10`

### 3. 噪声注入原理

噪声按以下方式注入到目标模块的权重中：

```
weight_noisy = weight + gaussian_noise * sigma
```

其中 `sigma` 随训练进度从 `sigma_start` 线性衰减到 `sigma_end`。

---

## 启动训练

### 1. 直接启动（推荐）

让 VERL 自动管理 Ray 集群，确保正确检测 NPU 资源：

```bash
cd /verl

RAY_DEDUP_LOGS=0 python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=/tmp/train_gsm8k_v4.parquet \
  data.val_files=/tmp/test_gsm8k_v4.parquet \
  data.train_batch_size=64 \
  data.max_prompt_length=512 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=True \
  actor_rollout_ref.model.path=/path/to/model \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  algorithm.use_kl_in_reward=False \
  custom_reward_function.path=/verl/verl/utils/reward_score/gsm8k.py \
  custom_reward_function.name=compute_score \
  trainer.critic_warmup=0 \
  trainer.logger=console \
  trainer.project_name=my_project \
  trainer.experiment_name=grpo_npu_test \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=20 \
  trainer.total_epochs=2 \
  trainer.val_before_train=True \
  trainer.noise_injection.enabled=True \
  trainer.noise_injection.sigma_start=0.05 \
  trainer.noise_injection.sigma_end=0.0005 \
  trainer.noise_injection.num_stages=10 \
  2>&1 | tee /tmp/training.log
```

### 2. 关于 Ray 启动的重要说明

**不要手动启动 Ray！** 直接运行训练脚本，让 VERL 自动初始化 Ray 并正确检测 NPU 资源。

如果手动执行 `ray start --head --num-gpus=8`，Ray 会注册 GPU 资源而非 NPU 资源，导致训练任务无法调度。

VERL 会自动：
1. 检测可用的 NPU 设备
2. 初始化 Ray 集群并注册 NPU 资源
3. 配置分布式训练环境

### 3. 使用 YAML 配置文件

也可以使用 Hydra 配置文件：

```bash
python3 -m verl.trainer.main_ppo --config-name=your_config
```

配置文件示例：

```yaml
# your_config.yaml
hydra:
  searchpath:
    - file://verl/trainer/config

defaults:
  - ppo_megatron_trainer
  - _self_

algorithm:
  adv_estimator: grpo

data:
  train_files: /tmp/train.parquet
  val_files: /tmp/val.parquet
  # ... 其他数据配置

trainer:
  device: npu
  n_gpus_per_node: 8
  noise_injection:
    enabled: true
    sigma_start: 0.05
    sigma_end: 0.0005
    num_stages: 10
```

---

## 常见问题

### 1. Ray 无法找到 NPU 资源

**症状**: 训练任务卡在 `Pending Demands: {'NPU': 1.0}`

**解决方案**:
- 不要手动启动 Ray
- 直接运行 `python3 -m verl.trainer.main_ppo ...`
- 让 VERL 自动初始化 Ray 并检测 NPU

### 2. 僵尸进程积累

**症状**: 容器中出现大量 zombie/defunct 进程

**解决方案**:
```bash
# 在容器外重启容器
docker restart <container_name>

# 或在容器内清理
ray stop --force
pkill -9 python
pkill -9 ray
```

### 3. OOM（内存不足）错误

**症状**: 训练过程中出现 OutOfMemory 错误

**解决方案**:
- 降低 `ppo_micro_batch_size_per_gpu`
- 降低 `log_prob_micro_batch_size_per_gpu`
- 降低 `gpu_memory_utilization`
- 减少 `max_response_length`

### 4. 验证精度指标

训练过程中会定期输出验证指标：
- `val-core/openai/gsm8k/acc/mean@1`: GSM8K 平均准确率

---

## 参考资料

- [VERL 官方文档](https://verl.readthedocs.io/)
- [QeRL 论文](https://arxiv.org/abs/xxxx.xxxxx)
- [华为昇腾开发者文档](https://www.hiascend.com/)

---

*文档版本: 2025-12-29*
