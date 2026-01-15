# Ascend 910C NPU Setup and Test Plan

**Created**: 2025-12-25
**Purpose**: Document NPU environment setup and test plan for QeRL MoE on Ascend 910C
**Status**: In Progress

---

## Table of Contents

1. [Host Information](#1-host-information)
2. [Docker Environment](#2-docker-environment)
3. [Proxy Configuration](#3-proxy-configuration)
4. [Test Plan](#4-test-plan)
5. [Operational Commands](#5-operational-commands)

---

## 1. Host Information

### 910C Server

| Field | Value |
|-------|-------|
| **Host** | `root@90.90.97.92` |
| **Architecture** | aarch64 |
| **OS** | openEuler 22.03 SP4 |
| **Kernel** | 5.10.0-294.0.0.197.oe2203sp4.aarch64 |
| **NPU** | Ascend 910C (16 cards) |

### SSH Access

```bash
# Add host key if needed
ssh-keyscan -H 90.90.97.92 >> ~/.ssh/known_hosts

# Connect
ssh root@90.90.97.92
```

---

## 2. Docker Environment

### Official VERL Image for 910C

| Field | Value |
|-------|-------|
| **Image** | `quay.io/ascend/verl:verl-8.3.rc1-a3-ubuntu22.04-py3.11-latest` |
| **Base** | Ubuntu 22.04 |
| **Python** | 3.11 |
| **CANN** | 8.3.RC1 |
| **Hardware** | Atlas 800T A3 (910C) |

### Container Creation (TODO after pull completes)

```bash
# Create container with NPU access
docker run -itd \
    --name verl-910c \
    --network host \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci8 \
    --device /dev/davinci9 \
    --device /dev/davinci10 \
    --device /dev/davinci11 \
    --device /dev/davinci12 \
    --device /dev/davinci13 \
    --device /dev/davinci14 \
    --device /dev/davinci15 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/sbin:/usr/local/sbin:ro \
    -v /data:/data \
    quay.io/ascend/verl:verl-8.3.rc1-a3-ubuntu22.04-py3.11-latest \
    /bin/bash

# Enter container
docker exec -it verl-910c bash
```

### Environment Variables (inside container)

```bash
# CANN environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# HCCL settings
export HCCL_BUFFSIZE="512"
export HCCL_OP_BASE_FFTS_MODE_ENABLE="TRUE"
export HCCL_WHITELIST_DISABLE="1"

# Ascend device visibility
export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"

# Prevent kernel panic on OOM
echo 0 | sudo tee /proc/sys/vm/panic_on_oom

# WandB offline (network restrictions)
export WANDB_MODE=offline
```

---

## 3. Proxy Configuration

### Docker Daemon Proxy

**Location**: `/etc/systemd/system/docker.service.d/http-proxy.conf`

```ini
[Service]
Environment="HTTP_PROXY=http://p_atlas:proxy%%40123@90.255.4.119:8890"
Environment="HTTPS_PROXY=http://p_atlas:proxy%%40123@90.255.4.119:8890"
Environment="NO_PROXY=localhost,127.0.0.1,.huawei.com"
```

**Note**: `%` must be escaped as `%%` in systemd files.

**Apply changes**:
```bash
systemctl daemon-reload
systemctl restart docker
systemctl show docker --property=Environment  # Verify
```

### Shell Proxy (for pip, git, etc.)

```bash
export http_proxy="http://p_atlas:proxy%40123@90.255.4.119:8890"
export https_proxy="http://p_atlas:proxy%40123@90.255.4.119:8890"
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 4. Test Plan

### Phase 1: Environment Validation

| Step | Test | Expected Result |
|------|------|-----------------|
| 1.1 | `npu-smi info` | Show 16 NPU cards |
| 1.2 | `python -c "import torch_npu; print(torch_npu.npu.device_count())"` | Output: 16 |
| 1.3 | `python -c "import vllm; print(vllm.__version__)"` | vllm-ascend version |
| 1.4 | HCCL all-reduce test | Communication works |

### Phase 2: Model Loading

| Step | Test | Expected Result |
|------|------|-----------------|
| 2.1 | Load Qwen3-30B-A3B-Base-NVFP4 | Model loads on NPU |
| 2.2 | vLLM inference (single prompt) | Generation works |
| 2.3 | Check memory usage | Within 64GB HBM per card |

### Phase 3: VERL Integration

| Step | Test | Expected Result |
|------|------|-----------------|
| 3.1 | Ray cluster init | Ray starts on NPU |
| 3.2 | FSDP worker spawn | Workers initialize |
| 3.3 | vLLM rollout | Inference works |
| 3.4 | Megatron training step | Forward/backward works |

### Phase 4: QeRL Features

| Step | Test | Expected Result |
|------|------|-----------------|
| 4.1 | R3 routing capture | Routing captured during vLLM inference |
| 4.2 | R3 routing replay | Routing replayed in Megatron |
| 4.3 | AQN noise injection | Noise applied to RMSNorm |
| 4.4 | Full training step | R3+AQN works together |

### Phase 5: Performance & Stability

| Step | Test | Expected Result |
|------|------|-----------------|
| 5.1 | Multi-step training | No OOM for 10+ steps |
| 5.2 | Memory profiling | Identify peak usage |
| 5.3 | Throughput measurement | tokens/sec baseline |

---

## 5. Operational Commands

### tmux on HOST (CRITICAL)

**Pattern**: tmux runs on HOST, docker inside tmux.

```bash
# Create session
ssh root@90.90.97.92 "tmux new-session -d -s SESSION_NAME 'COMMAND'"

# List sessions
ssh root@90.90.97.92 "tmux ls"

# Check output (non-interactive)
ssh root@90.90.97.92 "tmux capture-pane -t SESSION_NAME -p | tail -20"

# Attach (interactive)
ssh -t root@90.90.97.92 "tmux attach -t SESSION_NAME"

# Kill session
ssh root@90.90.97.92 "tmux kill-session -t SESSION_NAME"
```

### Docker Operations

```bash
# List containers
ssh root@90.90.97.92 "docker ps -a"

# Enter container
ssh root@90.90.97.92 "docker exec -it CONTAINER_NAME bash"

# Check image
ssh root@90.90.97.92 "docker images | grep verl"
```

### NPU Monitoring

```bash
# NPU status
ssh root@90.90.97.92 "npu-smi info"

# Memory usage
ssh root@90.90.97.92 "npu-smi info -t memory"

# Process list
ssh root@90.90.97.92 "npu-smi info -t usages"
```

---

## 6. Files to Transfer

After quantization completes on A100, transfer to 910C:

| Source (A100) | Destination (910C) |
|---------------|-------------------|
| `/data/z00637938/Qwen3-30B-A3B-Base-NVFP4` | `/data/models/Qwen3-30B-A3B-Base-NVFP4` |

Transfer command (run from A100):
```bash
rsync -avP /data/z00637938/Qwen3-30B-A3B-Base-NVFP4 root@90.90.97.92:/data/models/
```

---

## 7. Known Issues from team/verl-npu

From QA analysis of team's NPU work:

| Issue | Status | Notes |
|-------|--------|-------|
| OOM during training | Unresolved | Team stuck on this |
| Dynamic backend detection | Works | Keep our dynamic code |
| empty_cache on NPU | May need explicit check | Monitor during testing |

---

## 8. Next Steps

1. [ ] Wait for docker pull to complete
2. [ ] Create container with NPU device access
3. [ ] Validate environment (Phase 1)
4. [ ] Transfer quantized model from A100
5. [ ] Run test plan phases 2-5
6. [ ] Document results and issues

