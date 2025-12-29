# QeRL R3+AQN Feature Documentation

**Branch**: `feature/qerl-noise-injection-moe`
**Last Updated**: 2025-12-24
**Status**: R3+AQN Working, NVFP4 Quantization In Progress

---

## Table of Contents

1. [Overview](#overview)
2. [Development Environment](#development-environment)
3. [Development Workflow](#development-workflow)
4. [R3 (Rollout Router Replay)](#r3-rollout-router-replay)
5. [AQN (Adaptive Quantization Noise)](#aqn-adaptive-quantization-noise)
6. [Training Configuration](#training-configuration)
7. [NVFP4 Quantization](#nvfp4-quantization)
8. [LoRA Support](#lora-support)
9. [Implementation Details](#implementation-details)
10. [Troubleshooting](#troubleshooting)
11. [QA Review Summary](#qa-review-summary)

---

## Overview

This feature branch implements QeRL-style training for MoE (Mixture of Experts) models with:

- **R3 (Rollout Router Replay)**: Captures routing decisions during vLLM inference and replays them during Megatron training
- **AQN (Adaptive Quantization Noise)**: Injects Gaussian noise into RMSNorm layers with decay schedule
- **LoRA Support**: Parameter-efficient fine-tuning (requires Megatron-Bridge on GPU)
- **NVFP4 Quantization**: Post-training quantization following QeRL methodology

### Tested Configuration

| Component | Value |
|-----------|-------|
| **Model** | Qwen1.5-MoE-A2.7B-Chat (60 experts) |
| **Hardware** | 8x A100 80GB GPUs |
| **Parallelism** | TP=2, EP=2 |
| **Performance** | ~150 tokens/sec, ~127s/step |

### Validation Results (2025-12-24)

```
step:1 - actor/entropy:6.03 - timing_s/step:87.47s
step:2 - actor/entropy:6.29 - timing_s/step:61.73s
step:3 - actor/entropy:5.97 - timing_s/step:59.27s

Routing capture working throughout:
- Retrieved routing logs from worker: 24 layers, 9911 tokens -> (9911, 24, 4)
- Retrieved routing logs from worker: 24 layers, 1119 tokens -> (1119, 24, 4)
```

---

## Development Environment

### Environment Topology

This project uses a **split development environment**:

| Environment | Location | Purpose | Has GPU/Megatron |
|-------------|----------|---------|------------------|
| **Local WSL** | Your laptop | Code editing, git operations, documentation | NO |
| **Remote GPU Host** | `root@90.90.102.18` | Training, testing, quantization | YES (8x A100) |

**Critical Rule**: Training scripts (`test_*.sh`, `train_*.sh`) and any code importing `verl.trainer` or `megatron` **MUST** run on the remote GPU host.

### Remote Server Details

```
Server: root@90.90.102.18
Container: verl-r3-test
Workspace: /home/z00637938/workspace/verl
GPUs: 8x NVIDIA A100 80GB
```

### Quick Access Commands

```bash
# SSH to remote server
ssh root@90.90.102.18

# Enter the container
docker exec -it verl-r3-test bash

# Navigate to workspace
cd /home/z00637938/workspace/verl

# Check GPU status
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
```

---

## Development Workflow

### Git Workflow (Fork-Based)

This project uses a **fork-based development model**:

| Remote | URL | Access |
|--------|-----|--------|
| `origin` | https://github.com/volcengine/verl.git | **READ ONLY** |
| `personal` | https://github.com/zhshgmail/verl.git | **WRITE ACCESS** |

**Critical**: Default `git push` goes to `origin` and will **FAIL**. Always use:

```bash
git push personal <branch-name>
```

### Code Change → Test Workflow

Follow this **mandatory sequence** when making changes:

```
┌─────────────────────────────────────────────────────────────┐
│  LOCAL WSL                                                   │
│  1. Edit code                                                │
│  2. git add <files>                                          │
│  3. git commit -m "message"                                  │
│  4. git push personal feature/qerl-noise-injection-moe       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  REMOTE GPU HOST                                             │
│  5. ssh root@90.90.102.18                                    │
│  6. docker exec -it verl-r3-test bash                        │
│  7. cd /home/z00637938/workspace/verl                        │
│  8. git pull   ← CRITICAL: Sync before testing!              │
│  9. bash test_r3_mode_minimal.sh                             │
└─────────────────────────────────────────────────────────────┘
```

### Pre-Push Checklist

Before pushing code:
- [ ] All changes committed? (`git status`)
- [ ] Using correct remote? (`git push personal <branch>`)
- [ ] Branch name correct?

### Pre-Test Checklist (Remote)

Before running tests on remote:
- [ ] SSH'd into correct server (`root@90.90.102.18`)?
- [ ] Inside correct container (`verl-r3-test`)?
- [ ] In correct directory (`/home/z00637938/workspace/verl`)?
- [ ] Pulled latest changes (`git pull`)?
- [ ] Verified commit matches local (`git log -1`)?

### Process Monitoring for Long-Running Tests

For overnight training or quantization jobs:

```bash
# Record PID after starting test
echo "Test PID: $(pgrep -f 'python.*verl')"

# Check if process is still running
ps -p <PID>

# Check for OOM killer (if process disappeared)
dmesg | grep -i "killed process"

# Monitor GPU memory usage
watch -n 5 nvidia-smi

# Check tmux sessions
tmux list-sessions
```

### Common Workflow Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Running `git push` (without remote) | Permission denied | Use `git push personal <branch>` |
| Testing without `git pull` | Old code running | Run `git pull` on remote first |
| Running test on local WSL | `ModuleNotFoundError: megatron` | SSH to remote and run there |
| Forgetting container | Command not found errors | `docker exec -it verl-r3-test bash` |

---

## R3 (Rollout Router Replay)

### What is R3?

R3 ensures consistent expert routing between vLLM inference (rollout) and Megatron training by:

1. **Capturing** routing decisions (expert assignments) during vLLM inference
2. **Converting** format from vLLM to Megatron
3. **Replaying** routing indices during Megatron training

**Key Insight**: R3 replays routing INDICES but PROBABILITIES are computed from current router weights, allowing gradients to flow to the router.

### Configuration

```yaml
actor_rollout_ref:
  actor:
    router_replay:
      mode: R3  # Options: disabled, R1, R2, R3
  rollout:
    enable_rollout_routing_replay: true
```

### Implementation Files

| File | Description |
|------|-------------|
| `verl/utils/routing_playback.py` | Data structures for routing logs |
| `verl/utils/routing_playback_converter.py` | Format converter (vLLM -> Megatron) |
| `verl/workers/rollout/vllm_routing_capture.py` | vLLM capture hook |
| `verl/utils/megatron/router_replay_patch.py` | Megatron MoE patch for replay |
| `verl/workers/actor/megatron_actor.py` | Integration point |
| `verl/workers/config.py` | RouterReplayConfig dataclass |

### R3 Mode Data Flow

```
1. vLLM Rollout Phase:
   - Model generates responses
   - vLLM MoE layers capture routing decisions
   - Creates BatchRoutingLogs with expert indices

2. Format Conversion:
   - BatchRoutingLogs -> RoutingPlaybackBatch
   - Adapter for Megatron format

3. Megatron Training:
   - Receives RoutingPlaybackBatch
   - Patches MoE layers with RouterReplay
   - Uses captured routing instead of recomputing
```

---

## AQN (Adaptive Quantization Noise)

### What is AQN?

AQN simulates quantization effects during training to improve model robustness:

- Injects Gaussian noise into RMSNorm layers
- Noise level follows decay schedule: `sigma_start -> sigma_end` over `num_stages`
- Target modules: `post_attention_layernorm` (excludes router)

### Configuration

```yaml
noise_injection:
  enabled: true
  sigma_start: 0.01
  sigma_end: 0.001
  num_stages: 10
  target_modules: ["post_attention_layernorm"]
  exclude_patterns: ["input_layernorm", "router"]
```

### Benefits

1. **Increased Policy Entropy**: Quantization noise increases exploration in RL
2. **Better Strategy Discovery**: Enables RL to find better solutions
3. **Faster Reward Growth**: Models trained with QeRL show accelerated reward improvement
4. **Full-Parameter RL Parity**: Achieves performance matching full-precision RL

---

## Training Configuration

### Full Training Script (`train_qerl_gsm8k_full.sh`)

```bash
# Parallelism (8x A100)
TP=2                    # Tensor parallelism
EP=2                    # Expert parallelism
VLLM_INFER_TP=2         # vLLM inference TP

# Training
TRAIN_BATCH_SIZE=64
PPO_MINI_BATCH_SIZE=16
LEARNING_RATE=1e-5
MAX_RESPONSE_LENGTH=1024
TOTAL_TRAINING_STEPS=2000

# R3 Configuration
R3_ENABLED=True
actor_rollout_ref.actor.router_replay.mode=R3

# AQN Configuration
NOISE_ENABLED=True
SIGMA_START=0.01
SIGMA_END=0.001

# LoRA Configuration
ENABLE_LORA=True
LORA_RANK=32
LORA_ALPHA=64

# Memory Optimization
PARAM_OFFLOAD=True
OPTIMIZER_OFFLOAD=True
GPU_MEMORY_UTILIZATION=0.4
```

### Required Environment Variables

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VERL_DISABLE_DYNAMO=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
```

### Critical Configuration for MoE + Co-located Mode

- `offload=True` (param_offload, optimizer_offload, grad_offload)
- `gpu_memory_utilization=0.3-0.4` (low to share GPU with Megatron)
- `enforce_eager=True` (required for vLLM 0.11.0)
- `VERL_DISABLE_DYNAMO=1` (disable torch.compile)

---

## NVFP4 Quantization

### Overview

NVFP4 quantization follows QeRL methodology for post-training quantization of MoE models.

**Format**: 4-bit floating-point weights, BF16 activations

### Docker Container Selection (CRITICAL)

On A100 server (root@90.90.102.18), there are **multiple Docker containers**:

| Container | Purpose | Has llmcompressor |
|-----------|---------|-------------------|
| **`verl-fp8-container`** | **Quantization ONLY** | YES |
| `verl-r3-test` | Latest verl official image, RL training | NO |

**ALWAYS use `verl-fp8-container` for quantization work**:
```bash
ssh root@90.90.102.18
docker exec -it verl-fp8-container bash
```

### Prerequisites

Inside `verl-fp8-container`:
```bash
# llmcompressor and compressed_tensors should be pre-installed
pip show llmcompressor  # Should be 0.9.0+
pip show compressed-tensors  # Should be 0.13.0+

# If missing or broken, reinstall:
pip install llmcompressor>=0.9.0 compressed-tensors>=0.13.0
```

### Full Quantization Script

**Location**: Create at `/tmp/quantize_moe_nvfp4.py`

```python
#!/usr/bin/env python3
"""
NVFP4 Quantization Script for MoE Models
Usage: python quantize_moe_nvfp4.py --model_path /path/to/model --output_path /path/to/output
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to model (local or HF repo)")
    parser.add_argument("--output_path", required=True, help="Output path for quantized model")
    parser.add_argument("--num_samples", type=int, default=512, help="Calibration samples")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Max sequence length")
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Calibration dataset
    print("Loading calibration dataset...")
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ds.shuffle(seed=42).select(range(args.num_samples))

    def preprocess(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
    ds = ds.map(preprocess)

    # NVFP4 recipe - CRITICAL: exclude router/gate layers
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="NVFP4A16",  # 4-bit weights, 16-bit activations
        ignore=[
            "lm_head",                    # Output projection
            "re:.*mlp\\.gate$",           # MoE router (Qwen3, DeepSeek)
            "re:.*shared_expert_gate$",   # Shared expert gate (Qwen1.5-MoE)
        ]
    )

    print("Starting quantization...")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=args.num_samples,
    )

    print(f"Saving quantized model to: {args.output_path}")
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print("Done!")

if __name__ == "__main__":
    main()
```

### Running Quantization (MUST use tmux)

**CRITICAL**: tmux runs on HOST, docker runs INSIDE tmux session.

```bash
# 1. SSH to A100
ssh root@90.90.102.18

# 2. Create tmux session ON HOST
tmux new-session -s quant_work

# 3. INSIDE tmux, run docker and quantization
docker exec -it verl-fp8-container python /tmp/quantize_moe_nvfp4.py \
    --model_path /data/z00637938/hub/models--Qwen--Qwen3-30B-A3B-Base/snapshots/1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9 \
    --output_path /data/z00637938/Qwen3-30B-A3B-Base-NVFP4 \
    --num_samples 512

# 4. Detach from tmux: Ctrl+B, then D
# 5. Later, check status: tmux capture-pane -t quant_work -p | tail -20
```

### Layers Kept in BF16

| Layer Pattern | Reason |
|---------------|--------|
| `lm_head` | Output projection - sensitive to quantization |
| `re:.*mlp\\.gate$` | MoE router - routing decisions must be precise |
| `re:.*shared_expert_gate$` | Qwen1.5-MoE only - gate for shared expert |

### Model-Specific Ignore Patterns

| Model | Ignore Pattern |
|-------|----------------|
| **Qwen3-MoE** | `["lm_head", "re:.*mlp\\.gate$"]` |
| **Qwen1.5-MoE** | `["lm_head", "re:.*mlp\\.gate$", "re:.*shared_expert_gate$"]` |
| **Mixtral** | `["lm_head", "re:.*block_sparse_moe\\.gate"]` |
| **DeepSeek-V3** | `["lm_head", "re:.*mlp\\.gate$"]` |

### Completed Quantization Results

#### H100 - Qwen1.5-MoE-A2.7B-Chat (COMPLETED)
| Field | Value |
|-------|-------|
| **Model** | Qwen1.5-MoE-A2.7B-Chat (24 layers, 60 experts) |
| **Output** | `/data/z00637938/Qwen1.5-MoE-A2.7B-NVFP4` |
| **Size** | **8.4GB** (down from ~28GB BF16, ~3.3x compression) |
| **Status** | **COMPLETED** (2025-12-25) |

#### A100 - Qwen3-30B-A3B-Base (COMPLETED)
| Field | Value |
|-------|-------|
| **Model** | Qwen3-30B-A3B-Base (48 layers, 128 experts) |
| **Output** | `/data/z00637938/Qwen3-30B-A3B-Base-NVFP4` |
| **Size** | **17GB** (down from ~60GB BF16, ~3.5x compression) |
| **Status** | **COMPLETED** (2025-12-25) |

### Monitoring Commands

**CRITICAL**: tmux runs on HOST, docker runs INSIDE tmux session.

```bash
# List tmux sessions on host
ssh root@90.90.102.18 "tmux ls"

# Check output (tmux is on HOST, not in docker)
ssh root@90.90.102.18 "tmux capture-pane -t SESSION_NAME -p | tail -20"

# Attach to session interactively
ssh -t root@90.90.102.18 "tmux attach -t SESSION_NAME"

# Check GPU memory usage
ssh root@90.90.102.18 "nvidia-smi --query-gpu=memory.used --format=csv"
```

### Troubleshooting

#### Container Killed / Process Lost
If the docker container is killed during quantization:
1. Quantization has **NO checkpointing** - must restart from 0%
2. Output directory may have partial/corrupt files - delete and restart
3. Always use tmux inside container to survive container restarts

#### ImportError: InternalModule / is_fp4
```bash
# Upgrade compressed-tensors
pip install --upgrade compressed-tensors>=0.13.0
```

#### HFValidationError for local paths
If transformers validates paths as HuggingFace repo IDs:
```bash
# Create symlink with simple name
ln -s /long/path/to/model/snapshot/hash /tmp/ModelName
# Then use /tmp/ModelName as model_path
```

#### Wrong model loading (no MoE modules)
Ensure using correct snapshot path with full hash:
```bash
# Find correct snapshot
ls /data/z00637938/hub/models--Qwen--Qwen3-30B-A3B-Base/snapshots/
# Use the full path with hash
```

---

## LoRA Support

### Platform Summary

| Platform | LoRA Training | LoRA + RL | How |
|----------|---------------|-----------|-----|
| **A100** | YES | YES | verl + Megatron-Bridge |
| **H100** | YES | YES | verl + Megatron-Bridge |
| **Ascend 910C** | YES | Partial | MindSpeed-LLM |

### Enabling LoRA on GPU (A100/H100)

```bash
# 1. Install Megatron-Bridge
pip install megatron-bridge

# 2. Add to training script
actor_rollout_ref.actor.megatron.vanilla_mbridge=False
actor_rollout_ref.model.lora.rank=32
actor_rollout_ref.model.lora.alpha=64
actor_rollout_ref.model.lora.target_modules=["linear_qkv","linear_proj","linear_fc1","linear_fc2"]
```

### LoRA Merge Guide

After training with LoRA, merge adapters for deployment:

**Step 1: Convert Checkpoint to HuggingFace Format**
```bash
python -m verl.model_merger merge \
    --backend megatron \
    --local_dir /path/to/checkpoints/global_step_XXX/actor \
    --target_dir /path/to/output/merged_model \
    --trust-remote-code
```

**Step 2: Merge LoRA into Base Model**
```python
from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "/path/to/merged_model",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, "/path/to/merged_model/lora_adapter")
model = model.merge_and_unload()
model.save_pretrained("/path/to/final_merged_model")
```

---

## Implementation Details

### vLLM 0.11.0 Compatibility

**Solution Implemented**:

1. **Environment Variable**: `VERL_DISABLE_DYNAMO=1`
   - Disables `torch._dynamo.config.disable = True` before vLLM imports
   - Added in `vllm_async_server.py`

2. **Config Option**: `enforce_eager=True`
   - Disables CUDA graphs to ensure Python forward methods execute

### SharedFusedMoE Handling

Our monkey-patch captures routing **BEFORE** the fused kernel:

```python
# vllm_routing_capture.py forward_with_capture()

# Step 1: Gate/Router executes separately (NOT fused)
router_logits, _ = self.gate(hidden_states)

# Step 2: WE compute routing ourselves from router_logits
routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
routing_weights, expert_ids = torch.topk(routing_probs, k=self.experts.top_k, dim=-1)

# Step 3: Fused kernel runs (we don't depend on its output for routing)
final_hidden_states = self.experts(hidden_states, router_logits)
```

### Data Structures

```python
@dataclass
class BatchRoutingLogs:
    """vLLM format: per-layer routing decisions"""
    expert_indices: List[torch.Tensor]  # [num_layers]

@dataclass
class RoutingPlaybackBatch:
    """Megatron format: unified routing tensor"""
    routing_tensor: torch.Tensor  # [batch, seq, num_layers, topk]
```

---

## Troubleshooting

### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| Zero rewards, 100% clip ratio | Using BASE model instead of INSTRUCT | Switch to Chat/Instruct variant |
| `LoRA/PEFT only supported via Megatron-Bridge` | `megatron-bridge` not installed | Install package or disable LoRA |
| `lr_decay_steps > 0 assertion` | Missing `total_training_steps` | Add explicit `trainer.total_training_steps=N` |
| `routed_experts must be in data.batch.keys()` | R3 key mismatch during concat | Fixed in commits - ensure latest code |
| OOM during wake_up | GPU memory insufficient | Enable offloading |
| `is_fp4` import error | compressed_tensors version mismatch | Patch helpers.py (see below) |
| `static_minmax` observer error | NVFP4 scheme version mismatch | Remove observer field from scheme |

### Patching compressed_tensors for NVFP4

If you get `ImportError: cannot import name 'is_fp4'`:

```bash
# Add is_fp4 function to installed package
cat >> /path/to/site-packages/compressed_tensors/quantization/utils/helpers.py << 'EOF'

def is_fp4(quantization_args: QuantizationArgs):
    """Check if quantization args specify FP4 quantization."""
    return (
        quantization_args.num_bits == 4
        and quantization_args.type == QuantizationType.FLOAT
    )
EOF

# Add to __all__ and clear pycache
```

### Monitoring Commands

```bash
# Check training progress
grep -E "step:|Training Progress" /tmp/training.log | tail -20

# Check R3 routing
grep -E "routing replay|routed_experts" /tmp/training.log

# Check AQN noise
grep -E "Noise injection|sigma" /tmp/training.log

# Check for errors
grep -E "Error|Exception|Traceback" /tmp/training.log
```

---

## QA Review Summary

### QA Analysis (2025-12-23)

**Key Finding**: The routing capture patch IS applied correctly in worker actors, but there was initially no mechanism to retrieve the captured routing data from worker processes back to the main server process.

**Solution Implemented**: Added `get_captured_routing.remote()` Ray call to retrieve routing from worker actors.

### Architecture Analysis

```
vLLM Routing Capture Architecture:

┌─────────────────────────────────────────┐
│ Ray Actor (Worker Process)              │
│  ┌───────────────────────────────────┐  │
│  │ vLLM Engine                       │  │
│  │  - Model.forward() runs here      │  │
│  │  - Routing capture works          │  │
│  │  - _routing_layers_registry       │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
         ↕ Ray remote call
┌─────────────────────────────────────────┐
│ Main Process (vLLMHttpServer)           │
│  - get_captured_routing.remote()        │
│  - Receives routing via Ray             │
└─────────────────────────────────────────┘
```

### Bug Fixes in This Branch

1. **Config path error**: Noise injection at root level, not under trainer
2. **RolloutConfig fields**: Added noise injection fields to dataclass
3. **Dispatch method**: Added `update_noise_injection_step` dispatch
4. **TensorDict concat**: Handle key mismatch when some batches have `routed_experts`
5. **execute_all_async**: Use `ray.get()` for async returns
6. **SharedFusedMoE compatibility**: Handle tuple return from vLLM 0.11.0

---

## Training Failure Analysis: Zero Rewards Issue

### Problem

During initial training with Qwen1.5-MoE-A2.7B (BASE model):
- `response_length/clip_ratio: 1.0` - 100% of responses hit max length
- `critic/rewards/mean: 0.0` - All rewards were 0 throughout training

### Root Cause

**Qwen1.5-MoE-A2.7B is a BASE model**, not an INSTRUCT/CHAT model.

| Aspect | BASE Model | INSTRUCT Model |
|--------|------------|----------------|
| Training | Next-token prediction | Instruction following |
| Format instructions | Ignored | Understood |
| Task completion | Continues until max_length | Stops after answer |
| "#### answer" format | Never outputs | Reliably outputs |

### Solution

Use the CHAT/INSTRUCT variant:
```bash
HF_MODEL_PATH=/path/to/Qwen1.5-MoE-A2.7B-Chat
```

---

## References

### Documentation
- Example: `examples/router_replay/README.md`
- Config: `verl/workers/config.py:RouterReplayConfig`

### Test Files
- Script: `test_r3_mode_minimal.sh`
- Full training: `train_qerl_gsm8k_full.sh`

### Related Work
- Noise Injection (Phase 1-3): Complete
- Router Replay (Phase 4): Complete
- NVFP4 Quantization: In Progress
