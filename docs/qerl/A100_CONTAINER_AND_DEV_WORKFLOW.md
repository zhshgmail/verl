# A100 Container and Development Workflow

**Purpose**: Complete reference for A100 GPU environment, container access, and development workflow including git operations.

---

## ⚠️ CRITICAL RULES - READ FIRST ⚠️

### Rule 1: ALWAYS Restart Container Before Each Experiment

**MANDATORY**: You MUST restart the container before running ANY experiment to avoid zombie process accumulation.

**Why**: Ray/PyTorch distributed training creates many worker processes. If experiments fail or are interrupted, these become zombie processes (`<defunct>`) that accumulate and cause:
- Memory leaks
- Resource exhaustion
- Training hangs/crashes
- Unpredictable failures

**Evidence**: After E13h, 2280 zombie processes accumulated, causing the experiment to hang at step 27.

**Procedure**:
```bash
# On host (not in container)
ssh root@90.90.102.18

# Restart the container (kills all processes cleanly)
docker restart verl-r3-test

# Wait ~10 seconds for restart
sleep 10

# Re-enter container
docker exec -it verl-r3-test bash

# Verify no zombies
ps aux | grep defunct | wc -l  # Should be 0 or very small

# Navigate to workspace
cd /home/z00637938/workspace/verl

# Start Ray cluster (required after restart)
ray start --head --port=6379 --num-gpus=8 --disable-usage-stats

# Now run your experiment
bash scripts/your_experiment.sh
```

**Checklist Before Every Experiment**:
- [ ] Container restarted
- [ ] No zombie processes (check with `ps aux | grep defunct`)
- [ ] Ray cluster started
- [ ] GPU memory clear (check with `nvidia-smi` if available)

### Rule 2: Never Assume Experiment Completed Without Checking Final Step

**Problem**: Experiments can hang/crash silently without completing all training steps.

**What to Check**:
```bash
# Check if experiment completed (look for final validation or completion message)
grep -E "(Final validation|Training.*complete|step:2[89])" /path/to/training.log

# Check last logged step
grep "step:" /path/to/training.log | tail -1

# Check for zombie process of your experiment
ps aux | grep <PID> | grep defunct
```

**E13h Example**:
- Reported 56.41% at step 20 ✓
- Training hung at step 27 ✗
- Never reached step 28/29 or final validation ✗
- Process became zombie ✗

**Conclusion**: E13h did NOT complete successfully - need to re-run!

### Rule 3: Check Logs During Training

Don't wait until the end to check logs. Monitor periodically:
```bash
# Monitor live
tail -f /tmp/your_experiment/training.log

# Check progress every 10 minutes
grep "Training Progress:" /tmp/your_experiment/training.log | tail -1

# Check for errors
grep -i "error\|exception\|failed" /tmp/your_experiment/training.log | tail -20
```

---

## 1. Connection

```bash
# SSH to A100 server
ssh root@90.90.102.18

# Enter docker container
docker exec -it verl-r3-test bash

# Navigate to workspace
cd /home/z00637938/workspace/verl
```

---

## 2. Git Workflow

### 2.1 Remote Repositories

```bash
# Check configured remotes
git remote -v

# Expected output:
# origin   https://github.com/volcengine/verl.git (upstream - read-only)
# personal https://github.com/zhshgmail/verl.git (personal fork)
# team     https://github.com/EdisonAILab/verl.git (team fork)
```

### 2.2 Push Workflow

**IMPORTANT**: Always push to `personal` and `team` remotes, NOT `origin` (upstream is read-only).

```bash
# Push to both personal and team remotes
git push personal <branch-name>
git push team <branch-name>

# Example: Push feature branch
git push personal feature/npu-aqn-test
git push team feature/npu-aqn-test

# Push multiple remotes at once
git push personal <branch> && git push team <branch>
```

### 2.3 Common Git Operations

```bash
# Ensure you're on the feature branch
git checkout feature/npu-aqn-test

# Check status
git status

# Stage changes
git add <files>

# Commit with detailed message
git commit -m "feat: add E13h experiment

Detailed description here.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

# Push to both remotes (ALWAYS push to personal + team, NOT origin)
git push personal feature/npu-aqn-test
git push team feature/npu-aqn-test

# Pull latest changes from upstream (if needed)
git fetch origin
git merge origin/main

# View recent commits
git log --oneline -10
```

### 2.4 Branch Strategy

**Current branch**: `feature/npu-aqn-test`

**Note**: This project uses a **single long-lived feature branch** (`feature/npu-aqn-test`) for all NPU/AQN experiments and documentation. All commits for experiments E1-E13+ are made to this branch.

```bash
# Always work on the same branch
git checkout feature/npu-aqn-test

# Commit changes (experiments, docs, fixes all go to this branch)
git add .
git commit -m "feat: E13h MXFP4 W4A4 baseline experiment"

# Push to both remotes
git push personal feature/npu-aqn-test
git push team feature/npu-aqn-test
```

---

## 3. Model Paths

```bash
# E8c checkpoint (Qwen2.5-1.5B-Instruct fine-tuned)
MODEL_PATH="/home/z00637938/workspace/verl/checkpoints/noisy_ops_e8c_forward_only/e8c_forward_only_5e-2/global_step_116/merged_hf"

# Base tokenizer
TOKENIZER_PATH="/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

# Test data
TEST_DATA_PATH="/data/z00637938/gsm8k/test.parquet"
```

---

## 4. Common Commands

```bash
# Run SRDD diagnostic (validation mode with simulated fault)
python scripts/srdd_error_finder.py \
    --model_path $MODEL_PATH \
    --ground_truth_layer 10 \
    --fault_type dead_zone \
    --fault_magnitude 0.3

# Run SRDD diagnostic (production mode - no reference needed)
python scripts/srdd_error_finder.py \
    --model_path $MODEL_PATH

# Run layer sensitivity diagnosis
python scripts/layer_sensitivity_diagnosis.py \
    --checkpoint $MODEL_PATH \
    --tokenizer $TOKENIZER_PATH \
    --test-data $TEST_DATA_PATH
```

---

## 5. Network Proxy (REQUIRED for internet access)

```bash
# Source proxy settings BEFORE any network operations
source /home/z00637938/setup_proxy.sh

# Proxy details:
# HTTP_PROXY: http://p_atlas:proxy%40123@90.255.4.119:8890
# HF_ENDPOINT: https://hf-mirror.com (Hugging Face mirror)
# UV_INDEX_URL: https://pypi.tuna.tsinghua.edu.cn/simple (PyPI mirror)

# Example: git operations
source /home/z00637938/setup_proxy.sh
cd /home/z00637938/workspace/llm-compressor
git pull

# Example: pip install
source /home/z00637938/setup_proxy.sh
pip install llmcompressor
```

---

## 6. llm-compressor (Quantization Library)

```bash
# Location
LLM_COMPRESSOR_PATH="/home/z00637938/workspace/llm-compressor"

# Install (if needed)
source /home/z00637938/setup_proxy.sh
pip install -e /home/z00637938/workspace/llm-compressor

# Available schemes:
# - MXFP4: MicroScaling FP4 (group size 32)
# - NVFP4A16: NVIDIA FP4 weights, 16-bit activations (group size 16)
# - W4A4: 4-bit weights, 4-bit activations

# Example: MXFP4 quantization
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
recipe = QuantizationModifier(targets="Linear", scheme="MXFP4", ignore=["lm_head"])
oneshot(model=model, recipe=recipe)
```

---

## 7. Environment

- GPU: A100-SXM4-80GB (8x)
- Container: verl-r3-test
- Python env: Already configured in container
- Key packages: torch, transformers, scipy, llmcompressor
