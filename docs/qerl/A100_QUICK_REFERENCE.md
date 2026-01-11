# A100 Quick Reference for Agents

**Purpose**: Quick reference for accessing A100 GPU environment for testing.

## Connection

```bash
# SSH to A100 server
ssh root@90.90.102.18

# Enter docker container
docker exec -it verl-r3-test bash

# Navigate to workspace
cd /home/z00637938/workspace/verl
```

## Model Paths

```bash
# E8c checkpoint (Qwen2.5-1.5B-Instruct fine-tuned)
MODEL_PATH="/home/z00637938/workspace/verl/checkpoints/noisy_ops_e8c_forward_only/e8c_forward_only_5e-2/global_step_116/merged_hf"

# Base tokenizer
TOKENIZER_PATH="/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

# Test data
TEST_DATA_PATH="/data/z00637938/gsm8k/test.parquet"
```

## Common Commands

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

## Network Proxy (REQUIRED for internet access)

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

## llm-compressor (Quantization Library)

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

## Environment

- GPU: A100-SXM4-80GB (8x)
- Container: verl-r3-test
- Python env: Already configured in container
- Key packages: torch, transformers, scipy, llmcompressor
