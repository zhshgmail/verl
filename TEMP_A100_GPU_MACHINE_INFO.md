# A100 GPU Machine Information

## Connection Details

| Item | Value |
|------|-------|
| **SSH Host** | `root@90.90.102.18` |
| **Hostname** | `huawei` |
| **Container** | `verl-r3-test` |
| **GPUs** | 8x NVIDIA A100-SXM4-80GB |
| **Memory** | 81920 MiB per GPU (80GB) |

## Quick Access

```bash
# SSH to A100 machine
ssh root@90.90.102.18

# Enter container
docker exec -it verl-r3-test bash

# One-liner to run command in container
ssh root@90.90.102.18 "docker exec verl-r3-test bash -c 'YOUR_COMMAND'"
```

## Environment Setup (inside container)

```bash
cd /home/z00637938/workspace/verl
export MODEL_PATH=/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306
export TRAIN_DATA=/data/z00637938/gsm8k/train.parquet
export VAL_DATA=/data/z00637938/gsm8k/test.parquet
```

## Clean Ray Before Tests

```bash
# Clean Ray zombie processes
ssh root@90.90.102.18 "docker exec verl-r3-test bash -c 'ray stop --force; pkill -9 -f ray; pkill -9 -f python'"
```

## Run HW Error Injection Tests

```bash
# Linear layers (197 hooks, 95% FLOPs) - RECOMMENDED
ssh root@90.90.102.18 "docker exec verl-r3-test bash -c '
cd /home/z00637938/workspace/verl
export MODEL_PATH=/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306
export TRAIN_DATA=/data/z00637938/gsm8k/train.parquet
export VAL_DATA=/data/z00637938/gsm8k/test.parquet
bash scripts/test_hw_error_injection_a100.sh linear 8 1e-5
'"
```

## Other Containers on This Machine

| Container | Status | Purpose |
|-----------|--------|---------|
| `verl-r3-test` | Active | Main test container for HW error injection |
| `verl-fp8-container` | Active | FP8 experiments |

## Available Models

| Model | Path |
|-------|------|
| Qwen2.5-1.5B-Instruct | `/data/z00637938/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| DeepSeek-R1-Distill-Qwen-1.5B | `/data/z00637938/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` |
| Qwen3-30B-A3B-Base | `/data/z00637938/hub/models--Qwen--Qwen3-30B-A3B-Base/snapshots/1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9` |

## Related Documentation

- [HW_ERROR_INJECTION_EXPERIMENTS.md](docs/qerl/HW_ERROR_INJECTION_EXPERIMENTS.md)
- [AQN_ACCURACY_ANALYSIS.md](docs/qerl/AQN_ACCURACY_ANALYSIS.md)
- [SRDD_MXFP4_QUANT_EXPERIMENT.md](docs/qerl/SRDD_MXFP4_QUANT_EXPERIMENT.md)
