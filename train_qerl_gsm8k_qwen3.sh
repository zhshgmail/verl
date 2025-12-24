#!/bin/bash
# Full GSM8K training with QeRL-style R3 + AQN + LoRA on Qwen3-30B-A3B-Base
# Designed for training on 8x H100 80GB GPUs
# Updated: 2025-12-25 - Qwen3 MoE with 128 experts, 48 layers
set -x

# ============================================================================
# Model and Data Paths
# ============================================================================
# Qwen3-30B-A3B-Base: 30B total params, 3B activated, 128 experts, 48 layers
# NOTE: This is a BASE model - may need fine-tuning or special prompting for instruction following
HF_MODEL_PATH=/data/z00637938/hub/models--Qwen--Qwen3-30B-A3B-Base/snapshots/1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9
TRAIN_DATA_FILE=/tmp/gsm8k_processed/train.parquet
VAL_DATA_FILE=/tmp/gsm8k_processed/test.parquet

# Output directory for checkpoints and logs
OUTPUT_DIR=/data/z00637938/qerl_gsm8k_qwen3_training
mkdir -p $OUTPUT_DIR

# ============================================================================
# Environment Variables
# ============================================================================
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_CUDA_ARCH_LIST=9.0  # H100 architecture
export VERL_DISABLE_DYNAMO=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# TensorBoard directory
export TENSORBOARD_DIR=${OUTPUT_DIR}/tensorboard

# ============================================================================
# Parallelism Configuration (8x H100 80GB)
# Qwen3-30B-A3B: 128 experts, 48 layers, needs more parallelism
# ============================================================================
NODES=1
PP=1                    # Pipeline parallelism
VPP=None               # Virtual pipeline parallelism
TP=2                    # Tensor parallelism (hidden_size=2048, same as Qwen1.5)
EP=4                    # Expert parallelism (128 experts / 4 = 32 experts per rank)
ETP=1                   # Expert tensor parallelism
VLLM_INFER_TP=2        # vLLM inference TP

# ============================================================================
# Training Hyperparameters
# ============================================================================
# Batch sizes - reduced due to 2x more layers (48 vs 24)
TRAIN_BATCH_SIZE=32     # Reduced from 64 due to larger model
PPO_MINI_BATCH_SIZE=8   # Reduced from 16
MICRO_BATCH_SIZE=1      # Keep at 1 for memory efficiency

# Sequence lengths - same as Qwen1.5 config
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=1024

# Learning rate - slightly lower for larger model
LEARNING_RATE=5e-6

# Training duration
# GSM8K train has ~7.5K samples, batch_size=32 -> ~234 steps/epoch
# For similar training: ~1000 steps = ~4 epochs (2x steps per epoch due to smaller batch)
TOTAL_EPOCHS=1
TOTAL_TRAINING_STEPS=1000

# Checkpointing and evaluation
SAVE_FREQ=100           # Save checkpoint every N steps
TEST_FREQ=50            # Evaluate frequently to track progress

# ============================================================================
# LoRA Configuration (QeRL-style: train adapters, freeze base weights)
# Requires: pip install megatron-bridge
# ============================================================================
ENABLE_LORA=True
LORA_RANK=32
LORA_ALPHA=64
LORA_DROPOUT=0.05
# Target modules for LoRA (attention + MLP, but NOT router)
# Qwen3 uses GQA (32 heads, 4 KV heads) but LoRA targets are same

# ============================================================================
# AQN (Adaptive Quantization Noise) Configuration
# ============================================================================
NOISE_ENABLED=True
SIGMA_START=0.01        # Starting noise level
SIGMA_END=0.001         # Ending noise level
NUM_STAGES=10           # Number of decay stages

# ============================================================================
# R3 (Rollout Router Replay) Configuration
# ============================================================================
R3_ENABLED=True

# ============================================================================
# Offloading Configuration (for memory efficiency)
# More aggressive offloading for larger model
# ============================================================================
PARAM_OFFLOAD=True
OPTIMIZER_OFFLOAD=True
GRAD_OFFLOAD=True
GPU_MEMORY_UTILIZATION=0.35  # Lower than Qwen1.5 due to larger model

# ============================================================================
# Build Configuration Arrays
# ============================================================================
DATA=(
    data.train_files="$TRAIN_DATA_FILE"
    data.val_files="$VAL_DATA_FILE"
    data.train_batch_size=$TRAIN_BATCH_SIZE
    data.max_prompt_length=$MAX_PROMPT_LENGTH
    data.max_response_length=$MAX_RESPONSE_LENGTH
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.use_fused_kernels=False
    actor_rollout_ref.model.path="$HF_MODEL_PATH"
)

# LoRA configuration (megatron backend uses nested lora config)
if [ "$ENABLE_LORA" = "True" ]; then
    MODEL+=(
        actor_rollout_ref.model.lora.rank=$LORA_RANK
        actor_rollout_ref.model.lora.alpha=$LORA_ALPHA
        actor_rollout_ref.model.lora.dropout=$LORA_DROPOUT
        actor_rollout_ref.model.lora.type=lora
        # Target modules: attention (linear_qkv, linear_proj) + MLP (linear_fc1, linear_fc2)
        # This includes experts but NOT the MoE router
        'actor_rollout_ref.model.lora.target_modules=["linear_qkv","linear_proj","linear_fc1","linear_fc2"]'
    )
fi

ACTOR=(
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.entropy_coeff=0.001
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP
    actor_rollout_ref.actor.megatron.param_offload=$PARAM_OFFLOAD
    actor_rollout_ref.actor.megatron.optimizer_offload=$OPTIMIZER_OFFLOAD
    actor_rollout_ref.actor.megatron.grad_offload=$GRAD_OFFLOAD
)

# LoRA requires vanilla_mbridge=False for Megatron-Bridge provider
if [ "$ENABLE_LORA" = "True" ]; then
    ACTOR+=(
        actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    )
fi

# R3 configuration for actor
if [ "$R3_ENABLED" = "True" ]; then
    ACTOR+=("actor_rollout_ref.actor.router_replay.mode=R3")
fi

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=2048
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE
    actor_rollout_ref.rollout.tensor_model_parallel_size=$VLLM_INFER_TP
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION
    actor_rollout_ref.rollout.n=5
    actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LENGTH
)

# R3 configuration for rollout
if [ "$R3_ENABLED" = "True" ]; then
    ROLLOUT+=("actor_rollout_ref.rollout.enable_rollout_routing_replay=True")
fi

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$PP
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TP
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=$EP
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=$ETP
    actor_rollout_ref.ref.megatron.param_offload=$PARAM_OFFLOAD
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=2048
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
)

# AQN Noise Injection configuration
NOISE_CONFIG=()
if [ "$NOISE_ENABLED" = "True" ]; then
    NOISE_CONFIG=(
        +noise_injection.enabled=True
        +noise_injection.sigma_start=$SIGMA_START
        +noise_injection.sigma_end=$SIGMA_END
        +noise_injection.num_stages=$NUM_STAGES
        '+noise_injection.target_modules=["post_attention_layernorm"]'
        '+noise_injection.exclude_patterns=["input_layernorm","router"]'
    )
fi

TRAINER=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    trainer.critic_warmup=0
    'trainer.logger=["console","tensorboard"]'
    trainer.project_name=qerl_gsm8k
    trainer.experiment_name=qwen3_30b_a3b_r3_aqn_lora
    trainer.nnodes=$NODES
    trainer.n_gpus_per_node=8
    trainer.save_freq=$SAVE_FREQ
    trainer.test_freq=$TEST_FREQ
    trainer.total_epochs=$TOTAL_EPOCHS
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS
    trainer.balance_batch=False
    trainer.val_before_train=True
    trainer.default_local_dir=$OUTPUT_DIR
)

# ============================================================================
# Run Training
# ============================================================================
echo "=============================================="
echo "QeRL GSM8K Training - Qwen3-30B-A3B"
echo "=============================================="
echo "Model: $HF_MODEL_PATH"
echo "Architecture: 30B total params, 3B activated, 128 experts, 48 layers"
echo "Output: $OUTPUT_DIR"
echo "TensorBoard: $TENSORBOARD_DIR"
echo "Parallelism: TP=$TP, EP=$EP (128 experts / 4 = 32 per rank)"
echo "Batch: train=$TRAIN_BATCH_SIZE, mini=$PPO_MINI_BATCH_SIZE, micro=$MICRO_BATCH_SIZE"
echo "Response Length: $MAX_RESPONSE_LENGTH tokens"
echo "LoRA: enabled=$ENABLE_LORA, rank=$LORA_RANK, alpha=$LORA_ALPHA"
echo "AQN: sigma $SIGMA_START -> $SIGMA_END over $NUM_STAGES stages"
echo "R3: $R3_ENABLED"
echo "GRPO samples per prompt: n=5"
echo "=============================================="

# Use megatron config for MoE model with EP/TP parallelism
python3 -m verl.trainer.main_ppo \
    --config-name=ppo_megatron_trainer \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${NOISE_CONFIG[@]}" \
    "${TRAINER[@]}" \
    "$@"

# ============================================================================
# Post-Training Summary
# ============================================================================
echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "TensorBoard logs: $TENSORBOARD_DIR"
echo ""
echo "To view TensorBoard:"
echo "  tensorboard --logdir=$TENSORBOARD_DIR --port=6006"
echo ""
echo "To merge LoRA weights, see: docs/LoRA_Merge_Guide.md"
echo "=============================================="
