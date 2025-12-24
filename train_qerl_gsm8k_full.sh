#!/bin/bash
# Full GSM8K training with QeRL-style R3 + AQN + LoRA on Qwen1.5-MoE-A2.7B-Chat
# Designed for overnight training on 8x A100 GPUs
# Updated: 2025-12-24 - Using CHAT model, LoRA enabled, 1024 response length
set -x

# ============================================================================
# Model and Data Paths
# ============================================================================
# Using CHAT/INSTRUCT model (not BASE) for proper instruction following
HF_MODEL_PATH=/data/z00637938/models--Qwen--Qwen1.5-MoE-A2.7B-Chat/snapshots/latest
TRAIN_DATA_FILE=/tmp/gsm8k_processed/train.parquet
VAL_DATA_FILE=/tmp/gsm8k_processed/test.parquet

# Output directory for checkpoints and logs
OUTPUT_DIR=/data/z00637938/qerl_gsm8k_training_v2
mkdir -p $OUTPUT_DIR

# ============================================================================
# Environment Variables
# ============================================================================
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_CUDA_ARCH_LIST=8.0
export VERL_DISABLE_DYNAMO=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# TensorBoard directory
export TENSORBOARD_DIR=${OUTPUT_DIR}/tensorboard

# ============================================================================
# Parallelism Configuration (8x A100 80GB)
# ============================================================================
NODES=1
PP=1                    # Pipeline parallelism
VPP=None               # Virtual pipeline parallelism
TP=2                    # Tensor parallelism
EP=2                    # Expert parallelism
ETP=1                   # Expert tensor parallelism
VLLM_INFER_TP=2        # vLLM inference TP

# ============================================================================
# Training Hyperparameters
# ============================================================================
# Batch sizes - reduced for LoRA + longer sequences
TRAIN_BATCH_SIZE=64     # Total batch size per step
PPO_MINI_BATCH_SIZE=16  # Mini-batch size for PPO updates
MICRO_BATCH_SIZE=1      # Reduced for longer sequences with LoRA

# Sequence lengths - expanded for better math reasoning
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=1024

# Learning rate - higher for LoRA (only training adapters)
LEARNING_RATE=1e-5

# Training duration
# GSM8K train has ~7.5K samples, batch_size=64 -> ~117 steps/epoch
# For overnight run: ~2000 steps = ~17 epochs
# Using explicit steps instead of epochs (megatron optimizer needs this)
TOTAL_EPOCHS=1
TOTAL_TRAINING_STEPS=2000

# Checkpointing and evaluation
SAVE_FREQ=200           # Save checkpoint every N steps
TEST_FREQ=50            # Evaluate more frequently to track progress

# ============================================================================
# LoRA Configuration (QeRL-style: train adapters, freeze base weights)
# Requires: pip install megatron-bridge
# ============================================================================
ENABLE_LORA=True
LORA_RANK=32
LORA_ALPHA=64
LORA_DROPOUT=0.05
# Target modules for LoRA (attention + MLP, but NOT router)
# For Megatron: linear_qkv covers Q,K,V; linear_proj is output projection
# linear_fc1 and linear_fc2 are FFN layers (including experts)
# Router is excluded to maintain MoE routing stability

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
# ============================================================================
PARAM_OFFLOAD=True
OPTIMIZER_OFFLOAD=True
GRAD_OFFLOAD=True
GPU_MEMORY_UTILIZATION=0.4

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
    trainer.experiment_name=moe_chat_r3_aqn_lora_v2
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
echo "QeRL GSM8K Full Training v2"
echo "=============================================="
echo "Model: $HF_MODEL_PATH (CHAT/INSTRUCT)"
echo "Output: $OUTPUT_DIR"
echo "TensorBoard: $TENSORBOARD_DIR"
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
