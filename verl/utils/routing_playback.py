"""
Routing playback for MoE models in RL training.

This module provides data structures and utilities for:
1. Capturing routing decisions during rollout (vLLM)
2. Replaying routing decisions during training (Megatron)
3. Training router LoRA while maintaining expert selection consistency

NVFP4 Quantization Compatibility:
- Router weights are quantized to NVFP4 (4-bit floating point)
- LoRA adapters are FP16/BF16 (full precision, trainable)
- router_logits (captured) are FP16/BF16 (outputs after dequantization)
- expert_ids and routing_weights (captured) are derived from FP16 router_logits
- No special dtype handling needed for routing playback

Architecture:
  Input (FP16) → Router(NVFP4 + LoRA) → router_logits (FP16) → [CAPTURE]
                                                               ↓
                                        expert_ids, routing_weights
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch
import pickle


@dataclass
class RoutingLog:
    """
    Routing decisions for one MoE layer in one batch.

    Attributes:
        layer_id: Which MoE layer (0-indexed)
        expert_ids: Selected expert indices, shape (num_tokens, top_k)
        routing_weights: Weights for combining expert outputs, shape (num_tokens, top_k)
        router_logits: Raw router output, shape (num_tokens, num_experts). Optional for production.
        token_positions: Token positions within sequences, shape (num_tokens,). Optional for debugging.
        sequence_ids: Which sequence each token belongs to, shape (num_tokens,). Optional for debugging.
    """

    layer_id: int
    expert_ids: torch.Tensor  # (num_tokens, top_k)
    routing_weights: torch.Tensor  # (num_tokens, top_k)
    router_logits: Optional[torch.Tensor] = None  # (num_tokens, num_experts)

    # Optional metadata for debugging
    token_positions: Optional[torch.Tensor] = None  # (num_tokens,)
    sequence_ids: Optional[torch.Tensor] = None  # (num_tokens,)

    def __post_init__(self):
        """Validate tensor shapes."""
        num_tokens, top_k = self.expert_ids.shape
        assert self.routing_weights.shape == (num_tokens, top_k), \
            f"routing_weights shape mismatch: expected ({num_tokens}, {top_k}), got {self.routing_weights.shape}"

        if self.router_logits is not None:
            assert self.router_logits.shape[0] == num_tokens, \
                f"router_logits first dim mismatch: expected {num_tokens}, got {self.router_logits.shape[0]}"

        if self.token_positions is not None:
            assert self.token_positions.shape == (num_tokens,), \
                f"token_positions shape mismatch: expected ({num_tokens},), got {self.token_positions.shape}"

        if self.sequence_ids is not None:
            assert self.sequence_ids.shape == (num_tokens,), \
                f"sequence_ids shape mismatch: expected ({num_tokens},), got {self.sequence_ids.shape}"

    def to(self, device: torch.device) -> "RoutingLog":
        """Move all tensors to specified device."""
        return RoutingLog(
            layer_id=self.layer_id,
            expert_ids=self.expert_ids.to(device),
            routing_weights=self.routing_weights.to(device),
            router_logits=self.router_logits.to(device) if self.router_logits is not None else None,
            token_positions=self.token_positions.to(device) if self.token_positions is not None else None,
            sequence_ids=self.sequence_ids.to(device) if self.sequence_ids is not None else None,
        )

    def cpu(self) -> "RoutingLog":
        """Move all tensors to CPU."""
        return self.to(torch.device('cpu'))

    def memory_size(self) -> int:
        """Estimate memory size in bytes."""
        size = 0
        size += self.expert_ids.numel() * self.expert_ids.element_size()
        size += self.routing_weights.numel() * self.routing_weights.element_size()
        if self.router_logits is not None:
            size += self.router_logits.numel() * self.router_logits.element_size()
        if self.token_positions is not None:
            size += self.token_positions.numel() * self.token_positions.element_size()
        if self.sequence_ids is not None:
            size += self.sequence_ids.numel() * self.sequence_ids.element_size()
        return size


@dataclass
class BatchRoutingLogs:
    """
    All routing logs for one batch across all MoE layers.

    Attributes:
        batch_id: Unique identifier for this batch
        num_layers: Number of MoE layers
        num_tokens: Total number of tokens in batch
        top_k: Number of experts selected per token
        num_experts: Total number of experts per layer
        layers: List of RoutingLog, one per MoE layer
    """

    batch_id: int
    num_layers: int
    num_tokens: int
    top_k: int
    num_experts: int
    layers: List[RoutingLog] = field(default_factory=list)

    def __post_init__(self):
        """Validate consistency."""
        if self.layers:
            assert len(self.layers) == self.num_layers, \
                f"Expected {self.num_layers} layers, got {len(self.layers)}"

            for i, layer_log in enumerate(self.layers):
                assert layer_log.layer_id == i, \
                    f"Layer {i} has wrong layer_id: {layer_log.layer_id}"
                assert layer_log.expert_ids.shape[1] == self.top_k, \
                    f"Layer {i} has wrong top_k: {layer_log.expert_ids.shape[1]} != {self.top_k}"

    def to(self, device: torch.device) -> "BatchRoutingLogs":
        """Move all tensors to specified device."""
        return BatchRoutingLogs(
            batch_id=self.batch_id,
            num_layers=self.num_layers,
            num_tokens=self.num_tokens,
            top_k=self.top_k,
            num_experts=self.num_experts,
            layers=[layer.to(device) for layer in self.layers],
        )

    def cpu(self) -> "BatchRoutingLogs":
        """Move all tensors to CPU."""
        return self.to(torch.device('cpu'))

    def memory_size(self) -> int:
        """Estimate total memory size in bytes."""
        return sum(layer.memory_size() for layer in self.layers)

    def serialize(self) -> bytes:
        """Serialize to bytes for storage/transfer."""
        return pickle.dumps(self)

    @staticmethod
    def deserialize(data: bytes) -> "BatchRoutingLogs":
        """Deserialize from bytes."""
        return pickle.loads(data)


def merge_routing_logs(logs: List[RoutingLog]) -> RoutingLog:
    """
    Merge multiple RoutingLog captures from the same layer.

    This is needed when a layer's forward() is called multiple times
    during generation (e.g., auto-regressive decoding).

    Args:
        logs: List of RoutingLog from the same layer

    Returns:
        Merged RoutingLog with concatenated tensors
    """
    if not logs:
        raise ValueError("Cannot merge empty list of logs")

    if len(logs) == 1:
        return logs[0]

    # All logs should be from same layer
    layer_id = logs[0].layer_id
    assert all(log.layer_id == layer_id for log in logs), "All logs must be from same layer"

    # Concatenate tensors along token dimension
    merged = RoutingLog(
        layer_id=layer_id,
        expert_ids=torch.cat([log.expert_ids for log in logs], dim=0),
        routing_weights=torch.cat([log.routing_weights for log in logs], dim=0),
        router_logits=torch.cat([log.router_logits for log in logs], dim=0)
        if logs[0].router_logits is not None
        else None,
        token_positions=torch.cat([log.token_positions for log in logs], dim=0)
        if logs[0].token_positions is not None
        else None,
        sequence_ids=torch.cat([log.sequence_ids for log in logs], dim=0)
        if logs[0].sequence_ids is not None
        else None,
    )

    return merged


__all__ = [
    "RoutingLog",
    "BatchRoutingLogs",
    "merge_routing_logs",
]
