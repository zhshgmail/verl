"""
vLLM routing capture for MoE models.

This module provides hooks to capture routing decisions during vLLM generation,
without modifying vLLM source code (monkey-patching approach).

Supports both vLLM V0 and V1 architectures.
"""

import torch
import torch._dynamo

# CRITICAL: Disable torch.dynamo before any model operations
# This is required because our monkey-patch uses list.append() which dynamo cannot trace
# Must be done at module import time, before vLLM initializes
torch._dynamo.config.disable = True

import torch.nn.functional as F
from typing import Optional, List
from verl.utils.routing_playback import RoutingLog, BatchRoutingLogs, merge_routing_logs

# Module-level capture configuration
_capture_config = {
    'enabled': False,
    'capture_router_logits': True,
    'verbose': False,
}

# NOTE: vLLM V1 uses multiprocessing, but veRL's worker has direct in-process access
# Class attribute registry works fine because _load_model() runs inside worker process
# For standalone LLM() API usage, cross-process communication would be needed (not implemented)


def patch_vllm_moe_for_routing_capture(
    llm_engine=None,
    capture_router_logits: bool = True,
    verbose: bool = False
):
    """
    Patch vLLM MoE layers to capture routing decisions during generation.

    This monkey-patches Qwen2MoeSparseMoeBlock.forward() to intercept routing
    decisions before the CUDA kernel executes. Works with both V0 and V1 engines.

    Args:
        llm_engine: vLLM LLM engine instance (optional, for V0 compatibility)
        capture_router_logits: Whether to capture full router logits (for router LoRA training).
                               Set to False in production to save memory (4.75x reduction).
        verbose: Print debug information

    Returns:
        Number of MoE layers patched (0 if using V1 engine, actual count retrieved via get_routing_logs)

    Example:
        >>> from vllm import LLM
        >>> patch_vllm_moe_for_routing_capture(capture_router_logits=True)  # Patch before LLM creation
        >>> llm = LLM(model="Qwen/Qwen1.5-MoE-A2.7B")
        >>> outputs = llm.generate(prompts)
        >>> routing_logs = get_routing_logs_from_vllm()
    """
    try:
        from vllm.model_executor.models.qwen2_moe import Qwen2MoeSparseMoeBlock
    except ImportError:
        raise ImportError(
            "Cannot import Qwen2MoeSparseMoeBlock from vLLM. "
            "Make sure vLLM is installed and supports Qwen2 MoE models."
        )

    # Update global config
    _capture_config['enabled'] = True
    _capture_config['capture_router_logits'] = capture_router_logits
    _capture_config['verbose'] = verbose

    # Initialize class-level registry (survives multiprocessing)
    if not hasattr(Qwen2MoeSparseMoeBlock, '_routing_layers_registry'):
        Qwen2MoeSparseMoeBlock._routing_layers_registry = []

    # Store original forward method if not already patched
    if not hasattr(Qwen2MoeSparseMoeBlock, '_original_forward'):
        Qwen2MoeSparseMoeBlock._original_forward = Qwen2MoeSparseMoeBlock.forward

    # Apply torch._dynamo.disable to prevent torch.compile tracing issues
    # This is necessary because list.append() and other Python operations
    # used for routing capture are not supported by torch.dynamo
    @torch._dynamo.disable
    def forward_with_capture(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Modified forward that captures routing decisions.

        This is injected via monkey-patching. It wraps the original forward
        and adds routing capture after router, before FusedMoE kernel.

        Updated for vLLM 0.11.0+ which uses SharedFusedMoE to handle shared experts.
        """
        # Initialize layer on first call
        if not hasattr(self, '_routing_log_buffer'):
            self._routing_log_buffer = []
            self.layer_id = len(self.__class__._routing_layers_registry)
            self.__class__._routing_layers_registry.append(self)
            # Note: No print here - torch.compile doesn't support it

        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        # ===== ROUTER FORWARD (CAPTURE POINT) =====
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)

        # ===== CAPTURE ROUTING DECISIONS =====
        if _capture_config['enabled']:
            # Compute expert selection (same logic as vLLM CUDA kernel)
            routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
            routing_weights, expert_ids = torch.topk(
                routing_probs,
                k=self.experts.top_k,  # FusedMoE stores top_k
                dim=-1
            )

            # Normalize weights (same as vLLM)
            routing_weights = routing_weights.to(router_logits.dtype)

            # Move to CPU to free GPU memory immediately
            log = RoutingLog(
                layer_id=self.layer_id,
                expert_ids=expert_ids.cpu(),
                routing_weights=routing_weights.cpu(),
                router_logits=router_logits.cpu() if _capture_config['capture_router_logits'] else None,
            )

            self._routing_log_buffer.append(log)
            # Note: Verbose logging disabled - torch.compile doesn't support print()
        # ===== END CAPTURE =====

        # Continue with normal expert execution (SharedFusedMoE handles shared experts)
        # SharedFusedMoE.forward() returns tuple (shared_out, fused_out) per vLLM source:
        # vllm/model_executor/layers/fused_moe/shared_fused_moe.py:97
        result = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits
        )

        # Defensive handling for SharedFusedMoE return type
        # Tuple order verified from vLLM source: (shared_out, fused_out)
        if isinstance(result, tuple):
            shared_out, fused_out = result
            if shared_out is not None:
                final_hidden_states = shared_out + fused_out
            else:
                final_hidden_states = fused_out
        else:
            # Fallback for non-SharedFusedMoE (regular FusedMoE returns single tensor)
            final_hidden_states = result

        if self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states
            )

        return final_hidden_states.view(orig_shape)

    # Replace forward method
    Qwen2MoeSparseMoeBlock.forward = forward_with_capture

    if verbose:
        print(f"[Routing Capture] Patched Qwen2MoeSparseMoeBlock.forward()")
        print(f"[Routing Capture] capture_router_logits={capture_router_logits}")
        print(f"[Routing Capture] Layers will be registered on first forward pass")

    # Return 0 for V1, actual count will be available after model creation
    return 0


def get_routing_logs_from_vllm(
    llm_engine=None,
    batch_id: int = 0,
    clear_buffer: bool = True
) -> Optional[BatchRoutingLogs]:
    """
    Retrieve captured routing logs from vLLM after generation.

    Args:
        llm_engine: vLLM LLM engine instance (optional, not used in V1)
        batch_id: Unique identifier for this batch
        clear_buffer: Whether to clear buffers after retrieval

    Returns:
        BatchRoutingLogs containing all captured routing decisions,
        or None if no routing logs were captured

    Example:
        >>> outputs = llm.generate(prompts)
        >>> routing_logs = get_routing_logs_from_vllm(batch_id=0)
        >>> print(f"Captured {routing_logs.num_layers} layers, {routing_logs.num_tokens} tokens")
    """
    try:
        from vllm.model_executor.models.qwen2_moe import Qwen2MoeSparseMoeBlock
    except ImportError:
        return None

    if not hasattr(Qwen2MoeSparseMoeBlock, '_routing_layers_registry'):
        return None

    registry = Qwen2MoeSparseMoeBlock._routing_layers_registry
    if not registry:
        return None

    layers_logs = []
    num_tokens = None
    top_k = None
    num_experts = None

    for module in registry:
        if not hasattr(module, '_routing_log_buffer') or not module._routing_log_buffer:
            # No captures for this layer (shouldn't happen if patch worked)
            continue

        # Merge all captures from this layer
        # (auto-regressive generation may have multiple forward passes)
        layer_log = merge_routing_logs(module._routing_log_buffer)
        layers_logs.append(layer_log)

        # Infer metadata from first layer
        if num_tokens is None:
            num_tokens = layer_log.expert_ids.shape[0]
            top_k = layer_log.expert_ids.shape[1]
            if layer_log.router_logits is not None:
                num_experts = layer_log.router_logits.shape[1]
            else:
                # Infer from FusedMoE layer
                num_experts = module.experts.num_experts

        # Clear buffer if requested
        if clear_buffer:
            module._routing_log_buffer = []

    if not layers_logs:
        return None

    return BatchRoutingLogs(
        batch_id=batch_id,
        num_layers=len(layers_logs),
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_experts,
        layers=layers_logs,
    )


def disable_routing_capture(llm_engine=None):
    """
    Disable routing capture on all MoE layers.

    This stops capturing routing logs without unpatching the forward method.
    Useful for switching between training/inference modes.

    Args:
        llm_engine: vLLM LLM engine instance (optional, not used in V1)
    """
    global _capture_config
    _capture_config['enabled'] = False


def enable_routing_capture(llm_engine=None):
    """
    Enable routing capture on all MoE layers.

    This re-enables capturing after disable_routing_capture() was called.

    Args:
        llm_engine: vLLM LLM engine instance (optional, not used in V1)
    """
    global _capture_config
    _capture_config['enabled'] = True


__all__ = [
    'patch_vllm_moe_for_routing_capture',
    'get_routing_logs_from_vllm',
    'disable_routing_capture',
    'enable_routing_capture',
]
