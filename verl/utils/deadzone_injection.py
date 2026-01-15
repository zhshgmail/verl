# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
MXFP4 Deadzone Injection for SRDD-guided AQN Experiments.

This module provides unified deadzone injection that can be applied to:
1. SRDD diagnostic inference
2. vLLM rollout
3. Training forward/backward

Deadzone simulates MXFP4 quantization effects where small values are zeroed:
- Values with |x| < threshold * max(|x|) are set to 0
- This causes signal loss in layers with small activations
- AQN noise can help signals "break through" the deadzone

Usage:
    from verl.utils.deadzone_injection import (
        DeadzoneInjector,
        enable_deadzone_ops,
        disable_deadzone_ops,
    )

    # Method 1: Hook-based injection (for inference/vLLM)
    injector = DeadzoneInjector(
        model=model,
        fault_layer=15,
        threshold=0.01,  # 1% of max
    )
    injector.enable()

    # Method 2: Operator-level injection (for training)
    enable_deadzone_ops(
        layer_ids=[15],
        threshold=0.01,
    )
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Set
from contextlib import contextmanager
import threading

# ============================================================================
# Global Configuration
# ============================================================================

_DEADZONE_CONFIG_LOCK = threading.Lock()

# Global deadzone state
_DEADZONE_ENABLED = False
_DEADZONE_THRESHOLD = 0.01  # 1% of max by default
_DEADZONE_LAYERS: Set[int] = None  # None = all layers, set() = specific layers
_CURRENT_LAYER_ID: int = None

# Statistics
_DEADZONE_STATS = {
    'forward_calls': 0,
    'values_zeroed': 0,
    'total_values': 0,
}


def set_deadzone_layer(layer_id: int) -> None:
    """Set the current layer ID for deadzone injection."""
    global _CURRENT_LAYER_ID
    _CURRENT_LAYER_ID = layer_id


def get_deadzone_layer() -> int:
    """Get the current layer ID."""
    return _CURRENT_LAYER_ID


def should_apply_deadzone(layer_id: int = None) -> bool:
    """Check if deadzone should be applied to the given layer."""
    if not _DEADZONE_ENABLED:
        return False

    if _DEADZONE_LAYERS is None:
        return True  # All layers

    if layer_id is None:
        layer_id = _CURRENT_LAYER_ID

    if layer_id is None:
        return False  # No layer context

    return layer_id in _DEADZONE_LAYERS


def apply_deadzone(tensor: torch.Tensor, threshold: float = None) -> torch.Tensor:
    """
    Apply MXFP4 deadzone effect to a tensor.

    Args:
        tensor: Input tensor
        threshold: Deadzone threshold as fraction of max value (default: global config)

    Returns:
        Tensor with values below threshold zeroed
    """
    if threshold is None:
        threshold = _DEADZONE_THRESHOLD

    # Dynamic threshold based on tensor's max value
    max_val = tensor.abs().max()
    deadzone_threshold = threshold * max_val

    # Create mask for values in deadzone
    deadzone_mask = tensor.abs() < deadzone_threshold

    # Zero out values in deadzone
    result = tensor.masked_fill(deadzone_mask, 0.0)

    # Update stats
    global _DEADZONE_STATS
    _DEADZONE_STATS['forward_calls'] += 1
    _DEADZONE_STATS['values_zeroed'] += deadzone_mask.sum().item()
    _DEADZONE_STATS['total_values'] += tensor.numel()

    return result


# ============================================================================
# Hook-based Injection (for inference/vLLM)
# ============================================================================

class DeadzoneInjector:
    """
    Hook-based deadzone injector for inference and vLLM rollout.

    This class injects deadzone effects at the layer output level using
    forward hooks. It's suitable for:
    - SRDD diagnostic inference
    - vLLM rollout (after model is loaded)

    For training forward/backward, use the operator-level injection instead
    (enable_deadzone_ops).

    Args:
        model: The model to inject deadzone into
        fault_layer: Layer index to inject deadzone (0-indexed)
        threshold: Deadzone threshold as fraction of max value
        sparsity: Fraction of elements affected (1.0 = all)
    """

    def __init__(
        self,
        model,
        fault_layer: int,
        threshold: float = 0.01,
        sparsity: float = 1.0,
    ):
        self.model = model
        self.fault_layer = fault_layer
        self.threshold = threshold
        self.sparsity = sparsity
        self.hook_handle = None

        # Fixed sparse mask (created lazily)
        self._sparse_mask = None

        # Find target layer
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            self.target_module = model.model.layers[fault_layer]
        elif hasattr(model, 'layers'):
            self.target_module = model.layers[fault_layer]
        else:
            raise ValueError(f"Cannot find layer {fault_layer} in model")

        # RNG for sparse mask generation
        device = next(model.parameters()).device
        self.rng = torch.Generator(device=device)
        self.rng.seed()

    def _deadzone_hook(self, module, input, output):
        """Forward hook that applies deadzone effect."""
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        # Calculate dynamic threshold
        max_val = hidden_states.abs().max()
        deadzone_threshold = self.threshold * max_val

        # Create deadzone mask
        deadzone_mask = hidden_states.abs() < deadzone_threshold

        # Apply sparsity if needed
        if self.sparsity < 1.0:
            hidden_dim = hidden_states.shape[-1]
            if self._sparse_mask is None or self._sparse_mask.shape[0] != hidden_dim:
                self._sparse_mask = torch.rand(
                    hidden_dim,
                    device=hidden_states.device,
                    generator=self.rng,
                ) < self.sparsity
                num_affected = self._sparse_mask.sum().item()
                print(f"  [DEADZONE] Sparse mask: {num_affected}/{hidden_dim} "
                      f"({100*num_affected/hidden_dim:.1f}%) affected")

            # Only apply deadzone to sparse positions
            sparse_mask = self._sparse_mask.expand_as(hidden_states)
            deadzone_mask = deadzone_mask & sparse_mask

        # Apply deadzone
        hidden_states = hidden_states.masked_fill(deadzone_mask, 0.0)

        if rest is not None:
            return (hidden_states,) + rest
        return hidden_states

    def enable(self):
        """Enable deadzone injection."""
        if self.hook_handle is None:
            self.hook_handle = self.target_module.register_forward_hook(
                self._deadzone_hook
            )
            sparsity_str = f" (sparsity={self.sparsity*100:.0f}%)" if self.sparsity < 1.0 else ""
            print(f"[DEADZONE] Enabled on layer {self.fault_layer}: "
                  f"threshold={self.threshold*100:.1f}%{sparsity_str}")

    def disable(self):
        """Disable deadzone injection."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
            print(f"[DEADZONE] Disabled on layer {self.fault_layer}")

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, *args):
        self.disable()


# ============================================================================
# Operator-level Injection (for training forward/backward)
# ============================================================================

# Original operators (saved on first enable)
_ORIGINAL_MATMUL = None
_ORIGINAL_LINEAR = None
_ORIGINAL_BMM = None


class DeadzoneMatMul(torch.autograd.Function):
    """
    Matmul with deadzone effect in forward pass.

    The deadzone causes gradients to be zero for zeroed values,
    simulating the gradient vanishing effect of quantization deadzone.
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Use original matmul
        result = _ORIGINAL_MATMUL(a, b)

        # Apply deadzone if enabled and in target layer
        if should_apply_deadzone():
            result = apply_deadzone(result)

        ctx.save_for_backward(a, b, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        a, b, result = ctx.saved_tensors

        # Gradient is zero where deadzone zeroed the output
        if should_apply_deadzone():
            # Mask gradient where result was zeroed
            grad_mask = result == 0
            grad_output = grad_output.masked_fill(grad_mask, 0.0)

        # Normal backward
        grad_a = _ORIGINAL_MATMUL(grad_output, b.transpose(-1, -2))
        grad_b = _ORIGINAL_MATMUL(a.transpose(-1, -2), grad_output)

        return grad_a, grad_b


class DeadzoneBMM(torch.autograd.Function):
    """Batch matmul with deadzone effect."""

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        result = _ORIGINAL_BMM(a, b)

        if should_apply_deadzone():
            result = apply_deadzone(result)

        ctx.save_for_backward(a, b, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        a, b, result = ctx.saved_tensors

        if should_apply_deadzone():
            grad_mask = result == 0
            grad_output = grad_output.masked_fill(grad_mask, 0.0)

        grad_a = _ORIGINAL_BMM(grad_output, b.transpose(-1, -2))
        grad_b = _ORIGINAL_BMM(a.transpose(-1, -2), grad_output)

        return grad_a, grad_b


def deadzone_matmul(a: torch.Tensor, b: torch.Tensor, **kwargs) -> torch.Tensor:
    """Drop-in replacement for torch.matmul with deadzone."""
    if not _DEADZONE_ENABLED:
        return _ORIGINAL_MATMUL(a, b)
    return DeadzoneMatMul.apply(a, b)


def deadzone_bmm(a: torch.Tensor, b: torch.Tensor, **kwargs) -> torch.Tensor:
    """Drop-in replacement for torch.bmm with deadzone."""
    if not _DEADZONE_ENABLED:
        return _ORIGINAL_BMM(a, b)
    return DeadzoneBMM.apply(a, b)


def deadzone_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """Drop-in replacement for F.linear with deadzone."""
    if not _DEADZONE_ENABLED:
        return _ORIGINAL_LINEAR(input, weight, bias)

    output = DeadzoneMatMul.apply(input, weight.t())
    if bias is not None:
        output = output + bias
    return output


def enable_deadzone_ops(
    layer_ids: List[int] = None,
    threshold: float = 0.01,
) -> None:
    """
    Enable operator-level deadzone injection.

    This monkey-patches torch.matmul, torch.bmm, and F.linear to apply
    deadzone effects during training.

    Args:
        layer_ids: List of layer indices to apply deadzone to.
                   None = all layers (dangerous for training!)
        threshold: Deadzone threshold as fraction of max value
    """
    global _DEADZONE_ENABLED, _DEADZONE_THRESHOLD, _DEADZONE_LAYERS
    global _ORIGINAL_MATMUL, _ORIGINAL_LINEAR, _ORIGINAL_BMM

    if _DEADZONE_ENABLED:
        print("[DEADZONE OPS] Already enabled, updating config")

    # Save originals (only once)
    if _ORIGINAL_MATMUL is None:
        _ORIGINAL_MATMUL = torch.matmul
        _ORIGINAL_LINEAR = F.linear
        _ORIGINAL_BMM = torch.bmm

    # Set config
    with _DEADZONE_CONFIG_LOCK:
        _DEADZONE_THRESHOLD = threshold
        _DEADZONE_LAYERS = set(layer_ids) if layer_ids is not None else None
        _DEADZONE_ENABLED = True

    # Monkey-patch
    torch.matmul = deadzone_matmul
    torch.bmm = deadzone_bmm
    F.linear = deadzone_linear

    layers_str = f"layers {sorted(_DEADZONE_LAYERS)}" if _DEADZONE_LAYERS else "ALL layers"
    print(f"[DEADZONE OPS] Enabled: threshold={threshold*100:.1f}%, {layers_str}")


def disable_deadzone_ops() -> dict:
    """
    Disable operator-level deadzone injection.

    Returns:
        Statistics dict
    """
    global _DEADZONE_ENABLED

    if not _DEADZONE_ENABLED:
        return _DEADZONE_STATS.copy()

    # Restore originals
    if _ORIGINAL_MATMUL is not None:
        torch.matmul = _ORIGINAL_MATMUL
        torch.bmm = _ORIGINAL_BMM
        F.linear = _ORIGINAL_LINEAR

    _DEADZONE_ENABLED = False

    stats = _DEADZONE_STATS.copy()
    if stats['total_values'] > 0:
        zero_rate = stats['values_zeroed'] / stats['total_values'] * 100
        print(f"[DEADZONE OPS] Disabled. Stats: {stats['values_zeroed']:,} values zeroed "
              f"({zero_rate:.2f}%) in {stats['forward_calls']:,} calls")
    else:
        print("[DEADZONE OPS] Disabled. No statistics recorded.")

    return stats


@contextmanager
def deadzone_ops_context(
    layer_ids: List[int] = None,
    threshold: float = 0.01,
):
    """
    Context manager for deadzone injection during training.

    Example:
        with deadzone_ops_context(layer_ids=[15], threshold=0.01):
            output = model(input)
            loss.backward()
    """
    enable_deadzone_ops(layer_ids=layer_ids, threshold=threshold)
    try:
        yield
    finally:
        disable_deadzone_ops()


def get_deadzone_stats() -> dict:
    """Get deadzone injection statistics."""
    stats = _DEADZONE_STATS.copy()
    if stats['total_values'] > 0:
        stats['zero_rate'] = stats['values_zeroed'] / stats['total_values']
    else:
        stats['zero_rate'] = 0.0
    return stats


def reset_deadzone_stats() -> None:
    """Reset deadzone statistics."""
    global _DEADZONE_STATS
    _DEADZONE_STATS = {
        'forward_calls': 0,
        'values_zeroed': 0,
        'total_values': 0,
    }


# ============================================================================
# Layer Hooks for Tracking Current Layer
# ============================================================================

def register_deadzone_layer_hooks(model) -> int:
    """
    Register forward hooks to track current layer during forward pass.

    This enables layer-aware deadzone injection.

    Args:
        model: The PyTorch model (transformer)

    Returns:
        Number of hooks registered
    """
    import re

    hooks_registered = 0

    def make_hook(lid):
        def hook(module, input):
            set_deadzone_layer(lid)
        return hook

    for name, module in model.named_modules():
        if '.layers.' in name:
            match = re.search(r'\.layers\.(\d+)$', name)
            if match:
                layer_id = int(match.group(1))
                module.register_forward_pre_hook(make_hook(layer_id))
                hooks_registered += 1

    if hooks_registered > 0:
        print(f"[DEADZONE] Registered {hooks_registered} layer hooks")

    return hooks_registered


__all__ = [
    # Hook-based injection
    'DeadzoneInjector',
    # Operator-level injection
    'enable_deadzone_ops',
    'disable_deadzone_ops',
    'deadzone_ops_context',
    # Layer tracking
    'set_deadzone_layer',
    'get_deadzone_layer',
    'should_apply_deadzone',
    'register_deadzone_layer_hooks',
    # Utilities
    'apply_deadzone',
    'get_deadzone_stats',
    'reset_deadzone_stats',
]
