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
Operator-level HW Error Injection via Custom Autograd Functions.

This module provides noisy versions of core PyTorch operators (matmul, linear)
that inject errors in BOTH forward AND backward passes, simulating real HW errors
more accurately than module-level hooks.

Key differences from module-level injection:
1. Errors in backward pass (affects gradients)
2. No phase distinction (rollout vs training)
3. Operator granularity (not layer boundaries)

Usage:
    from verl.utils.noisy_ops import enable_noisy_ops, disable_noisy_ops

    # Enable globally with 1e-4 error scale
    enable_noisy_ops(error_scale=1e-4)

    # ... training code ...
    # All torch.matmul and F.linear calls will have errors injected

    # Disable when done
    disable_noisy_ops()

Alternative usage with context manager:
    with noisy_ops_context(error_scale=1e-4):
        # ... training code with noisy ops ...
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from contextlib import contextmanager
import logging
import threading

logger = logging.getLogger(__name__)

# Thread safety lock for phase configuration
_NOISE_CONFIG_LOCK = threading.Lock()

# Global state
_NOISY_OPS_ENABLED = False
_NOISY_OPS_FORWARD_ENABLED = True   # Inject noise in forward pass
_NOISY_OPS_BACKWARD_ENABLED = True  # Inject noise in backward pass
_ERROR_SCALE = 1e-5
_ERROR_TYPE = 'relative_gaussian'  # 'relative_gaussian', 'absolute_gaussian'
_ALL_OPS_MODE = False  # When True, inject noise into ALL operators (like quant_compute)
_ORIGINAL_MATMUL = None
_ORIGINAL_LINEAR = None
_ORIGINAL_BMM = None
_ORIGINAL_SOFTMAX = None
_ORIGINAL_SILU = None
_ORIGINAL_GELU = None
_ORIGINAL_LAYER_NORM = None
_CURRENT_PHASE = 'unknown'  # 'rollout', 'actor_forward', 'actor_backward', 'unknown'

# Selective layer injection (for diagnostic probing)
_SELECTIVE_LAYERS = None  # None = all layers, set() = specific layer IDs only
_CURRENT_LAYER_ID = None  # Current layer being processed (set via hooks)
_LAYER_INJECTION_STATS = {}  # Per-layer injection counts: {layer_id: {'forward': N, 'backward': M}}

# Selective operator injection (for fine-grained diagnostic probing)
_SELECTIVE_OPERATORS = None  # None = all operators, set() = specific operator types only
# Supported operator types: 'matmul', 'bmm', 'linear', 'softmax', 'silu', 'gelu', 'layer_norm'
_INJECTION_COUNT = {
    'rollout_forward': 0,
    'rollout_backward': 0,
    'actor_forward': 0,
    'actor_backward': 0,
    'unknown_forward': 0,
    'unknown_backward': 0,
}
_LOG_FIRST_N = 3  # Log first N injections per phase for debugging
_LOGGED_COUNT = {
    'rollout_forward': 0,
    'rollout_backward': 0,
    'actor_forward': 0,
    'actor_backward': 0,
}
_AUTO_ENABLED = False  # Track if we've done auto-enable from env var


def _auto_enable_from_env():
    """Auto-enable noisy ops if environment variables are set.

    IMPORTANT: Due to incompatibility with torch.compile (used by vLLM),
    auto-enable only works when VERL_NOISY_OPS_SKIP_VLLM is NOT set to "0".
    By default, we skip enabling in processes that might use vLLM.

    For training-only error injection, use enable_noisy_ops() explicitly
    in the trainer code, or set VERL_NOISY_OPS_TRAINING_ONLY=1.

    Environment variables:
        VERL_NOISY_OPS_ENABLED: Set to "1" or "true" to enable
        VERL_NOISY_OPS_SCALE: Error scale (default: 1e-4)
        VERL_NOISY_OPS_TYPE: Error type (default: relative_gaussian)
        VERL_NOISY_OPS_TRAINING_ONLY: Set to "1" to only enable during training
        VERL_NOISY_OPS_ALL_OPS: Set to "1" to enable ALL operators mode
                               (injects noise into softmax, silu, gelu, layer_norm too)
    """
    global _AUTO_ENABLED
    if _AUTO_ENABLED:
        return

    import os

    enabled = os.environ.get('VERL_NOISY_OPS_ENABLED', '').lower()
    if enabled not in ('1', 'true', 'yes'):
        return

    # If TRAINING_ONLY is set, don't auto-enable at import time
    # The trainer will enable/disable noisy ops around training steps
    training_only = os.environ.get('VERL_NOISY_OPS_TRAINING_ONLY', '').lower()
    if training_only in ('1', 'true', 'yes'):
        print("[NoisyOps] TRAINING_ONLY mode: will be enabled by trainer during actor training")
        _AUTO_ENABLED = True
        return

    scale = float(os.environ.get('VERL_NOISY_OPS_SCALE', '1e-4'))
    error_type = os.environ.get('VERL_NOISY_OPS_TYPE', 'relative_gaussian')
    all_ops = os.environ.get('VERL_NOISY_OPS_ALL_OPS', '').lower() in ('1', 'true', 'yes')

    # Check for forward-only or backward-only mode
    forward_only = os.environ.get('VERL_NOISY_OPS_FORWARD_ONLY', '').lower() in ('1', 'true', 'yes')
    backward_only = os.environ.get('VERL_NOISY_OPS_BACKWARD_ONLY', '').lower() in ('1', 'true', 'yes')

    if forward_only and backward_only:
        print("[NoisyOps] WARNING: Both FORWARD_ONLY and BACKWARD_ONLY set, using both=True")
        set_noise_phases(forward=True, backward=True)
    elif forward_only:
        set_noise_phases(forward=True, backward=False)
    elif backward_only:
        set_noise_phases(forward=False, backward=True)
    # else: default is both enabled

    enable_noisy_ops(error_scale=scale, error_type=error_type, all_ops_mode=all_ops)
    _AUTO_ENABLED = True
    print(f"[NoisyOps] Auto-enabled from environment: scale={scale}, type={error_type}, all_ops_mode={all_ops}")


def set_phase(phase: str) -> None:
    """
    Set the current execution phase for logging purposes.

    Args:
        phase: One of 'rollout', 'actor_forward', 'actor_backward', 'unknown'
    """
    global _CURRENT_PHASE
    assert phase in ['rollout', 'actor_forward', 'actor_backward', 'unknown'], \
        f"Invalid phase: {phase}"
    _CURRENT_PHASE = phase


def get_phase() -> str:
    """Get the current execution phase."""
    return _CURRENT_PHASE


def set_noise_phases(forward: bool = True, backward: bool = True) -> None:
    """
    Control which phases have noise injection.

    This allows testing the theory that forward noise (activations) vs
    backward noise (gradients) have different effects on model robustness.

    Args:
        forward: If True, inject noise in forward pass (affects activations)
        backward: If True, inject noise in backward pass (affects gradients)

    Example:
        # Forward-only noise (test activation robustness)
        set_noise_phases(forward=True, backward=False)

        # Backward-only noise (test gradient regularization)
        set_noise_phases(forward=False, backward=True)

        # Both (default, current AQN behavior)
        set_noise_phases(forward=True, backward=True)
    """
    global _NOISY_OPS_FORWARD_ENABLED, _NOISY_OPS_BACKWARD_ENABLED
    with _NOISE_CONFIG_LOCK:
        _NOISY_OPS_FORWARD_ENABLED = forward
        _NOISY_OPS_BACKWARD_ENABLED = backward
    logger.info(f"[NoisyOps] Phases set: forward={forward}, backward={backward}")
    print(f"[NoisyOps] Phases: forward={forward}, backward={backward}")


def get_noise_phases() -> dict:
    """Get current noise phase configuration."""
    return {
        'forward_enabled': _NOISY_OPS_FORWARD_ENABLED,
        'backward_enabled': _NOISY_OPS_BACKWARD_ENABLED,
    }


# ============================================================================
# Selective Layer Injection API (for Diagnostic Probing)
# ============================================================================

def set_selective_layers(layer_ids: list = None) -> None:
    """
    Enable noise injection only for specific layers (for diagnostic probing).

    This allows testing which layers are most sensitive to noise, enabling:
    1. Layer-wise sensitivity profiling
    2. Hardware error localization (which layers cause GPU/NPU divergence)
    3. Adaptive AQN targeting (apply more noise to robust layers)

    Args:
        layer_ids: List of layer indices to inject noise into.
                   None = all layers (default behavior)
                   [] = no layers (effectively disables noise)
                   [0, 5, 10] = only layers 0, 5, and 10

    Example:
        # Test sensitivity of early layers
        set_selective_layers([0, 1, 2, 3, 4])
        accuracy_early = evaluate(model, test_data)

        # Test sensitivity of late layers
        set_selective_layers([24, 25, 26, 27])
        accuracy_late = evaluate(model, test_data)

        # Compare to identify which region is more sensitive
    """
    global _SELECTIVE_LAYERS
    with _NOISE_CONFIG_LOCK:
        if layer_ids is None:
            _SELECTIVE_LAYERS = None
            logger.info("[NoisyOps] Selective layers: ALL (disabled filtering)")
            print("[NoisyOps] Selective layers: ALL")
        else:
            _SELECTIVE_LAYERS = set(layer_ids)
            logger.info(f"[NoisyOps] Selective layers: {sorted(_SELECTIVE_LAYERS)}")
            print(f"[NoisyOps] Selective layers: {sorted(_SELECTIVE_LAYERS)}")


def get_selective_layers() -> set:
    """Get current selective layer configuration."""
    return _SELECTIVE_LAYERS


def set_current_layer(layer_id: int) -> None:
    """
    Set the current layer ID being processed.

    This should be called by layer hooks during forward pass to track
    which layer's operations are currently executing.

    Args:
        layer_id: The transformer layer index (0-based)
    """
    global _CURRENT_LAYER_ID
    _CURRENT_LAYER_ID = layer_id


def get_current_layer() -> int:
    """Get the current layer ID being processed."""
    return _CURRENT_LAYER_ID


def should_inject_for_layer(layer_id: int = None) -> bool:
    """
    Check if noise should be injected for the given layer.

    Args:
        layer_id: Layer index to check. If None, uses _CURRENT_LAYER_ID.

    Returns:
        True if noise should be injected, False otherwise.
    """
    if _SELECTIVE_LAYERS is None:
        return True  # All layers

    if layer_id is None:
        layer_id = _CURRENT_LAYER_ID

    if layer_id is None:
        return True  # No layer context, default to inject

    return layer_id in _SELECTIVE_LAYERS


def _update_layer_stats(layer_id: int, direction: str) -> None:
    """Update per-layer injection statistics."""
    global _LAYER_INJECTION_STATS
    if layer_id is None:
        layer_id = 'unknown'

    if layer_id not in _LAYER_INJECTION_STATS:
        _LAYER_INJECTION_STATS[layer_id] = {'forward': 0, 'backward': 0}

    _LAYER_INJECTION_STATS[layer_id][direction] += 1


def get_layer_injection_stats() -> dict:
    """
    Get per-layer injection statistics.

    Returns:
        Dict mapping layer_id to {'forward': count, 'backward': count}
    """
    return _LAYER_INJECTION_STATS.copy()


def reset_layer_injection_stats() -> None:
    """Reset per-layer injection statistics."""
    global _LAYER_INJECTION_STATS
    _LAYER_INJECTION_STATS = {}


# ============================================================================
# Selective Operator Injection API (for Fine-Grained Diagnostic Probing)
# ============================================================================

_VALID_OPERATOR_TYPES = {'matmul', 'bmm', 'linear', 'softmax', 'silu', 'gelu', 'layer_norm'}


def set_selective_operators(op_types: list = None) -> None:
    """
    Enable noise injection only for specific operator types (for diagnostic probing).

    This allows testing which operators are most sensitive to noise, enabling:
    1. Operator-wise sensitivity profiling
    2. Hardware error localization (which ops cause GPU/NPU divergence)
    3. Fine-grained AQN targeting (apply noise to specific ops)

    Args:
        op_types: List of operator types to inject noise into.
                  None = all operators (default behavior)
                  [] = no operators (effectively disables noise)
                  ['matmul', 'softmax'] = only matmul and softmax ops

    Valid operator types:
        'matmul', 'bmm', 'linear', 'softmax', 'silu', 'gelu', 'layer_norm'

    Example:
        # Test sensitivity of attention (softmax) vs MLP (silu/gelu)
        set_selective_operators(['softmax'])
        accuracy_attention = evaluate(model, test_data)

        set_selective_operators(['silu', 'gelu'])
        accuracy_mlp = evaluate(model, test_data)

        # Combine with layer selection for fine-grained diagnosis
        set_selective_layers([10])
        set_selective_operators(['softmax'])
        # Now noise only affects softmax in layer 10
    """
    global _SELECTIVE_OPERATORS
    with _NOISE_CONFIG_LOCK:
        if op_types is None:
            _SELECTIVE_OPERATORS = None
            logger.info("[NoisyOps] Selective operators: ALL (disabled filtering)")
            print("[NoisyOps] Selective operators: ALL")
        else:
            # Validate operator types
            op_set = set(op_types)
            invalid_ops = op_set - _VALID_OPERATOR_TYPES
            if invalid_ops:
                raise ValueError(f"Invalid operator types: {invalid_ops}. "
                               f"Valid types: {_VALID_OPERATOR_TYPES}")
            _SELECTIVE_OPERATORS = op_set
            logger.info(f"[NoisyOps] Selective operators: {sorted(_SELECTIVE_OPERATORS)}")
            print(f"[NoisyOps] Selective operators: {sorted(_SELECTIVE_OPERATORS)}")


def get_selective_operators() -> set:
    """Get current selective operator configuration."""
    return _SELECTIVE_OPERATORS


def should_inject_for_operator(op_type: str) -> bool:
    """
    Check if noise should be injected for the given operator type.

    Args:
        op_type: Operator type to check (e.g., 'matmul', 'softmax')

    Returns:
        True if noise should be injected, False otherwise.
    """
    if _SELECTIVE_OPERATORS is None:
        return True  # All operators
    return op_type in _SELECTIVE_OPERATORS


def register_layer_hooks(model, layer_pattern: str = '.layers.') -> int:
    """
    Register forward hooks to track current layer during forward pass.

    This enables layer-aware noise injection for diagnostic probing.

    Args:
        model: The PyTorch model (typically a transformer)
        layer_pattern: Pattern to identify layer modules (default: '.layers.')

    Returns:
        Number of hooks registered

    Example:
        model = AutoModelForCausalLM.from_pretrained(...)
        num_hooks = register_layer_hooks(model)
        print(f"Registered {num_hooks} layer hooks")

        # Now noise injection will be layer-aware
        set_selective_layers([0, 10, 20])
        output = model(input_ids)  # Only layers 0, 10, 20 get noise
    """
    import re

    hooks_registered = 0

    def make_hook(lid):
        def hook(module, input):
            set_current_layer(lid)
        return hook

    for name, module in model.named_modules():
        if layer_pattern in name:
            # Extract layer ID: 'model.layers.12.mlp' → 12
            match = re.search(r'\.layers\.(\d+)', name)
            if match:
                layer_id = int(match.group(1))
                # Only register for top-level layer module (not sub-modules)
                if name.count('.layers.') == 1 and name.endswith(f'.layers.{layer_id}'):
                    module.register_forward_pre_hook(make_hook(layer_id))
                    hooks_registered += 1

    if hooks_registered > 0:
        print(f"[NoisyOps] Registered {hooks_registered} layer hooks for diagnostic probing")

    return hooks_registered


def _log_injection(phase: str, direction: str, shape: tuple, error_mean: float, error_max: float) -> None:
    """Log injection event if within first N for this phase."""
    global _LOGGED_COUNT
    key = f"{phase}_{direction}"
    if key in _LOGGED_COUNT and _LOGGED_COUNT[key] < _LOG_FIRST_N:
        _LOGGED_COUNT[key] += 1
        print(f"[NoisyOps] {direction.upper()} injection #{_LOGGED_COUNT[key]} in {phase}: "
              f"shape={shape}, mean_error={error_mean:.2e}, max_error={error_max:.2e}")


def _compute_error(tensor: torch.Tensor, error_type: str = None, scale: float = None) -> torch.Tensor:
    """Compute error tensor based on error type."""
    if error_type is None:
        error_type = _ERROR_TYPE
    if scale is None:
        scale = _ERROR_SCALE

    if error_type == 'relative_gaussian':
        # Error proportional to value magnitude (most realistic HW error model)
        noise = torch.randn_like(tensor)
        return noise * tensor.abs() * scale
    elif error_type == 'absolute_gaussian':
        # Constant magnitude error
        return torch.randn_like(tensor) * scale
    else:
        raise ValueError(f"Unknown error type: {error_type}")


class NoisyMatMul(torch.autograd.Function):
    """
    Noisy matrix multiplication with error injection in forward AND backward.

    Forward: result = matmul(a, b) + error
    Backward: grad_a = matmul(grad, b.T) + error
              grad_b = matmul(a.T, grad) + error
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        global _INJECTION_COUNT
        # Use original matmul to avoid recursion
        result = _ORIGINAL_MATMUL(a, b)

        if (_NOISY_OPS_ENABLED and _NOISY_OPS_FORWARD_ENABLED
                and should_inject_for_layer() and should_inject_for_operator('matmul')):
            error = _compute_error(result)
            result = result + error

            # Track and log by phase
            phase = _CURRENT_PHASE
            key = f"{phase}_forward"
            _INJECTION_COUNT[key] = _INJECTION_COUNT.get(key, 0) + 1

            # Track per-layer stats
            _update_layer_stats(_CURRENT_LAYER_ID, 'forward')

            # Log first few injections per phase
            _log_injection(phase, 'forward', tuple(result.shape),
                          error.abs().mean().item(), error.abs().max().item())

        ctx.save_for_backward(a, b)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        global _INJECTION_COUNT
        a, b = ctx.saved_tensors

        # Compute gradients using original matmul
        grad_a = _ORIGINAL_MATMUL(grad_output, b.transpose(-1, -2))
        grad_b = _ORIGINAL_MATMUL(a.transpose(-1, -2), grad_output)

        if (_NOISY_OPS_ENABLED and _NOISY_OPS_BACKWARD_ENABLED
                and should_inject_for_layer() and should_inject_for_operator('matmul')):
            # Inject error into gradients too
            error_a = _compute_error(grad_a)
            error_b = _compute_error(grad_b)
            grad_a = grad_a + error_a
            grad_b = grad_b + error_b

            # Track and log by phase
            phase = _CURRENT_PHASE
            key = f"{phase}_backward"
            _INJECTION_COUNT[key] = _INJECTION_COUNT.get(key, 0) + 1

            # Track per-layer stats
            _update_layer_stats(_CURRENT_LAYER_ID, 'backward')

            # Log first few injections per phase
            _log_injection(phase, 'backward', tuple(grad_a.shape),
                          error_a.abs().mean().item(), error_a.abs().max().item())

        return grad_a, grad_b


class NoisyBMM(torch.autograd.Function):
    """Noisy batch matrix multiplication."""

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        global _INJECTION_COUNT
        result = _ORIGINAL_BMM(a, b)

        if (_NOISY_OPS_ENABLED and _NOISY_OPS_FORWARD_ENABLED
                and should_inject_for_layer() and should_inject_for_operator('bmm')):
            error = _compute_error(result)
            result = result + error

            phase = _CURRENT_PHASE
            key = f"{phase}_forward"
            _INJECTION_COUNT[key] = _INJECTION_COUNT.get(key, 0) + 1

            # Track per-layer stats
            _update_layer_stats(_CURRENT_LAYER_ID, 'forward')

        ctx.save_for_backward(a, b)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        global _INJECTION_COUNT
        a, b = ctx.saved_tensors

        grad_a = _ORIGINAL_BMM(grad_output, b.transpose(-1, -2))
        grad_b = _ORIGINAL_BMM(a.transpose(-1, -2), grad_output)

        if (_NOISY_OPS_ENABLED and _NOISY_OPS_BACKWARD_ENABLED
                and should_inject_for_layer() and should_inject_for_operator('bmm')):
            error_a = _compute_error(grad_a)
            error_b = _compute_error(grad_b)
            grad_a = grad_a + error_a
            grad_b = grad_b + error_b

            phase = _CURRENT_PHASE
            key = f"{phase}_backward"
            _INJECTION_COUNT[key] = _INJECTION_COUNT.get(key, 0) + 1

            # Track per-layer stats
            _update_layer_stats(_CURRENT_LAYER_ID, 'backward')

        return grad_a, grad_b


class NoisySoftmax(torch.autograd.Function):
    """Noisy softmax with error injection."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, dim: int) -> torch.Tensor:
        global _INJECTION_COUNT
        result = _ORIGINAL_SOFTMAX(input, dim=dim)

        if (_NOISY_OPS_ENABLED and _ALL_OPS_MODE and _NOISY_OPS_FORWARD_ENABLED
                and should_inject_for_layer() and should_inject_for_operator('softmax')):
            error = _compute_error(result)
            result = result + error
            # Re-normalize to keep it a valid probability distribution
            result = result.clamp(min=0)
            result = result / result.sum(dim=dim, keepdim=True).clamp(min=1e-12)

            phase = _CURRENT_PHASE
            key = f"{phase}_forward"
            _INJECTION_COUNT[key] = _INJECTION_COUNT.get(key, 0) + 1

            # Track per-layer stats
            _update_layer_stats(_CURRENT_LAYER_ID, 'forward')

        ctx.save_for_backward(result)
        ctx.dim = dim
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        global _INJECTION_COUNT
        result, = ctx.saved_tensors
        dim = ctx.dim

        # Softmax backward: grad_input = softmax * (grad_output - sum(grad_output * softmax))
        sum_term = (grad_output * result).sum(dim=dim, keepdim=True)
        grad_input = result * (grad_output - sum_term)

        if (_NOISY_OPS_ENABLED and _ALL_OPS_MODE and _NOISY_OPS_BACKWARD_ENABLED
                and should_inject_for_layer() and should_inject_for_operator('softmax')):
            error = _compute_error(grad_input)
            grad_input = grad_input + error

            phase = _CURRENT_PHASE
            key = f"{phase}_backward"
            _INJECTION_COUNT[key] = _INJECTION_COUNT.get(key, 0) + 1

            # Track per-layer stats
            _update_layer_stats(_CURRENT_LAYER_ID, 'backward')

        return grad_input, None


class NoisySiLU(torch.autograd.Function):
    """Noisy SiLU (Swish) activation with error injection."""

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        global _INJECTION_COUNT
        result = _ORIGINAL_SILU(input)

        if (_NOISY_OPS_ENABLED and _ALL_OPS_MODE and _NOISY_OPS_FORWARD_ENABLED
                and should_inject_for_layer() and should_inject_for_operator('silu')):
            error = _compute_error(result)
            result = result + error

            phase = _CURRENT_PHASE
            key = f"{phase}_forward"
            _INJECTION_COUNT[key] = _INJECTION_COUNT.get(key, 0) + 1

            # Track per-layer stats
            _update_layer_stats(_CURRENT_LAYER_ID, 'forward')

        ctx.save_for_backward(input)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        global _INJECTION_COUNT
        input, = ctx.saved_tensors

        # SiLU backward: grad_input = sigmoid(x) * (1 + x * (1 - sigmoid(x))) * grad_output
        sigmoid_x = torch.sigmoid(input)
        grad_input = grad_output * sigmoid_x * (1 + input * (1 - sigmoid_x))

        if (_NOISY_OPS_ENABLED and _ALL_OPS_MODE and _NOISY_OPS_BACKWARD_ENABLED
                and should_inject_for_layer() and should_inject_for_operator('silu')):
            error = _compute_error(grad_input)
            grad_input = grad_input + error

            phase = _CURRENT_PHASE
            key = f"{phase}_backward"
            _INJECTION_COUNT[key] = _INJECTION_COUNT.get(key, 0) + 1

            # Track per-layer stats
            _update_layer_stats(_CURRENT_LAYER_ID, 'backward')

        return grad_input


class NoisyGeLU(torch.autograd.Function):
    """Noisy GELU activation with error injection."""

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        global _INJECTION_COUNT
        result = _ORIGINAL_GELU(input)

        if (_NOISY_OPS_ENABLED and _ALL_OPS_MODE and _NOISY_OPS_FORWARD_ENABLED
                and should_inject_for_layer() and should_inject_for_operator('gelu')):
            error = _compute_error(result)
            result = result + error

            phase = _CURRENT_PHASE
            key = f"{phase}_forward"
            _INJECTION_COUNT[key] = _INJECTION_COUNT.get(key, 0) + 1

            # Track per-layer stats
            _update_layer_stats(_CURRENT_LAYER_ID, 'forward')

        ctx.save_for_backward(input)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        global _INJECTION_COUNT
        input, = ctx.saved_tensors

        # GELU backward approximation
        # Using the exact formula: gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = 0.7978845608028654
        coeff = 0.044715
        inner = sqrt_2_over_pi * (input + coeff * input.pow(3))
        tanh_inner = torch.tanh(inner)
        sech2_inner = 1 - tanh_inner.pow(2)
        inner_deriv = sqrt_2_over_pi * (1 + 3 * coeff * input.pow(2))
        grad_input = grad_output * 0.5 * (1 + tanh_inner + input * sech2_inner * inner_deriv)

        if (_NOISY_OPS_ENABLED and _ALL_OPS_MODE and _NOISY_OPS_BACKWARD_ENABLED
                and should_inject_for_layer() and should_inject_for_operator('gelu')):
            error = _compute_error(grad_input)
            grad_input = grad_input + error

            phase = _CURRENT_PHASE
            key = f"{phase}_backward"
            _INJECTION_COUNT[key] = _INJECTION_COUNT.get(key, 0) + 1

            # Track per-layer stats
            _update_layer_stats(_CURRENT_LAYER_ID, 'backward')

        return grad_input


class NoisyLayerNorm(torch.autograd.Function):
    """Noisy layer normalization with error injection."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, normalized_shape, weight, bias, eps) -> torch.Tensor:
        global _INJECTION_COUNT
        result = _ORIGINAL_LAYER_NORM(input, normalized_shape, weight, bias, eps)

        if (_NOISY_OPS_ENABLED and _ALL_OPS_MODE and _NOISY_OPS_FORWARD_ENABLED
                and should_inject_for_layer() and should_inject_for_operator('layer_norm')):
            error = _compute_error(result)
            result = result + error

            phase = _CURRENT_PHASE
            key = f"{phase}_forward"
            _INJECTION_COUNT[key] = _INJECTION_COUNT.get(key, 0) + 1

            # Track per-layer stats
            _update_layer_stats(_CURRENT_LAYER_ID, 'forward')

        ctx.save_for_backward(input, weight)
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        global _INJECTION_COUNT
        input, weight = ctx.saved_tensors
        normalized_shape = ctx.normalized_shape
        eps = ctx.eps

        # Use PyTorch's native backward for correctness
        # Just inject noise into the final gradients
        input.requires_grad_(True)
        with torch.enable_grad():
            output = _ORIGINAL_LAYER_NORM(input, normalized_shape, weight, None, eps)
            grad_input, = torch.autograd.grad(output, input, grad_output, retain_graph=False)

        if (_NOISY_OPS_ENABLED and _ALL_OPS_MODE and _NOISY_OPS_BACKWARD_ENABLED
                and should_inject_for_layer() and should_inject_for_operator('layer_norm')):
            error = _compute_error(grad_input)
            grad_input = grad_input + error

            phase = _CURRENT_PHASE
            key = f"{phase}_backward"
            _INJECTION_COUNT[key] = _INJECTION_COUNT.get(key, 0) + 1

            # Track per-layer stats
            _update_layer_stats(_CURRENT_LAYER_ID, 'backward')

        # Gradient for weight and bias
        grad_weight = (grad_output * ((input - input.mean(dim=-1, keepdim=True)) /
                       (input.var(dim=-1, keepdim=True, unbiased=False) + eps).sqrt())).sum(dim=tuple(range(len(input.shape) - len(normalized_shape))))
        grad_bias = grad_output.sum(dim=tuple(range(len(input.shape) - len(normalized_shape)))) if True else None

        return grad_input, None, grad_weight, grad_bias, None


# Note: We use a simple flag check without torch.compile detection.
# vLLM and other torch.compile users will skip these noisy ops entirely
# because they run in separate processes where noisy ops isn't enabled,
# or they use torch.compile which forces fallback to original ops via graph break.


def _noisy_matmul_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Implementation of noisy matmul - called only when noisy ops is enabled."""
    return NoisyMatMul.apply(a, b)


def _noisy_bmm_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Implementation of noisy bmm - called only when noisy ops is enabled."""
    return NoisyBMM.apply(a, b)


def _noisy_linear_impl(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Implementation of noisy linear - called only when noisy ops is enabled.

    Note: Linear uses matmul internally. We check for 'linear' operator type here,
    but the actual noise injection happens in NoisyMatMul which checks for 'matmul'.
    To inject noise specifically for linear ops (not standalone matmuls), use:
        set_selective_operators(['linear'])  # This is treated as ['matmul'] for injection
    """
    # If 'linear' is in selective operators but 'matmul' is not, we need special handling
    # For now, linear is implemented via matmul, so they share the same noise injection
    if _SELECTIVE_OPERATORS is not None and 'linear' in _SELECTIVE_OPERATORS:
        # Temporarily allow matmul for this linear operation
        old_ops = _SELECTIVE_OPERATORS.copy()
        _SELECTIVE_OPERATORS.add('matmul')
        try:
            output = NoisyMatMul.apply(input, weight.t())
        finally:
            _SELECTIVE_OPERATORS.clear()
            _SELECTIVE_OPERATORS.update(old_ops)
    else:
        output = NoisyMatMul.apply(input, weight.t())
    if bias is not None:
        output = output + bias
    return output


def _noisy_softmax_impl(input: torch.Tensor, dim: int = -1, dtype=None) -> torch.Tensor:
    """Implementation of noisy softmax."""
    if dtype is not None:
        input = input.to(dtype)
    return NoisySoftmax.apply(input, dim)


def _noisy_silu_impl(input: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """Implementation of noisy SiLU."""
    return NoisySiLU.apply(input)


def _noisy_gelu_impl(input: torch.Tensor, approximate: str = 'none') -> torch.Tensor:
    """Implementation of noisy GELU."""
    # Note: we ignore approximate parameter for simplicity
    return NoisyGeLU.apply(input)


def _noisy_layer_norm_impl(input: torch.Tensor, normalized_shape, weight=None, bias=None, eps=1e-5) -> torch.Tensor:
    """Implementation of noisy layer norm."""
    return NoisyLayerNorm.apply(input, normalized_shape, weight, bias, eps)


# Mark implementations as incompatible with torch.compile
# This forces torch.compile to use the original ops when it encounters these
try:
    if hasattr(torch, '_dynamo') and hasattr(torch._dynamo, 'disable'):
        _noisy_matmul_impl = torch._dynamo.disable(_noisy_matmul_impl, recursive=False)
        _noisy_bmm_impl = torch._dynamo.disable(_noisy_bmm_impl, recursive=False)
        _noisy_linear_impl = torch._dynamo.disable(_noisy_linear_impl, recursive=False)
        _noisy_softmax_impl = torch._dynamo.disable(_noisy_softmax_impl, recursive=False)
        _noisy_silu_impl = torch._dynamo.disable(_noisy_silu_impl, recursive=False)
        _noisy_gelu_impl = torch._dynamo.disable(_noisy_gelu_impl, recursive=False)
        _noisy_layer_norm_impl = torch._dynamo.disable(_noisy_layer_norm_impl, recursive=False)
except Exception:
    pass


def noisy_matmul(a: torch.Tensor, b: torch.Tensor, **kwargs) -> torch.Tensor:
    """Drop-in replacement for torch.matmul with error injection."""
    if not _NOISY_OPS_ENABLED:
        return _ORIGINAL_MATMUL(a, b)
    # Call implementation which handles torch.compile via @dynamo.disable
    return _noisy_matmul_impl(a, b)


def noisy_bmm(a: torch.Tensor, b: torch.Tensor, **kwargs) -> torch.Tensor:
    """Drop-in replacement for torch.bmm with error injection."""
    if not _NOISY_OPS_ENABLED:
        return _ORIGINAL_BMM(a, b)
    return _noisy_bmm_impl(a, b)


def noisy_linear(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
    """
    Drop-in replacement for F.linear with error injection.

    F.linear computes: output = input @ weight.T + bias
    We inject error into the matmul result.
    """
    if not _NOISY_OPS_ENABLED:
        return _ORIGINAL_LINEAR(input, weight, bias)
    return _noisy_linear_impl(input, weight, bias)


def noisy_softmax(input: torch.Tensor, dim: int = -1, dtype=None, **kwargs) -> torch.Tensor:
    """Drop-in replacement for F.softmax with error injection (only in ALL_OPS_MODE).

    Note: **kwargs is used to absorb extra arguments from torch.compile (e.g., _stacklevel).
    """
    if not _NOISY_OPS_ENABLED or not _ALL_OPS_MODE:
        return _ORIGINAL_SOFTMAX(input, dim=dim, dtype=dtype)
    return _noisy_softmax_impl(input, dim, dtype)


def noisy_silu(input: torch.Tensor, inplace: bool = False, **kwargs) -> torch.Tensor:
    """Drop-in replacement for F.silu with error injection (only in ALL_OPS_MODE)."""
    if not _NOISY_OPS_ENABLED or not _ALL_OPS_MODE:
        return _ORIGINAL_SILU(input, inplace=inplace)
    return _noisy_silu_impl(input, inplace)


def noisy_gelu(input: torch.Tensor, approximate: str = 'none', **kwargs) -> torch.Tensor:
    """Drop-in replacement for F.gelu with error injection (only in ALL_OPS_MODE)."""
    if not _NOISY_OPS_ENABLED or not _ALL_OPS_MODE:
        return _ORIGINAL_GELU(input, approximate=approximate)
    return _noisy_gelu_impl(input, approximate)


def noisy_layer_norm(input: torch.Tensor, normalized_shape, weight=None, bias=None, eps=1e-5, **kwargs) -> torch.Tensor:
    """Drop-in replacement for F.layer_norm with error injection (only in ALL_OPS_MODE)."""
    if not _NOISY_OPS_ENABLED or not _ALL_OPS_MODE:
        return _ORIGINAL_LAYER_NORM(input, normalized_shape, weight, bias, eps)
    return _noisy_layer_norm_impl(input, normalized_shape, weight, bias, eps)


def enable_noisy_ops(
    error_scale: float = 1e-5,
    error_type: str = 'relative_gaussian',
    all_ops_mode: bool = False,
) -> None:
    """
    Enable operator-level error injection globally.

    This monkey-patches torch.matmul, torch.bmm, and F.linear to inject
    errors in both forward and backward passes.

    Args:
        error_scale: Scale of error relative to tensor magnitude
        error_type: 'relative_gaussian' or 'absolute_gaussian'
        all_ops_mode: If True, also inject errors into softmax, silu, gelu, layer_norm
                      (simulating broader quantization-like errors)
    """
    global _NOISY_OPS_ENABLED, _ERROR_SCALE, _ERROR_TYPE, _ALL_OPS_MODE
    global _ORIGINAL_MATMUL, _ORIGINAL_LINEAR, _ORIGINAL_BMM, _INJECTION_COUNT
    global _ORIGINAL_SOFTMAX, _ORIGINAL_SILU, _ORIGINAL_GELU, _ORIGINAL_LAYER_NORM

    if _NOISY_OPS_ENABLED:
        logger.warning("Noisy ops already enabled, updating config")

    # Save originals (only once)
    if _ORIGINAL_MATMUL is None:
        _ORIGINAL_MATMUL = torch.matmul
        _ORIGINAL_LINEAR = F.linear
        _ORIGINAL_BMM = torch.bmm
    if _ORIGINAL_SOFTMAX is None:
        _ORIGINAL_SOFTMAX = F.softmax
        _ORIGINAL_SILU = F.silu
        _ORIGINAL_GELU = F.gelu
        _ORIGINAL_LAYER_NORM = F.layer_norm

    # Set config
    _ERROR_SCALE = error_scale
    _ERROR_TYPE = error_type
    _ALL_OPS_MODE = all_ops_mode
    _NOISY_OPS_ENABLED = True
    _INJECTION_COUNT = {'forward': 0, 'backward': 0}

    # Monkey-patch core ops (always)
    torch.matmul = noisy_matmul
    torch.bmm = noisy_bmm
    F.linear = noisy_linear

    # Monkey-patch additional ops (only in all_ops_mode)
    if all_ops_mode:
        F.softmax = noisy_softmax
        F.silu = noisy_silu
        F.gelu = noisy_gelu
        F.layer_norm = noisy_layer_norm

    ops_list = "matmul, bmm, linear"
    if all_ops_mode:
        ops_list += ", softmax, silu, gelu, layer_norm"

    logger.info(f"[NoisyOps] Enabled operator-level error injection: "
                f"scale={error_scale}, type={error_type}, all_ops_mode={all_ops_mode}")
    print(f"[NoisyOps] Enabled: scale={error_scale}, type={error_type}, "
          f"all_ops_mode={all_ops_mode}, affects: {ops_list}")


def disable_noisy_ops() -> dict:
    """
    Disable operator-level error injection and restore original ops.

    Returns:
        Dictionary with injection statistics
    """
    global _NOISY_OPS_ENABLED, _INJECTION_COUNT, _ALL_OPS_MODE

    if not _NOISY_OPS_ENABLED:
        logger.warning("Noisy ops not enabled")
        return _INJECTION_COUNT.copy()

    # Restore originals (core ops)
    if _ORIGINAL_MATMUL is not None:
        torch.matmul = _ORIGINAL_MATMUL
        torch.bmm = _ORIGINAL_BMM
        F.linear = _ORIGINAL_LINEAR

    # Restore originals (additional ops from all_ops_mode)
    if _ORIGINAL_SOFTMAX is not None:
        F.softmax = _ORIGINAL_SOFTMAX
        F.silu = _ORIGINAL_SILU
        F.gelu = _ORIGINAL_GELU
        F.layer_norm = _ORIGINAL_LAYER_NORM

    _NOISY_OPS_ENABLED = False
    _ALL_OPS_MODE = False

    stats = _INJECTION_COUNT.copy()
    # Sum all forward/backward injections across phases
    forward_total = sum(v for k, v in stats.items() if 'forward' in k)
    backward_total = sum(v for k, v in stats.items() if 'backward' in k)
    logger.info(f"[NoisyOps] Disabled. Stats: {stats}")
    print(f"[NoisyOps] Disabled. Forward injections: {forward_total}, "
          f"Backward injections: {backward_total}")

    return stats


@contextmanager
def noisy_ops_context(
    error_scale: float = 1e-5,
    error_type: str = 'relative_gaussian',
    all_ops_mode: bool = False,
):
    """
    Context manager for enabling noisy ops temporarily.

    Example:
        with noisy_ops_context(error_scale=1e-4):
            output = model(input)
            loss = criterion(output, target)
            loss.backward()

        # With all_ops_mode (injects noise into more operators):
        with noisy_ops_context(error_scale=1e-3, all_ops_mode=True):
            output = model(input)
    """
    enable_noisy_ops(error_scale=error_scale, error_type=error_type, all_ops_mode=all_ops_mode)
    try:
        yield
    finally:
        disable_noisy_ops()


def get_injection_stats() -> dict:
    """Get current injection statistics by phase."""
    counts = _INJECTION_COUNT.copy()
    forward_total = sum(v for k, v in counts.items() if 'forward' in k)
    backward_total = sum(v for k, v in counts.items() if 'backward' in k)
    return {
        'enabled': _NOISY_OPS_ENABLED,
        'error_scale': _ERROR_SCALE,
        'error_type': _ERROR_TYPE,
        'current_phase': _CURRENT_PHASE,
        'forward_enabled': _NOISY_OPS_FORWARD_ENABLED,
        'backward_enabled': _NOISY_OPS_BACKWARD_ENABLED,
        'counts': counts,
        'total_forward': forward_total,
        'total_backward': backward_total,
    }


def print_injection_summary() -> None:
    """Print a summary of injection counts by phase."""
    if not any(_INJECTION_COUNT.values()):
        print("[NoisyOps] No injections recorded yet")
        return

    print(f"\n[NoisyOps] Injection Summary (scale={_ERROR_SCALE}):")
    print("-" * 60)
    for phase in ['rollout', 'actor', 'unknown']:
        fwd_key = f"{phase}_forward"
        bwd_key = f"{phase}_backward"
        fwd = _INJECTION_COUNT.get(fwd_key, 0)
        bwd = _INJECTION_COUNT.get(bwd_key, 0)
        if fwd > 0 or bwd > 0:
            print(f"  {phase:15s}: forward={fwd:,}, backward={bwd:,}")
    print("-" * 60)


def reset_injection_stats() -> None:
    """Reset injection counters and logged counts."""
    global _INJECTION_COUNT, _LOGGED_COUNT
    _INJECTION_COUNT = {
        'rollout_forward': 0,
        'rollout_backward': 0,
        'actor_forward': 0,
        'actor_backward': 0,
        'unknown_forward': 0,
        'unknown_backward': 0,
    }
    _LOGGED_COUNT = {
        'rollout_forward': 0,
        'rollout_backward': 0,
        'actor_forward': 0,
        'actor_backward': 0,
    }


__all__ = [
    # Core API
    'enable_noisy_ops',
    'disable_noisy_ops',
    'noisy_ops_context',
    # Phase control
    'set_phase',
    'get_phase',
    'set_noise_phases',
    'get_noise_phases',
    # Selective layer injection (diagnostic probing)
    'set_selective_layers',
    'get_selective_layers',
    'set_current_layer',
    'get_current_layer',
    'should_inject_for_layer',
    'register_layer_hooks',
    'get_layer_injection_stats',
    'reset_layer_injection_stats',
    # Selective operator injection (fine-grained diagnostic probing)
    'set_selective_operators',
    'get_selective_operators',
    'should_inject_for_operator',
    # Statistics
    'get_injection_stats',
    'print_injection_summary',
    'reset_injection_stats',
    # Noisy operator classes
    'NoisyMatMul',
    'NoisyBMM',
    'NoisySoftmax',
    'NoisySiLU',
    'NoisyGeLU',
    'NoisyLayerNorm',
    # Noisy operator functions
    'noisy_matmul',
    'noisy_bmm',
    'noisy_linear',
    'noisy_softmax',
    'noisy_silu',
    'noisy_gelu',
    'noisy_layer_norm',
    # Internal
    '_auto_enable_from_env',
]

# Auto-enable from environment at import time
# This ensures noisy ops is enabled in Ray workers when env vars are set
_auto_enable_from_env()
