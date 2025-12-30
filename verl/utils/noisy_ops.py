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

logger = logging.getLogger(__name__)

# Global state
_NOISY_OPS_ENABLED = False
_ERROR_SCALE = 1e-5
_ERROR_TYPE = 'relative_gaussian'  # 'relative_gaussian', 'absolute_gaussian'
_ORIGINAL_MATMUL = None
_ORIGINAL_LINEAR = None
_ORIGINAL_BMM = None
_INJECTION_COUNT = {'forward': 0, 'backward': 0}


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

        if _NOISY_OPS_ENABLED:
            error = _compute_error(result)
            result = result + error
            _INJECTION_COUNT['forward'] += 1

        ctx.save_for_backward(a, b)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        global _INJECTION_COUNT
        a, b = ctx.saved_tensors

        # Compute gradients using original matmul
        grad_a = _ORIGINAL_MATMUL(grad_output, b.transpose(-1, -2))
        grad_b = _ORIGINAL_MATMUL(a.transpose(-1, -2), grad_output)

        if _NOISY_OPS_ENABLED:
            # Inject error into gradients too
            grad_a = grad_a + _compute_error(grad_a)
            grad_b = grad_b + _compute_error(grad_b)
            _INJECTION_COUNT['backward'] += 1

        return grad_a, grad_b


class NoisyBMM(torch.autograd.Function):
    """Noisy batch matrix multiplication."""

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        global _INJECTION_COUNT
        result = _ORIGINAL_BMM(a, b)

        if _NOISY_OPS_ENABLED:
            error = _compute_error(result)
            result = result + error
            _INJECTION_COUNT['forward'] += 1

        ctx.save_for_backward(a, b)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        global _INJECTION_COUNT
        a, b = ctx.saved_tensors

        grad_a = _ORIGINAL_BMM(grad_output, b.transpose(-1, -2))
        grad_b = _ORIGINAL_BMM(a.transpose(-1, -2), grad_output)

        if _NOISY_OPS_ENABLED:
            grad_a = grad_a + _compute_error(grad_a)
            grad_b = grad_b + _compute_error(grad_b)
            _INJECTION_COUNT['backward'] += 1

        return grad_a, grad_b


def noisy_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for torch.matmul with error injection."""
    if not _NOISY_OPS_ENABLED:
        return _ORIGINAL_MATMUL(a, b)
    return NoisyMatMul.apply(a, b)


def noisy_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for torch.bmm with error injection."""
    if not _NOISY_OPS_ENABLED:
        return _ORIGINAL_BMM(a, b)
    return NoisyBMM.apply(a, b)


def noisy_linear(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Drop-in replacement for F.linear with error injection.

    F.linear computes: output = input @ weight.T + bias
    We inject error into the matmul result.
    """
    if not _NOISY_OPS_ENABLED:
        return _ORIGINAL_LINEAR(input, weight, bias)

    # Compute matmul with noise
    output = NoisyMatMul.apply(input, weight.t())

    if bias is not None:
        output = output + bias

    return output


def enable_noisy_ops(
    error_scale: float = 1e-5,
    error_type: str = 'relative_gaussian',
) -> None:
    """
    Enable operator-level error injection globally.

    This monkey-patches torch.matmul, torch.bmm, and F.linear to inject
    errors in both forward and backward passes.

    Args:
        error_scale: Scale of error relative to tensor magnitude
        error_type: 'relative_gaussian' or 'absolute_gaussian'
    """
    global _NOISY_OPS_ENABLED, _ERROR_SCALE, _ERROR_TYPE
    global _ORIGINAL_MATMUL, _ORIGINAL_LINEAR, _ORIGINAL_BMM, _INJECTION_COUNT

    if _NOISY_OPS_ENABLED:
        logger.warning("Noisy ops already enabled, updating config")

    # Save originals (only once)
    if _ORIGINAL_MATMUL is None:
        _ORIGINAL_MATMUL = torch.matmul
        _ORIGINAL_LINEAR = F.linear
        _ORIGINAL_BMM = torch.bmm

    # Set config
    _ERROR_SCALE = error_scale
    _ERROR_TYPE = error_type
    _NOISY_OPS_ENABLED = True
    _INJECTION_COUNT = {'forward': 0, 'backward': 0}

    # Monkey-patch
    torch.matmul = noisy_matmul
    torch.bmm = noisy_bmm
    F.linear = noisy_linear

    logger.info(f"[NoisyOps] Enabled operator-level error injection: "
                f"scale={error_scale}, type={error_type}")
    print(f"[NoisyOps] Enabled: scale={error_scale}, type={error_type}, "
          f"affects forward+backward passes globally")


def disable_noisy_ops() -> dict:
    """
    Disable operator-level error injection and restore original ops.

    Returns:
        Dictionary with injection statistics
    """
    global _NOISY_OPS_ENABLED, _INJECTION_COUNT

    if not _NOISY_OPS_ENABLED:
        logger.warning("Noisy ops not enabled")
        return _INJECTION_COUNT.copy()

    # Restore originals
    if _ORIGINAL_MATMUL is not None:
        torch.matmul = _ORIGINAL_MATMUL
        torch.bmm = _ORIGINAL_BMM
        F.linear = _ORIGINAL_LINEAR

    _NOISY_OPS_ENABLED = False

    stats = _INJECTION_COUNT.copy()
    logger.info(f"[NoisyOps] Disabled. Stats: {stats}")
    print(f"[NoisyOps] Disabled. Forward injections: {stats['forward']}, "
          f"Backward injections: {stats['backward']}")

    return stats


@contextmanager
def noisy_ops_context(
    error_scale: float = 1e-5,
    error_type: str = 'relative_gaussian',
):
    """
    Context manager for enabling noisy ops temporarily.

    Example:
        with noisy_ops_context(error_scale=1e-4):
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
    """
    enable_noisy_ops(error_scale=error_scale, error_type=error_type)
    try:
        yield
    finally:
        disable_noisy_ops()


def get_injection_stats() -> dict:
    """Get current injection statistics."""
    return {
        'enabled': _NOISY_OPS_ENABLED,
        'error_scale': _ERROR_SCALE,
        'error_type': _ERROR_TYPE,
        'forward_injections': _INJECTION_COUNT['forward'],
        'backward_injections': _INJECTION_COUNT['backward'],
    }


def reset_injection_stats() -> None:
    """Reset injection counters."""
    global _INJECTION_COUNT
    _INJECTION_COUNT = {'forward': 0, 'backward': 0}


__all__ = [
    'enable_noisy_ops',
    'disable_noisy_ops',
    'noisy_ops_context',
    'get_injection_stats',
    'reset_injection_stats',
    'NoisyMatMul',
    'NoisyBMM',
    'noisy_matmul',
    'noisy_bmm',
    'noisy_linear',
]
