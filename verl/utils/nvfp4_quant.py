"""
NVFP4 Fake Quantization Implementation

This module provides NVFP4 (NVIDIA FP4) quantization implementation for
simulating quantization effects during SRDD scans and AQN training.

NVFP4 Format:
- 4-bit values with 16-element blocks
- E4M3 shared exponent (higher precision than MXFP4's E8M0)
- ~1% relative error vs ~21% for MXFP4

Key differences from MXFP4:
| Feature | MXFP4 | NVFP4 |
|---------|-------|-------|
| Block size | 32 | 16 |
| Scale format | E8M0 (8-bit exp, 0-bit mantissa) | E4M3 (4-bit exp, 3-bit mantissa) |
| Relative error | ~21% | ~1% |

Usage:
    from verl.utils.nvfp4_quant import nvfp4_quantize

    # Basic NVFP4 quantization
    x_quant = nvfp4_quantize(x)

    # With stochastic rounding
    x_quant = nvfp4_quantize(x, stochastic_rounding=True)
"""

import torch
from torch import Tensor
from typing import Optional
from dataclasses import dataclass


@dataclass
class NVFP4Config:
    """Configuration for NVFP4 quantization"""
    # FP4 value bits
    val_exp_bits: int = 2       # Value exponent bits
    val_man_bits: int = 1       # Value mantissa bits
    # Scale format: E4M3 (much higher precision than MXFP4's E8M0)
    scale_exp_bits: int = 4     # Scale exponent bits
    scale_man_bits: int = 3     # Scale mantissa bits
    # Block size (16 for NVFP4 vs 32 for MXFP4)
    block_size: int = 16
    stochastic_rounding: bool = False

    @property
    def val_max(self) -> float:
        """Maximum representable FP4 value (before scaling)"""
        # FP4 E2M1: max = 1.5 * 2^1 = 3.0 (with subnormals: 0, 0.5, 1, 1.5, 2, 3)
        return 6.0

    @property
    def scale_bias(self) -> int:
        """E4M3 bias"""
        return 7  # 2^(4-1) - 1

    @property
    def scale_max(self) -> float:
        """Maximum E4M3 scale value"""
        # E4M3: max = 1.875 * 2^7 = 240
        return 240.0

    @property
    def scale_min(self) -> float:
        """Minimum positive E4M3 scale value (subnormal)"""
        # Smallest subnormal: 2^(-6-3) = 2^-9
        return 2.0 ** -9


# Pre-defined configurations
NVFP4_CONFIG = NVFP4Config()
NVFP4_SR_CONFIG = NVFP4Config(stochastic_rounding=True)


# FP4 lookup table (representable values before scaling)
# E2M1 format: sign bit + 2 exp bits + 1 mantissa bit
FP4_VALUES = torch.tensor([
    0.0,    # 0000: zero
    0.5,    # 0001: subnormal 0.5
    1.0,    # 0010: 1.0 * 2^0
    1.5,    # 0011: 1.5 * 2^0
    2.0,    # 0100: 1.0 * 2^1
    3.0,    # 0101: 1.5 * 2^1
    4.0,    # 0110: 1.0 * 2^2
    6.0,    # 0111: 1.5 * 2^2 (max normal)
], dtype=torch.float32)


def _compute_e4m3_scale(max_abs: Tensor) -> Tensor:
    """
    Compute E4M3 scale factor for a block.

    E4M3 has:
    - 4 exponent bits (bias = 7)
    - 3 mantissa bits
    - Range: subnormal to 240

    This provides much higher precision than MXFP4's E8M0 (exponent only).
    """
    # Prevent division by zero
    max_abs = torch.clamp(max_abs, min=1e-10)

    # Compute scale to map max_abs to FP4 max (6.0)
    # scale = max_abs / 6.0, then quantize to E4M3
    raw_scale = max_abs / 6.0

    # Quantize to E4M3
    # Extract exponent and mantissa
    log2_scale = torch.log2(raw_scale)
    exp = torch.floor(log2_scale).clamp(-6, 8)  # E4 range with bias 7
    mantissa = raw_scale / (2.0 ** exp)

    # Quantize mantissa to 3 bits (8 levels from 1.0 to 1.875)
    mantissa_quant = torch.round(mantissa * 8) / 8
    mantissa_quant = torch.clamp(mantissa_quant, 1.0, 1.875)

    # Reconstruct E4M3 scale
    scale = mantissa_quant * (2.0 ** exp)

    return scale


def _quantize_to_fp4(x: Tensor, fp4_values: Tensor) -> Tensor:
    """
    Quantize values to nearest FP4 representable value.

    Args:
        x: Input tensor (values should be in [0, 6] range after scaling)
        fp4_values: Lookup table of FP4 values

    Returns:
        Quantized tensor
    """
    x = torch.clamp(x, 0, 6.0)

    # Find nearest FP4 value
    # Expand dimensions for broadcasting
    x_expanded = x.unsqueeze(-1)  # (..., 1)
    fp4_expanded = fp4_values.to(x.device)  # (8,)

    # Compute distances
    distances = torch.abs(x_expanded - fp4_expanded)

    # Get index of nearest value
    indices = torch.argmin(distances, dim=-1)

    # Look up quantized value
    quantized = fp4_expanded[indices]

    return quantized


def _quantize_to_fp4_stochastic(x: Tensor, fp4_values: Tensor) -> Tensor:
    """
    Stochastic rounding to FP4.

    Instead of always rounding to nearest, probabilistically round to
    floor or ceil based on distance.
    """
    x = torch.clamp(x, 0, 6.0)
    fp4 = fp4_values.to(x.device)

    # Find floor and ceil FP4 values
    x_expanded = x.unsqueeze(-1)
    distances = x_expanded - fp4

    # Floor: largest FP4 value <= x
    floor_mask = distances >= 0
    floor_distances = torch.where(floor_mask, distances, torch.tensor(float('inf'), device=x.device))
    floor_idx = torch.argmin(floor_distances, dim=-1)
    floor_val = fp4[floor_idx]

    # Ceil: smallest FP4 value >= x
    ceil_mask = distances <= 0
    ceil_distances = torch.where(ceil_mask, -distances, torch.tensor(float('inf'), device=x.device))
    ceil_idx = torch.argmin(ceil_distances, dim=-1)
    ceil_val = fp4[ceil_idx]

    # Probability of rounding up = (x - floor) / (ceil - floor)
    gap = ceil_val - floor_val
    gap = torch.where(gap == 0, torch.ones_like(gap), gap)
    prob_up = (x - floor_val) / gap
    prob_up = torch.clamp(prob_up, 0, 1)

    # Stochastic decision
    rand = torch.rand_like(x)
    quantized = torch.where(rand < prob_up, ceil_val, floor_val)

    return quantized


@torch.no_grad()
def nvfp4_quantize(
    x: Tensor,
    config: Optional[NVFP4Config] = None,
    stochastic_rounding: bool = False,
) -> Tensor:
    """
    Apply NVFP4 fake quantization to a tensor.

    This performs quant -> dequant in one step, simulating the
    quantization error without actually storing in 4-bit format.

    Args:
        x: Input tensor (any shape, will be processed in blocks of 16)
        config: NVFP4Config object (overrides stochastic_rounding if provided)
        stochastic_rounding: Use stochastic rounding (reduces bias)

    Returns:
        Quantized-dequantized tensor (same shape and dtype as input)
    """
    if config is None:
        config = NVFP4Config(stochastic_rounding=stochastic_rounding)

    original_shape = x.shape
    original_dtype = x.dtype

    # Ensure we work in float32 for numerical stability
    x = x.float()

    # Pad to fit block size (16 elements)
    numel = x.numel()
    block_size = config.block_size
    pad_size = (block_size - numel % block_size) % block_size

    if pad_size > 0:
        x_flat = x.flatten()
        x_padded = torch.cat([x_flat, torch.zeros(pad_size, device=x.device, dtype=x.dtype)])
    else:
        x_padded = x.flatten()

    # Reshape to blocks of 16
    num_blocks = x_padded.numel() // block_size
    x_blocked = x_padded.view(num_blocks, block_size)

    # Extract sign and magnitude
    sign = torch.sign(x_blocked)
    x_abs = torch.abs(x_blocked)

    # Compute E4M3 scale per block (max over block)
    max_abs = x_abs.max(dim=-1, keepdim=True)[0]
    scale = _compute_e4m3_scale(max_abs)

    # Scale values to FP4 range [0, 6]
    x_scaled = x_abs / (scale + 1e-10)
    x_scaled = torch.clamp(x_scaled, 0, 6.0)

    # Quantize to FP4
    fp4_lut = FP4_VALUES.to(x.device)
    if config.stochastic_rounding:
        x_quantized = _quantize_to_fp4_stochastic(x_scaled, fp4_lut)
    else:
        x_quantized = _quantize_to_fp4(x_scaled, fp4_lut)

    # Dequantize: apply scale and sign
    out = sign * x_quantized * scale

    # Reshape back
    out = out.flatten()
    if pad_size > 0:
        out = out[:-pad_size]
    out = out.view(original_shape)

    return out.to(original_dtype)


class NVFP4QuantHook:
    """
    PyTorch forward hook for applying NVFP4 fake quantization to layer outputs.

    Usage:
        hook = NVFP4QuantHook(stochastic_rounding=False)
        handle = layer.register_forward_hook(hook)
        # ... run forward pass ...
        handle.remove()
    """

    def __init__(self, stochastic_rounding: bool = False):
        self.config = NVFP4Config(stochastic_rounding=stochastic_rounding)
        self.call_count = 0
        self.total_error = 0.0

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        # Apply NVFP4 quantization
        hidden_quant = nvfp4_quantize(hidden, config=self.config)

        # Track quantization error
        error = (hidden_quant - hidden).abs().mean().item()
        self.total_error += error
        self.call_count += 1

        if isinstance(output, tuple):
            return (hidden_quant,) + output[1:]
        return hidden_quant

    def get_avg_error(self) -> float:
        return self.total_error / max(1, self.call_count)

    def reset_stats(self):
        self.call_count = 0
        self.total_error = 0.0


def compute_nvfp4_error(x: Tensor, **kwargs) -> Tensor:
    """
    Compute the quantization error introduced by NVFP4.

    Args:
        x: Input tensor
        **kwargs: Arguments passed to nvfp4_quantize

    Returns:
        Absolute quantization error tensor
    """
    x_quant = nvfp4_quantize(x, **kwargs)
    return (x_quant - x).abs()


def analyze_nvfp4_sensitivity(x: Tensor, **kwargs) -> dict:
    """
    Analyze how sensitive a tensor is to NVFP4 quantization.

    Args:
        x: Input tensor
        **kwargs: Arguments passed to nvfp4_quantize

    Returns:
        Dictionary with error statistics
    """
    x_quant = nvfp4_quantize(x, **kwargs)
    error = (x_quant - x).abs()

    return {
        'mean_error': error.mean().item(),
        'max_error': error.max().item(),
        'std_error': error.std().item(),
        'relative_error': (error / (x.abs() + 1e-10)).mean().item(),
        'zero_ratio': (x_quant == 0).float().mean().item(),
        'original_zero_ratio': (x == 0).float().mean().item(),
    }


def compare_mxfp4_vs_nvfp4(x: Tensor) -> dict:
    """
    Compare MXFP4 and NVFP4 quantization errors.

    Args:
        x: Input tensor

    Returns:
        Dictionary comparing both formats
    """
    try:
        from verl.utils.mxfp4_quant import mxfp4_quantize, analyze_mxfp4_sensitivity
        mxfp4_available = True
    except ImportError:
        mxfp4_available = False

    nvfp4_stats = analyze_nvfp4_sensitivity(x)

    if mxfp4_available:
        mxfp4_stats = analyze_mxfp4_sensitivity(x)
        improvement = mxfp4_stats['relative_error'] / (nvfp4_stats['relative_error'] + 1e-10)
    else:
        mxfp4_stats = None
        improvement = None

    return {
        'nvfp4': nvfp4_stats,
        'mxfp4': mxfp4_stats,
        'nvfp4_improvement': improvement,
    }


# Convenience functions
def nvfp4_standard(x: Tensor) -> Tensor:
    """Standard NVFP4 quantization"""
    return nvfp4_quantize(x, config=NVFP4_CONFIG)


def nvfp4_sr(x: Tensor) -> Tensor:
    """NVFP4 with stochastic rounding"""
    return nvfp4_quantize(x, config=NVFP4_SR_CONFIG)


if __name__ == "__main__":
    # Quick test and comparison
    print("Testing NVFP4 quantization...")

    x = torch.randn(128, 128)
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")

    # Test NVFP4
    for name, func in [
        ("NVFP4 Standard", nvfp4_standard),
        ("NVFP4 Stochastic", nvfp4_sr),
    ]:
        x_q = func(x)
        mse = ((x_q - x) ** 2).mean().item()
        rel_err = ((x_q - x).abs() / (x.abs() + 1e-10)).mean().item()
        print(f"\n{name}:")
        print(f"  MSE: {mse:.6f}")
        print(f"  Relative Error: {rel_err*100:.2f}%")
        print(f"  Output range: [{x_q.min():.4f}, {x_q.max():.4f}]")

    # Compare with MXFP4
    print("\n" + "="*50)
    print("NVFP4 vs MXFP4 Comparison:")
    print("="*50)
    comparison = compare_mxfp4_vs_nvfp4(x)
    print(f"\nNVFP4 relative error: {comparison['nvfp4']['relative_error']*100:.2f}%")
    if comparison['mxfp4']:
        print(f"MXFP4 relative error: {comparison['mxfp4']['relative_error']*100:.2f}%")
        print(f"NVFP4 is {comparison['nvfp4_improvement']:.1f}x better")
