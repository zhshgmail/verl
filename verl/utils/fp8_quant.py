"""
FP8 Fake Quantization Implementation

This module provides FP8 (E4M3) fake quantization for simulating
quantization effects during training. FP8 has much lower quantization
error than FP4 formats (~0.1% vs ~1-21%).

FP8 E4M3 Format:
- 1 sign bit, 4 exponent bits, 3 mantissa bits
- Dynamic range: ~±448
- Precision: ~0.1% relative error

Usage:
    from verl.utils.fp8_quant import fp8_quantize

    # Basic FP8 quantization
    x_quant = fp8_quantize(x)

    # With stochastic rounding
    x_quant = fp8_quantize(x, stochastic_rounding=True)
"""

import torch
from torch import Tensor
from typing import Optional
from dataclasses import dataclass


@dataclass
class FP8Config:
    """Configuration for FP8 quantization"""
    exp_bits: int = 4       # Exponent bits (E4M3)
    man_bits: int = 3       # Mantissa bits
    exp_bias: int = 7       # Exponent bias for E4M3
    stochastic_rounding: bool = False

    @property
    def max_val(self) -> float:
        """Maximum representable value in FP8 E4M3: 448"""
        # E4M3: max = 2^(15-7) * (1 + 7/8) = 256 * 1.875 = 480
        # But E4M3 reserves some values, practical max is 448
        return 448.0

    @property
    def min_val(self) -> float:
        """Minimum positive value in FP8 E4M3"""
        # 2^(1-7) * (1 + 0/8) = 2^-6 = 0.015625
        return 2 ** (-6)


# Pre-defined configurations
FP8_E4M3_CONFIG = FP8Config()
FP8_E4M3_SR_CONFIG = FP8Config(stochastic_rounding=True)


def _fp8_e4m3_clamp(x: Tensor, config: FP8Config) -> Tensor:
    """Clamp values to FP8 E4M3 representable range"""
    return torch.clamp(x, -config.max_val, config.max_val)


@torch.no_grad()
def fp8_quantize(
    x: Tensor,
    config: Optional[FP8Config] = None,
    stochastic_rounding: bool = False,
) -> Tensor:
    """
    Apply FP8 E4M3 fake quantization to a tensor.

    This performs quant -> dequant in one step, simulating the
    quantization error without actually storing in 8-bit format.

    Args:
        x: Input tensor (any shape)
        config: FP8Config object (default: FP8_E4M3_CONFIG)
        stochastic_rounding: Use stochastic rounding (reduces bias)

    Returns:
        Tensor with FP8 quantization error applied
    """
    if config is None:
        config = FP8_E4M3_SR_CONFIG if stochastic_rounding else FP8_E4M3_CONFIG

    original_dtype = x.dtype
    x = x.float()

    # Handle zeros
    zero_mask = x == 0

    # Clamp to representable range
    x_clamped = _fp8_e4m3_clamp(x, config)

    # Extract sign
    sign = torch.sign(x_clamped)
    x_abs = torch.abs(x_clamped)

    # Avoid log of zero
    x_abs = torch.clamp(x_abs, min=config.min_val)

    # Compute exponent and mantissa
    # FP8 E4M3: value = sign * 2^(exp - bias) * (1 + mantissa/8)
    log2_x = torch.log2(x_abs)
    exp = torch.floor(log2_x)

    # Clamp exponent to valid range [1-bias, 15-bias] for E4M3
    exp = torch.clamp(exp, min=1 - config.exp_bias, max=14 - config.exp_bias)

    # Compute mantissa: x_abs = 2^exp * (1 + m/8), so m = (x_abs/2^exp - 1) * 8
    mantissa_raw = (x_abs / (2.0 ** exp) - 1.0) * (2 ** config.man_bits)

    # Quantize mantissa
    if config.stochastic_rounding or stochastic_rounding:
        # Stochastic rounding: round up with probability = fractional part
        noise = torch.rand_like(mantissa_raw)
        mantissa_quant = torch.floor(mantissa_raw + noise)
    else:
        # Round to nearest
        mantissa_quant = torch.round(mantissa_raw)

    # Clamp mantissa to valid range [0, 7] for 3-bit mantissa
    mantissa_quant = torch.clamp(mantissa_quant, min=0, max=2**config.man_bits - 1)

    # Dequantize: reconstruct value
    x_quant = sign * (2.0 ** exp) * (1.0 + mantissa_quant / (2 ** config.man_bits))

    # Restore zeros
    x_quant = torch.where(zero_mask, torch.zeros_like(x_quant), x_quant)

    return x_quant.to(original_dtype)


@torch.no_grad()
def compute_fp8_error(x: Tensor, config: Optional[FP8Config] = None) -> dict:
    """
    Compute FP8 quantization error statistics.

    Args:
        x: Input tensor
        config: FP8Config object

    Returns:
        Dict with error statistics
    """
    x_quant = fp8_quantize(x, config)

    error = x - x_quant
    relative_error = torch.abs(error) / (torch.abs(x) + 1e-10)

    return {
        "abs_error_mean": error.abs().mean().item(),
        "abs_error_max": error.abs().max().item(),
        "rel_error_mean": relative_error.mean().item(),
        "rel_error_max": relative_error.max().item(),
        "sqnr_db": (10 * torch.log10(
            (x ** 2).mean() / ((error ** 2).mean() + 1e-10)
        )).item(),
    }
