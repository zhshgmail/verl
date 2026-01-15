"""
Minimal MXFP4 Fake Quantization Implementation

This module provides a standalone MXFP4 quantization implementation for
simulating quantization effects during SRDD scans and AQN training.

MXFP4 Format (E2M1K8B32):
- 2 exponent bits, 1 mantissa bit
- 8 shared exponent bits per block of 32 elements
- Representable values: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (with scaling)

Usage:
    from verl.utils.mxfp4_quant import mxfp4_quantize

    # Basic MXFP4 quantization
    x_quant = mxfp4_quantize(x)

    # With stochastic rounding
    x_quant = mxfp4_quantize(x, stochastic_rounding=True)

    # 2D block quantization (32x32 blocks)
    x_quant = mxfp4_quantize(x, block_2d=True)
"""

import torch
from torch import Tensor
from typing import Optional
from dataclasses import dataclass


@dataclass
class MXFP4Config:
    """Configuration for MXFP4 quantization"""
    exp_bits: int = 2       # Exponent bits
    man_bits: int = 1       # Mantissa bits
    k_bits: int = 8         # Shared exponent bits
    block_h: int = 1        # Block height (1 for 1D, 32 for 2D)
    block_w: int = 32       # Block width
    stochastic_rounding: bool = False
    truncation_free: bool = False

    @property
    def exp_max(self) -> int:
        return 2 ** (self.exp_bits - 1)

    @property
    def exp_min(self) -> int:
        return -2 ** (self.exp_bits - 1) + 2

    @property
    def k_max(self) -> int:
        return 2 ** (self.k_bits - 1) - 1

    @property
    def exp_offset(self) -> int:
        return self.exp_max

    @property
    def man_shift_bit(self) -> int:
        return self.man_bits

    @property
    def fp_val_max(self) -> float:
        return 2 ** self.exp_max * float(2 ** (self.man_bits + 1) - 1) / 2 ** self.man_bits

    @property
    def Qmax(self) -> float:
        Elow = -2 ** (self.exp_bits - 1) + 2
        Ehigh = Elow + 2 ** self.exp_bits - 2
        Mhigh = 2 ** self.man_bits - 1
        return (1 + Mhigh / (Mhigh + 1)) * (2 ** Ehigh)

    @property
    def Qmin(self) -> float:
        return -self.Qmax


# Pre-defined configurations
MXFP4_CONFIG = MXFP4Config()
MXFP4_SR_CONFIG = MXFP4Config(stochastic_rounding=True)
MXFP4_TF_CONFIG = MXFP4Config(truncation_free=True)
MXFP4_2D_CONFIG = MXFP4Config(block_h=32, block_w=32)


def mxfp4_quantize(
    x: Tensor,
    config: Optional[MXFP4Config] = None,
    stochastic_rounding: bool = False,
    truncation_free: bool = False,
    block_2d: bool = False,
) -> Tensor:
    """
    Apply MXFP4 fake quantization to a tensor with gradient support.

    Note: @torch.no_grad() decorator removed to allow gradient flow through
    quantization for proper QAT/W4A4 training.

    This performs quant -> dequant in one step, simulating the
    quantization error without actually storing in 4-bit format.

    Args:
        x: Input tensor (any shape, will be processed in blocks of 32)
        config: MXFP4Config object (overrides other args if provided)
        stochastic_rounding: Use stochastic rounding (reduces bias)
        truncation_free: Use truncation-free scaling (more precise)
        block_2d: Use 32x32 blocks instead of 1x32

    Returns:
        Quantized-dequantized tensor (same shape and dtype as input)
    """
    if config is None:
        config = MXFP4Config(
            stochastic_rounding=stochastic_rounding,
            truncation_free=truncation_free,
            block_h=32 if block_2d else 1,
            block_w=32,
        )

    original_shape = x.shape
    original_dtype = x.dtype

    # Ensure we work in float32 for numerical stability
    x = x.float()

    # Pad to fit block size
    numel = x.numel()
    block_size = config.block_h * config.block_w
    pad_size = (block_size - numel % block_size) % block_size

    if pad_size > 0:
        x_flat = x.flatten()
        x_padded = torch.cat([x_flat, torch.zeros(pad_size, device=x.device, dtype=x.dtype)])
    else:
        x_padded = x.flatten()

    # Reshape to blocks
    x_blocked = x_padded.view(-1, config.block_h, config.block_w)

    # Apply quantization
    out = _mxfp4_quant_dequant(x_blocked, config)

    # Reshape back
    out = out.flatten()
    if pad_size > 0:
        out = out[:-pad_size]
    out = out.view(original_shape)

    return out.to(original_dtype)


def _mxfp4_quant_dequant(x_blocked: Tensor, config: MXFP4Config) -> Tensor:
    """
    Core MXFP4 quantization-dequantization logic.

    Args:
        x_blocked: Tensor of shape (num_blocks, block_h, block_w)
        config: MXFP4Config

    Returns:
        Quantized-dequantized tensor of same shape
    """
    x_unsigned = torch.abs(x_blocked)
    sign = torch.sign(x_blocked)
    exp_offset = config.exp_offset

    # Compute shared exponent per block (max over block)
    max_inner = x_unsigned.amax(dim=(-1, -2), keepdim=True)

    if config.truncation_free:
        # Truncation-free scaling: more precise shared exponent
        max_inner = torch.where(max_inner == 0, torch.tensor(1e-7, device=max_inner.device), max_inner)
        shared_exp = torch.ceil(torch.log2(2 * max_inner / (config.Qmax - config.Qmin)))
    else:
        # Standard scaling
        shared_exp = torch.floor(torch.log2(max_inner + 1e-7)) - config.exp_max

    # Clamp shared exponent to valid range
    shared_exp = torch.clamp(shared_exp, -config.k_max - exp_offset, config.k_max - exp_offset)

    # Compute private exponent for each element
    x_scaled = x_unsigned / torch.exp2(shared_exp)
    private_exp = torch.floor(torch.log2(x_scaled + 1e-7))
    private_exp = torch.clamp(private_exp, config.exp_min, config.exp_max)

    # Compute mantissa
    mant = x_scaled / (2 ** private_exp)

    if config.stochastic_rounding:
        # Stochastic rounding: add random noise before rounding
        mant_int = torch.floor(mant)
        mant_frac = (mant - mant_int) * (2 ** config.man_shift_bit)
        mant_frac = mant_frac + torch.empty_like(mant_frac).uniform_(-0.5, 0.5)
        mant_frac = torch.clamp(mant_frac, 0, 2 ** config.man_shift_bit).round()
        mant_trunc = mant_int + mant_frac / (2 ** config.man_shift_bit)
    else:
        # Standard rounding
        mant_shifted = mant * (2 ** config.man_shift_bit)
        mant_trunc = torch.floor(mant_shifted + 0.5)

    # Compute final quantized value
    fp_value = mant_trunc / (2 ** config.man_shift_bit) * (2 ** private_exp)
    fp_value = torch.clamp(fp_value, -config.fp_val_max, config.fp_val_max)

    # Handle underflow (very small values -> 0)
    underflow_mask = shared_exp < -127
    fp_value = torch.where(
        underflow_mask.expand_as(fp_value),
        torch.zeros_like(fp_value),
        fp_value
    )

    # Dequantize: apply shared exponent and sign
    out = sign * (2 ** shared_exp) * fp_value

    return out


class MXFP4QuantHook:
    """
    PyTorch forward hook for applying MXFP4 fake quantization to layer outputs.

    Usage:
        hook = MXFP4QuantHook(stochastic_rounding=False)
        handle = layer.register_forward_hook(hook)
        # ... run forward pass ...
        handle.remove()
    """

    def __init__(
        self,
        stochastic_rounding: bool = False,
        truncation_free: bool = False,
        block_2d: bool = False,
    ):
        self.config = MXFP4Config(
            stochastic_rounding=stochastic_rounding,
            truncation_free=truncation_free,
            block_h=32 if block_2d else 1,
            block_w=32,
        )
        self.call_count = 0
        self.total_error = 0.0

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        # Apply MXFP4 quantization
        hidden_quant = mxfp4_quantize(hidden, config=self.config)

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


def compute_mxfp4_error(x: Tensor, **kwargs) -> Tensor:
    """
    Compute the quantization error introduced by MXFP4.

    Args:
        x: Input tensor
        **kwargs: Arguments passed to mxfp4_quantize

    Returns:
        Absolute quantization error tensor
    """
    x_quant = mxfp4_quantize(x, **kwargs)
    return (x_quant - x).abs()


def analyze_mxfp4_sensitivity(
    x: Tensor,
    **kwargs
) -> dict:
    """
    Analyze how sensitive a tensor is to MXFP4 quantization.

    Args:
        x: Input tensor
        **kwargs: Arguments passed to mxfp4_quantize

    Returns:
        Dictionary with error statistics
    """
    x_quant = mxfp4_quantize(x, **kwargs)
    error = (x_quant - x).abs()

    return {
        'mean_error': error.mean().item(),
        'max_error': error.max().item(),
        'std_error': error.std().item(),
        'relative_error': (error / (x.abs() + 1e-10)).mean().item(),
        'zero_ratio': (x_quant == 0).float().mean().item(),
        'original_zero_ratio': (x == 0).float().mean().item(),
    }


# Convenience functions for common configurations
def mxfp4_standard(x: Tensor) -> Tensor:
    """Standard MXFP4 quantization (E2M1K8B32)"""
    return mxfp4_quantize(x, config=MXFP4_CONFIG)


def mxfp4_sr(x: Tensor) -> Tensor:
    """MXFP4 with stochastic rounding"""
    return mxfp4_quantize(x, config=MXFP4_SR_CONFIG)


def mxfp4_2d(x: Tensor) -> Tensor:
    """MXFP4 with 2D block quantization (32x32)"""
    return mxfp4_quantize(x, config=MXFP4_2D_CONFIG)


if __name__ == "__main__":
    # Quick test
    print("Testing MXFP4 quantization...")

    x = torch.randn(128, 128)
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")

    # Test different modes
    for name, func in [
        ("Standard", mxfp4_standard),
        ("Stochastic Rounding", mxfp4_sr),
        ("2D Block", mxfp4_2d),
    ]:
        x_q = func(x)
        mse = ((x_q - x) ** 2).mean().item()
        print(f"\n{name}:")
        print(f"  MSE: {mse:.6f}")
        print(f"  Output range: [{x_q.min():.4f}, {x_q.max():.4f}]")

    # Test sensitivity analysis
    print("\nSensitivity analysis:")
    stats = analyze_mxfp4_sensitivity(x)
    for k, v in stats.items():
        print(f"  {k}: {v:.6f}")
