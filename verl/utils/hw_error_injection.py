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
Hardware Heterogeneous Error Injection for simulating GPU/NPU differences.

This module injects synthetic errors into operator OUTPUTS (activations) during
forward pass, similar to fake quantization. This is different from noise_injection.py
which modifies model WEIGHTS.

Use case:
- Simulate NPU-like numerical errors on GPU
- Test if AQN (Adaptive Quantization Noise) can improve robustness to HW errors
- Study model sensitivity to different operators' numerical precision

Design inspired by:
- quant_compute library's fake quantization (quant/dequant pattern)
- QeRL's noise injection for quantization robustness
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Callable, Tuple
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# STE (Straight-Through Estimator) Implementation for Proper QAT
# =============================================================================

class STEQuantizeWeight(torch.autograd.Function):
    """
    Straight-Through Estimator for weight quantization.

    Forward: Returns quantized weight
    Backward: Passes gradient through unchanged (as if no quantization)

    This is the standard QAT approach used in PyTorch's quantization-aware training.
    The key insight is that during backward:
    - grad_W is computed correctly (uses x from forward with quantized weights)
    - grad_x = grad_output @ W_quant (NOT W_orig!)

    Usage:
        quantized_weight = STEQuantizeWeight.apply(weight, quantize_fn)
    """

    @staticmethod
    def forward(ctx, weight: torch.Tensor, quantize_fn: Callable) -> torch.Tensor:
        """
        Forward pass: apply quantization.

        Args:
            weight: Original weight tensor
            quantize_fn: Function that takes weight and returns quantized weight

        Returns:
            Quantized weight tensor
        """
        # Apply quantization
        quantized = quantize_fn(weight)
        # Save quantized weight for backward (used in grad_x computation)
        ctx.save_for_backward(quantized)
        return quantized

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Backward pass: straight-through estimator.

        The gradient passes through unchanged, as if quantization was identity.
        This is the standard STE approach for QAT.

        Args:
            grad_output: Gradient from downstream

        Returns:
            Tuple of (gradient for weight, None for quantize_fn)
        """
        # STE: pass gradient through unchanged
        return grad_output, None


class STEQuantizeActivation(torch.autograd.Function):
    """
    Straight-Through Estimator for activation quantization.

    Forward: Returns quantized activation
    Backward: Passes gradient through unchanged

    Used for W16A4 (weight FP16, activation FP4) scenarios.
    """

    @staticmethod
    def forward(ctx, activation: torch.Tensor, quantize_fn: Callable) -> torch.Tensor:
        """Apply quantization to activation."""
        return quantize_fn(activation)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """STE: pass gradient through unchanged."""
        return grad_output, None


class HWErrorConfig:
    """Configuration for HW error injection."""

    def __init__(
        self,
        enabled: bool = False,
        error_scale: float = 1e-5,
        error_type: str = 'relative_gaussian',
        injection_point: str = 'input',  # 'input' or 'output'
        target_modules: Optional[List[str]] = None,
        target_layers: Optional[List[int]] = None,  # v2.0: Specific layer indices
        apply_during: str = 'both',  # 'rollout', 'training', 'both'
        seed: Optional[int] = None,
        # Deadzone-specific config (for error_type='deadzone')
        deadzone_threshold: float = 0.01,  # 1% of max value
        # MXFP4-specific config (for error_type='mxfp4')
        mxfp4_stochastic_rounding: bool = False,
        mxfp4_block_2d: bool = False,
        # NVFP4-specific config (for error_type='nvfp4')
        nvfp4_stochastic_rounding: bool = False,
        # Module exclusion list (for production PTQ recipes)
        exclude_modules: Optional[List[str]] = None,
        # STE (Straight-Through Estimator) mode for proper QAT gradient flow
        # IMPORTANT: Must be True for FullFT training with weight quantization!
        use_ste: bool = True,
    ):
        """
        Args:
            enabled: Whether to enable error injection
            error_scale: Scale of the error (relative to tensor magnitude)
                - 1e-6: ~BF16 precision level (very small)
                - 1e-5: Moderate (recommended starting point)
                - 1e-4: Aggressive (may cause training instability)
                - For deadzone/mxfp4: not used
            error_type: Type of error to inject
                - 'relative_gaussian': error = randn() * |tensor| * scale
                - 'absolute_gaussian': error = randn() * scale
                - 'systematic_bias': error = sign(tensor) * scale (simulates rounding bias)
                - 'deadzone': MXFP4 deadzone - values < threshold*max are zeroed
                - 'mxfp4': Full MXFP4 fake quantization (E2M1K8B32, ~21% error)
                - 'nvfp4': NVFP4 fake quantization (E4M3 scale, ~1% error, QeRL compatible)
                - 'fp8': FP8 E4M3 fake quantization (~0.1% error, highest precision)
            injection_point: Where to inject error (like quant_compute fake quantization)
                - 'input': Inject error to operator INPUT activations (before computation)
                          Like fake quantization: y = operator(x + error)
                          Simulates error in activation representation (W16A4)
                - 'output': Inject error to operator OUTPUT (after computation)
                          Like: y = operator(x) + error
                          Simulates error in operator's HW implementation
                - 'weight': Inject error to WEIGHT tensors (W4A16 - QeRL style)
                          Like: y = operator_with_quantized_weight(x)
                          Simulates static weight quantization with FP16 activations
            target_modules: List of module name patterns to target
                - Default: ['rmsnorm'] - only RMSNorm layers
                - Options: ['rmsnorm', 'down_proj', 'o_proj', 'linear']
                - For mxfp4: recommend ['linear'] to target all linear layers
            target_layers: List of layer indices to target (None = all layers)
                - Example: [15] - only layer 15
                - Used with SRDD-guided injection
            apply_during: When to apply error injection
                - 'rollout': Only during vLLM inference
                - 'training': Only during FSDP training
                - 'both': During both phases
            seed: Random seed for reproducibility (None = random)
            deadzone_threshold: For deadzone error type, values with |x| < threshold * max(|x|)
                              are set to zero (simulates MXFP4 quantization deadzone)
            mxfp4_stochastic_rounding: Use stochastic rounding for MXFP4 (reduces bias)
            mxfp4_block_2d: Use 32x32 2D blocks instead of 1x32 for MXFP4
            exclude_modules: List of module name patterns to EXCLUDE from quantization
                - Default for MXFP4/NVFP4: ['lm_head', 'embed_tokens'] (production PTQ recipe)
                - These layers are excluded because quantizing them is destructive:
                  - lm_head: Output logits require high precision for token prediction
                  - embed_tokens: Input embeddings need high precision for semantic encoding
            use_ste: Use Straight-Through Estimator for proper QAT gradient flow
                - True (default): Proper STE - backward uses quantized weights for grad_x
                - False: Legacy mode - backward uses original weights (BROKEN for FullFT!)
                - IMPORTANT: Must be True for FullFT training with trainable base weights
                - For LoRA (frozen base), either mode works but STE is safer
        """
        self.enabled = enabled
        self.error_scale = error_scale
        self.error_type = error_type
        self.injection_point = injection_point
        self.target_modules = target_modules or ['rmsnorm']
        self.target_layers = target_layers  # None = all layers
        self.apply_during = apply_during
        self.seed = seed
        self.deadzone_threshold = deadzone_threshold
        self.mxfp4_stochastic_rounding = mxfp4_stochastic_rounding
        self.mxfp4_block_2d = mxfp4_block_2d
        self.nvfp4_stochastic_rounding = nvfp4_stochastic_rounding
        # Default exclusion for MXFP4/NVFP4/FP8: exclude lm_head and embed_tokens
        # This follows production PTQ recipes that keep these layers in FP16
        if exclude_modules is None and error_type in ('mxfp4', 'nvfp4', 'fp8'):
            self.exclude_modules = ['lm_head', 'embed_tokens']
        else:
            self.exclude_modules = exclude_modules or []
        # STE mode for proper QAT gradient flow
        self.use_ste = use_ste

    def __repr__(self):
        if self.error_type == 'deadzone':
            return (f"HWErrorConfig(enabled={self.enabled}, type={self.error_type}, "
                    f"threshold={self.deadzone_threshold}, layers={self.target_layers})")
        if self.error_type == 'mxfp4':
            return (f"HWErrorConfig(enabled={self.enabled}, type={self.error_type}, "
                    f"sr={self.mxfp4_stochastic_rounding}, 2d={self.mxfp4_block_2d}, "
                    f"targets={self.target_modules}, exclude={self.exclude_modules}, layers={self.target_layers})")
        if self.error_type == 'nvfp4':
            return (f"HWErrorConfig(enabled={self.enabled}, type={self.error_type}, "
                    f"sr={self.nvfp4_stochastic_rounding}, "
                    f"targets={self.target_modules}, exclude={self.exclude_modules}, layers={self.target_layers})")
        if self.error_type == 'fp8':
            return (f"HWErrorConfig(enabled={self.enabled}, type={self.error_type}, "
                    f"targets={self.target_modules}, exclude={self.exclude_modules}, layers={self.target_layers})")
        return (f"HWErrorConfig(enabled={self.enabled}, scale={self.error_scale}, "
                f"type={self.error_type}, targets={self.target_modules}, layers={self.target_layers})")


class HWErrorInjector:
    """
    Inject simulated HW heterogeneous errors into model forward pass.

    Uses PyTorch forward hooks to modify operator outputs, similar to
    fake quantization in quant_compute library.

    Example:
        >>> config = HWErrorConfig(enabled=True, error_scale=1e-5)
        >>> injector = HWErrorInjector(config)
        >>> injector.register_hooks(model)
        >>> # Run forward pass - errors are injected automatically
        >>> output = model(input_ids)
        >>> # Remove hooks when done
        >>> injector.remove_hooks()
    """

    def __init__(self, config: HWErrorConfig):
        self.config = config
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.stats: Dict[str, Dict] = {}  # Track injection statistics
        self._phase = 'both'  # Current phase: 'rollout', 'training', or 'both'

        if config.seed is not None:
            self._rng = torch.Generator()
            self._rng.manual_seed(config.seed)
        else:
            self._rng = None

        # Initialize MXFP4 quantizer if needed
        self._mxfp4_quantize = None
        if config.error_type == 'mxfp4':
            try:
                from verl.utils.mxfp4_quant import mxfp4_quantize, MXFP4Config
                self._mxfp4_config = MXFP4Config(
                    stochastic_rounding=config.mxfp4_stochastic_rounding,
                    block_h=32 if config.mxfp4_block_2d else 1,
                    block_w=32,
                )
                self._mxfp4_quantize = mxfp4_quantize
                # MXFP4 should use input injection (quant before computation)
                # This simulates: output = linear(quant_dequant(input))
                # Where quant_dequant adds the MXFP4 error to input activations
                if config.injection_point == 'output':
                    logger.warning("[MXFP4] injection_point='output' will apply quant AFTER computation")
                logger.info(f"[MXFP4] Initialized MXFP4 quantizer: sr={config.mxfp4_stochastic_rounding}, 2d={config.mxfp4_block_2d}, injection={config.injection_point}")
            except ImportError as e:
                raise ImportError(f"MXFP4 error type requires verl.utils.mxfp4_quant: {e}")

        # Initialize NVFP4 quantizer if needed
        self._nvfp4_quantize = None
        self._nvfp4_quantize_columnwise = None
        if config.error_type == 'nvfp4':
            try:
                from verl.utils.nvfp4_quant import nvfp4_quantize, nvfp4_quantize_columnwise, NVFP4Config
                self._nvfp4_config = NVFP4Config(
                    stochastic_rounding=config.nvfp4_stochastic_rounding,
                )
                self._nvfp4_quantize = nvfp4_quantize
                self._nvfp4_quantize_columnwise = nvfp4_quantize_columnwise
                if config.injection_point == 'output':
                    logger.warning("[NVFP4] injection_point='output' will apply quant AFTER computation")
                logger.info(f"[NVFP4] Initialized NVFP4 quantizer (columnwise): sr={config.nvfp4_stochastic_rounding}, injection={config.injection_point}")
            except ImportError as e:
                raise ImportError(f"NVFP4 error type requires verl.utils.nvfp4_quant: {e}")

        # Initialize FP8 quantizer if needed
        self._fp8_quantize = None
        if config.error_type == 'fp8':
            try:
                from verl.utils.fp8_quant import fp8_quantize, FP8Config
                self._fp8_config = FP8Config()
                self._fp8_quantize = fp8_quantize
                if config.injection_point == 'output':
                    logger.warning("[FP8] injection_point='output' will apply quant AFTER computation")
                logger.info(f"[FP8] Initialized FP8 E4M3 quantizer: injection={config.injection_point}")
            except ImportError as e:
                raise ImportError(f"FP8 error type requires verl.utils.fp8_quant: {e}")

    def set_phase(self, phase: str):
        """Set current execution phase for conditional injection."""
        assert phase in ['rollout', 'training', 'both']
        self._phase = phase

    def _should_inject(self) -> bool:
        """Check if error should be injected based on config and phase."""
        if not self.config.enabled:
            return False
        if self.config.apply_during == 'both':
            return True
        return self.config.apply_during == self._phase

    def _compute_error(self, output: torch.Tensor) -> torch.Tensor:
        """
        Compute error tensor based on error type.

        Similar to quant_compute's pattern:
        - Quantization: x_quant = quant(x) -> dequant(x_quant) -> error baked in
        - Our approach: x_error = x + error_func(x) -> error added directly

        For deadzone: returns the MODIFIED output directly (not error to add).
        """
        scale = self.config.error_scale

        if self.config.error_type == 'relative_gaussian':
            # Error proportional to value magnitude (most common HW error model)
            # Similar to floating point relative error
            if self._rng is not None:
                noise = torch.randn(output.shape, generator=self._rng,
                                   device=output.device, dtype=torch.float32)
            else:
                noise = torch.randn_like(output, dtype=torch.float32)
            error = noise * output.abs().float() * scale

        elif self.config.error_type == 'absolute_gaussian':
            # Constant magnitude error (simulates fixed-point noise)
            if self._rng is not None:
                noise = torch.randn(output.shape, generator=self._rng,
                                   device=output.device, dtype=torch.float32)
            else:
                noise = torch.randn_like(output, dtype=torch.float32)
            error = noise * scale

        elif self.config.error_type == 'systematic_bias':
            # Systematic bias (simulates rounding mode differences)
            error = torch.sign(output.float()) * scale

        elif self.config.error_type == 'deadzone':
            # MXFP4 deadzone: values below threshold are zeroed
            # This REPLACES values, not adds error
            # Return None to signal special handling needed
            return None  # Special case handled in hook

        elif self.config.error_type == 'mxfp4':
            # Full MXFP4 fake quantization
            # Return None to signal special handling needed
            return None  # Special case handled in hook

        elif self.config.error_type == 'nvfp4':
            # Full NVFP4 fake quantization
            # Return None to signal special handling needed
            return None  # Special case handled in hook

        elif self.config.error_type == 'fp8':
            # Full FP8 E4M3 fake quantization
            # Return None to signal special handling needed
            return None  # Special case handled in hook

        else:
            raise ValueError(f"Unknown error type: {self.config.error_type}")

        return error.to(output.dtype)

    def _apply_deadzone(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply MXFP4 deadzone effect to tensor.

        Values with |x| < threshold * max(|x|) are set to zero.
        This simulates quantization deadzone where small values
        cannot be represented and are rounded to zero.

        Args:
            tensor: Input tensor

        Returns:
            Tensor with deadzone applied
        """
        threshold = self.config.deadzone_threshold
        max_val = tensor.abs().max()
        deadzone_threshold = threshold * max_val

        # Create mask for values in deadzone
        deadzone_mask = tensor.abs() < deadzone_threshold

        # Zero out values in deadzone
        result = tensor.masked_fill(deadzone_mask, 0.0)

        return result, deadzone_mask

    def _apply_mxfp4(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Apply full MXFP4 fake quantization to tensor.

        Uses the verl.utils.mxfp4_quant module for proper E2M1K8B32
        quantization simulation including:
        - Shared exponent per block of 32 elements
        - 2-bit exponent, 1-bit mantissa
        - Proper deadzone and saturation handling

        Args:
            tensor: Input tensor

        Returns:
            Tuple of (quantized tensor, stats dict)
        """
        if self._mxfp4_quantize is None:
            raise RuntimeError("MXFP4 quantizer not initialized")

        # Apply MXFP4 quantization
        quantized = self._mxfp4_quantize(tensor, config=self._mxfp4_config)

        # Compute error statistics
        error = (quantized - tensor).abs()
        stats = {
            'mean_error': error.mean().item(),
            'max_error': error.max().item(),
            'relative_error': (error / (tensor.abs() + 1e-10)).mean().item(),
        }

        return quantized, stats

    def _apply_nvfp4(self, tensor: torch.Tensor, is_weight: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Apply NVFP4 fake quantization to tensor.

        NVFP4 features:
        - 16-element blocks (vs 32 for MXFP4)
        - E4M3 shared exponent (higher precision than MXFP4's E8M0)
        - ~1% relative error (vs ~21% for MXFP4)

        This is the format used by QeRL and is more suitable for
        quantization-aware training.

        IMPORTANT: For 2D weight tensors, uses COLUMN-WISE blocking to match
        the quant_compute reference implementation (To_NVF4). This processes
        G=16 rows at a time per column, matching how the reference works.

        Args:
            tensor: Input tensor
            is_weight: If True, use column-wise blocking for 2D tensors

        Returns:
            Tuple of (quantized tensor, stats dict)
        """
        if self._nvfp4_quantize is None:
            raise RuntimeError("NVFP4 quantizer not initialized")

        # For 2D weight tensors, use column-wise blocking to match quant_compute reference
        if is_weight and tensor.dim() == 2 and self._nvfp4_quantize_columnwise is not None:
            quantized = self._nvfp4_quantize_columnwise(tensor, config=self._nvfp4_config)
        else:
            # For activations (3D) or non-weight tensors, use standard row-wise blocking
            quantized = self._nvfp4_quantize(tensor, config=self._nvfp4_config)

        # Compute error statistics
        error = (quantized - tensor).abs()
        stats = {
            'mean_error': error.mean().item(),
            'max_error': error.max().item(),
            'relative_error': (error / (tensor.abs() + 1e-10)).mean().item(),
        }

        return quantized, stats

    def _apply_fp8(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Apply FP8 E4M3 fake quantization to tensor.

        FP8 E4M3 features:
        - 4 exponent bits, 3 mantissa bits
        - Per-tensor quantization (no blocks)
        - ~0.1% relative error (highest precision)

        Args:
            tensor: Input tensor

        Returns:
            Tuple of (quantized tensor, stats dict)
        """
        if self._fp8_quantize is None:
            raise RuntimeError("FP8 quantizer not initialized")

        # Apply FP8 quantization
        quantized = self._fp8_quantize(tensor, config=self._fp8_config)

        # Compute error statistics
        error = (quantized - tensor).abs()
        stats = {
            'mean_error': error.mean().item(),
            'max_error': error.max().item(),
            'relative_error': (error / (tensor.abs() + 1e-10)).mean().item(),
        }

        return quantized, stats

    def _extract_layer_id(self, name: str) -> Optional[int]:
        """Extract layer ID from module name like 'model.layers.15.mlp.down_proj'."""
        import re
        match = re.search(r'\.layers\.(\d+)\.', name)
        if match:
            return int(match.group(1))
        return None

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        """Check if module should have error injection applied."""
        name_lower = name.lower()
        class_name = module.__class__.__name__.lower()

        # First check layer filter if specified
        if self.config.target_layers is not None:
            layer_id = self._extract_layer_id(name)
            if layer_id is None or layer_id not in self.config.target_layers:
                return False

        # Check exclusion list (e.g., lm_head, embed_tokens for production PTQ)
        for exclude_pattern in self.config.exclude_modules:
            if exclude_pattern.lower() in name_lower:
                return False

        # For deadzone, we target the decoder layer output (the whole layer)
        if self.config.error_type == 'deadzone':
            # Match decoder layer modules like "model.layers.15" or "layers.15"
            import re
            if re.match(r'.*\.layers\.\d+$', name) or re.match(r'layers\.\d+$', name):
                return True
            return False

        # For MXFP4/NVFP4/FP8, use target_modules (default to 'linear' for all linear layers)
        # This is different from deadzone which targets whole decoder layers
        if self.config.error_type in ('mxfp4', 'nvfp4', 'fp8'):
            # Default to linear layers if not specified
            targets = self.config.target_modules
            if not targets or targets == ['rmsnorm']:
                # Override default for MXFP4/NVFP4 - should target linear layers
                targets = ['linear']

            for target in targets:
                target_lower = target.lower()
                if target_lower == 'linear':
                    if isinstance(module, nn.Linear):
                        return True
                    if 'linear' in class_name:
                        return True
                if target_lower in name_lower or target_lower in class_name:
                    return True
            return False

        for target in self.config.target_modules:
            target_lower = target.lower()

            # Check by class name
            if target_lower == 'rmsnorm' and 'rmsnorm' in class_name:
                return True
            if target_lower == 'linear':
                # Standard PyTorch Linear
                if isinstance(module, nn.Linear):
                    return True
                # vLLM's custom Linear layers (ColumnParallelLinear, RowParallelLinear, etc.)
                if 'linear' in class_name:
                    return True
            if target_lower == 'layernorm' and 'layernorm' in class_name:
                return True

            # Check by module name pattern
            if target_lower in name_lower:
                return True

        return False

    def _create_forward_hook(self, name: str) -> Callable:
        """Create forward hook (post-hook) for OUTPUT injection."""
        def hook(module: nn.Module, input: Tuple, output: torch.Tensor) -> torch.Tensor:
            if not self._should_inject():
                return output

            # Handle tuple outputs (some modules return multiple tensors)
            if isinstance(output, tuple):
                tensor_to_modify = output[0]
                rest = output[1:]
                is_tuple = True
            else:
                tensor_to_modify = output
                rest = None
                is_tuple = False

            # Handle deadzone specially (it replaces values, not adds error)
            if self.config.error_type == 'deadzone':
                modified, deadzone_mask = self._apply_deadzone(tensor_to_modify)
                num_zeroed = deadzone_mask.sum().item()
                total = deadzone_mask.numel()
                zero_rate = num_zeroed / total * 100

                # Track statistics
                if name not in self.stats:
                    self.stats[name] = {'count': 0, 'zeroed': 0, 'total': 0, 'point': 'output', 'type': 'deadzone'}
                self.stats[name]['count'] += 1
                self.stats[name]['zeroed'] += num_zeroed
                self.stats[name]['total'] += total

                # Log first injection
                if self.stats[name]['count'] == 1:
                    print(f"[DEADZONE] First injection on {name}: "
                          f"shape={tuple(tensor_to_modify.shape)}, "
                          f"zeroed={num_zeroed}/{total} ({zero_rate:.1f}%), "
                          f"threshold={self.config.deadzone_threshold}")

                if is_tuple:
                    return (modified,) + rest
                return modified

            # Handle MXFP4/NVFP4/FP8 specially (full fake quantization)
            if self.config.error_type in ('mxfp4', 'nvfp4', 'fp8'):
                if self.config.error_type == 'mxfp4':
                    modified, quant_stats = self._apply_mxfp4(tensor_to_modify)
                elif self.config.error_type == 'nvfp4':
                    modified, quant_stats = self._apply_nvfp4(tensor_to_modify)
                else:
                    modified, quant_stats = self._apply_fp8(tensor_to_modify)

                # Track statistics
                if name not in self.stats:
                    self.stats[name] = {
                        'count': 0, 'mean_error': 0.0, 'max_error': 0.0,
                        'relative_error': 0.0, 'point': 'output', 'type': self.config.error_type
                    }
                self.stats[name]['count'] += 1
                # Exponential moving average for error tracking
                alpha = 0.1
                self.stats[name]['mean_error'] = (
                    alpha * quant_stats['mean_error'] +
                    (1 - alpha) * self.stats[name]['mean_error']
                )
                self.stats[name]['max_error'] = max(
                    self.stats[name]['max_error'], quant_stats['max_error']
                )
                self.stats[name]['relative_error'] = (
                    alpha * quant_stats['relative_error'] +
                    (1 - alpha) * self.stats[name]['relative_error']
                )

                # Log first injection
                if self.stats[name]['count'] == 1:
                    print(f"[MXFP4] First injection on {name}: "
                          f"shape={tuple(tensor_to_modify.shape)}, "
                          f"mean_error={quant_stats['mean_error']:.2e}, "
                          f"rel_error={quant_stats['relative_error']*100:.1f}%")

                if is_tuple:
                    return (modified,) + rest
                return modified

            # Standard error injection (add error to output)
            error = self._compute_error(tensor_to_modify)
            modified = tensor_to_modify + error

            # Track statistics
            if name not in self.stats:
                self.stats[name] = {'count': 0, 'mean_error': 0.0, 'max_error': 0.0, 'point': 'output'}
            self.stats[name]['count'] += 1
            error_abs_mean = error.abs().mean().item()
            error_abs_max = error.abs().max().item()
            self.stats[name]['mean_error'] = (
                error_abs_mean * 0.1 +
                self.stats[name]['mean_error'] * 0.9
            )
            self.stats[name]['max_error'] = max(self.stats[name]['max_error'], error_abs_max)

            # Log first injection per module for verification
            if self.stats[name]['count'] == 1:
                rel_error = error_abs_mean / (tensor_to_modify.abs().mean().item() + 1e-10)
                print(f"[HW Error] First injection on {name}: "
                      f"output_shape={tuple(tensor_to_modify.shape)}, "
                      f"mean_error={error_abs_mean:.2e}, "
                      f"rel_error={rel_error:.2e}")

            if is_tuple:
                return (modified,) + rest
            return modified

        return hook

    def _create_forward_pre_hook(self, name: str) -> Callable:
        """
        Create forward pre-hook for INPUT injection.

        This is similar to quant_compute's fake quantization approach:
        - Original: y = operator(x)
        - With injection: y = operator(quant_dequant(x))
        """
        def hook(module: nn.Module, input: Tuple) -> Tuple:
            if not self._should_inject():
                return input

            # input is a tuple, typically (tensor,) or (tensor, other_args...)
            if len(input) == 0:
                return input

            first_input = input[0]
            if not isinstance(first_input, torch.Tensor):
                return input

            # Handle MXFP4/NVFP4/FP8 specially (full fake quantization)
            if self.config.error_type in ('mxfp4', 'nvfp4', 'fp8'):
                if self.config.error_type == 'mxfp4':
                    modified_input, quant_stats = self._apply_mxfp4(first_input)
                elif self.config.error_type == 'nvfp4':
                    modified_input, quant_stats = self._apply_nvfp4(first_input)
                else:
                    modified_input, quant_stats = self._apply_fp8(first_input)

                # Track statistics
                if name not in self.stats:
                    self.stats[name] = {
                        'count': 0, 'mean_error': 0.0, 'max_error': 0.0,
                        'relative_error': 0.0, 'point': 'input', 'type': self.config.error_type
                    }
                self.stats[name]['count'] += 1
                alpha = 0.1
                self.stats[name]['mean_error'] = (
                    alpha * quant_stats['mean_error'] +
                    (1 - alpha) * self.stats[name]['mean_error']
                )
                self.stats[name]['max_error'] = max(
                    self.stats[name]['max_error'], quant_stats['max_error']
                )
                self.stats[name]['relative_error'] = (
                    alpha * quant_stats['relative_error'] +
                    (1 - alpha) * self.stats[name]['relative_error']
                )

                # Log first injection
                if self.stats[name]['count'] == 1:
                    print(f"[MXFP4] First INPUT injection on {name}: "
                          f"shape={tuple(first_input.shape)}, "
                          f"mean_error={quant_stats['mean_error']:.2e}, "
                          f"rel_error={quant_stats['relative_error']*100:.1f}%")

                return (modified_input,) + input[1:]

            # Handle deadzone specially
            if self.config.error_type == 'deadzone':
                modified_input, deadzone_mask = self._apply_deadzone(first_input)
                num_zeroed = deadzone_mask.sum().item()
                total = deadzone_mask.numel()

                if name not in self.stats:
                    self.stats[name] = {'count': 0, 'zeroed': 0, 'total': 0, 'point': 'input', 'type': 'deadzone'}
                self.stats[name]['count'] += 1
                self.stats[name]['zeroed'] += num_zeroed
                self.stats[name]['total'] += total

                if self.stats[name]['count'] == 1:
                    print(f"[DEADZONE] First INPUT injection on {name}: "
                          f"shape={tuple(first_input.shape)}, "
                          f"zeroed={num_zeroed}/{total} ({num_zeroed/total*100:.1f}%)")

                return (modified_input,) + input[1:]

            # Standard error injection (add error to input)
            error = self._compute_error(first_input)
            modified_input = first_input + error

            # Track statistics
            if name not in self.stats:
                self.stats[name] = {'count': 0, 'mean_error': 0.0, 'max_error': 0.0, 'point': 'input'}
            self.stats[name]['count'] += 1
            error_abs_mean = error.abs().mean().item()
            error_abs_max = error.abs().max().item()
            self.stats[name]['mean_error'] = (
                error_abs_mean * 0.1 +
                self.stats[name]['mean_error'] * 0.9
            )
            self.stats[name]['max_error'] = max(self.stats[name]['max_error'], error_abs_max)

            # Log first injection per module for verification
            if self.stats[name]['count'] == 1:
                rel_error = error_abs_mean / (first_input.abs().mean().item() + 1e-10)
                print(f"[HW Error] First injection on {name}: "
                      f"input_shape={tuple(first_input.shape)}, "
                      f"mean_error={error_abs_mean:.2e}, "
                      f"rel_error={rel_error:.2e}")

            # Return modified input tuple
            return (modified_input,) + input[1:]

        return hook

    def _create_weight_quant_pre_hook(self, name: str) -> Callable:
        """
        Create forward pre-hook for WEIGHT quantization (W4A16 - QeRL style).

        This applies MXFP4 quantization to weight tensors:
        - Original: y = linear(x, W)
        - With injection: y = linear(x, quant_dequant(W))

        The weight is quantized before forward and restored after.
        """
        def hook(module: nn.Module, input: Tuple) -> None:
            if not self._should_inject():
                return

            # Only quantize if module has weight attribute
            if not hasattr(module, 'weight') or module.weight is None:
                return

            weight = module.weight.data

            # Apply MXFP4/NVFP4 quantization to weight
            if self.config.error_type in ('mxfp4', 'nvfp4'):
                quantizer = self._mxfp4_quantize if self.config.error_type == 'mxfp4' else self._nvfp4_quantize
                if quantizer is None:
                    return

                # Save original weight for restoration
                module._original_weight = weight.clone()

                # Quantize weight (use column-wise blocking for NVFP4 to match quant_compute reference)
                if self.config.error_type == 'mxfp4':
                    quantized_weight, quant_stats = self._apply_mxfp4(weight)
                else:
                    quantized_weight, quant_stats = self._apply_nvfp4(weight, is_weight=True)
                module.weight.data = quantized_weight

                # Track statistics
                if name not in self.stats:
                    self.stats[name] = {
                        'count': 0, 'mean_error': 0.0, 'max_error': 0.0,
                        'relative_error': 0.0, 'point': 'weight', 'type': self.config.error_type
                    }
                self.stats[name]['count'] += 1
                alpha = 0.1
                self.stats[name]['mean_error'] = (
                    alpha * quant_stats['mean_error'] +
                    (1 - alpha) * self.stats[name]['mean_error']
                )
                self.stats[name]['max_error'] = max(
                    self.stats[name]['max_error'], quant_stats['max_error']
                )
                self.stats[name]['relative_error'] = (
                    alpha * quant_stats['relative_error'] +
                    (1 - alpha) * self.stats[name]['relative_error']
                )

                # Log first injection
                if self.stats[name]['count'] == 1:
                    print(f"[MXFP4-W4A16] First WEIGHT quant on {name}: "
                          f"shape={tuple(weight.shape)}, "
                          f"mean_error={quant_stats['mean_error']:.2e}, "
                          f"rel_error={quant_stats['relative_error']*100:.1f}%")

        return hook

    def _create_weight_restore_hook(self, name: str) -> Callable:
        """
        Create forward hook to restore original weight after forward pass.

        WARNING: This is the LEGACY (broken) implementation!
        Restoring weights after forward but before backward means grad_x uses W_orig
        instead of W_quant, which corrupts gradient flow for FullFT training.

        Only used when use_ste=False (not recommended for FullFT).
        """
        def hook(module: nn.Module, input: Tuple, output) -> None:
            if hasattr(module, '_original_weight'):
                module.weight.data = module._original_weight
                del module._original_weight

        return hook

    def _create_weight_restore_backward_hook(self, name: str) -> Callable:
        """
        Create BACKWARD hook to restore original weight AFTER backward pass.

        This is the CORRECT STE implementation for FullFT training:
        1. Pre-hook: W_orig -> W_quant (before forward)
        2. Forward: y = Linear(x, W_quant)
        3. Backward: grad_x = grad_output @ W_quant (CORRECT!)
        4. This hook: W_quant -> W_orig (after backward, for next iteration)

        The key difference from legacy mode:
        - Legacy: restore in forward_hook (before backward) -> grad_x uses W_orig (WRONG)
        - STE: restore in backward_hook (after backward) -> grad_x uses W_quant (CORRECT)
        """
        def hook(module: nn.Module, grad_input: Tuple, grad_output: Tuple) -> None:
            if hasattr(module, '_original_weight'):
                module.weight.data = module._original_weight
                del module._original_weight

        return hook

    def _create_activation_quant_pre_hook(self, name: str) -> Callable:
        """
        Create forward pre-hook for ACTIVATION quantization (for W4A4 mode).

        This applies MXFP4 quantization to input activations:
        - Original: y = linear(x, W_quant)
        - W4A4: y = linear(quant_dequant(x), W_quant)

        Used in conjunction with weight quantization hook for full W4A4.
        """
        def hook(module: nn.Module, input: Tuple) -> Tuple:
            if not self._should_inject():
                return input

            if len(input) == 0:
                return input

            first_input = input[0]
            if not isinstance(first_input, torch.Tensor):
                return input

            # Apply MXFP4/NVFP4 quantization to activation
            if self.config.error_type in ('mxfp4', 'nvfp4'):
                if self.config.error_type == 'mxfp4':
                    modified_input, quant_stats = self._apply_mxfp4(first_input)
                else:
                    modified_input, quant_stats = self._apply_nvfp4(first_input)

                # Track statistics (separate key for activation)
                act_name = f"{name}_activation"
                if act_name not in self.stats:
                    self.stats[act_name] = {
                        'count': 0, 'mean_error': 0.0, 'max_error': 0.0,
                        'relative_error': 0.0, 'point': 'activation', 'type': self.config.error_type
                    }
                self.stats[act_name]['count'] += 1
                alpha = 0.1
                self.stats[act_name]['mean_error'] = (
                    alpha * quant_stats['mean_error'] +
                    (1 - alpha) * self.stats[act_name]['mean_error']
                )
                self.stats[act_name]['max_error'] = max(
                    self.stats[act_name]['max_error'], quant_stats['max_error']
                )
                self.stats[act_name]['relative_error'] = (
                    alpha * quant_stats['relative_error'] +
                    (1 - alpha) * self.stats[act_name]['relative_error']
                )

                # Log first injection
                if self.stats[act_name]['count'] == 1:
                    print(f"[W4A4] First ACTIVATION quant on {name}: "
                          f"shape={tuple(first_input.shape)}, "
                          f"mean_error={quant_stats['mean_error']:.2e}, "
                          f"rel_error={quant_stats['relative_error']*100:.1f}%")

                return (modified_input,) + input[1:]

            return input

        return hook

    def register_hooks(self, model: nn.Module, verbose: bool = True) -> int:
        """
        Register forward hooks on target modules.

        Args:
            model: Model to inject errors into
            verbose: Print registration info

        Returns:
            Number of hooks registered
        """
        self.remove_hooks()  # Clear any existing hooks
        self.stats.clear()

        count = 0
        for name, module in model.named_modules():
            if self._is_target_module(name, module):
                # Choose hook type based on injection point
                if self.config.injection_point == 'weight':
                    # W4A16 mode: quantize weights, keep activations FP16 (QeRL style)
                    # Pre-hook quantizes weight before forward
                    pre_hook = module.register_forward_pre_hook(self._create_weight_quant_pre_hook(name))
                    self.hooks.append(pre_hook)

                    if self.config.use_ste:
                        # STE mode (default): Restore weight AFTER backward
                        # This ensures grad_x = grad_output @ W_quant (correct!)
                        backward_hook = module.register_full_backward_hook(
                            self._create_weight_restore_backward_hook(name)
                        )
                        self.hooks.append(backward_hook)
                        if verbose and count == 0:
                            logger.info("[STE] Using backward hook for proper QAT gradient flow")
                    else:
                        # Legacy mode (broken for FullFT): Restore weight after forward
                        # WARNING: grad_x = grad_output @ W_orig (wrong!)
                        post_hook = module.register_forward_hook(self._create_weight_restore_hook(name))
                        self.hooks.append(post_hook)
                        if verbose and count == 0:
                            logger.warning("[LEGACY] Using forward hook - NOT recommended for FullFT!")
                elif self.config.injection_point == 'input':
                    # W16A4 mode: quantize input activations (our previous approach)
                    # Pre-hook: inject error to input BEFORE operator processes it
                    # Like quant_compute's fake quantization: y = operator(fake_quant(x))
                    hook = module.register_forward_pre_hook(self._create_forward_pre_hook(name))
                    self.hooks.append(hook)
                elif self.config.injection_point == 'both':
                    # W4A4 mode: quantize BOTH weights AND activations
                    # This combines weight quantization (W4A16) + activation quantization
                    #
                    # IMPORTANT: Activation quantization uses POST-hook (output quantization)
                    # to avoid quantizing sensitive inputs like RMSNorm outputs!
                    #
                    # Correct W4A4 flow:
                    #   RMSNorm output (FP16) -> Linear with W4 weights -> Output (FP16) -> Quantize to A4
                    #
                    # Wrong approach (breaks model):
                    #   RMSNorm output (FP16) -> Quantize to A4 -> Linear with W4 weights (destroys normalization!)

                    # 1. Weight quantization pre-hook
                    pre_hook = module.register_forward_pre_hook(self._create_weight_quant_pre_hook(name))
                    self.hooks.append(pre_hook)

                    # 2. Weight restoration (STE mode)
                    if self.config.use_ste:
                        backward_hook = module.register_full_backward_hook(
                            self._create_weight_restore_backward_hook(name)
                        )
                        self.hooks.append(backward_hook)
                        if verbose and count == 0:
                            logger.info("[W4A4-STE] Using backward hook for proper QAT gradient flow")
                    else:
                        post_hook = module.register_forward_hook(self._create_weight_restore_hook(name))
                        self.hooks.append(post_hook)
                        if verbose and count == 0:
                            logger.warning("[W4A4-LEGACY] Using forward hook - NOT recommended for FullFT!")

                    # 3. Activation quantization PRE-hook (quantize INPUT activations)
                    #    This is TRUE W4A4: y = W_quant @ x_quant
                    #    QeRL uses this approach and achieves ~60% accuracy on GSM8K
                    #
                    #    Previous POST-hook approach was WRONG:
                    #    - POST-hook quantizes OUTPUT: y_quant = quant(W_quant @ x_fp16)
                    #    - This means input x remains FP16, not true W4A4!
                    act_hook = module.register_forward_pre_hook(self._create_activation_quant_pre_hook(name))
                    self.hooks.append(act_hook)

                    if verbose and count == 0:
                        logger.info("[W4A4] Quantizing weights (pre-hook) and INPUT activations (pre-hook) - TRUE W4A4")
                else:
                    # Output mode: inject error to output AFTER operator processes it
                    # Post-hook: Like: y = operator(x) + error
                    hook = module.register_forward_hook(self._create_forward_hook(name))
                    self.hooks.append(hook)

                count += 1

                if verbose and count <= 5:
                    logger.info(f"[HW Error] Registered {self.config.injection_point} hook on: {name} ({module.__class__.__name__})")

        if verbose:
            rank = 0
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            if rank == 0:
                ste_info = f", use_ste={self.config.use_ste}" if self.config.injection_point == 'weight' else ""
                print(f"[HW Error] Registered {count} {self.config.injection_point} hooks "
                      f"(scale={self.config.error_scale}, type={self.config.error_type}, "
                      f"targets={self.config.target_modules}{ste_info})")

        return count

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_stats(self) -> Dict[str, Dict]:
        """Get injection statistics."""
        return self.stats.copy()

    def print_stats(self):
        """Print injection statistics summary."""
        if not self.stats:
            print("[HW Error] No injection statistics available")
            return

        if self.config.error_type == 'deadzone':
            print(f"\n[DEADZONE] Injection Statistics (threshold={self.config.deadzone_threshold}):")
            print("-" * 60)
            total_zeroed = 0
            total_values = 0
            for name, stat in sorted(self.stats.items()):
                zero_rate = stat['zeroed'] / stat['total'] * 100 if stat['total'] > 0 else 0
                print(f"  {name}: count={stat['count']}, zeroed={stat['zeroed']:,}/{stat['total']:,} ({zero_rate:.1f}%)")
                total_zeroed += stat['zeroed']
                total_values += stat['total']
            overall_rate = total_zeroed / total_values * 100 if total_values > 0 else 0
            print("-" * 60)
            print(f"  TOTAL: {total_zeroed:,}/{total_values:,} ({overall_rate:.1f}%) values zeroed")
        elif self.config.error_type in ('mxfp4', 'nvfp4'):
            quant_type = self.config.error_type.upper()
            print(f"\n[{quant_type}] Injection Statistics:")
            print("-" * 60)
            total_count = 0
            avg_rel_error = 0.0
            for name, stat in sorted(self.stats.items()):
                print(f"  {name}: count={stat['count']}, "
                      f"mean_err={stat['mean_error']:.2e}, "
                      f"rel_err={stat['relative_error']*100:.1f}%")
                total_count += stat['count']
                avg_rel_error += stat['relative_error']
            if self.stats:
                avg_rel_error /= len(self.stats)
            print("-" * 60)
            print(f"  TOTAL: {total_count} injections, avg_rel_error={avg_rel_error*100:.1f}%")
        else:
            print(f"\n[HW Error] Injection Statistics (scale={self.config.error_scale}):")
            print("-" * 60)
            for name, stat in sorted(self.stats.items()):
                print(f"  {name}: count={stat['count']}, mean_error={stat['mean_error']:.2e}")
            print("-" * 60)


def create_hw_error_injector(
    enabled: bool = False,
    error_scale: float = 1e-5,
    error_type: str = 'relative_gaussian',
    injection_point: str = 'input',
    target_modules: Optional[List[str]] = None,
    apply_during: str = 'both',
) -> HWErrorInjector:
    """
    Factory function to create HW error injector.

    Args:
        enabled: Whether to enable error injection
        error_scale: Scale of error (1e-5 recommended for testing)
        error_type: 'relative_gaussian', 'absolute_gaussian', or 'systematic_bias'
        injection_point: Where to inject error
            - 'input': Inject to operator INPUT (like fake quantization in quant_compute)
                      y = operator(x + error)
            - 'output': Inject to operator OUTPUT
                      y = operator(x) + error
        target_modules: Module patterns to target (default: ['rmsnorm'])
        apply_during: 'rollout', 'training', or 'both'

    Returns:
        Configured HWErrorInjector instance

    Example:
        >>> # Input injection (like fake quantization)
        >>> injector = create_hw_error_injector(
        ...     enabled=True,
        ...     error_scale=1e-5,
        ...     injection_point='input',
        ...     target_modules=['rmsnorm', 'down_proj']
        ... )
        >>> injector.register_hooks(model)
    """
    config = HWErrorConfig(
        enabled=enabled,
        error_scale=error_scale,
        error_type=error_type,
        injection_point=injection_point,
        target_modules=target_modules,
        apply_during=apply_during,
    )
    return HWErrorInjector(config)


# Convenience function for quick testing
def inject_hw_error_once(
    tensor: torch.Tensor,
    error_scale: float = 1e-5,
    error_type: str = 'relative_gaussian',
) -> torch.Tensor:
    """
    Inject HW error into a single tensor (for testing/debugging).

    Args:
        tensor: Input tensor
        error_scale: Error scale
        error_type: Error type

    Returns:
        Tensor with injected error

    Example:
        >>> x = torch.randn(32, 2048)
        >>> x_noisy = inject_hw_error_once(x, error_scale=1e-4)
        >>> print(f"Relative error: {(x_noisy - x).abs().mean() / x.abs().mean():.2e}")
    """
    config = HWErrorConfig(enabled=True, error_scale=error_scale, error_type=error_type)
    injector = HWErrorInjector(config)
    return tensor + injector._compute_error(tensor)


__all__ = [
    'HWErrorConfig',
    'HWErrorInjector',
    'STEQuantizeWeight',
    'STEQuantizeActivation',
    'create_hw_error_injector',
    'inject_hw_error_once',
]
