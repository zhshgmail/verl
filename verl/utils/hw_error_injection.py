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


class HWErrorConfig:
    """Configuration for HW error injection."""

    def __init__(
        self,
        enabled: bool = False,
        error_scale: float = 1e-5,
        error_type: str = 'relative_gaussian',
        injection_point: str = 'input',  # 'input' or 'output'
        target_modules: Optional[List[str]] = None,
        apply_during: str = 'both',  # 'rollout', 'training', 'both'
        seed: Optional[int] = None,
    ):
        """
        Args:
            enabled: Whether to enable error injection
            error_scale: Scale of the error (relative to tensor magnitude)
                - 1e-6: ~BF16 precision level (very small)
                - 1e-5: Moderate (recommended starting point)
                - 1e-4: Aggressive (may cause training instability)
            error_type: Type of error to inject
                - 'relative_gaussian': error = randn() * |tensor| * scale
                - 'absolute_gaussian': error = randn() * scale
                - 'systematic_bias': error = sign(tensor) * scale (simulates rounding bias)
            injection_point: Where to inject error (like quant_compute fake quantization)
                - 'input': Inject error to operator INPUT (before computation)
                          Like fake quantization: y = operator(x + error)
                          Simulates error in data representation
                - 'output': Inject error to operator OUTPUT (after computation)
                          Like: y = operator(x) + error
                          Simulates error in operator's HW implementation
            target_modules: List of module name patterns to target
                - Default: ['rmsnorm'] - only RMSNorm layers
                - Options: ['rmsnorm', 'down_proj', 'o_proj', 'linear']
            apply_during: When to apply error injection
                - 'rollout': Only during vLLM inference
                - 'training': Only during FSDP training
                - 'both': During both phases
            seed: Random seed for reproducibility (None = random)
        """
        self.enabled = enabled
        self.error_scale = error_scale
        self.error_type = error_type
        self.injection_point = injection_point
        self.target_modules = target_modules or ['rmsnorm']
        self.apply_during = apply_during
        self.seed = seed

    def __repr__(self):
        return (f"HWErrorConfig(enabled={self.enabled}, scale={self.error_scale}, "
                f"type={self.error_type}, targets={self.target_modules})")


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

        else:
            raise ValueError(f"Unknown error type: {self.config.error_type}")

        return error.to(output.dtype)

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        """Check if module should have error injection applied."""
        name_lower = name.lower()
        class_name = module.__class__.__name__.lower()

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
                # Only modify the first tensor (main output)
                error = self._compute_error(output[0])
                modified = output[0] + error
                tensor_for_stats = output[0]
                result = (modified,) + output[1:]
            else:
                error = self._compute_error(output)
                modified = output + error
                tensor_for_stats = output
                result = modified

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
                rel_error = error_abs_mean / (tensor_for_stats.abs().mean().item() + 1e-10)
                print(f"[HW Error] First injection on {name}: "
                      f"output_shape={tuple(tensor_for_stats.shape)}, "
                      f"mean_error={error_abs_mean:.2e}, "
                      f"rel_error={rel_error:.2e}")

            return result

        return hook

    def _create_forward_pre_hook(self, name: str) -> Callable:
        """
        Create forward pre-hook for INPUT injection.

        This is similar to quant_compute's fake quantization approach:
        - Original: y = operator(x)
        - With injection: y = operator(x + error)
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
                if self.config.injection_point == 'input':
                    # Pre-hook: inject error to input BEFORE operator processes it
                    # Like quant_compute's fake quantization: y = operator(fake_quant(x))
                    hook = module.register_forward_pre_hook(self._create_forward_pre_hook(name))
                else:
                    # Post-hook: inject error to output AFTER operator processes it
                    # Like: y = operator(x) + error
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
                print(f"[HW Error] Registered {count} {self.config.injection_point} hooks "
                      f"(scale={self.config.error_scale}, type={self.config.error_type}, "
                      f"targets={self.config.target_modules})")

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
    'create_hw_error_injector',
    'inject_hw_error_once',
]
