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
Noise injection for 4-bit quantized model training.
Adapted from QeRL for veRL with MoE expert-only targeting.
"""

import torch
import logging

logger = logging.getLogger(__name__)


def get_sigma_by_step(step, total_steps, sigma_trend):
    """
    Calculate noise standard deviation based on current training step.

    Args:
        step: Current training step
        total_steps: Total number of training steps
        sigma_trend: List of sigma values for each interval

    Returns:
        Tuple of (sigma_id, sigma value)
    """
    step = min(step, total_steps)

    num_intervals = len(sigma_trend) + 1
    steps_per_interval = total_steps / num_intervals

    interval_id = int(step // steps_per_interval)

    if interval_id == 0:
        return interval_id, 0

    sigma_id = interval_id - 1
    sigma_id = min(sigma_id, len(sigma_trend) - 1)

    sigma = sigma_trend[sigma_id]
    return sigma_id, sigma


def _detect_moe_model(model):
    """
    Auto-detect if model is MoE by checking for expert modules.

    Returns:
        True if MoE model detected, False for dense models

    Note:
        We specifically check for 'experts' (plural) to distinguish from
        dense models that have 'gate_proj' in SwiGLU MLP (not MoE gate).
        We also check for 'shared_expert' which is specific to MoE.
    """
    # More specific MoE indicators (avoid false positives from gate_proj)
    moe_indicators = ['experts', 'shared_expert', 'num_experts', 'expert_']
    for name, _ in model.named_modules():
        name_lower = name.lower()
        if any(indicator in name_lower for indicator in moe_indicators):
            return True
    return False


def generate_expert_gaussian_noise(model, step, total_step, sigma_trend, target_modules=None, exclude_patterns=None, is_moe=None, verbose=True):
    """
    Generate and apply Gaussian noise to RMSNorm layers in quantized models.

    Behavior differs based on model type:

    For MoE models (is_moe=True):
        - Only targets post_attention_layernorm (after router/MLP)
        - Excludes input_layernorm (before router)
        - Avoids affecting router decisions (expert selection mismatch)

    For Dense models (is_moe=False):
        - Targets ALL RMSNorm layers (QeRL original behavior)
        - No exclusions
        - Follows QeRL paper's approach for quantized training

    Args:
        model: The model to inject noise into (typically vLLM inference model)
        step: Current training step
        total_step: Total training steps
        sigma_trend: List of sigma values for exponential decay schedule
        target_modules: List of module name patterns to target (None = auto based on is_moe)
        exclude_patterns: List of patterns to exclude (None = auto based on is_moe)
        is_moe: Whether model is MoE (None = auto-detect)
        verbose: Whether to print noise injection info

    Example:
        >>> # Auto-detect model type (recommended)
        >>> generate_expert_gaussian_noise(model, step=100, total_step=1000, sigma_trend=sigma_trend)
        >>>
        >>> # Force dense model behavior (QeRL original)
        >>> generate_expert_gaussian_noise(model, step=100, total_step=1000, sigma_trend=sigma_trend, is_moe=False)
        >>>
        >>> # Force MoE model behavior
        >>> generate_expert_gaussian_noise(model, step=100, total_step=1000, sigma_trend=sigma_trend, is_moe=True)
    """
    # Auto-detect model type if not specified
    if is_moe is None:
        is_moe = _detect_moe_model(model)

    # Set defaults based on model type
    if target_modules is None:
        if is_moe:
            target_modules = ["post_attention_layernorm"]  # MoE: only post-attention norms
        else:
            target_modules = []  # Dense: all RMSNorm (QeRL original)

    if exclude_patterns is None:
        if is_moe:
            exclude_patterns = ["input_layernorm"]  # MoE: exclude input norms (before router)
        else:
            exclude_patterns = []  # Dense: no exclusions (QeRL original)

    sigma_id, sigma = get_sigma_by_step(step, total_step, sigma_trend)

    if verbose and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            model_type = "MoE" if is_moe else "Dense"
            target_desc = target_modules if target_modules else "ALL RMSNorm"
            # Use print() to ensure visibility regardless of logging level
            print(f"[AQN] Noise injection ({model_type}) - Step: {step}/{total_step}, Sigma ID: {sigma_id}, Sigma: {sigma:.6f}, Target: {target_desc}")

    if sigma == 0:
        return

    # Detect RMSNorm by class name (works with vLLM's model implementations)
    # vLLM uses its own model classes, not transformers', so we check class name
    def _is_rmsnorm(module):
        """Check if module is an RMSNorm layer by class name and structure."""
        class_name = module.__class__.__name__.lower()
        # Check for RMSNorm in class name (handles Qwen2RMSNorm, LlamaRMSNorm, etc.)
        if 'rmsnorm' not in class_name:
            return False
        # Verify it has a weight parameter (all RMSNorm layers have this)
        if not hasattr(module, 'weight'):
            return False
        if not isinstance(module.weight, torch.Tensor):
            return False
        return True

    noise_count = 0
    skipped_count = 0

    for name, module in model.named_modules():
        # Skip if not RMSNorm (use class name detection for vLLM compatibility)
        if not _is_rmsnorm(module):
            continue

        # Check exclusion patterns first
        if exclude_patterns and any(pattern in name for pattern in exclude_patterns):
            skipped_count += 1
            continue

        # Check target patterns
        if target_modules and not any(pattern in name for pattern in target_modules):
            skipped_count += 1
            continue

        weight_tensor = module.weight

        # Generate noise
        noise = torch.normal(
            mean=0,
            std=sigma,
            size=weight_tensor.shape,
            dtype=torch.float32
        ).to(weight_tensor.device)

        # Convert to module dtype
        noise = noise.to(weight_tensor.dtype)

        # Apply noise in-place
        with torch.no_grad():
            module.weight.add_(noise)

        noise_count += 1

        if verbose and noise_count <= 3:  # Log first few for debugging
            logger.debug(f"Applied noise to: {name}")

    if verbose and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            # Use print() to ensure visibility regardless of logging level
            print(f"[AQN] Applied noise to {noise_count} RMSNorm layers (skipped {skipped_count} excluded layers)")


def get_sigma_schedule(sigma_start=0.01, sigma_end=0.001, num_stages=10):
    """
    Generate exponential decay schedule for noise standard deviation.

    Args:
        sigma_start: Starting sigma value
        sigma_end: Ending sigma value
        num_stages: Number of stages (intervals + 1)

    Returns:
        List of sigma values

    Example:
        >>> schedule = get_sigma_schedule(0.01, 0.001, 10)
        >>> # Returns [0.01, 0.0075, 0.0056, ..., 0.001] (9 values for 10 stages)
    """
    import numpy as np

    if num_stages <= 1:
        return []

    sigma_decay_schedule_np = sigma_start * (sigma_end / sigma_start) ** (
        np.arange(num_stages - 1) / (num_stages - 2)
    )
    return sigma_decay_schedule_np.tolist()


def get_epoch_aware_sigma_schedule(sigma_start=0.05, sigma_end=0.0005, num_epochs=2, stages_per_epoch=5):
    """
    Generate epoch-aware sigma schedule (Option C).

    Each epoch has its own sigma range with exponential decay within the epoch.
    This ensures meaningful noise levels throughout all epochs.

    Args:
        sigma_start: Starting sigma for epoch 1
        sigma_end: Ending sigma for final epoch
        num_epochs: Number of training epochs
        stages_per_epoch: Number of decay stages per epoch

    Returns:
        List of (epoch_ranges, stages_per_epoch) where epoch_ranges is list of (start, end) tuples

    Example for 2 epochs:
        Epoch 1: 0.05 → 0.01 (exploration)
        Epoch 2: 0.01 → 0.0005 (refinement)
    """
    import numpy as np

    if num_epochs <= 0:
        return [], stages_per_epoch

    # Calculate intermediate sigma values at epoch boundaries
    # Using exponential decay: sigma_i = sigma_start * (sigma_end/sigma_start)^(i/num_epochs)
    epoch_boundaries = [
        sigma_start * (sigma_end / sigma_start) ** (i / num_epochs)
        for i in range(num_epochs + 1)
    ]

    # Create ranges for each epoch
    epoch_ranges = [
        (epoch_boundaries[i], epoch_boundaries[i + 1])
        for i in range(num_epochs)
    ]

    return epoch_ranges, stages_per_epoch


def get_sigma_by_step_epoch_aware(step, steps_per_epoch, epoch_ranges, stages_per_epoch, current_epoch=None):
    """
    Calculate sigma for epoch-aware decay schedule (Option C).

    Args:
        step: Current global training step
        steps_per_epoch: Number of steps in one epoch
        epoch_ranges: List of (sigma_start, sigma_end) tuples per epoch
        stages_per_epoch: Number of decay stages per epoch
        current_epoch: Current epoch (0-indexed), auto-calculated if None

    Returns:
        Tuple of (stage_id, sigma value)
    """
    import numpy as np

    if not epoch_ranges:
        return 0, 0

    # Calculate current epoch if not provided
    if current_epoch is None:
        current_epoch = min(step // steps_per_epoch, len(epoch_ranges) - 1)

    # Get sigma range for current epoch
    sigma_start, sigma_end = epoch_ranges[current_epoch]

    # Calculate step within current epoch
    step_in_epoch = step % steps_per_epoch

    # Calculate intervals within epoch (including warmup)
    num_intervals = stages_per_epoch + 1  # +1 for warmup
    steps_per_interval = steps_per_epoch / num_intervals

    interval_id = int(step_in_epoch // steps_per_interval)

    # Warmup: first interval of each epoch has sigma=0
    if interval_id == 0:
        return (current_epoch * stages_per_epoch), 0

    # Calculate sigma with exponential decay within epoch
    stage_in_epoch = interval_id - 1
    stage_in_epoch = min(stage_in_epoch, stages_per_epoch - 1)

    if stages_per_epoch <= 1:
        sigma = sigma_start
    else:
        # Exponential decay within epoch
        decay_factor = (sigma_end / sigma_start) ** (stage_in_epoch / (stages_per_epoch - 1))
        sigma = sigma_start * decay_factor

    global_stage_id = current_epoch * stages_per_epoch + stage_in_epoch
    return global_stage_id, sigma


__all__ = [
    "generate_expert_gaussian_noise",
    "get_sigma_by_step",
    "get_sigma_schedule",
    "get_epoch_aware_sigma_schedule",
    "get_sigma_by_step_epoch_aware",
]
