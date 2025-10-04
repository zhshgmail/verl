# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
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
Decoupled PPO Loss for Asynchronous RL Training

Implements the decoupled PPO objective from AReaL paper (arXiv:2505.24298).

The key idea is to split the importance ratio into two parts:
1. Current policy vs. proximal policy (for PPO clipping)
2. Proximal policy vs. behavior policy (for staleness weighting)

This allows handling off-policy samples while maintaining PPO's stability.
"""

import torch
from typing import Dict, Optional, Tuple


def ppo_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    loss_mask: torch.Tensor,
    eps_clip_higher: Optional[float] = None,
    c_clip: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute standard PPO actor loss.

    Standard PPO objective:
        L = E[min(r * A, clip(r, 1-ε, 1+ε) * A)]
        where r = π_θ / π_old

    Args:
        logprobs: Log probabilities from current policy π_θ. Shape: [bs, max_seqlen]
        old_logprobs: Log probabilities from behavior policy π_old. Shape: [bs, max_seqlen]
        advantages: Advantage estimates. Shape: [bs, max_seqlen]
        eps_clip: Clip ratio for lower bound (e.g., 0.2 means clip to [0.8, ...])
        loss_mask: Mask indicating valid tokens (1 for valid, 0 for padding). Shape: [bs, max_seqlen]
        eps_clip_higher: Optional separate clip ratio for upper bound
        c_clip: Optional dual clip parameter (see PPO-penalty variant)

    Returns:
        loss: Scalar loss value
        stats: Dictionary with statistics for logging:
            - loss: Per-token loss values
            - importance_weight: Importance ratio (r)
            - approx_kl: Approximate KL divergence
            - clip_mask: Which tokens were clipped
            - dual_clip_mask: Which tokens were dual-clipped (if c_clip used)
    """
    return _compute_ppo_loss_impl(
        logprobs=logprobs,
        proximal_logprobs=old_logprobs,  # Standard PPO: proximal = old
        old_logprobs=old_logprobs,
        advantages=advantages,
        eps_clip=eps_clip,
        loss_mask=loss_mask,
        eps_clip_higher=eps_clip_higher,
        c_clip=c_clip,
        behav_imp_weight_cap=None,
        behav_imp_weight_floor=None,
        apply_behav_weighting=False,  # Standard PPO: no behavior weighting
    )


def decoupled_ppo_loss(
    logprobs: torch.Tensor,
    proximal_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    loss_mask: torch.Tensor,
    eps_clip_higher: Optional[float] = None,
    c_clip: Optional[float] = None,
    behav_imp_weight_cap: Optional[float] = None,
    behav_imp_weight_floor: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute decoupled PPO actor loss for asynchronous off-policy training.

    Decoupled PPO objective (from AReaL):
        L = E[w_behav * min(r * A, clip(r, 1-ε, 1+ε) * A)]
        where:
        - r = π_θ / π_prox  (ratio against proximal policy for PPO clipping)
        - w_behav = π_prox / π_old  (importance weight for stale behavior policy)

    This separates:
    1. PPO clipping (current vs proximal) - for stability
    2. Importance weighting (proximal vs behavior) - for staleness correction

    Args:
        logprobs: Log probabilities from current policy π_θ. Shape: [bs, max_seqlen]
        proximal_logprobs: Log probabilities from proximal policy π_prox (recent checkpoint
            used for regularization). Shape: [bs, max_seqlen]
        old_logprobs: Log probabilities from behavior policy π_old (policy that generated
            the data, may be stale). Shape: [bs, max_seqlen]
        advantages: Advantage estimates. Shape: [bs, max_seqlen]
        eps_clip: Clip ratio for lower bound (e.g., 0.2 means clip to [0.8, ...])
        loss_mask: Mask indicating valid tokens (1 for valid, 0 for padding). Shape: [bs, max_seqlen]
        eps_clip_higher: Optional separate clip ratio for upper bound
        c_clip: Optional dual clip parameter (see PPO-penalty variant)
        behav_imp_weight_cap: Maximum allowed behavior importance weight (e.g., 5.0).
            Samples with w_behav > cap are excluded. Default: None (no upper bound).
            AReaL typically uses 5.0.
        behav_imp_weight_floor: Minimum allowed behavior importance weight (e.g., 0.2).
            Samples with w_behav < floor are excluded. Default: None (no lower bound).
            For symmetric clipping, use 1/cap (e.g., floor=0.2 for cap=5.0).
            Note: AReaL does not use lower bounds.

    Returns:
        loss: Scalar loss value
        stats: Dictionary with statistics for logging:
            - loss: Per-token loss values
            - importance_weight: Current/proximal importance ratio (r)
            - approx_kl: Approximate KL between current and proximal policy
            - clip_mask: Which tokens were clipped
            - behave_imp_weight: Behavior importance weights (w_behav)
            - behave_approx_kl: Approximate KL between proximal and behavior policy
            - behave_mask: Which tokens passed the importance weight cap
            - staleness_fraction: Fraction of tokens with significant staleness
            - max_behav_imp_weight: Maximum behavior importance weight
            - mean_behav_imp_weight: Mean behavior importance weight
    """
    return _compute_ppo_loss_impl(
        logprobs=logprobs,
        proximal_logprobs=proximal_logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        eps_clip=eps_clip,
        loss_mask=loss_mask,
        eps_clip_higher=eps_clip_higher,
        c_clip=c_clip,
        behav_imp_weight_cap=behav_imp_weight_cap,
        behav_imp_weight_floor=behav_imp_weight_floor,
        apply_behav_weighting=True,  # Decoupled PPO: apply behavior weighting
    )


def _compute_ppo_loss_impl(
    logprobs: torch.Tensor,
    proximal_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    loss_mask: torch.Tensor,
    eps_clip_higher: Optional[float],
    c_clip: Optional[float],
    behav_imp_weight_cap: Optional[float],
    behav_imp_weight_floor: Optional[float],
    apply_behav_weighting: bool,
) -> Tuple[torch.Tensor, Dict]:
    """
    Internal implementation shared by both ppo_loss() and decoupled_ppo_loss().

    This function contains the actual computation logic, while the public functions
    provide clean, focused interfaces.
    """

    # Compute the number of valid tokens for normalization
    loss_mask_count = loss_mask.count_nonzero() or 1

    # Convert loss_mask to boolean if needed
    loss_mask_bool = loss_mask.bool()

    # Compute importance ratio: r = π_θ(a|s) / π_prox(a|s)
    # This is the ratio used for PPO clipping
    ratio = torch.where(loss_mask_bool, torch.exp(logprobs - proximal_logprobs), 0)

    # Apply PPO clipping to the ratio
    clipped_ratio = torch.clamp(
        ratio,
        1.0 - eps_clip,
        1.0 + (eps_clip if eps_clip_higher is None else eps_clip_higher),
    )

    # Compute policy gradient loss: L = -A * r (or -A * clip(r))
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    clip_mask = pg_loss1.detach() < pg_loss2.detach()  # Track which were clipped
    pg_loss = torch.max(pg_loss1, pg_loss2)

    # Optional: Apply dual clipping (PPO-penalty variant)
    if c_clip is not None:
        assert c_clip > 1.0, f"c_clip must be > 1.0, got {c_clip}"
        pg_loss3 = torch.sign(advantages) * c_clip * advantages
        dual_clip_mask = pg_loss3.detach() < pg_loss.detach()
        pg_loss = torch.min(pg_loss, pg_loss3)
    else:
        dual_clip_mask = torch.zeros_like(clip_mask)

    # Compute behavior importance weight: w_behav = π_prox(a|s) / π_old(a|s)
    # This captures how much the policy has changed since data was generated
    behav_kl = proximal_logprobs - old_logprobs
    behav_imp_weight = behav_kl.exp()

    # Apply importance weight bounds to limit impact of very stale or irrelevant samples
    if behav_imp_weight_cap is not None or behav_imp_weight_floor is not None:
        # Upper bound: reject samples where proximal policy is much better than behavior policy
        if behav_imp_weight_cap is not None:
            upper_mask = behav_imp_weight <= behav_imp_weight_cap
        else:
            upper_mask = torch.ones_like(behav_imp_weight, dtype=torch.bool)

        # Lower bound: reject samples where proximal policy is much worse than behavior policy
        if behav_imp_weight_floor is not None:
            lower_mask = behav_imp_weight >= behav_imp_weight_floor
        else:
            lower_mask = torch.ones_like(behav_imp_weight, dtype=torch.bool)

        # Combine bounds and loss mask
        behav_mask = upper_mask.logical_and(lower_mask).logical_and(loss_mask_bool)
    else:
        behav_mask = loss_mask_bool

    # Zero out importance weights for tokens that are outside the bounds
    behav_kl = torch.where(behav_mask, behav_kl, 0.0)
    behav_imp_weight = torch.where(behav_mask, behav_imp_weight, 0.0)

    # Apply behavior importance weighting to the loss
    # This is the key difference from standard PPO!
    if apply_behav_weighting:
        pg_loss = pg_loss * behav_imp_weight

    # Compute final loss (average over valid tokens)
    logging_loss = pg_loss.detach()
    pg_loss = torch.where(loss_mask_bool, pg_loss, 0).sum() / loss_mask_count

    # Update masks to only include valid tokens
    clip_mask.logical_and_(loss_mask_bool)
    dual_clip_mask.logical_and_(loss_mask_bool)

    # Collect statistics for logging
    stats = dict(
        loss=logging_loss,
        importance_weight=ratio.detach(),
        approx_kl=(logprobs - proximal_logprobs).detach(),
        clip_mask=clip_mask,
        dual_clip_mask=dual_clip_mask,
    )

    # Add behavior statistics (only meaningful when using decoupled loss)
    if apply_behav_weighting:
        stats["behave_imp_weight"] = behav_imp_weight
        stats["behave_approx_kl"] = behav_kl
        stats["behave_mask"] = behav_mask

        # Compute staleness metrics
        # A sample is "stale" if behav_imp_weight significantly differs from 1.0
        staleness_threshold = 1.2  # Consider stale if weight > 1.2 or < 0.8
        is_stale = ((behav_imp_weight > staleness_threshold) | (behav_imp_weight < 1.0 / staleness_threshold)) & loss_mask_bool
        stats["staleness_fraction"] = is_stale.float().sum() / loss_mask_count
        stats["max_behav_imp_weight"] = behav_imp_weight.max()
        stats["mean_behav_imp_weight"] = torch.where(loss_mask_bool, behav_imp_weight, 0).sum() / loss_mask_count

    return pg_loss, stats


def compute_advantages_with_version_tracking(
    rewards: torch.Tensor,
    values: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    kl_ctl: float,
    discount: float = 0.99,
    gae_lambda: float = 0.95,
    version_start: Optional[torch.Tensor] = None,
    version_end: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute advantages using GAE, with optional version tracking for staleness metrics.

    Args:
        rewards: Reward tensor. Shape: [bs]
        values: Value estimates. Shape: [bs, max_seqlen]
        old_logprobs: Log probs from behavior policy. Shape: [bs, max_seqlen]
        ref_logprobs: Log probs from reference policy. Shape: [bs, max_seqlen]
        attention_mask: Attention mask. Shape: [bs, max_seqlen]
        loss_mask: Loss mask (rolled by -1). Shape: [bs, max_seqlen]
        kl_ctl: KL penalty coefficient
        discount: Discount factor γ
        gae_lambda: GAE lambda parameter
        version_start: Policy version when generation started. Shape: [bs]
        version_end: Policy version when generation ended. Shape: [bs]

    Returns:
        advantages: Computed advantages. Shape: [bs, max_seqlen]
        stats: Dictionary with version tracking statistics if provided
    """

    bs = rewards.shape[0]
    max_seqlen = values.shape[1]
    device = rewards.device

    batch_indices = torch.arange(bs, device=device, dtype=torch.long)
    seqlens = attention_mask.sum(-1).long()
    seq_no_eos_mask = seqlens == max_seqlen

    # Compute KL-regularized rewards
    kl_rewards = -kl_ctl * (old_logprobs - ref_logprobs) * loss_mask
    step_rewards = kl_rewards.clone()

    # Add terminal reward at the end of sequence
    step_rewards[batch_indices, torch.clip(seqlens - 2, min=0)] += rewards

    # Compute GAE advantages
    advantages_reversed = [torch.zeros(bs, dtype=torch.float32, device=device)]
    lastgaelam = 0
    nextvalues = values[:, max_seqlen - 1] * seq_no_eos_mask

    for t in reversed(range(max_seqlen - 1)):
        delta = step_rewards[:, t] + discount * nextvalues - values[:, t]
        newgaelam = delta + discount * gae_lambda * lastgaelam

        # Skip tokens that do not contribute to the loss
        mask = loss_mask[:, t]
        nextvalues = nextvalues * (1 - mask) + values[:, t] * mask
        lastgaelam = lastgaelam * (1 - mask) + newgaelam * mask
        advantages_reversed.append(lastgaelam)

    advantages = torch.stack(advantages_reversed[::-1], dim=1)

    # Compute version staleness statistics if available
    stats = {}
    if version_start is not None and version_end is not None:
        version_diff = version_end - version_start
        stats["max_version_diff"] = version_diff.max().item()
        stats["mean_version_diff"] = version_diff.float().mean().item()
        stats["samples_with_staleness"] = (version_diff > 0).float().mean().item()

    return advantages, stats
