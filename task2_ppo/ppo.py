from __future__ import annotations

import torch

from common.metrics import masked_mean


def compute_gae(rewards, values, mask, gamma=1.0, lam=0.95):
    """Token-level GAE over response positions.

    rewards, values, mask: [batch, response_steps]. Padding positions must have mask=0.
    The final valid response position bootstraps with zero.
    """
    batch, steps = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(batch, device=rewards.device, dtype=rewards.dtype)  # A_{t+1}, carried backwards

    # Backward recursion A_t = delta_t + gamma * lambda * A_{t+1}, which unrolls to the manual's
    # A_t = sum_k (gamma lambda)^k delta_{t+k}.
    for t in reversed(range(steps)):
        current_valid = mask[:, t]
        if t + 1 < steps:
            next_valid = mask[:, t + 1]
            next_value = values[:, t + 1] * next_valid  # V(s_{t+1}); 0 once the response has ended
        else:
            next_valid = torch.zeros_like(current_valid)
            next_value = torch.zeros_like(last_adv)

        delta = rewards[:, t] + gamma * next_value - values[:, t]  # TD residual delta_t
        last_adv = delta + gamma * lam * next_valid * last_adv
        last_adv = last_adv * current_valid  # padding positions carry no advantage
        advantages[:, t] = last_adv

    returns = advantages + values  # critic regression target R_t = A_t + V(s_t)
    return advantages, returns


def shaped_rewards(task_reward, policy_logp, ref_logp, response_mask, beta_kl):
    """Sampled-action KL shaping plus terminal learned reward.

    r_t = r_task * 1[t = T] - beta_KL * (log pi(a_t|s_t) - log pi_ref(a_t|s_t)), shape [batch, steps].
    """
    rewards = -float(beta_kl) * (policy_logp - ref_logp) * response_mask  # per-token KL penalty
    for b in range(rewards.shape[0]):
        valid = int(response_mask[b].sum().item())  # response mask is contiguous, so T = valid - 1
        if valid > 0:
            rewards[b, valid - 1] += task_reward[b]  # learned reward arrives only at the last token
    return rewards


def ppo_policy_loss(new_logp, old_logp, advantage, mask, eps=0.2):
    """Return the PPO clipped policy loss and diagnostics.

    L = -E_t[min(rho_t A_t, clip(rho_t, 1-eps, 1+eps) A_t)], averaged over valid response tokens.

    FIX (deliberate starter defect): the release took torch.maximum of the two surrogates. The max
    is an optimistic bound - it always keeps the unclipped (or larger) branch, so large ratio moves
    in the direction of the advantage were never cut off and PPO lost its trust region. The
    pessimistic min is what removes the incentive to push rho beyond [1-eps, 1+eps].

    Diagnostics (all over the same response mask):
      clip_fraction     - rho outside [1-eps, 1+eps] before clipping (the manual's PPO clip fraction);
      affected_fraction - tokens where the clipped branch is the active minimum, i.e. the token's
                          gradient is actually zeroed (rho > 1+eps with A > 0, or rho < 1-eps with A < 0).
    """
    ratio = torch.exp(new_logp - old_logp)  # rho_t = pi_theta / pi_old, computed in log space
    surr1 = ratio * advantage  # unclipped surrogate
    surr2 = ratio.clamp(1.0 - eps, 1.0 + eps) * advantage  # clipped surrogate (zero gradient outside the band)
    objective = torch.minimum(surr1, surr2)  # pessimistic bound (was torch.maximum)

    loss = -masked_mean(objective, mask)  # maximize the objective -> minimize its negative, token mean
    outside = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    # The clip only bites when it is the smaller branch: rho too high for a good token, or too low for a bad one.
    affected = (((ratio > 1.0 + eps) & (advantage > 0)) | ((ratio < 1.0 - eps) & (advantage < 0))).float()
    return loss, {
        "ratio": ratio.detach(),
        "clip_fraction": masked_mean(outside, mask).detach(),
        "affected_fraction": masked_mean(affected, mask).detach(),
        "clipped_surrogate": masked_mean(objective.detach(), mask),
        "unclipped_surrogate": masked_mean(surr1.detach(), mask),
    }


def value_mse_loss(predicted_values, returns, mask):
    """Critic regression (V(s_t) - R_t)^2, token mean over the response (scaled by value_coef by the caller)."""
    return masked_mean((predicted_values - returns) ** 2, mask)


def normalize_advantages(advantages, mask, eps=1e-6):
    """Whiten advantages over valid tokens (zero mean, unit std); padding stays 0."""
    valid = advantages[mask.bool()]
    if valid.numel() <= 1:
        return advantages
    mean = valid.mean()
    std = valid.std(unbiased=False).clamp_min(eps)
    return ((advantages - mean) / std) * mask
