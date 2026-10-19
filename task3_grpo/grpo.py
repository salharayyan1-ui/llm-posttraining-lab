from __future__ import annotations

import torch

from common.metrics import masked_mean, sampled_kl, sample_entropy


def group_reward_stats(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = 1e-6):
    """Per-group reward mean and population std, and which groups are informative (std > eps).

    Returns (mean[G], std[G], informative[G], inverse[N]) where inverse[i] is completion i's group slot.
    """
    _, inverse = torch.unique(group_ids, return_inverse=True)  # map arbitrary prompt ids to slots 0..G-1
    n_groups = int(inverse.max()) + 1
    zeros = lambda: torch.zeros(n_groups, dtype=rewards.dtype, device=rewards.device)
    # Scatter-add sums per group: count, sum of rewards, sum of squared deviations.
    count = zeros().index_add_(0, inverse, torch.ones_like(rewards))
    mean = zeros().index_add_(0, inverse, rewards) / count
    # Population std (divide by K), the release helper's convention (std(unbiased=False)).
    std = (zeros().index_add_(0, inverse, (rewards - mean[inverse]) ** 2) / count).sqrt()
    return mean, std, std > eps, inverse


def group_relative_advantages(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = 1e-6):
    """Return one scalar advantage per sampled completion: A_k = (r_k - mu_g) / (sigma_g + eps).

    `group_ids[i]` identifies which prompt produced reward `rewards[i]`.

    FIX (deliberate starter defect): the release normalized with the mean/std of the *whole batch*.
    With several prompts per batch that compares completions of an easy prompt against those of a
    hard prompt, so a whole group can receive the same-signed advantage and an uninformative group
    (all rewards equal) still gets a non-zero signal. GRPO's baseline is the mean of the completions
    sampled for the same prompt, so the statistics are now computed per group.
    """
    mean, std, _, inverse = group_reward_stats(rewards, group_ids, eps)
    # A group with identical rewards gets exactly 0 advantage: it carries no relative signal.
    return (rewards - mean[inverse]) / (std[inverse] + eps)


def grpo_policy_loss(
    new_logp,
    old_logp,
    seq_adv,
    token_mask,
    ref_logp,
    eps,
    beta,
    loss_type="grpo",
    max_completion_length: int | None = None,
):
    """PPO-style clipped GRPO loss for already-sampled completions.

    Shapes: new/old/ref_logp and token_mask are [K_total, steps]; seq_adv is [K_total], one
    advantage per completion, broadcast to all of its tokens.
    L = -mean_k [ norm_k * sum_t min(rho A_k, clip(rho) A_k) ] + beta * KL, with norm_k = 1/T_k
    (`grpo`) or 1/max_completion_length (`dr_grpo`).

    `token_mask` may be all-zero for a completion that was deliberately masked because it hit the
    maximum generation length.
    """
    ratio = torch.exp(new_logp - old_logp)
    adv = seq_adv[:, None]  # same advantage for every token of a completion
    s1 = ratio * adv
    s2 = ratio.clamp(1.0 - eps, 1.0 + eps) * adv
    objective = torch.minimum(s1, s2)  # pessimistic clipped surrogate, as in PPO

    token_sum = (objective * token_mask).sum(-1)  # per-completion sum over its valid tokens
    if loss_type == "grpo":
        # Canonical 1/T_k: each completion contributes equally, whatever its length.
        denom = token_mask.sum(-1).clamp_min(1.0)
        per_sequence = token_sum / denom
        policy_term = -per_sequence.mean()
    elif loss_type == "dr_grpo":
        if max_completion_length is None:
            raise ValueError("dr_grpo requires max_completion_length")
        # Constant normalization rather than dividing by each response's realized length.
        per_sequence = token_sum / float(max_completion_length)
        policy_term = -per_sequence.mean()
    else:
        raise ValueError(f"Unknown loss_type={loss_type!r}")

    # k3 estimator of KL(pi_theta || pi_ref) from DeepSeekMath: ref/pi - log(ref/pi) - 1 >= 0 per token.
    log_ratio_ref_over_policy = ref_logp - new_logp
    per_token_kl = torch.exp(log_ratio_ref_over_policy) - log_ratio_ref_over_policy - 1.0
    kl = masked_mean(per_token_kl, token_mask)  # token mean over the whole batch
    loss = policy_term + float(beta) * kl
    affected = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    return loss, {
        "policy_term": policy_term.detach(),
        "sampled_kl": kl.detach(),
        "clip_fraction": masked_mean(affected, token_mask).detach(),
        "ratio_mean": masked_mean(ratio.detach(), token_mask),
        "sample_entropy": sample_entropy(new_logp.detach(), token_mask),
    }


def mask_truncated_sequences(token_mask: torch.Tensor, truncated: list[bool] | torch.Tensor):
    """Zero the loss mask of completions that hit the generation cap (no EOS): their ending is unknown."""
    truncated = torch.as_tensor(truncated, device=token_mask.device, dtype=torch.bool)
    keep = (~truncated).to(token_mask.dtype)[:, None]
    return token_mask * keep
