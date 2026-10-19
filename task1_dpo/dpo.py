from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_margin(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
) -> torch.Tensor:
    """Per-pair preference margin m = [log pi(y+) - log ref(y+)] - [log pi(y-) - log ref(y-)]."""
    return (policy_chosen_logp - ref_chosen_logp) - (policy_rejected_logp - ref_rejected_logp)


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
):
    """Return scalar DPO loss plus lightweight diagnostics.

    L = -E[log sigmoid(beta * m)] with m the reference-relative margin above.

    FIX (deliberate starter defect): the release used beta * (policy_margin + ref_margin). Adding
    the reference margin rewards pairs the *reference* already ranks correctly and pushes the
    policy away from, not relative to, the reference; at initialization (policy == reference) the
    loss was not log 2 as it must be. The manual's objective subtracts the reference log-ratio.
    """
    # All inputs are [batch] sequence log-probs summed over response tokens only.
    policy_margin = policy_chosen_logp - policy_rejected_logp  # raw policy preference (diagnostic only)
    margin = dpo_margin(policy_chosen_logp, policy_rejected_logp, ref_chosen_logp, ref_rejected_logp)
    logits = beta * margin  # Bradley-Terry logit of the implicit reward difference

    loss = -F.logsigmoid(logits).mean()  # -log sigma(.) is numerically stable for large |logits|

    return loss, {
        "logit_mean": logits.detach().mean(),
        "policy_margin_mean": policy_margin.detach().mean(),
        "margin_mean": margin.detach().mean(),
        # Implicit DPO rewards beta * log(pi / ref) for each side.
        "chosen_reward_mean": (beta * (policy_chosen_logp - ref_chosen_logp)).detach().mean(),
        "rejected_reward_mean": (beta * (policy_rejected_logp - ref_rejected_logp)).detach().mean(),
        "preference_accuracy": (margin > 0).float().mean().detach(),
    }
