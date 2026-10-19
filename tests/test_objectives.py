"""CPU checks of the three corrected objectives (and GAE) against the manual's equations.

Run before any GPU job:  python -m pytest -q tests   (or: python -m tests.test_objectives)
Each test fails on the corresponding released starter defect.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from task1_dpo.dpo import dpo_loss
from task2_ppo.ppo import compute_gae, ppo_policy_loss
from task3_grpo.grpo import group_relative_advantages, group_reward_stats, grpo_policy_loss


# ----------------------------------------------------------------------------- DPO

def test_dpo_matches_manual_equation():
    pc, pr = torch.tensor([-10.0, -4.0]), torch.tensor([-12.0, -3.0])
    rc, rr = torch.tensor([-11.0, -5.0]), torch.tensor([-11.0, -2.0])
    beta = 0.1
    margin = (pc - rc) - (pr - rr)
    loss, stats = dpo_loss(pc, pr, rc, rr, beta)
    assert torch.allclose(loss, -F.logsigmoid(beta * margin).mean())
    assert math.isclose(float(stats["preference_accuracy"]), float((margin > 0).float().mean()))


def test_dpo_is_log2_when_policy_equals_reference():
    logp_c, logp_r = torch.tensor([-7.0, -20.0]), torch.tensor([-9.0, -15.0])
    loss, stats = dpo_loss(logp_c, logp_r, logp_c, logp_r, beta=0.3)
    assert math.isclose(float(loss), math.log(2.0), rel_tol=1e-6)
    assert float(stats["preference_accuracy"]) == 0.0


def test_dpo_gradient_raises_chosen_and_lowers_rejected():
    pc, pr = torch.tensor([-5.0], requires_grad=True), torch.tensor([-5.0], requires_grad=True)
    loss, _ = dpo_loss(pc, pr, torch.tensor([-5.0]), torch.tensor([-5.0]), beta=0.1)
    loss.backward()
    assert pc.grad.item() < 0 < pr.grad.item()


# ----------------------------------------------------------------------------- PPO

def test_ppo_uses_pessimistic_minimum():
    eps = 0.2
    ratio = torch.tensor([[1.5, 0.5, 1.5, 0.5, 1.0]])
    adv = torch.tensor([[1.0, 1.0, -1.0, -1.0, 2.0]])
    old = torch.zeros_like(ratio)
    new = ratio.log()
    mask = torch.ones_like(ratio)
    loss, stats = ppo_policy_loss(new, old, adv, mask, eps)
    expected = torch.minimum(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv)  # [1.2, 0.5, -1.5, -0.8, 2.0]
    assert torch.allclose(loss, -expected.mean())
    assert math.isclose(float(stats["clip_fraction"]), 4 / 5, rel_tol=1e-6)      # float32 token fraction
    assert math.isclose(float(stats["affected_fraction"]), 2 / 5, rel_tol=1e-6)  # clipped branch active


def test_ppo_clipped_tokens_have_zero_gradient():
    new = torch.tensor([[math.log(1.5), math.log(1.5)]], requires_grad=True)
    adv = torch.tensor([[1.0, -1.0]])
    loss, _ = ppo_policy_loss(new, torch.zeros(1, 2), adv, torch.ones(1, 2), eps=0.2)
    loss.backward()
    assert new.grad[0, 0].item() == 0.0   # A > 0 and rho > 1 + eps: no further push
    assert new.grad[0, 1].item() != 0.0   # A < 0: the unclipped (pessimistic) term still pulls rho down


def test_gae_reduces_to_reward_to_go():
    rewards = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    adv, returns = compute_gae(rewards, torch.zeros_like(rewards), mask, gamma=1.0, lam=1.0)
    assert torch.allclose(adv, torch.tensor([[1.0, 1.0, 1.0, 0.0]]))
    assert torch.allclose(returns, adv)


# ----------------------------------------------------------------------------- GRPO

def test_grpo_advantages_are_normalized_within_each_group():
    rewards = torch.tensor([1.0, 3.0, 100.0, 104.0, 5.0, 5.0])
    groups = torch.tensor([0, 0, 1, 1, 2, 2])
    adv = group_relative_advantages(rewards, groups)
    assert torch.allclose(adv[:4], torch.tensor([-1.0, 1.0, -1.0, 1.0]), atol=1e-5)
    assert torch.all(adv[4:] == 0)  # constant group carries no signal
    _, std, informative, _ = group_reward_stats(rewards, groups)
    assert informative.tolist() == [True, True, False]
    assert torch.allclose(std, torch.tensor([1.0, 2.0, 0.0]))


def test_grpo_normalizations_weight_lengths_differently():
    # Two completions with the same advantage: lengths 2 and 4 (of max 4), ratio = 1.
    logp = torch.zeros(2, 4, requires_grad=True)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    adv = torch.tensor([1.0, 1.0])
    for loss_type, expected_mass in [("grpo", [0.5, 0.5]), ("dr_grpo", [0.25, 0.5])]:
        loss, _ = grpo_policy_loss(logp, logp.detach(), adv, mask, logp.detach(), eps=0.2, beta=0.0,
                                   loss_type=loss_type, max_completion_length=4)
        (grad,) = torch.autograd.grad(loss, logp)
        mass = (grad.abs() * mask).sum(-1)
        assert torch.allclose(mass, torch.tensor(expected_mass)), (loss_type, mass)


if __name__ == "__main__":
    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_") and callable(obj)]
    for test in tests:
        test()
        print("ok", test.__name__)
    print(f"{len(tests)} objective checks passed")
