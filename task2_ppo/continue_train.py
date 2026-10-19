"""Task 2: PPO continuation from the supplied midpoint (standard run and every short fork).

One update:
  1. sample `prompts_per_update` prompts' responses from the current policy;
  2. score them once, without gradients: old log-probs + exact entropy (policy), reference
     log-probs (adapter disabled), critic values V(s_t), and the learned reward (minus
     `missing_eos_penalty` when the response never emitted EOS);
  3. token rewards r_t = -beta_KL (log pi_old - log pi_ref) + r_task 1[t = T], then GAE;
  4. `ppo_epochs` passes of the clipped surrogate for the policy and value regression for the critic.

Every run walks the same seeded permutation of the training prompt pool, so a short fork sees
exactly the first prompts of the standard run and forks differ only in the setting under study.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, prompt_schedule, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs, scored_response_logprobs
from common.logging_utils import append_jsonl, peak_vram, reset_peak_vram, save_json, set_seed, wall_timer
from common.metrics import explained_variance, masked_mean, sample_entropy, sampled_kl
from common.models import (
    disable_dropout,
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards, value_mse_loss


def prepare_ppo_continuation(config_path: str):
    """Supplied midpoint policy + critic, frozen reward model, prompt pool and both optimizers (seeded)."""
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    # With dropout off, the first PPO epoch reproduces the rollout log-probs exactly (ratio = 1).
    disable_dropout(policy)
    disable_dropout(value_model)
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def normalize(advantages, mask, mode: str):
    """`full` = whiten (shift + scale); `scale_only` keeps the sign; `none` leaves GAE untouched.

    With one prompt per update, full whitening would centre the advantages of a single response and
    erase whether that response was better or worse than the critic expected, so the default
    (configs/ppo.yaml) is scale_only.
    """
    if mode == "full":
        return normalize_advantages(advantages, mask)
    if mode == "scale_only":
        valid = advantages[mask.bool()]  # only real response tokens enter the statistics
        return advantages / valid.std(unbiased=False).clamp_min(1e-6) * mask if valid.numel() > 1 else advantages
    if mode == "none":
        return advantages
    raise ValueError(f"Unknown advantage_normalization={mode!r}")


def response_values(value_model, rollout: dict) -> torch.Tensor:
    """Critic value V(s_t) for every response position t (state = prompt + response[:t]).

    The value read at sequence position i summarizes tokens 0..i, i.e. the state *before* token i+1
    is emitted, so response token t (position prompt_width + t) gets the value at prompt_width - 1 + t.
    """
    values = token_values(value_model, rollout["sequences"], rollout["attention_mask"])  # [batch, seq_len]
    start = rollout["prompt_width"] - 1
    # Rollout rewards/GAE and policy loss live on the policy GPU. This small copy
    # remains differentiable, so critic regression still updates the other GPU.
    return values[:, start:start + rollout["response_ids"].shape[1]].float().to(rollout["sequences"].device)


@torch.no_grad()
def collect_rollout(bundle: dict, rows: list[dict], cfg: dict) -> dict:
    """Sample responses with the current policy and compute everything the update needs (no gradients)."""
    policy, tokenizer = bundle["policy"], bundle["tokenizer"]
    prompts = [prompt_messages(r) for r in rows]
    gen = batch_generate(
        policy, tokenizer, prompts,
        int(cfg["max_prompt_length"]), int(cfg["max_response_length"]), **cfg["generation"],
    )
    mask = gen["response_mask"]  # [batch, steps], 1 up to and including EOS

    # pi_old: the policy that generated the samples; its log-probs stay fixed for all ppo_epochs.
    old_logp, entropy = scored_response_logprobs(policy, gen)
    # pi_ref: same weights with the LoRA adapter disabled (frozen Qwen2.5-1.5B-Instruct).
    with reference_mode(policy):
        ref_logp, _ = scored_response_logprobs(policy, gen, with_entropy=False)
    values = response_values(bundle["value_model"], gen)

    # Learned reward of the full response; truncated responses (no EOS) are penalized as in the release config.
    raw_reward = score_reward_pairs(
        bundle["reward_model"], bundle["reward_tokenizer"], gen["prompt_messages"], gen["responses"],
        max_length=int(cfg["reward_max_length"]),
    ).to(mask.device)
    missing_eos = torch.tensor([not t for t in gen["terminated_with_eos"]], dtype=torch.float32, device=mask.device)
    task_reward = raw_reward - float(cfg["missing_eos_penalty"]) * missing_eos

    # Manual eq.: r_t = r_task 1[t=T] - beta_KL (log pi - log pi_ref), then GAE(gamma, lambda) with the critic.
    rewards = shaped_rewards(task_reward, old_logp, ref_logp, mask, float(cfg["kl_beta"]))
    advantages, returns = compute_gae(rewards, values, mask, float(cfg["gamma"]), float(cfg["gae_lambda"]))
    return {
        **gen,
        "prompts": prompts,
        "old_logp": old_logp,
        "ref_logp": ref_logp,
        "entropy": entropy,
        "values": values,
        "raw_reward": raw_reward,
        "task_reward": task_reward,
        "advantages": normalize(advantages, mask, cfg.get("advantage_normalization", "scale_only")),
        "returns": returns,
    }


def ppo_epoch(bundle: dict, rollout: dict, cfg: dict) -> dict:
    """One clipped-surrogate step for the policy and one regression step for the critic."""
    policy, value_model = bundle["policy"], bundle["value_model"]
    mask = rollout["response_mask"]

    # Policy: re-score the fixed rollout under the current weights (with grad) -> ratio vs pi_old.
    new_logp, _ = response_token_logprobs(
        policy, rollout["sequences"], rollout["attention_mask"], rollout["prompt_width"], rollout["response_ids"]
    )
    policy_loss, stats = ppo_policy_loss(new_logp, rollout["old_logp"], rollout["advantages"], mask, float(cfg["clip_epsilon"]))
    bundle["policy_optimizer"].zero_grad(set_to_none=True)
    policy_loss.backward()
    grad_norm = clip_grad_norm_(trainable_parameters(policy), float(cfg["max_grad_norm"]))  # pre-clip norm
    bundle["policy_optimizer"].step()

    # Critic: regress V(s_t) onto the GAE returns computed at rollout time (separate optimizer and model).
    values = response_values(value_model, rollout)
    value_loss = float(cfg["value_coef"]) * value_mse_loss(values, rollout["returns"], mask)
    bundle["value_optimizer"].zero_grad(set_to_none=True)
    value_loss.backward()
    value_grad_norm = clip_grad_norm_(trainable_parameters(value_model), float(cfg["max_grad_norm"]))
    bundle["value_optimizer"].step()

    return {
        "policy_loss": float(policy_loss),
        "value_loss": float(value_loss),
        "grad_norm": float(grad_norm),
        "value_grad_norm": float(value_grad_norm),
        "clip_fraction": float(stats["clip_fraction"]),
        "affected_fraction": float(stats["affected_fraction"]),
    }


@torch.no_grad()
def policy_step_size(policy, rollout: dict) -> dict:
    """How far this update moved the policy on its own rollout: k3 estimate of KL(old || new) and max |log ratio|."""
    new_logp, _ = scored_response_logprobs(policy, rollout, with_entropy=False)
    log_ratio = new_logp - rollout["old_logp"]  # log(pi_new / pi_old) on the tokens sampled from pi_old
    mask = rollout["response_mask"]
    return {
        # k3 = (r - 1) - log r >= 0 is an unbiased, low-variance estimate of KL(pi_old || pi_new).
        "step_kl_old_new": float(masked_mean(torch.exp(log_ratio) - 1.0 - log_ratio, mask)),
        "max_abs_log_ratio": float((log_ratio.abs() * mask).max()),
    }


def update_record(update: int, rows: list[dict], rollout: dict, epochs: list[dict], step: dict) -> dict:
    """One JSON log line per update: rollout statistics + epoch-averaged optimization diagnostics."""
    mask = rollout["response_mask"]
    mean = lambda key: float(np.mean([e[key] for e in epochs]))
    return {
        "update": update,
        "prompt_ids": [r["prompt_id"] for r in rows],
        "reward_raw": float(rollout["raw_reward"].mean()),  # reward-model score as is
        "reward": float(rollout["task_reward"].mean()),  # after the missing-EOS penalty (what PPO optimizes)
        "kl": float(sampled_kl(rollout["old_logp"], rollout["ref_logp"], mask)),  # course helper: token mean
        "kl_sequence": float(((rollout["old_logp"] - rollout["ref_logp"]) * mask).sum(-1).mean()),
        "policy_loss": mean("policy_loss"),
        "value_loss": mean("value_loss"),
        "entropy": float(masked_mean(rollout["entropy"], mask)),
        "sample_entropy": float(sample_entropy(rollout["old_logp"], mask)),
        "grad_norm": mean("grad_norm"),
        "grad_norm_max": max(e["grad_norm"] for e in epochs),
        "value_grad_norm": mean("value_grad_norm"),
        # Epoch 0 always has ratio = 1, so the mean over epochs is the clip rate of the whole update.
        "clip_fraction": mean("clip_fraction"),
        "clip_fraction_last_epoch": epochs[-1]["clip_fraction"],
        "affected_fraction": mean("affected_fraction"),
        **step,
        "response_tokens": float(np.mean(rollout["response_lengths"])),
        "eos_rate": float(np.mean(rollout["terminated_with_eos"])),
        "truncation_rate": float(np.mean(rollout["truncated"])),
        # Critic behaviour (the manual asks not to assume it is accurate): EV near 1 = good, <= 0 = no better than a constant.
        "value_mean": float(masked_mean(rollout["values"], mask)),
        "return_mean": float(masked_mean(rollout["returns"], mask)),
        "explained_variance": explained_variance(rollout["values"], rollout["returns"], mask),
    }


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    """Run `updates` PPO updates from the supplied midpoint; CLI overrides define the forks."""
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    log_dir = repo_path(cfg["results_dir"]) / "train" / run_name
    for name in ["update_log.jsonl", "rollouts.jsonl"]:
        (log_dir / name).unlink(missing_ok=True)

    per_update = int(cfg["prompts_per_update"])
    schedule = prompt_schedule(bundle["prompt_rows"], int(cfg["updates"]) * per_update, int(cfg["seed"]))
    elapsed = wall_timer()
    reset_peak_vram()
    generated_tokens = 0
    effective_generation = None

    for u in range(int(cfg["updates"])):
        rows = schedule[u * per_update:(u + 1) * per_update]
        rollout = collect_rollout(bundle, rows, cfg)  # 1) sample + score with pi_old
        effective_generation = rollout["effective_generation_config"]
        epochs = [ppo_epoch(bundle, rollout, cfg) for _ in range(int(cfg["ppo_epochs"]))]  # 2) optimize on it
        record = update_record(u + 1, rows, rollout, epochs, policy_step_size(bundle["policy"], rollout))  # 3) log
        generated_tokens += int(sum(rollout["response_lengths"]))
        record.update(generated_tokens=generated_tokens, elapsed_s=elapsed(), **peak_vram())
        append_jsonl(log_dir / "update_log.jsonl", record)
        for i, row in enumerate(rows):
            append_jsonl(log_dir / "rollouts.jsonl", {
                "update": u + 1,
                "prompt_id": row["prompt_id"],
                "prompt": rollout["prompts"][i][-1]["content"],
                "effective_prompt_messages": rollout["prompt_messages"][i],
                "prompt_truncated": rollout["prompt_truncated"][i],
                "response": rollout["responses"][i],
                "response_tokens": int(rollout["response_lengths"][i]),
                "reward_raw": float(rollout["raw_reward"][i]),
                "reward": float(rollout["task_reward"][i]),
            })
        print(f"[{run_name}] update {u + 1}/{cfg['updates']}: reward={record['reward']:.3f} kl={record['kl']:.4f} "
              f"clip={record['clip_fraction']:.3f} len={record['response_tokens']:.0f} ev={record['explained_variance']:.2f}", flush=True)
        del rollout

    # Policy LoRA adapter (used by Task 4) plus the critic's LoRA + score head for completeness.
    bundle["policy"].save_pretrained(str(out))
    bundle["tokenizer"].save_pretrained(str(out))
    bundle["value_model"].save_pretrained(str(out / "value"))
    save_json(log_dir / "summary.json", {
        "run_name": run_name,
        "output": str(out),
        "updates": int(cfg["updates"]),
        "clip_epsilon": float(cfg["clip_epsilon"]),
        "kl_beta": float(cfg["kl_beta"]),
        "generated_tokens": generated_tokens,
        "maximum_generated_token_budget": int(cfg["updates"]) * per_update * int(cfg["max_response_length"]),
        "budget_matching": "equal updates and token caps; actual emitted tokens reported separately",
        "effective_generation_config": effective_generation,
        "wall_clock_s": elapsed(),
        **peak_vram(),
        "prompt_ids": [r["prompt_id"] for r in schedule],
        "config": cfg,
    })
    print(f"[{run_name}] saved policy adapter to {out} ({elapsed():.0f}s, {peak_vram()['peak_allocated_gib']:.2f} GiB peak)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
