"""Task 3: GRPO continuation from the supplied midpoint (standard run and normalization forks).

One update:
  1. sample K = `num_generations` completions for each of `prompts_per_update` prompts;
  2. score them once without gradients: old log-probs + exact entropy, reference log-probs (adapter
     disabled) and the learned reward of every completion;
  3. group-relative advantages A_k = (r_k - mu_g) / (sigma_g + eps) inside each prompt's group.
     Completions that hit the generation cap are masked out of the loss
     (`mask_truncated_completions`) but still count in their group's reward statistics;
  4. `policy_epochs` steps of the clipped objective with the chosen sequence normalization
     (`grpo`: 1/T_k per completion, `dr_grpo`: constant 1/max_completion_length).

Besides the per-update log, each completion's length, advantage and gradient allocation is saved:
`grad_mass` = sum_t |dL/d log pi(y_t)|, the exact weight the loss places on that completion's token
log-probs. It is the length-conditioned statistic used to compare the two normalizations.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, prompt_schedule, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs, scored_response_logprobs
from common.logging_utils import append_jsonl, peak_vram, reset_peak_vram, save_json, scalars, set_seed, wall_timer
from common.metrics import masked_mean, sampled_kl
from common.models import disable_dropout, load_policy, load_reward_model, load_tokenizer, reference_mode, trainable_parameters
from task3_grpo.grpo import group_relative_advantages, group_reward_stats, grpo_policy_loss, mask_truncated_sequences


def prepare_grpo_continuation(config_path: str):
    """Supplied GRPO midpoint, frozen reward model, prompt pool and AdamW (seeded)."""
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    disable_dropout(policy)  # first-epoch ratios are then exactly 1 up to kernel numerics
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


@torch.no_grad()
def collect_groups(bundle: dict, rows: list[dict], cfg: dict) -> dict:
    """Sample K completions per prompt and compute rewards, advantages and log-probs (no gradients)."""
    policy, k = bundle["policy"], int(cfg["num_generations"])
    # Each prompt repeated K times in one batch: [p0]*K + [p1]*K + ...; sampling makes the copies differ.
    prompts = [prompt_messages(r) for r in rows for _ in range(k)]
    gen = batch_generate(
        policy, bundle["tokenizer"], prompts,
        int(cfg["max_prompt_length"]), int(cfg["max_completion_length"]), **cfg["generation"],
    )
    mask = gen["response_mask"]
    old_logp, entropy = scored_response_logprobs(policy, gen)  # pi_old, fixed during the update
    with reference_mode(policy):  # pi_ref = adapter disabled
        ref_logp, _ = scored_response_logprobs(policy, gen, with_entropy=False)
    rewards = score_reward_pairs(  # one scalar learned reward per completion, [K_total]
        bundle["reward_model"], bundle["reward_tokenizer"], gen["prompt_messages"], gen["responses"],
        max_length=int(cfg.get("reward_max_length", 1024)),
    ).to(mask.device)

    group_ids = torch.arange(len(rows), device=mask.device).repeat_interleave(k)  # [0]*K + [1]*K + ...
    _, group_std, informative, _ = group_reward_stats(rewards, group_ids)
    # Truncated completions leave the loss but keep their reward in the group statistics above.
    loss_mask = mask_truncated_sequences(mask, gen["truncated"]) if cfg["mask_truncated_completions"] else mask
    return {
        **gen,
        "prompts": prompts,
        "group_ids": group_ids,
        "old_logp": old_logp,
        "ref_logp": ref_logp,
        "entropy": entropy,
        "rewards": rewards,
        "advantages": group_relative_advantages(rewards, group_ids),
        "group_std": group_std,
        "informative": informative,
        "loss_mask": loss_mask,
    }


def grpo_epoch(bundle: dict, groups: dict, cfg: dict, loss_type: str):
    """One clipped GRPO step. Returns (diagnostics, per-completion grad_mass)."""
    policy = bundle["policy"]
    new_logp, _ = response_token_logprobs(
        policy, groups["sequences"], groups["attention_mask"], groups["prompt_width"], groups["response_ids"]
    )
    loss, stats = grpo_policy_loss(
        new_logp, groups["old_logp"], groups["advantages"], groups["loss_mask"], groups["ref_logp"],
        float(cfg["clip_epsilon"]), float(cfg["kl_beta"]),
        loss_type=loss_type, max_completion_length=int(cfg["max_completion_length"]),
    )
    # dL/d log pi(y_t): how strongly the loss pushes each sampled token's log-prob. It differentiates
    # only the loss head (new_logp -> loss), not the network, so it is cheap; retain_graph keeps the
    # graph alive for the real backward pass below.
    (token_grad,) = torch.autograd.grad(loss, new_logp, retain_graph=True)
    bundle["optimizer"].zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = clip_grad_norm_(trainable_parameters(policy), float(cfg["max_grad_norm"]))  # pre-clip norm
    bundle["optimizer"].step()

    # Per-completion gradient mass: under `grpo` ~ |A_k| regardless of length, under `dr_grpo` ~ |A_k| * T_k.
    grad_mass = (token_grad.abs() * groups["loss_mask"]).sum(-1)
    return {"loss": float(loss), **scalars(stats), "grad_norm": float(grad_norm)}, grad_mass


def completion_records(update: int, rows: list[dict], groups: dict, grad_mass: torch.Tensor) -> list[dict]:
    """One row per sampled completion (length, reward, advantage, gradient allocation, text) for the analyses."""
    k = len(groups["prompts"]) // len(rows)
    loss_tokens = groups["loss_mask"].sum(-1)
    records = []
    for i in range(len(groups["prompts"])):
        g = i // k
        records.append({
            "update": update,
            "prompt_id": rows[g]["prompt_id"],
            "group": g,
            "k": i % k,
            "response_tokens": int(groups["response_lengths"][i]),
            "truncated": bool(groups["truncated"][i]),
            "in_loss": bool(loss_tokens[i] > 0),
            "reward": float(groups["rewards"][i]),
            "advantage": float(groups["advantages"][i]),
            "group_reward_std": float(groups["group_std"][g]),
            "grad_mass": float(grad_mass[i]),
            "grad_per_token": float(grad_mass[i] / loss_tokens[i].clamp_min(1.0)),
            "response": groups["responses"][i],
            "effective_prompt_messages": groups["prompt_messages"][i],
            "prompt_truncated": groups["prompt_truncated"][i],
        })
    return records


def update_record(update: int, rows: list[dict], groups: dict, epochs: list[dict]) -> dict:
    """One JSON log line per update with the diagnostics the manual lists for the GRPO continuation."""
    mask = groups["response_mask"]
    mean = lambda key: float(np.mean([e[key] for e in epochs]))
    return {
        "update": update,
        "prompt_ids": [r["prompt_id"] for r in rows],
        "reward": float(groups["rewards"].mean()),
        "reward_std_within_group": float(groups["group_std"].mean()),
        "frac_uninformative_groups": float(1.0 - groups["informative"].float().mean()),  # std <= 1e-6 -> zero advantages
        "kl": float(sampled_kl(groups["old_logp"], groups["ref_logp"], mask)),  # course helper, same as PPO/DPO
        "kl_k3": mean("sampled_kl"),  # the k3 term actually inside the GRPO loss
        "policy_loss": mean("policy_term"),
        "loss": mean("loss"),
        "grad_norm": mean("grad_norm"),
        "entropy": float(masked_mean(groups["entropy"], mask)),
        "sample_entropy": mean("sample_entropy"),
        "clip_fraction": mean("clip_fraction"),
        "response_tokens": float(np.mean(groups["response_lengths"])),
        "truncation_rate": float(np.mean(groups["truncated"])),
    }


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo", run_name: str = "standard"):
    """Run `updates` GRPO updates from the supplied midpoint; `loss_type` selects the sequence normalization."""
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    log_dir = repo_path(cfg["results_dir"]) / "train" / run_name
    for name in ["update_log.jsonl", "completions.jsonl"]:
        (log_dir / name).unlink(missing_ok=True)

    per_update = int(cfg["prompts_per_update"])
    schedule = prompt_schedule(bundle["prompt_rows"], int(cfg["updates"]) * per_update, int(cfg["seed"]))
    elapsed = wall_timer()
    reset_peak_vram()
    generated_tokens = 0
    effective_generation = None

    for u in range(int(cfg["updates"])):
        rows = schedule[u * per_update:(u + 1) * per_update]
        groups = collect_groups(bundle, rows, cfg)  # 1) sample K per prompt + score
        effective_generation = groups["effective_generation_config"]
        epochs, masses = [], []
        for _ in range(int(cfg["policy_epochs"])):  # 2) optimize (1 epoch in the release config)
            stats, mass = grpo_epoch(bundle, groups, cfg, loss_type)
            epochs.append(stats)
            masses.append(mass)

        generated_tokens += int(sum(groups["response_lengths"]))
        record = update_record(u + 1, rows, groups, epochs)
        record.update(loss_type=loss_type, generated_tokens=generated_tokens, elapsed_s=elapsed(), **peak_vram())
        append_jsonl(log_dir / "update_log.jsonl", record)
        # Gradient allocation of the first epoch, i.e. on the freshly sampled batch.
        for rec in completion_records(u + 1, rows, groups, masses[0]):
            append_jsonl(log_dir / "completions.jsonl", rec)
        print(f"[{run_name}] update {u + 1}/{cfg['updates']}: reward={record['reward']:.3f} "
              f"std={record['reward_std_within_group']:.3f} kl={record['kl']:.4f} len={record['response_tokens']:.0f}", flush=True)
        del groups

    bundle["policy"].save_pretrained(str(out))
    bundle["tokenizer"].save_pretrained(str(out))
    save_json(log_dir / "summary.json", {
        "run_name": run_name,
        "output": str(out),
        "loss_type": loss_type,
        "updates": int(cfg["updates"]),
        "generated_tokens": generated_tokens,
        "maximum_generated_token_budget": int(cfg["updates"]) * per_update * int(cfg["num_generations"]) * int(cfg["max_completion_length"]),
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
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()
