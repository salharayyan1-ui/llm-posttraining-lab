"""Task 2, step 2: clipping study.

Part A - supplied cached batch (no generation).
  Rebuild the 32 cached rollouts as one fixed batch, form KL-shaped rewards and GAE advantages from
  the cached old/reference log-probs and critic values, then for every epsilon start from the
  identical midpoint policy and take `ppo_epochs` clipped-surrogate steps on that batch. Before each
  step we record the clip fraction, the affected-token fraction, the clipped and unclipped surrogate
  and ratio statistics, all over the same response mask. Step 0 is the static geometry of the
  as-loaded midpoint against the cached old log-probs; later steps show how quickly each epsilon
  starts constraining the update.

Part B - matched short forks (task2_ppo/forks.py): `fork_updates` updates per epsilon with
  identical reward/KL settings, prompts and seed, followed by the common held-out evaluation.
"""

from __future__ import annotations

import argparse

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, render_prompt, repo_path
from common.generation import response_token_logprobs, scored_response_logprobs
from common.logging_utils import save_json
from common.models import clear_gpu, disable_dropout, load_policy, load_tokenizer, trainable_parameters
from task2_ppo.continue_train import normalize
from task2_ppo.forks import fork_name, fork_summary, run_fork, trajectories
from task2_ppo.ppo import compute_gae, ppo_policy_loss, shaped_rewards


def load_cached_rollouts(path):
    """Supplied PPO rollout cache (released loader; unifies two equivalent field names)."""
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


# ----------------------------------------------------------------------------- Part A

def pack_rollouts(prompt_ids: list[list[int]], response_ids: list[list[int]], pad_id: int) -> dict:
    """Same tensor layout as `batch_generate`: left-padded prompts, right-padded responses."""
    width, steps = max(map(len, prompt_ids)), max(map(len, response_ids))
    seqs, attn, rmask = [], [], []
    for p, r in zip(prompt_ids, response_ids):
        seqs.append([pad_id] * (width - len(p)) + p + r + [pad_id] * (steps - len(r)))
        attn.append([0] * (width - len(p)) + [1] * (len(p) + len(r)) + [0] * (steps - len(r)))
        rmask.append([1] * len(r) + [0] * (steps - len(r)))
    sequences = torch.tensor(seqs, dtype=torch.long)
    return {
        "sequences": sequences,
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "prompt_width": width,
        "response_ids": sequences[:, width:],
        "response_mask": torch.tensor(rmask, dtype=torch.float32),
    }


def build_cached_batch(cfg: dict, tokenizer, rows: list[dict]):
    """Re-tokenize cached prompts/responses and align them with the cached per-token tensors.

    The cache stores decoded text, so the response is re-encoded (+ EOS if it terminated); a row
    whose token count no longer matches its cached log-prob vector cannot be aligned and is skipped.
    """
    pool = {r["prompt_id"]: r for key in ("rl_prompt_eval", "rl_prompt_train") for r in read_jsonl(cfg["paths"][key])}
    prompt_ids, response_ids, kept, skipped = [], [], [], []
    for row in rows:
        rendered = render_prompt(tokenizer, prompt_messages(pool[row["prompt_id"]]))
        p_ids = tokenizer(rendered, truncation=True, max_length=int(cfg["max_prompt_length"]))["input_ids"]
        r_ids = tokenizer(row["response"], add_special_tokens=False)["input_ids"]
        if row.get("terminated_with_eos"):
            r_ids = r_ids + [tokenizer.eos_token_id]
        if len(r_ids) != len(row["old_logprobs"]):
            skipped.append({"prompt_id": row["prompt_id"], "retokenized": len(r_ids), "cached": len(row["old_logprobs"])})
            continue
        prompt_ids.append(p_ids)
        response_ids.append(r_ids)
        kept.append(row)

    batch = pack_rollouts(prompt_ids, response_ids, tokenizer.pad_token_id)
    steps = batch["response_ids"].shape[1]
    padded = lambda key: torch.stack([F.pad(torch.as_tensor(r[key], dtype=torch.float32), (0, steps - len(r[key]))) for r in kept])
    batch.update(
        old_logp=padded("old_logprobs"),
        ref_logp=padded("ref_logprobs"),
        values=padded("values"),
        task_reward=torch.tensor([float(r["effective_terminal_reward"]) for r in kept]),
    )
    return batch, skipped


def cached_advantages(batch: dict, cfg: dict) -> torch.Tensor:
    """KL-shaped rewards + GAE from the cached old/ref log-probs and critic values, normalized as in training."""
    mask = batch["response_mask"]
    rewards = shaped_rewards(batch["task_reward"], batch["old_logp"], batch["ref_logp"], mask, float(cfg["kl_beta"]))
    advantages, _ = compute_gae(rewards, batch["values"], mask, float(cfg["gamma"]), float(cfg["gae_lambda"]))
    return normalize(advantages, mask, cfg.get("advantage_normalization", "scale_only"))


def row_chunks(batch: dict, chunk: int):
    """Yield (row slice, sub-batch) pairs so long sequences are processed a few rows at a time."""
    for start in range(0, batch["sequences"].shape[0], chunk):
        rows = slice(start, start + chunk)
        yield rows, {
            "sequences": batch["sequences"][rows],
            "attention_mask": batch["attention_mask"][rows],
            "prompt_width": batch["prompt_width"],
            "response_ids": batch["response_ids"][rows],
        }


def measure(policy, batch: dict, eps: float, chunk: int) -> dict:
    """Clip/affected fractions and surrogates of the current policy on the fixed batch (no update)."""
    new_logp, _ = scored_response_logprobs(policy, batch, chunk_rows=chunk, with_entropy=False)
    mask = batch["response_mask"]
    _, stats = ppo_policy_loss(new_logp, batch["old_logp"], batch["advantages"], mask, eps)
    log_ratio = (new_logp - batch["old_logp"])[mask.bool()]
    return {
        "clip_fraction": float(stats["clip_fraction"]),
        "affected_fraction": float(stats["affected_fraction"]),
        "clipped_surrogate": float(stats["clipped_surrogate"]),
        "unclipped_surrogate": float(stats["unclipped_surrogate"]),
        "mean_abs_log_ratio": float(log_ratio.abs().mean()),
        "ratio_min": float(log_ratio.min().exp()),
        "ratio_max": float(log_ratio.max().exp()),
    }


def surrogate_step(policy, optimizer, batch: dict, eps: float, chunk: int, max_grad_norm: float) -> float:
    """One optimizer step on the full cached batch, accumulated over row chunks.

    Each chunk's token-mean loss is weighted by its share of valid tokens, so the accumulated
    gradient equals that of the full-batch token mean.
    """
    mask = batch["response_mask"]
    total_tokens = mask.sum()
    optimizer.zero_grad(set_to_none=True)
    for rows, part in row_chunks(batch, chunk):
        new_logp, _ = response_token_logprobs(policy, part["sequences"], part["attention_mask"], part["prompt_width"], part["response_ids"])
        loss, _ = ppo_policy_loss(new_logp, batch["old_logp"][rows], batch["advantages"][rows], mask[rows], eps)
        (loss * mask[rows].sum() / total_tokens).backward()
    grad_norm = clip_grad_norm_(trainable_parameters(policy), max_grad_norm)
    optimizer.step()
    return float(grad_norm)


def cached_rollout_study(cfg: dict) -> pd.DataFrame:
    """Part A: per epsilon, a fresh midpoint alternates measure -> step on the fixed cached batch; writes clipping_cached.*"""
    tokenizer = load_tokenizer(cfg["base_model"])
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    batch, skipped = build_cached_batch(cfg, tokenizer, rows)
    batch["advantages"] = cached_advantages(batch, cfg)
    chunk = int(cfg.get("cached_chunk_rows", 2))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    records = []
    for eps in cfg["clip_values"]:
        policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
        disable_dropout(policy)
        optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["policy_learning_rate"]))
        for step in range(int(cfg["ppo_epochs"]) + 1):
            record = {"epsilon": float(eps), "step": step, **measure(policy, batch, float(eps), chunk)}
            if step < int(cfg["ppo_epochs"]):
                record["grad_norm"] = surrogate_step(policy, optimizer, batch, float(eps), chunk, float(cfg["max_grad_norm"]))
            records.append(record)
            print(f"[cached eps={eps}] step {step}: clip={record['clip_fraction']:.4f} affected={record['affected_fraction']:.4f}", flush=True)
        del policy, optimizer
        clear_gpu()

    table = pd.DataFrame(records)
    results = repo_path(cfg["results_dir"])
    results.mkdir(parents=True, exist_ok=True)
    table.to_csv(results / "clipping_cached.csv", index=False)
    save_json(results / "clipping_cached.json", {
        "n_cached_rollouts": len(rows),
        "n_used": int(batch["sequences"].shape[0]),
        "skipped_misaligned": skipped,
        "valid_tokens": int(batch["response_mask"].sum()),
        "kl_beta": float(cfg["kl_beta"]),
        "advantage_normalization": cfg.get("advantage_normalization", "scale_only"),
        "records": records,
    })
    return table


# ----------------------------------------------------------------------------- Part B

def fork_study(config_path: str, cfg: dict, train: bool) -> pd.DataFrame:
    """Part B: train/evaluate the epsilon forks (finished ones are skipped) and tabulate them."""
    names, rows = [], []
    kl_beta = float(cfg["kl_beta"])
    for eps in map(float, cfg["clip_values"]):
        if train:
            run_fork(config_path, cfg, eps, kl_beta)
        names.append(fork_name(eps, kl_beta))
        rows.append(fork_summary(cfg, names[-1], epsilon=eps, kl_beta=kl_beta))
    results = repo_path(cfg["results_dir"])
    table = pd.DataFrame(rows)
    table.to_csv(results / "clipping_forks.csv", index=False)
    trajectories(cfg, names).to_csv(results / "clipping_trajectories.csv", index=False)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--part", choices=["cached", "forks", "all"], default="all")
    ap.add_argument("--summary-only", action="store_true", help="forks: only rebuild the table from saved results")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if args.part in {"cached", "all"}:
        print(cached_rollout_study(cfg).to_string(index=False))
    if args.part in {"forks", "all"}:
        print(fork_study(args.config, cfg, train=not args.summary_only).to_string(index=False))


if __name__ == "__main__":
    main()
