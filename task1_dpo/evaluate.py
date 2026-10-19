"""Task 1 evaluation of one DPO adapter (or the SFT base policy with `--adapter none`).

Writes to results/task1_dpo/eval/<name>/:
  pairs_standard.jsonl    per-pair policy/reference log-probs and margin on the fixed held-out pairs
  pairs_stratified.jsonl  the same on the length-stratified held-out set (per-stratum analysis)
  generations.jsonl       one sampled response per held-out prompt with reward / KL / entropy / length
  word_limit.jsonl        sampled responses to the common word-limit prompts
  metrics.json            every aggregate reported for Task 1
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from tqdm import tqdm

from common.data import load_yaml, prompt_messages_from_preference, read_jsonl, repo_path, write_jsonl
from common.evaluation import adapter_or_none, evaluate_generations, summarize_generations
from common.generation import batch_generate
from common.logging_utils import save_json, set_seed
from common.metrics import length_stats, parse_word_limit, word_count, word_limit_compliance
from common.models import load_policy, load_reward_model, load_tokenizer
from task1_dpo.dpo import dpo_loss, dpo_margin
from task1_dpo.train import make_collate, pair_logps


def load_evaluation_bundle(config_path: str, adapter: str | None):
    """Config, held-out pairs, tokenizer, frozen policy (adapter or base) and the fixed reward model."""
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["dpo_standard_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


# ----------------------------------------------------------------------------- preference pairs

@torch.no_grad()
def score_pairs(model, tokenizer, rows: list[dict], max_length: int, batch_size: int, max_prompt_length=None) -> list[dict]:
    """Per-pair sequence log-probs under the policy and the reference (adapter disabled) and the DPO margin m."""
    collate = make_collate(tokenizer, max_length, max_prompt_length)
    records = []
    for start in tqdm(range(0, len(rows), batch_size), desc="held-out pairs"):
        chunk = rows[start:start + batch_size]
        batch = collate(chunk)
        pc, pr, rc, rr = pair_logps(model, batch)
        margin = dpo_margin(pc, pr, rc, rr)
        n_chosen = batch["chosen"]["response_mask"].sum(-1)
        n_rejected = batch["rejected"]["response_mask"].sum(-1)
        for i, row in enumerate(chunk):
            records.append({
                "prompt_id": row["prompt_id"],
                "length_stratum": row.get("length_stratum"),
                "policy_chosen": float(pc[i]),
                "policy_rejected": float(pr[i]),
                "ref_chosen": float(rc[i]),
                "ref_rejected": float(rr[i]),
                "margin": float(margin[i]),
                "chosen_tokens": int(n_chosen[i]),
                "rejected_tokens": int(n_rejected[i]),
            })
    return records


def pair_metrics(records: list[dict], beta: float) -> dict:
    """Held-out DPO loss at this run's beta plus the beta-free margin and preference accuracy (m > 0)."""
    col = lambda key: torch.tensor([r[key] for r in records], dtype=torch.float64)
    loss, stats = dpo_loss(col("policy_chosen"), col("policy_rejected"), col("ref_chosen"), col("ref_rejected"), beta)
    return {
        "n_pairs": len(records),
        "dpo_loss": float(loss),
        "preference_accuracy": float(stats["preference_accuracy"]),
        "margin_mean": float(stats["margin_mean"]),
        "chosen_reward_mean": float(stats["chosen_reward_mean"]),
        "rejected_reward_mean": float(stats["rejected_reward_mean"]),
    }


def stratified_metrics(records: list[dict], beta: float) -> dict:
    """Pair metrics on the whole length-stratified set and per stratum (preferred_longer / length_matched / rejected_longer)."""
    strata = sorted({r["length_stratum"] for r in records if r.get("length_stratum")})
    return {
        "all": pair_metrics(records, beta),
        "by_stratum": {s: pair_metrics([r for r in records if r["length_stratum"] == s], beta) for s in strata},
    }


# ----------------------------------------------------------------------------- generation

def heldout_prompt_items(cfg: dict, rows: list[dict], tokenizer):
    """The entire fixed prefix; long contexts use the shared prompt fitting rule."""
    return [{"prompt_id": row["prompt_id"], "messages": prompt_messages_from_preference(row)}
            for row in rows[: int(cfg["eval_generation_prompts"])]], []


def word_limit_eval(policy, tokenizer, cfg: dict) -> list[dict]:
    """Sample `word_limit_samples` responses per common word-limit prompt with the shared decoding config."""
    rows = read_jsonl(cfg["paths"]["word_limit_prompts"])
    samples = int(cfg.get("word_limit_samples", 1))
    jobs = [(row, s) for row in rows for s in range(samples)]
    batch_size = int(cfg.get("eval_batch_size", 8))
    set_seed(int(cfg["seed"]))
    records = []
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start:start + batch_size]
        gen = batch_generate(
            policy, tokenizer, [row["messages"] for row, _ in chunk],
            int(cfg["max_prompt_length"]), int(cfg["max_generation_tokens"]), **cfg["generation"],
        )
        for (row, sample), response, n_tok in zip(chunk, gen["responses"], gen["response_lengths"]):
            prompt = row["messages"][-1]["content"]
            records.append({
                "prompt_id": row["prompt_id"],
                "sample": sample,
                "prompt": prompt,
                "response": response,
                "response_tokens": int(n_tok),
                "word_limit": parse_word_limit(prompt),
                "words": word_count(response),
                "compliant": word_limit_compliance(prompt, response),
            })
    return records


def word_limit_summary(records: list[dict]) -> dict:
    """Compliance rate with the released parser plus word and token length statistics."""
    scored = [r for r in records if r["compliant"] is not None]
    return {
        "n_responses": len(scored),
        "compliance_rate": float(np.mean([r["compliant"] for r in scored])) if scored else float("nan"),
        "words": length_stats(r["words"] for r in scored),
        "words_over_limit_ratio_mean": float(np.mean([r["words"] / r["word_limit"] for r in scored])) if scored else float("nan"),
        "response_tokens": length_stats(r["response_tokens"] for r in scored),
    }


# ----------------------------------------------------------------------------- main

def evaluate_adapter(config_path: str, adapter: str | None, name: str, beta: float | None = None) -> dict:
    """Full Task 1 evaluation of one adapter: both held-out pair sets, sampled generations, word-limit prompts."""
    adapter = adapter_or_none(adapter)
    bundle = load_evaluation_bundle(config_path, adapter)
    cfg, tokenizer, policy = bundle["cfg"], bundle["tokenizer"], bundle["policy"]
    reward_model, reward_tokenizer = bundle["reward"]
    beta = float(cfg["beta"] if beta is None else beta)
    max_len = int(cfg["max_sequence_length"])
    pair_bs = int(cfg.get("eval_pair_batch_size", 4))
    pair_limit = cfg.get("eval_pair_limit")  # None = every held-out pair (smoke configs set a cap)
    out_dir = repo_path(cfg["results_dir"]) / "eval" / name

    standard_rows, dropped_standard = bundle["rows"][:pair_limit], []
    stratified_rows, dropped_stratified = read_jsonl(cfg["paths"]["dpo_length_eval"])[:pair_limit], []
    prompt_cap = int(cfg["max_prompt_length"])
    standard_pairs = score_pairs(policy, tokenizer, standard_rows, max_len, pair_bs, prompt_cap)
    stratified_pairs = score_pairs(policy, tokenizer, stratified_rows, max_len, pair_bs, prompt_cap)
    write_jsonl(out_dir / "pairs_standard.jsonl", standard_pairs)
    write_jsonl(out_dir / "pairs_stratified.jsonl", stratified_pairs)

    items, skipped_prompts = heldout_prompt_items(cfg, bundle["rows"], tokenizer)
    generations = evaluate_generations(
        policy, tokenizer, reward_model, reward_tokenizer, items,
        max_prompt_length=int(cfg["max_prompt_length"]),
        max_new_tokens=int(cfg["max_generation_tokens"]),
        generation=cfg["generation"],
        seed=int(cfg["seed"]),
        batch_size=int(cfg.get("eval_batch_size", 8)),
        reward_max_length=int(cfg.get("reward_max_length", 1024)),
    )
    write_jsonl(out_dir / "generations.jsonl", generations)

    word_limit = word_limit_eval(policy, tokenizer, cfg)
    write_jsonl(out_dir / "word_limit.jsonl", word_limit)

    metrics = {
        "name": name,
        "adapter": adapter,
        "beta": beta,
        "heldout_pairs": pair_metrics(standard_pairs, beta),
        "stratified_pairs": stratified_metrics(stratified_pairs, beta),
        "generation": summarize_generations(generations),
        "word_limit": word_limit_summary(word_limit),
        "dropped_overlong_pairs": {"standard": dropped_standard, "stratified": dropped_stratified},
        "skipped_generation_prompts": skipped_prompts,
        "prompt_truncation": "oldest context first; preserve latest role and assistant header; identical policy/reward conversation",
        "effective_generation_config": generations[0]["effective_generation_config"] if generations else {},
        "decoding": {**cfg["generation"], "max_new_tokens": int(cfg["max_generation_tokens"]), "seed": int(cfg["seed"])},
    }
    save_json(out_dir / "metrics.json", metrics)
    print(f"[{name}] acc={metrics['heldout_pairs']['preference_accuracy']:.3f} "
          f"loss={metrics['heldout_pairs']['dpo_loss']:.4f} reward={metrics['generation']['reward_mean']:.3f} "
          f"kl={metrics['generation']['kl_token_mean']:.4f} len={metrics['generation']['response_length']['mean']:.1f} -> {out_dir}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter directory, or 'none' for the SFT base policy")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--beta", type=float, help="beta used for the held-out DPO loss (default: config beta)")
    args = ap.parse_args()
    evaluate_adapter(args.config, args.adapter, args.name, args.beta)


if __name__ == "__main__":
    main()
