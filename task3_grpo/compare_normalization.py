"""Task 3, step 3: canonical GRPO vs Dr. GRPO-style sequence normalization.

Two matched short continuations from the identical midpoint (`fork_updates` updates each, same
prompts, seed, reward, beta, epsilon, K and generation settings); only `loss_type` differs. Then:

* held-out reward / KL / entropy / length with the common protocol (midpoint evaluated as reference);
* length-conditioned gradient allocation from each fork's completions.jsonl, where for every
  completion in the loss grad_mass = sum_t |dL/d log pi(y_t)| and grad_per_token = grad_mass / T.
  Canonical GRPO divides each completion by its own length T, so grad_mass should not grow with T
  (grad_per_token ~ 1/T); Dr. GRPO divides by a constant, so grad_mass should grow with T.
  Reported: Spearman correlation of length with both statistics, the share of gradient mass on
  completions longer than their group's median, and length-quartile means split by advantage sign
  (quartile edges pooled over both forks so the bins are identical).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.runs import run_module

FORKS = {"grpo": "norm_grpo", "dr_grpo": "norm_dr_grpo"}


def run_forks(config_path: str, cfg: dict) -> None:
    """Train both normalization forks and evaluate them plus the midpoint (finished steps are skipped)."""
    for loss_type, name in FORKS.items():
        output = f"{cfg['fork_output_dir']}/{name}"
        run_module(
            "task3_grpo.continue_train", "--config", config_path, "--run-name", name, "--output", output,
            "--updates", cfg["fork_updates"], "--loss-type", loss_type,
            skip_if_exists=f"{output}/adapter_config.json",
        )
    evals = {name: f"{cfg['fork_output_dir']}/{name}" for name in FORKS.values()}
    evals["midpoint"] = cfg["paths"]["grpo_midpoint_policy"]
    for name, adapter in evals.items():
        run_module("task3_grpo.evaluate", "--config", config_path, "--adapter", adapter, "--name", name,
                   skip_if_exists=f"{cfg['results_dir']}/eval/{name}/metrics.json")


def spearman(a, b) -> float:
    """Rank correlation without scipy (Pearson correlation of ranks)."""
    return float(pd.Series(np.asarray(a, dtype=float)).rank().corr(pd.Series(np.asarray(b, dtype=float)).rank()))


def completions(cfg: dict) -> pd.DataFrame:
    """Completions that actually carry policy-gradient signal (in the loss, non-zero advantage)."""
    frames = []
    for loss_type, name in FORKS.items():
        df = pd.DataFrame(read_jsonl(repo_path(cfg["results_dir"]) / "train" / name / "completions.jsonl"))
        frames.append(df.assign(loss_type=loss_type))
    df = pd.concat(frames, ignore_index=True)
    return df[df["in_loss"] & (df["advantage"].abs() > 1e-8)].copy()


def allocation_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per loss type: length/gradient correlations and the gradient share of longer-than-median completions."""
    rows = []
    for loss_type, d in df.groupby("loss_type"):
        group_median = d.groupby(["update", "group"])["response_tokens"].transform("median")
        longer = d["response_tokens"] > group_median
        rows.append({
            "loss_type": loss_type,
            "n_completions": len(d),
            "mean_length": float(d["response_tokens"].mean()),
            "spearman_length_vs_grad_mass": spearman(d["response_tokens"], d["grad_mass"]),
            "spearman_length_vs_grad_per_token": spearman(d["response_tokens"], d["grad_per_token"]),
            "grad_mass_share_longer_than_group_median": float(d.loc[longer, "grad_mass"].sum() / d["grad_mass"].sum()),
        })
    return pd.DataFrame(rows)


def length_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Mean gradient statistics per length quartile and advantage sign."""
    df["length_bin"] = pd.qcut(df["response_tokens"], q=4, duplicates="drop").astype(str)
    df["advantage_sign"] = np.where(df["advantage"] > 0, "positive", "negative")
    return (df.groupby(["loss_type", "length_bin", "advantage_sign"])
              .agg(n=("grad_mass", "size"), mean_length=("response_tokens", "mean"),
                   grad_mass=("grad_mass", "mean"), grad_per_token=("grad_per_token", "mean"))
              .reset_index())


def heldout_table(cfg: dict) -> pd.DataFrame:
    """Held-out reward / KL / entropy / length of the midpoint and both forks."""
    rows = []
    for name in ["midpoint", *FORKS.values()]:
        m = load_json(repo_path(cfg["results_dir"]) / "eval" / name / "metrics.json")
        train = load_json(repo_path(cfg["results_dir"]) / "train" / name / "summary.json") if name != "midpoint" else {}
        rows.append({
            "condition": name,
            "updates": train.get("updates"),
            "generated_tokens": train.get("generated_tokens"),
            "maximum_generated_token_budget": train.get("maximum_generated_token_budget"),
            "reward_mean": m["reward_mean"],
            "kl_token_mean": m["kl_token_mean"],
            "entropy_token_mean": m["entropy_token_mean"],
            "length_mean": m["response_length"]["mean"],
            "length_std": m["response_length"]["std"],
            "truncation_rate": m["truncation_rate"],
        })
    return pd.DataFrame(rows)


def length_contrast_examples(cfg: dict, n: int = 5, chars: int = 600) -> list[dict]:
    """Held-out prompts where the two forks' response lengths differ most (for qualitative reading)."""
    load = lambda name: {r["prompt_id"]: r for r in read_jsonl(repo_path(cfg["results_dir"]) / "eval" / name / "generations.jsonl")}
    grpo, dr = load(FORKS["grpo"]), load(FORKS["dr_grpo"])
    shared = sorted((pid for pid in grpo if pid in dr), key=lambda pid: -abs(grpo[pid]["response_tokens"] - dr[pid]["response_tokens"]))
    return [{
        "prompt_id": pid,
        "prompt": grpo[pid]["prompt"][:chars],
        "grpo": {k: grpo[pid][k] for k in ("response_tokens", "reward", "truncated")} | {"response": grpo[pid]["response"][:chars]},
        "dr_grpo": {k: dr[pid][k] for k in ("response_tokens", "reward", "truncated")} | {"response": dr[pid]["response"][:chars]},
    } for pid in shared[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--summary-only", action="store_true", help="only analyze saved results")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if not args.summary_only:
        run_forks(args.config, cfg)

    out = repo_path(cfg["results_dir"]) / "normalization"
    out.mkdir(parents=True, exist_ok=True)
    df = completions(cfg)
    allocation = allocation_summary(df)
    bins = length_bins(df)
    heldout = heldout_table(cfg)
    trajectories = pd.concat([
        pd.DataFrame(read_jsonl(repo_path(cfg["results_dir"]) / "train" / name / "update_log.jsonl")).drop(columns=["prompt_ids"])
        for name in FORKS.values()
    ], ignore_index=True)

    allocation.to_csv(out / "gradient_allocation.csv", index=False)
    bins.to_csv(out / "gradient_by_length_bin.csv", index=False)
    heldout.to_csv(out / "heldout.csv", index=False)
    trajectories.to_csv(out / "trajectories.csv", index=False)
    save_json(out / "length_contrast_examples.json", length_contrast_examples(cfg))
    print(heldout.to_string(index=False))
    print(allocation.to_string(index=False))
    print("Wrote", out)


if __name__ == "__main__":
    main()
