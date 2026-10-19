"""Matched short PPO forks shared by the clipping (step 2) and KL-pressure (step 3) studies.

A fork is identified by (clip_epsilon, kl_beta); the eps=0.20 / beta_KL=0.10 fork belongs to both
studies and is trained once. Every fork runs `fork_updates` updates from the identical supplied
policy + critic midpoint with the same prompt schedule and seed, then the common held-out evaluation.

Stability statistics (defined once, computed from the per-update training log):
  step_kl_mean / step_kl_max - k3 estimate of KL(pi_old || pi_new) on each update's own rollout,
                               i.e. how far one update moved the policy;
  grad_norm_max              - largest pre-clipping policy gradient norm within the fork;
  clip_fraction_mean         - mean fraction of response tokens with rho outside [1-eps, 1+eps].
"""

from __future__ import annotations

import pandas as pd

from common.data import read_jsonl, repo_path
from common.logging_utils import load_json
from common.runs import run_module


def fork_name(clip_epsilon: float, kl_beta: float) -> str:
    """Run name of a fork, e.g. fork_eps0.2_kl0.1."""
    return f"fork_eps{clip_epsilon:g}_kl{kl_beta:g}"


def run_fork(config_path: str, cfg: dict, clip_epsilon: float, kl_beta: float) -> str:
    """Train (if missing) and evaluate (if missing) one fork; returns its name."""
    name = fork_name(clip_epsilon, kl_beta)
    output = f"{cfg['fork_output_dir']}/{name}"
    run_module(
        "task2_ppo.continue_train", "--config", config_path, "--run-name", name, "--output", output,
        "--updates", cfg["fork_updates"], "--clip-epsilon", clip_epsilon, "--kl-beta", kl_beta,
        skip_if_exists=f"{output}/adapter_config.json",
    )
    run_module(
        "task2_ppo.evaluate", "--config", config_path, "--adapter", output, "--name", name,
        skip_if_exists=f"{cfg['results_dir']}/eval/{name}/metrics.json",
    )
    return name


def update_log(cfg: dict, name: str) -> pd.DataFrame:
    """Per-update training log of one fork as a DataFrame."""
    return pd.DataFrame(read_jsonl(repo_path(cfg["results_dir"]) / "train" / name / "update_log.jsonl"))


def fork_summary(cfg: dict, name: str, **labels) -> dict:
    """One table row: held-out metrics, token budget and the stability statistics defined above."""
    results = repo_path(cfg["results_dir"])
    log = update_log(cfg, name)
    heldout = load_json(results / "eval" / name / "metrics.json")
    train = load_json(results / "train" / name / "summary.json")
    return {
        **labels,
        "fork": name,
        "updates": len(log),
        "generated_tokens": train["generated_tokens"],
        "maximum_generated_token_budget": train.get("maximum_generated_token_budget"),
        "heldout_reward": heldout["reward_mean"],
        "heldout_kl": heldout["kl_token_mean"],
        "heldout_entropy": heldout["entropy_token_mean"],
        "heldout_length_mean": heldout["response_length"]["mean"],
        "heldout_length_std": heldout["response_length"]["std"],
        "heldout_truncation_rate": heldout["truncation_rate"],
        "train_reward_mean": float(log["reward"].mean()),
        "train_kl_last": float(log["kl"].iloc[-1]),
        "step_kl_mean": float(log["step_kl_old_new"].mean()),
        "step_kl_max": float(log["step_kl_old_new"].max()),
        "grad_norm_max": float(log["grad_norm_max"].max()),
        "clip_fraction_mean": float(log["clip_fraction"].mean()),
        "affected_fraction_mean": float(log["affected_fraction"].mean()),
    }


def trajectories(cfg: dict, names: list[str]) -> pd.DataFrame:
    """Long-format per-update logs of several forks (one row per fork x update) for plotting."""
    frames = [update_log(cfg, name).drop(columns=["prompt_ids"]).assign(fork=name) for name in names]
    return pd.concat(frames, ignore_index=True)
