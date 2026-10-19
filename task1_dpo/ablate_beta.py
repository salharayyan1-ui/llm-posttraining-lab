"""Task 1, step 2: regularization-strength (beta) study.

Each fork starts from the original policy initialization and trains one pass over the first
`short_ablation_examples` pairs of the standard training file with the same seed, optimizer, data
order and LoRA initialization; only beta changes. Every fork is evaluated with the common protocol
(task1_dpo.evaluate). The summary also lists the one-epoch standard run and the SFT base so the
table can be reported in one place; the `train_examples` / `budget` columns make the different
training budgets explicit, as the manual requires.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common.data import load_yaml, repo_path
from common.logging_utils import load_json
from common.runs import run_module


def fork_name(beta: float) -> str:
    """Run name of a beta fork, e.g. beta_0.03."""
    return f"beta_{beta:g}"


def summary_row(results_dir, name: str, budget: str) -> dict | None:
    """Flatten one condition's eval metrics (+ training budget) into a table row; None if not evaluated."""
    eval_path = results_dir / "eval" / name / "metrics.json"
    if not eval_path.exists():
        return None
    m = load_json(eval_path)
    train_path = results_dir / "train" / name / "run_config.json"
    train = load_json(train_path) if train_path.exists() else {}
    gen, pairs = m["generation"], m["heldout_pairs"]
    return {
        "condition": name,
        "budget": budget,
        "beta": m["beta"],
        "train_examples": train.get("examples", 0),
        "optimizer_steps": train.get("optimizer_steps", 0),
        "heldout_dpo_loss": pairs["dpo_loss"],
        "preference_accuracy": pairs["preference_accuracy"],
        "margin_mean": pairs["margin_mean"],
        "kl_token_mean": gen["kl_token_mean"],
        "kl_sequence_mean": gen["kl_sequence_mean"],
        "reward_mean": gen["reward_mean"],
        "reward_std": gen["reward_std"],
        "entropy_token_mean": gen["entropy_token_mean"],
        "length_mean": gen["response_length"]["mean"],
        "length_std": gen["response_length"]["std"],
        "length_iqr": gen["response_length"]["iqr"],
        "truncation_rate": gen["truncation_rate"],
        "word_limit_compliance": m["word_limit"]["compliance_rate"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--summary-only", action="store_true", help="only rebuild the summary table")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results_dir = repo_path(cfg["results_dir"])
    n_short = int(cfg["short_ablation_examples"])

    if not args.summary_only:
        for beta in cfg["betas"]:
            name = fork_name(beta)
            output = (Path(cfg["standard_output"]).parent / name).as_posix()  # next to the standard adapter
            run_module(
                "task1_dpo.train", "--config", args.config, "--run-name", name, "--output", output,
                "--beta", beta, "--max-examples", n_short,
                skip_if_exists=f"{output}/adapter_config.json",
            )
            run_module(
                "task1_dpo.evaluate", "--config", args.config, "--adapter", output, "--name", name, "--beta", beta,
                skip_if_exists=f"{cfg['results_dir']}/eval/{name}/metrics.json",
            )

    conditions = [("sft", "no training"), ("standard", "full epoch")]
    conditions += [(fork_name(b), f"short run ({n_short} pairs)") for b in cfg["betas"]]
    rows = [r for r in (summary_row(results_dir, name, budget) for name, budget in conditions) if r]
    table = pd.DataFrame(rows)
    results_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(results_dir / "beta_summary.csv", index=False)
    print(table.to_string(index=False))
    print("Wrote", results_dir / "beta_summary.csv")


if __name__ == "__main__":
    main()
