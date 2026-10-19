"""Task 2, step 3: reward-overoptimization (KL-pressure) study.

Matched short forks (task2_ppo/forks.py) with beta_KL in `kl_values` at the release epsilon, all
from the identical supplied midpoint. Writes
  kl_forks.csv         held-out reward / KL / entropy / length + stability statistics per beta_KL;
  kl_trajectories.csv  per-update reward, KL, entropy, length, ... of every fork, to see which
                       observable moves first when KL pressure is weakened.
Held-out generations for qualitative inspection are in results/task2_ppo/eval/<fork>/generations.jsonl.
"""

from __future__ import annotations

import argparse

import pandas as pd

from common.data import load_yaml, repo_path
from task2_ppo.forks import fork_name, fork_summary, run_fork, trajectories


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--summary-only", action="store_true", help="only rebuild the tables from saved results")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    eps = float(cfg["clip_epsilon"])

    names, rows = [], []
    for kl_beta in map(float, cfg["kl_values"]):
        if not args.summary_only:
            run_fork(args.config, cfg, eps, kl_beta)
        name = fork_name(eps, kl_beta)
        names.append(name)
        rows.append(fork_summary(cfg, name, epsilon=eps, kl_beta=kl_beta))

    results = repo_path(cfg["results_dir"])
    table = pd.DataFrame(rows)
    table.to_csv(results / "kl_forks.csv", index=False)
    trajectories(cfg, names).to_csv(results / "kl_trajectories.csv", index=False)
    print(table.to_string(index=False))
    print("Wrote", results / "kl_forks.csv", "and", results / "kl_trajectories.csv")


if __name__ == "__main__":
    main()
