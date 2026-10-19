"""Task 3, step 2: group-size study on the supplied completion/reward cache (no training).

The cache holds 8 sampled completions with learned-reward scores for each prompt. Every K in
`group_sizes` spends exactly that budget (8 x n_prompts generations):

  partition rule  - each prompt's 8 completions are split into 8/K disjoint groups of size K
                    (K=2 -> 4 groups per prompt, K=4 -> 2, K=8 -> 1); every completion is used once.
                    The primary partition is contiguous in generation_index; the same metrics are
                    also averaged over `group_size_resamples` seeded random partitions so the result
                    does not depend on that arbitrary order.
  difficulty bins - prompts are ranked by mean reward over all 8 completions and split into
                    tertiles: hard (lowest mean reward), medium, easy. Defined once, reused for every K.

Metrics (overall and per difficulty bin), using the same helpers as training (task3_grpo.grpo):
  informative_rate           groups whose reward std > 1e-6 (the released helper's tolerance)
  informative_rate_practical groups whose std > practical_std_threshold; the reward model is
                             continuous, so tiny differences are "informative" yet carry little signal
  mean_group_std             mean within-group population std of the reward
  centered_signal_var        variance of r - mu_group (size of the raw relative signal)
  advantage_var              variance of the normalized group-relative advantage
  sign_agreement_k8          fraction of completions whose advantage sign equals the sign it gets in
                             its full 8-completion group (does a small group push the same way?)
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from task3_grpo.grpo import group_relative_advantages, group_reward_stats

FULL_K = 8
BINS = ["hard", "medium", "easy"]


def load_k8_cache(path):
    """Cached completions grouped by prompt and sorted by generation_index (released loader)."""
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int, rng: np.random.Generator | None = None):
    """Split each prompt's first 8 completions into 8 // k disjoint groups of size k.

    rng=None keeps generation_index order (contiguous groups); otherwise each prompt's completions
    are permuted first. Returns a list of (prompt_key, [rows]) groups.
    """
    if FULL_K % k:
        raise ValueError(f"K={k} must divide {FULL_K}")
    groups = []
    for pid, rows in by_prompt.items():
        rows = rows[:FULL_K]
        order = np.arange(FULL_K) if rng is None else rng.permutation(FULL_K)
        for g in range(FULL_K // k):
            groups.append((pid, [rows[i] for i in order[g * k:(g + 1) * k]]))
    return groups


def group_tensors(groups):
    """Flatten groups into rewards [N], group ids [N] and (prompt, generation_index) keys."""
    k = len(groups[0][1])
    rewards = torch.tensor([r["reward"] for _, g in groups for r in g], dtype=torch.float64)
    ids = torch.arange(len(groups)).repeat_interleave(k)
    keys = [(pid, int(r["generation_index"])) for pid, g in groups for r in g]
    return rewards, ids, keys


def full_group_signs(by_prompt) -> dict:
    """Advantage sign of every completion inside its full 8-completion group (the reference)."""
    rewards, ids, keys = group_tensors(regroup_equal_generation_budget(by_prompt, FULL_K))
    signs = torch.sign(group_relative_advantages(rewards, ids))
    return dict(zip(keys, signs.tolist()))


def group_metrics(groups, reference_signs: dict, practical: float) -> dict:
    """Every study metric for one set of equal-size groups (definitions in the module docstring)."""
    rewards, ids, keys = group_tensors(groups)
    mean, std, informative, inverse = group_reward_stats(rewards, ids)
    adv = group_relative_advantages(rewards, ids)
    ref = torch.tensor([reference_signs[key] for key in keys], dtype=torch.float64)
    decided = ref != 0
    return {
        "n_groups": len(groups),
        "n_generations": int(rewards.numel()),
        "informative_rate": float(informative.double().mean()),
        "informative_rate_practical": float((std > practical).double().mean()),
        "mean_group_std": float(std.mean()),
        "centered_signal_var": float((rewards - mean[inverse]).var(unbiased=False)),
        "advantage_var": float(adv.var(unbiased=False)),
        "sign_agreement_k8": float((torch.sign(adv)[decided] == ref[decided]).double().mean()) if decided.any() else float("nan"),
    }


def difficulty_bins(by_prompt) -> pd.DataFrame:
    """Tertiles of per-prompt mean reward over all 8 completions: hard < medium < easy."""
    table = pd.DataFrame([{
        "prompt_key": pid,
        "prompt_id": rows[0].get("prompt_id"),
        "mean_reward": float(np.mean([r["reward"] for r in rows[:FULL_K]])),
        "std_reward": float(np.std([r["reward"] for r in rows[:FULL_K]])),
        "clipped_at_max": int(sum(bool(r.get("clipped_at_max")) for r in rows[:FULL_K])),
    } for pid, rows in by_prompt.items()])
    table["bin"] = pd.qcut(table["mean_reward"].rank(method="first"), q=len(BINS), labels=BINS).astype(str)
    return table.sort_values("mean_reward").reset_index(drop=True)


def study(by_prompt, cfg) -> pd.DataFrame:
    """Metrics for every K x difficulty subset, for the contiguous and the random partitions."""
    practical = float(cfg.get("practical_std_threshold", 0.1))
    resamples = int(cfg.get("group_size_resamples", 0))
    reference = full_group_signs(by_prompt)
    bins = difficulty_bins(by_prompt).set_index("prompt_key")["bin"].to_dict()
    subsets = {"all": lambda pid: True, **{b: (lambda pid, b=b: bins[pid] == b) for b in BINS}}

    rows = []
    for k in map(int, cfg["group_sizes"]):
        contiguous = regroup_equal_generation_budget(by_prompt, k)
        rng = np.random.default_rng(int(cfg["seed"]))
        random_partitions = [regroup_equal_generation_budget(by_prompt, k, rng) for _ in range(resamples)]
        for subset, keep in subsets.items():
            pick = lambda groups: [g for g in groups if keep(g[0])]
            rows.append({"K": k, "subset": subset, "partition": "contiguous", **group_metrics(pick(contiguous), reference, practical)})
            if random_partitions:
                draws = pd.DataFrame([group_metrics(pick(groups), reference, practical) for groups in random_partitions])
                rows.append({"K": k, "subset": subset, "partition": "random_mean", **draws.mean().to_dict()})
                rows.append({"K": k, "subset": subset, "partition": "random_std", **draws.std(ddof=0).to_dict()})
    return pd.DataFrame(rows)


def group_examples(by_prompt, n: int = 2, chars: int = 300) -> dict:
    """Lowest- and highest-spread prompts with truncated completions, for qualitative evidence."""
    table = difficulty_bins(by_prompt).sort_values("std_reward")
    show = lambda pid: [{"generation_index": r["generation_index"], "reward": r["reward"],
                         "completion_tokens": r.get("completion_tokens"), "completion": r["completion"][:chars]}
                        for r in by_prompt[pid][:FULL_K]]
    return {
        "lowest_reward_spread": [{"prompt_key": p, "completions": show(p)} for p in table["prompt_key"].head(n)],
        "highest_reward_spread": [{"prompt_key": p, "completions": show(p)} for p in table["prompt_key"].tail(n)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    print("Cached prompts:", len(by_prompt), "| generations:", FULL_K * len(by_prompt), "| K:", cfg["group_sizes"])

    out = repo_path(cfg["results_dir"]) / "group_size"
    out.mkdir(parents=True, exist_ok=True)
    summary = study(by_prompt, cfg)
    summary.to_csv(out / "group_size_summary.csv", index=False)
    difficulty_bins(by_prompt).to_csv(out / "prompt_difficulty.csv", index=False)
    save_json(out / "group_examples.json", group_examples(by_prompt))
    save_json(out / "protocol.json", {
        "generations_per_prompt": FULL_K,
        "n_prompts": len(by_prompt),
        "partition": "8/K disjoint groups per prompt; contiguous by generation_index + seeded random partitions",
        "resamples": int(cfg.get("group_size_resamples", 0)),
        "difficulty_bins": "tertiles of per-prompt mean reward over all 8 completions (hard < medium < easy)",
        "informative_tolerance": 1e-6,
        "practical_std_threshold": float(cfg.get("practical_std_threshold", 0.1)),
    })
    print(summary[summary["partition"] != "random_std"].to_string(index=False))
    print("Wrote", out)


if __name__ == "__main__":
    main()
