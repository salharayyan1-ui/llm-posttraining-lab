"""Task 5: final RLVR-vs-RLAIF comparison combining the in-domain, diagnostic and transfer results.

Reads results/task5_feedback/{gsm,transfer}/metrics.json and diagnostics/summary.json; writes to
results/task5_feedback/:
  feedback_comparison.csv  per policy: GSM8K vs SVAMP accuracy, format compliance, AI win rate vs
                           SFT, mean length, and the in-domain -> transfer drop (absolute)
  failure_types.csv        failure-type counts per policy and dataset
  feedback_source.json     inputs for the feedback-source discussion:
                             coverage       how often each signal separates two responses
                             noise          judge order inconsistency, verifier-judge disagreement
                             exploitability wrong-preference rates on filler / distractor pairs
                             inference cost judge seconds per comparison vs a regex verifier
"""

from __future__ import annotations

import argparse

import pandas as pd

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json

POLICIES = ["sft", "rlvr", "rlaif"]
DATASETS = {"gsm": "in_domain", "transfer": "transfer"}


def policy_table(root) -> pd.DataFrame:
    """Per-policy in-domain vs transfer metrics and the drops between them."""
    metrics = {key: load_json(root / key / "metrics.json") for key in DATASETS}
    rows = []
    for policy in POLICIES:
        row = {"policy": policy}
        for key, label in DATASETS.items():
            m = metrics[key]["policies"][policy]
            row[f"{label}_accuracy"] = m["exact_accuracy"]
            row[f"{label}_format_compliance"] = m["format_compliance"]
            row[f"{label}_length_mean"] = m["response_tokens"]["mean"]
            row[f"{label}_length_std"] = m["response_tokens"]["std"]
            row[f"{label}_win_rate_vs_sft"] = metrics[key]["pairwise_vs_sft"].get(policy, {}).get("win_rate", 0.5 if policy == "sft" else float("nan"))
        row["accuracy_drop"] = row["in_domain_accuracy"] - row["transfer_accuracy"]
        row["win_rate_drop"] = row["in_domain_win_rate_vs_sft"] - row["transfer_win_rate_vs_sft"]
        rows.append(row)
    return pd.DataFrame(rows)


def failure_table(root) -> pd.DataFrame:
    """Failure-type counts per dataset and policy."""
    rows = []
    for key, label in DATASETS.items():
        for policy, m in load_json(root / key / "metrics.json")["policies"].items():
            rows.append({"dataset": label, "policy": policy, **m["failure_types"]})
    return pd.DataFrame(rows).fillna(0)


def feedback_source(root) -> dict:
    """Coverage / noise / exploitability / cost numbers collected from the saved results."""
    in_domain = load_json(root / "gsm" / "metrics.json")["pairwise_vs_sft"]
    diag = load_json(root / "diagnostics" / "summary.json")
    pairs = pd.read_csv(root / "diagnostics" / "pair_summary.csv").set_index(["pair", "mechanism"])
    wrong = lambda pair, mech: float(pairs.loc[(pair, mech), "wrong_rate"])
    tie = lambda pair, mech: float(pairs.loc[(pair, mech), "tie_rate"])
    return {
        "coverage": {
            "verifier_decisive_rate_vs_sft": {p: m["verifier_decisive_rate"] for p, m in in_domain.items()},
            "judge_decisive_when_verifier_ties": {p: m["judge_decisive_when_verifier_ties"] for p, m in in_domain.items()},
            "reasoning_pair_tie_rate": {"rlvr": tie("reasoning", "rlvr"), "rlaif": tie("reasoning", "rlaif")},
        },
        "noise": {
            "judge_order_consistency": diag["judge_order_consistency"],
            "verifier_judge_agreement_vs_sft": {p: m["verifier_judge_agreement"] for p, m in in_domain.items()},
            "judge_agreement_when_verifier_decisive": {p: m["agreement_when_verifier_decisive"] for p, m in in_domain.items()},
        },
        "exploitability": {
            pair: {"rlvr_wrong_rate": wrong(pair, "rlvr"), "rlaif_wrong_rate": wrong(pair, "rlaif")}
            for pair in ["filler", "distractor", "outcome", "reasoning"]
        },
        "sensitivity": diag["sensitivity"],
        "inference_cost": {
            "judge_seconds_per_comparison_in_domain": {p: m["judge_seconds_per_comparison"] for p, m in in_domain.items()},
            "judge_seconds_per_comparison_diagnostics": diag["judge_cost"]["seconds_per_new_comparison"],
            "judge_calls_per_group": "K*(K-1)/2 = 6 for K = 4 (one 3B forward + 4-token decode each)",
            "verifier": "one regex per response, no model call",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    root = repo_path(cfg["results_dir"]) / "task5_feedback"

    table = policy_table(root)
    failures = failure_table(root)
    table.to_csv(root / "feedback_comparison.csv", index=False)
    failures.to_csv(root / "failure_types.csv", index=False)
    save_json(root / "feedback_source.json", feedback_source(root))
    print(table.to_string(index=False))
    print(failures.to_string(index=False))
    print("Wrote", root / "feedback_comparison.csv", "and", root / "feedback_source.json")


if __name__ == "__main__":
    main()
