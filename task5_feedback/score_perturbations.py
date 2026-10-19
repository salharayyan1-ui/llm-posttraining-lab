"""Task 5, step 2: reward sensitivity on the controlled diagnostic set (20 problems x 5 variants).

Controlled pairs (diagnostically better response first):
  reasoning   clean_correct vs corrupt_reasoning_correct_final  same correct final, reasoning degraded
  outcome     clean_correct vs good_reasoning_wrong_final       reasoning ~fixed, final answer changed
  distractor  clean_correct vs gold_distractor_wrong_final      gold number mentioned, wrong final
  filler      clean_correct vs persuasive_filler_correct        same final, irrelevant persuasion added
Each mechanism prefers the better response, ties, or prefers the worse one ("wrong"):
  RLVR  - exact verifier reward of each response (binary, so ties are expected when finals agree);
  RLAIF - the fixed judge's head-to-head decision. The reversed presentation is judged as well
          (`rlaif_reversed`), which measures position/order consistency, i.e. judge noise.
S_reason  = better-rate on the reasoning pair;
S_outcome = better-rate over the outcome-changing pairs (outcome + distractor).
Every variant also gets the direct-RLAIF group reward (wins + 0.5 ties) / (K - 1) from a round
robin over the problem's 5 variants, so all five categories are scored by both mechanisms.

Outputs: results/task5_feedback/diagnostics/{pairs.jsonl, variant_scores.csv, pair_summary.csv,
summary.json, disagreement_examples.json}
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import PairwiseAIJudge

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}
VARIANT_ORDER = [  # clean first, so the round robin's compare(clean, X) calls are the controlled pairs
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "gold_distractor_wrong_final",
    "persuasive_filler_correct",
]
PAIRS = {
    "reasoning": ("clean_correct", "corrupt_reasoning_correct_final"),
    "outcome": ("clean_correct", "good_reasoning_wrong_final"),
    "distractor": ("clean_correct", "gold_distractor_wrong_final"),
    "filler": ("clean_correct", "persuasive_filler_correct"),
}
OUTCOME_PAIRS = ["outcome", "distractor"]
MECHANISMS = ["rlvr", "rlaif", "rlaif_reversed"]
JUDGE_TO_OUTCOME = {"A": "better", "B": "wrong", "TIE": "tie"}  # A = the better response


def load_diagnostic_groups(path):
    """Diagnostic rows grouped by problem: {problem_id: {variant_type: row}} (released loader)."""
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def verifier_outcome(better: float, worse: float) -> str:
    """better / tie / wrong from the verifier rewards of the better and the worse response."""
    return "better" if better > worse else "wrong" if better < worse else "tie"


def score_problem(judge: PairwiseAIJudge, pid: str, variants: dict):
    """Score one problem's 5 variants with both mechanisms; returns per-variant rows and per-pair rows."""
    question = variants["clean_correct"]["question"]
    responses = {v: variants[v]["response"] for v in VARIANT_ORDER}
    exact = {v: exact_reward(responses[v], variants[v]["gold_final"]) for v in VARIANT_ORDER}
    round_robin = dict(zip(VARIANT_ORDER, judge.group_rewards(question, [responses[v] for v in VARIANT_ORDER])))

    variant_rows = [{
        "problem_id": pid,
        "variant": v,
        "rlvr_reward": exact[v],
        "expected_exact_reward": float(variants[v]["expected_exact_reward"]),
        "rlaif_round_robin": round_robin[v],
    } for v in VARIANT_ORDER]

    pair_rows = []
    for pair, (better, worse) in PAIRS.items():
        # Explicit physical A/B reversal. Independent hash balancing could present
        # the same physical order twice and inflate the consistency statistic.
        forward = JUDGE_TO_OUTCOME[judge.compare(question, responses[better], responses[worse], balance_orientation=False)]
        backward = {"A": "wrong", "B": "better", "TIE": "tie"}[judge.compare(question, responses[worse], responses[better], balance_orientation=False)]
        pair_rows.append({
            "problem_id": pid,
            "pair": pair,
            "rlvr": verifier_outcome(exact[better], exact[worse]),
            "rlaif": forward,
            "rlaif_reversed": backward,
            "rlaif_order_consistent": forward == backward,
        })
    return variant_rows, pair_rows


def rates(outcomes: pd.Series) -> dict:
    """better / tie / wrong rates of a column of outcomes."""
    return {f"{k}_rate": float((outcomes == k).mean()) for k in ["better", "tie", "wrong"]}


def pair_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    """better / tie / wrong rates per controlled pair and mechanism."""
    return pd.DataFrame([
        {"pair": pair, "mechanism": m, "n": len(d), **rates(d[m])}
        for pair, d in pairs.groupby("pair") for m in MECHANISMS
    ])


def sensitivity(pairs: pd.DataFrame) -> dict:
    """S_reason, S_outcome and their tie / wrong rates per mechanism."""
    out = {}
    for m in MECHANISMS:
        reason = rates(pairs.loc[pairs["pair"] == "reasoning", m])
        outcome = rates(pairs.loc[pairs["pair"].isin(OUTCOME_PAIRS), m])
        out[m] = {
            "S_reason": reason["better_rate"], "reason_tie_rate": reason["tie_rate"], "reason_wrong_rate": reason["wrong_rate"],
            "S_outcome": outcome["better_rate"], "outcome_tie_rate": outcome["tie_rate"], "outcome_wrong_rate": outcome["wrong_rate"],
        }
    return out


def disagreement_examples(pairs: pd.DataFrame, groups: dict, per_pair: int = 2, chars: int = 700) -> dict:
    """Pairs on which the verifier and the judge reach different outcomes, per perturbation type."""
    out = {}
    for pair, (better, worse) in PAIRS.items():
        d = pairs[(pairs["pair"] == pair) & (pairs["rlvr"] != pairs["rlaif"])].head(per_pair)
        out[pair] = [{
            "problem_id": r["problem_id"],
            "rlvr": r["rlvr"],
            "rlaif": r["rlaif"],
            "question": groups[r["problem_id"]][better]["question"],
            "better_response": groups[r["problem_id"]][better]["response"][:chars],
            "worse_response": groups[r["problem_id"]][worse]["response"][:chars],
        } for _, r in d.iterrows()]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    groups = dict(list(groups.items())[: cfg.get("diagnostic_problems")])  # None = all 20 (smoke configs cap it)
    print("Diagnostic problems:", len(groups))

    judge = PairwiseAIJudge(cfg, cfg["judge_cache"])
    cached_before = len(judge.cache)
    start = time.perf_counter()
    variant_rows, pair_rows = [], []
    for pid, variants in groups.items():
        v_rows, p_rows = score_problem(judge, pid, variants)
        variant_rows += v_rows
        pair_rows += p_rows
    judge_seconds = time.perf_counter() - start
    new_comparisons = len(judge.cache) - cached_before

    out = repo_path(cfg["results_dir"]) / "task5_feedback" / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    pairs = pd.DataFrame(pair_rows)
    variants = pd.DataFrame(variant_rows)
    write_jsonl(out / "pairs.jsonl", pair_rows)
    variants.to_csv(out / "variant_scores.csv", index=False)
    summary_table = pair_summary(pairs)
    summary_table.to_csv(out / "pair_summary.csv", index=False)
    summary = {
        "n_problems": len(groups),
        "sensitivity": sensitivity(pairs),
        "variant_means": variants.groupby("variant")[["rlvr_reward", "rlaif_round_robin"]].mean().to_dict(orient="index"),
        "verifier_vs_expected_mismatches": int((variants["rlvr_reward"] != variants["expected_exact_reward"]).sum()),
        "judge_order_consistency": float(pairs["rlaif_order_consistent"].mean()),
        "judge_cost": {
            "new_comparisons": new_comparisons,
            "seconds": judge_seconds,
            "seconds_per_new_comparison": judge_seconds / new_comparisons if new_comparisons else None,
            "comparisons_per_group_of_K": "K*(K-1)/2 (6 for K=4); the verifier needs one regex per response",
        },
    }
    save_json(out / "summary.json", summary)
    save_json(out / "disagreement_examples.json", disagreement_examples(pairs, groups))
    print(summary_table.to_string(index=False))
    print(pd.DataFrame(summary["sensitivity"]).T.to_string())


if __name__ == "__main__":
    main()
