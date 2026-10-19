"""Task 1, step 3: length-confounding study.

1. Dataset side: how strongly is response length tied to the preference label in each file?
   (A property of the data, reported separately from the policy's own length shift.)
2. Train the length-balanced condition: same config as the standard run, balanced training file.
3. Evaluate it (and, if missing, the standard and SFT models) with the common protocol.
4. Standard vs length-balanced: per-stratum preference accuracy/margin, which pairs each model
   ranks correctly, generated length and word-limit compliance.
5. Dump qualitative candidates for manual reading (reward up while length/compliance gets worse).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import load_yaml, preference_responses, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.metrics import length_stats
from common.models import load_tokenizer
from common.runs import run_module

MODELS = ["sft", "standard", "length_balanced"]


# ----------------------------------------------------------------------------- dataset side

def dataset_length_stats(rows: list[dict], tokenizer) -> dict:
    """Token-length relation between chosen and rejected responses (same tokenizer for every file)."""
    n_tok = lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"])
    lengths = np.array([[n_tok(c), n_tok(r)] for c, r in map(preference_responses, rows)], dtype=float)
    diff = lengths[:, 0] - lengths[:, 1]
    out = {
        "n_pairs": len(rows),
        "chosen_tokens": length_stats(lengths[:, 0]),
        "rejected_tokens": length_stats(lengths[:, 1]),
        "length_difference": length_stats(diff),
        "frac_chosen_longer": float((diff > 0).mean()),
        "frac_rejected_longer": float((diff < 0).mean()),
    }
    if "score_chosen" in rows[0]:
        gap = pd.Series([r["score_chosen"] - r["score_rejected"] for r in rows])
        # Spearman = Pearson on ranks (avoids a scipy dependency).
        out["spearman_score_gap_vs_length_diff"] = float(gap.rank().corr(pd.Series(diff).rank()))
    if "length_stratum" in rows[0]:
        out["strata"] = pd.Series([r["length_stratum"] for r in rows]).value_counts().to_dict()
    return out


# ----------------------------------------------------------------------------- model side

def eval_dir(results_dir, name):
    """Evaluation output folder of one condition."""
    return results_dir / "eval" / name


def strata_table(results_dir) -> pd.DataFrame:
    """Per-stratum held-out metrics of the standard and length-balanced models (one row per model x stratum)."""
    rows = []
    for name in ["standard", "length_balanced"]:
        by_stratum = load_json(eval_dir(results_dir, name) / "metrics.json")["stratified_pairs"]["by_stratum"]
        for stratum, m in by_stratum.items():
            rows.append({"model": name, "stratum": stratum, **m})
    return pd.DataFrame(rows)


def agreement_table(results_dir) -> pd.DataFrame:
    """Per stratum: fraction of held-out pairs ranked correctly (m > 0) by both / only one / neither model."""
    load = lambda name: {r["prompt_id"]: r for r in read_jsonl(eval_dir(results_dir, name) / "pairs_stratified.jsonl")}
    std, bal = load("standard"), load("length_balanced")
    shared = [pid for pid in std if pid in bal]
    rows = []
    for stratum in sorted({std[pid]["length_stratum"] for pid in shared}):
        ids = [pid for pid in shared if std[pid]["length_stratum"] == stratum]
        a = np.array([std[pid]["margin"] > 0 for pid in ids])
        b = np.array([bal[pid]["margin"] > 0 for pid in ids])
        rows.append({
            "stratum": stratum,
            "n": len(ids),
            "both_correct": float((a & b).mean()),
            "only_standard": float((a & ~b).mean()),
            "only_length_balanced": float((~a & b).mean()),
            "neither": float((~a & ~b).mean()),
        })
    return pd.DataFrame(rows)


def behaviour_summary(results_dir) -> dict:
    """Generated-behaviour numbers per model: reward, KL, length and word-limit compliance."""
    out = {}
    for name in MODELS:
        m = load_json(eval_dir(results_dir, name) / "metrics.json")
        out[name] = {
            "reward_mean": m["generation"]["reward_mean"],
            "kl_token_mean": m["generation"]["kl_token_mean"],
            "response_length": m["generation"]["response_length"],
            "truncation_rate": m["generation"]["truncation_rate"],
            "word_limit": m["word_limit"],
        }
    return out


# ----------------------------------------------------------------------------- qualitative

def qualitative_candidates(results_dir, top_k: int = 8) -> dict:
    """Candidates only: a human still has to judge which ones show a real quality problem."""
    gens = {name: {r["prompt_id"]: r for r in read_jsonl(eval_dir(results_dir, name) / "generations.jsonl")} for name in MODELS}
    reward_vs_length = []
    for pid, std in gens["standard"].items():
        sft = gens["sft"].get(pid)
        if sft and std["reward"] > sft["reward"] and std["response_tokens"] >= 1.5 * max(sft["response_tokens"], 1):
            reward_vs_length.append({
                "prompt_id": pid,
                "prompt": std["prompt"],
                "reward_gain": std["reward"] - sft["reward"],
                "tokens": {"sft": sft["response_tokens"], "standard": std["response_tokens"]},
                "truncated_standard": std["truncated"],
                "sft_response": sft["response"],
                "standard_response": std["response"],
            })
    reward_vs_length.sort(key=lambda r: -r["reward_gain"])

    word_limit_violations = []
    for name in ["standard", "length_balanced"]:
        for r in read_jsonl(eval_dir(results_dir, name) / "word_limit.jsonl"):
            if r["compliant"] == 0.0:
                word_limit_violations.append({"model": name, **r, "overshoot": r["words"] / r["word_limit"]})
    word_limit_violations.sort(key=lambda r: -r["overshoot"])
    return {
        "reward_up_and_much_longer_than_sft": reward_vs_length[:top_k],
        "word_limit_violations": word_limit_violations[: 2 * top_k],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--analysis-only", action="store_true", help="skip training/evaluation, only analyze saved results")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results_dir = repo_path(cfg["results_dir"])
    eval_metrics = lambda name: f"{cfg['results_dir']}/eval/{name}/metrics.json"

    if not args.analysis_only:
        run_module(
            "task1_dpo.train", "--config", args.config, "--run-name", "length_balanced",
            "--dataset", cfg["paths"]["dpo_length_train"], "--output", cfg["length_output"],
            skip_if_exists=f"{cfg['length_output']}/adapter_config.json",
        )
        for name, adapter in [("length_balanced", cfg["length_output"]), ("standard", cfg["standard_output"]), ("sft", "none")]:
            run_module("task1_dpo.evaluate", "--config", args.config, "--adapter", adapter, "--name", name,
                       skip_if_exists=eval_metrics(name))

    tokenizer = load_tokenizer(cfg["base_model"])
    datasets = {
        "standard_train": cfg["paths"]["dpo_standard_train"],
        "standard_eval": cfg["paths"]["dpo_standard_eval"],
        "length_balanced_train": cfg["paths"]["dpo_length_train"],
        "length_stratified_eval": cfg["paths"]["dpo_length_eval"],
    }
    dataset_side = {key: dataset_length_stats(read_jsonl(path), tokenizer) for key, path in datasets.items()}

    strata = strata_table(results_dir)
    agreement = agreement_table(results_dir)
    strata.to_csv(results_dir / "length_strata.csv", index=False)
    agreement.to_csv(results_dir / "length_agreement.csv", index=False)
    save_json(results_dir / "length_study.json", {
        "dataset_length_structure": dataset_side,
        "per_stratum": strata.to_dict(orient="records"),
        "pair_agreement": agreement.to_dict(orient="records"),
        "generated_behaviour": behaviour_summary(results_dir),
    })
    save_json(results_dir / "qualitative_candidates.json", qualitative_candidates(results_dir))
    print(strata.to_string(index=False))
    print(agreement.to_string(index=False))
    print("Wrote", results_dir / "length_study.json")


if __name__ == "__main__":
    main()
