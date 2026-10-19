"""Validate experiment cardinalities/identities and report actual fork budgets.

Does not write the course report or invent missing human audit labels.
"""
from __future__ import annotations

import argparse
import math

from common.data import load_yaml, prompt_schedule, read_jsonl, repo_path
from common.logging_utils import load_json, save_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_ids(path, expected, key):
    rows = read_jsonl(path)
    actual = [r[key] for r in rows]
    require(len(actual) == len(expected) and set(actual) == set(expected), f"Wrong rows in {path}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    root = "configs/smoke" if args.smoke else "configs"
    dpo, ppo, grpo, feedback = [load_yaml(f"{root}/{name}.yaml") for name in ["dpo", "ppo", "grpo", "feedback"]]
    result = {"scope": "smoke" if args.smoke else "full", "checks": [], "fork_budgets": {}}

    dpo_names = ["standard", "length_balanced", *[f"beta_{b:g}" for b in dpo["betas"]]]
    for name in dpo_names:
        limit = dpo["short_ablation_examples"] if name.startswith("beta_") else dpo.get("max_train_examples")
        dataset = dpo["paths"]["dpo_length_train" if name == "length_balanced" else "dpo_standard_train"]
        expected = [r["prompt_id"] for r in read_jsonl(dataset)[:limit]]
        train = repo_path(dpo["results_dir"]) / "train" / name
        order = load_json(train / "train_order.json")
        require(len(order) == len(expected) * int(dpo["epochs"]) and set(order) == set(expected), f"DPO subset mismatch: {name}")
        cfg = load_json(train / "run_config.json")
        require(cfg["examples"] == len(expected) and not cfg["dropped_overlong_prompt_ids"], f"Dropped DPO rows: {name}")
    for name in ["sft", *dpo_names]:
        out = repo_path(dpo["results_dir"]) / "eval" / name
        for filename, dataset in [("pairs_standard.jsonl", "dpo_standard_eval"), ("pairs_stratified.jsonl", "dpo_length_eval")]:
            expected = [r["prompt_id"] for r in read_jsonl(dpo["paths"][dataset])[:dpo.get("eval_pair_limit")]]
            check_ids(out / filename, expected, "prompt_id")
        expected = [r["prompt_id"] for r in read_jsonl(dpo["paths"]["dpo_standard_eval"])[:int(dpo["eval_generation_prompts"])]]
        check_ids(out / "generations.jsonl", expected, "prompt_id")
    result["checks"].append("DPO full fixed train/evaluation subsets retained")

    for label, cfg in [("ppo", ppo), ("grpo", grpo)]:
        folder = repo_path(cfg["results_dir"]) / "train"
        names = ["standard"]
        names += ([f"fork_eps{e:g}_kl{cfg['kl_beta']:g}" for e in cfg["clip_values"]] +
                  [f"fork_eps{cfg['clip_epsilon']:g}_kl{b:g}" for b in cfg["kl_values"]]) if label == "ppo" else ["norm_grpo", "norm_dr_grpo"]
        fork_rows = []
        for name in sorted(set(names)):
            train = load_json(folder / name / "summary.json")
            log = read_jsonl(folder / name / "update_log.jsonl")
            n = int(cfg["updates"] if name == "standard" else cfg["fork_updates"])
            expected = [r["prompt_id"] for r in prompt_schedule(read_jsonl(cfg["paths"]["rl_prompt_train"]), n * int(cfg["prompts_per_update"]), int(cfg["seed"]))]
            require(train["updates"] == len(log) == n and train["prompt_ids"] == expected, f"Wrong continuation schedule: {label}/{name}")
            for row in log:
                for key in ["reward", "kl", "entropy", "response_tokens"]:
                    require(math.isfinite(row[key]), f"Nonfinite {label}/{name}/{key}")
            if name != "standard":
                maximum = n * int(cfg["prompts_per_update"]) * (int(cfg["max_response_length"]) if label == "ppo" else int(cfg["num_generations"]) * int(cfg["max_completion_length"]))
                fork_rows.append({"name": name, "updates": n, "generated_tokens": train["generated_tokens"],
                                  "maximum_generated_token_budget": maximum,
                                  "allowance_source": "training summary" if "maximum_generated_token_budget" in train else "derived from the recorded configuration"})
        expected_eval = [r["prompt_id"] for r in read_jsonl(cfg["paths"]["rl_prompt_eval"])[:int(cfg["eval_prompts"])]]
        for name in ["midpoint", *sorted(set(names))]:
            check_ids(repo_path(cfg["results_dir"]) / "eval" / name / "generations.jsonl", expected_eval, "prompt_id")
        result["fork_budgets"][label] = {"runs": fork_rows,
            "actual_token_counts_equal": len({r["generated_tokens"] for r in fork_rows}) == 1,
            "maximum_token_allowances_equal": len({r["maximum_generated_token_budget"] for r in fork_rows}) == 1,
            "manual_caveat": "Identical updates/caps do not establish exact equality of actual emitted tokens."}
    cached = load_json(repo_path(ppo["results_dir"]) / "clipping_cached.json")
    require(cached["n_used"] == cached["n_cached_rollouts"] == 32 and not cached["skipped_misaligned"], "Released PPO cache lost/misaligned rows")
    result["checks"].append("PPO/GRPO continuation schedules, held-out IDs and all 32 cached PPO rollouts")

    from task4_safety.generate_responses import load_xstest
    expected = list(map(int, load_xstest(feedback)["xstest_id"]))
    safety = repo_path(feedback["results_dir"]) / "task4_safety"
    for name in ["sft", "dpo", "ppo", "grpo"]:
        generated = check_ids(safety / f"generated_{name}.jsonl", expected, "xstest_id")
        judged = check_ids(safety / f"judged_{name}.jsonl", expected, "xstest_id")
        by_id = {r["xstest_id"]: r for r in judged}
        require(all(a["response"] == by_id[a["xstest_id"]]["response"] for a in generated), "Stale safety verdicts")
    result["checks"].append("All four safety policies generated and judged on fixed XSTest IDs")
    for dataset, key in [("gsm", "gsm_eval"), ("transfer", "math_transfer_eval")]:
        expected = [r["source_index"] for r in read_jsonl(feedback["paths"][key])[:feedback.get("math_eval_limit")]]
        out = repo_path(feedback["results_dir"]) / "task5_feedback" / dataset
        for policy in ["sft", "rlvr", "rlaif"]:
            check_ids(out / f"generated_{policy}.jsonl", expected, "source_index")
        pairs = read_jsonl(out / "pairwise.jsonl")
        require(len(pairs) == 2 * len(expected), f"Missing math comparisons: {dataset}")
    diag = load_json(repo_path(feedback["results_dir"]) / "task5_feedback" / "diagnostics" / "summary.json")
    require(diag["n_problems"] == (feedback.get("diagnostic_problems") or 20) and diag["verifier_vs_expected_mismatches"] == 0, "Diagnostic set incomplete or verifier mismatch")
    result["checks"].append("Fixed math/transfer IDs and validated controlled diagnostics")
    result["human_work_pending"] = ["Blind safety labels and agreement analysis", "Qualitative inspection", "Human-authored coursework report"]
    result["status"] = "passed_with_token_budget_caveat"
    save_json("logs/outputs_verified_smoke.json" if args.smoke else "logs/outputs_verified_full.json", result)
    print(result)


if __name__ == "__main__":
    main()
