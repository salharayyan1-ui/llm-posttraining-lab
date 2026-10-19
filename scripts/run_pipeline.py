"""Run every required PA2 experiment in dependency order, resuming whatever already finished.

    python -m scripts.run_pipeline                # all tasks with the real configs
    python -m scripts.run_pipeline --tasks 2 3    # only some tasks
    python -m scripts.run_pipeline --smoke        # toy-size end-to-end check (configs/smoke -> results_smoke/)

Each step runs in its own process (common/runs.py). Training/evaluation steps are skipped when
their adapter or metrics file already exists; analysis steps are cheap and always re-run.
Task 4 needs the standard DPO/PPO/GRPO adapters, so run Tasks 1-3 first. The Task 4 manual audit
is a human step: label the sheet, then re-run `--tasks 4` to add the agreement analysis.
"""

from __future__ import annotations

import argparse

from common.data import load_yaml
from common.runs import run_module


def task1(c: dict) -> None:
    """Task 1: standard DPO + its eval and the SFT eval, then the beta study and the length study."""
    cfg = load_yaml(c["dpo"])
    run_module("task1_dpo.train", "--config", c["dpo"], "--run-name", "standard",
               skip_if_exists=f"{cfg['standard_output']}/adapter_config.json")
    for name, adapter in [("standard", cfg["standard_output"]), ("sft", "none")]:
        run_module("task1_dpo.evaluate", "--config", c["dpo"], "--adapter", adapter, "--name", name,
                   skip_if_exists=f"{cfg['results_dir']}/eval/{name}/metrics.json")
    run_module("task1_dpo.ablate_beta", "--config", c["dpo"])
    run_module("task1_dpo.analyze_length", "--config", c["dpo"])


def rl_task(c: dict, key: str, package: str, midpoint_key: str) -> None:
    """Shared Task 2/3 prologue: standard 20-update continuation + held-out eval of it and the midpoint."""
    cfg = load_yaml(c[key])
    run_module(f"{package}.continue_train", "--config", c[key], "--run-name", "standard",
               skip_if_exists=f"{cfg['output']}/adapter_config.json")
    for name, adapter in [("standard", cfg["output"]), ("midpoint", cfg["paths"][midpoint_key])]:
        run_module(f"{package}.evaluate", "--config", c[key], "--adapter", adapter, "--name", name,
                   skip_if_exists=f"{cfg['results_dir']}/eval/{name}/metrics.json")


def task2(c: dict) -> None:
    """Task 2: standard PPO + evals, cached clipping study, epsilon forks, beta_KL forks."""
    rl_task(c, "ppo", "task2_ppo", "ppo_midpoint_policy")
    cfg = load_yaml(c["ppo"])
    run_module("task2_ppo.analyze_clipping", "--config", c["ppo"], "--part", "cached",
               skip_if_exists=f"{cfg['results_dir']}/clipping_cached.csv")
    run_module("task2_ppo.analyze_clipping", "--config", c["ppo"], "--part", "forks")
    run_module("task2_ppo.ablate_kl", "--config", c["ppo"])


def task3(c: dict) -> None:
    """Task 3: standard GRPO + evals, group-size study (CPU only), normalization forks."""
    rl_task(c, "grpo", "task3_grpo", "grpo_midpoint_policy")
    run_module("task3_grpo.analyze_group_size", "--config", c["grpo"])
    run_module("task3_grpo.compare_normalization", "--config", c["grpo"])


def task4(c: dict) -> None:
    """Task 4: generate, judge, blind audit sheet, analysis (each step resumes or keeps existing outputs)."""
    for module in ["generate_responses", "judge_responses", "make_audit_sheet", "evaluate_safety"]:
        run_module(f"task4_safety.{module}", "--config", c["feedback"])


def task5(c: dict) -> None:
    """Task 5: in-domain evaluation, controlled diagnostics, transfer evaluation, final comparison."""
    run_module("task5_feedback.evaluate_math", "--config", c["feedback"], "--dataset", "gsm")
    run_module("task5_feedback.score_perturbations", "--config", c["feedback"])
    run_module("task5_feedback.evaluate_math", "--config", c["feedback"], "--dataset", "transfer")
    run_module("task5_feedback.compare_feedback", "--config", c["feedback"])


TASKS = {"1": task1, "2": task2, "3": task3, "4": task4, "5": task5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    ap.add_argument("--smoke", action="store_true", help="use configs/smoke/*.yaml (toy sizes, separate outputs)")
    ap.add_argument("--dual-gpu", action="store_true", help="policy on GPU 0, critic/reward/judge on GPU 1")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()
    root = "configs/smoke" if args.smoke else "configs"
    if args.dual_gpu:
        root = "configs/dual_gpu/smoke" if args.smoke else "configs/dual_gpu"
    configs = {name: f"{root}/{name}.yaml" for name in ["dpo", "ppo", "grpo", "feedback"]}

    if not args.skip_tests:
        run_module("tests.test_objectives")
    for task in args.tasks:
        print(f"\n===== Task {task} ({root}) =====", flush=True)
        TASKS[task](configs)
    if not args.smoke:
        run_module("scripts.plot_results")


if __name__ == "__main__":
    main()
