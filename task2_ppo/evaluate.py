"""Held-out PPO evaluation: common protocol from common/evaluation.py at the frozen-evaluation cap.

Writes results/task2_ppo/eval/<name>/{metrics.json, generations.jsonl}. Evaluate the supplied
midpoint as well (`--adapter checkpoints/ppo_midpoint_policy --name midpoint`) as the starting point.
"""

from __future__ import annotations

import argparse

from common.data import load_yaml
from common.evaluation import evaluate_rl_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter directory, or 'none' for the SFT base policy")
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    evaluate_rl_policy(args.config, args.adapter, args.name, int(cfg["eval_max_response_length"]))


if __name__ == "__main__":
    main()
