from __future__ import annotations

import argparse
import pandas as pd

from common.data import load_yaml, repo_path, write_jsonl
from common.generation import batch_generate
from common.models import clear_gpu, load_policy, load_tokenizer
from common.runs import artifact_is_current, file_digest, mark_artifact_complete, run_fingerprint


def policy_specs(cfg):
    """The four fixed Task 4 policies: base model and the three standard adapters."""
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    """Fixed XSTest CSV in its released order (optionally capped per class for smoke tests)."""
    df = pd.read_csv(repo_path(cfg["paths"]["xstest"]))
    per_class = cfg.get("xstest_limit_per_class")  # smoke configs only; real runs use all 450 prompts
    return df.groupby("benchmark_class", sort=False).head(int(per_class)) if per_class else df


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4):
    """Greedy responses of one policy to every XSTest prompt (released generation settings)."""
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    df = load_xstest(cfg)
    records = []
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=int(cfg["safety_max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for (_, row), response, n_tok in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"]):
            records.append({
                "xstest_id": int(row["xstest_id"]),
                "policy": policy_name,
                "prompt": str(row["prompt"]),
                "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]),
                "response": response,
                "response_tokens": int(n_tok),
            })
    return records


def main():
    """Greedy responses of the four frozen Task 4 policies, one file per policy, in XSTest order.

    Decoding (greedy, `safety_max_new_tokens`, prompt cap, batch size) is identical for every
    policy. Existing files are kept unless --force, so the step can be resumed policy by policy.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policies", nargs="+", choices=["sft", "dpo", "ppo", "grpo"], default=["sft", "dpo", "ppo", "grpo"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    specs = policy_specs(cfg)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    print("XSTest rows:", len(load_xstest(cfg)))

    for name in args.policies:
        path = outdir / f"generated_{name}.jsonl"
        fingerprint = run_fingerprint({"task": "safety_generate", "config": cfg, "policy": name,
                                       "batch_size": args.batch_size,
                                       "adapter": file_digest(specs[name]) if specs[name] else None})
        if artifact_is_current(path, fingerprint) and not args.force:
            print(f"[skip] verified {path}")
            continue
        if specs[name] and not (repo_path(specs[name]) / "adapter_config.json").exists():
            raise FileNotFoundError(f"{name}: standard-policy adapter missing at {specs[name]} (run its Task 1-3 step 1 first)")
        records = generate_for_policy(cfg, name, batch_size=args.batch_size)
        write_jsonl(path, records)
        mark_artifact_complete(path, fingerprint)
        clear_gpu()
        print(f"[{name}] wrote {len(records)} responses -> {path}")


if __name__ == "__main__":
    main()
