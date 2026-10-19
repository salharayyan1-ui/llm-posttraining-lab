"""Task 5, steps 1 and 3: SFT vs RLVR vs RLAIF on GSM8K (`--dataset gsm`) or SVAMP (`--dataset transfer`).

Stage `generate`: one greedy response per problem and policy with identical decoding, prompt budget
and generation cap, scored by the released exact verifier:
  correct      designated final (`#### <number>`) parsed and equal to gold   -> exact-answer accuracy
  format_ok    a designated final was parsed at all                         -> format compliance
  failure_type correct / wrong_final / truncated (no final, hit the cap) / no_final
Stage `judge`: the fixed pairwise AI judge compares each RL policy's response with SFT's on the same
problem (the judge never sees the gold answer); win = 1, tie = 0.5, loss = 0. The verifier's own
preference for the same pair (exact reward of A vs B) is stored next to it -> verifier-judge agreement.

Outputs: results/task5_feedback/<dataset>/{generated_<policy>.jsonl, pairwise.jsonl, metrics.json}
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json
from common.metrics import length_stats
from common.models import clear_gpu, load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final

RL_POLICIES = ["rlvr", "rlaif"]


def policy_specs(cfg):
    """SFT base plus the two supplied frozen RL adapters."""
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    """GSM8K evaluation subset or the fixed SVAMP transfer file."""
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    """Config, evaluation rows (optionally capped for smoke tests) and tokenizer."""
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))[: cfg.get("math_eval_limit")]  # None = all (smoke configs cap it)
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    """Frozen policy by name (adapter or base model)."""
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def failure_type(correct: float, pred, truncated: bool) -> str:
    """Classify one response for the failure-type breakdown."""
    if correct:
        return "correct"
    if pred is not None:
        return "wrong_final"
    return "truncated" if truncated else "no_final"


def generate(cfg: dict, name: str, rows: list[dict], tokenizer) -> list[dict]:
    """Greedy responses of one policy, scored with the exact verifier."""
    model = load_frozen_policy(cfg, name)
    batch_size = int(cfg.get("math_batch_size", 8))
    records = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        gen = batch_generate(
            model, tokenizer, [prompt_messages(r) for r in chunk],
            max_prompt_length=int(cfg["math_max_prompt_length"]),
            max_new_tokens=int(cfg["math_max_new_tokens"]),
            temperature=0.0, top_p=1.0, do_sample=False,
        )
        for row, response, n_tok, truncated in zip(chunk, gen["responses"], gen["response_lengths"], gen["truncated"]):
            pred = extract_designated_final(response)
            correct = exact_reward(response, row["gold_final"])
            records.append({
                "source_index": row["source_index"],
                "policy": name,
                "question": row["question"],
                "gold_final": str(row["gold_final"]),
                "response": response,
                "response_tokens": int(n_tok),
                "truncated": bool(truncated),
                "pred_final": pred,
                "format_ok": pred is not None,
                "correct": correct,
                "failure_type": failure_type(correct, pred, truncated),
            })
        print(f"[{name}] {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    del model
    clear_gpu()
    return records


def preference_from_scores(a: float, b: float) -> str:
    """Pairwise preference implied by two scalar scores: A, B or TIE."""
    return "A" if a > b else "B" if a < b else "TIE"


def judge_against_sft(cfg: dict, outdir) -> list[dict]:
    """Policy response = candidate A, SFT response = candidate B (the judge balances orientation itself)."""
    gens = {p: {r["source_index"]: r for r in read_jsonl(outdir / f"generated_{p}.jsonl")} for p in ["sft", *RL_POLICIES]}
    judge = PairwiseAIJudge(cfg, cfg["judge_cache"])
    records = []
    for policy in RL_POLICIES:
        for sid, a in gens[policy].items():
            b = gens["sft"][sid]
            cache_hit = judge._key(a["question"], a["response"], b["response"]) in judge.cache
            start = time.perf_counter()
            pref = judge.compare(a["question"], a["response"], b["response"])
            records.append({
                "policy": policy,
                "opponent": "sft",
                "source_index": sid,
                "judge": pref,
                "verifier": preference_from_scores(a["correct"], b["correct"]),
                "policy_correct": a["correct"],
                "sft_correct": b["correct"],
                "policy_tokens": a["response_tokens"],
                "sft_tokens": b["response_tokens"],
                "judge_seconds": time.perf_counter() - start,
                "judge_cache_hit": cache_hit,
            })
    return records


def summarize(outdir) -> dict:
    """Per-policy accuracy / format / length / failure types and the pairwise judge statistics vs SFT."""
    out = {"policies": {}, "pairwise_vs_sft": {}}
    for policy in ["sft", *RL_POLICIES]:
        df = pd.DataFrame(read_jsonl(outdir / f"generated_{policy}.jsonl"))
        out["policies"][policy] = {
            "n": len(df),
            "exact_accuracy": float(df["correct"].mean()),
            "format_compliance": float(df["format_ok"].mean()),
            "truncation_rate": float(df["truncated"].mean()),
            "response_tokens": length_stats(df["response_tokens"]),
            "failure_types": df["failure_type"].value_counts().to_dict(),
        }
    pairs_path = outdir / "pairwise.jsonl"
    if pairs_path.exists():
        pairs = pd.DataFrame(read_jsonl(pairs_path))
        for policy, d in pairs.groupby("policy"):
            decisive = d["verifier"] != "TIE"
            fresh = d.loc[~d["judge_cache_hit"], "judge_seconds"]
            out["pairwise_vs_sft"][policy] = {
                "n": len(d),
                "wins": int((d["judge"] == "A").sum()),
                "ties": int((d["judge"] == "TIE").sum()),  # unparsable judge outputs are folded into TIE by the released judge
                "losses": int((d["judge"] == "B").sum()),
                "win_rate": float(((d["judge"] == "A") + 0.5 * (d["judge"] == "TIE")).mean()),
                "verifier_judge_agreement": float((d["judge"] == d["verifier"]).mean()),
                "agreement_when_verifier_decisive": float((d.loc[decisive, "judge"] == d.loc[decisive, "verifier"]).mean()) if decisive.any() else float("nan"),
                "judge_decisive_when_verifier_ties": float((d.loc[~decisive, "judge"] != "TIE").mean()) if (~decisive).any() else float("nan"),
                "verifier_decisive_rate": float(decisive.mean()),
                "judge_seconds_per_comparison": float(fresh.mean()) if len(fresh) else float("nan"),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--stage", choices=["generate", "judge", "summary", "all"], default="all")
    ap.add_argument("--force", action="store_true", help="regenerate existing response files")
    args = ap.parse_args()
    cfg, rows, tokenizer = load_math_evaluation(args.config, args.dataset)
    from common.runs import artifact_is_current, file_digest, mark_artifact_complete, run_fingerprint
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback" / args.dataset
    print("Rows:", len(rows), "| policies:", list(policy_specs(cfg)))

    if args.stage in {"generate", "all"}:
        for name in policy_specs(cfg):
            path = outdir / f"generated_{name}.jsonl"
            adapter = policy_specs(cfg)[name]
            fingerprint = run_fingerprint({"task": "math_generate", "dataset": args.dataset, "config": cfg,
                                           "policy": name, "adapter": file_digest(adapter) if adapter else None})
            if artifact_is_current(path, fingerprint) and not args.force:
                print(f"[skip] verified {path}")
                continue
            write_jsonl(path, generate(cfg, name, rows, tokenizer))
            mark_artifact_complete(path, fingerprint)
    if args.stage in {"judge", "all"}:
        write_jsonl(outdir / "pairwise.jsonl", judge_against_sft(cfg, outdir))
    if args.stage in {"summary", "judge", "all"}:
        metrics = {"dataset": args.dataset, "decoding": "greedy", "max_new_tokens": int(cfg["math_max_new_tokens"]), **summarize(outdir)}
        save_json(outdir / "metrics.json", metrics)
        for policy, m in metrics["policies"].items():
            win = metrics["pairwise_vs_sft"].get(policy, {}).get("win_rate", float("nan"))
            print(f"[{args.dataset}/{policy}] acc={m['exact_accuracy']:.3f} format={m['format_compliance']:.3f} "
                  f"len={m['response_tokens']['mean']:.0f} win_vs_sft={win:.3f}")


if __name__ == "__main__":
    main()
