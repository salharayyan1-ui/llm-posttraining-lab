"""Task 4, step 3: fixed manual-audit subset and a blind labelling sheet.

The 60 audit prompts (30 SAFE + 30 UNSAFE, seeded) are drawn exactly as released. Their responses
from every requested policy go into one shuffled sheet without the policy name or the AI label, so
manual labels are assigned blind; the mapping back is kept in a separate key file.

  results/task4_safety/manual_audit_ids.csv    the fixed audit prompt IDs
  results/task4_safety/manual_audit_sheet.csv  fill `manual_label` with one of the five judge labels
  results/task4_safety/manual_audit_key.csv    audit_key -> (policy, xstest_id); do not open while labelling

An existing sheet is never overwritten (it may already hold manual labels) unless --force.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    """Seeded draw of `per_class` SAFE and `per_class` UNSAFE prompt ids (released rule)."""
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def blind_sheet(outdir: Path, ids: list[int], policies: list[str], seed: int):
    """Shuffled (policy, audit prompt) responses without policy name or AI label, plus the key that undoes it."""
    rows = []
    for policy in policies:
        generated = {r["xstest_id"]: r for r in read_jsonl(outdir / f"generated_{policy}.jsonl")}
        for xid in ids:
            g = generated[xid]
            rows.append({"policy": policy, "xstest_id": xid, "prompt": g["prompt"], "response": g["response"],
                         "benchmark_class": g["benchmark_class"], "type": g["type"]})
    table = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    table.insert(0, "audit_key", [f"A{i:03d}" for i in range(1, len(table) + 1)])
    sheet = table[["audit_key", "prompt", "response", "benchmark_class", "type"]].assign(manual_label="", notes="")
    key = table[["audit_key", "policy", "xstest_id"]]
    return sheet, key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policies", nargs="+", default=["sft", "dpo", "ppo", "grpo"],
                    help="policies whose responses to the audit prompts are labelled")
    ap.add_argument("--force", action="store_true", help="overwrite an existing (possibly labelled) sheet")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))
    ids = fixed_audit_ids(read_jsonl(src), int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    sheet_path, key_path = outdir / "manual_audit_sheet.csv", outdir / "manual_audit_key.csv"
    sheet, key = blind_sheet(outdir, ids, args.policies, int(cfg["seed"]))
    if sheet_path.exists() and not args.force:
        existing = pd.read_csv(sheet_path, keep_default_na=False)
        same = existing.drop(columns=["manual_label", "notes"]).equals(sheet.drop(columns=["manual_label", "notes"]))
        same = same and key_path.exists() and pd.read_csv(key_path).equals(key)
        if not same:
            raise ValueError("Audit sheet/key belongs to different responses. Preserve any manual labels, then use --force to create a new blind audit.")
        print(f"[keep] {sheet_path} already exists (may contain manual labels); use --force to rebuild")
        return
    pd.DataFrame({"xstest_id": ids}).to_csv(outdir / "manual_audit_ids.csv", index=False)
    sheet.to_csv(sheet_path, index=False)
    key.to_csv(key_path, index=False)
    print(f"Wrote blind sheet with {len(sheet)} rows: {sheet_path}")
    print("Label `manual_label` with SAFE_ANSWER / JUSTIFIED_REFUSAL / UNSAFE_COMPLIANCE / OVER_REFUSAL / AMBIGUOUS "
          "before looking at any judged_*.jsonl file.")


if __name__ == "__main__":
    main()
