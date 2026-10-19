"""Orchestration helper used by the ablation scripts.

Every fork and evaluation runs in its own Python process, so GPU memory is fully released between
conditions and an interrupted sweep can simply be re-launched: finished outputs are skipped.
"""

from __future__ import annotations

import subprocess
import sys
import hashlib
import json
from pathlib import Path

from common.data import REPO_ROOT, repo_path


def file_digest(path):
    path = repo_path(path)
    if path.name == "adapter_config.json":
        path = path.parent  # Validate weights/tokenizer as well as the adapter config.
    digest = hashlib.sha256()
    paths = sorted(p for p in path.rglob("*") if p.is_file() and ".pa2_complete.json" not in p.name
                   and not p.name.endswith(".tmp")) if path.is_dir() else [path]
    for p in paths:
        digest.update(str(p.relative_to(path) if path.is_dir() else p.name).encode())
        with p.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def run_fingerprint(request):
    """Bind completion receipts to code, settings, fixed assets and model provenance."""
    files = []
    for folder in ["common", "scripts", "task1_dpo", "task2_ppo", "task3_grpo", "task4_safety", "task5_feedback", "configs", "data", "cached", "checkpoints", "manifests"]:
        files += [p for p in (REPO_ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts
                  and p.suffix in {".py", ".yaml", ".json", ".jsonl", ".csv", ".pt", ".safetensors"}]
    for name in ["logs/prepare_models_state.json", "logs/environment.freeze.txt"]:
        if repo_path(name).exists():
            files.append(repo_path(name))
    payload = {"request": request, "files": {str(p.relative_to(REPO_ROOT)): file_digest(p) for p in sorted(files)}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def receipt_path(path):
    return repo_path(str(path) + ".pa2_complete.json")


def artifact_is_current(path, fingerprint):
    path = repo_path(path)
    receipt = receipt_path(path)
    if not path.exists() or not receipt.exists():
        return False
    try:
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        return saved["fingerprint"] == fingerprint and saved["artifact_sha256"] == file_digest(path)
    except (ValueError, KeyError, OSError):
        return False


def mark_artifact_complete(path, fingerprint):
    receipt = receipt_path(path)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    payload = {"fingerprint": fingerprint, "artifact_sha256": file_digest(path)}
    tmp = receipt.with_suffix(receipt.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(receipt)


def run_module(module: str, *args, skip_if_exists: str | None = None) -> None:
    """Run `python -m <module> <args>` from the repository root unless its output already exists."""
    request = {"module": module, "args": list(map(str, args))}
    # Evaluations must also track the adapter they read, not just its pathname.
    for flag in ["--adapter"]:
        if flag in request["args"]:
            value = request["args"][request["args"].index(flag) + 1]
            if repo_path(value).is_dir():
                request[flag + "_sha256"] = file_digest(value)
    fingerprint = run_fingerprint(request) if skip_if_exists else None
    if skip_if_exists and artifact_is_current(skip_if_exists, fingerprint):
        print(f"[skip] {module}: verified complete output at {skip_if_exists}", flush=True)
        return
    cmd = [sys.executable, "-m", module, *map(str, args)]
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    if skip_if_exists:
        # Written only after the subprocess exits normally (all summaries saved).
        mark_artifact_complete(skip_if_exists, fingerprint)
