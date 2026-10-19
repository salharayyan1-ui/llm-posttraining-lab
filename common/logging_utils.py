from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common.data import repo_path


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU + all GPUs)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent(path: str | Path) -> Path:
    """Resolve a path and create its parent folder."""
    p = repo_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one record to a JSON-lines log."""
    p = ensure_parent(path)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(path: str | Path, obj: Any) -> None:
    """Write an object as indented JSON."""
    p = ensure_parent(path)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    """Read a JSON file under the repository root."""
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def wall_timer():
    """Return a function giving seconds elapsed since this call."""
    start = time.perf_counter()
    return lambda: time.perf_counter() - start


def reset_peak_vram() -> None:
    """Start a new peak-memory measurement window."""
    if torch.cuda.is_available():
        for device in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device)


def peak_vram() -> dict[str, Any]:
    """Track each GPU; existing scalar fields report the highest per-card peak."""
    if not torch.cuda.is_available():
        return {"peak_allocated_gib": 0.0, "peak_reserved_gib": 0.0}
    by_device = {
        f"cuda:{i}": {
            "peak_allocated_gib": torch.cuda.max_memory_allocated(i) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(i) / 2**30,
        }
        for i in range(torch.cuda.device_count())
    }
    return {
        "peak_allocated_gib": max(v["peak_allocated_gib"] for v in by_device.values()),
        "peak_reserved_gib": max(v["peak_reserved_gib"] for v in by_device.values()),
        "peak_vram_by_device": by_device,
    }


def scalars(stats: dict[str, Any]) -> dict[str, float]:
    """Convert a dict of 0-d tensors / numbers into plain floats for JSON logs."""
    return {k: float(v.item() if torch.is_tensor(v) else v) for k, v in stats.items()}
