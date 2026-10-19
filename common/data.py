from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(path: str | Path) -> Path:
    """Resolve a path relative to the repository root (absolute paths pass through)."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _deep_merge(a: dict, b: dict) -> dict:
    """Recursive dict merge; values in `b` win."""
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str | Path) -> dict:
    """Load a config, recursively merged over its `base_config`."""
    path = repo_path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg.get("base_config"):
        # Resolved recursively so overlays (configs/smoke/*.yaml) can extend a task config.
        base = load_yaml(cfg["base_config"])
        cfg = _deep_merge(base, {k: v for k, v in cfg.items() if k != "base_config"})
    return cfg


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSON-lines file into a list of dicts."""
    rows = []
    with repo_path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    """Write dicts as JSON lines (parent folders are created)."""
    p = repo_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV under the repository root."""
    return pd.read_csv(repo_path(path))


def last_assistant_text(messages) -> str:
    """Content of the last assistant turn of a chat (or the string itself)."""
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list):
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "assistant":
                return str(m.get("content", ""))
    return str(messages)


def prompt_messages_from_preference(row: dict) -> list[dict]:
    """Prompt turns of a preference row: the chosen chat without its final assistant reply."""
    chosen = row.get("chosen")
    if isinstance(chosen, list) and chosen:
        out = list(chosen)
        if isinstance(out[-1], dict) and out[-1].get("role") == "assistant":
            out = out[:-1]
        return out
    prompt = str(row.get("prompt", ""))
    return [{"role": "user", "content": prompt}]


def preference_responses(row: dict) -> tuple[str, str]:
    """(chosen, rejected) assistant texts of a preference row."""
    return last_assistant_text(row["chosen"]), last_assistant_text(row["rejected"])


def prompt_messages(row: dict) -> list[dict]:
    """Chat messages of an RL / evaluation row (messages, prompt list, or plain prompt/question)."""
    if isinstance(row.get("messages"), list):
        return row["messages"]
    if isinstance(row.get("prompt"), list):
        return row["prompt"]
    return [{"role": "user", "content": str(row.get("prompt", row.get("question", "")))}]


def prompt_schedule(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Fixed RL prompt schedule: the first n rows of a seeded permutation of the pool.

    Every PPO/GRPO run uses it, so a short fork sees exactly the first prompts of the standard run.
    """
    order = np.random.default_rng(seed).permutation(len(rows))
    if n > len(order):
        raise ValueError(f"Requested {n} prompts but the pool has only {len(order)}")
    return [rows[i] for i in order[:n]]


def render_prompt(tokenizer, messages: list[dict]) -> str:
    """Chat-template text of a prompt, ending with the assistant header."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def fit_prompt_messages(tokenizer, messages: list[dict], max_tokens: int) -> list[dict]:
    """Keep the latest chat context within a token cap, preserving the assistant header.

    Trim oldest content first, leaving the last turn and its role intact. Both the
    policy and reward model receive this same effective conversation. No rows are
    removed from the fixed course datasets. Cached released rollouts retain their
    original encoding and do not call this helper.
    """
    out = [dict(m) for m in messages]
    if not out:
        raise ValueError("A prompt must contain at least one message")
    def size():
        return len(tokenizer.apply_chat_template(out, tokenize=True, add_generation_prompt=True))
    while size() > max_tokens:
        # Removing an old turn also removes its template overhead. Preserve a
        # leading system message while there are other old turns to remove.
        if len(out) > 2:
            del out[1 if out[0].get("role") == "system" else 0]
            continue
        if len(out) == 2:
            del out[0]
            continue
        content = str(out[0].get("content", ""))
        ids = tokenizer.encode(content, add_special_tokens=False)
        if not ids:
            raise ValueError(f"Chat template alone exceeds max_prompt_length={max_tokens}")
        # Re-render after each trim: decode/encode boundaries can change token counts.
        excess = size() - max_tokens
        out[0]["content"] = tokenizer.decode(ids[min(len(ids), max(1, excess)):], skip_special_tokens=False)
    return out


def encode_prompt_response(
    tokenizer,
    messages: list[dict],
    response: str,
    max_length: int,
    max_prompt_length: int | None = None,
):
    """
    Encode prompt + response for DPO.

    Both responses share the same effective prompt, shortened from the oldest
    context if needed. Reserve response space and truncate the response right.
    """

    messages = fit_prompt_messages(tokenizer, messages, min(max_prompt_length or max_length - 1, max_length - 1))
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )

    response_ids = tokenizer(
        response,
        add_special_tokens=False,
    )["input_ids"]

    response_budget = max_length - len(prompt_ids)

    # Keep an EOS token when the tokenizer defines one.
    eos_id = tokenizer.eos_token_id

    if eos_id is not None:
        # Reserve one token for EOS.
        content_budget = max(0, response_budget - 1)
        response_ids = response_ids[:content_budget] + [eos_id]
    else:
        response_ids = response_ids[:response_budget]

    ids = prompt_ids + response_ids

    response_mask = (
        [0] * len(prompt_ids)
        + [1] * len(response_ids)
    )

    assert len(ids) <= max_length
    assert len(ids) == len(response_mask)

    return ids, response_mask


def pad_batch(tokenizer, examples: list[tuple[list[int], list[int]]]):
    """Left-pad (ids, response_mask) pairs into input_ids / attention_mask / response_mask tensors."""
    import torch

    max_len = max(len(ids) for ids, _ in examples)
    pad_id = tokenizer.pad_token_id
    input_ids, attention_mask, response_mask = [], [], []

    for ids, rmask in examples:
        pad = max_len - len(ids)
        input_ids.append([pad_id] * pad + ids)
        attention_mask.append([0] * pad + [1] * len(ids))
        response_mask.append([0] * pad + rmask)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "response_mask": torch.tensor(response_mask, dtype=torch.float32),
    }
