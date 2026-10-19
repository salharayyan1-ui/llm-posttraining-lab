from __future__ import annotations

import re
import numpy as np
import torch
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of x over positions where mask = 1 (0 if the mask is empty)."""
    mask = mask.to(dtype=x.dtype)
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def sampled_kl(policy_logp: torch.Tensor, ref_logp: torch.Tensor, mask: torch.Tensor):
    """Course KL estimator: token mean of log pi - log pi_ref on sampled tokens."""
    return masked_mean(policy_logp - ref_logp, mask)


def sample_entropy(sampled_logp: torch.Tensor, mask: torch.Tensor):
    """One-sample entropy estimate: token mean of -log pi(sampled token)."""
    return -masked_mean(sampled_logp, mask)


@torch.no_grad()
def token_entropies(logits: torch.Tensor, chunk_rows: int = 1) -> torch.Tensor:
    """Exact per-token entropy -sum_v p log p, shape [batch, steps].

    `sample_entropy` above is a one-sample estimate; this is the full-vocabulary value used for the
    manual's "mean token-level policy entropy". Rows are processed in chunks because a float32
    copy of [rows, steps, 151k] logits is large.
    """
    out = []
    for start in range(0, logits.shape[0], chunk_rows):
        logp = F.log_softmax(logits[start:start + chunk_rows].float(), dim=-1)
        out.append(-(logp.exp() * logp).sum(-1))
    return torch.cat(out)


def explained_variance(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """1 - Var[target - pred] / Var[target] over valid tokens (1 = perfect critic, <= 0 = useless)."""
    valid = mask.bool()
    y, r = target[valid].float(), (target - pred)[valid].float()
    if y.numel() < 2 or float(y.var()) == 0.0:
        return float("nan")
    return float(1.0 - r.var() / y.var())


def mean_response_length(mask: torch.Tensor):
    """Mean number of valid tokens per row."""
    return float(mask.sum(-1).float().mean().item())


def length_stats(lengths) -> dict:
    """Mean plus dispersion (std, median, IQR) as required by the manual's length convention."""
    x = np.asarray(list(lengths), dtype=float)
    if x.size == 0:
        return {"n": 0}
    q1, median, q3 = np.percentile(x, [25, 50, 75])
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
    }


def preference_accuracy(chosen_logp, rejected_logp):
    """Fraction of rows where the chosen score beats the rejected one."""
    return float((chosen_logp > rejected_logp).float().mean().item())


def word_count(text: str) -> int:
    """Number of words, i.e. alphanumeric runs between word boundaries."""
    return len(re.findall(r"\b\w+\b", text))


def parse_word_limit(prompt: str):
    """Word limit stated in a prompt (e.g. 'in at most 40 words'), or None."""
    patterns = [
        r"(?:at most|no more than|under|within)\s+(\d+)\s+words?",
        r"(?:in|use)\s+(\d+)\s+words?\s+(?:or fewer|max(?:imum)?)",
        r"(?:maximum|max)\s+(?:of\s+)?(\d+)\s+words?",
    ]
    lower = str(prompt).lower()
    for pattern in patterns:
        m = re.search(pattern, lower)
        if m:
            return int(m.group(1))
    return None


def word_limit_compliance(prompt: str, response: str):
    """1.0 / 0.0 if the response respects the prompt's word limit; None if the prompt has none."""
    limit = parse_word_limit(prompt)
    if limit is None:
        return None
    return float(word_count(response) <= limit)


def safe_corr(a, b):
    """Pearson correlation, NaN when undefined (fewer than 2 points or zero variance)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
