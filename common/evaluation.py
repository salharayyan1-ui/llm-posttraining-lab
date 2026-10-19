"""Shared held-out generation protocol for Tasks 1-3.

Every compared condition is evaluated with the same prompts, decoding settings, generation cap and
seed. For each prompt one response is sampled and we record:

* reward  - scalar score from the fixed course reward model;
* KL      - sampled-response estimate sum_t [log pi(y_t|.) - log pi_ref(y_t|.)], where the reference
            is the same network with its LoRA adapter disabled (the frozen base policy);
* entropy - exact token-level policy entropy over the generated tokens;
* length  - generated response tokens (EOS included, padding excluded).

Aggregation follows the course helper `sampled_kl`: KL and entropy are token means over all valid
response tokens. The per-sequence KL mean is stored as well but is never mixed across conditions.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, score_reward_pairs, scored_response_logprobs
from common.logging_utils import save_json, set_seed
from common.metrics import length_stats
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode


def adapter_or_none(adapter: str | None) -> str | None:
    """CLI convenience: `--adapter none` evaluates the untouched base (SFT) policy."""
    return None if adapter is None or adapter.lower() in {"none", "sft", "base"} else adapter


def evaluate_generations(
    policy,
    tokenizer,
    reward_model,
    reward_tokenizer,
    items: list[dict],
    *,
    max_prompt_length: int,
    max_new_tokens: int,
    generation: dict,
    seed: int,
    batch_size: int = 8,
    reward_max_length: int = 1024,
) -> list[dict]:
    """Sample one response per item and score it. Items are dicts with `prompt_id` and `messages`."""
    set_seed(seed)
    records = []
    for start in tqdm(range(0, len(items), batch_size), desc="generate+score"):
        batch = items[start:start + batch_size]
        prompts = [item["messages"] for item in batch]
        gen = batch_generate(policy, tokenizer, prompts, max_prompt_length, max_new_tokens, **generation)
        mask = gen["response_mask"]

        logp, entropy = scored_response_logprobs(policy, gen)
        with reference_mode(policy):
            ref_logp, _ = scored_response_logprobs(policy, gen, with_entropy=False)
        rewards = score_reward_pairs(reward_model, reward_tokenizer, gen["prompt_messages"], gen["responses"], max_length=reward_max_length)

        kl_sum = ((logp - ref_logp) * mask).sum(-1)
        entropy_sum = (entropy * mask).sum(-1)
        for i, item in enumerate(batch):
            records.append({
                "prompt_id": item["prompt_id"],
                "prompt": prompts[i][-1]["content"],
                "effective_prompt_messages": gen["prompt_messages"][i],
                "prompt_truncated": gen["prompt_truncated"][i],
                "effective_generation_config": gen["effective_generation_config"],
                "response": gen["responses"][i],
                "response_tokens": int(gen["response_lengths"][i]),
                "terminated_with_eos": bool(gen["terminated_with_eos"][i]),
                "truncated": bool(gen["truncated"][i]),
                "reward": float(rewards[i]),
                "kl_sum": float(kl_sum[i]),
                "entropy_sum": float(entropy_sum[i]),
            })
    return records


def summarize_generations(records: list[dict]) -> dict:
    """Aggregate per-prompt records: reward mean/std, token- and sequence-mean KL, entropy, length stats, EOS/truncation rates."""
    tokens = max(sum(r["response_tokens"] for r in records), 1)
    rewards = np.array([r["reward"] for r in records], dtype=float)
    return {
        "n_prompts": len(records),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "kl_token_mean": sum(r["kl_sum"] for r in records) / tokens,
        "kl_sequence_mean": float(np.mean([r["kl_sum"] for r in records])),
        "entropy_token_mean": sum(r["entropy_sum"] for r in records) / tokens,
        "response_length": length_stats(r["response_tokens"] for r in records),
        "eos_rate": float(np.mean([r["terminated_with_eos"] for r in records])),
        "truncation_rate": float(np.mean([r["truncated"] for r in records])),
    }


def evaluate_rl_policy(config_path: str, adapter: str | None, name: str, max_new_tokens: int) -> dict:
    """Held-out evaluation shared by PPO and GRPO: first `eval_prompts` rows of the fixed eval pool."""
    cfg = load_yaml(config_path)
    rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])[: int(cfg["eval_prompts"])]
    items = [{"prompt_id": r["prompt_id"], "messages": prompt_messages(r)} for r in rows]

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter_or_none(adapter), trainable=False)
    reward_model, reward_tokenizer = load_reward_model(cfg)
    records = evaluate_generations(
        policy, tokenizer, reward_model, reward_tokenizer, items,
        max_prompt_length=int(cfg["max_prompt_length"]),
        max_new_tokens=int(max_new_tokens),
        generation=cfg["generation"],
        seed=int(cfg["seed"]),
        batch_size=int(cfg.get("eval_batch_size", 8)),
        reward_max_length=int(cfg.get("reward_max_length", 1024)),
    )

    out_dir = repo_path(cfg["results_dir"]) / "eval" / name
    write_jsonl(out_dir / "generations.jsonl", records)
    metrics = {
        "name": name,
        "adapter": adapter,
        "max_new_tokens": int(max_new_tokens),
        "generation": cfg["generation"],
        "seed": int(cfg["seed"]),
        "effective_generation_config": records[0]["effective_generation_config"] if records else {},
        **summarize_generations(records),
    }
    save_json(out_dir / "metrics.json", metrics)
    print(f"[{name}] reward={metrics['reward_mean']:.3f} kl={metrics['kl_token_mean']:.4f} "
          f"len={metrics['response_length']['mean']:.1f} -> {out_dir}")
    return metrics
