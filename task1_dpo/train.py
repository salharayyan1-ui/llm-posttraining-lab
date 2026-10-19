from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, peak_vram, reset_peak_vram, save_json, set_seed, wall_timer
from common.models import load_policy, load_tokenizer, reference_mode, trainable_parameters
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length, max_prompt_length=None):
    """Batch builder: chosen and rejected are encoded with the same prompt and padded separately."""
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length, max_prompt_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length, max_prompt_length))
        return {
            "prompt_ids": [row["prompt_id"] for row in rows],
            "chosen": pad_batch(tokenizer, chosen),
            "rejected": pad_batch(tokenizer, rejected),
        }
    return collate


def drop_overlong_prompts(rows: list[dict], tokenizer, max_length: int):
    """`encode_prompt_response` refuses prompts that alone fill `max_length`; drop and report them up front."""
    kept, dropped = [], []
    for row in rows:
        n_prompt = len(tokenizer.apply_chat_template(
            prompt_messages_from_preference(row), tokenize=True, add_generation_prompt=True
        ))
        (kept if n_prompt < max_length else dropped).append(row)
    return kept, [row["prompt_id"] for row in dropped]


def pair_logps(model, batch: dict):
    """Response-only sequence log-probs of chosen/rejected under the policy and the reference.

    The reference is the same network with its LoRA adapter disabled, i.e. the frozen
    Qwen2.5-1.5B-Instruct initialization, so no second model copy is needed.
    Returns (policy_chosen, policy_rejected, ref_chosen, ref_rejected); reference values carry no grad.
    """
    device = next(model.parameters()).device
    chosen = {k: v.to(device) for k, v in batch["chosen"].items()}
    rejected = {k: v.to(device) for k, v in batch["rejected"].items()}
    policy_chosen = response_sequence_logprobs(model, chosen)[0]
    policy_rejected = response_sequence_logprobs(model, rejected)[0]
    with torch.no_grad(), reference_mode(model):
        ref_chosen = response_sequence_logprobs(model, chosen)[0]
        ref_rejected = response_sequence_logprobs(model, rejected)[0]
    return policy_chosen, policy_rejected, ref_chosen, ref_rejected


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    """Data (optionally the first `max_examples` pairs), fresh-LoRA policy, shuffled loader and AdamW; seeded so every DPO condition shares one start."""
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    max_examples = max_examples if max_examples is not None else cfg.get("max_train_examples")  # smoke configs only
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])
    dropped = []  # Preserve the entire fixed subset, including all length strata.
    # Fresh LoRA on the base policy: every DPO condition starts from the identical initialization
    # (same seed -> same LoRA init and same shuffle order).
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"]), int(cfg["max_prompt_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "dataset_path": path,
        "dropped_prompt_ids": dropped,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    """One pass (`epochs`) of DPO with gradient accumulation; saves the adapter and a per-step log.

    Effective batch = batch_size x grad_accum_steps pairs (2 x 8 = 16 in the release config).
    """
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg, model, loader, optimizer = bundle["cfg"], bundle["model"], bundle["loader"], bundle["optimizer"]
    beta = bundle["beta"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)

    log_dir = repo_path(cfg["results_dir"]) / "train" / run_name
    log_path = log_dir / "train_log.jsonl"
    log_path.unlink(missing_ok=True)  # a re-run starts a fresh log

    accum = int(cfg["grad_accum_steps"])
    epochs = int(cfg["epochs"])
    total_micro = len(loader) * epochs
    params = trainable_parameters(model)
    elapsed = wall_timer()
    reset_peak_vram()

    # step = optimizer steps taken, micro = micro-batches seen, window = diagnostics of the current step.
    step, micro, seen_ids, window = 0, 0, [], []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        for batch in loader:
            # Accumulate pair sums; divide gradients by the actual pair count
            # before clipping, including unequal final micro-batches.
            group = min(accum, total_micro - step * accum)
            loss, stats = dpo_loss(*pair_logps(model, batch), beta)
            n_pairs = len(batch["prompt_ids"])
            (loss * n_pairs).backward()
            window.append({"loss": loss.detach(), **stats, "n_pairs": n_pairs})
            seen_ids.extend(batch["prompt_ids"])
            micro += 1
            if micro - step * accum < group:
                continue  # keep accumulating

            normalize_accumulated_gradients(params, sum(w["n_pairs"] for w in window))
            grad_norm = clip_grad_norm_(params, float(cfg["max_grad_norm"]))  # returns the pre-clip norm
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            # One log line per optimizer step: diagnostics averaged over its micro-batches.
            record = {k: float(np.average([float(w[k]) for w in window], weights=[w["n_pairs"] for w in window]))
                      for k in window[0] if k != "n_pairs"}
            record.update(step=step, epoch=epoch, examples_seen=len(seen_ids), grad_norm=float(grad_norm), elapsed_s=elapsed())
            append_jsonl(log_path, record)
            window = []
            if step == 1 or step % 10 == 0:
                print(f"[{run_name}] step {step}: loss={record['loss']:.4f} acc={record['preference_accuracy']:.3f} "
                      f"margin={record['margin_mean']:.3f} grad={record['grad_norm']:.3f}", flush=True)

    # Only the LoRA adapter is saved (a few MB); load_policy re-attaches it to the base model.
    model.save_pretrained(str(output))
    bundle["tokenizer"].save_pretrained(str(output))
    save_json(log_dir / "train_order.json", seen_ids)  # exact data order, for reproducibility
    save_json(log_dir / "run_config.json", {
        "run_name": run_name,
        "output": str(output),
        "dataset": bundle["dataset_path"],
        "beta": beta,
        "max_examples": max_examples,
        "examples": len(bundle["rows"]),
        "dropped_overlong_prompt_ids": bundle["dropped_prompt_ids"],
        "optimizer_steps": step,
        "wall_clock_s": elapsed(),
        **peak_vram(),
        "config": cfg,
    })
    print(f"[{run_name}] saved adapter to {output} after {step} optimizer steps")


def normalize_accumulated_gradients(params, n_pairs):
    """Convert accumulated pair-sum gradients to a mean over actual pairs."""
    if n_pairs <= 0:
        raise ValueError("An accumulation window must contain pairs")
    for param in params:
        if param.grad is not None:
            param.grad.div_(n_pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
