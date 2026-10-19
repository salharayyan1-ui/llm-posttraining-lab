from __future__ import annotations

import gc
from contextlib import contextmanager
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from common.data import repo_path


def model_device(cfg: dict, role: str):
    """Explicit per-model placement; omit `devices` for the original CPU/GPU-0 default."""
    configured = cfg.get("devices", {}).get(role)
    device = torch.device(configured or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"{role} requests {device}, but CUDA is unavailable")
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"{role} requests {device}, but only {torch.cuda.device_count()} CUDA GPUs are visible")
        device = torch.device("cuda", index)
    return device


def resolve_dtype(name: str):
    """Map a config dtype name to a torch dtype (bf16 falls back to fp16 without hardware support)."""
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def load_tokenizer(model_id: str, padding_side: str = "left"):
    """Tokenizer with left padding (needed for batched generation) and a pad token."""
    tok = AutoTokenizer.from_pretrained(model_id, padding_side=padding_side, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = padding_side
    return tok


def make_lora_config(cfg: dict) -> LoraConfig:
    """Policy LoRA config from the `lora` block of the config."""
    lc = cfg["lora"]
    return LoraConfig(
        r=int(lc["r"]),
        lora_alpha=int(lc["alpha"]),
        lora_dropout=float(lc["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lc["target_modules"]),
    )


def make_value_lora_config(cfg: dict) -> LoraConfig:
    """Critic LoRA config; the score head is trained in full (`modules_to_save`)."""
    lc = cfg.get("value_lora", cfg["lora"])
    return LoraConfig(
        r=int(lc["r"]),
        lora_alpha=int(lc["alpha"]),
        lora_dropout=float(lc["dropout"]),
        bias="none",
        task_type="SEQ_CLS",
        target_modules=list(lc["target_modules"]),
        modules_to_save=["score"],
    )


def load_policy(cfg: dict, adapter_path: str | None = None, trainable: bool = False, fresh_lora: bool = False):
    """Base policy, optionally with a saved adapter or a fresh LoRA; trainable -> train mode + gradient checkpointing."""
    dtype = resolve_dtype(cfg.get("dtype", "float16"))
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    tok_for_config = load_tokenizer(cfg["base_model"])
    model.config.pad_token_id = tok_for_config.pad_token_id

    if adapter_path:
        model = PeftModel.from_pretrained(
            model,
            str(repo_path(adapter_path)),
            is_trainable=trainable,
        )
    elif fresh_lora:
        model = get_peft_model(model, make_lora_config(cfg))

    model = model.to(model_device(cfg, "policy"))

    if trainable:
        upcast_trainable_to_fp32(model)
        model.train()
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        model.eval()
    return model


@contextmanager
def reference_mode(model):
    """Context manager: run the policy with its adapter disabled, i.e. as the frozen reference."""
    was_training = model.training
    model.eval()
    try:
        if isinstance(model, PeftModel):
            with model.disable_adapter():
                yield
        else:
            yield
    finally:
        if was_training:
            model.train()


def _quant_config(bits: int | None, dtype):
    """bitsandbytes 8-/4-bit config for frozen models (None on CPU)."""
    if bits is None or not torch.cuda.is_available():
        return None
    if bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
    raise ValueError(bits)


def load_reward_model(cfg: dict):
    """Frozen course reward model (8-bit when quantization is enabled) with the base-policy tokenizer."""
    dtype = resolve_dtype(cfg.get("dtype", "float16"))
    device = model_device(cfg, "reward")
    qcfg = _quant_config(8 if cfg.get("quantize_frozen_models", True) else None, dtype) if device.type == "cuda" else None
    kwargs = {"num_labels": 1, "low_cpu_mem_usage": True}
    if qcfg is not None:
        kwargs.update({"quantization_config": qcfg, "device_map": {"": str(device)}})
    else:
        kwargs["dtype"] = dtype

    model = AutoModelForSequenceClassification.from_pretrained(cfg["reward_model"], **kwargs)
    if qcfg is None:
        model = model.to(device)

    # The RM repository tokenizer was incompatible with the pinned Transformers build during
    # instructor preparation. Use the canonical base-policy tokenizer intentionally.
    tok = load_tokenizer(cfg.get("reward_tokenizer", cfg["base_model"]), padding_side="left")
    model.config.pad_token_id = tok.pad_token_id
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def load_value_model(cfg: dict, checkpoint: str, train_mode: str = "lora_head"):
    """Supplied PPO critic; `lora_head` adds a zero-initialized LoRA so training starts from the exact released values."""
    dtype = resolve_dtype(cfg.get("dtype", "float16"))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(repo_path(checkpoint)),
        num_labels=1,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    tok_for_config = load_tokenizer(cfg["base_model"])
    model.config.pad_token_id = tok_for_config.pad_token_id
    model.config.use_cache = False

    if train_mode == "lora_head":
        # The released critic is a full merged checkpoint. Wrapping it with a fresh,
        # zero-initialized LoRA adapter preserves the exact starting value function while
        # giving the continuation run critic capacity comparable to instructor preparation.
        model = get_peft_model(model, make_value_lora_config(cfg))
    elif train_mode == "head_only":
        for name, p in model.named_parameters():
            p.requires_grad_("score" in name or "classifier" in name)
    elif train_mode == "frozen":
        for p in model.parameters():
            p.requires_grad_(False)
        model = model.to(model_device(cfg, "value"))
        model.eval()
        return model
    elif train_mode == "full":
        pass
    else:
        raise ValueError(f"Unknown value train_mode={train_mode!r}")

    upcast_trainable_to_fp32(model)
    model = model.to(model_device(cfg, "value"))
    model.train()
    return model


def upcast_trainable_to_fp32(model) -> None:
    """Keep trainable weights in fp32 while the frozen backbone stays in fp16.

    PEFT already upcasts LoRA matrices, but `modules_to_save` copies (the critic's score head)
    stay fp16; AdamW on fp16 weights underflows its second-moment estimate and produces NaNs.
    """
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()


def disable_dropout(model) -> None:
    """Zero every dropout probability so old/new log-probs agree exactly at the first PPO/GRPO epoch."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


def value_parameter_groups(model, lora_lr: float, head_lr: float):
    """Return optimizer groups for a PEFT critic without implementing the PPO loop."""
    lora_params, head_params, other = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(p)
        elif "score" in name or "classifier" in name:
            head_params.append(p)
        else:
            other.append((name, p))
    if other:
        names = ", ".join(name for name, _ in other[:8])
        raise RuntimeError(f"Unexpected trainable critic parameters: {names}")
    groups = []
    if lora_params:
        groups.append({"params": lora_params, "lr": float(lora_lr), "name": "critic_lora"})
    if head_params:
        groups.append({"params": head_params, "lr": float(head_lr), "name": "critic_head"})
    if not groups:
        raise RuntimeError("Critic has no trainable parameters")
    return groups


def token_values(value_model, input_ids, attention_mask):
    """Scalar value at every sequence position, [batch, seq_len], from the critic's backbone + score head."""
    # PEFT's `.model` forwarding can resolve to the whole classifier, which would
    # prematurely apply its fp32 score head to fp16 hidden states. Unwrap only the
    # outer PEFT wrapper: the decoder still contains its active LoRA layers.
    classifier = value_model.get_base_model() if isinstance(value_model, PeftModel) else value_model
    backbone = getattr(classifier, classifier.base_model_prefix)
    device = next(backbone.parameters()).device
    outputs = backbone(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        return_dict=True,
        use_cache=False,
    )
    hidden = outputs.last_hidden_state
    if hasattr(classifier, "score"):
        head = classifier.score
    elif hasattr(classifier, "classifier"):
        head = classifier.classifier
    else:
        raise RuntimeError("Could not locate scalar value head")
    # The trainable head copy is fp32 (see upcast_trainable_to_fp32); match its dtype.
    head_dtype = next((p.dtype for p in head.parameters() if p.requires_grad), hidden.dtype)
    return head(hidden.to(head_dtype)).squeeze(-1)


def trainable_parameters(model):
    """Parameters with requires_grad (what the optimizer updates)."""
    return [p for p in model.parameters() if p.requires_grad]


def count_parameters(model):
    """(total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def clear_gpu(*objects):
    """Garbage-collect and release cached GPU memory."""
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
