"""Offline tiny-Qwen regression tests; no pretrained weights or dataset downloads.

Run: python -m tests.test_model_execution
The cross-GPU test requires two CUDA GPUs and otherwise reports a skip.
"""
from __future__ import annotations

import unittest

import torch
from peft import get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2ForSequenceClassification

from common.generation import batch_generate, response_token_logprobs
from common.models import make_value_lora_config, token_values, upcast_trainable_to_fp32
from task2_ppo.continue_train import response_values


def tiny_config():
    return Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
        num_labels=1, pad_token_id=0, eos_token_id=2,
    )


def wrapped_critic(device="cpu"):
    model = Qwen2ForSequenceClassification(tiny_config()).to(device).half().eval()
    cfg = {"lora": {"r": 2, "alpha": 4, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]}}
    model = get_peft_model(model, make_value_lora_config(cfg))
    upcast_trainable_to_fp32(model)
    return model


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        return [3, 4, 5] if kwargs.get("tokenize") else "tiny prompt"

    def __call__(self, texts, **kwargs):
        ids = torch.tensor([[3, 4, 5] for _ in texts])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids.tolist()))


class ModelExecutionTests(unittest.TestCase):
    def test_critic_initial_values_and_mixed_dtype_backward(self):
        torch.manual_seed(12)
        base = Qwen2ForSequenceClassification(tiny_config()).half().eval()
        ids = torch.tensor([[3, 4, 5, 6]])
        mask = torch.ones_like(ids)
        with torch.no_grad():
            expected = base.score(base.model(input_ids=ids, attention_mask=mask).last_hidden_state).squeeze(-1).float()
        cfg = {"lora": {"r": 2, "alpha": 4, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]}}
        model = get_peft_model(base, make_value_lora_config(cfg))
        upcast_trainable_to_fp32(model)
        actual = token_values(model, ids, mask)
        torch.testing.assert_close(actual.float(), expected, atol=1e-3, rtol=1e-3)
        actual.float().square().mean().backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertTrue(any(g.abs().sum() > 0 for g in grads))
        lora_grads = [p.grad for n, p in model.named_parameters() if "lora_" in n and p.grad is not None]
        self.assertTrue(any(g.abs().sum() > 0 for g in lora_grads))

    def test_rollout_indices_can_be_saved_for_policy_backward(self):
        torch.manual_seed(13)
        policy = Qwen2ForCausalLM(tiny_config()).train()
        gen = batch_generate(policy, TinyTokenizer(), [[{"role": "user", "content": "x"}]], 8, 3, do_sample=False)
        self.assertTrue(policy.training)
        self.assertFalse(gen["sequences"].is_inference())
        self.assertFalse(gen["response_ids"].is_inference())
        logp, _ = response_token_logprobs(policy, gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"])
        (-logp.mean()).backward()
        self.assertTrue(torch.isfinite(policy.lm_head.weight.grad).all())
        self.assertGreater(float(policy.lm_head.weight.grad.abs().sum()), 0)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA GPUs")
    def test_critic_backward_across_two_gpus(self):
        torch.manual_seed(14)
        critic = wrapped_critic("cuda:1")
        ids = torch.tensor([[3, 4, 5, 6, 7]], device="cuda:0")
        rollout = {
            "sequences": ids, "attention_mask": torch.ones_like(ids),
            "prompt_width": 3, "response_ids": ids[:, 3:],
        }
        values = response_values(critic, rollout)
        self.assertEqual(values.device, ids.device)
        self.assertEqual(tuple(values.shape), (1, 2))
        before = {n: p.detach().clone() for n, p in critic.named_parameters() if p.requires_grad}
        optimizer = torch.optim.AdamW([p for p in critic.parameters() if p.requires_grad], lr=1e-3)
        (values - 1.0).square().mean().backward()
        grads = [p.grad for p in critic.parameters() if p.requires_grad and p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(g.device == torch.device("cuda:1") and torch.isfinite(g).all() for g in grads))
        optimizer.step()
        self.assertTrue(any(not torch.equal(before[n], p.detach()) for n, p in critic.named_parameters() if n in before))


if __name__ == "__main__":
    unittest.main()
