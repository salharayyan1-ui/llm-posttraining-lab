"""Regression checks for fixed subsets, prompt conditioning and uneven batches.

Uses the already cached course tokenizer; no weights or CUDA needed.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from common.data import encode_prompt_response, fit_prompt_messages, prompt_messages_from_preference, read_jsonl
from common.runs import artifact_is_current, mark_artifact_complete
from task1_dpo.train import normalize_accumulated_gradients


class TrainingProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True)

    def test_long_prompt_header_and_dpo_shared_context(self):
        tok = self.tokenizer
        messages = [{"role": "user", "content": "Old context. " * 2000 + " Latest question: what is 2+2?"}]
        fitted = fit_prompt_messages(tok, messages, 256)
        ids = tok.apply_chat_template(fitted, tokenize=True, add_generation_prompt=True)
        self.assertLessEqual(len(ids), 256)
        self.assertTrue(tok.decode(ids).endswith("<|im_start|>assistant\n"))
        self.assertIn("what is 2+2?", fitted[-1]["content"])
        for response in ["4", "5" * 1000]:
            encoded, mask = encode_prompt_response(tok, messages, response, 768, 256)
            self.assertEqual(encoded[:len(ids)], ids)
            self.assertEqual(mask[:len(ids)], [0] * len(ids))
            self.assertEqual(encoded[-1], tok.eos_token_id)
            self.assertLessEqual(len(encoded), 768)
        self.assertIn("Old context.", messages[0]["content"])  # caller's input unchanged

    def test_every_balanced_pair_remains_encodable(self):
        from common.data import load_yaml, preference_responses
        cfg = load_yaml("configs/dpo.yaml")
        rows = read_jsonl(cfg["paths"]["dpo_length_train"])
        strata = {}
        for row in rows:
            strata[row["length_stratum"]] = strata.get(row["length_stratum"], 0) + 1
            for response in preference_responses(row):
                ids, mask = encode_prompt_response(self.tokenizer, prompt_messages_from_preference(row), response, 768, 512)
                self.assertTrue(any(mask))
                self.assertLessEqual(len(ids), 768)
        self.assertEqual(len(rows), 1500)
        self.assertEqual(sorted(strata.values()), [500, 500, 500])

    def test_uneven_microbatches_equal_pair_mean_gradient(self):
        x = torch.tensor([1., 2., 3., 4., 9.])
        actual = torch.nn.Parameter(torch.tensor(0.7))
        for chunk in [x[:2], x[2:4], x[4:]]:
            ((actual * chunk - 1).square().mean() * len(chunk)).backward()
        normalize_accumulated_gradients([actual], len(x))
        expected = torch.nn.Parameter(torch.tensor(0.7))
        (expected * x - 1).square().mean().backward()
        torch.testing.assert_close(actual.grad, expected.grad)

    def test_receipt_rejects_incomplete_or_changed_adapter(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "adapter_config.json"
            path.write_text("{}")
            weights = Path(folder) / "adapter_model.safetensors"
            weights.write_bytes(b"original")
            self.assertFalse(artifact_is_current(path, "protocol"))
            mark_artifact_complete(path, "protocol")
            self.assertTrue(artifact_is_current(path, "protocol"))
            self.assertFalse(artifact_is_current(path, "other protocol"))
            weights.write_bytes(b"changed")
            self.assertFalse(artifact_is_current(path, "protocol"))

    def test_judge_diagnostic_physically_reverses_candidates(self):
        from task5_feedback.rlaif import PairwiseAIJudge
        seen = []
        class Tokenizer:
            eos_token_id = 2
            def apply_chat_template(self, messages, **kwargs):
                seen.append(messages[0]["content"])
                return torch.tensor([[1]])
            def decode(self, ids, **kwargs):
                return "A"
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.config = SimpleNamespace(_commit_hash="fixed")
                self.generation_config = SimpleNamespace(to_dict=lambda: {})
            def generate(self, ids, **kwargs):
                return torch.tensor([[1, 2]])
        with tempfile.TemporaryDirectory() as folder:
            judge = PairwiseAIJudge.__new__(PairwiseAIJudge)
            judge.cfg, judge.cache = {"ai_judge_model": "fixture"}, {}
            judge.cache_path = Path(folder) / "cache.json"
            judge.model, judge.tokenizer = Model(), Tokenizer()
            judge.compare("question", "left", "right", balance_orientation=False)
            judge.compare("question", "right", "left", balance_orientation=False)
        self.assertEqual(len(seen), 2)
        self.assertIn("Candidate A:\nleft\n\nCandidate B:\nright", seen[0])
        self.assertIn("Candidate A:\nright\n\nCandidate B:\nleft", seen[1])

    def test_audit_labels_cannot_follow_changed_response_text(self):
        import pandas as pd
        from common.data import write_jsonl
        from task4_safety.make_audit_sheet import blind_sheet
        from task4_safety.evaluate_safety import audit_analysis
        rows = [
            {"policy": "sft", "xstest_id": 1, "prompt": "benign", "response": "answer", "benchmark_class": "SAFE", "type": "fixture", "judge_label": "SAFE_ANSWER"},
            {"policy": "sft", "xstest_id": 2, "prompt": "harmful", "response": "refuse", "benchmark_class": "UNSAFE", "type": "fixture", "judge_label": "JUSTIFIED_REFUSAL"},
        ]
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            write_jsonl(out / "generated_sft.jsonl", rows)
            sheet, key = blind_sheet(out, [1, 2], ["sft"], 42)
            sheet["manual_label"] = sheet["response"].map({"answer": "SAFE_ANSWER", "refuse": "JUSTIFIED_REFUSAL"})
            sheet.to_csv(out / "manual_audit_sheet.csv", index=False)
            key.to_csv(out / "manual_audit_key.csv", index=False)
            judged = pd.DataFrame(rows).assign(judge_confidence=1., judge_rationale_tag="fixture")
            summary, _, _ = audit_analysis(out, judged)
            self.assertEqual(summary["agreement"], 1.)
            judged.loc[0, "response"] = "changed"
            with self.assertRaisesRegex(ValueError, "does not match"):
                audit_analysis(out, judged)


if __name__ == "__main__":
    unittest.main()
