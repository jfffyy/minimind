"""SFT 数据模板、标签和空监督批次的回归测试。"""

from argparse import Namespace
from contextlib import nullcontext
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import torch
from torch.amp.grad_scaler import GradScaler
from transformers.models.auto.tokenization_auto import AutoTokenizer

from dataset.lm_dataset import SFTDataset
from model.model import MiniMindBConfig, MiniMindBForCausalLM
from trainer import train_full_sft as sft


class SFTDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tokenizer_path = Path(__file__).resolve().parents[1] / "model"
        cls.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), local_files_only=True, trust_remote_code=False,
        )

    def make_dataset(self, messages, max_length=512):
        with patch("dataset.lm_dataset.load_dataset", return_value=[{"conversations": messages}]):
            return SFTDataset("unused.jsonl", self.tokenizer, max_length)

    def test_tool_dialogue_parses_json_fields_without_mutating_input(self):
        definitions = [
            {"type": "function", "function": {"name": "lookup_color", "parameters": {}}}
        ]
        calls = [{"function": {"name": "lookup_color", "arguments": {}}}]
        for definitions_key, value in (("tools", json.dumps(definitions)), ("functions", definitions)):
            with self.subTest(definitions_field=definitions_key):
                messages = [
                    {"role": "system", "content": "SYSTEM_NOTE", definitions_key: value},
                    {"role": "user", "content": "USER_QUESTION"},
                    {"role": "assistant", "content": None, "tool_calls": json.dumps(calls)},
                    {"role": "tool", "content": "TOOL_RESULT"},
                    {"role": "assistant", "content": "FINAL_ANSWER"},
                ]
                before = json.dumps(messages)
                data = self.make_dataset(messages)
                ids, labels, mask = data[0]
                full_text = self.tokenizer.decode(ids[mask.bool()])
                targets = self.tokenizer.decode(labels[labels != -100])
                self.assertIn("<tools>", full_text)
                self.assertIn("TOOL_RESULT", full_text)
                self.assertIn("<tool_call>", targets)
                self.assertIn("lookup_color", targets)
                self.assertIn("FINAL_ANSWER", targets)
                self.assertIn(self.tokenizer.eos_token, targets)
                for context in ("SYSTEM_NOTE", "USER_QUESTION", "TOOL_RESULT"):
                    self.assertNotIn(context, targets)
                self.assertTrue((labels[mask == 0] == -100).all())
                self.assertEqual(json.dumps(messages), before)

    def test_truncated_prompt_has_no_targets(self):
        data = self.make_dataset([
            {"role": "system", "content": "中文上下文。" * 50},
            {"role": "user", "content": "测试"},
            {"role": "assistant", "content": "回答"},
        ], max_length=32)
        ids, labels, mask = data[0]
        self.assertEqual(ids.shape, mask.shape)
        self.assertTrue((labels == -100).all())


class SFTLoopTests(unittest.TestCase):
    def test_empty_supervision_does_not_update_adamw_or_produce_nan(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        model = MiniMindBForCausalLM(MiniMindBConfig(
            hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, vocab_size=16,
            max_position_embeddings=8, dropout=0.0,
        ))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        args = Namespace(
            device="cpu", epochs=1, learning_rate=1e-3, accumulation_steps=1,
            grad_clip=1.0, log_interval=1, save_interval=100,
        )
        ids = torch.tensor([[1, 4, 2, 0]])
        mask = (ids != 0).long()
        valid = (ids, torch.tensor([[-100, 4, 2, -100]]), mask)
        empty = (ids, torch.full_like(ids, -100), mask)
        wandb = Mock()
        with patch.multiple(
            sft, args=args, model=model, optimizer=optimizer,
            scaler=GradScaler("cpu", enabled=False), autocast_ctx=nullcontext(),
            is_main_process=lambda: False, Logger=Mock(), create=True,
        ):
            sft.train_epoch(0, [valid], 1, wandb=wandb)
            # 先建立 AdamW 动量，再检查空批不会因动量或权重衰减更新。
            before = model.lm_head.weight.detach().clone()
            sft.train_epoch(0, [empty], 1, wandb=wandb)
        self.assertGreater(wandb.log.call_args_list[0].args[0]["loss"], 0)
        self.assertEqual(wandb.log.call_args_list[1].args[0]["loss"], 0)
        self.assertTrue(torch.equal(before, model.lm_head.weight))
