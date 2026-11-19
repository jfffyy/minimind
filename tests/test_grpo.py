"""GRPO 的输入、奖励编码和短梯度累积回归测试。"""

from argparse import Namespace
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import torch
from torch.amp.grad_scaler import GradScaler
from transformers.models.auto.tokenization_auto import AutoTokenizer

from trainer import train_grpo as grpo


class GRPOTests(unittest.TestCase):
    def test_fit_prompt_preserves_user_and_assistant_boundary(self):
        tokenizer = AutoTokenizer.from_pretrained(
            str(Path(__file__).resolve().parents[1] / "model"), local_files_only=True,
        )
        turns = [
            {"role": "user", "content": "Old conversation " * 40},
            {"role": "assistant", "content": "Old answer"},
            {"role": "user", "content": "One plus one?"},
            {"role": "assistant", "content": "Reference answer"},
        ]
        with patch.object(grpo, "load_dataset", return_value=[{"conversations": turns}]):
            data = grpo.GRPOPromptDataset("irrelevant.jsonl")
        text, context = grpo.fit_prompt(tokenizer, data[0], max_length=64)
        self.assertEqual(context, [turns[2]])
        self.assertLessEqual(len(tokenizer.encode(text, add_special_tokens=False)), 64)
        self.assertTrue(text.endswith(f"{tokenizer.bos_token}assistant\n"))
        self.assertNotIn("Reference answer", text)

    def test_mask_stops_at_first_eos_and_ignores_padding(self):
        ids = torch.tensor([[9, 2, 2, 0], [4, 5, 0, 0], [8, 7, 6, 3]])
        self.assertEqual(grpo.completion_mask(ids, 0, 2).tolist(), [
            [True, True, False, False], [True, True, False, False], [True, True, True, True],
        ])
        self.assertEqual(grpo.completion_mask(ids[:1], 2, 2).tolist(), [[True, True, False, False]])

    def test_reward_token_ids_remapped_before_embedding_lookup(self):
        tokenizer = Mock(return_value={"input_ids": torch.tensor([[1, 12, 3, 11]])})
        ids = grpo.encode_reward_chat(tokenizer, [], {12: 8, 11: 9}, reward_id=9, vocab_size=10)
        self.assertEqual(ids.tolist(), [[1, 8, 3, 9]])
        with self.assertRaisesRegex(ValueError, "超出权重词表"):
            grpo.encode_reward_chat(tokenizer, [], {12: 8}, reward_id=9, vocab_size=10)

    def test_accumulation_updates_tail_and_saves_at_boundary(self):
        model = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.constant_(model.weight, 1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        scaler = GradScaler("cpu", enabled=False)
        args = Namespace(
            device="cpu", dtype="bfloat16", accumulation_steps=2, grad_clip=1.0,
            log_interval=100, save_interval=1,
        )
        with patch.object(grpo, "rollout_loss", side_effect=lambda *unused: (model.weight.square().sum(), 1.0, 2.0)):
            with patch.object(grpo, "Logger"), patch.object(optimizer, "step", wraps=optimizer.step) as step:
                save = Mock()
                grpo.train_epoch(0, [None] * 3, 0, model, None, None, None, optimizer,
                                 scheduler, scaler, args, save)
        self.assertEqual(step.call_count, 2)
        self.assertEqual([call.args for call in save.call_args_list], [(0, 2), (0, 3)])
