"""预训练循环检查：CPU 小模型、字典批次及临时检查点。"""

from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, Dataset, Subset

from model.model import MiniMindBConfig, MiniMindBForCausalLM
from trainer import trainer_pretrain as pretrain


class TensorSampleDataset(Dataset[dict[str, torch.Tensor]]):
    """把测试样本包装成 DataLoader 要求的 Dataset 接口。"""

    def __init__(self, samples: list[dict[str, torch.Tensor]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


class PretrainLoopTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, original_threads)
        temporary = tempfile.TemporaryDirectory(prefix="minimind-pretrain-test-")
        self.save_dir = Path(temporary.name).resolve()
        self.assertTrue(self.save_dir.is_relative_to(Path(tempfile.gettempdir()).resolve()))
        self.assertTrue(self.save_dir.name.startswith("minimind-pretrain-test-"))
        self.addCleanup(temporary.cleanup)
        self.config = MiniMindBConfig(
            hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, vocab_size=16,
            max_position_embeddings=8, dropout=0.0,
        )
        self.model = MiniMindBForCausalLM(self.config)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        self.args = Namespace(
            device="cpu", epochs=1, learning_rate=1e-3, accumulation_steps=2,
            grad_clip=1000.0, log_interval=100, save_interval=100,
            save_dir=str(self.save_dir), save_weight="pretrain",
        )
        self.samples: list[dict[str, torch.Tensor]] = []
        for token in range(3, 8):
            ids = torch.tensor([1, token, 2, 0])
            labels = ids.clone()
            labels[-1] = -100
            self.samples.append({
                "input_ids": ids, "labels": labels,
                "attention_mask": torch.tensor([1, 1, 1, 0]),
            })
        self.dataset = TensorSampleDataset(self.samples)
        environment = patch.multiple(
            pretrain, args=self.args, model=self.model, optimizer=self.optimizer,
            scaler=GradScaler("cuda", enabled=False),
            autocast_ctx=nullcontext(), lm_config=self.config,
            CHECKPOINT_DIR=str(self.save_dir / "checkpoints"), create=True,
        )
        environment.start()
        self.addCleanup(environment.stop)

    def test_dictionary_batches_loss_and_final_accumulation(self):
        before = self.model.lm_head.weight.detach().clone()
        loader = DataLoader(self.dataset, batch_size=1)
        with patch.object(self.optimizer, "step", wraps=self.optimizer.step) as update:
            pretrain.train_epoch(0, loader, len(loader))
        self.assertEqual(update.call_count, 3)  # 2+2+1，最后不足两批也需更新。
        self.assertFalse(torch.equal(before, self.model.lm_head.weight))
        self.assertTrue(torch.isfinite(self.model.lm_head.weight).all())
        saved = pretrain.lm_checkpoint(
            self.config, weight="pretrain", save_dir=pretrain.CHECKPOINT_DIR,
        )
        if saved is None:
            self.fail("没有读到刚保存的训练检查点")
        self.assertEqual(saved["step"], 5)
        self.assertIn("scaler", saved)

    def test_partial_accumulation_matches_reference_batches(self):
        reference = MiniMindBForCausalLM(self.config).train()
        reference.load_state_dict(self.model.state_dict())
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=1e-3)
        loader = DataLoader(self.dataset, batch_size=1)
        with patch.object(pretrain, "get_lr", return_value=1e-3):
            pretrain.train_epoch(0, loader, len(loader))

        # 参考实现每次直接放入 2 条，最后一组只放入 1 条。
        for batch in DataLoader(self.dataset, batch_size=2):
            logits = reference(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            ).logits
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                batch["labels"][:, 1:].reshape(-1), ignore_index=-100,
            )
            loss.backward()
            reference_optimizer.step()
            reference_optimizer.zero_grad()
        for name, tensor in self.model.state_dict().items():
            torch.testing.assert_close(
                tensor, reference.state_dict()[name], rtol=1e-4, atol=1e-6,
            )

    def test_resume_counts_only_remaining_batches(self):
        loader = DataLoader(Subset(self.dataset, range(3, len(self.dataset))), batch_size=1)
        with patch.object(self.optimizer, "step", wraps=self.optimizer.step) as update:
            pretrain.train_epoch(0, loader, iters=5, start_step=3)
        self.assertEqual(update.call_count, 1)

    def test_checkpoint_waits_for_accumulation_boundary(self):
        self.args.save_interval = 3
        loader = DataLoader(self.dataset, batch_size=1)
        with patch.object(pretrain, "lm_checkpoint", wraps=pretrain.lm_checkpoint) as save:
            pretrain.train_epoch(0, loader, len(loader))
        self.assertEqual([call.kwargs["step"] for call in save.call_args_list], [4, 5])

    def test_invalid_accumulation_steps(self):
        self.args.accumulation_steps = 0
        loader = DataLoader(self.dataset, batch_size=1)
        with self.assertRaisesRegex(ValueError, "accumulation_steps"):
            pretrain.train_epoch(0, loader, len(loader))


if __name__ == "__main__":
    unittest.main()
