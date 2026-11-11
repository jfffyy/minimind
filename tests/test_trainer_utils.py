"""训练工具回归检查：使用 CPU 和本地临时权重，不下载模型或 tokenizer。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, sentinel

import torch

from model.model import MiniMindBConfig, MiniMindBForCausalLM
from trainer.trainer_utils import (
    SkipBatchSampler,
    get_lr,
    init_distributed_mode,
    init_model,
    lm_checkpoint,
)


class TrainerUtilsTests(unittest.TestCase):
    def setUp(self):
        self.config = MiniMindBConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=16,
            max_position_embeddings=8,
        )
        temporary = tempfile.TemporaryDirectory(prefix="minimind-trainer-utils-test-")
        self.save_dir = Path(temporary.name).resolve()
        self.assertTrue(self.save_dir.is_relative_to(Path(tempfile.gettempdir()).resolve()))
        self.assertTrue(self.save_dir.name.startswith("minimind-trainer-utils-test-"))
        self.addCleanup(temporary.cleanup)

    def test_init_model_from_scratch_and_local_weights(self):
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=sentinel.tokenizer) as load_tokenizer:
            model, tokenizer = init_model(self.config, from_weight="none", device="cpu")
            self.assertIsInstance(model, MiniMindBForCausalLM)
            self.assertIs(tokenizer, sentinel.tokenizer)
            self.assertEqual(next(model.parameters()).device.type, "cpu")
            self.assertEqual(load_tokenizer.call_args.args[0], str(Path(__file__).resolve().parents[1] / "model"))

            torch.save(model.state_dict(), self.save_dir / "pretrain_16.pth")
            restored, _ = init_model(
                self.config,
                from_weight="pretrain",
                tokenizer_path="local-tokenizer",
                save_dir=str(self.save_dir),
                device="cpu",
            )
            load_tokenizer.assert_called_with("local-tokenizer")
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], tensor)

    def test_checkpoint_validation_and_restore(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model(torch.ones(1, 2)).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad()

        with self.assertRaisesRegex(ValueError, "optimizer"):
            lm_checkpoint(self.config, model=model, save_dir=str(self.save_dir))
        self.assertEqual(list(self.save_dir.iterdir()), [])

        lm_checkpoint(
            self.config, model=model, optimizer=optimizer,
            epoch=1, step=7, save_dir=str(self.save_dir),
        )
        saved = lm_checkpoint(self.config, save_dir=str(self.save_dir))
        if saved is None:
            self.fail("没有读到刚保存的训练检查点")
        self.assertEqual((saved["epoch"], saved["step"]), (1, 7))
        self.assertTrue(saved["optimizer"]["state"])
        restored = torch.optim.AdamW(model.parameters(), lr=0.1)
        restored.load_state_dict(saved["optimizer"])
        self.assertEqual(restored.param_groups[0]["lr"], 1e-4)
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(saved["model"][name], tensor)
        self.assertEqual(len(list(self.save_dir.glob("*.pth"))), 2)
        self.assertEqual(list(self.save_dir.glob("*.tmp")), [])

    def test_default_weight_directory_is_project_relative(self):
        source = MiniMindBForCausalLM(self.config)
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=sentinel.tokenizer):
            with patch("torch.load", return_value=source.state_dict()) as load_weights:
                init_model(self.config, from_weight="pretrain", device="cpu")
        expected = Path(__file__).resolve().parents[1] / "out" / "pretrain_16.pth"
        load_weights.assert_called_once_with(str(expected), map_location="cpu")

    def test_learning_rate_and_skipped_batches(self):
        for step, factor in ((0, 1.0), (50, 0.55), (100, 0.1)):
            self.assertAlmostEqual(get_lr(step, 100, 1e-4), factor * 1e-4)
        batches = [list(range(4)), list(range(4, 8)), [8, 9]]
        for skipped in range(5):
            sampler = SkipBatchSampler(range(10), batch_size=4, skip_batches=skipped)
            self.assertEqual(list(sampler), batches[skipped:])
            self.assertEqual(len(sampler), len(batches[skipped:]))

    def test_single_process_initialization(self):
        with patch.dict("os.environ", {}, clear=True), patch("torch.distributed.init_process_group") as initialize:
            self.assertEqual(init_distributed_mode(), 0)
            initialize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
