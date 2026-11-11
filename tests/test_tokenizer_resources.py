"""本地真实分词器检查：不下载预训练权重，不启动正式训练。"""

from pathlib import Path
import unittest

import torch
import torch.nn.functional as F
from transformers.models.auto.tokenization_auto import AutoTokenizer

from model.model import MiniMindBConfig
from trainer.trainer_utils import init_model


class TokenizerResourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        cls.tokenizer_dir = Path(__file__).resolve().parents[1] / "model"
        cls.tokenizer = AutoTokenizer.from_pretrained(
            str(cls.tokenizer_dir), local_files_only=True, trust_remote_code=False,
        )

    def test_vocab_and_special_tokens_match_model(self):
        config = MiniMindBConfig()
        self.assertEqual(len(self.tokenizer), config.vocab_size)
        self.assertEqual(self.tokenizer.bos_token_id, config.bos_token_id)
        self.assertEqual(self.tokenizer.eos_token_id, config.eos_token_id)
        self.assertEqual(self.tokenizer.pad_token_id, 0)

    def test_chinese_and_english_round_trip(self):
        text = "你好，MiniMindB! 123"
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        self.assertTrue(ids)
        self.assertTrue(all(0 <= token_id < 6400 for token_id in ids))
        self.assertEqual(self.tokenizer.decode(ids), text)

    def test_real_tokenizer_model_forward_and_backward(self):
        config = MiniMindBConfig(
            hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=64,
        )
        model, tokenizer = init_model(config, from_weight="none", device="cpu")
        tokens = (
            [tokenizer.bos_token_id]
            + tokenizer.encode("你好，欢迎学习大语言模型。", add_special_tokens=False)
            + [tokenizer.eos_token_id]
        )
        input_ids = torch.tensor([tokens], dtype=torch.long)
        logits = model(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
        ).logits
        self.assertEqual(logits.shape, (1, len(tokens), 6400))
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, 6400), input_ids[:, 1:].reshape(-1),
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertIsNotNone(model.get_parameter("lm_head.weight").grad)


if __name__ == "__main__":
    unittest.main()
