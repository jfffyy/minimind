"""小模型回归检查：不下载模型，不需要 GPU。"""

import unittest

import torch
import torch.nn.functional as F

from model.model import MiniMindBConfig, MiniMindBForCausalLM


class CausalLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        torch.manual_seed(7)
        config = MiniMindBConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=32,
            max_position_embeddings=32,
            dropout=0.0,
        )
        self.model = MiniMindBForCausalLM(config).eval()
        self.ids = torch.tensor([[1, 4, 5, 6]])

    def test_forward_and_shared_weights(self):
        with torch.no_grad():
            output = self.model(self.ids)
        self.assertEqual(output.logits.shape, (1, 4, 32))
        self.assertTrue(torch.isfinite(output.logits).all())
        self.assertIsNone(output.past_key_values)
        self.assertIs(self.model.model.embed_tokens.weight, self.model.lm_head.weight)
        self.assertEqual(self.model.model.freqs_cos.shape, (32, 8))

    def test_cached_decoding_matches_full_sequence(self):
        with torch.no_grad():
            full = self.model(self.ids).logits
            prefix = self.model(self.ids[:, :2], use_cache=True)
            decoded = self.model(
                self.ids[:, 2:], past_key_values=prefix.past_key_values, use_cache=True
            )
        torch.testing.assert_close(decoded.logits, full[:, 2:], rtol=1e-4, atol=1e-5)
        self.assertEqual(prefix.past_key_values.get_seq_length(), 2)
        self.assertEqual(decoded.past_key_values.get_seq_length(), 4)
        self.assertEqual(decoded.past_key_values.to_legacy_cache()[0][0].shape, (1, 2, 4, 8))

    def test_internal_list_cache_is_accepted(self):
        with torch.no_grad():
            _, cache_list = self.model.model(self.ids[:, :2], use_cache=True)
            self.assertIsInstance(cache_list, list)
            decoded = self.model(
                self.ids[:, 2:], past_key_values=cache_list, use_cache=True
            )
            full = self.model(self.ids)
        torch.testing.assert_close(decoded.logits, full.logits[:, 2:], rtol=1e-4, atol=1e-5)

    def test_logits_selection(self):
        with torch.no_grad():
            full = self.model(self.ids).logits
            last = self.model(self.ids, logits_to_keep=1).logits
            selected = self.model(self.ids, logits_to_keep=torch.tensor([0, 2])).logits
        torch.testing.assert_close(last, full[:, -1:])
        torch.testing.assert_close(selected, full[:, [0, 2]])
        with self.assertRaises(ValueError):
            self.model(self.ids, logits_to_keep=-1)

    def test_outputs_are_independent(self):
        with torch.no_grad():
            first = self.model(self.ids, use_cache=True)
            second = self.model(self.ids[:, :1])
        self.assertIsNot(first, second)
        self.assertEqual(first.logits.shape, (1, 4, 32))
        self.assertEqual(first.past_key_values.get_seq_length(), 4)

    def test_padding_mask_prevents_reading_masked_token(self):
        modified = self.ids.clone()
        modified[0, 1] = 9
        mask = torch.tensor([[1, 0, 1, 1]])
        with torch.no_grad():
            original = self.model(self.ids, attention_mask=mask).logits
            changed = self.model(modified, attention_mask=mask).logits
        torch.testing.assert_close(original[:, 2:], changed[:, 2:], rtol=1e-4, atol=1e-5)

    def test_shifted_cross_entropy_backward(self):
        self.model.train()
        logits = self.model(self.ids).logits
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 32), self.ids[:, 1:].reshape(-1))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))
        self.assertIsNotNone(self.model.model.layers[0].self_attn.q_proj.weight.grad)

    def test_greedy_generation_cache_parity(self):
        options = dict(
            max_new_tokens=3,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=0,
            attention_mask=torch.ones_like(self.ids),
        )
        with torch.no_grad():
            cached = self.model.generate(self.ids, use_cache=True, **options)
            uncached = self.model.generate(self.ids, use_cache=False, **options)
        self.assertEqual(cached.shape, (1, 7))
        self.assertTrue(torch.equal(cached, uncached))

    def test_invalid_input_and_cache_length(self):
        with self.assertRaises(ValueError):
            self.model()
        with self.assertRaises(ValueError):
            self.model(self.ids, past_key_values=[None])
        with self.assertRaises(ValueError):
            self.model(torch.ones((1, 33), dtype=torch.long))


if __name__ == '__main__':
    unittest.main()
