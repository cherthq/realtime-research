"""Actual CPU-tensor cache tests; no GPU or weights needed."""

import importlib.util
import unittest

from scripts.liveavatar_cache import grow_conditioning_cache, install_cache_guard


@unittest.skipUnless(importlib.util.find_spec("torch"), "Requires CPU-capable PyTorch")
class CacheTensorTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        self.cache = {
            "cond_k": torch.ones((1, 2800, 2, 4), dtype=torch.bfloat16),
            "cond_v": torch.full((1, 2800, 2, 4), 2, dtype=torch.bfloat16),
            "cond_end": torch.tensor([1200]),
        }

    def test_growth_preserves_prefix_device_dtype_and_cursor(self):
        self.assertTrue(grow_conditioning_cache(self.cache, 2980))
        for name, expected in (("cond_k", 1), ("cond_v", 2)):
            tensor = self.cache[name]
            self.assertEqual(tensor.shape, (1, 2980, 2, 4))
            self.assertEqual(tensor.device.type, "cpu")
            self.assertEqual(tensor.dtype, self.torch.bfloat16)
            self.assertTrue((tensor[:, :2800] == expected).all())
            self.assertTrue((tensor[:, 2800:] == 0).all())
        self.assertEqual(self.cache["cond_end"].item(), 1200)

    def test_existing_capacity_is_not_reallocated(self):
        original = self.cache["cond_k"]
        self.assertFalse(grow_conditioning_cache(self.cache, 2800))
        self.assertIs(original, self.cache["cond_k"])

    def test_prefill_can_write_all_2980_tokens_without_truncation(self):
        grow_conditioning_cache(self.cache, 2980)
        source = self.torch.arange(2980).view(1, 2980, 1, 1).expand(1, 2980, 2, 4)
        self.cache["cond_k"][:, :2980] = source
        self.assertEqual(
            self.cache["cond_k"][0, -1, 0, 0],
            source.to(self.torch.bfloat16)[0, -1, 0, 0],
        )

    def test_unexpected_capacity_is_rejected(self):
        with self.assertRaises(ValueError):
            grow_conditioning_cache(self.cache, 20000)


class HookTests(unittest.TestCase):
    def test_upstream_signature_change_is_rejected(self):
        class Changed:
            def forward(self, x):
                return x

        with self.assertRaisesRegex(ValueError, "signature changed"):
            install_cache_guard(Changed)


if __name__ == "__main__":
    unittest.main()
