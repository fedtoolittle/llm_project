import os
import tempfile
import unittest

import torch

from generate import CheckpointGenerator, _coerce_mappings, _load_checkpoint
from transformer import TransformerModel


class GenerateTests(unittest.TestCase):
    def _write_checkpoint(self, path, max_len=8, with_mappings=True):
        vocab = ["a", "b", "c", " "]
        char_to_idx = {ch: i for i, ch in enumerate(vocab)}
        model = TransformerModel(
            vocab_size=len(vocab),
            embed_size=8,
            num_heads=2,
            num_layers=1,
            max_len=max_len,
        )
        ckpt = {
            "checkpoint_version": 2,
            "model_state": model.state_dict(),
            "vocab_size": len(vocab),
            "embed_size": 8,
            "num_heads": 2,
            "num_layers": 1,
            "max_len": max_len,
        }
        if with_mappings:
            ckpt["char_to_idx"] = char_to_idx
            ckpt["idx_to_char"] = {i: ch for ch, i in char_to_idx.items()}
        torch.save(ckpt, path)

    def test_load_checkpoint_missing_required_keys(self):
        with tempfile.TemporaryDirectory() as td:
            bad_path = os.path.join(td, "bad.pth")
            torch.save({"vocab_size": 4}, bad_path)
            with self.assertRaises(KeyError):
                _load_checkpoint(bad_path)

    def test_coerce_mappings_is_strict_by_default(self):
        with self.assertRaises(ValueError):
            _coerce_mappings({"vocab_size": 2, "model_state": {}})

    def test_coerce_mappings_allow_legacy_rebuild(self):
        with tempfile.TemporaryDirectory() as td:
            data_path = os.path.join(td, "data.txt")
            with open(data_path, "w", encoding="utf-8") as f:
                f.write("abba")
            char_to_idx, idx_to_char = _coerce_mappings(
                {"vocab_size": 2, "model_state": {}},
                allow_rebuild_from_data=True,
                data_path=data_path,
            )
            self.assertEqual(set(char_to_idx.keys()), {"a", "b"})
            self.assertEqual({idx_to_char[i] for i in idx_to_char}, {"a", "b"})

    def test_generation_clips_context_to_max_window(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt_path = os.path.join(td, "ok.pth")
            self._write_checkpoint(ckpt_path, max_len=8)

            generator = CheckpointGenerator(checkpoint_path=ckpt_path, device="cpu")
            # Start sequence intentionally longer than checkpoint max_len.
            out = generator.generate(start_seq="ab cab cab cab", max_len=12, temperature=1.0)
            self.assertIsInstance(out, str)
            self.assertGreaterEqual(len(out), 12)


if __name__ == "__main__":
    unittest.main()
