"""Fast smoke tests (no full training required).

Run with:  python -m pytest tests -q   or   python tests/test_smoke.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import prepare_dataset  # noqa: E402
from src.metrics import coverage_rate, dcr_ratio, mmd_rbf  # noqa: E402
from src.models import GaussianDiffusion, TabularDenoiser, TabularGenerator, TabularVAE, vae_loss  # noqa: E402


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = prepare_dataset("adult", 0.05, seed=0, test_frac=0.2, val_frac=0.2)
        cls.n_num = cls.data.preprocessor.n_num
        cls.n_cat = cls.data.preprocessor.n_cat

    def test_encoding_shapes(self):
        d = self.data
        self.assertEqual(d.train.enc.num.shape[1], self.n_num)
        self.assertEqual(d.train.enc.cat.shape[1], self.n_cat)
        self.assertAlmostEqual(float(d.train.enc.y.mean()), 0.05, places=2)
        self.assertEqual(len(d.preprocessor.cat_slices), len(d.preprocessor.cat_cols))

    def test_decode_roundtrip(self):
        enc = self.data.train.enc
        df = self.data.preprocessor.to_frame(enc.num[:10], enc.cat[:10], mode="prob")
        self.assertEqual(len(df), 10)
        self.assertFalse(df.isna().any().any())

    def test_vae_forward(self):
        num = torch.tensor(self.data.train.enc.num[:32])
        cat = torch.tensor(self.data.train.enc.cat[:32])
        y = torch.tensor(self.data.train.enc.y[:32])
        m = TabularVAE(self.n_num, self.n_cat)
        loss, parts = vae_loss(m, num, cat, y, self.data.preprocessor.cat_slices)
        self.assertTrue(torch.isfinite(loss))
        n2, c2 = m.sample(8, y[:8], "cpu")
        self.assertEqual(tuple(n2.shape), (8, self.n_num))
        self.assertEqual(tuple(c2.shape), (8, self.n_cat))

    def test_gan_forward(self):
        g = TabularGenerator(self.n_num, self.n_cat)
        n, c = g(torch.randn(8, 32), torch.ones(8, dtype=torch.long))
        self.assertEqual(tuple(n.shape), (8, self.n_num))
        self.assertEqual(tuple(c.shape), (8, self.n_cat))

    def test_ddpm_loss_and_sampling(self):
        d = self.data
        m = TabularDenoiser(self.n_num, self.n_cat)
        diff = GaussianDiffusion(50, device="cpu")
        x0 = torch.tensor(np.hstack([d.train.enc.num[:16], d.train.enc.cat[:16]]), dtype=torch.float32)
        y = torch.tensor(d.train.enc.y[:16])
        loss = diff.training_loss_hybrid(m, x0, y, self.n_num, d.preprocessor.cat_slices)
        self.assertTrue(torch.isfinite(loss))
        num, probs = diff.ddim_sample_hybrid(m, 4, y[:4], self.n_num, d.preprocessor.cat_slices, steps=5)
        self.assertEqual(tuple(num.shape), (4, self.n_num))
        self.assertEqual(tuple(probs.shape), (4, self.n_cat))

    def test_metric_ranges(self):
        rng = np.random.default_rng(0)
        real = rng.normal(size=(120, 12))
        syn = rng.normal(size=(300, 12)) * 1.2
        self.assertGreaterEqual(mmd_rbf(real, syn), 0.0)
        self.assertTrue(0.0 <= coverage_rate(real, syn) <= 1.0)
        self.assertGreater(dcr_ratio(real, syn), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
