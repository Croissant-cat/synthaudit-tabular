"""Train generators on a *reduced* real minority-label budget.

This is the leakage-controlled counterpart of run_label_budget.py: the generator
only sees the labels that the downstream classifier is allowed to use.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import DATASET_SPECS, Encoding, Split, prepare_dataset  # noqa: E402
from src.evaluate import needed_synthetic_count  # noqa: E402
from src.metrics import fidelity_report  # noqa: E402
from src.train import TrainConfig, load_generator, sample_synthetic, train_generator  # noqa: E402
from src.utils import RUN_DIR, get_logger, save_json  # noqa: E402

log = get_logger()


def reduce_labels(data, budget: float, seed: int):
    pos = np.flatnonzero(data.train.enc.y == 1)
    perm = np.random.default_rng(seed).permutation(pos)[: max(int(round(budget * len(pos))), 20)]
    keep = np.concatenate([np.flatnonzero(data.train.enc.y == 0), perm])
    keep.sort()
    enc = Encoding(num=data.train.enc.num[keep], cat=data.train.enc.cat[keep], y=data.train.enc.y[keep])
    df = data.train.df.iloc[keep].reset_index(drop=True)
    data.train = Split(name="train", df=df, enc=enc,
                       meta={"n": len(df), "n_pos": int(enc.y.sum()), "prevalence": float(enc.y.mean())})
    data.meta["splits"]["train"] = data.train.meta
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", nargs="+", type=float, default=[0.10, 0.25, 0.50])
    ap.add_argument("--models", nargs="+", default=["ddpm"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--dataset", default="adult", choices=["adult", "bank"])
    ap.add_argument("--natural", action="store_true", help="keep the natural prevalence (Bank)")
    ap.add_argument("--ratio", type=float, default=0.05)
    args = ap.parse_args()
    ratio = None if args.natural else args.ratio
    base_tag = f"{args.dataset}_nat" if ratio is None else f"{args.dataset}_r{int(round(ratio * 100)):02d}"
    target = DATASET_SPECS[args.dataset]["target"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig()

    for budget in args.budgets:
        tag = f"{base_tag}_b{int(round(budget * 100)):02d}"
        n_all = None
        for seed in args.seeds:
            # The generator must see exactly the subset that the downstream
            # classifier will use for this seed (no label leakage).
            data = reduce_labels(prepare_dataset(args.dataset, ratio, seed=0), budget, seed)
            n_all = needed_synthetic_count(data)
            real = data.val.df[data.val.df[target] == 1].reset_index(drop=True)
            renc = np.hstack([data.val.enc.num[data.val.enc.y == 1], data.val.enc.cat[data.val.enc.y == 1]])
            log.info("[%s s%d] n_train_pos=%d n_all=%d", tag, seed, int(data.train.enc.y.sum()), n_all)
            for model in args.models:
                rd = RUN_DIR / tag / f"{model}_s{seed}"
                if (rd / "samples_pool.npz").exists():
                    continue
                t0 = time.time()
                train_generator(model, data, seed, cfg, device, rd)
                loaded, cfg_used = load_generator(model, rd, data, device)
                df_all, enc_all = sample_synthetic(model, loaded, data, n_all, seed, device, cfg_used)
                df_pool, enc_pool = sample_synthetic(model, loaded, data, 2 * n_all, seed + 1, device, cfg_used)
                np.savez_compressed(rd / "samples_all.npz", num=enc_all.num, cat=enc_all.cat, y=enc_all.y)
                np.savez_compressed(rd / "samples_pool.npz", num=enc_pool.num, cat=enc_pool.cat, y=enc_pool.y)
                df_all.to_csv(rd / "samples_all.csv", index=False)
                df_pool.to_csv(rd / "samples_pool.csv", index=False)
                fid = fidelity_report(real, df_all, renc, np.hstack([enc_all.num, enc_all.cat]),
                                      data.preprocessor.num_cols, data.preprocessor.cat_cols, seed=seed)
                save_json({"model": model, "seed": seed, "budget": budget, "n_syn": n_all, **fid},
                          rd / "fidelity.json")
                log.info("[%s] %s s%d %.0fs KS=%.3f TV=%.3f MMD=%.3f", tag, model, seed,
                         time.time() - t0, fid["ks_mean"], fid["tv_mean"], fid["mmd"])


if __name__ == "__main__":
    main()
