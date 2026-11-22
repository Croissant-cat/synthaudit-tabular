"""Train the three generators and persist synthetic samples.

Example
-------
python scripts/run_generation.py --models vae gan ddpm --seeds 0 1 2
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

from src.data import DATASET_SPECS, prepare_dataset  # noqa: E402
from src.evaluate import needed_synthetic_count  # noqa: E402
from src.metrics import fidelity_report  # noqa: E402
from src.train import TrainConfig, load_generator, sample_synthetic, train_generator  # noqa: E402
from src.utils import METRIC_DIR, RUN_DIR, get_logger, save_json  # noqa: E402

log = get_logger()

SETTINGS = [
    ("adult", 0.05, list(range(5))),
    ("adult", 0.10, [0, 1]),
    ("bank", None, [0, 1, 2]),
]


def tag_for(dataset: str, ratio: float | None) -> str:
    if ratio is None:
        return f"{dataset}_nat"
    return f"{dataset}_r{int(round(ratio * 100)):02d}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["vae", "gan", "ddpm"])
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="override seeds for every setting")
    p.add_argument("--setting", default=None, help="only this tag, e.g. adult_r05")
    p.add_argument("--vae-epochs", type=int, default=90)
    p.add_argument("--gan-epochs", type=int, default=30)
    p.add_argument("--ddpm-epochs", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=None, help="smoke-test cap per epoch loop")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)
    cfg = TrainConfig(vae_epochs=args.vae_epochs, gan_epochs=args.gan_epochs,
                      ddpm_epochs=args.ddpm_epochs, max_steps=args.max_steps)

    for dataset, ratio, seeds in SETTINGS:
        tag = tag_for(dataset, ratio)
        if args.setting and tag != args.setting:
            continue
        use_seeds = args.seeds if args.seeds is not None else seeds
        data = prepare_dataset(dataset, ratio, seed=0)
        spec = DATASET_SPECS[dataset]
        target = spec["target"]
        num_cols, cat_cols = spec["num_cols"], spec["cat_cols"]
        val = data.val
        real_mask = val.enc.y == 1
        real_df = val.df[real_mask].reset_index(drop=True)
        real_enc = np.hstack([val.enc.num[real_mask], val.enc.cat[real_mask]])
        n_all = needed_synthetic_count(data)
        n_pool = 2 * n_all
        log.info("[%s] n_train=%d n_pos=%d needed_all=%d pool=%d", tag,
                 len(data.train.enc.y), int((data.train.enc.y == 1).sum()), n_all, n_pool)

        for seed in use_seeds:
            for model in args.models:
                run_dir = RUN_DIR / tag / f"{model}_s{seed}"
                if (run_dir / "samples_pool.npz").exists() and not args.force:
                    log.info("skip existing %s", run_dir)
                    continue
                t0 = time.time()
                train_generator(model, data, seed, cfg, device, run_dir)
                loaded, cfg_used = load_generator(model, run_dir, data, device)
                df_all, enc_all = sample_synthetic(model, loaded, data, n_all, seed, device, cfg_used)
                df_pool, enc_pool = sample_synthetic(model, loaded, data, n_pool, seed + 1, device, cfg_used)
                np.savez_compressed(run_dir / "samples_all.npz",
                                    num=enc_all.num, cat=enc_all.cat, y=enc_all.y)
                np.savez_compressed(run_dir / "samples_pool.npz",
                                    num=enc_pool.num, cat=enc_pool.cat, y=enc_pool.y)
                df_all.to_csv(run_dir / "samples_all.csv", index=False)
                df_pool.to_csv(run_dir / "samples_pool.csv", index=False)
                fid = fidelity_report(real_df, df_all, real_enc,
                                      np.hstack([enc_all.num, enc_all.cat]),
                                      num_cols, cat_cols, seed=seed)
                save_json({"model": model, "seed": seed, "dataset": dataset,
                           "ratio": data.ratio, "n_syn": n_all, **fid},
                          run_dir / "fidelity.json")
                log.info("[%s] %s seed=%d done in %.1fs | KS=%.3f TV=%.3f MMD=%.4f DCR=%.2f",
                         tag, model, seed, time.time() - t0, fid["ks_mean"], fid["tv_mean"],
                         fid["mmd"], fid["dcr"])


if __name__ == "__main__":
    main()
