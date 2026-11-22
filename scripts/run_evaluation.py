"""Downstream utility experiments: train a classifier on real + synthetic data."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import prepare_dataset  # noqa: E402
from src.evaluate import evaluate_setting, load_bundle  # noqa: E402
from src.utils import METRIC_DIR, RUN_DIR, get_logger  # noqa: E402

log = get_logger()

SETTINGS = [
    ("adult", 0.05, list(range(5))),
    ("adult", 0.10, [0, 1]),
    ("bank", None, [0, 1, 2]),
]


def tag_for(dataset: str, ratio: float | None) -> str:
    return f"{dataset}_nat" if ratio is None else f"{dataset}_r{int(round(ratio * 100)):02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    args = ap.parse_args()
    for dataset, ratio, seeds in SETTINGS:
        tag = tag_for(dataset, ratio)
        if args.setting and tag != args.setting:
            continue
        use_seeds = args.seeds if args.seeds is not None else seeds
        data = prepare_dataset(dataset, ratio, seed=0)
        bundles = {}
        for s in use_seeds:
            for model in ("vae", "gan", "ddpm"):
                rd = RUN_DIR / tag / f"{model}_s{s}"
                if not (rd / "samples_pool.npz").exists():
                    log.warning("missing %s", rd)
                    continue
                bundles[(s, model)] = load_bundle(rd, model, s)
        if not bundles:
            continue
        rows = []
        for s in use_seeds:
            seed_bundles = {m: bundles[(s, m)] for m in ("vae", "gan", "ddpm") if (s, m) in bundles}
            if not seed_bundles:
                continue
            df = evaluate_setting(data, seed_bundles, [s])
            rows.append(df)
            log.info("[%s] seed %d done (%d rows)", tag, s, len(df))
        out = Path(METRIC_DIR) / f"utility_{tag}.csv"
        import pandas as pd
        pd.concat(rows, ignore_index=True).to_csv(out, index=False)
        log.info("saved %s", out)


if __name__ == "__main__":
    main()
