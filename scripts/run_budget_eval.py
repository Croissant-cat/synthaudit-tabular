"""Generic label-budget evaluation for Adult or Bank.

Protocols
---------
naive      : generator trained on the full labels (results/runs/<base_tag>)
controlled : generator retrained on the same budget (results/runs/<base_tag>_bXX)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import DATASET_SPECS, prepare_dataset  # noqa: E402
from src.evaluate import load_bundle, make_classifier  # noqa: E402
from src.metrics import best_f1_threshold, classification_metrics  # noqa: E402
from src.utils import RUN_DIR, TABLE_DIR, get_logger  # noqa: E402

log = get_logger()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bank", choices=["adult", "bank"])
    ap.add_argument("--natural", action="store_true")
    ap.add_argument("--ratio", type=float, default=0.05)
    ap.add_argument("--budgets", nargs="+", type=float, default=[0.10, 0.25, 0.50, 1.00])
    ap.add_argument("--models", nargs="+", default=["ddpm"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--controlled", action="store_true")
    ap.add_argument("--out", default="table_bank_budget.csv")
    args = ap.parse_args()
    ratio = None if args.natural else args.ratio
    base_tag = f"{args.dataset}_nat" if ratio is None else f"{args.dataset}_r{int(round(ratio * 100)):02d}"
    target = DATASET_SPECS[args.dataset]["target"]
    data = prepare_dataset(args.dataset, ratio, seed=0)
    maj = data.train.enc.y == 0
    pos = np.flatnonzero(data.train.enc.y == 1)
    base_X = np.hstack([data.train.enc.num[maj], data.train.enc.cat[maj]])
    base_y = data.train.enc.y[maj]
    Xv = np.hstack([data.val.enc.num, data.val.enc.cat])
    Xt = np.hstack([data.test.enc.num, data.test.enc.cat])

    rows = []
    for budget in args.budgets:
        for seed in args.seeds:
            perm = np.random.default_rng(seed).permutation(pos)[: max(int(round(budget * len(pos))), 20)]
            Xr = np.hstack([data.train.enc.num[perm], data.train.enc.cat[perm]])
            for model in args.models:
                tag = f"{base_tag}_b{int(round(budget * 100)):02d}" if args.controlled else base_tag
                rd = RUN_DIR / tag / f"{model}_s{seed}"
                if not (rd / "samples_all.npz").exists():
                    continue
                b = load_bundle(rd, model, seed)
                n_all = len(b.all_enc.y)
                for frac in [0.0, 0.125, 0.25, 0.50, 1.00]:
                    n = int(round(frac * n_all))
                    parts, labels = [base_X, Xr], [base_y, np.ones(len(perm))]
                    if n:
                        parts.append(np.hstack([b.all_enc.num[:n], b.all_enc.cat[:n]]))
                        labels.append(np.ones(n))
                    clf = make_classifier("hgb", seed).fit(np.vstack(parts), np.concatenate(labels))
                    pv = clf.predict_proba(Xv)[:, 1]
                    pt = clf.predict_proba(Xt)[:, 1]
                    thr, _ = best_f1_threshold(data.val.enc.y, pv)
                    m = classification_metrics(data.test.enc.y, pt, thr)
                    rows.append({"dataset": args.dataset, "protocol": "controlled" if args.controlled else "naive",
                                 "budget": budget, "seed": seed, "model": model,
                                 "n_real": len(perm), "frac": frac, "n_syn": n, **m})
        log.info("budget %.2f done", budget)
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / args.out, index=False)
    log.info("saved %s (%d rows)", TABLE_DIR / args.out, len(out))


if __name__ == "__main__":
    main()
