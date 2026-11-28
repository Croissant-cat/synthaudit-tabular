"""Leakage-controlled label-budget evaluation.

Generators in results/runs/adult_r05_b{10,25,50} were trained only on the
minority labels available at that budget, so the downstream comparison cannot
benefit from the generator having seen withheld labels.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import prepare_dataset  # noqa: E402
from src.evaluate import load_bundle, make_classifier  # noqa: E402
from src.metrics import best_f1_threshold, classification_metrics  # noqa: E402
from src.utils import RUN_DIR, TABLE_DIR, get_logger  # noqa: E402

log = get_logger()
BUDGETS = {0.10: ["ddpm"], 0.25: ["vae", "gan", "ddpm"], 0.50: ["ddpm"]}
FRACS = [0.0, 0.125, 0.25, 0.50, 1.00]


def main():
    data = prepare_dataset("adult", 0.05, seed=0)
    maj = data.train.enc.y == 0
    pos = np.flatnonzero(data.train.enc.y == 1)
    base_X = np.hstack([data.train.enc.num[maj], data.train.enc.cat[maj]])
    base_y = data.train.enc.y[maj]
    Xv = np.hstack([data.val.enc.num, data.val.enc.cat])
    Xt = np.hstack([data.test.enc.num, data.test.enc.cat])
    rows = []
    for budget, models in BUDGETS.items():
        tag = f"adult_r05_b{int(round(budget * 100)):02d}"
        for seed in range(5):
            perm = np.random.default_rng(seed).permutation(pos)[: max(int(round(budget * len(pos))), 20)]
            Xr = np.hstack([data.train.enc.num[perm], data.train.enc.cat[perm]])
            for model in models:
                rd = RUN_DIR / tag / f"{model}_s{seed}"
                if not (rd / "samples_all.npz").exists():
                    continue
                b = load_bundle(rd, model, seed)
                n_all = len(b.all_enc.y)
                for frac in FRACS:
                    n = int(round(frac * n_all))
                    parts = [base_X, Xr]
                    labels = [base_y, np.ones(len(perm))]
                    if n:
                        parts.append(np.hstack([b.all_enc.num[:n], b.all_enc.cat[:n]]))
                        labels.append(np.ones(n))
                    clf = make_classifier("hgb", seed).fit(np.vstack(parts), np.concatenate(labels))
                    pv = clf.predict_proba(Xv)[:, 1]
                    pt = clf.predict_proba(Xt)[:, 1]
                    thr, _ = best_f1_threshold(data.val.enc.y, pv)
                    m = classification_metrics(data.test.enc.y, pt, thr)
                    rows.append({"budget": budget, "seed": seed, "model": model,
                                 "frac": frac, "n_real": len(perm), "n_syn": n, **m})
        log.info("budget %.2f done", budget)
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "table_label_budget_controlled.csv", index=False)
    log.info("saved %d rows", len(out))


if __name__ == "__main__":
    main()
