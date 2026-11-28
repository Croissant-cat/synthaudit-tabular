"""Label-budget x synthetic-dose experiment (Adult 5%).

For each generator and seed, we vary (a) the fraction of *real* minority labels
available for downstream training and (b) the synthetic volume.  This separates
"synthetic as a label substitute" from "synthetic as an amplifier of existing
labels".
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import prepare_dataset  # noqa: E402
from src.evaluate import load_bundle, make_classifier, needed_synthetic_count  # noqa: E402
from src.metrics import best_f1_threshold, classification_metrics  # noqa: E402
from src.utils import RUN_DIR, TABLE_DIR, get_logger  # noqa: E402

log = get_logger()
BUDGETS = [0.10, 0.25, 0.50, 1.00]
FRACS = [0.0, 0.125, 0.25, 0.50, 1.00]
MODELS = ["vae", "gan", "ddpm"]


def main():
    data = prepare_dataset("adult", 0.05, seed=0)
    maj = data.train.enc.y == 0
    pos = np.flatnonzero(data.train.enc.y == 1)
    n_all = needed_synthetic_count(data)
    base_X = np.hstack([data.train.enc.num[maj], data.train.enc.cat[maj]])
    base_y = data.train.enc.y[maj]
    Xv = np.hstack([data.val.enc.num, data.val.enc.cat])
    Xt = np.hstack([data.test.enc.num, data.test.enc.cat])
    rows = []
    for seed in range(5):
        perm = np.random.default_rng(seed).permutation(pos)
        for model in MODELS:
            b = load_bundle(RUN_DIR / "adult_r05" / f"{model}_s{seed}", model, seed)
            for budget in BUDGETS:
                idx = perm[: max(int(round(budget * len(pos))), 20)]
                Xr = np.hstack([data.train.enc.num[idx], data.train.enc.cat[idx]])
                for frac in FRACS:
                    n = int(round(frac * n_all))
                    parts = [base_X, Xr]
                    labels = [base_y, np.ones(len(idx))]
                    if n:
                        parts.append(np.hstack([b.all_enc.num[:n], b.all_enc.cat[:n]]))
                        labels.append(np.ones(n))
                    clf = make_classifier("hgb", seed).fit(np.vstack(parts), np.concatenate(labels))
                    pv = clf.predict_proba(Xv)[:, 1]
                    pt = clf.predict_proba(Xt)[:, 1]
                    thr, _ = best_f1_threshold(data.val.enc.y, pv)
                    m = classification_metrics(data.test.enc.y, pt, thr)
                    rows.append({"seed": seed, "model": model, "budget": budget,
                                 "n_real": len(idx), "frac": frac, "n_syn": n, **m})
                log.info("seed=%d model=%s budget=%.2f done", seed, model, budget)
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "table_label_budget.csv", index=False)
    log.info("saved %s (%d rows)", TABLE_DIR / "table_label_budget.csv", len(out))


if __name__ == "__main__":
    main()
