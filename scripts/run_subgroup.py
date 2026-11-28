"""Subgroup performance of the low-budget augmentation regime (Adult 5%).

Compares no augmentation with DDPM augmentation when only 25% of real minority
labels are available, by sex, race group and age band.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import prepare_dataset  # noqa: E402
from src.evaluate import load_bundle, make_classifier  # noqa: E402
from src.metrics import best_f1_threshold  # noqa: E402
from src.utils import RUN_DIR, TABLE_DIR, get_logger  # noqa: E402

log = get_logger()


def main():
    data = prepare_dataset("adult", 0.05, seed=0)
    maj = data.train.enc.y == 0
    pos = np.flatnonzero(data.train.enc.y == 1)
    base_X = np.hstack([data.train.enc.num[maj], data.train.enc.cat[maj]])
    base_y = data.train.enc.y[maj]
    Xv = np.hstack([data.val.enc.num, data.val.enc.cat])
    Xt = np.hstack([data.test.enc.num, data.test.enc.cat])
    test = data.test.df.copy()
    test["age_bin"] = pd.cut(test["age"], [17, 35, 50, 65, 100], labels=["18-35", "36-50", "51-65", "65+"])
    test["race_group"] = np.where(test["race"] == "White", "White", "Non-White")
    rows = []
    for seed in range(5):
        perm = np.random.default_rng(seed).permutation(pos)[:293]
        Xr = np.hstack([data.train.enc.num[perm], data.train.enc.cat[perm]])
        b = load_bundle(RUN_DIR / "adult_r05" / f"ddpm_s{seed}", "ddpm", seed)
        syn_X = np.hstack([b.all_enc.num[:1100], b.all_enc.cat[:1100]])
        for strat, extra in [("none", None), ("ddpm_1100", syn_X)]:
            parts = [base_X, Xr] + ([extra] if extra is not None else [])
            y = np.concatenate([base_y, np.ones(len(perm) + (1100 if extra is not None else 0))])
            clf = make_classifier("hgb", seed).fit(np.vstack(parts), y)
            pv = clf.predict_proba(Xv)[:, 1]
            pt = clf.predict_proba(Xt)[:, 1]
            thr, _ = best_f1_threshold(data.val.enc.y, pv)
            pred = (pt >= thr).astype(int)
            for col in ["sex", "race_group", "age_bin"]:
                for val, idx in test.groupby(col, observed=True).groups.items():
                    idx = np.asarray(list(idx))
                    yt = test.loc[idx, "income"].to_numpy()
                    if len(np.unique(yt)) < 2:
                        auc = np.nan
                    else:
                        auc = roc_auc_score(yt, pt[idx])
                    rows.append({"seed": seed, "strategy": strat, "group": col, "value": str(val),
                                 "n": len(idx), "n_pos": int(yt.sum()), "auc": auc,
                                 "recall": float(pred[idx][yt == 1].mean()) if (yt == 1).any() else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "table_subgroup_perf.csv", index=False)
    print(out.groupby(["strategy", "group", "value"]).auc.mean().round(4).to_string())
    log.info("saved subgroup table")


if __name__ == "__main__":
    main()
