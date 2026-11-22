"""Downstream utility evaluation and synthetic-set construction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import Encoding, PreparedData
from .metrics import best_f1_threshold, classification_metrics
from .selection import select_drs, select_random


@dataclass
class Bundle:
    model: str
    seed: int
    all_enc: Encoding
    pool_enc: Encoding
    all_df: pd.DataFrame
    pool_df: pd.DataFrame


def load_bundle(run_dir: Path, model: str, seed: int) -> Bundle:
    run_dir = Path(run_dir)
    a = np.load(run_dir / "samples_all.npz")
    p = np.load(run_dir / "samples_pool.npz")
    all_enc = Encoding(num=a["num"], cat=a["cat"], y=a["y"])
    pool_enc = Encoding(num=p["num"], cat=p["cat"], y=p["y"])
    all_df = pd.read_csv(run_dir / "samples_all.csv")
    pool_df = pd.read_csv(run_dir / "samples_pool.csv")
    return Bundle(model=model, seed=seed, all_enc=all_enc, pool_enc=pool_enc,
                  all_df=all_df, pool_df=pool_df)


def _subset(enc: Encoding, idx: np.ndarray) -> Encoding:
    return Encoding(num=enc.num[idx], cat=enc.cat[idx], y=enc.y[idx])


def smote_synthetic(enc: Encoding, n_syn: int, k: int = 5, seed: int = 0) -> Encoding:
    """SMOTE on the encoded representation (numeric interpolation + neighbour
    category copy), a standard mixed-type baseline."""
    rng = np.random.default_rng(seed + 91)
    X = np.hstack([enc.num, enc.cat]).astype(np.float64)
    n = len(X)
    if n < k + 1:
        idx = rng.choice(n, size=n_syn, replace=True)
        return _subset(enc, idx)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, nbr = nn.kneighbors(X)
    base = rng.integers(0, n, size=n_syn)
    lam = rng.random((n_syn, 1))
    nb = nbr[base, rng.integers(1, k + 1, size=n_syn)]
    num = enc.num[base] + lam * (enc.num[nb] - enc.num[base])
    choose = (rng.random((n_syn, 1)) < 0.5)
    cat = np.where(choose, enc.cat[base], enc.cat[nb])
    return Encoding(num=num.astype(np.float32), cat=cat.astype(np.float32),
                    y=np.ones(n_syn, dtype=np.int64))


def needed_synthetic_count(data: PreparedData, target_prevalence: float = 0.20) -> int:
    n_real = int((data.train.enc.y == 1).sum())
    n_neg = int((data.train.enc.y == 0).sum())
    return max(int(round(target_prevalence / (1 - target_prevalence) * n_neg)) - n_real, 100)


def make_synthetic(strategy: str, data: PreparedData, bundle: Bundle | None,
                   seed: int) -> Encoding:
    n_real = int((data.train.enc.y == 1).sum())
    real_idx = np.flatnonzero(data.train.enc.y == 1)
    n_needed = needed_synthetic_count(data)

    if strategy == "none":
        return Encoding(num=np.zeros((0, data.preprocessor.n_num), dtype=np.float32),
                        cat=np.zeros((0, data.preprocessor.n_cat), dtype=np.float32),
                        y=np.zeros(0, dtype=np.int64))
    if strategy == "ros":
        rng = np.random.default_rng(seed + 7)
        idx = rng.choice(real_idx, size=n_needed, replace=True)
        return _subset(data.train.enc, idx)
    if strategy == "smote":
        return smote_synthetic(_subset(data.train.enc, real_idx), n_needed, seed=seed)

    if bundle is None:
        raise ValueError(f"strategy {strategy} needs a generated bundle")
    model, mode = strategy.rsplit("_", 1)
    if model != bundle.model:
        raise ValueError(f"bundle/model mismatch: {strategy} vs {bundle.model}")
    if mode == "all":
        return bundle.all_enc
    if mode == "rand":
        idx = select_random(len(bundle.pool_enc.y), n_needed, seed=seed)
        return _subset(bundle.pool_enc, idx)
    if mode == "drs":
        pool = np.hstack([bundle.pool_enc.num, bundle.pool_enc.cat])
        real = np.hstack([data.train.enc.num[real_idx], data.train.enc.cat[real_idx]])
        idx = select_drs(pool, real, n_needed, seed=seed)
        return _subset(bundle.pool_enc, idx)
    raise ValueError(strategy)


def make_classifier(name: str, seed: int):
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, class_weight="balanced",
            early_stopping=False, random_state=seed,
        )
    if name == "lr":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, random_state=seed),
        )
    raise ValueError(name)


def fit_evaluate(data: PreparedData, synth: Encoding, clf_name: str, seed: int) -> dict:
    X_base = np.hstack([data.train.enc.num, data.train.enc.cat])
    y_tr = data.train.enc.y
    if len(synth.y):
        X_tr = np.vstack([X_base, np.hstack([synth.num, synth.cat])])
        y_tr = np.concatenate([y_tr, synth.y])
    else:
        X_tr = X_base
    X_val = np.hstack([data.val.enc.num, data.val.enc.cat])
    X_test = np.hstack([data.test.enc.num, data.test.enc.cat])
    clf = make_classifier(clf_name, seed)
    clf.fit(X_tr, y_tr)
    p_val = clf.predict_proba(X_val)[:, 1]
    p_test = clf.predict_proba(X_test)[:, 1]
    thr, f1_val = best_f1_threshold(data.val.enc.y, p_val)
    m = classification_metrics(data.test.enc.y, p_test, thr)
    m.update({"threshold": thr, "f1_val": f1_val, "n_train": int(len(y_tr)),
              "n_syn": int(len(synth.y))})
    return m


def fit_tstr(data: PreparedData, synth: Encoding, clf_name: str, seed: int) -> dict:
    """Train on synthetic minority + real majority only (no real minority)."""
    maj = data.train.enc.y == 0
    X_tr = np.vstack([
        np.hstack([data.train.enc.num[maj], data.train.enc.cat[maj]]),
        np.hstack([synth.num, synth.cat]),
    ])
    y_tr = np.concatenate([np.zeros(int(maj.sum()), dtype=np.int64), synth.y])
    X_val = np.hstack([data.val.enc.num, data.val.enc.cat])
    X_test = np.hstack([data.test.enc.num, data.test.enc.cat])
    clf = make_classifier(clf_name, seed)
    clf.fit(X_tr, y_tr)
    p_val = clf.predict_proba(X_val)[:, 1]
    p_test = clf.predict_proba(X_test)[:, 1]
    thr, f1_val = best_f1_threshold(data.val.enc.y, p_val)
    m = classification_metrics(data.test.enc.y, p_test, thr)
    m.update({"threshold": thr, "f1_val": f1_val, "n_train": int(len(y_tr)), "n_syn": int(len(synth.y))})
    return m


def evaluate_setting(data: PreparedData, bundles: dict[str, Bundle], seeds: list[int]) -> pd.DataFrame:
    rows = []
    clfs = ["hgb", "lr"]
    for seed in seeds:
        base_strats = ["none", "ros", "smote"]
        for strat in base_strats:
            synth = make_synthetic(strat, data, None, seed)
            for clf in clfs:
                m = fit_evaluate(data, synth, clf, seed)
                rows.append({"dataset": data.name, "ratio": data.ratio, "seed": seed,
                             "model": "none", "strategy": strat, "classifier": clf, **m})
        for model, bundle in bundles.items():
            for mode in ["all", "rand", "drs"]:
                strat = f"{model}_{mode}"
                synth = make_synthetic(strat, data, bundle, seed)
                for clf in clfs:
                    m = fit_evaluate(data, synth, clf, seed)
                    rows.append({"dataset": data.name, "ratio": data.ratio, "seed": seed,
                                 "model": model, "strategy": strat, "classifier": clf, **m})
    return pd.DataFrame(rows)
