"""Distribution-level and predictive metrics.

Fidelity  : per-feature KS (numeric) / total-variation (categorical), correlation
            error, RBF-MMD in the shared encoded space.
Diversity : Vendi score ratio, category-combination coverage.
Privacy   : distance to closest real record (DCR), normalised by the real 5-NN
            distance.
Utility   : ROC-AUC, PR-AUC, F1 at a validation-tuned threshold, balanced
            accuracy, Brier score and ECE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors


# --------------------------------------------------------------------------- #
# distances / kernels
# --------------------------------------------------------------------------- #

def pairwise_sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d2, 0.0)


def median_bandwidth(X: np.ndarray, Y: np.ndarray, max_n: int = 1500, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    Xs = X if len(X) <= max_n else X[rng.choice(len(X), max_n, replace=False)]
    Ys = Y if len(Y) <= max_n else Y[rng.choice(len(Y), max_n, replace=False)]
    Z = np.vstack([Xs, Ys])
    d2 = pairwise_sq_dists(Z, Z)
    iu = np.triu_indices(len(Z), k=1)
    med = np.median(d2[iu])
    if not np.isfinite(med) or med <= 0:
        return 1.0
    return float(np.sqrt(med))


def mmd_rbf(X: np.ndarray, Y: np.ndarray, bandwidth: float | None = None) -> float:
    """Unbiased squared MMD with an RBF kernel."""
    if len(X) < 3 or len(Y) < 3:
        return float("nan")
    bw = bandwidth or median_bandwidth(X, Y)
    gamma = 1.0 / (2.0 * bw * bw)
    Kxx = np.exp(-gamma * pairwise_sq_dists(X, X))
    Kyy = np.exp(-gamma * pairwise_sq_dists(Y, Y))
    Kxy = np.exp(-gamma * pairwise_sq_dists(X, Y))
    n, m = len(X), len(Y)
    term_x = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
    term_y = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    mmd2 = term_x + term_y - 2.0 * Kxy.mean()
    return float(np.sqrt(max(mmd2, 0.0)))


def vendi_score(X: np.ndarray, max_n: int = 2000, seed: int = 0) -> float:
    """Vendi score = exp(entropy of normalised RBF kernel eigenvalues)."""
    rng = np.random.default_rng(seed)
    if len(X) > max_n:
        X = X[rng.choice(len(X), max_n, replace=False)]
    bw = median_bandwidth(X, X)
    gamma = 1.0 / (2.0 * bw * bw)
    K = np.exp(-gamma * pairwise_sq_dists(X, X))
    K = K / len(K)
    eig = np.linalg.eigvalsh(K)
    eig = np.clip(eig, 1e-12, None)
    eig = eig / eig.sum()
    return float(np.exp(-(eig * np.log(eig)).sum()))


# --------------------------------------------------------------------------- #
# fidelity / diversity / privacy
# --------------------------------------------------------------------------- #

def categorical_tv(real: pd.Series, syn: pd.Series) -> float:
    pr = real.astype(str).value_counts(normalize=True)
    ps = syn.astype(str).value_counts(normalize=True)
    idx = pr.index.union(ps.index)
    return float(0.5 * np.abs(pr.reindex(idx, fill_value=0) - ps.reindex(idx, fill_value=0)).sum())


def correlation_error(real: pd.DataFrame, syn: pd.DataFrame, num_cols: list[str]) -> float:
    if len(num_cols) < 2:
        return float("nan")
    cr = real[num_cols].corr().to_numpy()
    cs = syn[num_cols].corr().to_numpy()
    cr = np.nan_to_num(cr)
    cs = np.nan_to_num(cs)
    denom = np.linalg.norm(cr)
    return float(np.linalg.norm(cs - cr) / (denom + 1e-12))


def diversity_report(real_df: pd.DataFrame, syn_df: pd.DataFrame, cat_cols: list[str]) -> dict:
    real_combo = real_df[cat_cols].astype(str).agg("|".join, axis=1)
    syn_combo = syn_df[cat_cols].astype(str).agg("|".join, axis=1)
    return {
        "unique_cat_ratio": float(syn_combo.nunique() / max(len(syn_combo), 1)),
        "unique_cat_ratio_real": float(real_combo.nunique() / max(len(real_combo), 1)),
        "cat_combo_overlap": float(len(set(syn_combo) & set(real_combo)) / max(len(set(real_combo)), 1)),
    }


def coverage_rate(real: np.ndarray, syn: np.ndarray, k: int = 5) -> float:
    """Fraction of real points whose nearest synthetic point is closer than the
    distance to their k-th real neighbour (mode coverage)."""
    if len(real) <= k + 1 or len(syn) == 0:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=k + 1).fit(real)
    d_real, _ = nn.kneighbors(real)
    thresh = d_real[:, k]
    d_syn = NearestNeighbors(n_neighbors=1).fit(syn).kneighbors(real, return_distance=True)[0][:, 0]
    return float((d_syn <= thresh).mean())


def dcr_ratio(real: np.ndarray, syn: np.ndarray, k: int = 5) -> float:
    """Distance to closest real record, normalised by the real k-NN distance."""
    if len(real) <= k + 1 or len(syn) == 0:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=k + 1).fit(real)
    d_real, _ = nn.kneighbors(real)
    real_scale = float(d_real[:, 1:].mean())
    d_syn = NearestNeighbors(n_neighbors=1).fit(real).kneighbors(syn, return_distance=True)[0][:, 0]
    return float(d_syn.mean() / (real_scale + 1e-12))


def fidelity_report(real_df: pd.DataFrame, syn_df: pd.DataFrame, real_enc: np.ndarray,
                    syn_enc: np.ndarray, num_cols: list[str], cat_cols: list[str],
                    seed: int = 0) -> dict:
    out: dict[str, float] = {}
    ks = [ks_2samp(real_df[c].to_numpy(float), syn_df[c].to_numpy(float)).statistic for c in num_cols]
    out["ks_mean"] = float(np.mean(ks))
    out["ks_max"] = float(np.max(ks))
    tvs = [categorical_tv(real_df[c], syn_df[c]) for c in cat_cols]
    out["tv_mean"] = float(np.mean(tvs))
    out["tv_max"] = float(np.max(tvs))
    out["corr_err"] = correlation_error(real_df, syn_df, num_cols)
    for c in num_cols:
        sd_r = max(float(real_df[c].std()), 1e-9)
        out[f"mean_shift__{c}"] = float(abs(syn_df[c].mean() - real_df[c].mean()) / sd_r)
        out[f"std_ratio__{c}"] = float(syn_df[c].std() / sd_r)
    out["std_ratio_mean"] = float(np.mean([out[f"std_ratio__{c}"] for c in num_cols]))
    out["mean_shift_mean"] = float(np.mean([out[f"mean_shift__{c}"] for c in num_cols]))
    out["mmd"] = mmd_rbf(real_enc, syn_enc)
    out["coverage"] = coverage_rate(real_enc, syn_enc)
    out["vendi_ratio"] = float(vendi_score(syn_enc, seed=seed) / max(vendi_score(real_enc, seed=seed), 1e-9))
    out["dcr"] = dcr_ratio(real_enc, syn_enc)
    out.update(diversity_report(real_df, syn_df, cat_cols))
    return out


# --------------------------------------------------------------------------- #
# predictive utility
# --------------------------------------------------------------------------- #

def ece(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        conf = p[m].mean()
        acc = y_true[m].mean()
        total += m.mean() * abs(acc - conf)
    return float(total)


def best_f1_threshold(y_true: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        f = f1_score(y_true, (p >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_t, best_f1 = float(t), float(f)
    return best_t, float(best_f1)


def classification_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    y_pred = (p >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) > 1 else float("nan"),
        "ap": float(average_precision_score(y_true, p)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, p)),
        "ece": ece(y_true, p),
    }
