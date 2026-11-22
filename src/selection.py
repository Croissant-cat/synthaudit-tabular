"""Synthetic-sample selection.

The proposed selector (DRS, Density-Ratio Selection) re-weights generated
minority samples by an estimate of p_real(x) / p_model(x), so that the retained
set is closer to the real conditional distribution instead of inheriting the
generator's density bias (typically over-sampling dense regions and under-
sampling rare modes).  Density ratios are estimated with k-NN radii in a
10-dimensional PCA space, which is far more stable than k-NN density estimation
in the full ~90-dimensional one-hot space.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def density_ratio_weights(pool: np.ndarray, real: np.ndarray, k: int = 10,
                          n_pca: int = 10, clip: float = 3.0, seed: int = 0) -> np.ndarray:
    """Return normalised importance weights w_i ∝ p_real(x_i)/p_model(x_i)."""
    pool = np.asarray(pool, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)
    Z = np.vstack([pool, real])
    n_comp = int(min(n_pca, Z.shape[1], len(Z) - 1))
    pca = PCA(n_components=n_comp, random_state=seed).fit(Z)
    P = pca.transform(pool)
    R = pca.transform(real)
    k_eff = int(min(k, len(P) - 1, len(R) - 1))
    if k_eff < 1:
        return np.ones(len(pool)) / len(pool)
    d_model = NearestNeighbors(n_neighbors=k_eff + 1).fit(P).kneighbors(P, return_distance=True)[0][:, k_eff]
    d_real = NearestNeighbors(n_neighbors=k_eff).fit(R).kneighbors(P, return_distance=True)[0][:, k_eff - 1]
    eps = 1e-8
    logw = n_comp * (np.log(d_model + eps) - np.log(d_real + eps))
    logw = np.clip(logw, -clip, clip)
    w = np.exp(logw - logw.max())
    s = w.sum()
    return w / s if s > 0 else np.ones(len(pool)) / len(pool)


def select_drs(pool: np.ndarray, real: np.ndarray, n_select: int, seed: int = 0,
               k: int = 10, clip: float = 3.0) -> np.ndarray:
    pool = np.asarray(pool)
    n_select = min(n_select, len(pool))
    w = density_ratio_weights(pool, real, k=k, clip=clip, seed=seed)
    rng = np.random.default_rng(seed + 31)
    w = np.clip(w, 1e-9, None)
    w = w / w.sum()
    idx = rng.choice(len(pool), size=n_select, replace=False, p=w)
    return idx


def select_random(n_pool: int, n_select: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed + 53)
    return rng.choice(n_pool, size=min(n_select, n_pool), replace=False)


def confidence_filter(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    keep = np.flatnonzero(probs >= threshold)
    if len(keep) == 0:
        return np.arange(len(probs))
    return keep
