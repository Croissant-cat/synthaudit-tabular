"""Exploratory data analysis for the two datasets.

Produces the summary tables and figures used in the data-layer section of the
report.  All numbers are computed from the raw files (before imputation), so the
missing-value analysis is not masked by the modelling pipeline.
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import ADULT_COLS, ADULT_CAT, ADULT_NUM, DATASET_SPECS, load_dataset  # noqa: E402
from src.utils import FIG_DIR, RAW_DIR, TABLE_DIR, get_logger, save_json  # noqa: E402

log = get_logger()
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
})
C0, C1 = "#4C72B0", "#DD8452"


def read_adult_raw() -> pd.DataFrame:
    with zipfile.ZipFile(RAW_DIR / "adult.zip") as z:
        frames = []
        for name in ("adult.data", "adult.test"):
            text = z.read(name).decode("utf-8", errors="replace")
            lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("|")]
            frames.append(pd.read_csv(io.StringIO("\n".join(lines)), header=None, names=ADULT_COLS,
                                      skipinitialspace=True, na_values=["?"]))
    df = pd.concat(frames, ignore_index=True)
    df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")
    df["income"] = (df["income"] == ">50K").astype(int)
    return df


def plot_imbalance(dfs: dict[str, pd.DataFrame], target_map: dict[str, str]):
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.9))
    for ax, (name, df) in zip(axes, dfs.items()):
        y = df[target_map[name]]
        counts = y.value_counts().sort_index()
        bars = ax.bar(["majority (0)", "minority (1)"], counts.values, color=[C0, C1], width=0.55)
        for b, v in zip(bars, counts.values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v / len(y):.1%})",
                    ha="center", va="bottom", fontsize=8)
        ax.set_title(f"{name}: {len(y):,} rows, {y.mean():.1%} positive")
        ax.set_ylim(0, counts.max() * 1.22)
        ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig01_imbalance.png", bbox_inches="tight")
    plt.close(fig)


def plot_missing(raw_adult: pd.DataFrame):
    miss = raw_adult.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(4.8, 2.4))
    bars = ax.barh(miss.index[::-1], (miss / len(raw_adult) * 100)[::-1], color=C1)
    for b, v in zip(bars, (miss / len(raw_adult) * 100)[::-1]):
        ax.text(v + 0.05, b.get_y() + b.get_height() / 2, f"{v:.2f}%", va="center", fontsize=8)
    ax.set_xlabel("missing (%)")
    ax.set_title("Adult: missing values (encoded as '?')")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig02_missing.png", bbox_inches="tight")
    plt.close(fig)


def plot_numeric(df: pd.DataFrame, target: str, name: str):
    num = [c for c in df.columns if c in DATASET_SPECS[name]["num_cols"]]
    # log1p transform heavily skewed capital gains for display only
    show = num[:4]
    fig, axes = plt.subplots(1, len(show), figsize=(3.0 * len(show), 2.6))
    for ax, c in zip(np.atleast_1d(axes), show):
        a = df.loc[df[target] == 0, c].dropna()
        b = df.loc[df[target] == 1, c].dropna()
        bins = np.histogram_bin_edges(pd.concat([a, b]), bins=35)
        ax.hist(a, bins=bins, density=True, alpha=0.55, color=C0, label="majority")
        ax.hist(b, bins=bins, density=True, alpha=0.55, color=C1, label="minority")
        ax.set_title(c)
        ax.set_yticks([])
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{name}: numeric distributions by class", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"fig03_numeric_{name}.png", bbox_inches="tight")
    plt.close(fig)


def plot_corr(df: pd.DataFrame, name: str):
    num = [c for c in DATASET_SPECS[name]["num_cols"] if c in df.columns]
    corr = df[num].corr()
    fig, ax = plt.subplots(figsize=(0.55 * len(num) + 1.6, 0.55 * len(num) + 1.2))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num)), num, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(num)), num, fontsize=7)
    for i in range(len(num)):
        for j in range(len(num)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(corr.iloc[i, j]) > 0.55 else "black")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{name}: numeric correlation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"fig04_corr_{name}.png", bbox_inches="tight")
    plt.close(fig)


def plot_category_rates(df: pd.DataFrame, target: str, col: str, name: str, top: int = 8):
    rates = df.groupby(col)[target].agg(["mean", "count"]).sort_values("count", ascending=False)
    rates = rates.head(top).sort_values("mean")
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    bars = ax.barh(rates.index.astype(str), rates["mean"], color=C1)
    for b, (m, n) in zip(bars, zip(rates["mean"], rates["count"])):
        ax.text(m + 0.005, b.get_y() + b.get_height() / 2, f"n={int(n):,}", va="center", fontsize=7)
    ax.axvline(df[target].mean(), color="k", ls="--", lw=0.8)
    ax.set_xlabel("positive rate")
    ax.set_title(f"{name}: positive rate by {col} (dashed = overall)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"fig05_rate_{name}_{col}.png", bbox_inches="tight")
    plt.close(fig)


def plot_pca(df: pd.DataFrame, target: str, name: str):
    X = df[DATASET_SPECS[name]["num_cols"]].to_numpy(float)
    X = StandardScaler().fit_transform(X)
    xy = PCA(n_components=2, random_state=0).fit_transform(X)
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(xy), size=min(6000, len(xy)), replace=False)
    for cls, color, lab in [(0, C0, "majority"), (1, C1, "minority")]:
        m = (df[target].to_numpy()[idx] == cls)
        ax.scatter(xy[idx][m, 0], xy[idx][m, 1], s=4, alpha=0.45, color=color, label=lab, linewidths=0)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"{name}: PCA of numeric features")
    ax.legend(frameon=False, markerscale=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"fig06_pca_{name}.png", bbox_inches="tight")
    plt.close(fig)


def mi_table(df: pd.DataFrame, target: str, name: str) -> pd.DataFrame:
    cols = [c for c in df.columns if c != target]
    cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    M = np.zeros((len(df), len(cols)), dtype=np.float64)
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        E = enc.fit_transform(df[cat_cols].fillna("Unknown").astype(str))
        for j, c in enumerate(cat_cols):
            M[:, cols.index(c)] = E[:, j]
    for c in num_cols:
        v = pd.to_numeric(df[c], errors="coerce")
        v = v.fillna(v.median())
        M[:, cols.index(c)] = v.to_numpy(float)
    cat_mask = np.array([c in cat_cols for c in cols])
    y = df[target].to_numpy()
    mi = mutual_info_classif(M, y, discrete_features=cat_mask, random_state=0, n_neighbors=5)
    out = pd.DataFrame({"feature": cols, "mi": mi}).sort_values("mi", ascending=False)
    out["dataset"] = name
    return out


def main():
    raw_adult = read_adult_raw()
    bank = load_dataset("bank")
    dfs = {"adult": raw_adult, "bank": bank}
    targets = {"adult": "income", "bank": "y"}

    rows = []
    for name, df in dfs.items():
        y = df[targets[name]]
        spec = DATASET_SPECS[name]
        rows.append({
            "dataset": name, "n": len(df), "n_features": len(spec["num_cols"]) + len(spec["cat_cols"]),
            "n_numeric": len(spec["num_cols"]), "n_categorical": len(spec["cat_cols"]),
            "positive_rate": float(y.mean()), "n_positive": int(y.sum()),
            "missing_cells": int(df.isna().sum().sum()),
            "missing_pct": float(df.isna().sum().sum() / df.size * 100),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLE_DIR / "eda_summary.csv", index=False)
    log.info("summary\n%s", summary.to_string(index=False))

    plot_imbalance(dfs, targets)
    plot_missing(raw_adult)
    for name, df in dfs.items():
        plot_numeric(df, targets[name], name)
        plot_corr(df, name)
        plot_pca(df, targets[name], name)
    plot_category_rates(raw_adult, "income", "education", "adult", top=8)
    plot_category_rates(raw_adult, "income", "occupation", "adult", top=10)
    plot_category_rates(bank, "y", "poutcome", "bank", top=4)
    plot_category_rates(bank, "y", "month", "bank", top=10)

    mi = pd.concat([mi_table(raw_adult, "income", "adult"),
                    mi_table(bank, "y", "bank")], ignore_index=True)
    mi.to_csv(TABLE_DIR / "eda_mutual_info.csv", index=False)

    # numeric class-conditional stats
    stats = []
    for name, df in dfs.items():
        for c in DATASET_SPECS[name]["num_cols"]:
            for cls in (0, 1):
                v = pd.to_numeric(df.loc[df[targets[name]] == cls, c], errors="coerce").dropna()
                stats.append({"dataset": name, "feature": c, "class": cls, "n": len(v),
                              "mean": float(v.mean()), "std": float(v.std()),
                              "median": float(v.median()), "q25": float(v.quantile(.25)),
                              "q75": float(v.quantile(.75))})
    pd.DataFrame(stats).to_csv(TABLE_DIR / "eda_numeric_by_class.csv", index=False)

    # categorical rate table for a few informative columns
    cat_rows = []
    for name, df, cols in [("adult", raw_adult, ["education", "occupation", "marital_status", "sex"]),
                           ("bank", bank, ["poutcome", "month", "job", "housing"])]:
        t = targets[name]
        for c in cols:
            g = df.groupby(c)[t].agg(["mean", "count"])
            for val, r in g.iterrows():
                cat_rows.append({"dataset": name, "feature": c, "value": str(val),
                                 "positive_rate": float(r["mean"]), "n": int(r["count"])})
    pd.DataFrame(cat_rows).to_csv(TABLE_DIR / "eda_category_rates.csv", index=False)

    save_json({
        "n_adult": len(raw_adult), "n_bank": len(bank),
        "adult_positive_rate": float(raw_adult["income"].mean()),
        "bank_positive_rate": float(bank["y"].mean()),
        "adult_missing": raw_adult.isna().sum()[lambda s: s > 0].to_dict(),
        "top_mi_adult": mi[mi.dataset == "adult"].head(5)[["feature", "mi"]].to_dict("records"),
        "top_mi_bank": mi[mi.dataset == "bank"].head(5)[["feature", "mi"]].to_dict("records"),
    }, TABLE_DIR / "eda_stats.json")
    log.info("EDA done -> %s", FIG_DIR)


if __name__ == "__main__":
    main()
