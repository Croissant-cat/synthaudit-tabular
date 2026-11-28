"""Deep analyses built on the label-budget and audit experiments."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import FIG_DIR, TABLE_DIR, get_logger  # noqa: E402

log = get_logger()
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})
MODEL_COLORS = {"vae": "#8172B3", "gan": "#C44E52", "ddpm": "#55A868"}
MODEL_LABEL = {"vae": "VAE", "gan": "GAN", "ddpm": "DDPM"}


def label_budget():
    df = pd.read_csv(TABLE_DIR / "table_label_budget.csv")
    base = df[df.frac == 0].set_index(["seed", "model", "budget"]).auc
    df["delta"] = [r.auc - base.loc[(r.seed, r.model, r.budget)] for _, r in df.iterrows()]
    agg = df[df.frac > 0].groupby(["model", "budget", "frac"]).delta.agg(["mean", "std"]).reset_index()
    agg.to_csv(TABLE_DIR / "table_label_budget_delta.csv", index=False)

    # recovery of the gap between this budget and full labels
    rows = []
    full = df[(df.frac == 0) & (df.budget == 1.0)].set_index(["seed", "model"]).auc
    for (seed, model, budget), g in df.groupby(["seed", "model", "budget"]):
        if budget == 1.0:
            continue
        none = g[g.frac == 0].auc.iloc[0]
        best = g[g.frac > 0].auc.max()
        gap = full.loc[(seed, model)] - none
        rows.append({"seed": seed, "model": model, "budget": budget, "none": none,
                     "best_aug": best, "gap_to_full": gap, "gain": best - none,
                     "recovery": (best - none) / gap if gap > 0 else np.nan})
    rec = pd.DataFrame(rows)
    rec.to_csv(TABLE_DIR / "table_recovery.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), sharey=True)
    for ax, model in zip(axes, ["vae", "gan", "ddpm"]):
        sub = agg[agg.model == model]
        for b, color in zip([0.10, 0.25, 0.50, 1.00], ["#4C72B0", "#55A868", "#DD8452", "#8172B3"]):
            g = sub[sub.budget == b]
            if g.empty:
                continue
            ax.errorbar(g.frac, g["mean"], yerr=g["std"], marker="o", ms=3.5, capsize=2,
                        color=color, label=f"{int(b * 100)}% labels")
        ax.axhline(0, color="k", ls="--", lw=0.8)
        ax.set_title(MODEL_LABEL[model])
        ax.set_xlabel("synthetic / needed volume")
        ax.set_ylabel("Δ ROC-AUC vs. no augmentation")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Value of synthetic data depends on the real-label budget (Adult 5%)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig19_label_budget.png", bbox_inches="tight")
    plt.close(fig)

    # heatmap for DDPM
    hm = df[(df.model == "ddpm") & (df.frac > 0)].groupby(["budget", "frac"]).delta.mean().unstack()
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    im = ax.imshow(hm.values, cmap="RdBu_r", vmin=-0.02, vmax=0.02, aspect="auto")
    ax.set_xticks(range(len(hm.columns)), [f"{c:g}" for c in hm.columns])
    ax.set_yticks(range(len(hm.index)), [f"{int(b * 100)}%" for b in hm.index])
    ax.set_xlabel("synthetic / needed volume"); ax.set_ylabel("real minority labels")
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            ax.text(j, i, f"{hm.values[i, j]:+.3f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(hm.values[i, j]) > 0.012 else "black")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("DDPM: ΔAUC vs. no augmentation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig20_label_budget_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    return df, agg, rec


def subgroup():
    df = pd.read_csv(TABLE_DIR / "table_subgroup_perf.csv")
    agg = df.groupby(["group", "value", "strategy"]).agg(auc=("auc", "mean"), std=("auc", "std"),
                                                         n=("n", "mean"), n_pos=("n_pos", "mean")).reset_index()
    agg.to_csv(TABLE_DIR / "table_subgroup_agg.csv", index=False)
    groups = [("sex", ["Male", "Female"]), ("race_group", ["White", "Non-White"]),
              ("age_bin", ["18-35", "36-50", "51-65", "65+"])]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0))
    for ax, (g, vals) in zip(axes, groups):
        x = np.arange(len(vals))
        for k, (strat, label, color) in enumerate([("none", "no augmentation", "#4C72B0"),
                                                   ("ddpm_1100", "DDPM (+1,100)", "#55A868")]):
            sub = agg[(agg.group == g) & (agg.strategy == strat)].set_index("value").reindex(vals)
            ax.bar(x + (k - 0.5) * 0.36, sub.auc, width=0.34, yerr=sub["std"], capsize=2,
                   color=color, label=label)
        ax.set_xticks(x, vals, fontsize=7)
        ax.set_ylim(0.75, 0.95)
        ax.set_title({"sex": "sex", "race_group": "race group", "age_bin": "age band"}[g])
        ax.set_ylabel("ROC-AUC")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Subgroup AUC at 25% real minority labels", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig21_subgroup.png", bbox_inches="tight")
    plt.close(fig)
    return agg


def main():
    lb, agg, rec = label_budget()
    print("\nrecovery (% of gap to full labels closed by best augmentation)")
    print(rec.groupby(["model", "budget"]).recovery.mean().round(3).to_string())
    print("\nlabel-budget deltas")
    print(agg[agg.budget.isin([0.25, 1.0])].round(4).to_string(index=False))
    sg = subgroup()
    print("\nsubgroup AUC")
    print(sg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
