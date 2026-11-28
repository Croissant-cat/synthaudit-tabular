"""Convert AUC gains into 'equivalent real-label fraction'.

For a model trained with budget b real labels plus synthetic data, we find the
budget b* at which the *no-augmentation* model reaches the same AUC.  b* - b is
the apparent label saving; comparing the naive vs leakage-controlled protocol
separates real savings from label-access effects.
"""
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


def add_delta(df):
    base = df[df.frac == 0].set_index(["seed", "model", "budget"]).auc
    df = df.copy()
    df["delta"] = [r.auc - base.loc[(r.seed, r.model, r.budget)] for _, r in df.iterrows()]
    return df


def equivalent_labels(auc, curve):
    """curve: DataFrame with budget, auc for one seed/model (no augmentation)."""
    c = curve.sort_values("auc")
    xs, ys = c.auc.to_numpy(), c.budget.to_numpy()
    if auc <= xs[0]:
        return 0.0
    if auc >= xs[-1]:
        return 1.0
    return float(np.interp(auc, xs, ys))


def main():
    naive = add_delta(pd.read_csv(TABLE_DIR / "table_label_budget.csv"))
    ctrl = add_delta(pd.read_csv(TABLE_DIR / "table_label_budget_controlled.csv"))
    rows = []
    for proto, df in [("naive", naive), ("controlled", ctrl)]:
        for (model, budget, seed), g in df.groupby(["model", "budget", "seed"]):
            curve = df[(df.model == model) & (df.seed == seed) & (df.frac == 0)][["budget", "auc"]]
            aug = g[g.frac == 1.0].auc
            if aug.empty:
                continue
            eq = equivalent_labels(float(aug.iloc[0]), curve)
            rows.append({"protocol": proto, "model": model, "budget": budget, "seed": seed,
                         "auc": float(aug.iloc[0]), "eq_labels": eq, "saved": eq - budget})
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "table_label_equivalence.csv", index=False)

    agg = out.groupby(["protocol", "model", "budget"]).agg(
        auc=("auc", "mean"), eq=("eq_labels", "mean"), eq_std=("eq_labels", "std"),
        saved=("saved", "mean")).reset_index()
    agg.to_csv(TABLE_DIR / "table_label_equivalence_agg.csv", index=False)
    print(agg.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot([0, 1], [0, 1], color="#888888", ls="--", lw=0.9, label="no saving (y = x)")
    for proto, color, marker in [("naive", "#DD8452", "o"), ("controlled", "#4C72B0", "s")]:
        sub = agg[(agg.protocol == proto) & (agg.model == "ddpm")].sort_values("budget")
        ax.errorbar(sub["budget"], sub["eq"], yerr=sub["eq_std"], marker=marker, capsize=3,
                    color=color, label=f"DDPM {proto}")
        for _, r in sub.iterrows():
            ax.annotate(f"{r['budget']*100:.0f}%→{r['eq']*100:.0f}%", (r["budget"], r["eq"]),
                        textcoords="offset points", xytext=(4, -10), fontsize=6.5, color=color)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("real minority labels used")
    ax.set_ylabel("equivalent real labels (AUC-matched)")
    ax.set_title("Apparent vs true label savings (DDPM, Adult 5%)")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig23_label_equivalence.png", bbox_inches="tight")
    plt.close(fig)
    log.info("saved label-equivalence table/figure")


if __name__ == "__main__":
    main()
