"""Bank replication of the label-leakage audit (DDPM)."""
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
    base = df[df.frac == 0].set_index(["seed", "budget"]).auc
    df = df.copy()
    df["delta"] = [r.auc - base.loc[(r.seed, r.budget)] for _, r in df.iterrows()]
    return df


def main():
    naive = add_delta(pd.read_csv(TABLE_DIR / "table_bank_budget_naive.csv"))
    ctrl = add_delta(pd.read_csv(TABLE_DIR / "table_bank_budget_controlled.csv"))
    rows = []
    for budget in [0.10, 0.25, 0.50]:
        n = naive[(naive.budget == budget) & (naive.frac == 1.0)].delta.mean()
        c = ctrl[(ctrl.budget == budget) & (ctrl.frac == 1.0)].delta.mean()
        rows.append({"dataset": "bank", "budget": budget, "naive": n, "controlled": c, "gap": n - c})
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "table_bank_leakage.csv", index=False)

    # label equivalence for Bank
    eq_rows = []
    for proto, df in [("naive", naive), ("controlled", ctrl)]:
        for (budget, seed), g in df.groupby(["budget", "seed"]):
            curve = df[(df.seed == seed) & (df.frac == 0)][["budget", "auc"]].sort_values("auc")
            aug = g[g.frac == 1.0].auc
            if aug.empty:
                continue
            xs, ys = curve.auc.to_numpy(), curve.budget.to_numpy()
            a = float(aug.iloc[0])
            eq = 0.0 if a <= xs[0] else (1.0 if a >= xs[-1] else float(np.interp(a, xs, ys)))
            eq_rows.append({"protocol": proto, "budget": budget, "seed": seed, "eq": eq})
    eq = pd.DataFrame(eq_rows).groupby(["protocol", "budget"]).eq.mean().reset_index()
    eq.to_csv(TABLE_DIR / "table_bank_label_equivalence.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    x = np.arange(len(out))
    ax.bar(x - 0.18, out.naive, width=0.34, color="#DD8452", label="naive (generator saw all labels)")
    ax.bar(x + 0.18, out.controlled, width=0.34, color="#4C72B0", label="controlled (same labels)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [f"{int(b*100)}% labels" for b in out.budget])
    ax.set_ylabel("Δ ROC-AUC vs. no augmentation")
    ax.set_title("Bank: leakage explains most, but not all, of the apparent gain (DDPM)")
    ax.legend(frameon=False, fontsize=7)
    for i, r in out.iterrows():
        ax.text(i - 0.18, r.naive, f"{r.naive:+.3f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + 0.18, r.controlled, f"{r.controlled:+.3f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig24_bank_leakage.png", bbox_inches="tight")
    plt.close(fig)
    print(out.round(4).to_string(index=False))
    print(eq.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
