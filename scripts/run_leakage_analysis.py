"""Compare naive vs leakage-controlled label-budget augmentation."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import FIG_DIR, RUN_DIR, TABLE_DIR, get_logger  # noqa: E402

log = get_logger()
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})
COL = {"vae": "#8172B3", "gan": "#C44E52", "ddpm": "#55A868"}


def add_delta(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df.frac == 0].set_index(["seed", "model", "budget"]).auc
    df = df.copy()
    df["delta"] = [r.auc - base.loc[(r.seed, r.model, r.budget)] for _, r in df.iterrows()]
    return df


def fidelity(tag: str, model: str = "ddpm") -> float:
    vals = []
    for p in glob.glob(str(RUN_DIR / tag / f"{model}_s*" / "fidelity.json")):
        vals.append(json.load(open(p, encoding="utf-8"))["ks_mean"])
    return float(np.mean(vals)) if vals else np.nan


def main():
    naive = add_delta(pd.read_csv(TABLE_DIR / "table_label_budget.csv"))
    ctrl = add_delta(pd.read_csv(TABLE_DIR / "table_label_budget_controlled.csv"))
    a = naive.groupby(["model", "budget", "frac"]).delta.mean().rename("naive")
    b = ctrl.groupby(["model", "budget", "frac"]).delta.mean().rename("controlled")
    out = pd.concat([a, b], axis=1).reset_index()
    out["leakage_gap"] = out.naive - out.controlled
    out.to_csv(TABLE_DIR / "table_leakage_gap.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
    ax = axes[0]
    for budget, color in zip([0.10, 0.25, 0.50], ["#4C72B0", "#DD8452", "#8172B3"]):
        n = out[(out.model == "ddpm") & (out.budget == budget)]
        ax.plot(n.frac, n.naive, marker="o", ms=3.5, color=color, label=f"{int(budget*100)}% (naive)")
        ax.plot(n.frac, n.controlled, marker="o", ms=3.5, color=color, ls="--",
                label=f"{int(budget*100)}% (controlled)")
    ax.axhline(0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("synthetic / needed volume"); ax.set_ylabel("Δ ROC-AUC")
    ax.set_title("DDPM: generator label access inflates the gain")
    ax.legend(frameon=False, fontsize=6.5)

    ax = axes[1]
    for model in ["vae", "gan", "ddpm"]:
        n = out[(out.model == model) & (out.budget == 0.25)]
        ax.plot(n.frac, n.naive, marker="o", ms=3.5, color=COL[model], label=f"{model} naive")
        ax.plot(n.frac, n.controlled, marker="o", ms=3.5, color=COL[model], ls="--",
                label=f"{model} controlled")
    ax.axhline(0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("synthetic / needed volume"); ax.set_title("25% labels: all generators")
    ax.legend(frameon=False, fontsize=6.2)

    ax = axes[2]
    budgets = [0.10, 0.25, 0.50, 1.00]
    ks = [fidelity(f"adult_r05_b{int(round(b*100)):02d}") if b < 1 else fidelity("adult_r05")
          for b in budgets]
    ax.bar([f"{int(b*100)}%" for b in budgets], ks, color="#55A868", width=0.55)
    for i, v in enumerate(ks):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, 0.16)
    ax.set_ylabel("mean KS")
    ax.set_title("DDPM fidelity vs. label budget")
    fig.suptitle("Label leakage audit: the same synthetic samples, with and without generator access to withheld labels", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig22_leakage.png", bbox_inches="tight")
    plt.close(fig)

    print(out[out.frac == 1.0].round(4).to_string(index=False))
    return out


if __name__ == "__main__":
    main()
