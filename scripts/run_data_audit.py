"""Data-layer audit: duplicates, covariate shift, class overlap, missingness,
association preservation and subgroup base rates.

These checks complement the descriptive EDA: they test whether the modelling
pipeline is built on a sound split, whether classes are separable at all, and
whether the generators preserve *dependence structure* rather than only
marginals.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import DATASET_SPECS, load_dataset, prepare_dataset  # noqa: E402
from src.evaluate import load_bundle  # noqa: E402
from src.utils import FIG_DIR, RUN_DIR, TABLE_DIR, get_logger  # noqa: E402

log = get_logger()
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})
C0, C1 = "#4C72B0", "#DD8452"


def duplicates(df: pd.DataFrame, target: str) -> tuple[int, int]:
    feats = [c for c in df.columns if c != target]
    dup = int(df.duplicated(subset=feats + [target]).sum())
    g = df.groupby(feats, dropna=False)[target].nunique()
    conflict = int((g > 1).sum())
    return dup, conflict


def shift_table(name: str, data) -> dict:
    spec = DATASET_SPECS[name]
    tr, te = data.train.df, data.test.df
    ks = [ks_2samp(tr[c].to_numpy(float), te[c].to_numpy(float)).statistic for c in spec["num_cols"]]
    tv = []
    for c in spec["cat_cols"]:
        p = tr[c].astype(str).value_counts(normalize=True)
        q = te[c].astype(str).value_counts(normalize=True)
        idx = p.index.union(q.index)
        tv.append(float(0.5 * np.abs(p.reindex(idx, fill_value=0) - q.reindex(idx, fill_value=0)).sum()))
    X = np.vstack([np.hstack([data.train.enc.num, data.train.enc.cat]),
                   np.hstack([data.test.enc.num, data.test.enc.cat])])
    y = np.r_[np.zeros(len(data.train.enc.y)), np.ones(len(data.test.enc.y))]
    clf = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, random_state=0)
    auc = cross_val_score(clf, X, y, cv=3, scoring="roc_auc", n_jobs=1).mean()
    return {"ks_mean": float(np.mean(ks)), "ks_max": float(np.max(ks)),
            "tv_mean": float(np.mean(tv)), "tv_max": float(np.max(tv)),
            "shift_auc": float(auc)}


def overlap_table(data) -> dict:
    tr, te = data.train.enc, data.test.enc
    Xtr = np.hstack([tr.num, tr.cat])
    Xte = np.hstack([te.num, te.cat])
    maj = tr.y == 0
    mino = te.y == 1
    nn = NearestNeighbors(n_neighbors=5).fit(Xtr)
    _, idx = nn.kneighbors(Xte[mino])
    frac_maj = (tr.y[idx] == 0).mean(axis=1)
    knn = KNeighborsClassifier(n_neighbors=5).fit(Xtr, tr.y)
    err = 1.0 - knn.score(Xte, te.y)
    return {"overlap_rate": float((frac_maj >= 0.6).mean()),
            "minority_majority_nn_frac": float(frac_maj.mean()),
            "knn_error": float(err)}


def missing_patterns() -> pd.DataFrame:
    import io, zipfile
    from src.data import ADULT_COLS
    with zipfile.ZipFile(ROOT / "data" / "raw" / "adult.zip") as z:
        frames = []
        for nm in ("adult.data", "adult.test"):
            text = z.read(nm).decode("utf-8", errors="replace")
            lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("|")]
            frames.append(pd.read_csv(io.StringIO("\n".join(lines)), header=None, names=ADULT_COLS,
                                      skipinitialspace=True, na_values=["?"]))
    df = pd.concat(frames, ignore_index=True)
    df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")
    df["y"] = (df["income"] == ">50K").astype(int)
    cols = ["workclass", "occupation", "native_country"]
    pat = df[cols].isna().apply(lambda r: "|".join(c for c, v in zip(cols, r) if v) or "complete", axis=1)
    out = df.assign(pattern=pat).groupby("pattern")["y"].agg(["count", "mean"]).reset_index()
    out.columns = ["missing_pattern", "n", "positive_rate"]
    out["share"] = out.n / out.n.sum()
    return out.sort_values("n", ascending=False)


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    tab = pd.crosstab(a.astype(str), b.astype(str))
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(tab)[0]
    n = tab.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))


def association_table(name: str = "adult") -> pd.DataFrame:
    spec = DATASET_SPECS[name]
    data = prepare_dataset(name, 0.05, seed=0)
    tr = data.train.df
    real = data.val.df[data.val.df[spec["target"]] == 1]
    cols = spec["cat_cols"]
    pairs = [(a, b) for i, a in enumerate(cols) for b in cols[i + 1:]]
    v_real = np.array([cramers_v(real[a], real[b]) for a, b in pairs])
    rows = []
    for model in ["vae", "gan", "ddpm"]:
        rd = RUN_DIR / "adult_r05" / f"{model}_s0"
        if not (rd / "samples_all.csv").exists():
            continue
        syn = pd.read_csv(rd / "samples_all.csv")
        v_syn = np.array([cramers_v(syn[a], syn[b]) for a, b in pairs])
        rows.append({"model": model, "mean_abs_diff": float(np.abs(v_syn - v_real).mean()),
                     "fro_rel_err": float(np.linalg.norm(v_syn - v_real) / (np.linalg.norm(v_real) + 1e-12)),
                     "max_abs_diff": float(np.abs(v_syn - v_real).max())})
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "table_association.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.bar(out.model, out.fro_rel_err, color=[C0, C1, "#55A868"])
    for i, v in enumerate(out.fro_rel_err):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("relative Frobenius error")
    ax.set_title("Cramér's V structure: synthetic vs real")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig17_association.png", bbox_inches="tight")
    plt.close(fig)
    return out


def subgroup_rates() -> pd.DataFrame:
    data = prepare_dataset("adult", 0.05, seed=0)
    df = data.test.df.copy()
    df["age_bin"] = pd.cut(df["age"], [17, 35, 50, 65, 100], labels=["18-35", "36-50", "51-65", "65+"])
    df["race_group"] = np.where(df["race"] == "White", "White", "Non-White")
    rows = []
    for col in ["sex", "race_group", "age_bin"]:
        g = df.groupby(col, observed=True)["income"].agg(["count", "mean"]).reset_index()
        for _, r in g.iterrows():
            rows.append({"group": col, "value": str(r[col]), "n": int(r["count"]), "positive_rate": float(r["mean"])})
    return pd.DataFrame(rows)


def main():
    rows, groups = [], []
    for name in ["adult", "bank"]:
        raw = load_dataset(name)
        target = DATASET_SPECS[name]["target"]
        dup, conflict = duplicates(raw, target)
        data = prepare_dataset(name, None, seed=0)
        sh = shift_table(name, data)
        ov = overlap_table(data)
        rows.append({"dataset": name, "n": len(raw), "dup_rows": dup, "conflict_groups": conflict, **sh, **ov})
        groups.append((name, data))
        log.info("%s: %s", name, rows[-1])
    audit = pd.DataFrame(rows)
    audit.to_csv(TABLE_DIR / "table_data_audit.csv", index=False)

    mp = missing_patterns()
    mp.to_csv(TABLE_DIR / "table_missing_patterns.csv", index=False)

    assoc = association_table("adult")
    sr = subgroup_rates()
    sr.to_csv(TABLE_DIR / "table_subgroup_rates.csv", index=False)

    # one compact figure: shift per feature
    spec = DATASET_SPECS["adult"]
    data = prepare_dataset("adult", None, seed=0)
    tr, te = data.train.df, data.test.df
    ks = [ks_2samp(tr[c].to_numpy(float), te[c].to_numpy(float)).statistic for c in spec["num_cols"]]
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    ax.bar(spec["num_cols"], ks, color=C0)
    for i, v in enumerate(ks):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("KS(train, test)")
    ax.set_title("Covariate shift check (Adult numeric features)")
    plt.xticks(rotation=30, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig18_shift.png", bbox_inches="tight")
    plt.close(fig)
    log.info("audit done")


if __name__ == "__main__":
    main()
