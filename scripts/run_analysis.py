"""Aggregate results, run the follow-up analyses and produce figures/tables."""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import Encoding, prepare_dataset  # noqa: E402
from src.evaluate import fit_evaluate, fit_tstr, load_bundle, needed_synthetic_count  # noqa: E402
from src.metrics import fidelity_report  # noqa: E402
from src.selection import select_drs, select_random  # noqa: E402
from src.train import TrainConfig, load_generator, sample_synthetic  # noqa: E402
from src.utils import FIG_DIR, METRIC_DIR, RUN_DIR, TABLE_DIR, get_logger, save_json  # noqa: E402

log = get_logger()
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
})
C0, C1, C2 = "#4C72B0", "#DD8452", "#55A868"
MODEL_COLORS = {"vae": "#8172B3", "gan": "#C44E52", "ddpm": "#55A868"}
MODEL_LABEL = {"vae": "VAE", "gan": "GAN", "ddpm": "DDPM"}


def tag_meta(tag: str) -> tuple[str, str]:
    if tag.endswith("_nat"):
        return tag[:-4], "natural"
    ds, r = tag.rsplit("_r", 1)
    return ds, f"{int(r) / 100:.2f}"


def load_utility() -> pd.DataFrame:
    frames = []
    for p in sorted(METRIC_DIR.glob("utility_*.csv")):
        tag = p.stem.replace("utility_", "")
        ds, ratio = tag_meta(tag)
        df = pd.read_csv(p)
        df["tag"] = tag
        df["ratio_label"] = ratio
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_fidelity() -> pd.DataFrame:
    rows = []
    for p in sorted(RUN_DIR.glob("*/*/fidelity.json")):
        d = json.load(open(p, encoding="utf-8"))
        d["tag"] = p.parent.parent.name
        rows.append(d)
    df = pd.DataFrame(rows)
    df["ratio_label"] = df["tag"].map(lambda t: tag_meta(t)[1])
    return df


def boot_ci(x: np.ndarray, n_boot: int = 4000, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = [(k, v) for k, v in pvals.items() if np.isfinite(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        prev = max(prev, adj)
        out[k] = prev
    for k in pvals:
        out.setdefault(k, np.nan)
    return out


# --------------------------------------------------------------------------- #
# analyses
# --------------------------------------------------------------------------- #

def fidelity_tables(fid: pd.DataFrame):
    agg = fid.groupby(["tag", "ratio_label", "model"]).agg(
        n=("seed", "count"),
        ks_mean=("ks_mean", "mean"), ks_std=("ks_mean", "std"),
        tv_mean=("tv_mean", "mean"), tv_std=("tv_mean", "std"),
        mmd=("mmd", "mean"), mmd_std=("mmd", "std"),
        coverage=("coverage", "mean"),
        vendi=("vendi_ratio", "mean"), vendi_std=("vendi_ratio", "std"),
        dcr=("dcr", "mean"), dcr_std=("dcr", "std"),
        std_ratio=("std_ratio_mean", "mean"),
    ).reset_index()
    agg.to_csv(TABLE_DIR / "table_fidelity.csv", index=False)
    return agg


def plot_fidelity(agg: pd.DataFrame):
    main = agg[agg.tag == "adult_r05"]
    metrics = [("ks_mean", "KS (lower better)", 0, 0.30), ("tv_mean", "TV (lower better)", 0, 0.30),
               ("mmd", "MMD (lower better)", 0, 0.35), ("coverage", "coverage (higher better)", 0, 1.1),
               ("vendi", "Vendi ratio (1 = real)", 0, 2.0), ("dcr", "DCR (1 = real scale)", 0, 2.0)]
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 4.6))
    for ax, (col, title, lo, hi) in zip(axes.ravel(), metrics):
        vals = [main.loc[main.model == m, col].values for m in ["vae", "gan", "ddpm"]]
        bp = ax.bar(["VAE", "GAN", "DDPM"], [v.mean() for v in vals],
                    yerr=[v.std() for v in vals], capsize=3,
                    color=[MODEL_COLORS[m] for m in ["vae", "gan", "ddpm"]], width=0.6)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(lo, hi)
        if col == "dcr":
            ax.axhline(1.0, color="k", ls="--", lw=0.8)
        for b, v in zip(bp, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v.mean():.2f}", ha="center",
                    va="bottom", fontsize=7)
    fig.suptitle("Adult (5% minority): distributional quality of synthetic minority samples", y=1.00)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig10_fidelity.png", bbox_inches="tight")
    plt.close(fig)


def utility_table(util: pd.DataFrame):
    cols = ["auc", "ap", "f1", "recall", "precision", "balanced_acc", "brier", "ece"]
    agg = util.groupby(["dataset", "ratio_label", "classifier", "model", "strategy"])[cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(TABLE_DIR / "table_utility.csv", index=False)
    return agg


def delta_analysis(util: pd.DataFrame):
    rows = []
    for (tag, seed, clf), g in util.groupby(["tag", "seed", "classifier"]):
        base = g[g.strategy == "none"]
        if base.empty:
            continue
        base_auc = float(base.auc.iloc[0])
        base_ap = float(base.ap.iloc[0])
        base_f1 = float(base.f1.iloc[0])
        for _, r in g.iterrows():
            rows.append({"tag": tag, "seed": seed, "classifier": clf, "model": r.model,
                         "strategy": r.strategy, "d_auc": r.auc - base_auc,
                         "d_ap": r.ap - base_ap, "d_f1": r.f1 - base_f1})
    delta = pd.DataFrame(rows)
    out = []
    for (tag, clf, strategy), g in delta.groupby(["tag", "classifier", "strategy"]):
        rec = {"tag": tag, "classifier": clf, "strategy": strategy, "n": len(g)}
        for met in ["d_auc", "d_ap", "d_f1"]:
            v = g[met].to_numpy()
            lo, hi = boot_ci(v)
            rec[f"{met}_mean"] = float(v.mean())
            rec[f"{met}_lo"] = lo
            rec[f"{met}_hi"] = hi
            try:
                rec[f"{met}_p"] = float(wilcoxon(v).pvalue) if np.any(v != 0) and len(v) >= 4 else np.nan
            except ValueError:
                rec[f"{met}_p"] = np.nan
        out.append(rec)
    res = pd.DataFrame(out)
    for tag, clf in res[["tag", "classifier"]].drop_duplicates().itertuples(index=False):
        m = (res.tag == tag) & (res.classifier == clf)
        for met in ["d_auc", "d_ap", "d_f1"]:
            p = holm(dict(zip(res.loc[m, "strategy"], res.loc[m, f"{met}_p"])))
            for i, s in res.loc[m, "strategy"].items():
                res.loc[i, f"{met}_p_holm"] = p.get(s, np.nan)
    res.to_csv(TABLE_DIR / "table_delta_utility.csv", index=False)
    return delta, res


def plot_delta(res: pd.DataFrame):
    d = res[(res.tag == "adult_r05") & (res.classifier == "hgb") & (res.strategy != "none")].copy()
    order = ["ros", "smote", "vae_all", "vae_rand", "vae_drs", "gan_all", "gan_rand", "gan_drs",
             "ddpm_all", "ddpm_rand", "ddpm_drs"]
    d["order"] = d.strategy.map({s: i for i, s in enumerate(order)})
    d = d.sort_values("order")
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    colors = []
    for s in d.strategy:
        colors.append(MODEL_COLORS.get(s.split("_")[0], "#888888"))
    y = np.arange(len(d))
    ax.errorbar(d.d_auc_mean, y, xerr=[d.d_auc_mean - d.d_auc_lo, d.d_auc_hi - d.d_auc_mean],
                fmt="o", ms=4, color="k", ecolor="#555555", capsize=3, lw=1)
    ax.scatter(d.d_auc_mean, y, c=colors, s=45, zorder=3)
    ax.axvline(0, color="k", ls="--", lw=0.9)
    ax.set_yticks(y, d.strategy, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Δ ROC-AUC vs. no augmentation (95% bootstrap CI)")
    ax.set_title("Adult 5% / HistGradientBoosting")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig11_delta_auc.png", bbox_inches="tight")
    plt.close(fig)


def dose_response(tag: str, seeds: list[int]):
    ds, _ = tag_meta(tag)
    data = prepare_dataset(ds, 0.05, seed=0)
    n_all = needed_synthetic_count(data)
    fracs = [0.0, 0.125, 0.25, 0.5, 1.0]
    rows = []
    for seed in seeds:
        for model in ["vae", "gan", "ddpm"]:
            rd = RUN_DIR / tag / f"{model}_s{seed}"
            if not (rd / "samples_pool.npz").exists():
                continue
            b = load_bundle(rd, model, seed)
            for frac in fracs:
                n = int(round(frac * n_all))
                synth = Encoding(b.all_enc.num[:n], b.all_enc.cat[:n], b.all_enc.y[:n]) if n else Encoding(
                    np.zeros((0, data.preprocessor.n_num), np.float32),
                    np.zeros((0, data.preprocessor.n_cat), np.float32), np.zeros(0, np.int64))
                for clf in ["hgb", "lr"]:
                    m = fit_evaluate(data, synth, clf, seed)
                    rows.append({"tag": tag, "seed": seed, "model": model, "frac": frac,
                                 "n_syn": n, "classifier": clf, **m})
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "table_dose_response.csv", index=False)
    return df


def plot_dose(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    for ax, clf in zip(axes, ["hgb", "lr"]):
        sub = df[df.classifier == clf]
        base = sub[sub.frac == 0].groupby("seed").auc.mean()
        for model in ["vae", "gan", "ddpm"]:
            g = sub[sub.model == model].groupby(["seed", "frac"]).auc.mean().reset_index()
            g["d"] = g.apply(lambda r: r.auc - base.loc[r.seed], axis=1)
            agg = g.groupby("frac").d.agg(["mean", "std"]).reset_index()
            ax.errorbar(agg.frac, agg["mean"], yerr=agg["std"], marker="o", ms=4, capsize=3,
                        color=MODEL_COLORS[model], label=MODEL_LABEL[model])
        ax.axhline(0, color="k", ls="--", lw=0.8)
        ax.set_xlabel("synthetic / needed volume")
        ax.set_ylabel("Δ ROC-AUC vs. none")
        ax.set_title("HistGradientBoosting" if clf == "hgb" else "Logistic regression")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Dose-response of synthetic augmentation (Adult, 5% minority)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig12_dose_response.png", bbox_inches="tight")
    plt.close(fig)


def tstr_analysis(tag: str, seeds: list[int], util: pd.DataFrame):
    ds, _ = tag_meta(tag)
    data = prepare_dataset(ds, 0.05 if ds == "adult" else None, seed=0)
    rows = []
    for seed in seeds:
        for model in ["vae", "gan", "ddpm"]:
            rd = RUN_DIR / tag / f"{model}_s{seed}"
            if not (rd / "samples_all.npz").exists():
                continue
            b = load_bundle(rd, model, seed)
            for clf in ["hgb", "lr"]:
                m = fit_tstr(data, b.all_enc, clf, seed)
                rows.append({"tag": tag, "seed": seed, "model": model, "classifier": clf, **m})
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "table_tstr.csv", index=False)
    return df


def plot_tstr(df: pd.DataFrame, util: pd.DataFrame):
    tag = "adult_r05"
    d = df[df.tag == tag]
    base = util[(util.tag == tag) & (util.strategy == "none")].groupby("classifier").auc.mean()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for ax, clf in zip(axes, ["hgb", "lr"]):
        sub = d[d.classifier == clf]
        means = [sub[sub.model == m].auc.mean() for m in ["vae", "gan", "ddpm"]]
        stds = [sub[sub.model == m].auc.std() for m in ["vae", "gan", "ddpm"]]
        ax.bar(["VAE", "GAN", "DDPM"], means, yerr=stds, capsize=3,
               color=[MODEL_COLORS[m] for m in ["vae", "gan", "ddpm"]], width=0.55)
        ax.axhline(base[clf], color="k", ls="--", lw=1.0, label="real minority only")
        ax.set_ylim(0.75, 0.95)
        ax.set_ylabel("ROC-AUC")
        ax.set_title("TSTR · HistGradientBoosting" if clf == "hgb" else "TSTR · Logistic regression")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Train on synthetic minority + real majority, test on real data", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig13_tstr.png", bbox_inches="tight")
    plt.close(fig)


def meta_analysis(fid: pd.DataFrame, delta: pd.DataFrame, tstr: pd.DataFrame, util: pd.DataFrame):
    d = delta[(delta.model != "none") & (delta.strategy.str.endswith("_all"))]
    d = d.groupby(["tag", "seed", "model"]).d_auc.mean().reset_index()
    t = tstr.groupby(["tag", "seed", "model"]).auc.mean().reset_index().rename(columns={"auc": "tstr_auc"})
    m = fid.merge(d, on=["tag", "seed", "model"], how="inner").merge(t, on=["tag", "seed", "model"], how="inner")
    metrics = ["ks_mean", "tv_mean", "mmd", "coverage", "vendi_ratio", "dcr", "std_ratio_mean"]
    rows = []
    for met in metrics:
        for target in ["d_auc", "tstr_auc"]:
            r, p = spearmanr(m[met], m[target], nan_policy="omit")
            rows.append({"metric": met, "target": target, "spearman": float(r), "p": float(p), "n": len(m)})
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "table_meta_correlation.csv", index=False)
    m.to_csv(TABLE_DIR / "table_run_level.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0))
    for ax, met in zip(axes, ["mmd", "dcr", "coverage"]):
        ax.scatter(m[met], m.d_auc, c=[MODEL_COLORS[x] for x in m.model], s=28, alpha=0.8)
        r, p = spearmanr(m[met], m.d_auc, nan_policy="omit")
        ax.axhline(0, color="k", ls="--", lw=0.8)
        ax.set_xlabel(met); ax.set_ylabel("ΔAUC (augmentation)")
        ax.set_title(f"ρ={r:.2f}, p={p:.2f}", fontsize=9)
    fig.suptitle("Generator-level metrics vs. downstream augmentation gain", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig14_meta.png", bbox_inches="tight")
    plt.close(fig)
    return out


def selection_analysis(tag: str, seeds: list[int]):
    ds, _ = tag_meta(tag)
    data = prepare_dataset(ds, 0.05 if ds == "adult" else None, seed=0)
    pre = data.preprocessor
    real = data.val.df[data.val.df[pre.target_name] == 1].reset_index(drop=True)
    renc = np.hstack([data.val.enc.num[data.val.enc.y == 1], data.val.enc.cat[data.val.enc.y == 1]])
    real_tr = np.hstack([data.train.enc.num[data.train.enc.y == 1], data.train.enc.cat[data.train.enc.y == 1]])
    rows = []
    for seed in seeds:
        for model in ["vae", "gan", "ddpm"]:
            rd = RUN_DIR / tag / f"{model}_s{seed}"
            if not (rd / "samples_pool.npz").exists():
                continue
            b = load_bundle(rd, model, seed)
            pool = np.hstack([b.pool_enc.num, b.pool_enc.cat])
            n = len(b.all_enc.y)
            idxs = {"pool": np.arange(len(pool)), "random": select_random(len(pool), n, seed),
                    "drs": select_drs(pool, real_tr, n, seed)}
            for name, idx in idxs.items():
                cat_idx = np.column_stack([b.pool_enc.cat[idx][:, sl].argmax(1) for sl in pre.cat_slices])
                df = pre.to_frame(b.pool_enc.num[idx], b.pool_enc.cat[idx], cat_idx=cat_idx)
                f = fidelity_report(real, df, renc, pool[idx], pre.num_cols, pre.cat_cols, seed=seed)
                rows.append({"tag": tag, "seed": seed, "model": model, "selection": name, **f})
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "table_selection_fidelity.csv", index=False)
    return df


def plot_selection(sel: pd.DataFrame):
    main = sel[sel.tag == "adult_r05"]
    metrics = ["ks_mean", "tv_mean", "mmd", "coverage", "vendi_ratio"]
    fig, axes = plt.subplots(1, 5, figsize=(11, 2.5))
    for ax, met in zip(axes, metrics):
        for i, model in enumerate(["vae", "gan", "ddpm"]):
            sub = main[main.model == model]
            vals = [sub[sub.selection == s][met].mean() for s in ["pool", "random", "drs"]]
            ax.bar(np.arange(3) + i * 0.26, vals, width=0.24, color=MODEL_COLORS[model],
                   label=MODEL_LABEL[model] if met == metrics[0] else None)
        ax.set_xticks(np.arange(3) + 0.26, ["all", "random", "DRS"], fontsize=7)
        ax.set_title(met, fontsize=8)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Effect of selection on distributional quality (Adult 5%)", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig15_selection.png", bbox_inches="tight")
    plt.close(fig)


def diffusion_steps(tag="adult_r05", seed=0):
    ds, _ = tag_meta(tag)
    data = prepare_dataset(ds, 0.05, seed=0)
    pre = data.preprocessor
    real = data.val.df[data.val.df[pre.target_name] == 1]
    renc = np.hstack([data.val.enc.num[data.val.enc.y == 1], data.val.enc.cat[data.val.enc.y == 1]])
    model, cfg = load_generator("ddpm", RUN_DIR / tag / f"ddpm_s{seed}", data, "cuda")
    rows = []
    for steps in [10, 25, 50, 100, 200]:
        cfg2 = replace(cfg, ddim_steps=steps)
        df, enc = sample_synthetic("ddpm", model, data, 3000, seed, "cuda", cfg2)
        f = fidelity_report(real, df, renc, np.hstack([enc.num, enc.cat]), pre.num_cols, pre.cat_cols, seed=seed)
        rows.append({"steps": steps, **f})
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "table_diffusion_steps.csv", index=False)
    fig, ax1 = plt.subplots(figsize=(4.4, 3.0))
    ax1.plot(out.steps, out.ks_mean, marker="o", label="KS", color=C0)
    ax1.plot(out.steps, out.tv_mean, marker="s", label="TV", color=C1)
    ax1.set_xlabel("DDIM steps"); ax1.set_ylabel("error (lower better)")
    ax2 = ax1.twinx()
    ax2.plot(out.steps, out.mmd, marker="^", label="MMD", color=C2)
    ax2.set_ylabel("MMD")
    ax1.legend(frameon=False, fontsize=8, loc="upper right")
    ax1.set_title("DDPM sample quality vs. sampling steps")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig16_diffusion_steps.png", bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    util = load_utility()
    fid = load_fidelity()
    log.info("utility rows=%d fidelity runs=%d", len(util), len(fid))

    fagg = fidelity_tables(fid)
    plot_fidelity(fagg)
    uagg = utility_table(util)
    delta, dres = delta_analysis(util)
    plot_delta(dres)

    dose = dose_response("adult_r05", [0, 1, 2, 3, 4])
    plot_dose(dose)

    tstr = tstr_analysis("adult_r05", [0, 1, 2, 3, 4], util)
    plot_tstr(tstr, util)

    meta = meta_analysis(fid, delta, tstr, util)
    sel = selection_analysis("adult_r05", [0, 1, 2, 3, 4])
    plot_selection(sel)
    diff = diffusion_steps()

    summary = {
        "fidelity_adult": fagg[fagg.tag == "adult_r05"].to_dict("records"),
        "delta_adult_hgb": dres[(dres.tag == "adult_r05") & (dres.classifier == "hgb")].to_dict("records"),
        "meta": meta.to_dict("records"),
        "dose": dose.groupby(["model", "frac"]).auc.mean().reset_index().to_dict("records"),
    }
    save_json(summary, TABLE_DIR / "analysis_summary.json")
    log.info("analysis done")


if __name__ == "__main__":
    main()
