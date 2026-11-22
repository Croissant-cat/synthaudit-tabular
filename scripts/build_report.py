"""Fill report/report_template.md with the measured tables/numbers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import prepare_dataset  # noqa: E402
from src.evaluate import needed_synthetic_count  # noqa: E402
from src.utils import TABLE_DIR  # noqa: E402

T = TABLE_DIR
STRAT_LABEL = {
    "none": "不增广", "ros": "随机过采样", "smote": "SMOTE",
    "vae_all": "VAE（全部）", "vae_rand": "VAE（随机子集）", "vae_drs": "VAE + DRS",
    "gan_all": "GAN（全部）", "gan_rand": "GAN（随机子集）", "gan_drs": "GAN + DRS",
    "ddpm_all": "DDPM（全部）", "ddpm_rand": "DDPM（随机子集）", "ddpm_drs": "DDPM + DRS",
}
ORDER = ["none", "ros", "smote", "vae_all", "vae_rand", "vae_drs",
         "gan_all", "gan_rand", "gan_drs", "ddpm_all", "ddpm_rand", "ddpm_drs"]


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{nd}f}"


def pfmt(p):
    if p is None or not np.isfinite(p):
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def pm(mean, std, nd=3):
    if std is None or not np.isfinite(std):
        return fmt(mean, nd)
    return f"{mean:.{nd}f} ± {std:.{nd}f}"


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def load(name):
    p = T / name
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def table_datasets():
    df = load("eda_summary.csv")
    rows = []
    for _, r in df.iterrows():
        name = "Adult" if r.dataset == "adult" else "Bank Marketing"
        rows.append([name, f"{int(r.n):,}", int(r.n_features), f"{int(r.n_numeric)} / {int(r.n_categorical)}",
                     f"{r.positive_rate:.1%}", f"{int(r.missing_cells):,}"])
    return md_table(["数据集", "样本数", "原始特征", "数值 / 类别", "正类率", "缺失单元格"], rows)


def table_mi():
    df = load("eda_mutual_info.csv")
    rows = []
    for ds, name in [("adult", "Adult"), ("bank", "Bank")]:
        sub = df[df.dataset == ds].head(5)
        rows.append([name, "；".join(f"{r.feature} ({r.mi:.3f})" for _, r in sub.iterrows())])
    return md_table(["数据集", "互信息最高的 5 个特征（MI）"], rows)


def table_imbalance():
    rows = []
    for ds, ratio, label in [("adult", 0.05, "Adult 5%"), ("adult", 0.10, "Adult 10%"), ("bank", None, "Bank 自然 (11.7%)")]:
        data = prepare_dataset(ds, ratio, seed=0)
        tr, va, te = data.train.enc, data.val.enc, data.test.enc
        n_syn = needed_synthetic_count(data)
        rows.append([label, f"{len(tr.y):,}", f"{int(tr.y.sum()):,}", f"{len(va.y):,}", f"{len(te.y):,}",
                     f"{tr.y.mean():.1%}", f"{n_syn:,}"])
    return md_table(["设置", "训练集", "训练集少数类", "验证集", "测试集", "训练集正类率", "需合成量"], rows)


def table_hparam():
    rows = [
        ["VAE", "隐变量 16；MLP 256×2；ELBO（MSE + CE + β·KL，β 10 epoch 升到 1）；90 epoch；AdamW 2e-3；batch 512"],
        ["GAN", "WGAN-GP；噪声 32；MLP 256×2；n_critic=2；λ_GP=10；Gumbel-Softmax；30 epoch；AdamW 1e-4；batch 512"],
        ["DDPM", "T=400；线性 β 1e-4→0.02；噪声预测（数值）+ CE（类别）；EMA 0.999；100 epoch；AdamW 2e-3；DDIM 50 步"],
        ["共享", "类别嵌入 16；隐藏层 256；LayerNorm + SiLU；输出 90 维（5 数值 + 85 独热）"],
        ["下游", "HistGradientBoosting 250 轮 / lr 0.06 / 叶 31；LogisticRegression C=1.0；均 class_weight='balanced'"],
    ]
    return md_table(["组件", "设置"], rows)


def table_fidelity():
    df = load("table_fidelity.csv")
    sub = df[df.tag == "adult_r05"].set_index("model")
    rows = []
    for m, label in [("vae", "VAE"), ("gan", "GAN"), ("ddpm", "DDPM")]:
        r = sub.loc[m]
        rows.append([label, pm(r.ks_mean, r.ks_std), pm(r.tv_mean, r.tv_std), fmt(r.std_ratio),
                     pm(r.mmd, r.mmd_std), fmt(r.coverage), pm(r.vendi, r.vendi_std), pm(r.dcr, r.dcr_std)])
    return md_table(["模型", "KS ↓", "TV ↓", "数值标准差比", "MMD ↓", "覆盖率 ↑", "Vendi 比", "DCR"], rows)


def add_tag(df):
    df = df.copy()

    def mk(r):
        if str(r.ratio_label) == "natural":
            return f"{r.dataset}_nat"
        return f"{r.dataset}_r{int(round(float(r.ratio_label) * 100)):02d}"

    df["tag"] = df.apply(mk, axis=1)
    return df


def table_utility(tag="adult_r05", clf="hgb"):
    df = add_tag(load("table_utility.csv"))
    sub = df[(df.tag == tag) & (df.classifier == clf)].set_index("strategy")
    rows = []
    for s in ORDER:
        if s not in sub.index:
            continue
        r = sub.loc[s]
        rows.append([STRAT_LABEL[s], fmt(r.auc_mean), fmt(r.ap_mean), fmt(r.f1_mean),
                     fmt(r.recall_mean), fmt(r.precision_mean), fmt(r.ece_mean)])
    return md_table(["策略", "AUC ↑", "PR-AUC ↑", "F1 ↑", "召回 ↑", "精确率 ↑", "ECE ↓"], rows)


def table_delta(tag="adult_r05", clf="hgb"):
    df = load("table_delta_utility.csv")
    sub = df[(df.tag == tag) & (df.classifier == clf)].set_index("strategy")
    rows = []
    for s in ORDER:
        if s == "none" or s not in sub.index:
            continue
        r = sub.loc[s]
        rows.append([STRAT_LABEL[s], f"{r.d_auc_mean:+.4f} [{r.d_auc_lo:+.4f}, {r.d_auc_hi:+.4f}]",
                     pfmt(r.d_auc_p_holm)])
    return md_table(["策略", "ΔAUC [95% CI]", "Holm 校正 p"], rows)


def table_tstr():
    df = load("table_tstr.csv")
    sub = df[df.tag == "adult_r05"]
    util = add_tag(load("table_utility.csv"))
    auc_none = float(util[(util.tag == "adult_r05") & (util.strategy == "none")
                          & (util.classifier == "hgb")].auc_mean.iloc[0])
    rows = []
    for m, label in [("vae", "VAE"), ("gan", "GAN"), ("ddpm", "DDPM")]:
        h = float(sub[(sub.model == m) & (sub.classifier == "hgb")].auc.mean())
        l = float(sub[(sub.model == m) & (sub.classifier == "lr")].auc.mean())
        rows.append([label, fmt(h), fmt(l), f"{h / auc_none:.1%}"])
    return md_table(["生成器", "HGB TSTR AUC", "LR TSTR AUC", "相对真实标签基线"], rows)


def table_selection():
    df = load("table_selection_fidelity.csv")
    sub = df[df.tag == "adult_r05"]
    rows = []
    for m, label in [("vae", "VAE"), ("gan", "GAN"), ("ddpm", "DDPM")]:
        for sel, sl in [("pool", "全部候选"), ("random", "随机子集"), ("drs", "DRS 子集")]:
            g = sub[(sub.model == m) & (sub.selection == sel)]
            rows.append([label, sl, fmt(g.ks_mean.mean()), fmt(g.tv_mean.mean()), fmt(g.mmd.mean()), fmt(g.dcr.mean())])
    return md_table(["模型", "选择方式", "KS ↓", "TV ↓", "MMD ↓", "DCR"], rows)


def table_meta():
    df = load("table_meta_correlation.csv")
    rows = []
    for met in ["ks_mean", "tv_mean", "mmd", "coverage", "vendi_ratio", "dcr", "std_ratio_mean"]:
        a = df[(df.metric == met) & (df.target == "d_auc")].iloc[0]
        t = df[(df.metric == met) & (df.target == "tstr_auc")].iloc[0]
        rows.append([met, f"{a.spearman:+.3f}", pfmt(a.p), f"{t.spearman:+.3f}", pfmt(t.p)])
    return md_table(["指标", "ρ (ΔAUC)", "p", "ρ (TSTR AUC)", "p"], rows)


def table_steps():
    df = load("table_diffusion_steps.csv")
    rows = [[int(r.steps), fmt(r.ks_mean), fmt(r.tv_mean), fmt(r.mmd), fmt(r.coverage)] for _, r in df.iterrows()]
    return md_table(["DDIM 步数", "KS ↓", "TV ↓", "MMD ↓", "覆盖率 ↑"], rows)


def table_appendix_lr():
    return table_utility("adult_r05", "lr")


def table_appendix_other():
    df = add_tag(load("table_utility.csv"))
    rows = []
    for tag, label in [("adult_r10", "Adult 10%"), ("bank_nat", "Bank 自然")]:
        sub = df[(df.tag == tag) & (df.classifier == "hgb")].set_index("strategy")
        for s in ["none", "ros", "smote", "vae_all", "gan_all", "ddpm_all"]:
            if s in sub.index:
                rows.append([label, STRAT_LABEL[s], fmt(sub.loc[s, "auc_mean"]), fmt(sub.loc[s, "ap_mean"])])
    return md_table(["设置", "策略", "AUC", "PR-AUC"], rows)


CAPTIONS = {
    "TABLE_DATASETS": "表 1. 数据集概况",
    "TABLE_MI": "表 2. 互信息最高的 5 个特征",
    "TABLE_IMBALANCE": "表 3. 不平衡设置与数据规模",
    "TABLE_HPARAM": "表 4. 生成器与下游模型的超参数",
    "TABLE_FIDELITY": "表 5. Adult 5% 下的分布保真度、多样性与隐私（均值 ± 标准差，5 个种子）",
    "TABLE_UTILITY": "表 6. Adult 5% 下的下游分类性能（HistGradientBoosting，5 个种子均值）",
    "TABLE_TSTR": "表 7. TSTR：仅用合成少数类训练、在真实测试集上评估",
    "TABLE_SELECTION": "表 8. 不同筛选方式的保真度（Adult 5%）",
    "TABLE_META": "表 9. 生成器指标与下游收益的 Spearman 相关（n=15）",
    "TABLE_STEPS": "表 10. DDIM 采样步数与保真度",
    "TABLE_APPENDIX_LR": "表 A1. 逻辑回归上的完整下游结果（Adult 5%）",
    "TABLE_APPENDIX_OTHER": "表 A2. Adult 10% 与 Bank 的 AUC / PR-AUC",
}


def build():
    tpl = (ROOT / "report" / "report_template.md").read_text(encoding="utf-8")

    fid = load("table_fidelity.csv")
    f5 = fid[fid.tag == "adult_r05"].set_index("model")
    util = add_tag(load("table_utility.csv"))
    u5 = util[(util.tag == "adult_r05") & (util.classifier == "hgb")].set_index("strategy")
    delta = load("table_delta_utility.csv")
    d5 = delta[(delta.tag == "adult_r05") & (delta.classifier == "hgb")].set_index("strategy")
    tstr = load("table_tstr.csv")
    ts5 = tstr[tstr.tag == "adult_r05"]
    meta = load("table_meta_correlation.csv")
    sel = load("table_selection_fidelity.csv")
    sel5 = sel[sel.tag == "adult_r05"]
    dose = load("table_dose_response.csv")
    steps = load("table_diffusion_steps.csv")
    eda = load("eda_summary.csv")

    data5 = prepare_dataset("adult", 0.05, seed=0)
    n_syn = needed_synthetic_count(data5)
    n_real = int((data5.train.enc.y == 1).sum())

    dose_h = dose[(dose.model == "ddpm") & (dose.classifier == "hgb")]
    base = dose_h[dose_h.frac == 0].groupby("seed").auc.mean()

    def dose_delta(frac):
        g = dose_h[dose_h.frac == frac]
        v = [r.auc - base.loc[r.seed] for _, r in g.iterrows()]
        return float(np.mean(v))

    auc_none = float(u5.loc["none", "auc_mean"])
    tstr_ddpm = float(ts5[(ts5.model == "ddpm") & (ts5.classifier == "hgb")].auc.mean())
    mmd_a = meta[(meta.metric == "mmd") & (meta.target == "d_auc")].iloc[0]
    mmd_t = meta[(meta.metric == "mmd") & (meta.target == "tstr_auc")].iloc[0]

    vals = {
        "TABLE_DATASETS": table_datasets(),
        "TABLE_MI": table_mi(),
        "TABLE_IMBALANCE": table_imbalance(),
        "TABLE_HPARAM": table_hparam(),
        "TABLE_FIDELITY": table_fidelity(),
        "TABLE_UTILITY": table_utility(),
        "TABLE_DELTA": table_delta(),
        "TABLE_TSTR": table_tstr(),
        "TABLE_SELECTION": table_selection(),
        "TABLE_META": table_meta(),
        "TABLE_STEPS": table_steps(),
        "TABLE_APPENDIX_LR": table_appendix_lr(),
        "TABLE_APPENDIX_OTHER": table_appendix_other(),
        "FID_DDPM_KS": fmt(f5.loc["ddpm", "ks_mean"]),
        "FID_DDPM_TV": fmt(f5.loc["ddpm", "tv_mean"]),
        "FID_DDPM_MMD": fmt(f5.loc["ddpm", "mmd"]),
        "FID_GAN_KS": fmt(f5.loc["gan", "ks_mean"]),
        "FID_GAN_STD": fmt(f5.loc["gan", "std_ratio"]),
        "FID_VAE_DCR": fmt(f5.loc["vae", "dcr"], 2),
        "FID_DDPM_DCR": fmt(f5.loc["ddpm", "dcr"], 2),
        "FID_DDPM_VENDI": fmt(f5.loc["ddpm", "vendi"], 2),
        "MISSING_CELLS": f"{int(eda[eda.dataset == 'adult'].missing_cells.iloc[0]):,}",
        "N_TRAIN_ADULT5": f"{len(data5.train.enc.y):,}",
        "N_POS_ADULT5": f"{n_real:,}",
        "N_SYN_ADULT5": f"{n_syn:,}",
        "SYN_RATIO_ADULT5": f"{n_syn / n_real:.1f}",
        "AUC_NONE": fmt(auc_none, 4),
        "AUC_ROS": fmt(u5.loc["ros", "auc_mean"], 4),
        "AUC_SMOTE": fmt(u5.loc["smote", "auc_mean"], 4),
        "AUC_DDPM_ALL": fmt(u5.loc["ddpm_all", "auc_mean"], 4),
        "AUC_VAE_ALL": fmt(u5.loc["vae_all", "auc_mean"], 4),
        "AUC_GAN_ALL": fmt(u5.loc["gan_all", "auc_mean"], 4),
        "DAUC_DDPM": f"{d5.loc['ddpm_all', 'd_auc_mean']:+.4f}",
        "DAUC_DDPM_LO": f"{d5.loc['ddpm_all', 'd_auc_lo']:+.4f}",
        "DAUC_DDPM_HI": f"{d5.loc['ddpm_all', 'd_auc_hi']:+.4f}",
        "DOSE_25_DDPM": f"{dose_delta(0.25):+.4f}",
        "DOSE_100_DDPM": f"{dose_delta(1.0):+.4f}",
        "TSTR_DDPM": fmt(tstr_ddpm, 4),
        "TSTR_VAE": fmt(ts5[(ts5.model == "vae") & (ts5.classifier == "hgb")].auc.mean(), 4),
        "TSTR_GAN": fmt(ts5[(ts5.model == "gan") & (ts5.classifier == "hgb")].auc.mean(), 4),
        "TSTR_RETENTION": f"{tstr_ddpm / auc_none:.1%}",
        "SEL_VAE_MMD_ALL": fmt(sel5[(sel5.model == "vae") & (sel5.selection == "pool")].mmd.mean()),
        "SEL_VAE_MMD_DRS": fmt(sel5[(sel5.model == "vae") & (sel5.selection == "drs")].mmd.mean()),
        "META_RHO_MMD_DAUC": f"{mmd_a.spearman:+.2f}",
        "META_P_MMD_DAUC": pfmt(mmd_a.p),
        "META_RHO_MMD_TSTR": f"{mmd_t.spearman:+.2f}",
        "META_P_MMD_TSTR": pfmt(mmd_t.p),
    }
    for k, v in vals.items():
        if k in CAPTIONS:
            tpl = tpl.replace("{{" + k + "}}", f"**{CAPTIONS[k]}**\n\n{v}")
        else:
            tpl = tpl.replace("{{" + k + "}}", str(v))
    missing = [m for m in __import__("re").findall(r"\{\{([A-Z0-9_]+)\}\}", tpl)]
    if missing:
        raise SystemExit(f"unfilled placeholders: {sorted(set(missing))}")
    (ROOT / "report" / "report.md").write_text(tpl, encoding="utf-8")
    print("wrote report/report.md", len(tpl), "chars")


if __name__ == "__main__":
    build()
