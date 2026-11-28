"""Fill report/report_template_v2.md with tables/numbers from the three protocols."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_report as b  # noqa: E402

T = b.T


def add_delta(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df.frac == 0].set_index(["seed", "model", "budget"]).auc
    df = df.copy()
    df["delta"] = [r.auc - base.loc[(r.seed, r.model, r.budget)] for _, r in df.iterrows()]
    return df


def table_audit():
    df = pd.read_csv(T / "table_data_audit.csv")
    rows = []
    for _, r in df.iterrows():
        name = "Adult" if r.dataset == "adult" else "Bank Marketing"
        rows.append([name, f"{int(r.n):,}", f"{int(r.dup_rows):,} ({r.dup_rows / r.n:.1%})",
                     f"{int(r.conflict_groups):,}", f"{r.ks_mean:.4f} / {r.ks_max:.4f}",
                     f"{r.tv_mean:.4f}", f"{r.shift_auc:.3f}", f"{r.overlap_rate:.1%}", f"{r.knn_error:.3f}"])
    return b.md_table(["数据集", "样本数", "完全重复行", "标签冲突组", "KS 均值/最大", "TV 均值",
                       "两样本 AUC", "少数类重叠率", "5-NN 误差"], rows)


def table_missing():
    df = pd.read_csv(T / "table_missing_patterns.csv")
    rows = [[str(r.missing_pattern).replace("|", " + "), f"{int(r.n):,}", f"{r.share:.1%}", f"{r.positive_rate:.1%}"]
            for _, r in df.iterrows()]
    return b.md_table(["缺失模式", "样本数", "占比", "正类率"], rows)


def table_assoc():
    df = pd.read_csv(T / "table_association.csv")
    rows = [[m.upper(), f"{r.mean_abs_diff:.3f}", f"{r.fro_rel_err:.3f}", f"{r.max_abs_diff:.3f}"]
            for m, r in zip(df.model, df.itertuples())]
    return b.md_table(["生成器", "平均绝对差", "相对 Frobenius 误差", "最大绝对差"], rows)


def table_budget_naive(naive: pd.DataFrame, rec: pd.DataFrame):
    rows = []
    for model in ["ddpm", "vae", "gan"]:
        for budget in [0.10, 0.25, 0.50, 1.00]:
            g = naive[(naive.model == model) & (naive.budget == budget)]
            base = g[g.frac == 0].auc.mean()
            aug = g[g.frac == 1.0].auc.mean()
            r = rec[(rec.model == model) & (rec.budget == budget)].recovery.mean() if budget < 1 else np.nan
            rows.append([model.upper(), f"{int(budget * 100)}%", f"{base:.4f}", f"{aug:.4f}",
                         f"{aug - base:+.4f}", f"{r:.1%}" if np.isfinite(r) else "—"])
    return b.md_table(["生成器", "真实标签预算", "不增广 AUC", "1× 剂量 AUC", "ΔAUC", "标签差距恢复率"], rows)


def table_budget_ctrl(ctrl: pd.DataFrame):
    rows = []
    for (model, budget), g in ctrl.groupby(["model", "budget"]):
        base = g[g.frac == 0].auc.mean()
        aug = g[g.frac == 1.0].auc.mean()
        rows.append([model.upper(), f"{int(budget * 100)}%", f"{base:.4f}", f"{aug:.4f}", f"{aug - base:+.4f}"])
    return b.md_table(["生成器", "真实标签预算", "不增广 AUC", "1× 剂量 AUC", "ΔAUC"], rows)


def table_leak(leak: pd.DataFrame):
    rows = []
    for _, r in leak.iterrows():
        if not np.isfinite(r.controlled):
            continue
        rows.append([r.model.upper(), f"{int(r.budget * 100)}%", f"{r.naive:+.4f}",
                     f"{r.controlled:+.4f}", f"{r.leakage_gap:+.4f}"])
    return b.md_table(["生成器", "标签预算", "朴素 ΔAUC", "泄漏对照 ΔAUC", "标签访问效应"], rows)


def table_subgroup(df: pd.DataFrame):
    rows = []
    for (g, val), sub in df.groupby(["group", "value"]):
        s = sub.set_index("strategy")
        if "none" not in s.index or "ddpm_1100" not in s.index:
            continue
        rows.append([{"sex": "性别", "race_group": "种族", "age_bin": "年龄"}[g], str(val),
                     f"{s.loc['none', 'auc']:.3f}", f"{s.loc['ddpm_1100', 'auc']:.3f}",
                     f"{s.loc['ddpm_1100', 'auc'] - s.loc['none', 'auc']:+.3f}",
                     f"{int(s.loc['none', 'n_pos'])}"])
    return b.md_table(["维度", "子群体", "不增广 AUC", "DDPM 增广 AUC", "ΔAUC", "测试集正类数"], rows)


def table_label_eq():
    a = pd.read_csv(T / "table_label_equivalence_agg.csv")
    a = a[a.model == "ddpm"]
    n = a[a.protocol == "naive"].set_index("budget")
    c = a[a.protocol == "controlled"].set_index("budget")
    rows = []
    for budget in [0.10, 0.25, 0.50]:
        if budget not in n.index or budget not in c.index:
            continue
        rows.append([f"{int(budget * 100)}%", f"{n.loc[budget, 'auc']:.4f}", f"{n.loc[budget, 'eq']:.1%}",
                     f"{c.loc[budget, 'auc']:.4f}", f"{c.loc[budget, 'eq']:.1%}",
                     f"{(n.loc[budget, 'eq'] - budget) * 100:+.1f}", f"{(c.loc[budget, 'eq'] - budget) * 100:+.1f}"])
    return b.md_table(["真实标签预算", "朴素 AUC", "朴素等效标签", "对照 AUC", "对照等效标签",
                       "朴素节省（百分点）", "对照节省（百分点）"], rows)


def table_bank_leak():
    df = pd.read_csv(T / "table_bank_leakage.csv")
    rows = [[f"{int(r.budget * 100)}%", f"{r.naive:+.4f}", f"{r.controlled:+.4f}", f"{r.gap:+.4f}"]
            for _, r in df.iterrows()]
    return b.md_table(["真实标签预算", "朴素 ΔAUC", "泄漏对照 ΔAUC", "标签访问效应"], rows)


CAPTIONS = {
    "TABLE_DATASETS": "表 1. 数据集概况",
    "TABLE_MI": "表 2. 互信息最高的 5 个特征",
    "TABLE_AUDIT": "表 3. 数据层审计：重复、标签冲突、协变量漂移与类别重叠",
    "TABLE_MISSING": "表 4. Adult 的缺失模式与正类率",
    "TABLE_ASSOC": "表 5. 合成数据对类别特征关联结构的保持程度",
    "TABLE_IMBALANCE": "表 6. 不平衡设置与数据规模",
    "TABLE_HPARAM": "表 7. 生成器与下游模型的超参数",
    "TABLE_FIDELITY": "表 8. Adult 5% 下的分布保真度、多样性与隐私（均值 ± 标准差，5 个种子）",
    "TABLE_UTILITY": "表 9. Adult 5% 下的下游分类性能（HistGradientBoosting，5 个种子均值）",
    "TABLE_BUDGET_NAIVE": "表 10. 朴素标签预算协议：生成器用全部标签训练",
    "TABLE_BUDGET_CTRL": "表 11. 泄漏对照协议：生成器只用当前预算内的标签",
    "TABLE_LEAK": "表 12. 标签访问效应：朴素协议与对照协议的 ΔAUC 之差",
    "TABLE_LABEL_EQ": "表 12a. 等效真实标签：把 AUC 换算成“省了多少标签”（Adult，DDPM）",
    "TABLE_BANK_LEAK": "表 12b. Bank 复现：朴素增益与泄漏对照（DDPM）",
    "TABLE_TSTR": "表 13. TSTR：仅用合成少数类训练、在真实测试集上评估",
    "TABLE_SELECTION": "表 14. 不同筛选方式的保真度（Adult 5%）",
    "TABLE_META": "表 15. 生成器指标与下游收益的 Spearman 相关（n=15）",
    "TABLE_STEPS": "表 16. DDIM 采样步数与保真度",
    "TABLE_SUBGROUP": "表 17. 25% 标签下不同子群体的 AUC",
    "TABLE_APPENDIX_LR": "表 A1. 逻辑回归上的完整下游结果（Adult 5%）",
    "TABLE_APPENDIX_OTHER": "表 A2. Adult 10% 与 Bank 的 AUC / PR-AUC",
}


def main():
    tpl = (ROOT / "report" / "report_template_v2.md").read_text(encoding="utf-8")
    fid = b.load("table_fidelity.csv")
    f5 = fid[fid.tag == "adult_r05"].set_index("model")
    util = b.add_tag(b.load("table_utility.csv"))
    u5 = util[(util.tag == "adult_r05") & (util.classifier == "hgb")].set_index("strategy")
    delta = b.load("table_delta_utility.csv")
    d5 = delta[(delta.tag == "adult_r05") & (delta.classifier == "hgb")].set_index("strategy")
    tstr = b.load("table_tstr.csv")
    ts5 = tstr[tstr.tag == "adult_r05"]
    meta = b.load("table_meta_correlation.csv")
    sel = b.load("table_selection_fidelity.csv")
    sel5 = sel[sel.tag == "adult_r05"]
    eda = b.load("eda_summary.csv")
    audit = b.load("table_data_audit.csv").set_index("dataset")
    assoc = b.load("table_association.csv").set_index("model")
    naive = add_delta(b.load("table_label_budget.csv"))
    ctrl = add_delta(b.load("table_label_budget_controlled.csv"))
    rec = b.load("table_recovery.csv")
    leak = b.load("table_leakage_gap.csv")
    leak = leak[leak.frac == 1.0]
    sg = b.load("table_subgroup_agg.csv")

    data5 = b.prepare_dataset("adult", 0.05, seed=0)
    n_syn = b.needed_synthetic_count(data5)
    n_real = int((data5.train.enc.y == 1).sum())
    auc_none = float(u5.loc["none", "auc_mean"])

    def naive_delta(budget, model="ddpm"):
        g = naive[(naive.model == model) & (naive.budget == budget)]
        return float((g[g.frac == 1.0].auc.mean() - g[g.frac == 0].auc.mean()))

    def ctrl_delta(budget, model="ddpm"):
        g = ctrl[(ctrl.model == model) & (ctrl.budget == budget)]
        if g.empty:
            return float("nan")
        return float((g[g.frac == 1.0].auc.mean() - g[g.frac == 0].auc.mean()))

    def fval(budget, model="ddpm"):
        g = naive[(naive.model == model) & (naive.budget == budget)]
        return float(g[g.frac == 1.0].auc.mean())

    ks10 = []
    for p in glob.glob(str(ROOT / "results" / "runs" / "adult_r05_b10" / "ddpm_s*" / "fidelity.json")):
        ks10.append(json.load(open(p, encoding="utf-8"))["ks_mean"])

    def sgval(group, value, strategy):
        r = sg[(sg.group == group) & (sg.value == value) & (sg.strategy == strategy)]
        return float(r.auc.iloc[0]) if len(r) else np.nan

    eq = b.load("table_label_equivalence_agg.csv")
    bank = b.load("table_bank_leakage.csv")

    def eqv(proto, budget):
        r = eq[(eq.protocol == proto) & (eq.model == "ddpm") & (eq.budget == budget)]
        return float(r["eq"].iloc[0]) if len(r) else np.nan

    def bankv(budget, col):
        r = bank[bank.budget == budget]
        return float(r[col].iloc[0]) if len(r) else np.nan

    vals = {
        "TABLE_DATASETS": b.table_datasets(),
        "TABLE_MI": b.table_mi(),
        "TABLE_AUDIT": table_audit(),
        "TABLE_MISSING": table_missing(),
        "TABLE_ASSOC": table_assoc(),
        "TABLE_IMBALANCE": b.table_imbalance(),
        "TABLE_HPARAM": b.table_hparam(),
        "TABLE_FIDELITY": b.table_fidelity(),
        "TABLE_UTILITY": b.table_utility(),
        "TABLE_BUDGET_NAIVE": table_budget_naive(naive, rec),
        "TABLE_BUDGET_CTRL": table_budget_ctrl(ctrl),
        "TABLE_LEAK": table_leak(leak),
        "TABLE_LABEL_EQ": table_label_eq(),
        "TABLE_BANK_LEAK": table_bank_leak(),
        "TABLE_TSTR": b.table_tstr(),
        "TABLE_SELECTION": b.table_selection(),
        "TABLE_META": b.table_meta(),
        "TABLE_STEPS": b.table_steps(),
        "TABLE_SUBGROUP": table_subgroup(sg),
        "TABLE_APPENDIX_LR": b.table_utility("adult_r05", "lr"),
        "TABLE_APPENDIX_OTHER": b.table_appendix_other(),
        "MISSING_CELLS": f"{int(eda[eda.dataset == 'adult'].missing_cells.iloc[0]):,}",
        "DUP_ROWS": f"{int(audit.loc['adult', 'dup_rows']):,}",
        "DUP_PCT": f"{audit.loc['adult', 'dup_rows'] / audit.loc['adult', 'n']:.1%}",
        "CONFLICT": f"{int(audit.loc['adult', 'conflict_groups']):,}",
        "SHIFT_KS": f"{audit.loc['adult', 'ks_mean']:.4f}",
        "SHIFT_AUC": f"{audit.loc['adult', 'shift_auc']:.3f}",
        "OVERLAP": f"{audit.loc['adult', 'overlap_rate']:.1%}",
        "KNN_ERR": f"{audit.loc['adult', 'knn_error']:.3f}",
        "ASSOC_DDPM": f"{assoc.loc['ddpm', 'fro_rel_err']:.2f}",
        "ASSOC_VAE": f"{assoc.loc['vae', 'fro_rel_err']:.2f}",
        "ASSOC_GAN": f"{assoc.loc['gan', 'fro_rel_err']:.2f}",
        "N_SYN_ADULT5": f"{n_syn:,}",
        "SYN_RATIO_ADULT5": f"{n_syn / n_real:.1f}",
        "FID_DDPM_KS": b.fmt(f5.loc["ddpm", "ks_mean"]),
        "FID_DDPM_TV": b.fmt(f5.loc["ddpm", "tv_mean"]),
        "FID_DDPM_MMD": b.fmt(f5.loc["ddpm", "mmd"]),
        "FID_GAN_KS": b.fmt(f5.loc["gan", "ks_mean"]),
        "FID_GAN_STD": b.fmt(f5.loc["gan", "std_ratio"]),
        "FID_DDPM_VENDI": b.fmt(f5.loc["ddpm", "vendi"], 2),
        "FID_VAE_DCR": b.fmt(f5.loc["vae", "dcr"], 2),
        "FID_DDPM_DCR": b.fmt(f5.loc["ddpm", "dcr"], 2),
        "AUC_NONE": b.fmt(auc_none, 4),
        "AUC_ROS": b.fmt(u5.loc["ros", "auc_mean"], 4),
        "AUC_SMOTE": b.fmt(u5.loc["smote", "auc_mean"], 4),
        "AUC_DDPM_ALL": b.fmt(u5.loc["ddpm_all", "auc_mean"], 4),
        "AUC_VAE_ALL": b.fmt(u5.loc["vae_all", "auc_mean"], 4),
        "AUC_GAN_ALL": b.fmt(u5.loc["gan_all", "auc_mean"], 4),
        "DAUC_DDPM": f"{d5.loc['ddpm_all', 'd_auc_mean']:+.4f}",
        "DAUC_DDPM_LO": f"{d5.loc['ddpm_all', 'd_auc_lo']:+.4f}",
        "DAUC_DDPM_HI": f"{d5.loc['ddpm_all', 'd_auc_hi']:+.4f}",
        "NAIVE10_BASE": b.fmt(naive[(naive.model == "ddpm") & (naive.budget == 0.10) & (naive.frac == 0)].auc.mean(), 4),
        "NAIVE10_AUG": b.fmt(fval(0.10), 4),
        "NAIVE10": f"{naive_delta(0.10):+.4f}",
        "REC10": f"{rec[(rec.model == 'ddpm') & (rec.budget == 0.10)].recovery.mean():.1%}",
        "NAIVE25": f"{naive_delta(0.25):+.4f}",
        "REC25": f"{rec[(rec.model == 'ddpm') & (rec.budget == 0.25)].recovery.mean():.1%}",
        "NAIVE50": f"{naive_delta(0.50):+.4f}",
        "NAIVE100": f"{naive_delta(1.00):+.4f}",
        "CTRL10": f"{ctrl_delta(0.10):+.4f}",
        "CTRL25": f"{ctrl_delta(0.25):+.4f}",
        "CTRL50": f"{ctrl_delta(0.50):+.4f}",
        "LEAK10": f"{naive_delta(0.10) - ctrl_delta(0.10):+.4f}",
        "EQ_NAIVE10": f"{eqv('naive', 0.10):.1%}",
        "EQ_NAIVE25": f"{eqv('naive', 0.25):.1%}",
        "EQ_NAIVE50": f"{eqv('naive', 0.50):.1%}",
        "EQ_CTRL10": f"{eqv('controlled', 0.10):.1%}",
        "EQ_CTRL25": f"{eqv('controlled', 0.25):.1%}",
        "EQ_CTRL50": f"{eqv('controlled', 0.50):.1%}",
        "SAVE_NAIVE10": f"{(eqv('naive', 0.10) - 0.10) * 100:.1f} 个百分点",
        "BANK_NAIVE10": f"{bankv(0.10, 'naive'):+.4f}",
        "BANK_NAIVE25": f"{bankv(0.25, 'naive'):+.4f}",
        "BANK_NAIVE50": f"{bankv(0.50, 'naive'):+.4f}",
        "BANK_CTRL10": f"{bankv(0.10, 'controlled'):+.4f}",
        "BANK_CTRL25": f"{bankv(0.25, 'controlled'):+.4f}",
        "BANK_CTRL50": f"{bankv(0.50, 'controlled'):+.4f}",
        "BANK_KEEP10": f"{bankv(0.10, 'controlled') / bankv(0.10, 'naive'):.0%}",
        "BANK_KEEP25": f"{bankv(0.25, 'controlled') / bankv(0.25, 'naive'):.0%}",
        "BANK_KEEP50": f"{bankv(0.50, 'controlled') / bankv(0.50, 'naive'):.0%}",
        "BUDGET_KS10": b.fmt(float(np.mean(ks10))),
        "TSTR_DDPM": b.fmt(ts5[(ts5.model == "ddpm") & (ts5.classifier == "hgb")].auc.mean(), 4),
        "TSTR_VAE": b.fmt(ts5[(ts5.model == "vae") & (ts5.classifier == "hgb")].auc.mean(), 4),
        "TSTR_GAN": b.fmt(ts5[(ts5.model == "gan") & (ts5.classifier == "hgb")].auc.mean(), 4),
        "TSTR_RET": f"{ts5[(ts5.model == 'ddpm') & (ts5.classifier == 'hgb')].auc.mean() / auc_none:.1%}",
        "SEL_VAE_MMD_ALL": b.fmt(sel5[(sel5.model == "vae") & (sel5.selection == "pool")].mmd.mean()),
        "SEL_VAE_MMD_DRS": b.fmt(sel5[(sel5.model == "vae") & (sel5.selection == "drs")].mmd.mean()),
        "META_RHO_MMD_DAUC": f"{meta[(meta.metric == 'mmd') & (meta.target == 'd_auc')].spearman.iloc[0]:+.2f}",
        "META_P_MMD_DAUC": b.pfmt(meta[(meta.metric == 'mmd') & (meta.target == 'd_auc')].p.iloc[0]),
        "META_RHO_MMD_TSTR": f"{meta[(meta.metric == 'mmd') & (meta.target == 'tstr_auc')].spearman.iloc[0]:+.2f}",
        "META_P_MMD_TSTR": b.pfmt(meta[(meta.metric == 'mmd') & (meta.target == 'tstr_auc')].p.iloc[0]),
        "SUB_WHITE": f"{sgval('race_group', 'White', 'ddpm_1100') - sgval('race_group', 'White', 'none'):+.3f}",
        "SUB_NONWHITE": f"{sgval('race_group', 'Non-White', 'ddpm_1100') - sgval('race_group', 'Non-White', 'none'):+.3f}",
        "SUB_FEMALE": f"{sgval('sex', 'Female', 'ddpm_1100') - sgval('sex', 'Female', 'none'):+.3f}",
        "SUB_MALE": f"{sgval('sex', 'Male', 'ddpm_1100') - sgval('sex', 'Male', 'none'):+.3f}",
    }
    for k, v in vals.items():
        if k in CAPTIONS:
            tpl = tpl.replace("{{" + k + "}}", f"**{CAPTIONS[k]}**\n\n{v}")
        else:
            tpl = tpl.replace("{{" + k + "}}", str(v))
    import re
    missing = re.findall(r"\{\{([A-Z0-9_]+)\}\}", tpl)
    if missing:
        raise SystemExit(f"unfilled: {sorted(set(missing))}")
    (ROOT / "report" / "report.md").write_text(tpl, encoding="utf-8")
    print("wrote report.md", len(tpl))


if __name__ == "__main__":
    main()
