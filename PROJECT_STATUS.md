# 项目状态

更新时间：2026-11-23（v3 完成，准备上传）
状态：**v3 实验与报告完成；正在上传 GitHub（替换 v1）。**

## 1. 核心结论

- **全标签增广无益**：Adult 5%，AUC 0.9065 → DDPM 0.9034、GAN 0.8928、VAE 0.8903。
- **朴素标签预算看似有效**：Adult 10% + DDPM，0.8615 → 0.8954（+0.0339，等效 47.8% 真实标签）。
- **Adult 完整 3×3 泄漏对照**：
  - DDPM：−0.0078 / −0.0066 / −0.0063（10/25/50%）
  - VAE：−0.0282 / −0.0249 / −0.0214
  - GAN：**+0.0176** / −0.0148 / −0.0174
  - 唯一稳定正效应：GAN + 10% 标签；标签访问会双向失真（DDPM 高估、GAN 10% 低估）。
- **Bank 复现（DDPM）**：10% 朴素 +0.0662 → 对照 +0.0231（保留 35%）；25% +0.0229 → +0.0122；
  50% +0.0065 → +0.0048。Bank 25% 下 VAE/GAN 对照为 −0.0482 / −0.0086，残差为 DDPM 特有。
- **等效标签（Adult DDPM）**：朴素 10% ≈ 47.8% 真实标签；对照后 2.7%。25%：52.2% vs 23.1%。
- **保真度≠效用**：10% 预算生成器 KS 0.098；GAN 反例说明弱生成器噪声也可能正则化。
- **TSTR**：DDPM 0.8936（98.6%），但以生成器见过全部少数类标签为前提，衡量标签搬运。
- **数据审计**：Adult 6,374 条重复（13.1%）、1,029 组标签冲突、重叠率 38.8%、5-NN 误差 0.167；
  两样本 AUC 0.496；关联结构误差 DDPM 0.32 / VAE 0.39 / GAN 0.86。
- **子群体**：25% 标签下 White +0.014、Non-White −0.012；Female +0.017、Male +0.010（小样本，谨慎）。

## 2. 产物

- 报告（25 页，16 图、19 表）：`report/report.md`、`report/SynthAudit_报告.docx`、`.pdf`。
- 关键脚本：`run_generation_budget.py`、`run_budget_eval.py`、`run_label_budget*.py`、
  `run_bank_leakage.py`、`run_label_equivalence.py`、`run_data_audit.py`、`run_subgroup.py`、
  `run_leakage_analysis.py`、`build_report_v2.py`。
- 结果：`table_label_budget*.csv`、`table_leakage_gap.csv`、`table_bank_*.csv`、
  `table_label_equivalence*.csv`、`fig22/23/24`。

## 3. 上传说明

- `tools/gh_upload.py` 通过 GitHub REST API 上传（本机 github.com 不可达，api.github.com 可达）。
- 上传会替换仓库 v1 的 README、报告与代码；`results/runs` 只传 fidelity/train_log。
- 上传后检查：README、report/、scripts/、results/tables/ 与 figures/。
