# SynthAudit

**生成式表格数据增强的标签泄漏审计：VAE、GAN 与 DDPM 的效用边界**

本项目研究：当少数类标签稀缺时，生成模型合成的少数类样本，能否替代真实标签？
我们在同一混合类型表示、同一 MLP 容量和同一类别条件下，从头实现 VAE、WGAN-GP 与 DDPM，
并设计**三级审计协议**：

1. 全标签增广（生成器与下游都用完整标签）；
2. 朴素标签预算（生成器用完整标签，下游只用 10%–100%）；
3. **泄漏对照**（生成器只用下游可见的同一批标签重训）。

核心发现：低标签预算下的"增广红利"主要来自生成器对 withheld labels 的访问，而不是合成样本本身；
一旦做泄漏对照，增益消失。

**报告**：[report/SynthAudit_报告.pdf](report/SynthAudit_报告.pdf)（DOCX 同目录）

---

## 主要发现

1. **DDPM 保真度最好，GAN 最差。** Adult 5% 设置下，DDPM 的逐特征 KS 为 0.089、TV 为 0.045、MMD 为 0.064；
   VAE 次之（0.119 / 0.047 / 0.178）；GAN 最差（0.173 / 0.133 / 0.288），数值标准差仅为真实数据的 0.42 倍。
2. **全标签下强增广无益。** 把 Adult 5% 补到 20%（合成量约为真实少数类的 3.8 倍）后，AUC 从 0.9065 降到
   DDPM 0.9034、GAN 0.8928、VAE 0.8903；5 个种子方向一致（Wilcoxon 最小可达 p=0.0625，故以置信区间为准）。
3. **朴素标签预算给出"看似有效"的替代效应。** 生成器用全部标签训练、下游只用 10% 标签时，
   DDPM 增广把 AUC 从 0.8615 提到 0.8954（+0.0339，恢复 74.6% 的标签差距）；25% 标签下 +0.0116（恢复 59.9%）。
4. **泄漏对照大幅削弱上述结论。** Adult 上，生成器只用同样的 10% 标签重训后，DDPM 的 10%/25%/50% ΔAUC 为
   **−0.0078 / −0.0066 / −0.0063**；VAE 为 −0.0282 / −0.0249 / −0.0214。**GAN 是反例**：10% 标签下朴素
   −0.0183，对照后反而 **+0.0176**（5 个种子一致），25%/50% 为 −0.0148 / −0.0174。Bank 上，10% 标签的
   朴素增益 +0.0662 在对照后仍保留 **+0.0231**（约 35%），25% 保留 +0.0122。标签访问会系统性改变估计，
   方向取决于模型与预算——既可能高估，也可能低估。
5. **等效标签。** 把 AUC 换算成真实标签：朴素 10% + DDPM 相当于 **47.8%** 真实标签（看似省 37.8 个百分点）；
   泄漏对照后仅 **2.7%**。25% 预算：朴素 52.2% vs 对照 23.1%。
6. **TSTR 的前提。** 仅用合成少数类 + 真实多数类训练，DDPM 达到 AUC 0.894（真实标签基线的 98.6%）；
   但生成器本身是用全部少数类标签训练的，它衡量的是**标签搬运**，不是零标签合成。
7. **数据层审计。** Adult 在 12 维特征下有 6,374 条完全重复（13.1%）、1,029 组标签冲突；训练/测试两样本
   AUC 0.496（无明显漂移）；38.8% 的少数类测试点与多数类高度重叠；5-NN 误差 0.167。
   关联结构保持：Cramér's V 相对误差 DDPM 0.32、VAE 0.39、GAN 0.86。
8. **DRS 是负结果。** 密度比筛选把 VAE 的 MMD 从 0.177 降到 0.153，但下游 AUC 仅 +0.0012，无实质收益。

## 项目结构

```
src/
  data.py        数据层：加载、清洗、分位数变换 + 独热编码、分层划分、嵌套不平衡构造
  models.py      VAE / WGAN-GP / DDPM（共享 MLP 骨干、标签嵌入、混合参数化）
  train.py       训练循环、EMA、DDIM 采样、合成样本导出
  metrics.py     KS / TV / 相关误差 / MMD / 覆盖率 / Vendi / DCR / AUC / AP / F1 / ECE
  selection.py   密度比筛选 DRS（PCA-10 + kNN 密度比 + 无放回抽样）
  evaluate.py    下游协议、SMOTE/ROS 基线、TSTR
scripts/
  download_data.py            下载 UCI Adult 与 Bank Marketing
  run_eda.py                  探索性分析与图 1–5
  run_data_audit.py           重复/冲突/漂移/重叠/缺失/关联结构审计
  run_generation.py           全标签生成器（5 + 2 + 3 个种子）
  run_generation_budget.py    **泄漏对照**：只用预算内标签重训生成器
  run_evaluation.py           全标签下游评估
  run_label_budget.py         朴素标签预算（生成器见全部标签）
  run_label_budget_controlled.py  泄漏对照下游评估
  run_subgroup.py             25% 标签下的子群体分析
  run_analysis.py             聚合、统计检验、图 6–16
  run_deep_analysis.py        标签预算/恢复率/子群体图
  run_leakage_analysis.py     朴素 vs 对照的泄漏差与图 22
  build_report_v2.py          用实测表格填充 v2 报告
  build_docx.py               Markdown -> DOCX
tests/test_smoke.py           数据/模型/指标冒烟测试
report/                       报告（md / docx / pdf）与模板
results/figures/              所有图
results/tables/               所有结果表（CSV）
results/runs/                 每次生成器训练的 fidelity / train_log（模型与样本不入库）
```

## 快速开始

```bash
pip install -r requirements.txt          # Python 3.10+, PyTorch 2.x
python scripts/download_data.py          # 约 1.7 MB
python scripts/run_generation.py         # 全标签生成器，约 35 分钟（GTX 1650 Ti）
python scripts/run_evaluation.py         # 全标签下游评估，约 8 分钟
python scripts/run_generation_budget.py --budgets 0.1 0.25 0.5 --models ddpm --seeds 0 1 2 3 4
python scripts/run_generation_budget.py --budgets 0.25 --models vae gan --seeds 0 1 2 3 4
python scripts/run_label_budget.py       # 朴素标签预算，约 8 分钟
python scripts/run_label_budget_controlled.py  # 泄漏对照，约 3 分钟
python scripts/run_data_audit.py         # 数据审计，约 1 分钟
python scripts/run_subgroup.py           # 子群体，约 1 分钟
python scripts/run_analysis.py
python scripts/run_deep_analysis.py
python scripts/run_leakage_analysis.py
python scripts/build_report_v2.py        # 生成 report/report.md
python scripts/build_docx.py             # 需要 python-docx
```

冒烟测试：`python tests/test_smoke.py`

## 实验矩阵

| 设置 | 少数类比例 | 全标签种子 | 标签预算 |
|---|---|---|---|
| Adult 5% | 5.0% | 5 | 10% / 25% / 50% / 100% |
| Adult 10% | 10.0% | 2 | — |
| Bank 自然 | 11.7% | 3 | — |

全标签比较 9 种策略：不增广、随机过采样、SMOTE、三种生成器 × {全部、随机子集、DRS 子集}。
标签预算比较 5 档剂量（0、1/8、1/4、1/2、1）；泄漏对照覆盖 DDPM 的 10%/25%/50% 与 VAE/GAN 的 25%。
下游：HistGradientBoosting 与 LogisticRegression（均 `class_weight='balanced'`）。

## 复现与诚实性说明

- 所有随机种子、数据划分与超参数固定在代码中；`results/runs/*/train_log.json` 记录每次训练耗时与损失。
- Adult 5% 只有 5 个种子，Wilcoxon 最小可达双侧 p=0.0625；报告同时给出 95% bootstrap 置信区间与效应量。
- 泄漏对照是本文最重要的验证：朴素协议与对照协议的生成器**必须**用同一批标签训练，否则低预算增益会被严重高估。
- GAN 与 VAE/DDPM 使用相同训练轮数，WGAN-GP 收敛更慢，其较差表现可能部分来自训练预算。
- 未使用测试集做模型选择；阈值只在验证集上选取。

## 引用

- Kingma & Welling. Auto-Encoding Variational Bayes. ICLR 2014.
- Goodfellow et al. Generative Adversarial Nets. NeurIPS 2014.
- Ho, Jain & Abbeel. Denoising Diffusion Probabilistic Models. NeurIPS 2020.

## License

MIT
