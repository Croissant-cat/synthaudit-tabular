# SynthAudit

**生成式表格数据增强的效用审计：VAE、GAN 与 DDPM 的受控比较**

本项目研究一个实际问题：当少数类标签稀缺时，用生成模型合成少数类样本，能否提升下游表格分类？
我们在同一混合类型表示、同一 MLP 容量和同一类别条件下，从头实现了 VAE、WGAN-GP 与 DDPM，
并从**保真度、多样性、隐私、下游效用**四个层面做受控比较，另提出**密度比筛选（DRS）**并做剂量-响应、
TSTR 与元分析。代码、图表、结果表和报告均可复现。

**报告**：[report/SynthAudit_报告.pdf](report/SynthAudit_报告.pdf)（DOCX 同目录）

---

## 主要发现

1. **DDPM 保真度最好，GAN 最差。** Adult 5% 设置下，DDPM 的逐特征 KS 为 0.089、TV 为 0.045、MMD 为 0.064；
   VAE 次之（0.119 / 0.047 / 0.178）；GAN 最差（0.173 / 0.133 / 0.288），且数值标准差仅为真实数据的 0.42 倍，
   出现明显方差收缩。
2. **在强增广下，增广没有提升 AUC。** 把 Adult 5% 的少数类补到 20%（合成量约为真实少数类的 3.8 倍）后，
   HistGradientBoosting 的 AUC 从 0.9065 降到 DDPM 0.9034、VAE 0.8903、GAN 0.8928；
   5 个种子的配对差值方向一致（Wilcoxon 最小可达 p=0.0625，因此以置信区间为准）。
3. **剂量-响应非单调。** DDPM 在 1/8 剂量处有小幅正收益（+0.0025），1/4 剂量处与基线持平（-0.0002），
   1 倍剂量处转为负值（-0.0031）。Bank 自然不平衡下（0.89 倍剂量）DDPM 反而提升 +0.0046。
4. **TSTR：合成数据可以替代缺失的少数类标签。** 只用合成少数类 + 真实多数类训练，DDPM 在真实测试集上
   达到 AUC 0.894，约为真实少数类基线（0.9065）的 98.6%；VAE 为 0.837，GAN 为 0.736。
5. **DRS 改善保真度，但不改善效用。** DRS 把 VAE 的 MMD 从 0.177 降到 0.153、KS 从 0.117 降到 0.111，
   但下游 AUC 仅有 +0.0012 的变化（噪声范围内），三种生成器均无实质性提升。
6. **保真度是"风险指标"，不是"收益指标"。** 跨生成器看，MMD 与 ΔAUC 的 Spearman 相关为 -0.95，
   但这一相关由 GAN 的差保真与大幅下降驱动；保真度最高的 DDPM 仍略低于不增广。

## 项目结构

```
src/
  data.py        数据层：加载、清洗、分位数变换 + 独热编码、分层划分、嵌套不平衡构造
  models.py      VAE / WGAN-GP / DDPM（共享 MLP 骨干、标签嵌入、x0/噪声混合参数化）
  train.py       训练循环、EMA、DDIM 采样、合成样本导出
  metrics.py     KS / TV / 相关误差 / MMD / 覆盖率 / Vendi / DCR / AUC / AP / F1 / ECE
  selection.py   密度比筛选 DRS（PCA-10 + kNN 密度比 + 无放回抽样）
  evaluate.py    下游协议、SMOTE/ROS 基线、TSTR
scripts/
  download_data.py   下载 UCI Adult 与 Bank Marketing
  run_eda.py         探索性分析与图 1-5
  run_generation.py  训练三种生成器并保存合成样本
  run_evaluation.py  下游效用实验
  run_analysis.py    聚合、统计检验、剂量/TSTR/元分析、图 6-11
  build_report.py    用实测表格填充报告模板
  build_docx.py      Markdown -> DOCX
tests/test_smoke.py  数据/模型/指标冒烟测试
report/              报告（md / docx / pdf）与模板
results/
  figures/           图 1-16
  tables/            所有结果表（CSV）
  metrics/           每次下游评估的原始结果
  runs/              每次生成器训练的 fidelity / train_log（模型与样本不入库）
```

## 快速开始

```bash
pip install -r requirements.txt          # Python 3.10+, PyTorch 2.x
python scripts/download_data.py          # 约 1.7 MB
python scripts/run_generation.py         # 5 + 2 + 3 个种子，约 35 分钟（GTX 1650 Ti）
python scripts/run_evaluation.py         # 下游评估，约 8 分钟
python scripts/run_analysis.py           # 统计与图 6-11
python scripts/build_report.py           # 生成 report/report.md
python scripts/build_docx.py             # 需要 python-docx
```

冒烟测试：

```bash
python tests/test_smoke.py
```

## 实验矩阵

| 设置 | 少数类比例 | 种子数 | 测试集正类率 |
|---|---|---|---|
| Adult 5% | 5.0% | 5 | 5.0% |
| Adult 10% | 10.0% | 2 | 10.0% |
| Bank 自然 | 11.7% | 3 | 11.7% |

每个（设置，种子）训练 VAE / GAN / DDPM 各一个，比较 9 种增广策略：
不增广、随机过采样、SMOTE、以及三种生成器 × {全部、随机子集、DRS 子集}。
下游分类器：HistGradientBoosting 与 LogisticRegression（均 `class_weight='balanced'`）。

## 复现与诚实性说明

- 所有随机种子、数据划分与超参数固定在代码中；`results/runs/*/train_log.json` 记录了每次训练的耗时与损失。
- 统计上，Adult 5% 只有 5 个种子，Wilcoxon 的最小可达双侧 p 值为 0.0625；报告中同时给出 95% bootstrap 置信区间与效应量。
- GAN 与 VAE/DDPM 使用相同训练轮数，WGAN-GP 收敛更慢，其较差表现可能部分来自训练预算；这一点在报告的局限中明确说明。
- 未使用任何测试集信息做模型选择；阈值只在验证集上选取。

## 引用

如果使用本项目的代码或结果，请引用报告与三篇基础论文：

- Kingma & Welling. Auto-Encoding Variational Bayes. ICLR 2014.
- Goodfellow et al. Generative Adversarial Nets. NeurIPS 2014.
- Ho, Jain & Abbeel. Denoising Diffusion Probabilistic Models. NeurIPS 2020.

## License

MIT
