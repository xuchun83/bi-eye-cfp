# `assets/` — 图像资源与生成脚本

本目录存放 README 中引用的所有图像，以及**可一键复现**它们的 Python 脚本。
所有脚本均无副作用、不依赖训练好的模型权重，只需 `matplotlib + numpy` 即可生成。

## 文件清单

| 文件 | 内容 | 数据来源 |
|---|---|---|
| `architecture.png`            | DOX 模型整体架构示意图 | 根据论文图 2-3 由 `plot_architecture.py` 重绘 |
| `training_curves.png`         | 训练 / 验证 loss + AUC + F1 曲线 | 见下方 **数据来源声明** |
| `ablation_results.png`        | LoRA / Conv-Adapter / DBFEM 三组消融柱状图 | 论文表 3-5 原始数值，**未经任何修饰** |
| `preprocessing_comparison.png` | 三阶段预处理前后对比图 *(可选)* | `datademo/raw_data/` + 本仓库 `data_processing.py` |

## 一键复现

```bash
# 在仓库根目录执行
python assets/scripts/plot_architecture.py
python assets/scripts/plot_training_curves.py
python assets/scripts/plot_ablation.py

# 可选 (需要 opencv-python + IETK-Ret)
python assets/scripts/plot_preprocessing_comparison.py
```

---

## 数据来源声明 (学术诚信)

> 我们严格区分**真实实验数据**与**示意性可视化**。

### ✅ 完全基于真实数据的图

* **`architecture.png`** — 与论文图 2 / 图 3 在结构上完全一致，仅做了 matplotlib 风格统一；
* **`ablation_results.png`** — 数值取自论文表 3-5 (ODIR-5K On-site Test)，逐项与 README 消融实验表一致；
* **`preprocessing_comparison.png`** — 真实调用本仓库 `data_processing.preprocess_pipeline()` 生成，
  输入为 `datademo/raw_data/` 真实样例图像。

### ⚠️ 示意性图 (Schematic)

* **`training_curves.png`**
  - **最终 epoch 的 AUC=84.61% / F1=53.92%** 来自论文真实结果；
  - **中间 epoch 的曲线形状** 是基于 "warmup → cosine annealing + AsymmetricLoss" 的**典型收敛模式**插值得到，
    用作 README 的视觉示意，**不**代表逐 epoch 的真实日志；
  - 真实的逐 epoch 曲线由 `train.training` 在 `${work_dir}/train_loss.png` 自动产出，
    待训练复现完成后会替换此图。

如对此有疑问或希望我们补充真实日志，欢迎在 Issues 联系。
