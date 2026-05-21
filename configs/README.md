# Configs

本目录用 YAML 集中管理训练 / 评估超参，避免参数散落在 argparse 默认值与脚本各处。

## 文件清单

| 文件 | 用途 | 入口脚本 |
|---|---|---|
| `train_default.yaml` | 复现论文主实验的训练超参 | `python -m train.training --config configs/train_default.yaml` |
| `test_default.yaml`  | 论文报告指标使用的评估超参 (含类别差异化阈值) | `python -m train.test --config configs/test_default.yaml` |

## 设计原则

1. **YAML 是默认配置；命令行参数总是覆盖 YAML** —— 便于在保留默认值的同时做单次实验调整；
2. **绝对路径不入库** —— `trainset` / `train_csv_path` / `checkpoint` 等留给命令行传入，避免泄露本地路径；
3. **以 `configs/local_*.yaml` 存放个人私有配置** —— 已在 `.gitignore` 忽略。

## 新建一组配置

复制默认配置并改名即可：

```bash
cp configs/train_default.yaml configs/train_ablation_no_lora.yaml
# 然后编辑 r、adapter_dim 等进行消融
```
