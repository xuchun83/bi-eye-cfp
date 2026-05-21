# `train/` — 训练与评估

## 文件清单

| 文件 | 入口 | 内容 |
|---|---|---|
| [`training.py`](training.py) | `python -m train.training`   | 训练循环 (单卡 / DDP)、AdamW + Warmup-Cosine、ASL Loss、自动保 best/latest |
| [`test.py`](test.py)         | `python -m train.test`       | 评估脚本 (含 per-class / macro / micro 指标 + 多标签 Kappa) |
| [`utils.py`](utils.py)       | -                            | 日志、JSON 序列化等通用工具 |

## 训练 (单卡)

```bash
python -m train.training \
  --config configs/train_default.yaml \
  --pre-checkpoint /path/to/RETFound_mae_natureCFP.pth \
  --trainset       /path/to/ODIR-5K/Training_Images \
  --train-csv-path /path/to/ODIR-5K/labels.xlsx \
  --work-dir       work_dir/dox_run1
```

## 训练 (DDP)

```bash
torchrun --nproc_per_node=4 -m train.training \
  --config configs/train_default.yaml \
  --pre-checkpoint /path/to/RETFound_mae_natureCFP.pth \
  --trainset       /path/to/ODIR-5K/Training_Images \
  --train-csv-path /path/to/ODIR-5K/labels.xlsx \
  --work-dir       work_dir/dox_run1 \
  --distributed
```

## 评估

```bash
python -m train.test \
  --config        configs/test_default.yaml \
  --checkpoint    work_dir/dox_run1/checkpoint_best.pth \
  --testset       /path/to/ODIR-5K/off_site_test/Images \
  --test-csv-path /path/to/ODIR-5K/off_site_test/labels.xlsx \
  --pred-save-dir work_dir/dox_run1/eval_preds \
  --metric-json   work_dir/dox_run1/eval_metrics.json
```

## 训练产物

每次实验在 `${work_dir}/${exp_name}/` 下生成：

```
checkpoint_best.pth        # val loss 最低的检查点
checkpoint_latest.pth      # 最新一轮检查点 (用于 --resume)
checkpoint_epoch_NNN.pth   # 若 --save-interval 指定
train.log                  # 文本日志
train_loss.png             # train / val loss 曲线
```

## 复现论文超参

* AdamW (β₁=0.9, β₂=0.999, ε=1e-8)
* lr = 1e-3, weight_decay = 0.01
* 100 epochs + 10 epoch linear warmup → cosine annealing
* gradient clip = 0.1
* AsymmetricLoss (γ⁺=0, γ⁻=4, clip=0.05)
* batch_size = 32, LoRA rank r=4, adapter_dim=128
