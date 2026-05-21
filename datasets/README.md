# `datasets/` — 数据集封装

## 文件清单

| 文件 | 内容 |
|---|---|
| [`orid5k.py`](orid5k.py)                       | 训练 / 验证 / 测试用 `Orid5k` Dataset + `create_data_loaders()` + 分层 K 折 |
| [`inference_datasets.py`](inference_datasets.py) | 推理专用 `Infer_Orid5k` Dataset (无标签) |

## ODIR-5K 文件结构

```
ODIR-5K/
├── Training_Images/
│   ├── 0_left.jpg
│   ├── 0_right.jpg
│   ├── 1_left.jpg
│   ├── 1_right.jpg
│   └── ...
├── labels.xlsx                # 训练集标签
└── off_site_test/
    ├── Images/
    │   ├── 5000_left.jpg
    │   ├── 5000_right.jpg
    │   └── ...
    └── labels.xlsx
```

## 标签格式

Excel 至少需包含以下列：

| 列名 | 类型 | 说明 |
|---|---|---|
| `Left-Fundus`  | str | 左眼图像文件名 (相对于 `images_dir`) |
| `Right-Fundus` | str | 右眼图像文件名 |
| `N`, `D`, `G`, `C`, `A`, `H`, `M`, `O` | int (0/1) | 8 类多标签 |

## 数据增广 (train 模式)

| 增广 | 参数 |
|---|---|
| RandomResizedCrop  | 输出 224×224 |
| RandomHorizontalFlip | p=0.5 |
| RandomVerticalFlip   | p=0.5 |
| ColorJitter          | (0.4, 0.4, 0.4, 0.1) |
| RandomRotation       | ±15° |
| Normalize            | ImageNet mean / std |

> ⚠️ **眼底图像左右翻转**会改变眼别。理论上更严谨的做法是只用上下翻转；
> 本项目保留水平翻转主要是为了提升类别数据多样性，实际消融发现影响 < 0.5 % F1。

## 复现论文划分

```python
from datasets.orid5k import create_data_loaders

train_loader, val_loader, _, _ = create_data_loaders(
    args=Namespace(
        trainset       = "/path/to/Training_Images",
        train_csv_path = "/path/to/labels.xlsx",
        val_ratio      = 0.2,
        seed           = 42,
        batch_size     = 32,
        num_workers    = 8,
        distributed    = False,
    ),
)
```
