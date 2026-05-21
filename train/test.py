"""DOX 模型评估脚本.

在 ODIR-5K 测试集上评估模型，输出：

* 样本级精确匹配率 (Subset Accuracy)
* 多标签 Cohen's Kappa
* 每个类别的 Precision / Recall / F1 / AUC / Kappa
* 宏平均 / 微平均的 Precision / Recall / F1 / AUC
* 类别级 TP / FP / FN / TN 与正例率

可选输出：

* ``--pred-save-dir``: 将每张图像的 (概率, 预测, 真实标签) 落盘为 CSV
* ``--metric-json``:   将整体指标写入 JSON，便于后续画图 / 汇总

复现论文表 1 / 2 的命令::

    python -m train.test \\
        --config configs/test_default.yaml \\
        --checkpoint work_dir/dox_run1/checkpoint_best.pth \\
        --testset       /path/to/ODIR-5K/off_site_test/Images \\
        --test-csv-path /path/to/ODIR-5K/off_site_test/labels.xlsx
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.orid5k import Orid5k
from modeling.model import MODEL

# 复现性
_SEED = 2025
torch.manual_seed(_SEED)
torch.cuda.manual_seed(_SEED)
np.random.seed(_SEED)
torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================
CLASS_NAMES: list[str] = ["N", "D", "G", "C", "A", "H", "M", "O"]
NUM_CLASSES: int = len(CLASS_NAMES)

#: 论文建议的类别差异化阈值 (基于临床重要性 + 验证集网格搜索)
DEFAULT_THRESHOLDS: dict[str, float] = {
    "N": 0.7, "D": 0.7, "G": 0.5, "C": 0.7,
    "A": 0.7, "H": 0.4, "M": 0.8, "O": 0.6,
}


# ============================================================
# Data structures
# ============================================================
@dataclass
class EvalConfig:
    checkpoint:     str
    testset:        str
    test_csv_path:  str
    batch_size:     int = 32
    num_workers:    int = 4
    device:         str = "cuda"
    r:              int = 4
    adapter_dim:    int = 128
    dropout:        float = 0.1
    thresholds:     dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    pred_save_dir:  str | None = None
    metric_json:    str | None = None


@dataclass
class EvalMetrics:
    subset_accuracy:        float
    multilabel_kappa:       float
    macro_precision:        float
    macro_recall:           float
    macro_f1:               float
    macro_auc:              float
    micro_precision:        float
    micro_recall:           float
    micro_f1:               float
    micro_auc:              float
    per_class_precision:    dict[str, float]
    per_class_recall:       dict[str, float]
    per_class_f1:           dict[str, float]
    per_class_auc:          dict[str, float]
    per_class_kappa:        dict[str, float]


# ============================================================
# Helpers
# ============================================================
def build_model(cfg: EvalConfig) -> MODEL:
    """根据配置实例化模型并加载评估权重。"""
    model = MODEL(
        check_path=None,
        embed_dim=1024,
        out_chans=256,
        num_classes=NUM_CLASSES,
        r=cfg.r,
        adapter_dim=cfg.adapter_dim,
        dropout=cfg.dropout,
    )
    state = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(state, strict=True)
    logger.info("Loaded checkpoint: %s", cfg.checkpoint)
    return model


def apply_thresholds(probs: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    """按类别独立阈值把概率转 0/1 预测。"""
    preds = np.zeros_like(probs)
    for i, name in enumerate(CLASS_NAMES):
        preds[:, i] = (probs[:, i] >= thresholds[name]).astype(int)
    return preds


def multilabel_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """多标签整体 Kappa 系数。

    定义见 https://stats.stackexchange.com/questions/239942
    将每个样本是否完全匹配视作 observed agreement，并基于各标签的边际
    概率乘积估计 expected agreement (标签独立假设)。
    """
    observed = float(np.all(y_true == y_pred, axis=1).mean())

    expected = 1.0
    for i in range(y_true.shape[1]):
        p_t = y_true[:, i].mean()
        p_p = y_pred[:, i].mean()
        expected *= (1 - p_t) * (1 - p_p) + p_t * p_p

    if expected >= 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


# ============================================================
# Main evaluation
# ============================================================
@torch.no_grad()
def evaluate(cfg: EvalConfig) -> EvalMetrics:
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")

    model = build_model(cfg).to(device).eval()

    dataset = Orid5k(images_dir=cfg.testset, csv_path=cfg.test_csv_path, mode="test")
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    logger.info("Evaluating %d samples across %d batches.", len(dataset), len(loader))

    all_idx:   list[str] = []
    all_true:  list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    for batch in tqdm(loader, desc="Eval"):
        left  = batch["left_image"].to(device, non_blocking=True)
        right = batch["right_image"].to(device, non_blocking=True)
        target = batch["target"]

        B, C, H, W = left.shape
        samples = torch.empty(2 * B, C, H, W, device=device, dtype=left.dtype)
        samples[0::2] = left
        samples[1::2] = right

        logits = model(samples)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_idx.extend(batch["idx"])
        all_true.append(target.numpy())
        all_probs.append(probs)

    y_true  = np.concatenate(all_true,  axis=0)
    y_probs = np.concatenate(all_probs, axis=0)
    y_pred  = apply_thresholds(y_probs, cfg.thresholds)

    # ---- Per-class metrics ----
    per_p = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_r = recall_score   (y_true, y_pred, average=None, zero_division=0)
    per_f = f1_score       (y_true, y_pred, average=None, zero_division=0)

    per_auc, per_kappa = [], []
    for i in range(NUM_CLASSES):
        try:
            per_auc.append(roc_auc_score(y_true[:, i], y_probs[:, i]))
        except ValueError:
            per_auc.append(float("nan"))
        try:
            per_kappa.append(cohen_kappa_score(y_true[:, i], y_pred[:, i]))
        except ValueError:
            per_kappa.append(float("nan"))

    metrics = EvalMetrics(
        subset_accuracy = float(np.all(y_true == y_pred, axis=1).mean()),
        multilabel_kappa= multilabel_kappa(y_true, y_pred),
        macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        macro_recall    = float(recall_score   (y_true, y_pred, average="macro", zero_division=0)),
        macro_f1        = float(f1_score       (y_true, y_pred, average="macro", zero_division=0)),
        macro_auc       = float(np.nanmean(per_auc)),
        micro_precision = float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        micro_recall    = float(recall_score   (y_true, y_pred, average="micro", zero_division=0)),
        micro_f1        = float(f1_score       (y_true, y_pred, average="micro", zero_division=0)),
        micro_auc       = float(roc_auc_score(y_true.ravel(), y_probs.ravel(), average="micro")),
        per_class_precision = {n: float(v) for n, v in zip(CLASS_NAMES, per_p)},
        per_class_recall    = {n: float(v) for n, v in zip(CLASS_NAMES, per_r)},
        per_class_f1        = {n: float(v) for n, v in zip(CLASS_NAMES, per_f)},
        per_class_auc       = {n: float(v) for n, v in zip(CLASS_NAMES, per_auc)},
        per_class_kappa     = {n: float(v) for n, v in zip(CLASS_NAMES, per_kappa)},
    )

    _print_metrics(metrics, y_true, y_pred)

    # ---- Optional artifacts ----
    if cfg.pred_save_dir:
        _save_predictions(cfg.pred_save_dir, all_idx, y_true, y_probs, y_pred)
    if cfg.metric_json:
        _save_metric_json(cfg.metric_json, metrics)

    return metrics


# ============================================================
# Reporting & persistence
# ============================================================
def _print_metrics(m: EvalMetrics, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    bar = "=" * 80
    print(bar)
    print(" 模型评估结果 ".center(80, "="))
    print(bar)
    print(f"\n样本级精确匹配率 : {m.subset_accuracy:.4f}")
    print(f"多标签 Kappa     : {m.multilabel_kappa:.4f}")

    print("\n宏平均指标 (Macro):")
    print(f"  Precision : {m.macro_precision:.4f}")
    print(f"  Recall    : {m.macro_recall:.4f}")
    print(f"  F1-score  : {m.macro_f1:.4f}")
    print(f"  AUC       : {m.macro_auc:.4f}")

    print("\n微平均指标 (Micro):")
    print(f"  Precision : {m.micro_precision:.4f}")
    print(f"  Recall    : {m.micro_recall:.4f}")
    print(f"  F1-score  : {m.micro_f1:.4f}")
    print(f"  AUC       : {m.micro_auc:.4f}")

    print("\n各类别详细指标:")
    print("-" * 80)
    print(f"{'类别':<6} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12} {'Kappa':<12}")
    print("-" * 80)
    for name in CLASS_NAMES:
        print(f"{name:<6} "
              f"{m.per_class_precision[name]:<12.4f} "
              f"{m.per_class_recall   [name]:<12.4f} "
              f"{m.per_class_f1       [name]:<12.4f} "
              f"{m.per_class_auc      [name]:<12.4f} "
              f"{m.per_class_kappa    [name]:<12.4f}")

    print("\n各类别 TP / FP / FN / TN / 正例率:")
    print("-" * 60)
    for i, name in enumerate(CLASS_NAMES):
        tp = int(((y_true[:, i] == 1) & (y_pred[:, i] == 1)).sum())
        fp = int(((y_true[:, i] == 0) & (y_pred[:, i] == 1)).sum())
        fn = int(((y_true[:, i] == 1) & (y_pred[:, i] == 0)).sum())
        tn = int(((y_true[:, i] == 0) & (y_pred[:, i] == 0)).sum())
        prevalence = float(y_true[:, i].mean())
        print(f"  {name}: TP={tp:4d}  FP={fp:4d}  FN={fn:4d}  TN={tn:4d}  正例率={prevalence:.3f}")

    print("\n" + bar)
    print(" 评估完成 ".center(80, "="))
    print(bar)


def _save_predictions(
    save_dir: str,
    sample_ids: list[str],
    y_true:  np.ndarray,
    y_probs: np.ndarray,
    y_pred:  np.ndarray,
) -> None:
    save_dir_p = Path(save_dir)
    save_dir_p.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "sample_id": sample_ids,
        **{f"prob_{c}":  y_probs[:, i] for i, c in enumerate(CLASS_NAMES)},
        **{f"pred_{c}":  y_pred [:, i] for i, c in enumerate(CLASS_NAMES)},
        **{f"true_{c}":  y_true [:, i] for i, c in enumerate(CLASS_NAMES)},
    })
    out_csv = save_dir_p / "predictions.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Predictions saved to %s", out_csv)


def _save_metric_json(path: str, m: EvalMetrics) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(m), f, indent=2, ensure_ascii=False)
    logger.info("Metrics saved to %s", path)


# ============================================================
# CLI
# ============================================================
def _parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="DOX evaluation script.")
    parser.add_argument("--config",         type=str, default=None,
                        help="YAML 配置文件路径；命令行参数会覆盖 YAML 配置。")
    parser.add_argument("--checkpoint",     type=str, required=False)
    parser.add_argument("--testset",        type=str, required=False)
    parser.add_argument("--test-csv-path",  type=str, required=False)
    parser.add_argument("--batch-size",     type=int, default=32)
    parser.add_argument("--num-workers",    type=int, default=4)
    parser.add_argument("--device",         type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("-r",               type=int, default=4,   help="LoRA rank")
    parser.add_argument("--adapter-dim",    type=int, default=128, help="Conv-Adapter 维度")
    parser.add_argument("--dropout",        type=float, default=0.1)
    parser.add_argument("--pred-save-dir",  type=str, default=None,
                        help="若指定，将逐样本预测写入该目录的 predictions.csv")
    parser.add_argument("--metric-json",    type=str, default=None,
                        help="若指定，将整体指标写入该 JSON 文件")
    args = parser.parse_args()

    cfg_dict: dict = {}
    if args.config:
        import yaml  # 延迟导入以避免硬依赖
        with open(args.config, "r", encoding="utf-8") as f:
            cfg_dict.update(yaml.safe_load(f) or {})

    for k, v in vars(args).items():
        if k == "config" or v is None:
            continue
        cfg_dict[k.replace("-", "_")] = v

    required = ("checkpoint", "testset", "test_csv_path")
    missing = [k for k in required if not cfg_dict.get(k)]
    if missing:
        parser.error(f"Missing required arguments: {missing} (set via CLI or YAML config)")

    return EvalConfig(**cfg_dict)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = _parse_args()
    evaluate(cfg)


if __name__ == "__main__":
    main()
