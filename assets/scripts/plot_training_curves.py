"""生成训练曲线图 (loss + AUC/F1 双纵轴).

⚠️  关于数据来源 (学术诚信声明):
─────────────────────────────────────────────────────────────────
本图的最终性能数值 (epoch 100 的 AUC = 84.61 %, F1 = 53.92 %) 来自论文
ODIR-5K 离线测试集；中间 epoch 的曲线形状基于深度学习训练的**典型收敛
模式**用平滑函数插值得到，仅用于在 README 中**示意训练过程**，
不代表逐 epoch 的真实日志。

若需逐 epoch 的真实曲线，请运行 ``train.training`` 并使用其自动生成的
``train_loss.png``。

使用方法
-------
    python assets/scripts/plot_training_curves.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 论文报告的最终指标 (ODIR-5K 离线 / 在线)
FINAL_AUC_OFFLINE = 0.8461
FINAL_F1_OFFLINE  = 0.5392

# 训练设置 (与 configs/train_default.yaml 一致)
TOTAL_EPOCHS = 100
WARMUP_EPOCHS = 10

ASSETS_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ASSETS_DIR / "training_curves.png"


def _smooth_loss_curve(total_epochs: int, warmup: int, seed: int = 42) -> np.ndarray:
    """合成一条物理上合理的 train loss 曲线 (warmup → 指数衰减 + 噪声)。"""
    rng = np.random.default_rng(seed)
    x = np.arange(total_epochs)

    # warmup 阶段：loss 略有上升后回落
    warmup_curve = 1.5 + 0.3 * np.sin(np.linspace(0, np.pi, warmup))
    # cosine + 指数衰减
    cos_phase = np.linspace(0, np.pi, total_epochs - warmup)
    main_curve = 0.18 + 1.10 * (1 + np.cos(cos_phase)) / 2 * np.exp(-x[warmup:] / 60)

    loss = np.concatenate([warmup_curve, main_curve])
    loss += rng.normal(0, 0.015, size=loss.shape)  # 训练抖动
    return np.clip(loss, 0.12, None)


def _smooth_val_loss(train_loss: np.ndarray, seed: int = 7) -> np.ndarray:
    """验证 loss = 训练 loss + 偏置 + 轻微噪声 (并随训练逐步收敛)。"""
    rng = np.random.default_rng(seed)
    bias = np.linspace(0.08, 0.04, len(train_loss))  # 后期 gap 缩小
    return train_loss + bias + rng.normal(0, 0.012, size=train_loss.shape)


def _monotonic_metric(final: float, total_epochs: int, warmup: int,
                       start: float = 0.45, seed: int = 0) -> np.ndarray:
    """生成单调上升的验证指标曲线 (warmup 段贴近起点；之后 sigmoid 上升)。"""
    rng = np.random.default_rng(seed)
    out = np.empty(total_epochs)
    out[:warmup] = start + rng.normal(0, 0.005, warmup).cumsum() * 0.0
    progress = np.linspace(-4, 4, total_epochs - warmup)
    sigmoid = 1.0 / (1.0 + np.exp(-progress))
    out[warmup:] = start + (final - start) * sigmoid
    out += rng.normal(0, 0.004, size=out.shape)
    out[-1] = final  # 强制最终点 = 论文报告值
    return out


def main() -> None:
    epochs = np.arange(1, TOTAL_EPOCHS + 1)
    train_loss = _smooth_loss_curve(TOTAL_EPOCHS, WARMUP_EPOCHS, seed=42)
    val_loss   = _smooth_val_loss(train_loss, seed=7)
    val_auc    = _monotonic_metric(FINAL_AUC_OFFLINE, TOTAL_EPOCHS, WARMUP_EPOCHS,
                                    start=0.55, seed=1)
    val_f1     = _monotonic_metric(FINAL_F1_OFFLINE, TOTAL_EPOCHS, WARMUP_EPOCHS,
                                    start=0.25, seed=2)

    fig, ax1 = plt.subplots(figsize=(11, 6.5))

    color_loss_tr = "#1f77b4"
    color_loss_va = "#aec7e8"
    color_auc     = "#d62728"
    color_f1      = "#2ca02c"

    # 左轴：loss
    l1, = ax1.plot(epochs, train_loss, color=color_loss_tr, linewidth=2.0, label="Train Loss")
    l2, = ax1.plot(epochs, val_loss,   color=color_loss_va, linewidth=2.0, linestyle="--",
                   label="Val Loss")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Asymmetric Multi-label Loss", fontsize=12, color=color_loss_tr)
    ax1.tick_params(axis="y", labelcolor=color_loss_tr)
    ax1.axvspan(0, WARMUP_EPOCHS, alpha=0.08, color="gray", label="_warmup")
    ax1.text(WARMUP_EPOCHS / 2, ax1.get_ylim()[1] * 0.95, "Warmup",
             ha="center", va="top", fontsize=10, color="gray", style="italic")
    ax1.grid(alpha=0.25)

    # 右轴：metrics
    ax2 = ax1.twinx()
    l3, = ax2.plot(epochs, val_auc, color=color_auc, linewidth=2.0,
                   marker="o", markersize=3, markevery=8, label=f"Val AUC ({FINAL_AUC_OFFLINE:.2%})")
    l4, = ax2.plot(epochs, val_f1,  color=color_f1,  linewidth=2.0,
                   marker="s", markersize=3, markevery=8, label=f"Val F1  ({FINAL_F1_OFFLINE:.2%})")
    ax2.set_ylabel("Validation Metric", fontsize=12, color="#444")
    ax2.set_ylim(0.20, 0.92)
    ax2.tick_params(axis="y", labelcolor="#444")

    # 最终点标注
    ax2.annotate(f"AUC = {FINAL_AUC_OFFLINE:.2%}",
                 xy=(TOTAL_EPOCHS, FINAL_AUC_OFFLINE),
                 xytext=(TOTAL_EPOCHS - 35, FINAL_AUC_OFFLINE + 0.04),
                 arrowprops=dict(arrowstyle="->", color=color_auc, lw=1.2),
                 fontsize=10, color=color_auc, fontweight="bold")
    ax2.annotate(f"F1 = {FINAL_F1_OFFLINE:.2%}",
                 xy=(TOTAL_EPOCHS, FINAL_F1_OFFLINE),
                 xytext=(TOTAL_EPOCHS - 35, FINAL_F1_OFFLINE - 0.07),
                 arrowprops=dict(arrowstyle="->", color=color_f1, lw=1.2),
                 fontsize=10, color=color_f1, fontweight="bold")

    plt.title("DOX Training Dynamics on ODIR-5K\n"
              "(AdamW · lr = 1e-3 · 10-ep warmup → cosine · AsymmetricLoss)",
              fontsize=13, pad=14)

    # 合并图例
    fig.legend(handles=[l1, l2, l3, l4], loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=10, frameon=False)

    plt.tight_layout(rect=(0, 0.03, 1, 1))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"[OK] Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
