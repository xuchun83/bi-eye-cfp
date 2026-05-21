"""消融实验可视化柱状图.

数据来源: 论文 第四章 表 3-5 (ODIR-5K On-site test)。
全部数值与 README.md 的消融实验表完全一致，未经任何修饰。

使用方法
-------
    python assets/scripts/plot_ablation.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ASSETS_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ASSETS_DIR / "ablation_results.png"

# ============================================================
# 论文表 3-5 的原始数据 (单位: %)
# ============================================================
ABLATION_DATA: dict[str, dict[str, list[float]]] = {
    "LoRA Module (Table 3)": {
        "variants":  ["No LoRA",  "LoRA-Q",  "LoRA-V",  "Full Model"],
        "F1":        [44.21,      47.83,     48.97,     50.55],
        "AUC":       [76.85,      79.42,     80.15,     83.39],
        "Recall":    [48.12,      51.23,     52.34,     53.50],
        "Precision": [48.56,      51.58,     52.78,     53.86],
    },
    "Conv-Adapter Multi-Scale (Table 4)": {
        "variants":  ["No Adapter", "Single-Scale", "Full Model"],
        "F1":        [48.12,        49.38,          50.55],
        "AUC":       [81.56,        82.87,          83.39],
        "Recall":    [51.87,        52.91,          53.50],
        "Precision": [52.23,        53.35,          53.86],
    },
    "DBFEM Fusion (Table 5)": {
        "variants":  ["No Fusion", "Simple Avg", "Attention",  "Full Model"],
        "F1":        [47.91,       48.32,        49.18,        50.55],
        "AUC":       [81.23,       81.78,        82.56,        83.39],
        "Recall":    [50.87,       51.23,        52.34,        53.50],
        "Precision": [51.23,       51.58,        52.78,        53.86],
    },
}

METRIC_COLORS = {
    "F1":        "#d62728",
    "AUC":       "#1f77b4",
    "Recall":    "#2ca02c",
    "Precision": "#ff7f0e",
}


def _plot_one_panel(ax: plt.Axes, title: str, data: dict[str, list[float]]) -> None:
    variants = data["variants"]
    metrics = ["F1", "AUC", "Recall", "Precision"]
    n_v = len(variants)
    n_m = len(metrics)
    width = 0.8 / n_m
    x = np.arange(n_v)

    for i, metric in enumerate(metrics):
        values = data[metric]
        offset = (i - (n_m - 1) / 2) * width
        bars = ax.bar(x + offset, values, width,
                      label=metric, color=METRIC_COLORS[metric], edgecolor="white", linewidth=0.6)
        # 仅在最高指标 (F1) 上标注数值，避免太挤
        if metric == "F1":
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.4,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8.5,
                        color=METRIC_COLORS["F1"], fontweight="bold")

    # 高亮 Full Model
    full_idx = variants.index("Full Model")
    ax.axvspan(full_idx - 0.5, full_idx + 0.5, alpha=0.08, color="gold", zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=12, ha="right", fontsize=10)
    ax.set_ylim(40, 90)
    ax.set_ylabel("Score (%)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (title, data) in zip(axes, ABLATION_DATA.items()):
        _plot_one_panel(ax, title, data)

    # 全局图例
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, label=m)
        for m, c in METRIC_COLORS.items()
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04),
               ncol=4, fontsize=11, frameon=False)

    fig.suptitle(
        "Ablation Studies on ODIR-5K On-site Test Set\n"
        "(Higher is better. Gold band marks the Full Model.)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    plt.tight_layout(rect=(0, 0.02, 1, 1))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"[OK] Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
