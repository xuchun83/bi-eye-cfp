"""DOX 模型架构示意图.

用纯 matplotlib 绘制论文图 2-3 的简化版本，便于在 GitHub README 中展示。
不依赖 Graphviz / dot 等额外工具，确保任何人 ``pip install -r requirements.txt``
后就能复现该图。

使用方法
-------
    python assets/scripts/plot_architecture.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ASSETS_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ASSETS_DIR / "architecture.png"

# 配色
C_INPUT  = "#E8F4FD"
C_PRE    = "#FFF4E6"
C_BACK   = "#E8F8E8"
C_PEFT   = "#FDE8EC"
C_FUSE   = "#F0E8FD"
C_HEAD   = "#FFF8D6"
EDGE     = "#333333"


def _box(ax, x, y, w, h, text, fc, fontsize=10, fontweight="normal"):
    """画一个圆角矩形 + 居中文字。"""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2, edgecolor=EDGE, facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize, fontweight=fontweight)


def _arrow(ax, x1, y1, x2, y2, color=EDGE, lw=1.4, style="->,head_width=4,head_length=6"):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style, color=color, linewidth=lw,
                            mutation_scale=10)
    ax.add_patch(arrow)


def main() -> None:
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # ===== Inputs =====
    _box(ax, 0.2,  7.8, 1.6, 0.9, "Left CFP",  C_INPUT, fontsize=10, fontweight="bold")
    _box(ax, 0.2,  1.3, 1.6, 0.9, "Right CFP", C_INPUT, fontsize=10, fontweight="bold")

    # ===== Preprocessing pipeline =====
    pre_x = 2.3
    _box(ax, pre_x, 7.5, 3.5, 1.5,
         "3-Stage Preprocessing\n\n① Illumination Correction\n② Adaptive Sharpening\n③ Bilateral Denoising",
         C_PRE, fontsize=9)
    _box(ax, pre_x, 1.0, 3.5, 1.5,
         "3-Stage Preprocessing\n\n① Illumination Correction\n② Adaptive Sharpening\n③ Bilateral Denoising",
         C_PRE, fontsize=9)
    _arrow(ax, 1.85, 8.25, pre_x - 0.05, 8.25)
    _arrow(ax, 1.85, 1.75, pre_x - 0.05, 1.75)

    # ===== Shared ViT Backbone =====
    bb_x, bb_y, bb_w, bb_h = 6.5, 3.5, 4.5, 4.5
    _box(ax, bb_x, bb_y, bb_w, bb_h,
         "Shared RETFound ViT-L/16 Backbone\n(first 12 layers, FROZEN)\n\n"
         "Each Block:\n"
         "  - Self-Attention <- LoRA (rank=4) [trainable]\n"
         "  - MLP <- Conv-Adapter Multi-Scale [trainable]\n"
         "    (Dual-eye experts)",
         C_BACK, fontsize=10)
    _arrow(ax, pre_x + 3.55, 8.25, bb_x + 0.4, 7.5)
    _arrow(ax, pre_x + 3.55, 1.75, bb_x + 0.4, 4.0)

    # PEFT 标签
    _box(ax, 6.6, 8.2, 4.3, 0.55,
         "PEFT: trainable parameters < 2%",
         C_PEFT, fontsize=9, fontweight="bold")

    # ===== DBFEM Fusion =====
    fu_x = 11.6
    _box(ax, fu_x, 4.6, 3.6, 2.4,
         "DBFEM Fusion Module\n(Dual-stream Transformer)\n\n"
         "  AvgPool → Concat →\n"
         "  Transformer × 2 →\n"
         "  Softmax → Voxel Weight",
         C_FUSE, fontsize=10)
    _arrow(ax, bb_x + bb_w, 6.0, fu_x - 0.05, 5.8)

    # CLS token bypass
    _box(ax, fu_x, 3.0, 3.6, 1.0,
         "CLS-token Average\n(left + right) / 2",
         C_FUSE, fontsize=10)
    _arrow(ax, bb_x + bb_w, 5.0, fu_x - 0.05, 3.5)

    # ===== Classifier head =====
    _box(ax, fu_x, 1.0, 3.6, 1.5,
         "Classification Head\nLayerNorm → Linear → GELU →\nDropout → Linear",
         C_HEAD, fontsize=10)
    _arrow(ax, fu_x + 1.8, 4.55, fu_x + 1.8, 2.55)
    _arrow(ax, fu_x + 1.8, 2.95, fu_x + 1.8, 2.55)

    # ===== Output =====
    _box(ax, fu_x + 0.5, -0.05, 2.6, 0.85,
         "8-Class Multi-label Logits",
         "#FFE8E8", fontsize=10, fontweight="bold")
    _arrow(ax, fu_x + 1.8, 1.0, fu_x + 1.8, 0.85)

    # ===== Title & legend =====
    ax.text(8.0, 9.65,
            "DOX Architecture — Binocular CFP Multi-Disease Detection",
            ha="center", va="top", fontsize=14, fontweight="bold")

    legend = [
        mpatches.Patch(color=C_INPUT, label="Input"),
        mpatches.Patch(color=C_PRE,   label="Preprocessing"),
        mpatches.Patch(color=C_BACK,  label="Frozen Backbone"),
        mpatches.Patch(color=C_PEFT,  label="PEFT (trainable)"),
        mpatches.Patch(color=C_FUSE,  label="DBFEM Fusion"),
        mpatches.Patch(color=C_HEAD,  label="Classifier"),
    ]
    ax.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.0, -0.04),
              ncol=6, fontsize=9, frameon=False)

    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"[OK] Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
