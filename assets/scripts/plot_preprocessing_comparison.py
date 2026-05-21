"""预处理前后对比图.

策略 (按优先级降序):

1. **离线已有图对比** — 若 ``datademo/processing_data/`` 已有用本仓库
   预处理流水线生成好的成品图 (与 ``datademo/raw_data/`` 同名)，直接
   并排展示，**无需任何第三方依赖**。
2. **现场调用流水线** — 若步骤 1 找不到成品图，但环境中安装了 ``cv2`` 与
   ``IETK-Ret``，则现场调用 :func:`data_processing.preprocess_pipeline`
   重新生成。
3. **失败提示** — 若都不可用，则给出友好提示后退出。

使用方法
-------
    python assets/scripts/plot_preprocessing_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

ASSETS_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ASSETS_DIR / "preprocessing_comparison.png"

RAW_DIR = _REPO_ROOT / "datademo" / "raw_data"
PROCESSED_DIRS = [
    _REPO_ROOT / "datademo" / "processing_data" / "batch_test",
    _REPO_ROOT / "datademo" / "processing_data" / "single_test",
]

PREFERRED_SAMPLES = ["0_left.jpg", "0_right.jpg", "1_left.jpg", "1_right.jpg"]


def _find_processed(name: str) -> Path | None:
    """在所有候选 processed 目录里找同名文件。"""
    for d in PROCESSED_DIRS:
        p = d / name
        if p.is_file():
            return p
    return None


def _try_load_offline_pairs() -> list[tuple[Path, Path]]:
    """优先返回 (raw_path, processed_path) 对。"""
    if not RAW_DIR.is_dir():
        return []
    pairs: list[tuple[Path, Path]] = []

    names = PREFERRED_SAMPLES + sorted(
        p.name for p in RAW_DIR.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        and p.name not in PREFERRED_SAMPLES
    )
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        raw_p = RAW_DIR / name
        proc_p = _find_processed(name)
        if raw_p.is_file() and proc_p is not None:
            pairs.append((raw_p, proc_p))
        if len(pairs) >= 4:
            break
    return pairs


def _try_run_pipeline_live() -> tuple[list[tuple[Path, np.ndarray]], str]:
    """现场调用预处理流水线，返回 (raw_path, processed_array) 与流水线描述。"""
    try:
        import cv2
        from data_processing import PreprocessConfig, preprocess_pipeline
    except Exception as exc:
        raise RuntimeError(f"cannot run live pipeline: {exc}") from exc

    samples: list[tuple[Path, np.ndarray]] = []
    cfg = PreprocessConfig()
    desc = "Full 3-stage pipeline (illumination + sharpening + denoise)"

    candidate_names = PREFERRED_SAMPLES if RAW_DIR.is_dir() else []
    for name in candidate_names:
        p = RAW_DIR / name
        if not p.is_file():
            continue
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
        try:
            enhanced = preprocess_pipeline(rgb, cfg)
        except Exception as exc:
            print(f"[WARN] full pipeline failed on {name}: {exc}; "
                  "falling back to bilateral-only.")
            from data_processing import bilateral_denoise
            enhanced = bilateral_denoise(rgb)
            desc = "Bilateral denoise only (IETK unavailable)"
        samples.append((p, np.clip(enhanced, 0, 1)))
        if len(samples) >= 4:
            break
    return samples, desc


def _plot(pairs: list[tuple[Path, np.ndarray | Path]], pipeline_desc: str) -> None:
    n = len(pairs)
    if n == 0:
        print("[ERROR] No image pairs to plot.")
        sys.exit(1)

    fig, axes = plt.subplots(2, n, figsize=(3.4 * n, 7))
    if n == 1:
        axes = axes[:, None]

    for col, (raw_path, processed) in enumerate(pairs):
        raw_img = mpimg.imread(raw_path)
        if isinstance(processed, Path):
            proc_img = mpimg.imread(processed)
        else:
            proc_img = processed

        axes[0, col].imshow(raw_img)
        axes[0, col].set_title(raw_path.name, fontsize=10)
        axes[0, col].axis("off")

        axes[1, col].imshow(proc_img)
        axes[1, col].axis("off")

    axes[0, 0].text(-0.10, 0.5, "Raw CFP",
                    transform=axes[0, 0].transAxes,
                    rotation=90, ha="right", va="center",
                    fontsize=13, fontweight="bold", color="#444")
    axes[1, 0].text(-0.10, 0.5, "Preprocessed",
                    transform=axes[1, 0].transAxes,
                    rotation=90, ha="right", va="center",
                    fontsize=13, fontweight="bold", color="#0a7b3f")

    fig.suptitle(
        f"CFP Preprocessing Pipeline — {pipeline_desc}\n"
        "(Illumination correction -> Adaptive sharpening -> Bilateral denoising)",
        fontsize=12, fontweight="bold",
    )

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"[OK] Saved {OUTPUT_PATH}")


def main() -> None:
    offline_pairs = _try_load_offline_pairs()
    if offline_pairs:
        print(f"[INFO] Using {len(offline_pairs)} offline pairs from datademo/processing_data/")
        _plot(offline_pairs, "Full 3-stage pipeline")
        return

    print("[INFO] No offline processed images found; trying live pipeline...")
    try:
        live_samples, desc = _try_run_pipeline_live()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        print("        Please either pre-generate images into "
              "datademo/processing_data/, or `pip install opencv-python IETK-Ret`.")
        sys.exit(1)

    if not live_samples:
        print("[ERROR] No usable raw images found in datademo/raw_data/.")
        sys.exit(1)
    _plot(live_samples, desc)


if __name__ == "__main__":
    main()
