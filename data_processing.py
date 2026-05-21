"""CFP 眼底图像三阶段预处理流水线 (DOX preprocessing pipeline).

本模块实现论文 *《基于大模型的双目 CFP 眼部疾病智能检测》* 第三章 (一) 节
所描述的三阶段预处理流程，理论基础参考 Gaudio 等人 [28]:

    1. **光照校正 (illumination correction)**
       基于暗通道 / 亮通道先验估计透射图，参数化的大气光模型对
       局部亮度不均区域进行补偿。

       核心失真模型 (式 1):
           I(x) = J(x) · t(x) + A · (1 - t(x))

       清晰图像求解公式 (式 2):
           J(x) = (I(x) - A) / max(t(x), eps) + A

    2. **局部自适应锐化 (locally adaptive sharpening)**
       通过原图与模糊版本的差异图提取高频细节，并根据局部统计自适应
       决定锐化强度，增强血管 / 视盘等微细结构。

       式 3:
           J = (1/k) ⊙ I  -  (1/k - 1) ⊙ blur(I)

    3. **双边滤波降噪 (bilateral denoising)**
       在空间域与值域同时施加高斯权重，在抑制随机噪声的同时保持边缘。

实现说明
-------
* 默认走 ``IETK-Ret`` (参考文献 [28] 官方实现) 的 ``brighten_darken``
  操作完成 *光照校正 + 锐化* 联合优化，结果与论文实验一致；
* 双边滤波由 OpenCV 提供，保持论文中 σ_color / σ_space 的可配置性；
* 三个阶段均通过独立函数暴露，方便消融实验。

CLI 用法
-------
单张图像::

    python -m data_processing single \\
        --input  datademo/raw_data/0_left.jpg \\
        --output datademo/processed/single

批量处理::

    python -m data_processing batch \\
        --input-dir  datademo/raw_data \\
        --output-dir datademo/processed/batch \\
        --bilateral-d 9 \\
        --bilateral-sigma-color 75 \\
        --bilateral-sigma-space 75
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from tqdm import tqdm

try:
    from ietk import methods, util
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "IETK-Ret is required for illumination correction.\n"
        "Install with:  pip install IETK-Ret==0.1.1"
    ) from e


logger = logging.getLogger(__name__)
_SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")


# ============================================================
# Configuration
# ============================================================
@dataclass
class PreprocessConfig:
    """三阶段预处理的可配置参数。

    Attributes
    ----------
    illumination_method:
        IETK ``brighten_darken`` 使用的算子组合。
        ``"A+B+X"``  ── 论文使用的默认组合 (Amplification + Brighten + crossover)
    bilateral_d:
        双边滤波的邻域直径，0 时由 OpenCV 根据 sigma_space 自动推断。
    bilateral_sigma_color:
        值域高斯方差，越大对颜色差异越不敏感。
    bilateral_sigma_space:
        空间域高斯方差，越大滤波范围越大。
    enable_illumination:
        是否启用第 1 阶段 (光照校正)。
    enable_sharpening:
        是否启用第 2 阶段 (自适应锐化)。
    enable_denoise:
        是否启用第 3 阶段 (双边滤波降噪)。
    """

    illumination_method: str = "A+B+X"
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0

    enable_illumination: bool = True
    enable_sharpening: bool = True
    enable_denoise: bool = True


# ============================================================
# Stage 1 + 2: Illumination correction & adaptive sharpening
# ============================================================
def illumination_correction_and_sharpening(
    image: np.ndarray,
    method: str = "A+B+X",
) -> np.ndarray:
    """对图像执行论文第 1 阶段 (光照校正) 与第 2 阶段 (自适应锐化)。

    底层调用 ``ietk.methods.brighten_darken``。该方法基于暗 / 亮通道先验
    估计透射图并求解清晰图 (公式 1-2)，同时通过自适应反卷积掩模完成
    多尺度锐化 (公式 3)。

    Parameters
    ----------
    image:
        RGB 图像，浮点数组，取值范围 ``[0, 1]``，shape ``(H, W, 3)``。
    method:
        IETK 算子组合字符串，论文默认 ``"A+B+X"``。

    Returns
    -------
    np.ndarray
        校正与锐化后的图像，shape 与输入相同，已被 ``np.clip`` 到 ``[0, 1]``。
    """
    cropped, foreground_mask = util.center_crop_and_get_foreground_mask(image)
    enhanced = methods.brighten_darken(cropped, method, focus_region=foreground_mask)
    return np.clip(enhanced, 0.0, 1.0)


# ============================================================
# Stage 3: Bilateral denoising
# ============================================================
def bilateral_denoise(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """对图像执行论文第 3 阶段 ── 双边滤波降噪。

    双边滤波同时在空间域和值域施加高斯权重，能在抑制随机噪声的同时
    保留眼底血管边缘等高频结构 (论文中关键论断之一)。

    Parameters
    ----------
    image:
        RGB 图像，``float64`` 在 ``[0, 1]`` 区间，shape ``(H, W, 3)``。
    d:
        邻域直径，``0`` 时由 sigma_space 自动决定。
    sigma_color, sigma_space:
        值域 / 空间域的高斯方差。

    Returns
    -------
    np.ndarray
        降噪后的 ``float32`` 图像，仍在 ``[0, 1]`` 区间。
    """
    img_u8 = (image * 255.0).clip(0, 255).astype(np.uint8)
    denoised = cv2.bilateralFilter(img_u8, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    return denoised.astype(np.float32) / 255.0


# ============================================================
# Composite pipeline
# ============================================================
def preprocess_pipeline(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """完整三阶段预处理流水线。

    Parameters
    ----------
    image:
        RGB 浮点图像，取值范围 ``[0, 1]``，shape ``(H, W, 3)``。
    config:
        参数配置。可逐阶段开关，便于消融实验。
    """
    out = image
    if config.enable_illumination or config.enable_sharpening:
        out = illumination_correction_and_sharpening(out, method=config.illumination_method)
    if config.enable_denoise:
        out = bilateral_denoise(
            out,
            d=config.bilateral_d,
            sigma_color=config.bilateral_sigma_color,
            sigma_space=config.bilateral_sigma_space,
        )
    return out


# ============================================================
# Dataset iterator (lightweight, no torch dependency)
# ============================================================
class FundusImageFolder:
    """惰性遍历某目录下所有眼底图像，按 ``(filename, rgb_float)`` 产出。"""

    def __init__(self, img_dir: str | os.PathLike[str]) -> None:
        self.img_dir = Path(img_dir)
        if not self.img_dir.is_dir():
            raise NotADirectoryError(f"{self.img_dir} is not a directory")

        self.img_paths: list[Path] = sorted(
            Path(p) for p in glob(str(self.img_dir / "*"))
            if p.lower().endswith(_SUPPORTED_EXTS)
        )

    def __len__(self) -> int:
        return len(self.img_paths)

    def __iter__(self) -> Iterator[tuple[str, np.ndarray]]:
        for path in self.img_paths:
            img_bgr = cv2.imread(str(path))
            if img_bgr is None:
                logger.warning("Failed to read %s, skipped.", path)
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
            yield path.name, img_rgb


# ============================================================
# High-level public API
# ============================================================
def process_single_image(
    img_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] | None = None,
    config: PreprocessConfig | None = None,
) -> np.ndarray:
    """处理单张图像并 (可选) 落盘。

    Parameters
    ----------
    img_path:
        输入图像路径。
    output_dir:
        输出目录；若为 ``None``，仅返回处理结果而不写文件。
    config:
        预处理配置；若为 ``None``，使用论文默认参数。

    Returns
    -------
    np.ndarray
        预处理后的图像，RGB float ∈ ``[0, 1]``。
    """
    config = config or PreprocessConfig()
    img_path = Path(img_path)

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found or unreadable: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

    enhanced = preprocess_pipeline(img_rgb, config)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / img_path.name
        _save_image(enhanced, save_path)
        logger.info("Saved processed image: %s", save_path)

    return enhanced


def process_image_folder(
    input_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    config: PreprocessConfig | None = None,
) -> None:
    """批量处理某目录下的所有眼底图像。"""
    config = config or PreprocessConfig()
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = FundusImageFolder(input_dir)
    logger.info("Processing %d images from %s", len(dataset), input_dir)

    for filename, img_rgb in tqdm(dataset, total=len(dataset), desc="Preprocessing"):
        try:
            enhanced = preprocess_pipeline(img_rgb, config)
            _save_image(enhanced, output_dir / filename)
        except Exception as exc:
            logger.exception("Failed on %s: %s", filename, exc)

    logger.info("Done. Output saved to %s", output_dir)


def _save_image(img: np.ndarray, save_path: Path) -> None:
    """以 BGR-uint8 写出 ``img`` (RGB-float ∈ [0,1]) 到 ``save_path``。"""
    arr = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(save_path), arr_bgr):
        raise IOError(f"cv2.imwrite failed for {save_path}")


# ============================================================
# CLI
# ============================================================
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="data_processing",
        description="DOX ── 三阶段 CFP 预处理 (光照校正 + 自适应锐化 + 双边滤波)",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # single
    p_single = subparsers.add_parser("single", help="处理单张图像")
    p_single.add_argument("--input",  "-i", required=True, type=str, help="输入图像路径")
    p_single.add_argument("--output", "-o", required=True, type=str, help="输出目录")

    # batch
    p_batch = subparsers.add_parser("batch", help="批量处理目录")
    p_batch.add_argument("--input-dir",  "-i", required=True, type=str, help="输入目录")
    p_batch.add_argument("--output-dir", "-o", required=True, type=str, help="输出目录")

    # shared config
    for p in (p_single, p_batch):
        p.add_argument("--illumination-method", default="A+B+X",
                       help="IETK brighten_darken 算子组合 (论文默认 A+B+X)")
        p.add_argument("--bilateral-d", type=int, default=9, help="双边滤波邻域直径")
        p.add_argument("--bilateral-sigma-color", type=float, default=75.0)
        p.add_argument("--bilateral-sigma-space", type=float, default=75.0)
        p.add_argument("--no-illumination", action="store_true", help="跳过第 1 阶段 (光照校正)")
        p.add_argument("--no-sharpening",   action="store_true", help="跳过第 2 阶段 (自适应锐化)")
        p.add_argument("--no-denoise",      action="store_true", help="跳过第 3 阶段 (双边滤波)")

    return parser


def _args_to_config(args: argparse.Namespace) -> PreprocessConfig:
    return PreprocessConfig(
        illumination_method=args.illumination_method,
        bilateral_d=args.bilateral_d,
        bilateral_sigma_color=args.bilateral_sigma_color,
        bilateral_sigma_space=args.bilateral_sigma_space,
        enable_illumination=not args.no_illumination,
        enable_sharpening=not args.no_sharpening,
        enable_denoise=not args.no_denoise,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _build_arg_parser().parse_args()
    config = _args_to_config(args)

    if args.mode == "single":
        process_single_image(args.input, args.output, config)
    else:
        process_image_folder(args.input_dir, args.output_dir, config)


if __name__ == "__main__":
    main()
