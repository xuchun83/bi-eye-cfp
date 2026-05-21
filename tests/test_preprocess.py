"""测试三阶段预处理流水线 (光照校正 + 自适应锐化 + 双边滤波) 的形状与可关断性。"""

from __future__ import annotations

import numpy as np
import pytest

from data_processing import (
    PreprocessConfig,
    bilateral_denoise,
    preprocess_pipeline,
)


@pytest.fixture
def fake_fundus_image() -> np.ndarray:
    """构造一张 256×256 的模拟眼底图 (中心圆形亮区 + 噪声)。"""
    rng = np.random.default_rng(0)
    h = w = 256
    yy, xx = np.mgrid[0:h, 0:w]
    radius = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) ** 0.5
    mask = (radius < h * 0.45).astype(np.float64)
    img = np.stack([mask * 0.8, mask * 0.5, mask * 0.3], axis=-1)
    img += rng.normal(0, 0.03, size=img.shape)
    return np.clip(img, 0.0, 1.0)


def test_bilateral_denoise_preserves_shape(fake_fundus_image: np.ndarray) -> None:
    out = bilateral_denoise(fake_fundus_image, d=9, sigma_color=75, sigma_space=75)
    assert out.shape == fake_fundus_image.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_bilateral_denoise_reduces_noise(fake_fundus_image: np.ndarray) -> None:
    """降噪后的方差应比原图小 (定性验证)。"""
    out = bilateral_denoise(fake_fundus_image)
    assert out.var() < fake_fundus_image.var()


def test_preprocess_pipeline_all_stages_disabled_is_identity(fake_fundus_image: np.ndarray) -> None:
    """关闭所有阶段时应返回原图。"""
    cfg = PreprocessConfig(
        enable_illumination=False,
        enable_sharpening=False,
        enable_denoise=False,
    )
    out = preprocess_pipeline(fake_fundus_image, cfg)
    np.testing.assert_array_equal(out, fake_fundus_image)


def test_preprocess_pipeline_only_denoise(fake_fundus_image: np.ndarray) -> None:
    """只启用第 3 阶段时形状仍正确，可单独消融。"""
    cfg = PreprocessConfig(
        enable_illumination=False,
        enable_sharpening=False,
        enable_denoise=True,
    )
    out = preprocess_pipeline(fake_fundus_image, cfg)
    assert out.shape == fake_fundus_image.shape


def test_preprocess_config_defaults() -> None:
    """默认配置应严格匹配论文设置。"""
    cfg = PreprocessConfig()
    assert cfg.illumination_method == "A+B+X"
    assert cfg.enable_illumination is True
    assert cfg.enable_sharpening is True
    assert cfg.enable_denoise is True
