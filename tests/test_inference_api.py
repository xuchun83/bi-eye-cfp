"""测试 inference_single / inference_batch 库函数级 API 的契约 (不依赖真实权重)。"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from inference_single import CLASS_NAMES, inference_single
from modeling.model import MODEL


@pytest.fixture(scope="module")
def model() -> MODEL:
    return MODEL(
        check_path  = None,
        embed_dim   = 1024,
        out_chans   = 256,
        num_classes = 8,
        r           = 4,
        adapter_dim = 32,
        dropout     = 0.0,
    ).eval()


@pytest.fixture
def fake_pair(tmp_path) -> tuple[str, str]:
    """生成一对临时左右眼图像。"""
    arr = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    left  = tmp_path / "0_left.jpg"
    right = tmp_path / "0_right.jpg"
    Image.fromarray(arr).save(left)
    Image.fromarray(arr).save(right)
    return str(left), str(right)


def test_inference_single_keys_and_lengths(model, fake_pair) -> None:
    left, right = fake_pair
    result = inference_single(model, left, right, device="cpu")
    assert set(result.keys()) == {"probabilities", "predictions", "classes"}
    assert len(result["probabilities"]) == len(CLASS_NAMES)
    assert len(result["predictions"])   == len(CLASS_NAMES)


def test_inference_single_probabilities_in_range(model, fake_pair) -> None:
    left, right = fake_pair
    result = inference_single(model, left, right, device="cpu")
    for p in result["probabilities"]:
        assert 0.0 <= p <= 1.0, f"prob {p} 不在 [0, 1] 内"


def test_inference_single_predictions_are_binary(model, fake_pair) -> None:
    left, right = fake_pair
    result = inference_single(model, left, right, device="cpu")
    for pred in result["predictions"]:
        assert pred in (0, 1)


def test_inference_single_classes_are_subset(model, fake_pair) -> None:
    left, right = fake_pair
    result = inference_single(model, left, right, device="cpu")
    assert set(result["classes"]).issubset(CLASS_NAMES)
