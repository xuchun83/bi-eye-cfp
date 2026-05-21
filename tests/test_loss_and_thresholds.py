"""测试非对称多标签损失与类别差异化阈值逻辑。"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from timm.loss import AsymmetricLossMultiLabel

from train.test import DEFAULT_THRESHOLDS, apply_thresholds, multilabel_kappa


def test_asymmetric_loss_is_finite_and_positive() -> None:
    """损失应为有限正数，且对极端 logits 不发散。"""
    loss_fn = AsymmetricLossMultiLabel(gamma_pos=0, gamma_neg=4, clip=0.05)
    logits = torch.randn(8, 8)
    target = torch.randint(0, 2, (8, 8)).float()
    loss = loss_fn(logits, target)
    assert torch.isfinite(loss).all()
    assert loss.item() >= 0


def test_asymmetric_loss_handles_extreme_logits() -> None:
    """logits = ±100 不应导致 NaN / Inf。"""
    loss_fn = AsymmetricLossMultiLabel(gamma_pos=0, gamma_neg=4, clip=0.05)
    logits = torch.full((4, 8), 100.0)
    target = torch.zeros(4, 8)
    loss = loss_fn(logits, target)
    assert torch.isfinite(loss).all()


def test_default_thresholds_have_all_eight_classes() -> None:
    """阈值字典必须严格覆盖 8 个类别。"""
    expected = {"N", "D", "G", "C", "A", "H", "M", "O"}
    assert set(DEFAULT_THRESHOLDS.keys()) == expected
    for v in DEFAULT_THRESHOLDS.values():
        assert 0.0 < v < 1.0


def test_apply_thresholds_correctness() -> None:
    """同概率不同阈值应得到不同预测；阈值边界取大于等于。"""
    probs = np.array([
        [0.50, 0.71, 0.49, 0.71, 0.71, 0.41, 0.81, 0.61],
    ])
    preds = apply_thresholds(probs, DEFAULT_THRESHOLDS)
    # N=0.5<0.7→0, D=0.71>=0.7→1, G=0.49<0.5→0, C=0.71>=0.7→1, ...
    np.testing.assert_array_equal(preds[0], np.array([0, 1, 0, 1, 1, 1, 1, 1]))


def test_multilabel_kappa_perfect_agreement_is_one() -> None:
    """完全相同的标签 → Kappa = 1。"""
    y = np.random.randint(0, 2, size=(50, 8))
    assert multilabel_kappa(y, y) == pytest.approx(1.0)


def test_multilabel_kappa_random_is_low() -> None:
    """随机预测 → Kappa 接近 0。"""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=(200, 8))
    y_pred = rng.integers(0, 2, size=(200, 8))
    k = multilabel_kappa(y_true, y_pred)
    assert -0.3 < k < 0.3, f"随机情况下 Kappa 应接近 0，实际 {k}"
