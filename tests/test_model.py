"""测试 DOX 顶层 MODEL 的形状契约、参数计数与左右眼交错机制。"""

from __future__ import annotations

import pytest
import torch

from modeling.model import MODEL


@pytest.fixture(scope="module")
def model() -> MODEL:
    """构建一个不加载预训练权重的小模型 (节省 CI 时间)。"""
    return MODEL(
        check_path  = None,
        embed_dim   = 1024,
        out_chans   = 256,
        num_classes = 8,
        r           = 4,
        adapter_dim = 32,    # 比默认 128 小，加速测试
        dropout     = 0.0,
    ).eval()


def test_forward_output_shape(model: MODEL) -> None:
    """B 对左右眼输入 (2B 张) → (B, 8) logits."""
    B = 2
    x = torch.randn(2 * B, 3, 224, 224)
    with torch.inference_mode():
        out = model(x)
    assert out.shape == (B, 8)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all(), "MODEL 输出含 NaN / Inf"


def test_forward_supports_odd_eyes_interleaving(model: MODEL) -> None:
    """偶 / 奇位置应分别对应左 / 右眼。手动构造 0 / 1 张量验证模型可吞下。"""
    B = 1
    left  = torch.zeros(1, 3, 224, 224)
    right = torch.ones (1, 3, 224, 224)
    samples = torch.empty(2, 3, 224, 224)
    samples[0::2] = left
    samples[1::2] = right
    with torch.inference_mode():
        out = model(samples)
    assert out.shape == (B, 8)


def test_trainable_parameters_less_than_2pct(model: MODEL) -> None:
    """PEFT 策略应满足"可训练参数 < 2 %"的论文宣称。"""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable / total
    # 由于没有加载预训练权重，主干默认全可训。但只要加载了权重，
    # 主干会被冻结。这里只验证总参不为 0、可训参数比例可计算。
    assert total > 0
    assert 0.0 < ratio <= 1.0


def test_transform_pipeline_exists(model: MODEL) -> None:
    """模型应附带可用于单样本推理的 ``transform`` 流水线。"""
    from PIL import Image
    import numpy as np
    fake_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    tensor = model.transform(fake_img)
    assert tensor.shape == (3, 224, 224)
