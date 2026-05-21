"""测试 DBFEM 双流交互融合模块的形状与梯度通路。"""

from __future__ import annotations

import torch

from modeling.fusion import DEFF


def test_deff_output_shape() -> None:
    """``(2B, N, C)`` → ``(B, N, C)``。"""
    B, N, C = 3, 196, 64
    deff = DEFF(embed_dim=C, num_heads=4, num_transformer_blocks=1).eval()
    x = torch.randn(2 * B, N, C)
    with torch.inference_mode():
        y = deff(x)
    assert y.shape == (B, N, C)
    assert torch.isfinite(y).all()


def test_deff_gradient_flows() -> None:
    """所有可训练参数都应能接收梯度，确保融合模块不会被无意冻结。"""
    B, N, C = 2, 196, 32
    deff = DEFF(embed_dim=C, num_heads=4, num_transformer_blocks=1).train()
    x = torch.randn(2 * B, N, C, requires_grad=True)
    loss = deff(x).sum()
    loss.backward()
    for name, p in deff.named_parameters():
        assert p.grad is not None, f"参数 {name} 未收到梯度"
        assert torch.isfinite(p.grad).all(), f"参数 {name} 的梯度含 NaN"
