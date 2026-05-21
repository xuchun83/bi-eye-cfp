"""DBFEM ── 双流交互注意力融合模块 (Dual-stream Binocular Feature Enhancement Module).

对应论文 3.(二).2 节、图 3 及式 (5)-(9)。

核心计算：

.. math::

    F_L^{pool} &= \\mathrm{Pool}(F_L) \\\\
    F_R^{pool} &= \\mathrm{Pool}(F_R) \\\\
    Coef      &= \\mathrm{Transformer}\\bigl(\\mathrm{Concat}(F_L^{pool}, F_R^{pool})\\bigr) \\\\
    W         &= \\mathrm{Softmax}(Coef) \\\\
    Fuse      &= W_L \\odot F_L^{pool} + W_R \\odot F_R^{pool}

实现略与论文公式不同的工程细节：

* ``Pool`` 通过沿序列维度按 2 折叠 + 平均实现，等价于步幅 2 的 1D AvgPool；
* 输出之前用 ``F.interpolate`` 将 patch 数恢复到 ``N``，便于后续残差；
* 在融合输出上额外加了一项 ``(F_L+F_R)/2`` 的恒等项，提升训练稳定性。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DEFF(nn.Module):
    """双流交互融合模块。

    Parameters
    ----------
    embed_dim:
        输入 / 输出特征维度 (应与 :class:`MODEL` 的 ``out_chans`` 一致)。
    num_heads:
        Transformer 多头注意力的头数。
    num_transformer_blocks:
        融合 Transformer 块的数量。
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 8,
        num_transformer_blocks: int = 2,
    ) -> None:
        super().__init__()

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_transformer_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对左右眼交错排列的 patch tokens 进行双流融合。

        Parameters
        ----------
        x:
            ``(2B, N, C)``；偶数索引为左眼、奇数为右眼。

        Returns
        -------
        torch.Tensor
            融合后的特征，shape ``(B, N, C)``。
        """
        orig_left_features = x[0::2]  # B/2,N,C
        orig_right_features = x[1::2]
        average_features = (orig_left_features + orig_right_features) / 2  # B/2,N,C

        # 平均池化
        # left_cls, left_features = orig_left_features[:, :1, :], orig_left_features[:, 1:, :]
        # right_cls, right_features = orig_right_features[:, :1, :], orig_right_features[:, 1:, :]
        # batch_size, num_tokens, embed_dim = left_features.shape
        batch_size, num_tokens, embed_dim = orig_left_features.shape
        left_features = orig_left_features.view(batch_size, num_tokens // 2, 2, embed_dim)
        left_features = torch.mean(left_features, dim=2)
        right_features = orig_right_features.view(batch_size, num_tokens // 2, 2, embed_dim)
        right_features = torch.mean(right_features, dim=2) # B/2,N,C -> B/2,N/2,C

        # concatenated_tokens = torch.cat([left_cls, left_features, right_cls, right_features], dim=1) # B/2,N,C
        concatenated_tokens = torch.cat([left_features, right_features], dim=1) # B/2,N,C

        # 通过两个Transformer模块
        for transformer_block in self.transformer_blocks:
            concatenated_tokens = transformer_block(concatenated_tokens)

        # left_features_withcls = concatenated_tokens[:, :num_tokens//2+1, :]   # B/2,N/2,C
        # right_features_withcls = concatenated_tokens[:, num_tokens//2+1:, :]
        left_features = concatenated_tokens[:, :num_tokens//2, :]   # B/2,N/2,C
        right_features = concatenated_tokens[:, num_tokens//2:, :]
        # left_cls, left_features = left_features_withcls[:, :1, :], left_features_withcls[:, 1:, :]
        # right_cls, right_features = right_features_withcls[:, :1, :], right_features_withcls[:, 1:, :]

        upsampled_left_features = F.interpolate(
            left_features.transpose(1, 2), 
            size=num_tokens, 
            mode='linear',
            align_corners=False
        ).transpose(1, 2) # B,N,C
        upsampled_right_features = F.interpolate(
            right_features.transpose(1, 2), 
            size=num_tokens, 
            mode='linear',
            align_corners=False
        ).transpose(1, 2) 

        # left_cf = torch.cat([left_cls, upsampled_left_features], dim=1)
        # right_cf = torch.cat([right_cls, upsampled_right_features], dim=1)

        # left_weights = F.softmax(left_cf, dim=-1)
        # right_weights = F.softmax(right_cf, dim=-1)
        left_weights = F.softmax(upsampled_left_features, dim=-1)
        right_weights = F.softmax(upsampled_right_features, dim=-1)

        result_left_features = left_weights * orig_left_features  # B/2,N,C
        result_right_features = right_weights * orig_right_features

        fused = result_left_features + result_right_features + average_features  # [B/2, N, C]

        return fused


class TransformerBlock(nn.Module):
    """标准 pre-LN Transformer 块 (MHSA + FFN，两次残差)。

    供 :class:`DEFF` 内部使用，故未单独提取到 ``modeling.common``。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout:   float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # 多头自注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual #1: pre-LN self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attention(x, x, x)
        x = self.dropout(x)
        x = residual + x

        # Residual #2: pre-LN FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x
