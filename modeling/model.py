"""DOX 顶层模型定义.

本模块组合论文中的四大组件构成完整模型：

* **共享 ViT 主干** ── :class:`modeling.models_vit.RETFound_mae`
  RETFound 在 160 万张未标注视网膜图像上自监督预训练的 ViT-Large/16。
* **LoRA 微调注意力** ── :class:`modeling.fine_tune._LoRA_qkv`
  对每层自注意力的 Q / V 投影注入低秩适配。
* **多尺度卷积 Adapter** ── :class:`modeling.fine_tune._Conv_Adapter_MultiScale`
  替换每层 MLP，使用双路 (左 / 右眼) 专家网络。
* **DBFEM 双流交互融合** ── :class:`modeling.fusion.DEFF`
  对左右眼 patch token 进行双流 Transformer 融合。

输入约定：左右眼图像 *交错* 排列成单个 ``(2B, 3, 224, 224)`` 张量，
其中偶数位为左眼、奇数位为右眼。这样设计可让所有组件共享同一
``BatchNorm`` / ``LayerNorm`` 统计，同时通过 ``::2`` / ``1::2``
切片实现"双眼协同"机制，工程上完全零额外开销。
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms

from modeling.common import LayerNorm2d
from modeling.fine_tune import _Conv_Adapter_MultiScale, _LoRA_qkv
from modeling.fusion import DEFF
from modeling.models_vit import RETFound_mae


def build_model(checkpoint_path: str | None, device: str | torch.device) -> "MODEL":
    """构建并返回**评估模式**的 DOX 模型。

    Parameters
    ----------
    checkpoint_path:
        微调后的 ``.pth`` 检查点路径；为 ``None`` 时返回未加载权重的随机初始化模型
        (主要用于单元测试或 dry-run)。
    device:
        模型放置的设备，例如 ``"cuda"`` 或 ``"cpu"``。

    Returns
    -------
    MODEL
        已 ``eval()`` 的模型实例，可直接送入 ``inference_single`` /
        ``inference_batch``。
    """
    model = MODEL(
        check_path=None, 
        embed_dim=1024, 
        out_chans=256, 
        num_classes=8, 
        r=4, 
        adapter_dim=128, 
        dropout=0.1
    ).to(device).eval()
    if checkpoint_path is not None:
        with open(checkpoint_path, "rb") as f:
            state_dict = torch.load(f, map_location=torch.device('cpu'))['model']
        msg = model.load_state_dict(state_dict, False)
        print(msg)
    return model


class MODEL(nn.Module):
    """DOX 完整模型 (RETFound + LoRA + Conv-Adapter + DBFEM + Classifier).

    Parameters
    ----------
    check_path:
        RETFound 预训练权重路径 (``RETFound_mae_natureCFP.pth``)；为 ``None``
        时跳过加载，主要用于单元测试。
    embed_dim:
        ViT 主干隐藏维度，RETFound MAE 为 1024。
    out_chans:
        Neck 之后、融合 / 分类阶段的通道数。
    num_classes:
        多标签输出维度，ODIR-5K 为 8。
    r:
        LoRA 低秩，论文默认 4。
    adapter_dim:
        Conv-Adapter 内部通道数，论文默认 128。
    dropout:
        分类头 dropout 比例。
    """

    def __init__(
        self,
        check_path:  str | None = None,
        embed_dim:   int = 1024,
        out_chans:   int = 256,
        num_classes: int = 8,
        r:           int = 16,
        adapter_dim: int = 64,
        dropout:     float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        
        # 图像处理pipeline
        self.transform = transforms.Compose([
            transforms.Resize(256),                         # 先把短边缩放到 256
            transforms.CenterCrop(224),                     # 再裁出中心 224×224
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])

        # 创建图像编码器
        image_encoder = RETFound_mae()

        if check_path is not None:
            model_params = torch.load(check_path, map_location="cpu", weights_only=False)["model"]
            
            # 处理权重键名
            for k in list(model_params.keys()):
                if k.startswith("decoder_"):
                    del model_params[k] 
                if k == 'mask_token':
                    del model_params[k]
                if k.startswith("norm."):
                    del model_params[k]
            
            # 加载预训练权重
            msg = image_encoder.load_state_dict(model_params, strict=False)
            print(msg)
            
            # 冻结基础模型的ViT编码器参数
            for n, value in image_encoder.named_parameters():
                value.requires_grad = False  # 冻结所有参数
                if 'cls_token' in n: 
                    value.requires_grad = True

        self.w_As = []
        self.w_Bs = []
        for t_layer_i, blk in enumerate(image_encoder.blocks):
            w_qkv_linear = blk.attn.qkv  
            self.dim = w_qkv_linear.in_features
            
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)  # Q的A矩阵
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)  # Q的B矩阵
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)  # V的A矩阵
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)  # V的B矩阵
            
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
            blk.mlp = _Conv_Adapter_MultiScale(blk.mlp, adapter_dim)
        
        self.reset_parameters()
        
        self.image_encoder = image_encoder

        self.reduction = nn.Linear(embed_dim, out_chans)
        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

        self.fusion = DEFF(embed_dim=out_chans)
        
        # 分类头
        self.norm = nn.LayerNorm(out_chans*2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(out_chans*2, out_chans),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_chans, num_classes)
        )

    def reset_parameters(self) -> None:
        """LoRA 标准初始化：A 矩阵 Kaiming-Uniform，B 矩阵全零 (训练初始即恒等)。"""
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Parameters
        ----------
        x:
            左右眼交错 (``[L,R,L,R,...]``) 的张量，shape ``(2B, 3, 224, 224)``。

        Returns
        -------
        torch.Tensor
            多标签 logits，shape ``(B, num_classes)``，**未** 经过 sigmoid。
        """
        x = self.image_encoder.forward_features(x) # 2*B,3,224,224 -> 2*B,197,1024
        cls_token, patch_tokens = x[:, 0, :], x[:, 1:, :]
        cls_token = self.reduction(cls_token)  # 2*B, 256
        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)
        patch_tokens = patch_tokens.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # 2*B, 16, 16, 1024
        patch_tokens = self.neck(patch_tokens)  # 2*B, 16, 16, 256
        patch_tokens = patch_tokens.permute(0, 2, 3, 1).contiguous().view(B, H * W, -1)  # 2*B, 196, 256

        left_cls = cls_token[0::2]  # B,256
        right_cls = cls_token[1::2]
        cls_fused = (left_cls + right_cls) / 2  # B/2,256
        patch_tokens = self.fusion(patch_tokens) # 2*B,196,256 -> B,196,256
        patch_mean = patch_tokens.mean(dim=1) # B,256

        feat = torch.cat([cls_fused, patch_mean], dim=-1)  # B,512
        feat = self.norm(feat)
        feat = self.dropout(feat)
        logits  = self.classifier(feat) # B,8
        return logits


if __name__ == "__main__":
    import numpy as np

    model = MODEL(check_path=None, adapter_dim=128)
    # model = MODEL().eval()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"总参数数量：{params/1e6}M")
    for name, param in model.named_parameters():
        print(f"层名称: {name}, 参数形状: {param.shape}, 是否可训练: {param.requires_grad}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6}M")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params / 1e6}M")

    with torch.inference_mode():
        image_shape = (4, 3, 224, 224)  # (batch_size, channels, height, width)
        image = torch.randn(image_shape)
        out = model(image)
        print(out.shape)

    # image_shape = (4, 3, 224, 224)  # (batch_size, channels, height, width)
    # image = torch.randn(image_shape)
    # out = model(image)
    # print(out.shape)
