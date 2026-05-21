"""参数高效微调 (PEFT) 组件 ── LoRA 与多尺度 Conv-Adapter.

本模块实现论文 3.(二).1 节"Adapter-LoRA 混合高效微调方法"中的两个核心组件：

* :class:`_LoRA_qkv`
    用 *秩-r 低秩适配* 微调 Transformer 自注意力的 Q / V 投影
    (参考文献 [26] LoRA, Hu et al., ICLR 2022)。
    每次 forward 时把可学习的 ``B @ A`` 增量加到原 QKV 上，
    并通过一个 :class:`Gate` 模块控制注入强度。

* :class:`_Conv_Adapter_MultiScale`
    替换 Transformer Block 的 MLP 部分，提供"双路专家 (左 / 右眼) ×
    三尺度卷积分支"的旁路 Adapter (参考文献 [25] Adapter, Houlsby et al., ICML 2019)。
    通过逐点卷积压缩特征维度 → 多分支深度可分离卷积 → 逐点卷积恢复，
    与原 MLP 的输出按门控加和。

整套微调策略使**可训练参数 < 2 %**，在 ODIR-5K 上同时降低过拟合风险并
保留 RETFound 的强泛化特征。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Gate(nn.Module):
    """轻量门控模块：将输入压缩为单标量后过 sigmoid，用于控制旁路注入强度。"""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.dim = in_dim
        self.linear = nn.Linear(in_dim, 1)
        self.reset_parameters()

    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)
        if x.dim() == 4:
            x = rearrange(x, 'b h w c -> b (h w) c')
            return torch.mean(x, dim=1).unsqueeze(1).unsqueeze(2)
        else :
            return torch.mean(x, dim=1).unsqueeze(1)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

class _LoRA_qkv(nn.Module):
    def __init__(
            self,
            qkv: nn.Module,  
            linear_a_q: nn.Module, 
            linear_b_q: nn.Module, 
            linear_a_v: nn.Module,  
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.gate = Gate(self.dim)

    def forward(self, x):
        qkv = self.qkv(x)  
        
        new_q = self.linear_b_q(self.linear_a_q(x))  
        new_v = self.linear_b_v(self.linear_a_v(x))  

        qkv[:, :, : self.dim] += self.gate(x)*new_q
        qkv[:, :, -self.dim:] += self.gate(x)*new_v
        
        return qkv


class DWConvNormReLU(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels, 
            bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, in_channels, 1)
        self.norm = norm_layer(in_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class _Conv_Adapter_MultiScale(nn.Module):
    def __init__(self, mlp, adapter_dim):
        super().__init__()
        self.mlp = mlp
        self.in_dim = mlp.fc1.in_features
        self.adapter_dim = adapter_dim

        self.proj_down = nn.Linear(self.in_dim, self.adapter_dim)
        self.nonlinear = nn.ReLU(inplace=True)

        # 多尺度卷积路径
        self.branch_1x_left = DWConvNormReLU(adapter_dim)  
        self.branch_1x_right = DWConvNormReLU(adapter_dim)

        self.branch_2x_left = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DWConvNormReLU(adapter_dim),
            DWConvNormReLU(adapter_dim)
        )
        self.branch_2x_right = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DWConvNormReLU(adapter_dim),
            DWConvNormReLU(adapter_dim)
        )

        self.branch_4x_left = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DWConvNormReLU(adapter_dim),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DWConvNormReLU(adapter_dim),
            DWConvNormReLU(adapter_dim)
        )
        self.branch_4x_right = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DWConvNormReLU(adapter_dim),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DWConvNormReLU(adapter_dim),
            DWConvNormReLU(adapter_dim)
        )
        
        # 特征融合
        self.fusion_conv = nn.Conv2d(adapter_dim * 3, adapter_dim, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)

        self.proj_up = nn.Linear(self.adapter_dim, self.in_dim)
        self.gate = Gate(self.in_dim)
        self.reset_parameters()

    def forward(self, x):
        B, N, C = x.shape
        cls_token, patch_tokens = x[:, :1, :], x[:, 1:, :]
        x_mlp = self.mlp(x)

        H = W = int((N - 1) ** 0.5)
        patch_tokens = patch_tokens.view(B, H, W, C)
        x = self.proj_down(patch_tokens)
        x = x.permute(0, 3, 1, 2).contiguous()

        # 分奇偶样本
        left_x = x[0::2]
        right_x = x[1::2]

        # 路径1
        left_orig = self.branch_1x_left(left_x)
        right_orig = self.branch_1x_left(right_x)
        
        # 路径2
        left_mr = self.branch_2x_left(left_x)
        left_mr = F.interpolate(left_mr, size=left_x.shape[2:], mode='bilinear', align_corners=False)
        right_mr = self.branch_2x_right(right_x)
        right_mr = F.interpolate(right_mr, size=right_x.shape[2:], mode='bilinear', align_corners=False)

        # 路径3
        left_hr = self.branch_4x_left(left_x)
        left_hr = F.interpolate(left_hr, size=left_x.shape[2:], mode='bilinear', align_corners=False)
        right_hr = self.branch_4x_right(right_x)
        right_hr = F.interpolate(right_hr, size=right_x.shape[2:], mode='bilinear', align_corners=False)
        
        # 多尺度融合
        left_out = self.fusion_conv(torch.cat([left_orig, left_mr, left_hr], dim=1))
        right_out = self.fusion_conv(torch.cat([right_orig, right_mr, right_hr], dim=1))
        left_out = self.relu(left_out + left_x)
        right_out = self.relu(right_out + right_x)

        x_out = torch.empty_like(x)
        x_out[0::2] = left_out
        x_out[1::2] = right_out

        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        x_out = self.nonlinear(x_out)
        x_out = self.proj_up(x_out)
        x_out = x_out.view(B, H * W, C)

        x_mlp_cls, x_mlp_patch = x_mlp[:, :1, :], x_mlp[:, 1:, :]
        out_patch = x_mlp_patch + self.gate(x_mlp_patch)*x_out
        out = torch.cat([x_mlp_cls, out_patch], dim=1)

        return out

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj_down.weight)
        nn.init.xavier_uniform_(self.proj_up.weight)
        nn.init.constant_(self.proj_down.bias, 0)
        nn.init.constant_(self.proj_up.bias, 0)

