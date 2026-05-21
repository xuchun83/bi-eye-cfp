# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# -------------------------------------------------------- 

from functools import partial

import timm.models.vision_transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, g_pool=False, keep_layers=12, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        # 只保留前 keep_layers 层
        self.blocks = nn.Sequential(*list(self.blocks.children())[:keep_layers])

        self.g_pool = g_pool
        if self.g_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

        del self.norm  # remove the original norm
        del self.head

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        # if self.g_pool:
        #     x = x[:, 1:, :].mean(dim=1,keepdim=True)  # global pool without cls token
        #     outcome = self.fc_norm(x)
        # else:
        #     x = self.norm(x)
        #     outcome = x[:, 0]

        # return outcome
        # cls_token, patch_tokens = x[:, :1, :], x[:, 1:, :]
        # return cls_token, patch_tokens
        return x


def RETFound_mae(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



def RETFound_dinov2(args, **kwargs):
    model = timm.create_model(
        'vit_large_patch14_dinov2.lvd142m',
        pretrained=True,
        img_size=224,
        **kwargs
    )
    return model



if __name__ == "__main__":
    import numpy as np

    model = RETFound_mae()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"总参数数量：{params/1e6}M")
    # 获取每个层的参数数量和名称
    for name, param in model.named_parameters():
        print(f"层名称: {name}, 参数形状: {param.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6}M")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params / 1e6}M")

    image_shape = (2, 3, 224, 224)  # (batch_size, channels, height, width)
    image = torch.randn(image_shape)
    out = model.forward_features(image)
    print(out.shape)