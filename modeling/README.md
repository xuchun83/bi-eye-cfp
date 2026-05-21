# `modeling/` — 模型组件

实现论文方法论部分的全部模块。

## 文件清单

| 文件 | 内容 |
|---|---|
| [`model.py`](model.py)         | 顶层 `MODEL` 类与 `build_model()`，组合下方所有子模块 |
| [`models_vit.py`](models_vit.py) | RETFound MAE 主干 (ViT-Large/16)；仅保留前 12 层、去掉解码器 |
| [`fine_tune.py`](fine_tune.py) | LoRA 适配器、多尺度 Conv-Adapter、轻量 Gate |
| [`fusion.py`](fusion.py)       | DBFEM 双流交互融合 + 标准 Transformer 块 |
| [`common.py`](common.py)       | LayerNorm2d 等共享工具 |

## 数据流

```
左眼图像  ─┐
            ├── 交错拼成 (2B, 3, 224, 224)
右眼图像  ─┘
                │
                ▼
   ┌──────────────────────────────────────┐
   │ ViT-L/16 (RETFound, 前 12 层, 主干冻结) │
   │   每层 Block:                         │
   │     Self-Attn  ← LoRA (rank r)        │
   │     MLP        ← 双路 Conv-Adapter    │
   └──────────────────────────────────────┘
                │
   patch tokens │  cls token
                │     │
                ▼     ▼
      neck (1x1 + 3x3 conv) → DBFEM 融合 → mean
                                 │
              cls_fused = (left + right) / 2
                                 │
                                 ▼
              Concat → LayerNorm → Dropout → MLP head
                                 │
                                 ▼
                          8 维 logits
```

## 设计要点

1. **左右眼参数完全共享**：使用 `samples[0::2] / [1::2]` 实现交错，避免双流网络的额外开销；
2. **PEFT 哲学**：主干冻结，只训 LoRA + Adapter + Neck + Fusion + Classifier；
3. **DBFEM 是 patch-level 融合**：而非分类阶段融合 (后者已被消融实验证明效果更差)；
4. **cls token 走简单平均路径**：作为对 patch 融合特征的补充，类似 ResNet 的恒等映射。

## 参数预算

| 模块 | 训练状态 | 大致参数量 (`r=4, adapter_dim=128`) |
|---|---|---:|
| RETFound 主干 (前 12 层)   | ❄ 冻结      | ~150 M |
| LoRA (24 × Q/V × 2 × r=4) | ✅ 可训      | ~0.2 M |
| Conv-Adapter (12 × multi) | ✅ 可训      | ~2.5 M |
| Neck + Fusion + Head      | ✅ 可训      | ~0.8 M |
| **可训占比**              |              | **< 2 %** ✓ |

## 引用

* RETFound: Zhou et al., *Nature* 622, 156–163 (2023)
* LoRA:     Hu et al., *ICLR* (2022)
* Adapter:  Houlsby et al., *ICML* (2019)
