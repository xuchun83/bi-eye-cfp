"""单样本 (一对左右眼) 推理脚本.

提供两种使用方式：

1. **作为库函数**::

       from inference_single import inference_single
       from modeling.model import build_model

       model = build_model("checkpoints/checkpoint_latest.pth", device="cuda")
       result = inference_single(
           model,
           left_image_path  = "datademo/raw_data/0_left.jpg",
           right_image_path = "datademo/raw_data/0_right.jpg",
           device           = "cuda",
       )
       print(result["classes"])    # 例如：['N']

2. **作为 CLI**::

       python -m inference_single \\
           --checkpoint  checkpoints/checkpoint_latest.pth \\
           --left-image  datademo/raw_data/0_left.jpg \\
           --right-image datademo/raw_data/0_right.jpg \\
           --device cuda

返回的 ``sample_result`` 字典字段说明：

* ``probabilities``: ``List[float]``，每个类别的 sigmoid 概率，顺序为 ``['N','D','G','C','A','H','M','O']``
* ``predictions``:   ``List[int]``，与 probabilities 同顺序的 one-hot 预测
* ``classes``:       ``List[str]``，被预测为阳性的疾病类别名称
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image

from modeling.model import build_model

logger = logging.getLogger(__name__)

CLASS_NAMES: list[str] = ["N", "D", "G", "C", "A", "H", "M", "O"]
DEFAULT_THRESHOLDS: dict[str, float] = {
    "N": 0.7, "D": 0.7, "G": 0.5, "C": 0.7,
    "A": 0.7, "H": 0.4, "M": 0.8, "O": 0.6,
}


@torch.no_grad()
def inference_single(
    model:             torch.nn.Module,
    left_image_path:   str,
    right_image_path:  str,
    device:            str | torch.device = "cuda",
    thresholds:        dict[str, float] | None = None,
) -> dict[str, object]:
    """对一对左右眼 CFP 图像执行多标签预测。

    Parameters
    ----------
    model:
        通过 :func:`modeling.model.build_model` 构建的模型实例。
    left_image_path, right_image_path:
        左眼 / 右眼图像路径，支持 PIL 可读取的任意格式。
    device:
        推理 device，必须与构建模型时使用的 device 一致。
    thresholds:
        类别独立阈值；为 ``None`` 时使用论文默认 (:data:`DEFAULT_THRESHOLDS`)。

    Returns
    -------
    dict
        含 ``probabilities`` / ``predictions`` / ``classes`` 三个键。
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS
    device = torch.device(device)

    left_img  = Image.open(left_image_path ).convert("RGB")
    right_img = Image.open(right_image_path).convert("RGB")
    left_t    = model.transform(left_img ).unsqueeze(0)
    right_t   = model.transform(right_img).unsqueeze(0)

    samples = torch.empty(2, *left_t.shape[1:], device=device, dtype=left_t.dtype)
    samples[0::2] = left_t.to(device)
    samples[1::2] = right_t.to(device)

    logits = model(samples)
    probs  = torch.sigmoid(logits).cpu().numpy()

    preds = np.zeros_like(probs)
    for i, name in enumerate(CLASS_NAMES):
        preds[:, i] = (probs[:, i] >= thresholds[name]).astype(int)

    return {
        "probabilities": probs[0].tolist(),
        "predictions":   preds[0].astype(int).tolist(),
        "classes":       [CLASS_NAMES[i] for i, p in enumerate(preds[0]) if p == 1],
    }


# ============================================================
# CLI
# ============================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DOX single-sample inference.")
    parser.add_argument("--checkpoint",   required=True, type=str,
                        help="模型权重 (.pth) 路径")
    parser.add_argument("--left-image",   required=True, type=str)
    parser.add_argument("--right-image",  required=True, type=str)
    parser.add_argument("--device",       default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--json-output",  default=None, type=str,
                        help="若指定，将结果写入该 JSON 文件")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    if args.device == "cuda" and device == "cpu":
        logger.warning("CUDA unavailable, falling back to CPU.")

    model = build_model(checkpoint_path=args.checkpoint, device=device)
    logger.info("Loaded checkpoint from %s", args.checkpoint)

    t0 = perf_counter()
    result = inference_single(
        model            = model,
        left_image_path  = args.left_image,
        right_image_path = args.right_image,
        device           = device,
    )
    logger.info("Inference completed in %.2fs", perf_counter() - t0)

    fmt_probs = [f"{p:.4f}" for p in result["probabilities"]]
    print("\n类别顺序     :", CLASS_NAMES)
    print("Probabilities:", fmt_probs)
    print("Predictions  :", result["predictions"])
    print("Predicted    :", result["classes"] or ["(无阳性)"])

    if args.json_output:
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info("Saved JSON result to %s", args.json_output)


if __name__ == "__main__":
    main()
