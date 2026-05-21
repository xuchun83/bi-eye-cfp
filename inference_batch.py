"""批量推理脚本.

输入：

* ``--images-dir``: 包含所有左 / 右眼图像的文件夹；
* ``--csv-path``:   Excel/CSV 文件，含 ``Left-Fundus`` 与 ``Right-Fundus`` 两列，
  每行一对样本。

输出：

* 控制台逐样本打印；
* 可选 ``--output-csv``：每行 ``sample_id, prob_*, pred_*, classes``。

库函数形式::

    from inference_batch import inference_batch
    from modeling.model import build_model

    model = build_model("checkpoints/checkpoint_latest.pth", device="cuda")
    results = inference_batch(
        model       = model,
        images_dir  = "datademo/raw_data",
        csv_path    = "datademo/raw_data/Traning_Dataset.xlsx",
        batch_size  = 16,
        device      = "cuda",
    )
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.inference_datasets import Infer_Orid5k
from modeling.model import build_model

logger = logging.getLogger(__name__)

CLASS_NAMES: list[str] = ["N", "D", "G", "C", "A", "H", "M", "O"]
DEFAULT_THRESHOLDS: dict[str, float] = {
    "N": 0.7, "D": 0.7, "G": 0.5, "C": 0.7,
    "A": 0.7, "H": 0.4, "M": 0.8, "O": 0.6,
}


@torch.no_grad()
def inference_batch(
    model:        torch.nn.Module,
    images_dir:   str,
    csv_path:     str,
    batch_size:   int = 16,
    num_workers:  int = 4,
    device:       str | torch.device = "cuda",
    thresholds:   dict[str, float] | None = None,
) -> list[dict[str, object]]:
    """对一个文件夹中的所有样本批量推理。

    Returns
    -------
    list of dict
        每个元素含 ``idx`` / ``probabilities`` / ``predictions`` / ``classes``。
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS
    device = torch.device(device)

    dataset = Infer_Orid5k(images_dir=images_dir, csv_path=csv_path)
    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = (device.type == "cuda"),
    )
    logger.info("Loaded %d samples (%d batches).", len(dataset), len(loader))

    results: list[dict[str, object]] = []
    for batch in tqdm(loader, desc="Inference"):
        left   = batch["left_image"].to(device, non_blocking=True)
        right  = batch["right_image"].to(device, non_blocking=True)
        idxs   = batch["idx"]

        B, C, H, W = left.shape
        samples = torch.empty(2 * B, C, H, W, device=device, dtype=left.dtype)
        samples[0::2] = left
        samples[1::2] = right

        logits = model(samples)
        probs = torch.sigmoid(logits).cpu().numpy()

        preds = np.zeros_like(probs)
        for i, name in enumerate(CLASS_NAMES):
            preds[:, i] = (probs[:, i] >= thresholds[name]).astype(int)

        for i in range(B):
            results.append({
                "idx":           idxs[i],
                "probabilities": probs[i].tolist(),
                "predictions":   preds[i].astype(int).tolist(),
                "classes":       [CLASS_NAMES[j] for j, p in enumerate(preds[i]) if p == 1],
            })

    return results


def save_results_to_csv(results: list[dict[str, object]], output_csv: str | Path) -> None:
    """把 :func:`inference_batch` 的结果落盘为整齐的 CSV。"""
    rows = []
    for r in results:
        row = {"sample_id": r["idx"]}
        row.update({f"prob_{c}": r["probabilities"][i] for i, c in enumerate(CLASS_NAMES)})
        row.update({f"pred_{c}": r["predictions"]  [i] for i, c in enumerate(CLASS_NAMES)})
        row["classes"] = "|".join(r["classes"])  # type: ignore[arg-type]
        rows.append(row)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    logger.info("Saved %d predictions to %s", len(rows), output_csv)


# ============================================================
# CLI
# ============================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DOX batch inference.")
    parser.add_argument("--checkpoint",  required=True, type=str)
    parser.add_argument("--images-dir",  required=True, type=str)
    parser.add_argument("--csv-path",    required=True, type=str,
                        help="Excel/CSV 文件，含 Left-Fundus / Right-Fundus 两列")
    parser.add_argument("--output-csv",  default=None, type=str,
                        help="若指定，将逐样本结果写入该 CSV 文件")
    parser.add_argument("--batch-size",  type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device",      default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--quiet",       action="store_true",
                        help="不在控制台打印逐样本结果")
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

    results = inference_batch(
        model       = model,
        images_dir  = args.images_dir,
        csv_path    = args.csv_path,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        device      = device,
    )

    if not args.quiet:
        for r in results:
            print(f"\n样本 {r['idx']}")
            print(f"  Probabilities : {[f'{p:.4f}' for p in r['probabilities']]}")
            print(f"  Predictions   : {r['predictions']}")
            print(f"  Classes       : {r['classes'] or ['(无阳性)']}")

    if args.output_csv:
        save_results_to_csv(results, args.output_csv)


if __name__ == "__main__":
    main()
