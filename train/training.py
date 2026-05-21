"""DOX 模型训练脚本.

支持：

* 单卡 / 多卡 DDP 训练 (``torchrun --nproc_per_node=N -m train.training``)；
* 从 RETFound 预训练权重初始化 ViT 主干，并冻结主干以执行 PEFT 微调；
* AdamW + 线性 warmup + 余弦退火的两阶段学习率调度；
* 非对称多标签损失 (``timm.AsymmetricLossMultiLabel``) 缓解类别不平衡；
* 梯度裁剪、best / latest 双检查点保存、训练曲线自动绘制。

使用示例
-------
单卡::

    python -m train.training \\
        --config configs/train_default.yaml \\
        --pre-checkpoint /path/to/RETFound_mae_natureCFP.pth \\
        --trainset       /path/to/ODIR-5K/Training_Images \\
        --train-csv-path /path/to/ODIR-5K/labels.xlsx \\
        --work-dir work_dir/dox_run1

多卡 DDP::

    torchrun --nproc_per_node=4 -m train.training \\
        --config configs/train_default.yaml \\
        --pre-checkpoint /path/to/RETFound_mae_natureCFP.pth \\
        --trainset       /path/to/ODIR-5K/Training_Images \\
        --train-csv-path /path/to/ODIR-5K/labels.xlsx \\
        --work-dir work_dir/dox_run1 \\
        --distributed
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from time import time

import matplotlib

matplotlib.use("Agg")  # 服务器环境无显示
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.optim as optim
from timm.loss import AsymmetricLossMultiLabel
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from datasets.orid5k import create_data_loaders
from modeling.model import MODEL
from train.utils import setup_logger


# ============================================================
# Configuration
# ============================================================
@dataclass
class TrainConfig:
    """训练超参数集合 (与论文实验设置默认一致)。"""

    # --- I/O ---
    work_dir:        str  = "work_dir/dox_default"
    exp_name:        str  = "dox_default"
    pre_checkpoint:  str | None = None
    checkpoint:      str | None = None
    resume:          str | None = None
    trainset:        str  = ""
    train_csv_path:  str  = ""

    # --- Model ---
    net:             str  = "MODEL"
    r:               int  = 4
    adapter_dim:     int  = 128
    dropout:         float = 0.1
    is_show:         bool = False

    # --- Optimisation ---
    epoch:           int  = 100
    warmup_epoch:    int  = 10
    batch_size:      int  = 32
    lr:              float = 1e-3
    weight_decay:    float = 0.01
    gradient_clip:   float = 0.1

    # --- Loss ---
    asl_gamma_pos:   float = 0.0
    asl_gamma_neg:   float = 4.0
    asl_clip:        float = 0.05

    # --- Data ---
    val_ratio:       float = 0.2
    num_workers:     int  = 8
    seed:            int  = 42

    # --- Hardware / Distributed ---
    distributed:     bool = False
    gpu_device:      int  = 0

    # --- Logging ---
    save_interval:   int | None = None

    # 内部字段
    _ddp_args: dict = field(default_factory=dict)


# ============================================================
# Distributed helpers
# ============================================================
def setup_ddp() -> torch.device:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


# ============================================================
# Builders
# ============================================================
def build_model(cfg: TrainConfig) -> torch.nn.Module:
    if cfg.net != "MODEL":
        raise ValueError(f"Unsupported net type: {cfg.net!r}; only 'MODEL' is implemented.")

    net = MODEL(
        check_path  = cfg.pre_checkpoint,
        embed_dim   = 1024,
        out_chans   = 256,
        num_classes = 8,
        r           = cfg.r,
        adapter_dim = cfg.adapter_dim,
        dropout     = cfg.dropout,
    )

    if cfg.checkpoint is not None:
        state = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)["model"]
        msg = net.load_state_dict(state, strict=False)
        logging.getLogger(__name__).info("Loaded checkpoint %s: %s", cfg.checkpoint, msg)

    return net


def warmup_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_iters: int,
    warmup_factor: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """线性 warmup —— iter 0 时 lr = warmup_factor·lr，warmup_iters 时达到完整 lr。"""

    def f(step: int) -> float:
        if step >= warmup_iters:
            return 1.0
        alpha = float(step) / max(warmup_iters, 1)
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    iters_per_epoch: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    warmup_iters = cfg.warmup_epoch * iters_per_epoch
    cosine_iters = max((cfg.epoch - cfg.warmup_epoch) * iters_per_epoch, 1)

    warmup = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor=0.1)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_iters)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_iters]
    )


def build_optimizer(net: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    return optim.AdamW(
        net.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=cfg.weight_decay,
    )


# ============================================================
# Train / Val loops
# ============================================================
def _interleave_eyes(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """按论文双眼批次交错机制将左右眼图像组成 2B 输入。"""
    B, C, H, W = left.shape
    samples = torch.empty(2 * B, C, H, W, device=left.device, dtype=left.dtype)
    samples[0::2] = left
    samples[1::2] = right
    return samples


def _reduce_loss(loss: torch.Tensor) -> float:
    if dist.is_initialized():
        red = loss.detach().clone()
        dist.all_reduce(red, op=dist.ReduceOp.SUM)
        red /= dist.get_world_size()
        return red.item()
    return loss.item()


def train_one_epoch(
    net:        torch.nn.Module,
    loader:     torch.utils.data.DataLoader,
    optimizer:  torch.optim.Optimizer,
    scheduler:  torch.optim.lr_scheduler._LRScheduler,
    lossfunc:   torch.nn.Module,
    device:     torch.device,
    cfg:        TrainConfig,
    epoch:      int,
) -> float:
    net.train()
    pbar = tqdm(loader, disable=not is_main_process())
    epoch_losses: list[float] = []

    for batch in pbar:
        optimizer.zero_grad()
        left   = batch["left_image"].to(device, non_blocking=True)
        right  = batch["right_image"].to(device, non_blocking=True)
        target = batch["target"].float().to(device, non_blocking=True)

        samples = _interleave_eyes(left, right)
        logits  = net(samples)
        loss    = lossfunc(logits, target)

        loss.backward()
        if cfg.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.gradient_clip)
        optimizer.step()
        scheduler.step()

        loss_val = _reduce_loss(loss)
        epoch_losses.append(loss_val)

        if is_main_process():
            pbar.set_description(
                f"Epoch [{epoch+1}/{cfg.epoch}]  loss: {loss_val:.4f}  "
                f"lr: {optimizer.param_groups[0]['lr']:.2e}"
            )

    return sum(epoch_losses) / len(epoch_losses)


@torch.no_grad()
def validate(
    net:      torch.nn.Module,
    loader:   torch.utils.data.DataLoader,
    lossfunc: torch.nn.Module,
    device:   torch.device,
    epoch:    int,
    cfg:      TrainConfig,
) -> float:
    net.eval()
    pbar = tqdm(loader, disable=not is_main_process())
    losses: list[float] = []
    for batch in pbar:
        left   = batch["left_image"].to(device, non_blocking=True)
        right  = batch["right_image"].to(device, non_blocking=True)
        target = batch["target"].float().to(device, non_blocking=True)

        samples = _interleave_eyes(left, right)
        loss = lossfunc(net(samples), target)
        loss_val = _reduce_loss(loss)
        losses.append(loss_val)

        if is_main_process():
            pbar.set_description(f"Val   [{epoch+1}/{cfg.epoch}]  loss: {loss_val:.4f}")

    return sum(losses) / len(losses)


# ============================================================
# Checkpoint & plotting
# ============================================================
def save_checkpoint(
    work_dir:    Path,
    net:         torch.nn.Module,
    optimizer:   torch.optim.Optimizer,
    scheduler:   torch.optim.lr_scheduler._LRScheduler,
    epoch:       int,
    train_loss:  float,
    val_loss:    float,
    best_loss:   float,
    save_interval: int | None,
    total_epoch: int,
) -> float:
    model_weights = unwrap_model(net).state_dict()
    ckpt = {
        "model":     model_weights,
        "epoch":     epoch,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "loss":      train_loss,
        "val_loss":  val_loss,
        "best_loss": best_loss,
    }

    if val_loss < best_loss:
        logging.getLogger(__name__).info("New best val_loss: %.4f → %.4f", best_loss, val_loss)
        best_loss = val_loss
        ckpt["best_loss"] = best_loss
        torch.save(ckpt, work_dir / "checkpoint_best.pth")

    if save_interval is not None and (epoch % save_interval == 0 or epoch == total_epoch - 1):
        torch.save(ckpt, work_dir / f"checkpoint_epoch_{epoch:03d}.pth")

    torch.save(ckpt, work_dir / "checkpoint_latest.pth")
    return best_loss


def plot_curves(train_losses: list[float], val_losses: list[float], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses,   label="Val Loss")
    ax.set_title("Asymmetric Multi-label Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main(cfg: TrainConfig) -> None:
    device = setup_ddp() if cfg.distributed else torch.device("cuda", cfg.gpu_device)

    work_dir = Path(cfg.work_dir) / cfg.exp_name
    work_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(str(work_dir))

    if cfg.distributed:
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        rank, world = 0, 1

    train_loader, val_loader, train_sampler, _ = create_data_loaders(
        args=_NamespaceFromDataclass(cfg),
        rank=rank,
        world_size=world,
        pin_memory=True,
    )

    net = build_model(cfg)

    if cfg.is_show and is_main_process():
        for name, p in net.named_parameters():
            logger.info("  %-80s shape=%s trainable=%s", name, tuple(p.shape), p.requires_grad)
        total = sum(p.numel() for p in net.parameters()) / 1e6
        trainable = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6
        logger.info("Total params: %.2fM    Trainable: %.2fM  (%.1f%%)",
                    total, trainable, 100 * trainable / total)

    net = net.to(device)

    optimizer = build_optimizer(net, cfg)
    scheduler = build_scheduler(optimizer, cfg, iters_per_epoch=len(train_loader))

    lossfunc = AsymmetricLossMultiLabel(
        gamma_pos=cfg.asl_gamma_pos,
        gamma_neg=cfg.asl_gamma_neg,
        clip=cfg.asl_clip,
    )

    # ---- Resume ----
    start_epoch, best_loss = 0, float("inf")
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device)
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt["best_loss"]
        if is_main_process():
            logger.info("Resumed from %s (epoch=%d, best_loss=%.4f)",
                        cfg.resume, start_epoch, best_loss)

    if cfg.distributed:
        net = DDP(
            net,
            device_ids   = [int(os.environ["LOCAL_RANK"])],
            output_device= int(os.environ["LOCAL_RANK"]),
            find_unused_parameters=True,
        )

    # ---- Training loop ----
    train_losses, val_losses = [], []
    for epoch in range(start_epoch, cfg.epoch):
        t0 = time()
        if cfg.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            net, train_loader, optimizer, scheduler, lossfunc, device, cfg, epoch
        )
        val_loss = validate(net, val_loader, lossfunc, device, epoch, cfg)
        epoch_time = time() - t0

        train_losses.append(train_loss)
        val_losses  .append(val_loss)

        if is_main_process():
            logger.info(
                "Epoch [%d/%d]  train=%.4f  val=%.4f  time=%.1fs",
                epoch + 1, cfg.epoch, train_loss, val_loss, epoch_time,
            )
            best_loss = save_checkpoint(
                work_dir, net, optimizer, scheduler, epoch,
                train_loss, val_loss, best_loss,
                cfg.save_interval, cfg.epoch,
            )
            plot_curves(train_losses, val_losses, work_dir / "train_loss.png")

        if cfg.distributed:
            dist.barrier()

    cleanup_ddp()


# ============================================================
# CLI plumbing
# ============================================================
class _NamespaceFromDataclass(argparse.Namespace):
    """把 ``TrainConfig`` 包装成 ``argparse.Namespace``，复用旧版数据加载接口。"""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        for k, v in cfg.__dict__.items():
            setattr(self, k, v)


def _parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="DOX training script.")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML 配置文件路径；命令行参数会覆盖 YAML 配置。")

    # 组别 1: I/O
    parser.add_argument("--work-dir",        type=str, default=None)
    parser.add_argument("--exp-name",        type=str, default=None)
    parser.add_argument("--pre-checkpoint",  type=str, default=None)
    parser.add_argument("--checkpoint",      type=str, default=None)
    parser.add_argument("--resume",          type=str, default=None)
    parser.add_argument("--trainset",        type=str, default=None)
    parser.add_argument("--train-csv-path",  type=str, default=None)

    # 组别 2: Model
    parser.add_argument("--net",             type=str, default=None, choices=["MODEL"])
    parser.add_argument("-r",                type=int, default=None, dest="r")
    parser.add_argument("--adapter-dim",     type=int, default=None)
    parser.add_argument("--dropout",         type=float, default=None)
    parser.add_argument("--is-show",         action="store_true")

    # 组别 3: Optimisation
    parser.add_argument("--epoch",           type=int, default=None)
    parser.add_argument("--warmup-epoch",    type=int, default=None)
    parser.add_argument("--batch-size",      type=int, default=None)
    parser.add_argument("--lr",              type=float, default=None)
    parser.add_argument("--weight-decay",    type=float, default=None)
    parser.add_argument("--gradient-clip",   type=float, default=None)

    # 组别 4: Data
    parser.add_argument("--val-ratio",       type=float, default=None)
    parser.add_argument("--num-workers",     type=int, default=None)
    parser.add_argument("--seed",            type=int, default=None)

    # 组别 5: Distributed
    parser.add_argument("--distributed",     action="store_true")
    parser.add_argument("--gpu-device",      type=int, default=None)

    # 组别 6: Logging
    parser.add_argument("--save-interval",   type=int, default=None)

    args = parser.parse_args()

    cfg_dict: dict = {}
    if args.config:
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            cfg_dict.update(yaml.safe_load(f) or {})

    for k, v in vars(args).items():
        if k == "config":
            continue
        if v in (None, False):
            continue
        cfg_dict[k.replace("-", "_")] = v

    cfg = TrainConfig(**cfg_dict)

    if not cfg.trainset or not cfg.train_csv_path:
        parser.error("`trainset` and `train_csv_path` are required "
                     "(set via CLI or YAML config).")
    return cfg


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    main(_parse_args())
