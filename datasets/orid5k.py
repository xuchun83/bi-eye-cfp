"""ODIR-5K 训练 / 验证数据集封装.

数据格式约定：

* ``images_dir/``  下平铺存放所有左右眼 CFP 图像；
* ``csv_path``     为 Excel 文件，至少包含以下列：

  ============== ============================================================
  Left-Fundus    左眼图像文件名 (相对于 ``images_dir``)
  Right-Fundus   右眼图像文件名
  N D G C A H M O  8 列二值标签 (one-hot 多标签)
  ============== ============================================================

`mode="train"` 时会施加 RandomResizedCrop / 翻转 / 颜色抖动 / 旋转等
数据增广；`mode="val"` 或 `mode="test"` 仅做 Resize+CenterCrop。

本文件同时提供两种数据划分入口：

* :func:`create_data_loaders` ── 单次随机划分 (论文实验使用)；
* :func:`create_stratified_cv_data_loaders` ── 分层 K 折交叉验证。
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms


class Orid5k(Dataset):
    """ODIR-5K 双眼多标签数据集.

    Parameters
    ----------
    images_dir:
        包含全部左 / 右眼图像的目录。
    csv_path:
        Excel 标注文件路径。
    train_indices, val_indices:
        若给定，仅保留对应行；二者不应同时为非 ``None``。
    mode:
        ``"train"`` 启用增广，否则使用 (Resize+CenterCrop) 推理变换。
    """

    def __init__(
        self,
        images_dir:    str,
        csv_path:      str,
        train_indices: Optional[np.ndarray] = None,
        val_indices:   Optional[np.ndarray] = None,
        mode:          str = "train",
    ) -> None:
        self.mode = mode
        self.images_dir = images_dir
        self.datas = pd.read_excel(csv_path) 
        if train_indices is not None:  
            self.datas = self.datas.iloc[train_indices]  # 选取训练数据  
        elif val_indices is not None:  
            self.datas = self.datas.iloc[val_indices]  # 选取验证数据  

        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),              # 随机裁剪并缩放
                transforms.RandomHorizontalFlip(p=0.5),         # 随机水平翻转
                transforms.RandomVerticalFlip(p=0.5),           # 随机垂直翻转（可选）
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),     # 随机亮度/对比度/饱和度/色调变化（可选）
                transforms.RandomRotation(15),                  # 随机旋转（可选）
                transforms.ToTensor(),                          # 转张量
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),                         # 先把短边缩放到 256
                transforms.CenterCrop(224),                     # 再裁出中心 224×224
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
            ])
        
    def __getitem__(self, index):
        row = self.datas.iloc[[index]]
        label = np.array(row[['N','D','G', 'C', 'A', 'H', 'M', 'O']])
        label = torch.tensor(label[0])
        #print(label)
        left_imgpath = os.path.join(self.images_dir, row['Left-Fundus'].values[0])
        # left_imgpath = os.path.join(self.root, row['Left-Fundus'])
        right_imgpath = os.path.join(self.images_dir, row['Right-Fundus'].values[0])
        # right_imgpath = os.path.join(self.root, row['Right-Fundus'])

        # left_img = Image.open(left_imgpath.replace(".jpg", ".jpg.jpg")).convert('RGB')
        # right_img = Image.open(right_imgpath.replace(".jpg", ".jpg.jpg")).convert('RGB')
        left_img = Image.open(left_imgpath).convert('RGB')
        right_img = Image.open(right_imgpath).convert('RGB')
    
        left_img = self.transform(left_img)
        right_img = self.transform(right_img)

        data = {'left_image':left_img, 'right_image':right_img, 'target': label, 'idx': left_imgpath.replace("_left.jpg", "")}
        return data

    def __len__(self):
        return len(self.datas)

def create_data_loaders(
    args,
    rank: int = 0,          # 当前进程的 rank
    world_size: int = 1,    # 总进程数
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, Optional[DistributedSampler], Optional[DistributedSampler]] :
    data = pd.read_excel(args.train_csv_path) 
    train_indices, val_indices = train_test_split(data.index, test_size=args.val_ratio, random_state=args.seed)
    
    train_dataset = Orid5k(args.trainset, args.train_csv_path, train_indices=train_indices, mode="train")
    val_dataset = Orid5k(args.trainset, args.train_csv_path, val_indices=val_indices, mode="val")

    # ---------------- 分布式采样器 ----------------
    train_sampler = DistributedSampler(train_dataset, 
                                       num_replicas=world_size, 
                                       rank=rank, 
                                       shuffle=True) if args.distributed and world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, 
                                     num_replicas=world_size, 
                                     rank=rank, 
                                     shuffle=False) if args.distributed and world_size > 1 else None

    # ---------------- DataLoader ----------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, train_sampler, val_sampler


def create_stratified_cv_data_loaders(
    args,
    n_splits: int = 5,
    fold: int = 0,
    rank: int = 0,
    world_size: int = 1,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, Optional[DistributedSampler], Optional[DistributedSampler]]:
    """
    分层K折交叉验证，保持每折的类别分布一致
    """
    # 读取数据
    data = pd.read_excel(args.train_csv_path)
    
    # 获取标签（假设第一列是文件名，后面是标签）
    labels = data.iloc[:, 1:].values  # 调整索引根据你的数据格式
    # 或者如果是多标签，可以创建一个综合标签用于分层
    # 例如：将多标签转换为字符串表示
    label_str = labels.astype(str).sum(axis=1)
    
    # 创建分层K折分割器
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    
    # 获取当前折的训练和验证索引
    all_indices = list(range(len(data)))
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_indices, label_str)):
        if fold_idx == fold:
            train_indices = train_idx
            val_indices = val_idx
            break
    else:
        raise ValueError(f"Fold {fold} not found in {n_splits}-fold CV")
    
    # 创建完整数据集
    full_dataset = Orid5k(args.trainset, args.train_csv_path, mode="train")
    
    # 创建子集
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    print(f"Fold {fold + 1}/{n_splits}: Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # 其余部分与方案1相同...
    train_sampler = DistributedSampler(train_dataset, 
                                     num_replicas=world_size, 
                                     rank=rank, 
                                     shuffle=True) if args.distributed and world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, 
                                   num_replicas=world_size, 
                                   rank=rank, 
                                   shuffle=False) if args.distributed and world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, train_sampler, val_sampler


def unnormalize(img_tensor):
    """还原经过 ImageNet Normalize 的 Tensor 图像"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return img_tensor * std + mean  # 反归一化

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--trainset", type=str, default='datademo', help='path to train dataset')
    parser.add_argument("--train_csv_path", type=str, default='datademo/Traning_Dataset.xlsx', help='path to train csv file')
    parser.add_argument("--val_ratio", type=float, default=0.2, help='val ratio')
    parser.add_argument('-batch_size', type=int, default=1, help='batch size for dataloader')
    parser.add_argument('-seed', type=int, default=42, help='random seed')
    parser.add_argument('-num_workers', type=int, default=0)
    parser.add_argument('-distributed', type=bool, default=False, help="use distributed device to train or not")
    args = parser.parse_args()


    train_loader, val_loader, train_sampler, val_sampler = create_data_loaders(args)

    print(f"Train loader batches: {len(train_loader)}, Val loader batches: {len(val_loader)}")

    for batch_idx, batch in enumerate(val_loader):
        left_image = batch['left_image'] 
        right_image = batch['right_image']
        target = batch['target']
            
        print(f"Batch {batch_idx}:")
        print(f"left_image shape: {left_image.shape}")
        print(f"right_image shape: {right_image.shape}")
        print(f"target shape: {target.shape}")
        print(target.dtype)
        
        plt.figure(figsize=(20,10))
        plt.subplot(1,2,1)
        plt.imshow(unnormalize(left_image)[0].permute(1, 2, 0).clamp(0, 1))
        plt.title("left_image")
        plt.axis("off")

        plt.subplot(1,2,2)
        plt.imshow(unnormalize(right_image)[0].permute(1, 2, 0).clamp(0, 1))
        plt.title("right_image")
        plt.axis("off")

        plt.show()
            
        if batch_idx == 0:  
            break