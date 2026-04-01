from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _pil_rgb(path: str) -> Image.Image:
    with Image.open(path) as img:
        if img.mode == "P" and "transparency" in img.info:
            return img.convert("RGBA").convert("RGB")
        return img.convert("RGB")


def get_transforms(
    img_size: int = 224,
    randaugment: bool = True,
    randaugment_num_ops: int = 2,
    randaugment_magnitude: int = 9,
    random_erasing_p: float = 0.0,
):
    aug_list = [
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    ]
    if randaugment:
        aug_list.insert(
            1,
            transforms.RandAugment(
                num_ops=randaugment_num_ops, magnitude=randaugment_magnitude
            ),
        )
    aug_list.append(transforms.ToTensor())
    if random_erasing_p and random_erasing_p > 0:
        aug_list.append(
            transforms.RandomErasing(
                p=random_erasing_p, scale=(0.02, 0.2), ratio=(0.3, 3.3)
            )
        )
    aug_list.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    train_tf = transforms.Compose(aug_list)
    val_tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, val_tf


def stratified_indices(dataset: datasets.ImageFolder, val_ratio: float, seed: int):
    targets = np.array([t for _, t in dataset.samples])
    idx = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_ratio, stratify=targets, random_state=seed
    )
    return train_idx.tolist(), val_idx.tolist()


def make_loaders(
    data_root: str | Path,
    batch_size_train: int,
    batch_size_val: int,
    img_size: int = 224,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    randaugment: bool = True,
    randaugment_num_ops: int = 2,
    randaugment_magnitude: int = 9,
    random_erasing_p: float = 0.0,
):
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    train_tf, val_tf = get_transforms(
        img_size,
        randaugment=randaugment,
        randaugment_num_ops=randaugment_num_ops,
        randaugment_magnitude=randaugment_magnitude,
        random_erasing_p=random_erasing_p,
    )
    full_train = datasets.ImageFolder(
        str(data_root), transform=train_tf, loader=_pil_rgb
    )
    full_val_ref = datasets.ImageFolder(
        str(data_root), transform=val_tf, loader=_pil_rgb
    )
    train_idx, val_idx = stratified_indices(full_train, val_ratio=val_ratio, seed=seed)
    train_set = Subset(full_train, train_idx)
    val_set = Subset(full_val_ref, val_idx)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, full_train.classes
