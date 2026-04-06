from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.optim import AdamW

from src.data import make_loaders
from src.models import build_student
from src.utils import (
    dataloader_augment_kwargs,
    device_synchronize,
    get_device,
    load_config,
    project_root,
    resolve_data_root,
)


def run_train_epoch_ce(
    model,
    loader,
    criterion,
    optimizer,
    device,
    mixup_alpha: float,
    batch_scheduler=None,
):
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total = 0
    use_mixup = mixup_alpha > 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_mixup and images.size(0) > 1:
            lam = float(
                torch.distributions.Beta(mixup_alpha, mixup_alpha)
                .sample()
                .item()
            )
            idx = torch.randperm(images.size(0), device=device)
            mixed = lam * images + (1.0 - lam) * images[idx]
            ta, tb = targets, targets[idx]
            logits = model(mixed)
            loss = lam * criterion(logits, ta) + (1.0 - lam) * criterion(
                logits, tb
            )
            pred = logits.argmax(1)
            total_correct += (
                lam * (pred == ta).float() + (1.0 - lam) * (pred == tb).float()
            ).sum().item()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            total_correct += (logits.argmax(1) == targets).sum().item()

        loss.backward()
        optimizer.step()
        if batch_scheduler is not None:
            batch_scheduler.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total += bs

    device_synchronize(device)
    return total_loss / max(total, 1), total_correct / max(total, 1)


@torch.no_grad()
def run_val_epoch(model, loader, device):
    model.eval()
    total_correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        total_correct += (logits.argmax(1) == targets).sum().item()
        total += images.size(0)
    device_synchronize(device)
    return total_correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train MobileNetV3-Small with cross-entropy only (no KD). "
            "Saves to checkpoints.student_baseline — does not overwrite student_best.pt."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override student epochs from config",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = project_root()
    baseline_path = root / cfg["checkpoints"]["student_baseline"]
    baseline_path.parent.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    data_root = resolve_data_root(cfg)
    scfg = cfg["student"]
    epochs = int(args.epochs) if args.epochs is not None else int(scfg["epochs"])
    train_loader, val_loader, classes = make_loaders(
        data_root,
        batch_size_train=scfg["batch_size_train"],
        batch_size_val=scfg["batch_size_val"],
        img_size=cfg["img_size"],
        val_ratio=cfg["val_ratio"],
        seed=cfg["seed"],
        num_workers=cfg["num_workers"],
        **dataloader_augment_kwargs(cfg),
    )
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")
    print(f"Baseline checkpoint (CE-only): {baseline_path}", flush=True)

    ls = float(scfg.get("label_smoothing", 0.0))
    criterion = nn.CrossEntropyLoss(label_smoothing=ls)
    student = build_student(num_classes).to(device)
    base_lr = float(scfg["lr"])
    max_lr = float(scfg.get("max_lr", base_lr))
    sched_name = str(scfg.get("scheduler", "cosine")).lower()
    opt_lr = max_lr / 25.0 if sched_name == "onecycle" else base_lr
    optimizer = AdamW(
        student.parameters(),
        lr=opt_lr,
        weight_decay=float(scfg["weight_decay"]),
    )
    if sched_name == "onecycle":
        batch_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.15,
            div_factor=25.0,
            final_div_factor=1e4,
        )
        epoch_scheduler = None
    else:
        batch_scheduler = None
        epoch_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1), eta_min=base_lr * 1e-3
        )

    mixup_alpha = float(scfg.get("mixup_alpha", 0.0))
    best_acc = 0.0
    best_epoch = -1

    for epoch in range(epochs):
        train_loss, train_acc = run_train_epoch_ce(
            student,
            train_loader,
            criterion,
            optimizer,
            device,
            mixup_alpha=mixup_alpha,
            batch_scheduler=batch_scheduler,
        )
        val_acc = run_val_epoch(student, val_loader, device)
        if epoch_scheduler is not None:
            epoch_scheduler.step()

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f}",
            flush=True,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": student.state_dict(),
                    "classes": classes,
                    "num_classes": num_classes,
                    "val_acc": val_acc,
                    "epoch": epoch,
                    "training": "ce_only",
                    "config_meta": {
                        "img_size": cfg["img_size"],
                        "val_ratio": cfg["val_ratio"],
                        "seed": cfg["seed"],
                        "data_root": str(cfg["data_root"]),
                    },
                },
                baseline_path,
            )
            print(f"  saved best baseline -> {baseline_path}", flush=True)

    print(f"Best baseline val Top-1: {best_acc:.4f} at epoch {best_epoch + 1}")


if __name__ == "__main__":
    main()
