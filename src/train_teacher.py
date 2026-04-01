from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.optim import AdamW

from src.data import make_loaders
from src.models import (
    build_teacher,
    set_teacher_backbone_requires_grad,
    teacher_optimizer_param_groups,
)
from src.utils import (
    dataloader_augment_kwargs,
    device_synchronize,
    get_device,
    load_config,
    project_root,
    resolve_data_root,
)


@torch.no_grad()
def run_val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(1) == targets).sum().item()
        total += bs
    device_synchronize(device)
    return total_loss / max(total, 1), total_correct / max(total, 1)


def run_train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    mixup_alpha: float,
    batch_scheduler,
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


def main():
    parser = argparse.ArgumentParser(description="Train ResNet50 teacher")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = project_root()
    data_root = resolve_data_root(cfg)
    ckpt_path = root / cfg["checkpoints"]["teacher"]
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    tcfg = cfg["teacher"]
    epochs = int(args.epochs) if args.epochs is not None else int(tcfg["epochs"])
    train_loader, val_loader, classes = make_loaders(
        data_root,
        batch_size_train=tcfg["batch_size_train"],
        batch_size_val=tcfg["batch_size_val"],
        img_size=cfg["img_size"],
        val_ratio=cfg["val_ratio"],
        seed=cfg["seed"],
        num_workers=cfg["num_workers"],
        **dataloader_augment_kwargs(cfg),
    )
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")

    unfreeze_at = int(tcfg.get("unfreeze_epoch", 0) or 0)
    freeze_until = bool(tcfg.get("freeze_backbone_until_unfreeze", False))
    freeze_bb = bool(tcfg.get("freeze_backbone", False))
    mixup_alpha = float(tcfg.get("mixup_alpha", 0.0))

    two_phase = unfreeze_at > 0 and freeze_until
    if two_phase or freeze_bb:
        model = build_teacher(num_classes, freeze_backbone=True).to(device)
    else:
        model = build_teacher(num_classes, freeze_backbone=False).to(device)
    ls = float(tcfg.get("label_smoothing", 0.0))
    criterion = nn.CrossEntropyLoss(label_smoothing=ls)
    wd = float(tcfg["weight_decay"])
    sched_name = str(tcfg.get("scheduler", "cosine")).lower()
    base_lr = float(tcfg["lr"])
    max_lr = float(tcfg.get("max_lr", base_lr))
    phase1_lr = float(tcfg.get("phase1_lr", base_lr))
    bb_ratio = float(tcfg.get("backbone_lr_ratio", 0.1))

    def build_optimizer_head_only(lr: float):
        params = [p for p in model.parameters() if p.requires_grad]
        return AdamW(params, lr=lr, weight_decay=wd)

    def build_optimizer_two_group(head_lr: float):
        return AdamW(
            teacher_optimizer_param_groups(model, head_lr, bb_ratio),
            weight_decay=wd,
        )

    best_acc = 0.0
    best_epoch = -1
    global_epoch = 0

    def train_phase(
        phase_epochs: int,
        optimizer,
        batch_scheduler,
        epoch_scheduler,
        tag: str,
    ):
        nonlocal global_epoch, best_acc, best_epoch
        for _ in range(phase_epochs):
            train_loss, train_acc = run_train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                mixup_alpha,
                batch_scheduler,
            )
            val_loss, val_acc = run_val_epoch(
                model, val_loader, criterion, device
            )
            if epoch_scheduler is not None:
                epoch_scheduler.step()
            global_epoch += 1
            print(
                f"{tag} Epoch {global_epoch}/{epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
                flush=True,
            )
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = global_epoch - 1
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "classes": classes,
                        "num_classes": num_classes,
                        "val_acc": val_acc,
                        "epoch": global_epoch - 1,
                        "config_meta": {
                            "img_size": cfg["img_size"],
                            "val_ratio": cfg["val_ratio"],
                            "seed": cfg["seed"],
                            "data_root": str(cfg["data_root"]),
                        },
                    },
                    ckpt_path,
                )
                print(f"  saved best checkpoint -> {ckpt_path}", flush=True)

    if unfreeze_at > 0 and freeze_until:
        p1 = min(unfreeze_at, epochs)
        p2 = max(epochs - p1, 0)
        print(
            f"Teacher: two-phase — phase1 frozen backbone {p1} epochs, "
            f"then fine-tune full model {p2} epochs.",
            flush=True,
        )
        opt1 = build_optimizer_head_only(phase1_lr)
        if sched_name == "onecycle":
            bs1 = torch.optim.lr_scheduler.OneCycleLR(
                opt1,
                max_lr=phase1_lr,
                epochs=p1,
                steps_per_epoch=len(train_loader),
                pct_start=0.1,
                div_factor=25.0,
                final_div_factor=1e4,
            )
            es1 = None
        else:
            bs1 = None
            es1 = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt1, T_max=max(p1, 1), eta_min=phase1_lr * 1e-3
            )
        train_phase(p1, opt1, bs1, es1, tag="P1")

        if p2 > 0:
            set_teacher_backbone_requires_grad(model, True)
            opt2 = build_optimizer_two_group(base_lr)
            if sched_name == "onecycle":
                bs2 = torch.optim.lr_scheduler.OneCycleLR(
                    opt2,
                    max_lr=base_lr,
                    epochs=p2,
                    steps_per_epoch=len(train_loader),
                    pct_start=0.1,
                    div_factor=25.0,
                    final_div_factor=1e4,
                )
                es2 = None
            else:
                bs2 = None
                es2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt2, T_max=max(p2, 1), eta_min=base_lr * 1e-3
                )
            train_phase(p2, opt2, bs2, es2, tag="P2")
    else:
        if freeze_bb:
            print("Teacher: classifier head only (backbone frozen).", flush=True)
            opt = build_optimizer_head_only(phase1_lr)
            lr_for_sched = phase1_lr
        else:
            print("Teacher: full model fine-tune (param groups).", flush=True)
            opt = build_optimizer_two_group(base_lr)
            lr_for_sched = base_lr
        if sched_name == "onecycle":
            bs = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=max_lr,
                epochs=epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.1,
                div_factor=25.0,
                final_div_factor=1e4,
            )
            es = None
        else:
            bs = None
            es = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(epochs, 1), eta_min=lr_for_sched * 1e-3
            )
        train_phase(epochs, opt, bs, es, tag="")

    print(f"Best val Top-1: {best_acc:.4f} at epoch {best_epoch + 1}")


if __name__ == "__main__":
    main()
