from __future__ import annotations

import argparse

import torch

from src.data import make_loaders
from src.models import build_student, build_teacher
from src.utils import (
    dataloader_augment_kwargs,
    device_synchronize,
    get_device,
    load_config,
    project_root,
    resolve_data_root,
)


@torch.no_grad()
def evaluate_model(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        correct += (logits.argmax(1) == targets).sum().item()
        total += images.size(0)
    device_synchronize(device)
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Top-1 accuracy: teacher vs student")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (default: configs/default.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = project_root()
    device = get_device()
    print(f"Device: {device}")

    data_root = resolve_data_root(cfg)
    _, val_loader, classes_cfg = make_loaders(
        data_root,
        batch_size_train=cfg["teacher"]["batch_size_train"],
        batch_size_val=cfg["teacher"]["batch_size_val"],
        img_size=cfg["img_size"],
        val_ratio=cfg["val_ratio"],
        seed=cfg["seed"],
        num_workers=cfg["num_workers"],
        **dataloader_augment_kwargs(cfg),
    )
    num_classes = len(classes_cfg)

    teacher_path = root / cfg["checkpoints"]["teacher"]
    student_path = root / cfg["checkpoints"]["student"]

    def load_ckpt(path):
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=device)

    if teacher_path.is_file():
        ckpt_t = load_ckpt(teacher_path)
        nc = ckpt_t.get("num_classes", num_classes)
        teacher = build_teacher(nc).to(device)
        teacher.load_state_dict(ckpt_t["model_state_dict"])
        acc_t = evaluate_model(teacher, val_loader, device)
        print(f"Teacher (ResNet50) Top-1: {acc_t:.4f}")
    else:
        print(f"Teacher checkpoint missing: {teacher_path}")

    if student_path.is_file():
        ckpt_s = load_ckpt(student_path)
        nc = ckpt_s.get("num_classes", num_classes)
        student = build_student(nc).to(device)
        student.load_state_dict(ckpt_s["model_state_dict"])
        acc_s = evaluate_model(student, val_loader, device)
        print(f"Student (MobileNetV3-Small) Top-1: {acc_s:.4f}")
    else:
        print(f"Student checkpoint missing: {student_path}")


if __name__ == "__main__":
    main()
