from __future__ import annotations

import argparse
import time

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


def _load_ckpt(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def benchmark_model(
    model,
    images: torch.Tensor,
    device: torch.device,
    warmup_steps: int,
    timed_steps: int,
    name: str,
):
    model.eval()
    images = images.to(device, non_blocking=True)
    bs = images.size(0)

    for _ in range(warmup_steps):
        _ = model(images)
    device_synchronize(device)

    device_synchronize(device)
    t0 = time.perf_counter()
    for _ in range(timed_steps):
        _ = model(images)
    device_synchronize(device)
    t1 = time.perf_counter()

    total_time = t1 - t0
    total_images = timed_steps * bs
    ms_per_image = (total_time / total_images) * 1000.0
    images_per_sec = total_images / total_time
    print(
        f"{name}: batch_size={bs} warmup={warmup_steps} timed_steps={timed_steps} "
        f"-> {ms_per_image:.3f} ms/image, {images_per_sec:.1f} img/s (device)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Inference timing: teacher vs student on XPU/CUDA/CPU"
    )
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

    bcfg = cfg.get("benchmark", {})
    warmup = int(bcfg.get("warmup_steps", 20))
    timed = int(bcfg.get("timed_steps", 100))
    bench_bs = int(bcfg.get("batch_size", 1))

    data_root = resolve_data_root(cfg)
    _, val_loader, classes_cfg = make_loaders(
        data_root,
        batch_size_train=bench_bs,
        batch_size_val=bench_bs,
        img_size=cfg["img_size"],
        val_ratio=cfg["val_ratio"],
        seed=cfg["seed"],
        num_workers=cfg["num_workers"],
        **dataloader_augment_kwargs(cfg),
    )

    images, _ = next(iter(val_loader))
    if images.size(0) < bench_bs:
        pad = bench_bs - images.size(0)
        images = torch.cat([images, images[:pad]], dim=0)

    teacher_path = root / cfg["checkpoints"]["teacher"]
    student_path = root / cfg["checkpoints"]["student"]

    if teacher_path.is_file():
        ckpt = _load_ckpt(teacher_path, device)
        nc = ckpt.get("num_classes", len(classes_cfg))
        teacher = build_teacher(nc).to(device)
        teacher.load_state_dict(ckpt["model_state_dict"])
        benchmark_model(
            teacher, images, device, warmup, timed, "Teacher (ResNet50)"
        )
    else:
        print(f"Teacher checkpoint missing: {teacher_path}")

    if student_path.is_file():
        ckpt = _load_ckpt(student_path, device)
        nc = ckpt.get("num_classes", len(classes_cfg))
        student = build_student(nc).to(device)
        student.load_state_dict(ckpt["model_state_dict"])
        benchmark_model(
            student, images, device, warmup, timed, "Student (MobileNetV3-Small)"
        )
    else:
        print(f"Student checkpoint missing: {student_path}")


if __name__ == "__main__":
    main()
