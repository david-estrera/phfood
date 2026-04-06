from __future__ import annotations

import argparse

import torch

from src.data import make_loaders
from src.eval_metrics import inference_benchmark
from src.models import build_student, build_teacher
from src.utils import (
    dataloader_augment_kwargs,
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


def _print_bench(name: str, stats: dict[str, float]) -> None:
    print(
        f"{name}: batch_size={int(stats['batch_size'])} "
        f"warmup={int(stats['warmup_steps'])} "
        f"timed_steps={int(stats['timed_steps'])} "
        f"-> {stats['ms_per_image']:.3f} ms/image, "
        f"{stats['images_per_sec']:.1f} img/s (device)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Inference timing: teacher vs student(s) on XPU/CUDA/CPU"
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

    ck = cfg["checkpoints"]
    paths = [
        ("Teacher (ResNet50)", "teacher", ck["teacher"]),
        ("Student (MobileNetV3-Small, KD)", "student", ck["student"]),
    ]
    ha = ck.get("student_high_alpha")
    if ha:
        paths.append(
            ("Student (MobileNetV3-Small, KD high α)", "student", ha)
        )
    bl = ck.get("student_baseline")
    if bl:
        paths.append(
            ("Student (MobileNetV3-Small, CE-only)", "student", bl)
        )

    for label, kind, rel in paths:
        path = root / rel
        if not path.is_file():
            print(f"{label}: checkpoint missing: {path}")
            continue
        ckpt = _load_ckpt(path, device)
        nc = ckpt.get("num_classes", len(classes_cfg))
        if kind == "teacher":
            model = build_teacher(nc).to(device)
        else:
            model = build_student(nc).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        stats = inference_benchmark(model, images, device, warmup, timed)
        _print_bench(label, stats)


if __name__ == "__main__":
    main()
