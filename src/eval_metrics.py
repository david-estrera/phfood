from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Subset

from src.utils import device_synchronize


@torch.no_grad()
def collect_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        pred = logits.argmax(dim=1)
        y_true.extend(targets.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())
    device_synchronize(device)
    return np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)


@torch.no_grad()
def collect_misclassified_paths(
    model,
    val_loader,
    device,
    class_names: list[str],
) -> list[dict[str, Any]]:
    """
    List validation images where argmax(pred) != label.
    Requires val_loader.dataset to be Subset(ImageFolder) as built by make_loaders.
    """
    ds = val_loader.dataset
    if not isinstance(ds, Subset):
        raise TypeError(
            "collect_misclassified_paths expects Subset(ImageFolder); "
            "use the val_loader from make_loaders."
        )
    base = ds.dataset
    idx_map: list[int] = list(ds.indices)
    if not hasattr(base, "samples"):
        raise TypeError("Underlying dataset must be ImageFolder (has .samples).")

    model.eval()
    rows: list[dict[str, Any]] = []
    offset = 0
    for images, targets in val_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        pred = logits.argmax(dim=1)
        bs = images.size(0)
        for i in range(bs):
            t = int(targets[i].item())
            p = int(pred[i].item())
            if t == p:
                continue
            subset_pos = offset + i
            sample_idx = idx_map[subset_pos]
            path, _ = base.samples[sample_idx]
            rows.append(
                {
                    "image_path": str(path),
                    "true_index": t,
                    "predicted_index": p,
                    "true_class": class_names[t],
                    "predicted_class": class_names[p],
                }
            )
        offset += bs
    device_synchronize(device)
    return rows


def overall_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> list[dict[str, Any]]:
    cm = compute_confusion_matrix(y_true, y_pred, len(class_names))
    rows: list[dict[str, Any]] = []
    for i, name in enumerate(class_names):
        row_sum = cm[i].sum()
        support = int(row_sum)
        correct = int(cm[i, i])
        class_acc = correct / row_sum if row_sum > 0 else 0.0
        rows.append(
            {
                "class_index": i,
                "class_name": name,
                "support": support,
                "correct": correct,
                "class_accuracy": float(class_acc),
            }
        )
    return rows


def classification_report_dict(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> dict[str, Any]:
    return classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )


@torch.no_grad()
def inference_benchmark(
    model,
    images: torch.Tensor,
    device: torch.device,
    warmup_steps: int,
    timed_steps: int,
) -> dict[str, float]:
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
    return {
        "batch_size": float(bs),
        "ms_per_image": float(ms_per_image),
        "images_per_sec": float(images_per_sec),
        "warmup_steps": float(warmup_steps),
        "timed_steps": float(timed_steps),
    }


def _scalar_to_int(x: Any) -> int:
    if hasattr(x, "item"):
        return int(x.item())
    return int(x)


def forward_flop_stats(model: torch.nn.Module, img_size: int) -> dict[str, Any]:
    """
    Approximate forward-pass FLOPs and parameter count (batch size 1, square RGB input).
    Runs on CPU via thop for consistent results across CUDA/XPU/CPU training devices.
    """
    try:
        from thop import profile
    except ImportError:
        return {
            "status": "skipped",
            "reason": "thop not installed (pip install thop)",
        }

    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    h = w = int(img_size)
    try:
        model.cpu()
        dummy = torch.randn(1, 3, h, w, dtype=torch.float32)
        with torch.no_grad():
            flops, params = profile(model, inputs=(dummy,), verbose=False)
        fwd = _scalar_to_int(flops)
        prm = _scalar_to_int(params)
    except Exception as e:  # noqa: BLE001 — surface any hook/backend failure
        model.to(dev)
        if was_training:
            model.train()
        else:
            model.eval()
        return {"status": "error", "reason": repr(e)}

    model.to(dev)
    if was_training:
        model.train()
    else:
        model.eval()

    return {
        "status": "ok",
        "batch_size": 1,
        "input_size": h,
        "forward_flops": fwd,
        "forward_gflops": fwd / 1e9,
        "num_params": prm,
    }
