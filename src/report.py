from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.data import make_loaders
from src.eval_metrics import (
    classification_report_dict,
    collect_misclassified_paths,
    collect_predictions,
    compute_confusion_matrix,
    forward_flop_stats,
    inference_benchmark,
    overall_accuracy,
    per_class_metrics,
)
from src.models import build_student, build_teacher
from src.utils import (
    dataloader_augment_kwargs,
    get_device,
    load_config,
    project_root,
    resolve_data_root,
)


def _load_ckpt(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _image_src_for_html(image_abs: Path, out_dir: Path) -> str:
    """Relative path from HTML in out_dir, or file:// URI if needed (e.g. different drive)."""
    image_abs = image_abs.resolve()
    out_r = out_dir.resolve()
    try:
        rel = Path(os.path.relpath(image_abs, start=out_r))
        return rel.as_posix()
    except ValueError:
        return image_abs.as_uri()


def write_misclassified_gallery(
    out_dir: Path,
    model_safe_name: str,
    mistakes: list[dict],
) -> Path | None:
    if not mistakes:
        return None
    html_path = out_dir / f"misclassified_gallery_{model_safe_name}.html"
    title = html.escape(model_safe_name.replace("_", " "))
    chunks: list[str] = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>Misclassified — {title}</title>",
        "<style>",
        "body{font-family:system-ui,Segoe UI,sans-serif;margin:1rem;background:#fafafa;}",
        "h1{font-size:1.25rem;}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:1rem;}",
        ".card{border:1px solid #ccc;border-radius:8px;padding:.75rem;background:#fff;}",
        ".card img{width:100%;height:180px;object-fit:cover;border-radius:4px;}",
        ".cap{font-size:.8rem;margin-top:.5rem;line-height:1.35;color:#333;}",
        ".fname{font-size:.72rem;color:#666;word-break:break-all;}",
        "</style></head><body>",
        f"<h1>Misclassified — {title}</h1>",
        f"<p>{len(mistakes)} validation image(s). <strong>True</strong> → <strong>predicted</strong>.</p>",
        "<div class='grid'>",
    ]
    for m in mistakes:
        abs_p = Path(m["image_path"]).resolve()
        src = html.escape(_image_src_for_html(abs_p, out_dir))
        tcls = html.escape(str(m["true_class"])[:120])
        pcls = html.escape(str(m["predicted_class"])[:120])
        fname = html.escape(abs_p.name)
        chunks.append("<div class='card'>")
        chunks.append(f"<img src='{src}' alt='' loading='lazy'>")
        chunks.append("<div class='cap'>")
        chunks.append(f"<strong>True:</strong> {tcls}<br>")
        chunks.append(f"<strong>Pred:</strong> {pcls}")
        chunks.append(f"</div><div class='fname'>{fname}</div></div>")
    chunks.append("</div></body></html>")
    html_path.write_text("".join(chunks), encoding="utf-8")
    return html_path


def _bench_batch(val_loader, bench_bs: int, device) -> torch.Tensor:
    images, _ = next(iter(val_loader))
    if images.size(0) < bench_bs:
        pad = bench_bs - images.size(0)
        images = torch.cat([images, images[:pad]], dim=0)
    return images


def main():
    parser = argparse.ArgumentParser(
        description="Full report: accuracy, per-class metrics, confusion CSV, timing"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory for summary.json and CSV exports (under project root)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = project_root()
    out_dir = (root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}\n", flush=True)

    data_root = resolve_data_root(cfg)
    _, val_loader, class_names = make_loaders(
        data_root,
        batch_size_train=cfg["teacher"]["batch_size_train"],
        batch_size_val=cfg["teacher"]["batch_size_val"],
        img_size=cfg["img_size"],
        val_ratio=cfg["val_ratio"],
        seed=cfg["seed"],
        num_workers=cfg["num_workers"],
        **dataloader_augment_kwargs(cfg),
    )
    num_classes = len(class_names)
    bcfg = cfg.get("benchmark", {})
    warmup = int(bcfg.get("warmup_steps", 20))
    timed = int(bcfg.get("timed_steps", 100))
    bench_bs = int(bcfg.get("batch_size", 1))
    bench_images = _bench_batch(val_loader, bench_bs, device)

    ckpt_cfg = cfg["checkpoints"]
    jobs: list[tuple[str, str, str]] = [
        ("Teacher", "teacher", ckpt_cfg["teacher"]),
        ("Student_KD", "student", ckpt_cfg["student"]),
        (
            "Student_KD_HighAlpha",
            "student",
            ckpt_cfg.get("student_high_alpha", ""),
        ),
        ("Student_CE", "student", ckpt_cfg.get("student_baseline", "")),
    ]

    summary: dict = {
        "class_names": class_names,
        "models": {},
    }
    kd_acc: float | None = None
    kd_high_alpha_acc: float | None = None
    ce_acc: float | None = None

    for display_name, kind, rel_path in jobs:
        path = root / rel_path if rel_path else None
        if not path or not path.is_file():
            print(f"[skip] {display_name}: checkpoint not found ({rel_path})")
            summary["models"][display_name] = {"status": "missing"}
            continue

        ckpt = _load_ckpt(path, device)
        nc = ckpt.get("num_classes", num_classes)
        if kind == "teacher":
            model = build_teacher(nc).to(device)
        else:
            model = build_student(nc).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        flop_stats = forward_flop_stats(model, int(cfg["img_size"]))

        y_true, y_pred = collect_predictions(model, val_loader, device)
        mistakes = collect_misclassified_paths(
            model, val_loader, device, class_names
        )
        acc = overall_accuracy(y_true, y_pred)
        cm = compute_confusion_matrix(y_true, y_pred, nc)
        pcm = per_class_metrics(y_true, y_pred, class_names)
        rep = classification_report_dict(y_true, y_pred, class_names)
        bench = inference_benchmark(
            model, bench_images, device, warmup, timed
        )

        if display_name == "Student_KD":
            kd_acc = acc
        if display_name == "Student_KD_HighAlpha":
            kd_high_alpha_acc = acc
        if display_name == "Student_CE":
            ce_acc = acc

        model_entry: dict = {
            "status": "ok",
            "checkpoint": str(rel_path),
            "best_epoch": ckpt.get("epoch"),
            "overall_accuracy": acc,
            "misclassified_count": len(mistakes),
            "per_class": pcm,
            "classification_report": rep,
            "inference": bench,
            "flops": flop_stats,
        }
        if kind == "student" and isinstance(ckpt.get("distillation"), dict):
            model_entry["distillation"] = ckpt["distillation"]
        summary["models"][display_name] = model_entry

        print(f"=== {display_name} ({rel_path}) ===")
        print(f"Overall Top-1: {acc:.4f}")
        if flop_stats.get("status") == "ok":
            print(
                f"FLOPs (forward, bs=1, {cfg['img_size']}x{cfg['img_size']}): "
                f"{flop_stats['forward_gflops']:.3f} G  |  "
                f"params: {flop_stats['num_params']:,}"
            )
        else:
            print(
                f"FLOPs: n/a ({flop_stats.get('reason', flop_stats.get('status', ''))})"
            )
        print(
            f"Inference: {bench['ms_per_image']:.3f} ms/img, "
            f"{bench['images_per_sec']:.1f} img/s "
            f"(batch={int(bench['batch_size'])})"
        )
        print("Per-class (support / class-acc):")
        for row in pcm:
            print(
                f"  [{row['class_index']}] {row['class_name'][:50]:<50} "
                f"n={row['support']:4d}  acc={row['class_accuracy']:.4f}"
            )
        print()

        safe = display_name.replace(" ", "_")
        cm_path = out_dir / f"confusion_{safe}.csv"
        with cm_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([""] + class_names)
            for i, name in enumerate(class_names):
                w.writerow([name] + [int(x) for x in cm[i].tolist()])
        print(f"Wrote {cm_path}")

        mc_path = out_dir / f"misclassified_{safe}.csv"
        with mc_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "image_path",
                    "true_index",
                    "predicted_index",
                    "true_class",
                    "predicted_class",
                ]
            )
            for row in mistakes:
                w.writerow(
                    [
                        row["image_path"],
                        row["true_index"],
                        row["predicted_index"],
                        row["true_class"],
                        row["predicted_class"],
                    ]
                )
        print(
            f"Wrote {mc_path} ({len(mistakes)} misclassified)",
            flush=True,
        )

        gal = write_misclassified_gallery(out_dir, safe, mistakes)
        if gal is not None:
            print(f"Wrote {gal}", flush=True)

    val_size = len(val_loader.dataset)
    summary["validation"] = {
        "num_samples": val_size,
        "val_ratio": float(cfg["val_ratio"]),
        "split_seed": int(cfg["seed"]),
    }

    per_class_merged = out_dir / "per_class_long.csv"
    with per_class_merged.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["model", "class_index", "class_name", "support", "class_accuracy"]
        )
        for name, data in summary["models"].items():
            if data.get("status") != "ok":
                continue
            for row in data["per_class"]:
                w.writerow(
                    [
                        name,
                        row["class_index"],
                        row["class_name"],
                        row["support"],
                        f"{row['class_accuracy']:.6f}",
                    ]
                )
    print(f"Wrote {per_class_merged}")

    if kd_acc is not None and ce_acc is not None:
        delta = kd_acc - ce_acc
        summary["kd_vs_ce"] = {
            "student_kd_accuracy": kd_acc,
            "student_ce_accuracy": ce_acc,
            "absolute_gain_kd_minus_ce": delta,
        }
        print(f"KD vs CE-only (val Top-1): delta = {delta:+.4f} (KD - CE)")

    if kd_acc is not None and kd_high_alpha_acc is not None:
        delta_h = kd_high_alpha_acc - kd_acc
        summary["kd_default_vs_high_alpha"] = {
            "student_kd_accuracy": kd_acc,
            "student_kd_high_alpha_accuracy": kd_high_alpha_acc,
            "absolute_gain_high_alpha_minus_default": delta_h,
        }
        print(
            "KD high-alpha vs default KD (val Top-1): "
            f"delta = {delta_h:+.4f} (high_alpha - default)",
            flush=True,
        )

    json_path = out_dir / "summary.json"

    def json_sanitize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): json_sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [json_sanitize(v) for v in obj]
        return obj

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_sanitize(summary), f, indent=2)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
