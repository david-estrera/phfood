from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        path = project_root() / "configs" / "default.yaml"
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_data_root(cfg: dict[str, Any]) -> Path:
    root = project_root()
    return (root / cfg["data_root"]).resolve()


def get_device() -> torch.device:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def dataloader_augment_kwargs(cfg: dict) -> dict:
    return {
        "randaugment": bool(cfg.get("randaugment", True)),
        "randaugment_num_ops": int(cfg.get("randaugment_num_ops", 2)),
        "randaugment_magnitude": int(cfg.get("randaugment_magnitude", 9)),
        "random_erasing_p": float(cfg.get("random_erasing_p", 0.0)),
    }


def device_synchronize(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
