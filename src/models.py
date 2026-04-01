from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet50_Weights,
    mobilenet_v3_small,
    resnet50,
)


def build_teacher(num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    set_teacher_backbone_requires_grad(m, not freeze_backbone)
    return m


def set_teacher_backbone_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for name, p in model.named_parameters():
        if name.startswith("fc."):
            p.requires_grad = True
        else:
            p.requires_grad = requires_grad


def teacher_optimizer_param_groups(
    model: nn.Module, head_lr: float, backbone_lr_ratio: float
):
    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("fc."):
            head_params.append(p)
        else:
            backbone_params.append(p)
    return [
        {"params": backbone_params, "lr": head_lr * backbone_lr_ratio},
        {"params": head_params, "lr": head_lr},
    ]


def build_student(num_classes: int) -> nn.Module:
    m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(in_features, num_classes)
    return m
