from __future__ import annotations

import torch.nn.functional as F


def distillation_loss(
    student_logits,
    teacher_logits,
    labels,
    temperature: float,
    alpha: float,
    label_smoothing: float = 0.0,
):
    soft_teacher = F.softmax(teacher_logits.detach() / temperature, dim=-1)
    log_soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    kd = F.kl_div(log_soft_student, soft_teacher, reduction="batchmean") * (
        temperature**2
    )
    ce = F.cross_entropy(
        student_logits, labels, label_smoothing=label_smoothing
    )
    return alpha * kd + (1.0 - alpha) * ce
