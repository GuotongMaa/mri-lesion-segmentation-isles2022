from __future__ import annotations

import torch


def dice_score(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    pred = (torch.sigmoid(logits) >= threshold).float()
    target = target.float()
    inter = (pred * target).sum(dim=(1, 2, 3, 4))
    denom = pred.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
    score = (2 * inter + eps) / (denom + eps)
    return score


def iou_score(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    pred = (torch.sigmoid(logits) >= threshold).float()
    target = target.float()
    inter = (pred * target).sum(dim=(1, 2, 3, 4))
    union = pred.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4)) - inter
    score = (inter + eps) / (union + eps)
    return score


def lesion_recall(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    pred = (torch.sigmoid(logits) >= threshold).float()
    target = target.float()
    tp = (pred * target).sum(dim=(1, 2, 3, 4))
    fn = ((1 - pred) * target).sum(dim=(1, 2, 3, 4))
    recall = (tp + eps) / (tp + fn + eps)
    return recall
