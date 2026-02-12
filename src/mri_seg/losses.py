from __future__ import annotations

import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    def __init__(self, pos_weight: float = 10.0, bce_weight: float = 0.5, eps: float = 1e-6) -> None:
        super().__init__()
        self.register_buffer("_pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        self.bce_weight = bce_weight
        self.eps = eps

    def _dice_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        target = target.float()
        inter = (prob * target).sum(dim=(1, 2, 3, 4))
        denom = prob.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
        dice = (2 * inter + self.eps) / (denom + self.eps)
        return 1 - dice.mean()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pos_weight = self._pos_weight.to(logits.device)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target.float(), pos_weight=pos_weight)
        dice = self._dice_loss(logits, target)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, eps: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        target = target.float()

        tp = (prob * target).sum(dim=(1, 2, 3, 4))
        fp = (prob * (1 - target)).sum(dim=(1, 2, 3, 4))
        fn = ((1 - prob) * target).sum(dim=(1, 2, 3, 4))

        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return 1 - tversky.mean()
