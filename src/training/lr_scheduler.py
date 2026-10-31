"""Warmup and Cosine Annealing Learning Rate Scheduler."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """Linear warmup followed by cosine annealing decay."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int = 5,
        total_epochs: int = 120,
        min_lr: float = 1.0e-6,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            pass

        if self.last_epoch < self.warmup_epochs:
            # Linear warmup phase
            alpha = (self.last_epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]

        # Cosine annealing phase
        progress = (self.last_epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs
        )
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.min_lr + (base_lr - self.min_lr) * cosine_factor
            for base_lr in self.base_lrs
        ]
