"""Vertical Monotonicity and Physical Lapse Rate Constraint.

Penalizes unphysical super-adiabatic water vapor inversions unless supported
by documented thermodynamic capping inversion signatures.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VerticalMonotonicityLoss(nn.Module):
    """Penalizes non-physical vertical moisture lapse rate inversions."""

    def __init__(self, delta_q: float = 0.2) -> None:
        """Args:

        delta_q: Allowable tolerance threshold (g/kg) before penalizing inversions.
        """
        super().__init__()
        self.delta_q = delta_q

    def forward(self, q_pred: torch.Tensor) -> torch.Tensor:
        """Compute monotonicity loss.

        Args:
            q_pred: Specific humidity tensor of shape (B, L, H, W) in g/kg,
                   ordered from surface (p=1000 hPa at index 0) to top (p=700 hPa at index L-1).

        Returns:
            Scalar penalty term.
        """
        # q_pred[:, k+1, ...] should ideally be <= q_pred[:, k, ...] + delta_q
        # Penalize when: q_{k+1} - q_k - delta_q > 0
        diff = q_pred[:, 1:, :, :] - q_pred[:, :-1, :, :] - self.delta_q
        penalty = F.relu(diff)
        return torch.mean(penalty)
