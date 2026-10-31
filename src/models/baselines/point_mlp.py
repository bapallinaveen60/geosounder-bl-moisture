"""Point Radiance Multi-Layer Perceptron (MLP) Baseline.

A benchmark retrieval model mapping single-pixel sounder radiances directly to
vertical moisture profiles without spatial neighbor context or temporal history.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointRadianceMLP(nn.Module):
    """Point-wise Multi-Layer Perceptron for 1D radiance-to-profile inversion."""

    def __init__(
        self,
        in_features: int = 19,  # 18 IR bands + 1 Visible band
        out_levels: int = 5,
        hidden_dims: tuple[int, ...] = (128, 256, 128, 64),
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        cur_in = in_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(cur_in, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(0.1))
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
            cur_in = h_dim

        layers.append(nn.Linear(cur_in, out_levels))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

        x: Single pixel radiance vector (B, In_features) or spatial tensor (B, C, H, W).

        Returns:
            Specific humidity profile (B, L=5) or (B, L=5, H, W).
        """
        if x.ndim == 4:
            # Spatial tensor passed: process pixel-wise
            b, c, h, w = x.shape
            x_perm = x.permute(0, 2, 3, 1).reshape(-1, c)  # (B*H*W, C)
            out = F.relu(self.network(x_perm))  # (B*H*W, L)
            out = out.view(b, h, w, -1).permute(0, 3, 1, 2)  # (B, L, H, W)
            return out

        return F.relu(self.network(x))
