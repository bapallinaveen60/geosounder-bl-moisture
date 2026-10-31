"""Horizontal Moisture Flux Convergence (MFC) Operator via 2D Finite Differences.

Computes the horizontal divergence of boundary-layer moisture flux:
    MFC = - nabla_h . (v_h * q) = - ( d(u * q)/dx + d(v * q)/dy )
using differentiable 2D Sobel convolutional filters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoistureFluxConvergenceOperator(nn.Module):
    """Computes Moisture Flux Convergence (MFC) using differentiable spatial finite-difference kernels."""

    def __init__(self, dx_meters: float = 4000.0, dy_meters: float = 4000.0) -> None:
        """Args:

        dx_meters: Grid spacing in horizontal x-direction (meters).
        dy_meters: Grid spacing in horizontal y-direction (meters).
        """
        super().__init__()
        self.dx = dx_meters
        self.dy = dy_meters

        # Sobel horizontal gradient filter (d/dx)
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3) / (8.0 * self.dx)

        # Sobel vertical gradient filter (d/dy)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3) / (8.0 * self.dy)

        self.register_buffer("kernel_x", sobel_x)
        self.register_buffer("kernel_y", sobel_y)

    def forward(self, q_field: torch.Tensor, u_wind: torch.Tensor, v_wind: torch.Tensor) -> torch.Tensor:
        """Compute instantaneous horizontal Moisture Flux Convergence (MFC).

        Args:
            q_field: 2D specific humidity slice (e.g. at 925 hPa) of shape (B, 1, H, W) in g/kg.
            u_wind: Horizontal zonal wind component (B, 1, H, W) in m/s.
            v_wind: Horizontal meridional wind component (B, 1, H, W) in m/s.

        Returns:
            MFC tensor of shape (B, 1, H, W) in g/(kg * s).
            Positive values denote convergence; negative values denote divergence.
        """
        # Moisture flux vectors: F_x = u * q, F_y = v * q
        flux_x = u_wind * q_field
        flux_y = v_wind * q_field

        # Replicate padding to preserve spatial dimensions
        flux_x_pad = F.pad(flux_x, (1, 1, 1, 1), mode="replicate")
        flux_y_pad = F.pad(flux_y, (1, 1, 1, 1), mode="replicate")

        # Convolve with Sobel operators
        d_flux_x_dx = F.conv2d(flux_x_pad, self.kernel_x)
        d_flux_y_dy = F.conv2d(flux_y_pad, self.kernel_y)

        # MFC = - (d(u*q)/dx + d(v*q)/dy)
        mfc = -(d_flux_x_dx + d_flux_y_dy)
        return mfc
