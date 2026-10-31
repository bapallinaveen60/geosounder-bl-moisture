"""Composite Physics-Informed Loss Function for ThermoBL-Net.

Formulation:
    L_total = lambda_1 * L_profile + lambda_2 * L_column + lambda_3 * L_monotonic + lambda_4 * L_conv
Combines:
  1. Huber profile reconstruction loss on collocated radiosondes
  2. Invariant GNSS hydrostatic column mass conservation
  3. Vertical moisture lapse rate monotonicity constraint
  4. Horizontal Moisture Flux Convergence (MFC) finite-difference gradient consistency
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.physics.mass_conservation import ColumnMassConservationLoss
from src.physics.moisture_flux import MoistureFluxConvergenceOperator
from src.physics.monotonicity import VerticalMonotonicityLoss


class CompositePhysicsLoss(nn.Module):
    """Composite Physics-Informed Loss Function."""

    def __init__(
        self,
        lambda_profile: float = 1.0,
        lambda_column: float = 1.8,
        lambda_monotonic: float = 0.5,
        lambda_conv: float = 0.8,
        huber_delta: float = 0.5,
        monotonic_delta_q: float = 0.2,
        pressure_levels_hpa: List[float] = [1000.0, 950.0, 925.0, 850.0, 700.0],
        rho_w: float = 1000.0,
        g: float = 9.80665,
        dx_meters: float = 4000.0,
        dy_meters: float = 4000.0,
    ) -> None:
        super().__init__()
        self.lambda_profile = lambda_profile
        self.lambda_column = lambda_column
        self.lambda_monotonic = lambda_monotonic
        self.lambda_conv = lambda_conv

        # 1. Profile reconstruction criterion: Huber Loss (smooth L1)
        self.huber_delta = huber_delta

        # 2. GNSS Mass Conservation loss
        self.column_loss_fn = ColumnMassConservationLoss(
            pressure_levels_hpa=pressure_levels_hpa,
            rho_w=rho_w,
            g=g,
        )

        # 3. Monotonicity constraint
        self.monotonic_loss_fn = VerticalMonotonicityLoss(delta_q=monotonic_delta_q)

        # 4. Moisture flux convergence operator
        self.mfc_operator = MoistureFluxConvergenceOperator(
            dx_meters=dx_meters,
            dy_meters=dy_meters,
        )

    def forward(
        self,
        q_pred: torch.Tensor,
        q_true: torch.Tensor,
        sonde_mask: Optional[torch.Tensor],
        gnss_pwv_true: torch.Tensor,
        upper_pwv_residual: torch.Tensor,
        gnss_mask: Optional[torch.Tensor],
        winds_uv: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute composite physics-informed loss and its individual diagnostic terms.

        Args:
            q_pred: Predicted specific humidity (B, L=5, H, W) in g/kg.
            q_true: True specific humidity (B, L=5, H, W) in g/kg.
            sonde_mask: Mask of collocated in-situ radiosonde pixels (B, L, H, W).
            gnss_pwv_true: Ground truth GNSS PWV (B, H, W) in mm.
            upper_pwv_residual: Upper tropospheric PWV above 700 hPa (B, H, W) in mm.
            gnss_mask: Mask of active GNSS receiver stations (B, H, W).
            winds_uv: Horizontal surface/boundary wind vectors (B, 2, H, W) in m/s.

        Returns:
            Dictionary containing:
                - 'total_loss': Weighted sum of all loss components
                - 'loss_profile': Radiosonde profile Huber loss
                - 'loss_column': GNSS hydrostatic column mass loss
                - 'loss_monotonic': Monotonicity penalty
                - 'loss_conv': Moisture flux convergence divergence loss
        """
        # --- 1. Radiosonde Profile Huber Loss ---
        if sonde_mask is not None and sonde_mask.sum() > 0:
            diff_profile = torch.abs(q_pred[sonde_mask] - q_true[sonde_mask])
            loss_profile = F.huber_loss(
                q_pred[sonde_mask],
                q_true[sonde_mask],
                delta=self.huber_delta,
                reduction="mean",
            )
        else:
            loss_profile = F.huber_loss(q_pred, q_true, delta=self.huber_delta, reduction="mean")

        # --- 2. Invariant GNSS Column Mass Conservation Loss ---
        if self.lambda_column > 0:
            loss_column = self.column_loss_fn(
                q_pred=q_pred,
                gnss_pwv_true=gnss_pwv_true,
                upper_pwv_residual=upper_pwv_residual,
                gnss_mask=gnss_mask,
            )
        else:
            loss_column = torch.tensor(0.0, device=q_pred.device, dtype=q_pred.dtype)

        # --- 3. Vertical Monotonicity Constraint ---
        if self.lambda_monotonic > 0:
            loss_monotonic = self.monotonic_loss_fn(q_pred)
        else:
            loss_monotonic = torch.tensor(0.0, device=q_pred.device, dtype=q_pred.dtype)

        # --- 4. Horizontal Moisture Flux Convergence (MFC) Divergence Loss ---
        if self.lambda_conv > 0:
            # 925 hPa level corresponds to index 2 in [1000, 950, 925, 850, 700]
            q_925_pred = q_pred[:, 2:3, :, :]
            q_925_true = q_true[:, 2:3, :, :]
            u_wind = winds_uv[:, 0:1, :, :]
            v_wind = winds_uv[:, 1:2, :, :]

            mfc_pred = self.mfc_operator(q_925_pred, u_wind, v_wind)
            mfc_true = self.mfc_operator(q_925_true, u_wind, v_wind)

            loss_conv = F.l1_loss(mfc_pred, mfc_true)
        else:
            loss_conv = torch.tensor(0.0, device=q_pred.device, dtype=q_pred.dtype)

        # --- Composite Objective ---
        total_loss = (
            self.lambda_profile * loss_profile
            + self.lambda_column * loss_column
            + self.lambda_monotonic * loss_monotonic
            + self.lambda_conv * loss_conv
        )

        return {
            "total_loss": total_loss,
            "loss_profile": loss_profile,
            "loss_column": loss_column,
            "loss_monotonic": loss_monotonic,
            "loss_conv": loss_conv,
        }
