"""GNSS Hydrostatic Column Mass Invariant and Precipitable Water Vapor Integral Operator.

Calculates the physical vertical integral of reconstructed specific humidity:
    PWV = 1 / (rho_w * g) * Integral( q(p) dp )
and enforces consistency against independent ground GNSS receiver measurements.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


def integrate_boundary_layer_pwv(
    q_profiles: torch.Tensor,
    pressure_levels_hpa: List[float] = [1000.0, 950.0, 925.0, 850.0, 700.0],
    rho_w: float = 1000.0,
    g: float = 9.80665,
) -> torch.Tensor:
    """Perform trapezoidal hydrostatic integration of specific humidity to derive boundary-layer PWV.

    Formula:
        PWV_bl = sum_{k=1}^{L-1} [ (q_k + q_{k+1}) / 2 * Delta p_k ] / (rho_w * g)

    Args:
        q_profiles: Specific humidity tensor of shape (B, L=5, H, W) in g/kg.
        pressure_levels_hpa: Ordered standard pressure levels in hPa (surface to top).
        rho_w: Density of liquid water in kg/m^3 (default: 1000.0).
        g: Gravitational acceleration in m/s^2 (default: 9.80665).

    Returns:
        PWV_bl in mm (B, H, W).
    """
    device = q_profiles.device
    num_levels = q_profiles.shape[1]
    assert num_levels == len(pressure_levels_hpa), "Level count mismatch."

    # Convert specific humidity from g/kg to kg/kg: 1 g/kg = 1e-3 kg/kg
    q_kg_kg = q_profiles * 1.0e-3

    total_pwv_meters = torch.zeros(
        (q_profiles.shape[0], q_profiles.shape[2], q_profiles.shape[3]),
        device=device,
        dtype=q_profiles.dtype,
    )

    for k in range(num_levels - 1):
        p_bottom = pressure_levels_hpa[k]
        p_top = pressure_levels_hpa[k + 1]
        # Delta p in Pascals: (p_bottom - p_top) * 100 Pa/hPa
        delta_p_pa = (p_bottom - p_top) * 100.0

        # Mean layer moisture in kg/kg
        layer_mean_q = 0.5 * (q_kg_kg[:, k, :, :] + q_kg_kg[:, k + 1, :, :])

        # Layer mass increment: delta_pwv = (layer_mean_q * delta_p) / (rho_w * g) in meters
        delta_pwv_m = (layer_mean_q * delta_p_pa) / (rho_w * g)
        total_pwv_meters += delta_pwv_m

    # Convert meters to millimeters: 1 m = 1000 mm
    pwv_bl_mm = total_pwv_meters * 1000.0
    return pwv_bl_mm


class ColumnMassConservationLoss(nn.Module):
    """Loss module enforcing total column mass consistency with ground GNSS PWV observations."""

    def __init__(
        self,
        pressure_levels_hpa: List[float] = [1000.0, 950.0, 925.0, 850.0, 700.0],
        rho_w: float = 1000.0,
        g: float = 9.80665,
    ) -> None:
        super().__init__()
        self.pressure_levels_hpa = pressure_levels_hpa
        self.rho_w = rho_w
        self.g = g

    def forward(
        self,
        q_pred: torch.Tensor,
        gnss_pwv_true: torch.Tensor,
        upper_pwv_residual: torch.Tensor,
        gnss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute column mass conservation loss.

        Args:
            q_pred: Predicted specific humidity (B, L=5, H, W) in g/kg.
            gnss_pwv_true: Ground truth GNSS PWV (B, H, W) in mm.
            upper_pwv_residual: Upper tropospheric PWV above 700 hPa (B, H, W) in mm.
            gnss_mask: Boolean mask of valid station coordinates (B, H, W).

        Returns:
            Scalar L1 column mass penalty.
        """
        # Integrate boundary-layer specific humidity
        pwv_bl = integrate_boundary_layer_pwv(
            q_pred,
            pressure_levels_hpa=self.pressure_levels_hpa,
            rho_w=self.rho_w,
            g=self.g,
        )

        # Full column predicted PWV = Boundary-Layer PWV + Sounder Upper-Level Residual
        pwv_pred_total = pwv_bl + upper_pwv_residual

        diff = torch.abs(pwv_pred_total - gnss_pwv_true)

        if gnss_mask is not None and gnss_mask.sum() > 0:
            loss = (diff * gnss_mask.float()).sum() / (gnss_mask.float().sum() + 1e-7)
        else:
            loss = torch.mean(diff)

        return loss
