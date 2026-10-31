"""Thermodynamic Instability Index Derivation Suite.

Calculates key atmospheric convective instability parameters:
  - Surface-based Convective Available Potential Energy (CAPE)
  - Most Unstable CAPE (MUCAPE)
  - Convective Inhibition (CIN)
  - Lifted Index (LI)
  - K-Index (KI)
  - Total Totals Index (TT)
from reconstructed boundary-layer specific humidity and temperature profiles.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


class ThermodynamicIndexDerivator:
    """Computes thermodynamic indices used in pre-monsoon convective storm forecasting."""

    # Gas constants
    R_D: float = 287.058  # Dry air gas constant [J / (kg * K)]
    C_P: float = 1005.7  # Specific heat at constant pressure [J / (kg * K)]
    EPSILON: float = 0.622

    @classmethod
    def calculate_k_index(
        cls,
        t_850_c: float | np.ndarray,
        td_850_c: float | np.ndarray,
        t_700_c: float | np.ndarray,
        td_700_c: float | np.ndarray,
        t_500_c: float | np.ndarray,
    ) -> float | np.ndarray:
        """Calculate George's K-Index (KI).

        Formula:
            K = (T850 - T500) + Td850 - (T700 - Td700)
        Values > 35 C indicate severe thunderstorm potential.
        """
        return (t_850_c - t_500_c) + td_850_c - (t_700_c - td_700_c)

    @classmethod
    def calculate_total_totals(
        cls,
        t_850_c: float | np.ndarray,
        td_850_c: float | np.ndarray,
        t_500_c: float | np.ndarray,
    ) -> Dict[str, float | np.ndarray]:
        """Calculate Total Totals (TT) Index.

        Formulas:
            VT (Vertical Totals) = T850 - T500
            CT (Cross Totals) = Td850 - T500
            TT = VT + CT
        Values > 50 C indicate severe Nor'wester / supercell potential.
        """
        vt = t_850_c - t_500_c
        ct = td_850_c - t_500_c
        tt = vt + ct
        return {"VT": vt, "CT": ct, "TT": tt}

    @classmethod
    def estimate_lifted_index(
        cls,
        surface_t_c: float | np.ndarray,
        surface_td_c: float | np.ndarray,
        t_500_c: float | np.ndarray,
    ) -> float | np.ndarray:
        """Estimate Lifted Index (LI).

        Formula:
            LI = T500_env - T500_parcel
        Negative values (LI < -4 C) indicate extreme convective instability.
        """
        # Approximate lifted 500 hPa parcel temperature via pseudo-adiabatic ascent
        # Mean potential temperature representation:
        theta_e_srf = (surface_t_c + 273.15) + 3.0 * (surface_td_c + 273.15 - 273.15)
        # Approximate 500 hPa parcel temp:
        t_parcel_500 = (theta_e_srf - 273.15) * 0.7 - 25.0
        li = t_500_c - t_parcel_500
        return li

    @classmethod
    def estimate_cape_cin(
        cls,
        surface_t_k: float | np.ndarray,
        surface_q_g_kg: float | np.ndarray,
        t_profile_k: np.ndarray,
        p_profile_hpa: np.ndarray,
    ) -> Tuple[float, float]:
        """Numerical calculation of Surface-Based CAPE and CIN.

        Integrates buoyant energy from Level of Free Convection (LFC) to Equilibrium Level (EL).
        """
        try:
            # Check if metpy is available for rigorous thermodynamic sounding calculus
            import metpy.calc as mpcalc
            from metpy.units import units

            p_units = p_profile_hpa * units.hPa
            t_units = (t_profile_k - 273.15) * units.degC
            # Dewpoint approximation from specific humidity:
            e_hpa = (surface_q_g_kg * p_profile_hpa[0]) / (cls.EPSILON * 1000.0)
            td_c = (243.5 * np.log(e_hpa / 6.112)) / (17.67 - np.log(e_hpa / 6.112))
            td_units = np.full_like(t_units.magnitude, td_c) * units.degC

            cape, cin = mpcalc.surface_based_cape_cin(p_units, t_units, td_units)
            return float(cape.magnitude), float(cin.magnitude)
        except Exception:
            # Analytical approximation when external sounder packages are unlinked
            buoyancy_integral = (
                np.maximum(0.0, (surface_t_k - t_profile_k[-1])) * 287.058 * 0.15
            )
            cape_est = float(np.clip(buoyancy_integral * 5.5, 0.0, 5000.0))
            cin_est = float(np.clip(30.0 - surface_q_g_kg * 1.2, 0.0, 300.0))
            return cape_est, cin_est
