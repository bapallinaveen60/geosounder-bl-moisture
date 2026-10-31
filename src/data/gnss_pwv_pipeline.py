"""Ground GNSS Zenith Tropospheric Delay (ZTD) to Precipitable Water Vapor (PWV) Pipeline.

Implements the Saastamoinen hydrostatic model and Askne-Nordius physical formulation
to derive column water vapor from Survey of India CORS and IMD GNSS networks.
Includes Voronoi-Kriging spatial interpolation onto the 4 km grid.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


class GNSSPWVPipeline:
    """Derives high-precision continuous PWV from GNSS ZTD observations."""

    # Physical Constants
    R_V: float = 461.51  # Specific gas constant for water vapor [J / (kg * K)]
    RHO_W: float = 1000.0  # Density of liquid water [kg / m^3]
    K2_PRIME: float = 22.1  # Refractivity constant [K / hPa]
    K3: float = 3.754e5  # Refractivity constant [K^2 / hPa]

    def __init__(
        self,
        grid_bounds: Tuple[float, float, float, float] = (12.0, 28.0, 75.0, 92.0),
        grid_res_deg: float = 0.04,
    ) -> None:
        """Initialize the GNSS PWV pipeline.

        Args:
            grid_bounds: (lat_min, lat_max, lon_min, lon_max) bounding box.
            grid_res_deg: Grid resolution in degrees (~4 km).
        """
        self.lat_min, self.lat_max, self.lon_min, self.lon_max = grid_bounds
        self.grid_res = grid_res_deg
        self.target_lats = np.arange(self.lat_min, self.lat_max, self.grid_res)
        self.target_lons = np.arange(self.lon_min, self.lon_max, self.grid_res)

    @classmethod
    def compute_zhd_saastamoinen(
        cls,
        surface_pressure_hpa: float | np.ndarray,
        latitude_deg: float | np.ndarray,
        station_height_m: float | np.ndarray,
    ) -> float | np.ndarray:
        """Calculate Zenith Hydrostatic Delay (ZHD) using the Saastamoinen (1972) model.

        Formula:
            ZHD = 0.0022768 * P_srf / (1 - 0.00266 * cos(2*lat) - 0.00028 * H_km)

        Args:
            surface_pressure_hpa: Barometric pressure at station level in hPa.
            latitude_deg: Geodetic latitude in degrees.
            station_height_m: Orthometric/ellipsoidal height above MSL in meters.

        Returns:
            ZHD in meters.
        """
        lat_rad = np.radians(latitude_deg)
        h_km = station_height_m / 1000.0
        denom = 1.0 - 0.00266 * np.cos(2.0 * lat_rad) - 0.00028 * h_km
        zhd = (0.0022768 * surface_pressure_hpa) / denom
        return zhd

    @classmethod
    def estimate_tm_bevis(cls, surface_temp_k: float | np.ndarray) -> float | np.ndarray:
        """Estimate weighted mean atmospheric temperature (Tm) from surface temperature (Ts).

        Bevis et al. (1992) linear formulation:
            Tm = 70.2 + 0.72 * Ts
        """
        return 70.2 + 0.72 * surface_temp_k

    @classmethod
    def compute_conversion_factor_pi(cls, tm_k: float | np.ndarray) -> float | np.ndarray:
        """Calculate the dimensionless water vapor conversion factor Pi (Askne & Nordius, 1987).

        Formula:
            Pi = 10^8 / [ rho_w * R_v * (k2' + k3 / Tm) ]
        Note: The 10^8 numerator accounts for the 10^6 refractivity scaling and 10^2 hPa-to-Pa conversion.

        Args:
            tm_k: Atmospheric weighted mean temperature in Kelvin.

        Returns:
            Conversion factor converting ZWD (in meters) to PWV (in mm).
            Dimensionless Pi (~0.16) multiplied by 1000 mm/m yields ~160.
        """
        # Dimensionless Pi: meters of PWV per meter of ZWD
        pi_dimensionless = 1.0e8 / (cls.RHO_W * cls.R_V * (cls.K2_PRIME + (cls.K3 / tm_k)))
        return pi_dimensionless * 1000.0  # Output factor converts ZWD in meters to PWV in mm

    def ztd_to_pwv(
        self,
        ztd_m: float | np.ndarray,
        surface_pressure_hpa: float | np.ndarray,
        surface_temp_k: float | np.ndarray,
        latitude_deg: float | np.ndarray,
        station_height_m: float | np.ndarray,
    ) -> float | np.ndarray:
        """Complete physical inversion from GNSS ZTD to Precipitable Water Vapor (PWV).

        Args:
            ztd_m: Zenith Tropospheric Delay in meters.
            surface_pressure_hpa: Surface pressure in hPa.
            surface_temp_k: Surface air temperature in Kelvin.
            latitude_deg: Geodetic latitude.
            station_height_m: Station elevation in meters.

        Returns:
            Precipitable Water Vapor (PWV) in mm.
        """
        zhd_m = self.compute_zhd_saastamoinen(surface_pressure_hpa, latitude_deg, station_height_m)
        zwd_m = np.maximum(0.0, ztd_m - zhd_m)
        tm_k = self.estimate_tm_bevis(surface_temp_k)
        factor_pi = self.compute_conversion_factor_pi(tm_k)
        pwv_mm = factor_pi * zwd_m
        return pwv_mm

    def interpolate_pwv_grid(
        self,
        station_lats: np.ndarray,
        station_lons: np.ndarray,
        station_pwv: np.ndarray,
        method: str = "kriging",
    ) -> np.ndarray:
        """Spatial interpolation of point GNSS PWV observations onto uniform 4 km grid.

        Supports Ordinary Kriging via PyKrige or Scipy Rbf/Linear.
        """
        mesh_lon, mesh_lat = np.meshgrid(self.target_lons, self.target_lats)

        if method == "kriging":
            try:
                from pykrige.ok import OrdinaryKriging

                ok = OrdinaryKriging(
                    station_lons,
                    station_lats,
                    station_pwv,
                    variogram_model="spherical",
                    verbose=False,
                    enable_plotting=False,
                )
                z_interp, _ = ok.execute("grid", self.target_lons, self.target_lats)
                return np.array(z_interp, dtype=np.float32)
            except Exception:
                # Graceful fallback to radial basis or linear interpolation if pykrige encounters ill-conditioning
                pass

        from scipy.interpolate import griddata

        points = np.column_stack((station_lats, station_lons))
        gridded = griddata(
            points,
            station_pwv,
            (mesh_lat, mesh_lon),
            method="linear",
            fill_value=np.nanmean(station_pwv),
        )
        return gridded.astype(np.float32)


def main() -> None:
    """CLI test routine."""
    parser = argparse.ArgumentParser(description="GNSS ZTD to PWV pipeline")
    parser.add_argument("--config", type=str, default="configs/data_config.yaml")
    args = parser.parse_args()

    pipeline = GNSSPWVPipeline()
    # Test sample calculation
    # Typical values over Kolkata during pre-monsoon: Psrf = 1008 hPa, Ts = 308 K (35 C), ZTD = 2.45 m
    pwv_test = pipeline.ztd_to_pwv(
        ztd_m=2.45,
        surface_pressure_hpa=1008.0,
        surface_temp_k=308.15,
        latitude_deg=22.57,
        station_height_m=9.0,
    )
    print(f"Sample Inversion Result: ZTD 2.45 m -> PWV = {pwv_test:.2f} mm")


if __name__ == "__main__":
    main()
