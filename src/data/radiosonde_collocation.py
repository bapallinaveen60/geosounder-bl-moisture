"""IMD Radiosonde / GPS-Sonde Vertical Interpolation and Spatiotemporal Collocation.

Ingests IMD upper-air operational soundings from 14 primary pre-monsoon storm stations,
interpolates specific humidity profiles to standard boundary-layer levels
[1000, 950, 925, 850, 700] hPa, and matches with INSAT pixels within +/- 30 min and 0.05 deg.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


class RadiosondeCollocator:
    """Collocation and interpolation engine for radiosonde ground truth soundings."""

    TARGET_LEVELS_HPA: List[int] = [1000, 950, 925, 850, 700]

    # Primary IMD sounding network stations in pre-monsoon corridor
    PRIMARY_STATIONS: Dict[str, Tuple[float, float, float]] = {
        "Kolkata": (22.65, 88.45, 6.0),
        "Bhubaneswar": (20.25, 85.82, 42.0),
        "Ranchi": (23.32, 85.32, 652.0),
        "Patna": (25.60, 85.10, 60.0),
        "Agartala": (23.88, 91.25, 15.0),
        "Hyderabad": (17.45, 78.47, 545.0),
        "Machilipatnam": (16.20, 81.15, 3.0),
        "Visakhapatnam": (17.72, 83.25, 3.0),
        "Gopalpur": (19.27, 84.87, 17.0),
        "Nagpur": (21.10, 79.05, 310.0),
        "Raipur": (21.23, 81.65, 298.0),
        "Gorakhpur": (26.75, 83.37, 77.0),
        "Digha": (21.62, 87.50, 4.0),
        "Cuttack": (20.47, 85.88, 36.0),
    }

    def __init__(
        self,
        time_window_mins: int = 30,
        spatial_radius_deg: float = 0.05,
    ) -> None:
        self.time_window_mins = time_window_mins
        self.spatial_radius_deg = spatial_radius_deg

    @staticmethod
    def dewpoint_to_specific_humidity(
        td_celsius: float | np.ndarray,
        pressure_hpa: float | np.ndarray,
    ) -> float | np.ndarray:
        """Convert dewpoint temperature and barometric pressure to specific humidity (q).

        Formula:
            e = 6.112 * exp((17.67 * Td) / (Td + 243.5))   [vapor pressure in hPa]
            w = 0.622 * e / (p - e)                        [mixing ratio in kg/kg]
            q = (w / (1 + w)) * 1000                       [specific humidity in g/kg]
        """
        # Tetens formulation for vapor pressure over water
        e = 6.112 * np.exp((17.67 * td_celsius) / (td_celsius + 243.5))
        # Clamp vapor pressure strictly below total atmospheric pressure
        e = np.minimum(e, pressure_hpa * 0.99)
        mixing_ratio = (0.622 * e) / (pressure_hpa - e)
        q_g_per_kg = (mixing_ratio / (1.0 + mixing_ratio)) * 1000.0
        return np.maximum(0.0, q_g_per_kg)

    def interpolate_profile(
        self,
        pressure_levels: np.ndarray,
        specific_humidity: np.ndarray,
    ) -> np.ndarray:
        """Interpolate specific humidity profile to target levels [1000, 950, 925, 850, 700] hPa.

        Uses interpolation in log-pressure coordinates: ln(p).
        """
        valid_mask = ~np.isnan(pressure_levels) & ~np.isnan(specific_humidity)
        p_clean = pressure_levels[valid_mask]
        q_clean = specific_humidity[valid_mask]

        if len(p_clean) < 2:
            return np.full(len(self.TARGET_LEVELS_HPA), np.nan, dtype=np.float32)

        # Sort descending by pressure (surface to upper levels)
        sort_idx = np.argsort(p_clean)[::-1]
        p_sorted = p_clean[sort_idx]
        q_sorted = q_clean[sort_idx]

        log_p = np.log(p_sorted)
        target_log_p = np.log(np.array(self.TARGET_LEVELS_HPA, dtype=np.float64))

        # Linear interpolation in log(p)
        interp_q = np.interp(target_log_p, log_p[::-1], q_sorted[::-1], left=np.nan, right=np.nan)
        return interp_q.astype(np.float32)

    def find_collocated_pixels(
        self,
        station_lat: float,
        station_lon: float,
        grid_lats: np.ndarray,
        grid_lons: np.ndarray,
    ) -> Tuple[int, int]:
        """Find the row and column indices of the grid pixel closest to station coordinates."""
        lat_diff = np.abs(grid_lats - station_lat)
        lon_diff = np.abs(grid_lons - station_lon)

        row_idx = int(np.argmin(lat_diff))
        col_idx = int(np.argmin(lon_diff))

        # Check distance threshold
        dist_sq = (grid_lats[row_idx] - station_lat) ** 2 + (grid_lons[col_idx] - station_lon) ** 2
        if np.sqrt(dist_sq) > self.spatial_radius_deg:
            # Beyond allowable spatial collocation tolerance
            return -1, -1

        return row_idx, col_idx


def main() -> None:
    """CLI test routine."""
    collocator = RadiosondeCollocator()
    p_raw = np.array([1006.0, 980.0, 940.0, 915.0, 850.0, 780.0, 700.0, 500.0])
    td_raw = np.array([24.0, 22.5, 20.0, 18.2, 14.0, 8.0, 2.0, -15.0])
    q_raw = collocator.dewpoint_to_specific_humidity(td_raw, p_raw)

    q_interpolated = collocator.interpolate_profile(p_raw, q_raw)
    print("Test Sounding Interpolation across [1000, 950, 925, 850, 700] hPa:")
    for lvl, q_val in zip(collocator.TARGET_LEVELS_HPA, q_interpolated):
        print(f"  Level {lvl:4d} hPa: {q_val:6.2f} g/kg")


if __name__ == "__main__":
    main()
