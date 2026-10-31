"""Vertical Atmospheric Cross-Sections and Sounding Comparison.

Plots vertical cross-sections through pre-convective dryline boundaries,
comparing Operational INSAT-3DR L2 Sounder retrievals, ThermoBL-Net predictions,
and IMD Radiosonde benchmark soundings.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_vertical_cross_section(
    pressure_levels_hpa: List[int],
    insat_l2_cross: np.ndarray,
    thermobonet_cross: np.ndarray,
    radiosonde_cross: np.ndarray,
    longitude_transect: np.ndarray,
    save_path: Optional[str | Path] = None,
) -> None:
    """Generate 3-panel vertical cross-section figure comparing L2, Model, and Sonde.

    Args:
        pressure_levels_hpa: Vertical levels [1000, 950, 925, 850, 700].
        insat_l2_cross: (5, X) array of operational L2 moisture.
        thermobonet_cross: (5, X) array of ThermoBL-Net moisture.
        radiosonde_cross: (5, X) array of interpolated in-situ sounding moisture.
        longitude_transect: 1D array of longitude along west-to-east transect.
        save_path: File path to save figure.
    """
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, sharey=True, dpi=300)
    cmap = plt.get_cmap("Spectral_r")
    levels = np.linspace(2.0, 20.0, 37)

    panels = [
        ("Operational INSAT-3DR L2 Retrieval (Exhibiting Dry Bias)", insat_l2_cross),
        ("ThermoBL-Net (Physics-Constrained with GNSS Mass Anchor)", thermobonet_cross),
        ("In-Situ Radiosonde Analysis Benchmark", radiosonde_cross),
    ]

    for ax, (title, data) in zip(axes, panels):
        cf = ax.contourf(
            longitude_transect,
            pressure_levels_hpa,
            data,
            levels=levels,
            cmap=cmap,
            extend="both",
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Pressure [hPa]", fontsize=11)
        ax.invert_yaxis()  # Standard meteorological vertical coordinate: surface at bottom
        ax.grid(True, linestyle=":", alpha=0.5)

    axes[-1].set_xlabel("Longitude [$^\\circ$E] (West-to-East Dryline Transect)", fontsize=11)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cb = fig.colorbar(cf, cax=cbar_ax)
    cb.set_label("Specific Humidity $q$ [$\\mathrm{g\\ kg^{-1}}$]", fontsize=12)

    plt.subplots_adjust(right=0.90, hspace=0.25)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved cross-section figure to {save_path}")
    plt.close()


def main() -> None:
    levels = [1000, 950, 925, 850, 700]
    lons = np.linspace(82.0, 90.0, 100)

    # Synthetic cross-sections: sharp moisture jump from 85.5 to 87.5 E
    sig = 1.0 / (1.0 + np.exp(-(lons - 86.2) * 4.0))
    vert_falloff = np.array([1.0, 0.85, 0.75, 0.55, 0.28])[:, None]

    truth = (6.0 + 13.0 * sig[None, :]) * vert_falloff
    # Operational L2 exhibits dry bias in lower boundary:
    insat_l2 = truth * np.array([0.78, 0.82, 0.86, 0.92, 0.98])[:, None]
    # ThermoBL-Net restores true boundary-layer mass
    thermobonet = truth + np.random.normal(0, 0.3, size=truth.shape)

    plot_vertical_cross_section(
        pressure_levels_hpa=levels,
        insat_l2_cross=insat_l2,
        thermobonet_cross=thermobonet,
        radiosonde_cross=truth,
        longitude_transect=lons,
        save_path="figures/demo_vertical_cross_section.png",
    )


if __name__ == "__main__":
    main()
