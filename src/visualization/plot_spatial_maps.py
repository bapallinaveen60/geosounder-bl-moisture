"""2D Spatial Maps of Boundary-Layer Specific Humidity and Drylines.

Generates publication-quality 2D horizontal contour maps of specific humidity
at 925 hPa and 1000 hPa showing sharp dryline moisture discontinuities over Eastern India.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_dryline_moisture_map(
    q_field_925: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    save_path: Optional[str | Path] = None,
    title: str = "ThermoBL-Net: 925 hPa Specific Humidity & Dryline Front",
) -> None:
    """Plot 2D specific humidity field with dryline gradient annotation.

    Args:
        q_field_925: 2D array of q at 925 hPa in g/kg.
        lats: 1D array of latitude coordinates.
        lons: 1D array of longitude coordinates.
        save_path: Output file path.
        title: Figure title.
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # Specific humidity contour
    levels = np.linspace(4.0, 22.0, 37)
    cmap = plt.get_cmap("YlGnBu")

    mesh_lon, mesh_lat = np.meshgrid(lons, lats)
    cf = ax.contourf(
        mesh_lon, mesh_lat, q_field_925, levels=levels, cmap=cmap, extend="both"
    )
    cb = fig.colorbar(cf, ax=ax, orientation="vertical", pad=0.03, aspect=25)
    cb.set_label(
        "Specific Humidity $q$ at 925 hPa [$\\mathrm{g\\ kg^{-1}}$]", fontsize=12
    )

    # Highlight dryline boundary (sharp gradient corridor around 12-14 g/kg)
    cs = ax.contour(
        mesh_lon,
        mesh_lat,
        q_field_925,
        levels=[13.0],
        colors="crimson",
        linewidths=2.5,
        linestyles="--",
    )
    ax.clabel(cs, fmt="Dryline: %1.0f g/kg", inline=True, fontsize=11, colors="crimson")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Longitude [$^\\circ$E]", fontsize=12)
    ax.set_ylabel("Latitude [$^\\circ$N]", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved dryline map to {save_path}")
    plt.close()


def main() -> None:
    lats = np.linspace(18.0, 27.0, 200)
    lons = np.linspace(80.0, 92.0, 250)
    mesh_lon, mesh_lat = np.meshgrid(lons, lats)

    # Simulated dryline: sharp moisture step between dry continental air and moist Bay of Bengal air
    sigmoid = 1.0 / (1.0 + np.exp(-(mesh_lon - 86.5) * 3.0))
    q_sim = 6.0 + 13.0 * sigmoid + np.random.normal(0, 0.4, size=mesh_lon.shape)

    plot_dryline_moisture_map(
        q_sim, lats, lons, save_path="figures/demo_dryline_925hpa.png"
    )


if __name__ == "__main__":
    main()
