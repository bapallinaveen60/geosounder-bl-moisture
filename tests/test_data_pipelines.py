"""Unit tests for data ingestion, physics formulas, and spatial transformations."""

import numpy as np

from src.data.gnss_pwv_pipeline import GNSSPWVPipeline
from src.data.radar_dwr_pipeline import RadarDWRPipeline
from src.data.radiosonde_collocation import RadiosondeCollocator
from src.data.topography_auxiliary import TopographyAuxiliaryEngine


def test_saastamoinen_zhd():
    pipeline = GNSSPWVPipeline()
    # At MSL, lat=45 deg, Psrf=1013.25 hPa: ZHD is ~2.27 m
    zhd = pipeline.compute_zhd_saastamoinen(
        surface_pressure_hpa=1013.25,
        latitude_deg=45.0,
        station_height_m=0.0,
    )
    assert 2.25 < zhd < 2.35


def test_askne_nordius_pi():
    pipeline = GNSSPWVPipeline()
    # For Tm ~ 290 K, Pi is approximately 0.15 - 0.16
    pi_factor = pipeline.compute_conversion_factor_pi(tm_k=290.0)
    assert 140.0 < pi_factor < 170.0


def test_radiosonde_dewpoint_conversion():
    collocator = RadiosondeCollocator()
    # At Td = 20 C, p = 1000 hPa -> q is ~14.7 g/kg
    q_calc = collocator.dewpoint_to_specific_humidity(
        td_celsius=20.0, pressure_hpa=1000.0
    )
    assert 13.5 < q_calc < 16.0


def test_radiosonde_interpolation():
    collocator = RadiosondeCollocator()
    p_raw = np.array([1010.0, 960.0, 920.0, 840.0, 690.0])
    q_raw = np.array([18.0, 15.0, 12.0, 8.0, 4.0])

    q_interp = collocator.interpolate_profile(p_raw, q_raw)
    assert len(q_interp) == 5
    # Target levels: 1000, 950, 925, 850, 700 hPa
    assert q_interp[0] > q_interp[1] > q_interp[2] > q_interp[3] > q_interp[4]


def test_topography_morphometrics():
    dem = np.zeros((50, 50), dtype=np.float32)
    # Create an incline
    dem += np.linspace(0, 1000, 50)[:, None]
    morph = TopographyAuxiliaryEngine.compute_terrain_morphometrics(dem)

    assert "slope" in morph
    assert "aspect" in morph
    assert "curvature" in morph
    assert morph["slope"].shape == (50, 50)


def test_radar_ci_detection():
    radar = RadarDWRPipeline()
    grid = np.full((100, 100), 20.0)
    # Create convective cell with 45 dBZ (> 35 dBZ threshold)
    grid[45:55, 45:55] = 45.0

    res = radar.detect_convective_initiation(grid)
    assert res["new_ci_mask"].sum() == 100

    pooled = radar.spatial_box_pooling(res["new_ci_mask"], box_size_pixels=5)
    assert pooled.shape == (20, 20)
    assert pooled.sum() > 0
