"""Unit tests for validation metrics and statistical scoring."""

import numpy as np
import pytest

from src.evaluation.ci_nowcasting import ConvectiveInitiationNowcaster
from src.evaluation.gnss_verification import GNSSColumnMassVerifier
from src.evaluation.profile_metrics import ProfileMetricsEvaluator


def test_profile_metrics():
    evaluator = ProfileMetricsEvaluator()
    truth = np.array([[15.0, 12.0, 10.0, 7.0, 3.0]])
    pred = np.array([[16.0, 12.0, 9.0, 7.0, 3.0]])

    metrics = evaluator.evaluate_levels(pred, truth)
    # Check 1000 hPa error (diff = 1.0)
    assert np.isclose(metrics["1000 hPa"]["RMSE_g_kg"], 1.0)
    assert np.isclose(metrics["1000 hPa"]["Bias_g_kg"], 1.0)
    # Check 950 hPa (exact match)
    assert np.isclose(metrics["950 hPa"]["RMSE_g_kg"], 0.0)


def test_gnss_column_mass_metrics():
    verifier = GNSSColumnMassVerifier()
    truth = np.array([40.0, 45.0, 50.0, 55.0])
    pred = np.array([41.0, 44.0, 51.0, 54.0])

    res = verifier.compute_column_metrics(pred, truth)
    assert np.isclose(res["PWV_RMSE_mm"], 1.0)
    assert np.isclose(res["PWV_Bias_mm"], 0.0)
    assert res["Pearson_r"] > 0.95


def test_ci_contingency_scores():
    nowcaster = ConvectiveInitiationNowcaster()
    # 100 samples
    pred = np.array([True, True, False, False, True])
    truth = np.array([True, False, True, False, True])

    scores = nowcaster.compute_contingency_scores(pred, truth)
    # Hits: [0, 4] -> 2
    # Misses: [2] -> 1
    # False Alarms: [1] -> 1
    # POD: 2 / (2 + 1) = 2/3 = 0.667
    # FAR: 1 / (2 + 1) = 1/3 = 0.333
    # CSI: 2 / (2 + 1 + 1) = 2/4 = 0.50
    assert np.isclose(scores["Hits"], 2)
    assert np.isclose(scores["Misses"], 1)
    assert np.isclose(scores["False_Alarms"], 1)
    assert np.isclose(scores["POD"], 2.0 / 3.0)
    assert np.isclose(scores["FAR"], 1.0 / 3.0)
    assert np.isclose(scores["CSI"], 0.50)
