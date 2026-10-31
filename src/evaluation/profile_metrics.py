"""Vertical Specific Humidity Profile Verification against Radiosondes.

Calculates level-by-level Root Mean Square Error (RMSE), Mean Bias,
and Pearson correlation (r) across [1000, 950, 925, 850, 700] hPa
to benchmark against JGR: Atmospheres criteria.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


class ProfileMetricsEvaluator:
    """Computes standard atmospheric vertical profile verification metrics."""

    TARGET_LEVELS_HPA: List[int] = [1000, 950, 925, 850, 700]

    @classmethod
    def compute_rmse(cls, pred: np.ndarray, truth: np.ndarray) -> float:
        """Root Mean Square Error in g/kg."""
        return float(np.sqrt(np.nanmean((pred - truth) ** 2)))

    @classmethod
    def compute_bias(cls, pred: np.ndarray, truth: np.ndarray) -> float:
        """Mean Bias in g/kg (pred - truth)."""
        return float(np.nanmean(pred - truth))

    @classmethod
    def compute_pearson_r(cls, pred: np.ndarray, truth: np.ndarray) -> float:
        """Pearson correlation coefficient."""
        valid = ~np.isnan(pred) & ~np.isnan(truth)
        p = pred[valid]
        t = truth[valid]
        if len(p) < 2:
            return float("nan")
        p_dev = p - np.mean(p)
        t_dev = t - np.mean(t)
        denom = np.sqrt(np.sum(p_dev**2) * np.sum(t_dev**2))
        if denom == 0:
            return 0.0
        return float(np.sum(p_dev * t_dev) / denom)

    def evaluate_levels(
        self,
        pred_profiles: np.ndarray,
        true_profiles: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics for each vertical level.

        Args:
            pred_profiles: Array of shape (N, L=5) in g/kg.
            true_profiles: Array of shape (N, L=5) in g/kg.

        Returns:
            Dictionary mapped by level label (e.g. '1000 hPa') with RMSE, Bias, and r.
        """
        results: Dict[str, Dict[str, float]] = {}

        for idx, lvl in enumerate(self.TARGET_LEVELS_HPA):
            p_slice = pred_profiles[:, idx]
            t_slice = true_profiles[:, idx]

            lvl_name = f"{lvl} hPa"
            results[lvl_name] = {
                "RMSE_g_kg": self.compute_rmse(p_slice, t_slice),
                "Bias_g_kg": self.compute_bias(p_slice, t_slice),
                "Pearson_r": self.compute_pearson_r(p_slice, t_slice),
            }

        return results


def main() -> None:
    evaluator = ProfileMetricsEvaluator()
    # Mock evaluation with synthetic test profiles
    n = 100
    true_q = np.array([16.0, 13.5, 11.2, 7.8, 3.5]) + np.random.normal(0, 1.0, (n, 5))
    pred_q = true_q + np.random.normal(0.1, 0.9, (n, 5))

    metrics = evaluator.evaluate_levels(pred_q, true_q)
    print("Level-by-Level Radiosonde Verification:")
    for lvl, vals in metrics.items():
        print(
            f"  {lvl:10s} -> RMSE: {vals['RMSE_g_kg']:.3f} g/kg | "
            f"Bias: {vals['Bias_g_kg']:+.3f} g/kg | "
            f"r: {vals['Pearson_r']:.3f}"
        )


if __name__ == "__main__":
    main()
