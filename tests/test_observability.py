"""Unit tests for calibration metrics, drift metrics, and rank correlation bounds."""

from __future__ import annotations

import pytest

from gemini_decision_router.observability import (
    compute_calibration_metrics,
    spearman_rank_correlation,
)


@pytest.mark.parametrize("invalid_bins", [0, -1, -10])
def test_compute_calibration_metrics_non_positive_bins_raises_value_error(
    invalid_bins: int,
) -> None:
    with pytest.raises(ValueError, match="num_bins must be positive"):
        compute_calibration_metrics([(0.8, 1.0)], num_bins=invalid_bins)

    with pytest.raises(ValueError, match="num_bins must be positive"):
        compute_calibration_metrics([], num_bins=invalid_bins)


@pytest.mark.parametrize(
    ("xs", "ys", "expected"),
    [
        # Tied-rank pairs whose un-clamped float ratio exceeds +/-1.0 by ~2e-16
        ([1.0, 1.0, 2.0, 3.0], [1.0, 1.0, 2.0, 3.0], 1.0),
        ([1.0, 1.0, 2.0, 3.0], [3.0, 3.0, 2.0, 1.0], -1.0),
        ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [60.0, 50.0, 40.0, 30.0, 20.0, 10.0], -1.0),
    ],
)
def test_spearman_rank_correlation_clamps_float_drift_to_unit_interval(
    xs: list[float],
    ys: list[float],
    expected: float,
) -> None:
    rho = spearman_rank_correlation(xs, ys)

    assert -1.0 <= rho <= 1.0
    assert rho == expected
