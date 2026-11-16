"""Tests for post-hoc Platt calibration and Platt-calibrated `noul_eu` routing."""

import pytest

from gemini_decision_router import (
    PlattParams,
    RouterPolicyConfig,
    fit_per_tier_platt_scaling,
    fit_platt_scaling,
)
from gemini_decision_router.router import select_tier_from_noul_eu


def test_platt_params_and_fit_fixes_systematic_underconfidence() -> None:
    raw_probs = [0.50, 0.55, 0.60, 0.62, 0.65, 0.70, 0.75, 0.80, 0.45, 0.58] * 3
    labels = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0] * 3
    params = fit_platt_scaling(raw_probs, labels)
    calibrated = [params.apply(p) for p in raw_probs]

    assert sum(calibrated) / len(calibrated) > sum(raw_probs) / len(raw_probs)
    assert all(0.01 <= p <= 0.99 for p in calibrated)
    assert PlattParams().apply(0.5) == pytest.approx(0.5)


def test_fit_platt_scaling_validation_and_per_tier() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        fit_platt_scaling([], [])
    with pytest.raises(ValueError, match="equal length"):
        fit_platt_scaling([0.5], [1.0, 0.0])

    per_tier = fit_per_tier_platt_scaling(
        {"t0_flash_lite_min": [0.2, 0.8, 0.3, 0.9], "t1_hydrogen_low": []},
        {"t0_flash_lite_min": [0.0, 1.0, 0.0, 1.0]},
    )
    assert set(per_tier) == {"t0_flash_lite_min", "t1_hydrogen_low"}
    assert per_tier["t1_hydrogen_low"] == PlattParams(a=1.0, b=0.0)


def test_fit_platt_scaling_falls_back_to_identity_on_non_positive_slope() -> None:
    inverted = fit_platt_scaling([0.35, 0.45, 0.55, 0.65] * 3, [1.0, 1.0, 0.0, 0.0] * 3)
    assert inverted == PlattParams(a=1.0, b=0.0)


def test_noul_eu_applies_platt_params_before_pava() -> None:
    raw_nouls = {
        "t0_flash_lite_min": 0.40,
        "t1_hydrogen_low": 0.55,
        "t2_hydrogen_med": 0.85,
        "t3_hydrogen_high": 0.95,
        "t4_hydrogen_xhigh": 0.97,
    }
    uncal_cfg = RouterPolicyConfig(policy="noul_eu", eu_lambda=2.0)
    uncal_tid, _, _ = select_tier_from_noul_eu(raw_nouls, 500, uncal_cfg)

    cal_cfg = RouterPolicyConfig(
        policy="noul_eu",
        eu_lambda=2.0,
        noul_platt_params={"t1_hydrogen_low": PlattParams(a=1.0, b=3.0)},
    )
    cal_tid, cal_probs, _ = select_tier_from_noul_eu(raw_nouls, 500, cal_cfg)
    assert cal_probs["t1_hydrogen_low"] > 0.90
    assert cal_tid == "t1_hydrogen_low"
    assert uncal_tid != cal_tid or cal_probs["t1_hydrogen_low"] > 0.90
