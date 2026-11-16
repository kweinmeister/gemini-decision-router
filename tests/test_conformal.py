"""Tests for finite-sample Conformal Risk Control (CRC) routing."""

from types import SimpleNamespace
from typing import Any

import pytest

from gemini_decision_router import (
    DecisionRouter,
    RouterPolicyConfig,
    calibrate_conformal_cdf_threshold,
    select_tier_from_cdf_threshold,
)

TIERS = (
    "t0_flash_lite_min",
    "t1_hydrogen_low",
    "t2_hydrogen_med",
    "t3_hydrogen_high",
    "t4_hydrogen_xhigh",
)


@pytest.fixture
def crc_calibration_sample() -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    probs_list = [
        {"t0_flash_lite_min": 0.85, "t1_hydrogen_low": 0.15},
        {"t0_flash_lite_min": 0.60, "t1_hydrogen_low": 0.40},
        {"t0_flash_lite_min": 0.10, "t1_hydrogen_low": 0.85, "t2_hydrogen_med": 0.05},
        {"t0_flash_lite_min": 0.05, "t1_hydrogen_low": 0.25, "t2_hydrogen_med": 0.70},
    ] * 25
    accs_list = [
        {
            "t0_flash_lite_min": 1.0,
            "t1_hydrogen_low": 1.0,
            "t2_hydrogen_med": 1.0,
            "t3_hydrogen_high": 1.0,
            "t4_hydrogen_xhigh": 1.0,
        },
        {
            "t0_flash_lite_min": 1.0,
            "t1_hydrogen_low": 1.0,
            "t2_hydrogen_med": 1.0,
            "t3_hydrogen_high": 1.0,
            "t4_hydrogen_xhigh": 1.0,
        },
        {
            "t0_flash_lite_min": 0.0,
            "t1_hydrogen_low": 1.0,
            "t2_hydrogen_med": 1.0,
            "t3_hydrogen_high": 1.0,
            "t4_hydrogen_xhigh": 1.0,
        },
        {
            "t0_flash_lite_min": 0.0,
            "t1_hydrogen_low": 0.0,
            "t2_hydrogen_med": 1.0,
            "t3_hydrogen_high": 1.0,
            "t4_hydrogen_xhigh": 1.0,
        },
    ] * 25
    return probs_list, accs_list


def test_calibrate_conformal_cdf_threshold_respects_alpha(
    crc_calibration_sample: tuple[list[dict[str, float]], list[dict[str, float]]],
) -> None:
    probs_list, accs_list = crc_calibration_sample
    res = calibrate_conformal_cdf_threshold(probs_list, accs_list, TIERS, target_alpha=0.02)
    assert res.finite_sample_bound <= 0.02
    assert 0.30 <= res.threshold <= 0.99


def test_calibrate_conformal_cdf_threshold_weighted_respects_alpha(
    crc_calibration_sample: tuple[list[dict[str, float]], list[dict[str, float]]],
) -> None:
    probs_list, accs_list = crc_calibration_sample
    weights = [2.0 if i % 4 == 0 else 1.0 for i in range(len(probs_list))]
    res = calibrate_conformal_cdf_threshold(
        probs_list, accs_list, TIERS, target_alpha=0.05, weights=weights
    )
    assert res.finite_sample_bound <= 0.05


def test_calibrate_conformal_cdf_threshold_falls_back_to_one_when_unattainable(
    crc_calibration_sample: tuple[list[dict[str, float]], list[dict[str, float]]],
) -> None:
    probs_list, accs_list = crc_calibration_sample
    res = calibrate_conformal_cdf_threshold(probs_list[:5], accs_list[:5], TIERS, target_alpha=0.05)
    assert res.threshold == 1.0
    assert res.empirical_risk == pytest.approx(0.0)
    assert res.finite_sample_bound == pytest.approx(1.0 / 6.0)

    # Non-zero regret at theta=1.0 (best tier is t1=1.0, last tier t4=0.0) must be reflected
    regret_probs = [{"t0_flash_lite_min": 0.5, "t4_hydrogen_xhigh": 0.5}] * 4
    regret_accs = [{"t0_flash_lite_min": 0.0, "t1_hydrogen_low": 1.0, "t4_hydrogen_xhigh": 0.0}] * 4
    regret_res = calibrate_conformal_cdf_threshold(
        regret_probs, regret_accs, TIERS, target_alpha=0.10
    )
    assert regret_res.threshold == 1.0
    assert regret_res.empirical_risk == pytest.approx(1.0)
    assert regret_res.finite_sample_bound == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("probs", "accs", "tiers", "kwargs", "match"),
    [
        ([], [], TIERS, {}, "non-empty and equal length"),
        ([{"t0": 1.0}], [], TIERS, {}, "non-empty and equal length"),
        ([{"t0": 1.0}], [{"t0": 1.0}], (), {}, "ordered_tier_ids must not be empty"),
        ([{"t0": 1.0}], [{"t0": 1.0}], ("t0",), {"target_alpha": 0.0}, "target_alpha"),
        ([{"t0": 1.0}], [{"t0": 1.0}], ("t0",), {"target_alpha": 1.5}, "target_alpha"),
        ([{"t0": 1.0}], [{"t0": 1.0}], ("t0",), {"weights": [1.0, 1.0]}, "weights length"),
        (
            [{"t0": 1.0}, {"t0": 1.0}],
            [{"t0": 1.0}, {"t0": 1.0}],
            ("t0",),
            {"weights": [-2.0, 5.0]},
            "finite and non-negative",
        ),
        (
            [{"t0": 1.0}, {"t0": 1.0}],
            [{"t0": 1.0}, {"t0": 1.0}],
            ("t0",),
            {"weights": [float("nan"), 1.0]},
            "finite and non-negative",
        ),
        (
            [{"t0": 1.0}, {"t0": 1.0}],
            [{"t0": 1.0}, {"t0": 1.0}],
            ("t0",),
            {"weights": [float("inf"), 1.0]},
            "finite and non-negative",
        ),
        ([{"t0": 1.0}], [{"t0": 1.0}], ("t0",), {"weights": [0.0]}, "positive sum"),
        ([{"t0": 1.0}], [{"t0": 1.0}], ("t0",), {"grid_size": 0}, "grid_size must be positive"),
        ([{"t0": 1.0}], [{"t0": 1.0}], ("t0",), {"grid_size": -4}, "grid_size must be positive"),
        (
            [{"t0": 1.0}],
            [{"t0": 1.0}],
            ("t0",),
            {"upper_bound_b": 0.0},
            "upper_bound_b must be finite and positive",
        ),
        (
            [{"t0": 1.0}],
            [{"t0": 1.0}],
            ("t0",),
            {"upper_bound_b": -1.0},
            "upper_bound_b must be finite and positive",
        ),
        (
            [{"t0": 1.0}],
            [{"t0": 1.0}],
            ("t0",),
            {"upper_bound_b": float("nan")},
            "upper_bound_b must be finite and positive",
        ),
    ],
)
def test_calibrate_conformal_cdf_threshold_validation_errors(
    probs: list[dict[str, float]],
    accs: list[dict[str, float]],
    tiers: tuple[str, ...],
    kwargs: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        calibrate_conformal_cdf_threshold(probs, accs, tiers, **kwargs)


def test_select_tier_from_cdf_threshold_picks_first_reaching_mass() -> None:
    chosen, conf = select_tier_from_cdf_threshold(
        {"t0_flash_lite_min": 0.30, "t1_hydrogen_low": 0.45, "t2_hydrogen_med": 0.25},
        TIERS,
        0.70,
    )
    assert chosen == "t1_hydrogen_low"
    assert conf == pytest.approx(0.75)


def test_select_tier_from_cdf_threshold_fallback_and_empty_error() -> None:
    chosen, conf = select_tier_from_cdf_threshold(
        {"t0_flash_lite_min": 0.10, "t1_hydrogen_low": 0.10},
        TIERS,
        0.95,
    )
    assert chosen == TIERS[-1]
    assert conf == 1.0

    with pytest.raises(ValueError, match="ordered_tier_ids must not be empty"):
        select_tier_from_cdf_threshold({"t0_flash_lite_min": 1.0}, (), 0.5)


def test_decision_router_choice_crc_policy_routes_end_to_end() -> None:
    fake_resp = SimpleNamespace(
        answers={
            "target_tier": SimpleNamespace(
                probabilities={
                    "t0_flash_lite_min": 0.50,
                    "t1_hydrogen_low": 0.35,
                    "t2_hydrogen_med": 0.15,
                },
                choice="t0_flash_lite_min",
            )
        },
        usage_metadata=SimpleNamespace(prompt_token_count=100),
    )
    fake_client = SimpleNamespace(
        decisions=SimpleNamespace(create=lambda **_kw: fake_resp),
        interactions=SimpleNamespace(create=lambda **_kw: None),
    )
    router = DecisionRouter(
        config=RouterPolicyConfig(policy="choice_crc", conformal_cdf_threshold=0.80),
        client=fake_client,
        downstream_client=fake_client,
    )
    decision = router.route("Explain why CRC guarantees bounded risk.")
    assert decision.selected_tier_id == "t1_hydrogen_low"
    assert decision.policy_name == "choice_crc"


def test_decision_router_choice_crc_uses_ladder_conformal_cdf_threshold() -> None:
    import dataclasses

    from gemini_decision_router.ladder import default_ladder

    ladder = dataclasses.replace(default_ladder(), conformal_cdf_threshold=0.80)
    fake_resp = SimpleNamespace(
        answers={
            "target_tier": SimpleNamespace(
                probabilities={
                    "t0_flash_lite_min": 0.50,
                    "t1_hydrogen_low": 0.35,
                    "t2_hydrogen_med": 0.15,
                },
                choice="t0_flash_lite_min",
            )
        },
        usage_metadata=SimpleNamespace(prompt_token_count=100),
    )
    fake_client = SimpleNamespace(
        decisions=SimpleNamespace(create=lambda **_kw: fake_resp),
        interactions=SimpleNamespace(create=lambda **_kw: None),
    )
    router = DecisionRouter(
        config=RouterPolicyConfig(policy="choice_crc"),
        client=fake_client,
        downstream_client=fake_client,
        ladder=ladder,
    )
    decision = router.route("Explain why CRC guarantees bounded risk.")
    assert decision.selected_tier_id == "t1_hydrogen_low"
