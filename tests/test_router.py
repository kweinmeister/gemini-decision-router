"""Unit tests for DecisionRouter policies, PAVA isotonic projection, length-aware cost, and safety fallbacks."""

from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any, cast, get_args

import pytest

from gemini_decision_router.compute import calculate_compute_units
from gemini_decision_router.ladder import default_ladder
from gemini_decision_router.models import (
    pava_isotonic_increasing,
    validate_thinking_level,
)
from gemini_decision_router.observability import RouterObserver
from gemini_decision_router.router import (
    POLICY_FAMILIES,
    DecisionRouter,
    PolicyMode,
    RouterPolicyConfig,
    build_noul_solvability_questions,
    select_tier_from_choice_distribution,
    select_tier_from_noul_eu,
)
from gemini_decision_router.usage import Usage


def _decision_response(
    answers: Any,
    *,
    input_tokens: int = 200,
    prompt_feedback: Any = None,
) -> SimpleNamespace:
    """Build a minimal Decisions API response stub."""
    return SimpleNamespace(
        usage=SimpleNamespace(total_input_tokens=input_tokens),
        prompt_feedback=prompt_feedback,
        answers=answers,
    )


def _client_returning(resp: Any) -> Any:
    """A fake `genai.Client` whose `decisions.create` and `aio.decisions.create` return `resp`."""

    async def _acreate(**kwargs: Any) -> Any:
        return resp

    return SimpleNamespace(
        decisions=SimpleNamespace(create=lambda **kwargs: resp),
        aio=SimpleNamespace(decisions=SimpleNamespace(create=_acreate)),
    )


@pytest.mark.parametrize(
    ("model", "level"),
    [
        ("gemini-3.5-flash-lite", "minimal"),
        ("gemini-3.8-flash", "low"),
        ("gemini-4-hydrogen", "xhigh"),
        ("gemini-4-hydrogen", "low"),
    ],
)
def test_validate_thinking_level_allows_supported_levels(model: str, level: str) -> None:
    assert validate_thinking_level(model, level) == level


@pytest.mark.parametrize(
    "model_no_minimal",
    [
        "gemini-3.8-flash",
        "gemini-4-hydrogen",
        "gemini-4-argon",
        "gemini-4-carbon",
    ],
)
def test_validate_thinking_level_rejects_minimal_when_unsupported(
    model_no_minimal: str,
) -> None:
    with pytest.raises(ValueError, match="does not support thinking_level='minimal'"):
        validate_thinking_level(model_no_minimal, "minimal")


def test_pava_isotonic_increasing_monotonicity() -> None:
    raw = [0.80, 0.60, 0.70, 0.90, 0.85]
    proj = pava_isotonic_increasing(raw)
    assert len(proj) == 5
    for i in range(len(proj) - 1):
        assert proj[i] <= proj[i + 1] + 1e-12
    assert proj[:3] == pytest.approx([0.70, 0.70, 0.70])
    assert proj[3:] == pytest.approx([0.875, 0.875])


def test_pava_isotonic_increasing_properties() -> None:
    rng = random.Random(20261112)
    # Empty and single-element inputs
    assert pava_isotonic_increasing([]) == []
    assert pava_isotonic_increasing([0.42]) == [0.42]
    # Clamping out-of-range floats to [0.01, 0.99]
    assert pava_isotonic_increasing([-5.0, 10.0]) == [0.01, 0.99]

    for _ in range(50):
        length = rng.randint(2, 20)
        # Draw probabilities within the valid domain [0.01, 0.99]
        raw = [rng.uniform(0.01, 0.99) for _ in range(length)]
        proj = pava_isotonic_increasing(raw)

        # Property 1: Weakly monotonic increasing
        assert len(proj) == length
        for i in range(length - 1):
            assert proj[i] <= proj[i + 1] + 1e-11

        # Property 2: Idempotent (PAVA on monotonic sequence is identity)
        re_proj = pava_isotonic_increasing(proj)
        assert re_proj == pytest.approx(proj, abs=1e-11)

        # Property 3: Bounds preservation within [0.01, 0.99]
        assert min(proj) >= min(raw) - 1e-11
        assert max(proj) <= max(raw) + 1e-11

        # Property 4: Preserves already sorted sequences
        sorted_raw = sorted(raw)
        assert pava_isotonic_increasing(sorted_raw) == pytest.approx(sorted_raw, abs=1e-11)


def test_length_aware_compute_shifts_noul_eu_choice() -> None:
    raw_nouls = {
        "t0_flash_lite_min": 0.70,
        "t1_hydrogen_low": 0.74,
        "t2_hydrogen_med": 0.74,
        "t3_hydrogen_high": 0.75,
        "t4_hydrogen_xhigh": 0.76,
    }
    cfg = RouterPolicyConfig(policy="noul_eu", eu_lambda=4.0)

    # Short prompt (100 input tokens): t1_hydrogen_low (+0.04 solvability) beats t0,
    # while t4_hydrogen_xhigh's high output/thought tokens are too heavy for a +0.02 gain.
    tid_short, proj_short, conf_short = select_tier_from_noul_eu(
        raw_nouls, known_input_tokens=100, config=cfg
    )
    assert tid_short == "t1_hydrogen_low"
    assert conf_short == proj_short[tid_short] == pytest.approx(0.74)

    # Long prompt (40,000 input tokens): input compute weight (1.0 vs 0.4) penalizes Hydrogen tiers
    tid_long, proj_long, conf_long = select_tier_from_noul_eu(
        raw_nouls, known_input_tokens=40_000, config=cfg
    )
    assert tid_long == "t0_flash_lite_min"
    assert conf_long == proj_long[tid_long] == pytest.approx(0.70)


def test_noul_questions_express_cumulative_capability() -> None:
    qs = build_noul_solvability_questions()
    assert list(qs) == [
        "can_t0_flash_lite_min_solve",
        "can_t1_hydrogen_low_solve",
        "can_t2_hydrogen_med_solve",
        "can_t3_hydrogen_high_solve",
        "can_t4_hydrogen_xhigh_solve",
    ]
    # Rank 0 uses its base description directly; higher ranks frame cumulative subsumption
    assert (
        "handles all simpler tasks"
        not in qs["can_t0_flash_lite_min_solve"]["criteria"]["true_description"]
    )
    for name in (
        "can_t1_hydrogen_low_solve",
        "can_t2_hydrogen_med_solve",
        "can_t3_hydrogen_high_solve",
        "can_t4_hydrogen_xhigh_solve",
    ):
        assert (
            "handles all simpler tasks from lower tiers" in qs[name]["criteria"]["true_description"]
        )
        assert "strictly subsume all lower tiers" in qs[name]["instructions"]


def test_compute_units_scales_across_models() -> None:
    usage = Usage(input=100_000, output=10_000)
    flash_units = calculate_compute_units("gemini-4-hydrogen", usage)
    argon_units = calculate_compute_units("gemini-4-argon", usage)
    assert argon_units > flash_units

    dec_units = calculate_compute_units("gemini-decision-1", Usage(input=1_000_000))
    assert dec_units == pytest.approx(0.067)


def test_thinking_tokens_weighted_at_output_rate() -> None:
    # gemini-3.8-flash compute_weight: input=1.0, output=5.0 (thinking weighted as output)
    usage = Usage(input=1_000, output=100, thought=900)
    assert usage.billed_output == 1_000
    units = calculate_compute_units("gemini-3.8-flash", usage)
    assert units == pytest.approx((1_000 * 1.0 + 1_000 * 5.0) / 1e6)
    no_thought = calculate_compute_units("gemini-3.8-flash", Usage(input=1_000, output=100))
    assert units - no_thought == pytest.approx(900 * 5.0 / 1e6)


def test_usage_from_response_reads_thought_tokens_and_defaults_missing_to_zero() -> None:
    resp = SimpleNamespace(
        usage=SimpleNamespace(
            total_input_tokens=1_200,
            total_output_tokens=300,
            total_thought_tokens=4_500,
            input_tokens_by_modality=[
                SimpleNamespace(modality="text", tokens=200),
                SimpleNamespace(modality="image", tokens=1_000),
            ],
        )
    )
    u = Usage.from_response(resp)
    assert (u.input, u.output, u.thought, u.billed_output) == (1_200, 300, 4_500, 4_800)
    assert dict(u.by_modality) == {"text": 200, "image": 1_000}

    # Decision responses report no output/thought tokens; absent fields (or usage) count as 0
    assert Usage.from_response(
        SimpleNamespace(usage=SimpleNamespace(total_input_tokens=7))
    ) == Usage(input=7)
    assert Usage.from_response(SimpleNamespace(usage=None)) == Usage()
    assert Usage.from_response(SimpleNamespace()) == Usage()


@pytest.fixture
def score_mock_response() -> SimpleNamespace:
    return _decision_response(
        [
            SimpleNamespace(
                name="complexity_tier",
                type="score",
                score=1.80,
                probabilities={"0": 0.40, "1": 0.15, "2": 0.05, "3": 0.40},
                confidence=0.40,
            ),
            SimpleNamespace(name="needs_precision", type="noul", noul=0.80, confidence=0.80),
        ]
    )


def test_router_score_tail_and_argmax_mock_client(
    score_mock_response: SimpleNamespace,
) -> None:
    fake_client = _client_returning(score_mock_response)
    decision_weight = default_ladder().compute_weight("gemini-decision-1").input

    # Argmax picks index 0 ("0": 0.40) -> t0_flash_lite_min
    r_argmax = DecisionRouter(config=RouterPolicyConfig(policy="score_argmax"), client=fake_client)
    d_argmax = r_argmax.route("Solve this puzzle.")
    assert d_argmax.selected_tier_id == "t0_flash_lite_min"
    # confidence = P(selected tier suffices) = mass at or below the selected level
    assert d_argmax.confidence == pytest.approx(0.40)
    assert d_argmax.router_compute_units == pytest.approx(200 * decision_weight / 1e6)

    # Tail policy sees frontier tail mass on {t3_hydrogen_high, t4_hydrogen_xhigh} (0.40 >= 0.35) -> escalates to t3_hydrogen_high
    r_tail = DecisionRouter(config=RouterPolicyConfig(policy="score_tail"), client=fake_client)
    d_tail = r_tail.route("Solve this puzzle.")
    assert d_tail.selected_tier_id == "t3_hydrogen_high"
    assert d_tail.confidence == pytest.approx(1.0)
    assert d_tail.needs_precision_noul == pytest.approx(0.80)


def test_score_tail_precision_floor_bumps_rank0_without_frontier_tail() -> None:
    # Expected score = 0.25 (< cutoff 0.75 -> t0) and frontier tail mass = 0.05 (< 0.35),
    # so only needs_precision (0.80 >= 0.65) triggers the bump to precision_floor_tier (t1).
    resp = _decision_response(
        [
            SimpleNamespace(
                name="complexity_tier",
                type="score",
                score=0.25,
                probabilities={"0": 0.80, "1": 0.10, "2": 0.05, "3": 0.05, "4": 0.00},
                confidence=0.80,
            ),
            SimpleNamespace(name="needs_precision", type="noul", noul=0.80, confidence=0.80),
        ],
        input_tokens=150,
    )
    router = DecisionRouter(
        config=RouterPolicyConfig(policy="score_tail"), client=_client_returning(resp)
    )
    dec = router.route("Compute exact checksum.")
    assert dec.selected_tier_id == "t1_hydrogen_low"
    assert dec.confidence == pytest.approx(0.90)
    assert dec.needs_precision_noul == pytest.approx(0.80)


_CHOICE_PROBS = {
    "t0_flash_lite_min": 0.2,
    "t1_hydrogen_low": 0.3,
    "t2_hydrogen_med": 0.3,
    "t3_hydrogen_high": 0.1,
    "t4_hydrogen_xhigh": 0.1,
}
_BIMODAL_CHOICE_PROBS = {
    "t0_flash_lite_min": 0.55,
    "t1_hydrogen_low": 0.05,
    "t2_hydrogen_med": 0.00,
    "t3_hydrogen_high": 0.20,
    "t4_hydrogen_xhigh": 0.20,
}


@pytest.mark.parametrize(
    ("probs", "config", "raw_choice", "expected_tier", "expected_conf"),
    [
        (
            _CHOICE_PROBS,
            RouterPolicyConfig(policy="choice_tail", choice_cdf_threshold=0.7),
            None,
            "t2_hydrogen_med",
            0.8,
        ),
        (
            _CHOICE_PROBS,
            RouterPolicyConfig(policy="choice_argmax"),
            None,
            "t1_hydrogen_low",
            0.5,
        ),
        (
            _CHOICE_PROBS,
            RouterPolicyConfig(policy="choice_argmax"),
            "t2_hydrogen_med",
            "t2_hydrogen_med",
            0.8,
        ),
        (
            _BIMODAL_CHOICE_PROBS,
            RouterPolicyConfig(
                policy="choice_tail",
                score_cutoffs=(0.50, 1.50, 2.50, 3.50),
                score_frontier_tail_threshold=0.35,
            ),
            None,
            "t3_hydrogen_high",
            0.8,
        ),
    ],
    ids=[
        "choice-tail-cdf",
        "choice-argmax-modal",
        "choice-argmax-raw-tiebreak",
        "choice-tail-score-cutoffs",
    ],
)
def test_choice_policies_report_cumulative_confidence(
    probs: dict[str, float],
    config: RouterPolicyConfig,
    raw_choice: str | None,
    expected_tier: str,
    expected_conf: float,
) -> None:
    tid, conf = select_tier_from_choice_distribution(probs, config, raw_choice=raw_choice)
    assert (tid, conf) == (expected_tier, pytest.approx(expected_conf))


def test_router_choice_tail_end_to_end() -> None:
    resp = _decision_response(
        [
            SimpleNamespace(
                name="target_tier",
                type="choice",
                choice="t0_flash_lite_min",
                probabilities={
                    "t0_flash_lite_min": 0.45,
                    "t1_hydrogen_low": 0.30,
                    "t2_hydrogen_med": 0.15,
                    "t3_hydrogen_high": 0.05,
                    "t4_hydrogen_xhigh": 0.05,
                },
                confidence=0.45,
            )
        ],
        input_tokens=120,
    )
    router = DecisionRouter(
        config=RouterPolicyConfig(policy="choice_tail", choice_cdf_threshold=0.70),
        client=_client_returning(resp),
    )
    dec = router.route("Refactor this helper.")
    assert dec.selected_tier_id == "t1_hydrogen_low"
    assert dec.confidence == pytest.approx(0.75)


def test_empirical_tier_order_governs_pava_and_validates_permutation() -> None:
    with pytest.raises(ValueError, match="must be a permutation"):
        RouterPolicyConfig(tier_order=("t0_flash_lite_min", "t1_hydrogen_low"))

    # Suppose on a training fold t2_hydrogen_med has lower realized cost than t1_hydrogen_low
    permuted = (
        "t0_flash_lite_min",
        "t2_hydrogen_med",
        "t1_hydrogen_low",
        "t3_hydrogen_high",
        "t4_hydrogen_xhigh",
    )
    raw = {
        "t0_flash_lite_min": 0.40,
        "t2_hydrogen_med": 0.80,
        "t1_hydrogen_low": 0.60,
        "t3_hydrogen_high": 0.90,
        "t4_hydrogen_xhigh": 0.95,
    }
    cfg = RouterPolicyConfig(policy="noul_eu", eu_lambda=2.0, tier_order=permuted)
    _, proj, _ = select_tier_from_noul_eu(raw, known_input_tokens=100, config=cfg)
    # PAVA enforces monotonicity along `permuted`: t0 <= t2 <= t1 <= t3 <= t4
    assert proj["t2_hydrogen_med"] == pytest.approx(0.70)
    assert proj["t1_hydrogen_low"] == pytest.approx(0.70)


def test_router_records_exact_api_input_tokens_on_uncompressed_payload() -> None:
    mock_resp = _decision_response(
        [
            SimpleNamespace(
                name="target_tier",
                type="choice",
                choice="t2_hydrogen_med",
                probabilities={
                    "t0_flash_lite_min": 0.1,
                    "t1_hydrogen_low": 0.4,
                    "t2_hydrogen_med": 0.4,
                    "t3_hydrogen_high": 0.05,
                    "t4_hydrogen_xhigh": 0.05,
                },
            )
        ],
        input_tokens=412,
    )
    obs = RouterObserver()
    router = DecisionRouter(
        config=RouterPolicyConfig(policy="choice_argmax"),
        client=_client_returning(mock_resp),
        observer=obs,
    )
    dec = router.route("Uncompressed text prompt")
    assert dec.selected_tier_id == "t2_hydrogen_med"
    assert obs.observations[0].known_input_tokens == 412


def test_every_policy_mode_has_a_family_and_unknown_policy_fails_fast() -> None:
    assert set(POLICY_FAMILIES) == set(get_args(PolicyMode))

    def _must_not_be_called(**kwargs: object) -> None:
        raise AssertionError("client must not be used for an invalid config")

    client = SimpleNamespace(decisions=SimpleNamespace(create=_must_not_be_called))
    with pytest.raises(ValueError, match="Unknown router policy: 'bogus'"):
        DecisionRouter(config=RouterPolicyConfig(policy=cast(Any, "bogus")), client=client)


def test_router_safety_fallback_on_http_error() -> None:
    def _raise_safety(**kwargs: object) -> None:
        raise RuntimeError("400 Bad Request: prohibited_content block triggered")

    client_err = SimpleNamespace(decisions=SimpleNamespace(create=_raise_safety))
    dec = DecisionRouter(client=client_err).route("Sensitive prompt")
    assert dec.fallback_triggered is True
    assert dec.selected_tier_id == "t2_hydrogen_med"
    assert "prohibited_content" in (dec.fallback_reason or "")
    assert dec.confidence == 0.0
    assert dec.router_compute_units == 0.0


def test_router_safety_fallback_on_inband_block() -> None:
    decision_weight = default_ladder().compute_weight("gemini-decision-1").input
    inband_resp = _decision_response(
        [],
        input_tokens=120,
        prompt_feedback=SimpleNamespace(block_reason="SAFETY"),
    )
    dec = DecisionRouter(client=_client_returning(inband_resp)).route("In-band blocked prompt")
    assert dec.fallback_triggered is True
    assert dec.selected_tier_id == "t2_hydrogen_med"
    assert dec.fallback_reason == "in_band_block:SAFETY"
    # The blocked decision call still records its input compute units
    assert dec.router_compute_units == pytest.approx(120 * decision_weight / 1e6)


@pytest.mark.parametrize(
    ("policy", "answers", "expected_reason"),
    [
        ("score_tail", [], "empty_answers"),
        (
            "score_tail",
            [SimpleNamespace(name="needs_precision", type="noul", noul=0.9)],
            "missing_complexity_tier",
        ),
        (
            "choice_tail",
            [SimpleNamespace(name="needs_precision", type="noul", noul=0.9)],
            "missing_target_tier",
        ),
    ],
    ids=["empty-answers", "missing-complexity-tier", "missing-target-tier"],
)
def test_router_falls_back_on_empty_or_incomplete_answers(
    policy: PolicyMode, answers: list[Any], expected_reason: str
) -> None:
    resp = _decision_response(answers, input_tokens=50)
    dec = DecisionRouter(
        config=RouterPolicyConfig(policy=policy),
        client=_client_returning(resp),
    ).route("hi")
    assert (dec.fallback_triggered, dec.fallback_reason) == (True, expected_reason)
    assert dec.selected_tier_id == "t2_hydrogen_med"


def test_missing_credentials_raises_environment_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_ENTERPRISE", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    with pytest.raises(EnvironmentError, match="Missing credentials"):
        DecisionRouter()


def test_google_api_key_fallback_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_ENTERPRISE", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    router = DecisionRouter()
    assert router.client is not None


def test_gemini_api_key_takes_precedence_over_google_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    router = DecisionRouter()
    assert router.client is not None


def test_enterprise_mode_defaults_to_global_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-enterprise-proj")
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    router = DecisionRouter()
    assert router.client is not None


def test_router_observer_tracks_savings_and_calibration() -> None:
    mock_resp = _decision_response(
        {
            "complexity_tier": SimpleNamespace(
                type="score",
                score=0.20,
                probabilities={"0": 0.85, "1": 0.10, "2": 0.03, "3": 0.02},
                confidence=0.85,
            ),
            "needs_precision": SimpleNamespace(type="noul", noul=0.05, confidence=0.95),
        },
        input_tokens=150,
    )
    observer = RouterObserver(baseline_tier_id="t4_hydrogen_xhigh")
    router = DecisionRouter(client=_client_returning(mock_resp), observer=observer)

    dec = router.route("Format this date as YYYY-MM-DD.")
    assert dec.selected_tier_id == "t0_flash_lite_min"
    assert dec.confidence == pytest.approx(0.85)
    # Record the downstream outcome; this updates the observation route() already recorded
    usage = Usage(input=45, output=110, thought=64)
    obs = observer.record_outcome(dec, success=True, usage=usage)
    assert len(observer.observations) == 1
    assert obs.success is True
    # Thinking tokens are weighted at output rate
    expected_downstream = calculate_compute_units("gemini-3.5-flash-lite", usage)
    assert obs.realized_downstream_compute_units == pytest.approx(expected_downstream)

    rep = observer.summary()
    assert rep.total_requests == 1
    assert rep.tier_counts["t0_flash_lite_min"] == 1
    assert rep.total_estimated_compute_units == pytest.approx(
        dec.router_compute_units + expected_downstream, abs=1e-6
    )
    assert rep.estimated_compute_savings_pct > 50.0
    assert rep.calibration["count"] == 1.0
    assert rep.calibration["brier_score"] == pytest.approx((1.0 - 0.85) ** 2)

    # Fallback decisions are counted as requests but excluded from calibration
    blocked_resp = _decision_response(
        [],
        input_tokens=150,
        prompt_feedback=SimpleNamespace(block_reason="SAFETY"),
    )
    blocked_router = DecisionRouter(client=_client_returning(blocked_resp), observer=observer)
    fb = blocked_router.route("Blocked prompt")
    observer.record_outcome(fb, success=False)
    rep = observer.summary()
    assert rep.total_requests == 2
    assert rep.fallback_rate == pytest.approx(0.5)
    assert rep.calibration["count"] == 1.0

    # A decision the observer never saw is appended once, with the caller's input length
    unseen = DecisionRouter(client=_client_returning(mock_resp)).route("Another prompt")
    obs_unseen = observer.record_outcome(unseen, success=True, known_input_tokens=500)
    assert obs_unseen.known_input_tokens == 500
    assert observer.summary().total_requests == 3


@pytest.mark.anyio
async def test_router_async_aroute(score_mock_response: SimpleNamespace) -> None:
    fake_client = _client_returning(score_mock_response)
    decision_weight = default_ladder().compute_weight("gemini-decision-1").input
    router = DecisionRouter(config=RouterPolicyConfig(policy="score_tail"), client=fake_client)

    dec = await router.aroute("Solve this puzzle.")
    assert dec.selected_tier_id == "t3_hydrogen_high"
    assert dec.confidence == pytest.approx(1.0)
    assert dec.needs_precision_noul == pytest.approx(0.80)
    assert dec.router_compute_units == pytest.approx(200 * decision_weight / 1e6)


def test_route_and_execute_compresses_router_only_and_forwards_kwargs() -> None:
    router_inputs: list[Any] = []
    downstream_calls: list[dict[str, Any]] = []

    mock_resp = _decision_response(
        [
            SimpleNamespace(
                name="complexity_tier",
                type="score",
                score=1.0,
                probabilities={"0": 0.10, "1": 0.70, "2": 0.15, "3": 0.05},
                confidence=0.70,
            ),
            SimpleNamespace(name="needs_precision", type="noul", noul=0.10, confidence=0.10),
        ],
        input_tokens=150,
    )

    def _create_decision(**kwargs: Any) -> Any:
        router_inputs.append(kwargs["input"])
        return mock_resp

    def _create_interaction(**kwargs: Any) -> Any:
        downstream_calls.append(kwargs)
        return SimpleNamespace(output_text="ok")

    fake_client = SimpleNamespace(
        decisions=SimpleNamespace(create=_create_decision),
        interactions=SimpleNamespace(create=_create_interaction),
    )
    router = DecisionRouter(config=RouterPolicyConfig(policy="score_tail"), client=fake_client)

    payload = [
        {"type": "image", "mime_type": "image/png", "data": "AAAA"},
        {"type": "text", "text": "Describe this image."},
    ]
    decision, interaction = router.route_and_execute(
        payload,
        system_instruction="Be concise.",
        safety_settings=[{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}],
        generation_config={"temperature": 0.2},
    )

    assert interaction.output_text == "ok"
    assert decision.selected_tier_id == "t1_hydrogen_low"
    # Router received compressed low-res image; downstream received original untouched payload
    assert router_inputs[0][0]["resolution"] == "low"
    assert "resolution" not in payload[0]
    assert downstream_calls[0]["input"] is payload
    assert downstream_calls[0]["model"] == decision.model
    assert downstream_calls[0]["generation_config"] == {
        "thinking_level": decision.thinking_level,
        "temperature": 0.2,
    }
    assert downstream_calls[0]["system_instruction"] == "Be concise."
    assert len(downstream_calls[0]["safety_settings"]) == 1


@pytest.mark.anyio
async def test_route_and_aroute_reraise_non_safety_errors() -> None:
    def _boom(**_kwargs: Any) -> Any:
        raise RuntimeError("transient network failure")

    async def _aboom(**_kwargs: Any) -> Any:
        raise RuntimeError("transient async failure")

    fake_client = SimpleNamespace(
        decisions=SimpleNamespace(create=_boom),
        aio=SimpleNamespace(decisions=SimpleNamespace(create=_aboom)),
    )
    router = DecisionRouter(client=fake_client)

    with pytest.raises(RuntimeError, match="transient network failure"):
        router.route("hello")

    with pytest.raises(RuntimeError, match="transient async failure"):
        await router.aroute("hello")


def test_is_safety_api_error_detects_http_400_safety_and_ignores_generic() -> None:
    from google.genai import errors as genai_errors

    from gemini_decision_router.router import _is_safety_api_error

    api_err = genai_errors.APIError(400, {"error": {"message": "Request was blocked"}})
    assert _is_safety_api_error(api_err) == (True, "http_400_safety")
    assert _is_safety_api_error(ValueError("bad input")) == (False, "")


def test_noul_eu_missing_probes_default_to_half() -> None:
    mock_resp = _decision_response(
        [
            SimpleNamespace(name="can_t0_flash_lite_min_solve", type="noul", noul=0.95),
            # Remaining per-tier probes omitted -> default to 0.5; PAVA pools [0.95, 0.5, 0.5, 0.5, 0.5] -> 0.59
        ],
        input_tokens=100,
    )
    router = DecisionRouter(
        config=RouterPolicyConfig(policy="noul_eu"),
        client=_client_returning(mock_resp),
    )
    dec = router.route("Simple question")
    assert dec.selected_tier_id == "t0_flash_lite_min"
    assert dec.raw_probabilities is not None
    assert dec.raw_probabilities["t1_hydrogen_low"] == pytest.approx(0.5)
    assert dec.solvability_probabilities is not None
    assert dec.solvability_probabilities["t1_hydrogen_low"] == pytest.approx(0.59)


def test_decision_router_rejects_conflicting_ladders() -> None:
    from gemini_decision_router.ladder import TierLadder, default_ladder

    lad1 = default_ladder()
    # Distinct ladder instances passed in both config and constructor must raise.
    with pytest.raises(ValueError, match="not both"):
        DecisionRouter(
            config=RouterPolicyConfig(ladder=lad1),
            ladder=TierLadder.from_dict(
                {
                    "models": {
                        "gemini-decision-1": {"compute_weight": {"input": 0.1, "output": 0.0}},
                        "m": {
                            "thinking_levels": ["low"],
                            "compute_weight": {"input": 0.2, "output": 0.4},
                        },
                    },
                    "tiers": [
                        {
                            "id": "only",
                            "model": "m",
                            "thinking_level": "low",
                            "can_solve": "all",
                            "cannot_solve": "none",
                            "expected_output_tokens": {"default": 10},
                        }
                    ],
                }
            ),
            client=_client_returning(SimpleNamespace(usage=None, prompt_feedback=None, answers={})),
        )


def test_router_observer_max_observations() -> None:
    from gemini_decision_router.models import RouteDecision

    with pytest.raises(ValueError, match="positive"):
        RouterObserver(max_observations=0)

    observer = RouterObserver(max_observations=2)
    for idx in range(3):
        dec = RouteDecision(
            selected_tier_id="t0_flash_lite_min",
            model="gemini-3.5-flash-lite",
            thinking_level="minimal",
            policy_name="score_tail",
            confidence=0.9,
            router_latency_ms=10.0 + idx,
            router_input_tokens=50,
            router_compute_units=0.0001,
            estimated_downstream_compute_units=0.0002,
        )
        observer.record_decision(dec, known_input_tokens=50 + idx)

    assert len(observer.observations) == 2
    assert observer.observations[0].known_input_tokens == 51
    assert observer.observations[1].known_input_tokens == 52


def test_vertexai_env_var_and_hybrid_downstream_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_ENTERPRISE", raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-vertex-proj")
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)

    vertex_router = DecisionRouter()
    assert vertex_router.client is not None
    assert vertex_router.downstream_client is vertex_router.client

    # Hybrid mode: GEMINI_API_KEY for routing + GOOGLE_GENAI_USE_VERTEXAI for downstream
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    hybrid_router = DecisionRouter()
    assert hybrid_router.client is not None
    assert hybrid_router.downstream_client is not hybrid_router.client
