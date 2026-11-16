"""Tests for pure 2-stage cascades and Hybrid Cascade-Routing."""

from types import SimpleNamespace
from typing import Any

import pytest

from gemini_decision_router import (
    CascadeConfig,
    DecisionRouter,
    HybridCascadeRouter,
    RouterPolicyConfig,
    build_verification_questions,
)


class _FakeCascadeClient:
    def __init__(self, prefill_probs: dict[str, float], verifier_prob: float) -> None:
        self.prefill_probs = prefill_probs
        self.verifier_prob = verifier_prob
        self.decision_calls: list[dict[str, Any]] = []
        self.interaction_calls: list[dict[str, Any]] = []
        self.decisions = SimpleNamespace(create=self._create_decision)
        self.interactions = SimpleNamespace(create=self._create_interaction)

    def _create_decision(self, **kwargs: Any) -> Any:
        self.decision_calls.append(kwargs)
        q_names = set(kwargs["questions"].keys())
        if "target_tier" in q_names:
            return SimpleNamespace(
                answers={
                    "target_tier": SimpleNamespace(
                        probabilities=self.prefill_probs,
                        choice="t0_flash_lite_min",
                    )
                },
                usage_metadata=SimpleNamespace(prompt_token_count=80),
            )
        assert "draft_solves_request" in q_names
        return SimpleNamespace(
            answers={"draft_solves_request": SimpleNamespace(noul=self.verifier_prob)},
            usage_metadata=SimpleNamespace(prompt_token_count=140),
        )

    def _create_interaction(self, **kwargs: Any) -> Any:
        self.interaction_calls.append(kwargs)
        return SimpleNamespace(
            output_text=f"response from {kwargs['model']}",
            usage=SimpleNamespace(
                total_input_tokens=100,
                total_output_tokens=200,
                total_thought_tokens=50,
            ),
        )


def test_build_verification_questions_includes_draft_solves_request() -> None:
    assert "draft_solves_request" in build_verification_questions()


def test_hybrid_cascade_prefill_skips_cheap_and_verifier_on_hard_prompt() -> None:
    client = _FakeCascadeClient(
        prefill_probs={"t0_flash_lite_min": 0.1, "t1_hydrogen_low": 0.2, "t2_hydrogen_med": 0.7},
        verifier_prob=0.1,
    )
    router = DecisionRouter(
        config=RouterPolicyConfig(policy="choice_tail", choice_cdf_threshold=0.70),
        client=client,
        downstream_client=client,
    )
    res = HybridCascadeRouter(router).run("Hard debugging prompt")
    assert res.prefill_skipped_cheap is True
    assert res.escalated is False
    assert res.final_tier_id == "t2_hydrogen_med"
    assert len(client.interaction_calls) == 1
    assert len(client.decision_calls) == 1


def test_hybrid_cascade_prefill_fast_accepts_high_confidence_cheap_prompt() -> None:
    client = _FakeCascadeClient(
        prefill_probs={"t0_flash_lite_min": 0.92, "t1_hydrogen_low": 0.08},
        verifier_prob=0.10,
    )
    router = DecisionRouter(
        config=RouterPolicyConfig(policy="choice_tail", choice_cdf_threshold=0.70),
        client=client,
        downstream_client=client,
    )
    res = HybridCascadeRouter(
        router,
        CascadeConfig(use_prefill_skip=True, prefill_accept_threshold=0.80, accept_threshold=0.70),
    ).run("Easy factual prompt")
    assert res.prefill_skipped_cheap is False
    assert res.escalated is False
    assert res.verifier_confidence is None
    assert res.final_tier_id == "t0_flash_lite_min"
    assert len(client.interaction_calls) == 1
    assert len(client.decision_calls) == 1


def test_hybrid_cascade_verifies_borderline_draft_and_escalates_on_low_confidence() -> None:
    client = _FakeCascadeClient(
        prefill_probs={"t0_flash_lite_min": 0.70, "t1_hydrogen_low": 0.30},
        verifier_prob=0.30,
    )
    router = DecisionRouter(
        config=RouterPolicyConfig(policy="choice_tail", choice_cdf_threshold=0.70),
        client=client,
        downstream_client=client,
    )
    res = HybridCascadeRouter(
        router,
        CascadeConfig(use_prefill_skip=True, prefill_accept_threshold=0.80, accept_threshold=0.70),
    ).run("Borderline prompt")
    assert res.prefill_skipped_cheap is False
    assert res.escalated is True
    assert res.final_tier_id == "t1_hydrogen_low"
    assert res.verifier_confidence == pytest.approx(0.30)
    assert len(client.interaction_calls) == 2
    assert len(client.decision_calls) == 2


@pytest.mark.parametrize(
    ("verifier_prob", "expected_escalated", "expected_tier", "expected_interactions"),
    [
        (0.92, False, "t0_flash_lite_min", 1),
        (0.25, True, "t1_hydrogen_low", 2),
    ],
)
def test_pure_two_stage_cascade_accepts_or_escalates_by_verifier_confidence(
    verifier_prob: float,
    expected_escalated: bool,
    expected_tier: str,
    expected_interactions: int,
) -> None:
    client = _FakeCascadeClient(prefill_probs={}, verifier_prob=verifier_prob)
    router = DecisionRouter(
        config=RouterPolicyConfig(policy="choice_tail"),
        client=client,
        downstream_client=client,
    )
    res = HybridCascadeRouter(
        router, CascadeConfig(use_prefill_skip=False, accept_threshold=0.70)
    ).run("Lookup prompt")
    assert res.escalated is expected_escalated
    assert res.final_tier_id == expected_tier
    assert res.verifier_confidence == pytest.approx(verifier_prob)
    assert len(client.interaction_calls) == expected_interactions
    assert len(client.decision_calls) == 1
