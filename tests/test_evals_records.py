"""Tests for the evals' pure record builders: compute_units must weight thinking tokens."""

from __future__ import annotations

import asyncio
import itertools
import json
from types import SimpleNamespace
from typing import Any

import pytest

from evals.collect_matrix import build_matrix_record
from evals.routers import (
    _eval_flash_lite_json,
    _serialize_decision_resp,
    flash_lite_router_schema,
    flash_lite_router_system_prompt,
)
from gemini_decision_router.compute import calculate_compute_units
from gemini_decision_router.ladder import TierLadder, default_ladder
from gemini_decision_router.usage import Usage


def _tier(tid: str, level: str) -> dict[str, Any]:
    return {
        "id": tid,
        "model": "m",
        "thinking_level": level,
        "can_solve": "x",
        "cannot_solve": "y",
        "expected_output_tokens": {"default": 10},
    }


def _custom_ladder() -> TierLadder:
    """Two tiers with custom compute weights."""
    return TierLadder.from_dict(
        {
            "router": {},
            "models": {
                "gemini-decision-1": {
                    "compute_weight": {"input": 0.1, "output": 0.0},
                },
                "m": {
                    "thinking_levels": ["low", "high"],
                    "compute_weight": {"input": 1.0, "output": 2.0},
                },
            },
            "tiers": [_tier("a", "low"), _tier("b", "high")],
        }
    )


def test_calculate_compute_units_weights_thinking_tokens() -> None:
    usage = Usage(input=1_000, output=200, thought=800)
    lad = _custom_ladder()
    assert calculate_compute_units("m", usage, ladder=lad) == pytest.approx(
        (1_000 * 1.0 + 1_000 * 2.0) / 1e6
    )

    bundled = calculate_compute_units("gemini-3.8-flash", usage, ladder=default_ladder())
    assert bundled == pytest.approx((1_000 * 1.0 + 1_000 * 5.0) / 1e6)


def test_build_matrix_record_weights_thinking() -> None:
    lad = _custom_ladder()
    usage = Usage(input=1_000, output=50, thought=950, by_modality={"text": 1_000})
    rec = build_matrix_record(
        {"id": "s3_001", "slice": "s3", "modality": "text"},
        lad.tier("b"),
        2,
        response_text="   ",
        usage=usage,
        latency_ms=12.3456,
        attempts=1,
        passed=False,
        grade_reason="no_final_answer",
        interaction_status="incomplete",
        ladder=lad,
    )
    assert (rec["tier_id"], rec["model"], rec["thinking_level"], rec["replicate"]) == (
        "b",
        "m",
        "high",
        2,
    )
    assert (rec["input_tokens"], rec["output_tokens"], rec["thought_tokens"]) == (1_000, 50, 950)
    weight = lad.compute_weight("m")
    expected = (1_000 * weight.input + (50 + 950) * weight.output) / 1e6
    assert rec["compute_units"] == pytest.approx(expected)
    assert rec["latency_ms"] == 12.35
    assert rec["empty_response"] is True
    assert rec["input_tokens_by_modality"] == {"text": 1_000}
    json.dumps(rec)  # rows are written as JSONL


def test_serialize_decision_resp_records_compute_units_and_answers() -> None:
    lad = default_ladder()
    resp = SimpleNamespace(
        usage=SimpleNamespace(total_input_tokens=2_000),
        answers=[
            SimpleNamespace(
                name="complexity_tier",
                type="score",
                score=1.2,
                probabilities={0: 0.5, 1: 0.5},
                confidence=0.5,
            ),
            SimpleNamespace(name="needs_precision", type="noul", noul=0.3),
        ],
    )
    out = _serialize_decision_resp(resp, 12.3, ladder=lad)
    assert out["compute_units"] == pytest.approx(2_000 * 0.067 / 1e6)
    assert out["answers"] == {
        "complexity_tier": {
            "type": "score",
            "score": 1.2,
            "probabilities": {"0": 0.5, "1": 0.5},
            "confidence": 0.5,
        },
        "needs_precision": {"type": "noul", "noul": 0.3},
    }
    assert (out["latency_ms"], out["attempts"], out["input_tokens"]) == (12.3, 1, 2_000)


def test_flash_lite_router_schema_and_prompt_follow_the_ladder() -> None:
    lad = _custom_ladder()
    schema = flash_lite_router_schema(lad)
    assert schema["properties"]["choice_tier"]["enum"] == ["a", "b"]
    for key in ("tier_probabilities", "solvability_probabilities"):
        assert schema["properties"][key]["required"] == ["a", "b"]
    assert "to 1.0 (frontier)" in schema["properties"]["complexity_score"]["description"]
    prompt = flash_lite_router_system_prompt(lad)
    assert "- a (rank 0)" in prompt and "- b (rank 1)" in prompt


def _fake_interactions_client(text: str, captured: dict[str, Any]) -> Any:
    interaction = SimpleNamespace(
        output_text=text,
        usage=SimpleNamespace(
            total_input_tokens=500, total_output_tokens=100, total_thought_tokens=400
        ),
    )

    async def _create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return interaction

    return SimpleNamespace(aio=SimpleNamespace(interactions=SimpleNamespace(create=_create)))


def test_eval_flash_lite_json_bills_thinking_and_normalizes_output() -> None:
    lad = default_ladder()
    ids = lad.ordered_tier_ids
    body = {
        "choice_tier": ids[1],
        "complexity_score": 1.0,
        "needs_precision": False,
        "tier_probabilities": {ids[0]: 2.0, ids[1]: 2.0, ids[2]: 0.0, ids[3]: 0.0},
        "solvability_probabilities": {ids[0]: 0.9, ids[1]: 0.5, ids[2]: 0.7, ids[3]: 1.5},
    }
    captured: dict[str, Any] = {}
    out = asyncio.run(
        _eval_flash_lite_json(
            _fake_interactions_client(json.dumps(body), captured), "hi", ladder=lad
        )
    )

    # The router baseline runs on the cheapest tier with a schema over exactly the ladder's tiers
    assert (captured["model"], captured["generation_config"]) == (
        lad.tiers[0].model,
        {"thinking_level": lad.tiers[0].thinking_level},
    )
    assert captured["response_format"]["properties"]["choice_tier"]["enum"] == list(ids)
    # flash-lite weights: 0.2/1M input, 1.67/1M output; thinking weighted as output
    assert out["compute_units"] == pytest.approx((500 * 0.2 + (100 + 400) * 1.67) / 1e6)
    assert (out["output_tokens"], out["thought_tokens"]) == (100, 400)

    parsed = out["parsed"]
    assert list(parsed["tier_probabilities"]) == list(ids)
    assert sum(parsed["tier_probabilities"].values()) == pytest.approx(1.0)
    assert parsed["solvability_probabilities"][ids[3]] == 0.99  # clamped
    proj = list(parsed["pava_projected"].values())
    assert all(a <= b + 1e-12 for a, b in itertools.pairwise(proj))


def test_eval_flash_lite_json_falls_back_on_unparseable_output() -> None:
    lad = default_ladder()
    out = asyncio.run(
        _eval_flash_lite_json(_fake_interactions_client("not json", {}), "hi", ladder=lad)
    )
    assert out["parsed"]["choice_tier"] == lad.fallback_tier_id
    assert set(out["parsed"]["tier_probabilities"]) == set(lad.ordered_tier_ids)
