"""Tests for TOML-configurable tier ladders."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from gemini_decision_router.cli import main as cli_main
from gemini_decision_router.compute import (
    calculate_compute_units,
    estimate_tier_compute_units,
)
from gemini_decision_router.ladder import (
    LADDER_ENV_VAR,
    TierLadder,
    default_ladder,
    default_score_cutoffs,
)
from gemini_decision_router.router import (
    DecisionRouter,
    RouterPolicyConfig,
    build_choice_questions,
    build_noul_solvability_questions,
    build_score_questions,
    select_tier_from_score_distribution,
)
from gemini_decision_router.usage import Usage

REPO_ROOT = Path(__file__).resolve().parents[1]

CARBON_LADDER_TOML = """
schema_version = 1

[router]
fallback_tier = "mid"
frontier_tiers = ["argon", "carbon"]
score_cutoffs = [0.65, 1.45, 2.35, 3.25]

[models."gemini-decision-1"]
compute_weight = { input = 0.15, output = 0.0 }

[models."gemini-3.5-flash-lite"]
thinking_levels = ["minimal", "low"]
compute_weight = { input = 0.4, output = 3.5 }

[models."gemini-3.8-flash"]
thinking_levels = ["low", "medium"]
compute_weight = { input = 1.0, output = 5.0 }

[models."gemini-4-argon"]
thinking_levels = ["medium"]
compute_weight = { input = 2.5, output = 13.0 }

[models."gemini-4-carbon"]
thinking_levels = ["high"]
compute_weight = { input = 2.5, output = 13.0 }

[[tiers]]
id = "lite"
model = "gemini-3.5-flash-lite"
thinking_level = "minimal"
can_solve = "trivial"
cannot_solve = "non-trivial"
expected_output_tokens = { default = 100 }

[[tiers]]
id = "low"
model = "gemini-3.8-flash"
thinking_level = "low"
can_solve = "routine"
cannot_solve = "deep"
expected_output_tokens = { default = 200 }

[[tiers]]
id = "mid"
model = "gemini-3.8-flash"
thinking_level = "medium"
can_solve = "moderate"
cannot_solve = "frontier"
expected_output_tokens = { default = 400, code = 900 }

[[tiers]]
id = "argon"
model = "gemini-4-argon"
thinking_level = "medium"
can_solve = "complex"
cannot_solve = "max depth"
expected_output_tokens = { default = 1000 }

[[tiers]]
id = "carbon"
model = "gemini-4-carbon"
thinking_level = "high"
can_solve = "max depth"
cannot_solve = "ill-posed"
expected_output_tokens = { default = 3000 }
"""


@pytest.fixture
def carbon_ladder(tmp_path: Path) -> TierLadder:
    p = tmp_path / "tiers.toml"
    p.write_text(CARBON_LADDER_TOML)
    return TierLadder.from_toml(p)


def _minimal_ladder_dict(**router: object) -> dict[str, object]:
    return {
        "router": router,
        "models": {
            "gemini-decision-1": {"compute_weight": {"input": 0.1, "output": 0.0}},
            "m": {
                "thinking_levels": ["low", "high"],
                "compute_weight": {"input": 1.0, "output": 2.0},
            },
        },
        "tiers": [
            {
                "id": "a",
                "model": "m",
                "thinking_level": "low",
                "can_solve": "x",
                "cannot_solve": "y",
                "expected_output_tokens": {"default": 10},
            },
            {
                "id": "b",
                "model": "m",
                "thinking_level": "high",
                "can_solve": "x",
                "cannot_solve": "y",
                "expected_output_tokens": {"default": 20},
            },
        ],
    }


def test_bundled_ladder_is_five_tiers_hydrogen_without_carbon() -> None:
    lad = default_ladder()
    assert lad.ordered_tier_ids == (
        "t0_flash_lite_min",
        "t1_hydrogen_low",
        "t2_hydrogen_med",
        "t3_hydrogen_high",
        "t4_hydrogen_xhigh",
    )
    assert "gemini-4-hydrogen" in lad.models
    assert lad.models["gemini-4-hydrogen"].thinking_levels == frozenset(
        {"low", "medium", "high", "xhigh"}
    )
    assert "gemini-4-carbon" in lad.models
    assert all(t.model != "gemini-4-carbon" for t in lad.tiers)
    assert len(lad.score_cutoffs) == 4
    assert lad.frontier_tier_ids == ("t3_hydrogen_high", "t4_hydrogen_xhigh")
    assert len(build_score_questions()["complexity_tier"]["criteria"]) == 5
    assert default_ladder() is lad  # cached process-wide


def test_bad_env_ladder_fails_lazily_not_at_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A broken $GEMINI_ROUTER_TIERS must not break imports or explicit `--tiers` paths."""
    missing = str(tmp_path / "nonexistent.toml")
    env = {**os.environ, LADDER_ENV_VAR: missing, "PYTHONPATH": f"src{os.pathsep}."}
    proc = subprocess.run(
        [sys.executable, "-c", "import gemini_decision_router, evals.analyze, evals.plots"],
        check=False,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr

    good = tmp_path / "tiers.toml"
    good.write_text(CARBON_LADDER_TOML)
    monkeypatch.setenv(LADDER_ENV_VAR, missing)
    default_ladder.cache_clear()
    try:
        assert cli_main(["tiers", "--tiers", str(good)]) == 0
        assert json.loads(capsys.readouterr().out)["tiers"][-1]["id"] == "carbon"
        with pytest.raises(FileNotFoundError):
            default_ladder()
    finally:
        monkeypatch.delenv(LADDER_ENV_VAR)
        default_ladder.cache_clear()
    assert len(default_ladder()) == 5


def test_custom_five_tier_ladder_routes_with_carbon(carbon_ladder: TierLadder) -> None:
    assert len(carbon_ladder) == 5
    assert len(build_score_questions(carbon_ladder)["complexity_tier"]["criteria"]) == 5
    assert set(build_choice_questions(carbon_ladder)["target_tier"]["criteria"]) == set(
        carbon_ladder.ordered_tier_ids
    )
    assert "can_carbon_solve" in build_noul_solvability_questions(carbon_ladder)

    cfg = RouterPolicyConfig(policy="score_tail", ladder=carbon_ladder)
    probs = {"0": 0.40, "1": 0.15, "2": 0.05, "3": 0.25, "4": 0.15}
    # tail mass on {argon, carbon} = 0.40 >= 0.35 -> escalate to cheapest frontier tier (argon)
    assert select_tier_from_score_distribution(1.8, probs, None, cfg)[0] == "argon"
    # score above the last cutoff -> top tier
    assert select_tier_from_score_distribution(3.6, probs, None, cfg)[0] == "carbon"

    resp = SimpleNamespace(
        usage=SimpleNamespace(total_input_tokens=100),
        prompt_feedback=None,
        answers={
            "complexity_tier": SimpleNamespace(score=0.1, probabilities={"0": 0.9, "1": 0.1}),
        },
    )
    client = SimpleNamespace(decisions=SimpleNamespace(create=lambda **kw: resp))
    router = DecisionRouter(client=client, ladder=carbon_ladder)
    dec = router.route("hi")
    assert dec.selected_tier_id == "lite"
    assert dec.model == "gemini-3.5-flash-lite"


def test_custom_ladder_compute_weights_and_token_priors(carbon_ladder: TierLadder) -> None:
    units = calculate_compute_units("gemini-4-carbon", Usage(input=1_000_000), ladder=carbon_ladder)
    assert units == pytest.approx(2.5)
    est = estimate_tier_compute_units("mid", 0, slice_name="code", ladder=carbon_ladder)
    assert est == pytest.approx(900 * 5.0 / 1e6)
    # Unknown slice falls back to the tier's default prior
    est_default = estimate_tier_compute_units("mid", 0, slice_name="nope", ladder=carbon_ladder)
    assert est_default == pytest.approx(400 * 5.0 / 1e6)


def test_router_defaults_derived_when_omitted() -> None:
    lad = TierLadder.from_dict(_minimal_ladder_dict())
    assert lad.score_cutoffs == default_score_cutoffs(2) == (0.5,)
    assert lad.fallback_tier_id == "b"
    assert lad.frontier_tier_ids == ("b",)
    assert lad.precision_floor_tier_id == "b"
    assert lad.baseline_tier_id == "b"


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda d: d["tiers"][0].update(thinking_level="minimal"),
            "does not support thinking_level",
        ),
        (lambda d: d["tiers"][0].update(model="ghost"), "not defined in \\[models\\]"),
        (lambda d: d["tiers"][1].update(id="a"), "duplicate tier ids"),
        (lambda d: d["router"].update(score_cutoffs=[0.5, 1.5]), "needs 1 values"),
        (lambda d: d["router"].update(fallback_tier="zzz"), "not a defined tier id"),
        (
            lambda d: d["models"]["m"]["compute_weight"].update(input=-1.0),
            "greater than or equal to 0",
        ),
        (
            lambda d: d["models"]["m"]["compute_weight"].update(input=float("inf")),
            "finite number",
        ),
        (
            lambda d: d["router"].update(score_cutoffs=[float("nan")]),
            "finite number",
        ),
        (lambda d: d["tiers"][0].update(expected_output_tokens={"code": 5}), "default is required"),
        (lambda d: d["tiers"][0].update(typo_field=1), "Extra inputs"),
    ],
)
def test_invalid_ladders_rejected(mutate: Callable[[dict[str, Any]], None], match: str) -> None:
    d = _minimal_ladder_dict()
    mutate(d)
    with pytest.raises(ValidationError, match=match):
        TierLadder.from_dict(d)


def test_config_cutoff_count_mismatch_fails_fast(carbon_ladder: TierLadder) -> None:
    client = SimpleNamespace(decisions=SimpleNamespace(create=lambda **kw: None))
    with pytest.raises(ValueError, match="needs 4 values"):
        DecisionRouter(
            config=RouterPolicyConfig(score_cutoffs=(1.0, 2.0, 3.0)),
            client=client,
            ladder=carbon_ladder,
        )


def test_cli_tiers_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = tmp_path / "tiers.toml"
    p.write_text(CARBON_LADDER_TOML)
    assert cli_main(["tiers", "--tiers", str(p)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert [t["id"] for t in out["tiers"]] == ["lite", "low", "mid", "argon", "carbon"]
    assert out["baseline_tier"] == "carbon"
