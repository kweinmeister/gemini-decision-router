"""Integration and unit tests for the gdr command-line interface (gemini_decision_router.cli)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from gemini_decision_router import cli
from gemini_decision_router.models import RouteDecision, ThinkingLevel


def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Test that gdr --help displays usage and exits with code 0."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "gdr" in captured.out
    assert "route" in captured.out
    assert "tiers" in captured.out


def test_cli_tiers_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Test that gdr tiers prints valid ladder summary JSON and returns 0."""
    code = cli.main(["tiers"])
    assert code == 0
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert "tiers" in summary
    assert "baseline_tier" in summary
    assert len(summary["tiers"]) >= 4


def _make_decision(
    selected_tier_id: str = "t1_hydrogen_low",
    thinking_level: ThinkingLevel = "low",
    confidence: float = 0.85,
    router_latency_ms: float = 120.0,
    estimated_downstream_compute_units: float = 0.001,
) -> RouteDecision:
    return RouteDecision(
        selected_tier_id=selected_tier_id,
        model="gemini-4-hydrogen",
        thinking_level=thinking_level,
        policy_name="score_tail",
        confidence=confidence,
        router_latency_ms=router_latency_ms,
        router_input_tokens=250,
        router_compute_units=0.00002,
        estimated_downstream_compute_units=estimated_downstream_compute_units,
    )


def test_cli_route_command_mocked(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test gdr route routes prompt with mocked DecisionRouter."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    fake_decision = _make_decision()

    monkeypatch.setattr(
        cli.DecisionRouter,
        "route",
        MagicMock(return_value=fake_decision),
    )

    code = cli.main(["route", "Write a python function to compute fibonacci."])
    assert code == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["selected_tier_id"] == "t1_hydrogen_low"
    assert result["model"] == "gemini-4-hydrogen"
    assert result["confidence"] == 0.85


def test_cli_route_and_execute_mocked(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test gdr route --execute routes and executes prompt with mocked downstream interaction."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    fake_decision = _make_decision(
        selected_tier_id="t2_hydrogen_med",
        thinking_level="medium",
        confidence=0.92,
        router_latency_ms=130.0,
        estimated_downstream_compute_units=0.002,
    )
    fake_interaction = SimpleNamespace(
        outputs=[SimpleNamespace(type="text", text="def fib(n): ...")]
    )

    monkeypatch.setattr(
        cli.DecisionRouter,
        "route_and_execute",
        MagicMock(return_value=(fake_decision, fake_interaction)),
    )

    code = cli.main(["route", "Solve problem", "--execute"])
    assert code == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["decision"]["selected_tier_id"] == "t2_hydrogen_med"
    assert "def fib(n): ..." in result["downstream_output"]


def test_cli_route_with_attachment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test gdr route with an attachment inside allowed-dir."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    img_file = tmp_path / "test.png"
    # Minimal 1x1 PNG bytes
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff"
        b"?\x00\x05\xfe\x02\xfe\xa76\x814\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    img_file.write_bytes(png_bytes)

    captured_payload: list[Any] = []

    def mock_route(self: Any, payload: Any) -> RouteDecision:
        captured_payload.append(payload)
        return RouteDecision(
            selected_tier_id="t0_flash_lite_min",
            model="gemini-3.5-flash-lite",
            thinking_level="minimal",
            policy_name="score_tail",
            confidence=0.99,
            router_latency_ms=110.0,
            router_input_tokens=300,
            router_compute_units=0.00001,
            estimated_downstream_compute_units=0.0005,
        )

    monkeypatch.setattr(cli.DecisionRouter, "route", mock_route)

    code = cli.main(
        [
            "route",
            "What is in this image?",
            "--attach",
            str(img_file),
            "--allowed-dir",
            str(tmp_path),
        ]
    )
    assert code == 0
    assert len(captured_payload) == 1
    blocks = captured_payload[0]
    assert isinstance(blocks, list)
    assert len(blocks) == 2
    assert blocks[0]["type"] == "image"
    assert blocks[1]["type"] == "text"
    assert blocks[1]["text"] == "What is in this image?"
