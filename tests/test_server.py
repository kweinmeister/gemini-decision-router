"""Tests for the FastAPI Cloud Run HTTP wrapper (gemini_decision_router.server)."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from gemini_decision_router.observability import RouterObserver
from gemini_decision_router.router import DecisionRouter, PolicyMode, RouterPolicyConfig
from gemini_decision_router.server import create_app


def _fake_client(
    *,
    score: float = 0.25,
    probs: dict[str, float] | None = None,
    downstream_text: str = "Hello from downstream!",
    calls: list[str] | None = None,
) -> Any:
    prob_map = probs or {"0": 0.85, "1": 0.10, "2": 0.03, "3": 0.01, "4": 0.01}

    async def _create_decision(**_kwargs: Any) -> Any:
        if calls is not None:
            calls.append("decisions.create")
        return SimpleNamespace(
            usage=SimpleNamespace(total_input_tokens=120),
            prompt_feedback=None,
            answers={
                "complexity_tier": SimpleNamespace(
                    type="score",
                    score=score,
                    probabilities=prob_map,
                    confidence=0.85,
                ),
                "needs_precision": SimpleNamespace(type="noul", noul=0.1, confidence=0.9),
            },
        )

    async def _create_interaction(**_kwargs: Any) -> Any:
        if calls is not None:
            calls.append("interactions.create")
        return SimpleNamespace(
            output_text=downstream_text,
            usage=SimpleNamespace(
                total_input_tokens=120, total_output_tokens=10, total_thought_tokens=0
            ),
        )

    return SimpleNamespace(
        aio=SimpleNamespace(
            decisions=SimpleNamespace(create=_create_decision),
            interactions=SimpleNamespace(create=_create_interaction),
        )
    )


@pytest.fixture
def make_test_client() -> Callable[..., TestClient]:
    def _factory(
        *,
        client: Any | None = None,
        downstream_client: Any | None = None,
        policy: PolicyMode = "score_tail",
    ) -> TestClient:
        obs = RouterObserver(max_observations=10)
        router = DecisionRouter(
            config=RouterPolicyConfig(policy=policy),
            client=client or _fake_client(),
            downstream_client=downstream_client,
            observer=obs,
        )
        return TestClient(create_app(router=router, observer=obs))

    return _factory


def test_server_health_and_tiers(make_test_client: Callable[..., TestClient]) -> None:
    with make_test_client() as client:
        assert client.get("/health").json() == {"status": "ok"}
        tiers_resp = client.get("/v1/tiers")
        assert tiers_resp.status_code == 200
        assert len(tiers_resp.json()["tiers"]) == 5


def test_server_route_and_metrics(make_test_client: Callable[..., TestClient]) -> None:
    with make_test_client() as client:
        resp = client.post("/v1/route", json={"input": "Say hello."})
        assert resp.status_code == 200
        body = resp.json()
        assert body["selected_tier_id"] == "t0_flash_lite_min"
        assert body["policy_name"] == "score_tail"

        metrics = client.get("/v1/metrics").json()
        assert metrics["total_requests"] == 1
        assert metrics["tier_counts"]["t0_flash_lite_min"] == 1


def test_server_route_allows_per_request_policy_override(
    make_test_client: Callable[..., TestClient],
) -> None:
    with make_test_client(policy="score_tail") as client:
        resp = client.post("/v1/route", json={"input": "Say hello.", "policy": "score_argmax"})
        assert resp.status_code == 200
        assert resp.json()["policy_name"] == "score_argmax"


def test_server_route_and_execute_uses_downstream_client(
    make_test_client: Callable[..., TestClient],
) -> None:
    router_calls: list[str] = []
    downstream_calls: list[str] = []
    routing_client = _fake_client(calls=router_calls)
    exec_client = _fake_client(downstream_text="42", calls=downstream_calls)

    with make_test_client(client=routing_client, downstream_client=exec_client) as client:
        ok = client.post(
            "/v1/route-and-execute",
            json={
                "input": [{"type": "text", "text": "What is 6 * 7?"}],
                "max_output_tokens": 64,
                "system_instruction": "Be concise.",
            },
        )
        assert ok.status_code == 200
        data = ok.json()
        assert data["decision"]["selected_tier_id"] == "t0_flash_lite_min"
        assert data["downstream_output"] == "42"
        assert router_calls == ["decisions.create"]
        assert downstream_calls == ["interactions.create"]


@pytest.mark.parametrize(
    ("payload", "expected_status"),
    [
        ({"input": ""}, 422),
        ({"input": []}, 422),
        ({"input": "hi", "unknown_field": 1}, 422),
        ({"input": "hi", "policy": "not_a_policy"}, 422),
        ({"input": [{"type": "text", "data": "wrong"}]}, 422),
        ({"input": [{"type": "image"}]}, 422),
    ],
)
def test_server_rejects_invalid_requests(
    make_test_client: Callable[..., TestClient],
    payload: dict[str, Any],
    expected_status: int,
) -> None:
    with make_test_client() as client:
        resp = client.post("/v1/route", json=payload)
        assert resp.status_code == expected_status


def test_server_emits_structured_json_log_with_trace_and_redacts_prompt(
    make_test_client: Callable[..., TestClient],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import json

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-log-proj")
    secret_prompt = "SECRET_USER_PII_DO_NOT_LOG"
    valid_trace_id = "0123456789abcdef0123456789abcdef"

    with make_test_client() as client:
        capsys.readouterr()
        resp = client.post(
            "/v1/route",
            json={"input": secret_prompt},
            headers={"X-Cloud-Trace-Context": f"{valid_trace_id}/12345;o=1"},
        )
        assert resp.status_code == 200

    captured = capsys.readouterr().out.strip().splitlines()
    assert len(captured) == 1
    entry = json.loads(captured[0])
    assert entry["severity"] == "INFO"
    assert entry["message"] == "route_decision"
    assert (
        entry["logging.googleapis.com/trace"] == f"projects/test-log-proj/traces/{valid_trace_id}"
    )
    assert entry["selected_tier_id"] == "t0_flash_lite_min"
    assert entry["input_type"] == "text"
    assert secret_prompt not in captured[0]


def test_server_omits_malformed_trace_and_enforces_limits_and_bad_b64(
    make_test_client: Callable[..., TestClient],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import json

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-log-proj")
    with make_test_client() as client:
        capsys.readouterr()
        resp = client.post(
            "/v1/route",
            json={"input": "hello"},
            headers={"X-Cloud-Trace-Context": "not-32-hex/12345;o=1"},
        )
        assert resp.status_code == 200
        entry = json.loads(capsys.readouterr().out.strip())
        assert "logging.googleapis.com/trace" not in entry

        # Malformed base64 PDF returns 422 rather than 500
        bad_pdf_resp = client.post(
            "/v1/route",
            json={
                "input": [{"type": "document", "mime_type": "application/pdf", "data": "!!bad!!"}]
            },
        )
        assert bad_pdf_resp.status_code == 422

    # Configurable GEMINI_ROUTER_MAX_* limits reject oversized inputs with 422
    monkeypatch.setenv("GEMINI_ROUTER_MAX_TEXT_CHARS", "10")
    monkeypatch.setenv("GEMINI_ROUTER_MAX_MEDIA_B64_CHARS", "8")
    monkeypatch.setenv("GEMINI_ROUTER_MAX_BLOCKS", "1")
    monkeypatch.setenv("GEMINI_ROUTER_MAX_SYSTEM_INSTRUCTION_CHARS", "5")
    with make_test_client() as client:
        assert client.post("/v1/route", json={"input": "x" * 11}).status_code == 422
        assert (
            client.post(
                "/v1/route",
                json={"input": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/v1/route",
                json={"input": [{"type": "image", "mime_type": "image/png", "data": "A" * 9}]},
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/v1/route-and-execute",
                json={"input": "ok", "system_instruction": "too long"},
            ).status_code
            == 422
        )

    from gemini_decision_router.server import RouteRequest, ServerLimits

    with pytest.raises(ValueError, match="must be > 0"):
        ServerLimits(max_blocks=0)

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RouteRequest(input="   ")
    with pytest.raises(ValidationError):
        RouteRequest(input=[{"type": "text", "text": "  \t "}])
    with pytest.raises(ValidationError):
        RouteRequest(input="ok", policy="choice_crc", conformal_cdf_threshold=1.5)
    req = RouteRequest(input="ok", policy="choice_crc", conformal_cdf_threshold=0.85)
    assert req.conformal_cdf_threshold == 0.85
