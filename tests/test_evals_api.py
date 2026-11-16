"""Unit tests for shared evals API retry loop, latency timing, and JSONL persistence."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from evals._api import (
    append_jsonl,
    call_with_retry_and_latency,
    is_retryable_error,
    read_jsonl,
)


class FakeHTTPError(Exception):
    def __init__(self, code: int, message: str = "HTTP error") -> None:
        super().__init__(message)
        self.code = code


def test_is_retryable_error_distinguishes_transient_from_permanent() -> None:
    # Non-retryable programmer / data errors
    assert not is_retryable_error(ValueError("bad value"))
    assert not is_retryable_error(TypeError("bad type"))
    assert not is_retryable_error(KeyError("missing key"))
    assert not is_retryable_error(AssertionError("assertion failed"))

    # Non-retryable HTTP client errors (400, 401, 403, 404, etc.)
    assert not is_retryable_error(FakeHTTPError(400, "Bad Request"))
    assert not is_retryable_error(FakeHTTPError(401, "Unauthorized"))
    assert not is_retryable_error(FakeHTTPError(403, "Forbidden"))
    assert not is_retryable_error(FakeHTTPError(404, "Not Found"))

    # Retryable transient errors (429 rate limit, 503 service unavailable, timeouts)
    assert is_retryable_error(FakeHTTPError(429, "Rate limit exceeded"))
    assert is_retryable_error(FakeHTTPError(500, "Internal Server Error"))
    assert is_retryable_error(FakeHTTPError(503, "Service Unavailable"))
    assert is_retryable_error(TimeoutError())
    assert is_retryable_error(ConnectionError("connection reset"))


def test_retry_loop_succeeds_on_first_attempt() -> None:
    calls = 0

    async def _op() -> str:
        nonlocal calls
        calls += 1
        return "success"

    sleeps: list[float] = []

    async def _mock_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    res, latency_ms, attempt = asyncio.run(call_with_retry_and_latency(_op, sleep_fn=_mock_sleep))
    assert res == "success"
    assert calls == 1
    assert attempt == 1
    assert latency_ms >= 0.0
    assert sleeps == []


def test_retry_loop_retries_transient_error_with_backoff_and_records_winning_latency() -> None:
    calls = 0

    async def _op() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise FakeHTTPError(503, "Temporary failure")
        return "recovered"

    sleeps: list[float] = []

    async def _mock_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    res, latency_ms, attempt = asyncio.run(
        call_with_retry_and_latency(_op, base_delay=1.0, max_attempts=4, sleep_fn=_mock_sleep)
    )
    assert res == "recovered"
    assert calls == 3
    assert attempt == 3
    assert latency_ms >= 0.0
    # Exponential backoff: base_delay * 2^(attempt-1): 1.0 * 1 = 1.0, 1.0 * 2 = 2.0
    assert sleeps == [1.0, 2.0]


def test_retry_loop_fails_fast_on_non_retryable_error_without_sleeping() -> None:
    calls = 0

    async def _op() -> str:
        nonlocal calls
        calls += 1
        raise FakeHTTPError(400, "Invalid argument")

    sleeps: list[float] = []

    async def _mock_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    with pytest.raises(FakeHTTPError) as exc_info:
        asyncio.run(call_with_retry_and_latency(_op, max_attempts=4, sleep_fn=_mock_sleep))

    assert exc_info.value.code == 400
    assert calls == 1
    assert sleeps == []


def test_retry_loop_raises_after_exhausting_max_attempts() -> None:
    calls = 0

    async def _op() -> str:
        nonlocal calls
        calls += 1
        raise FakeHTTPError(500, "Persistent 500")

    sleeps: list[float] = []

    async def _mock_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    with pytest.raises(FakeHTTPError) as exc_info:
        asyncio.run(
            call_with_retry_and_latency(_op, max_attempts=3, base_delay=0.5, sleep_fn=_mock_sleep)
        )

    assert exc_info.value.code == 500
    assert calls == 3
    assert sleeps == [0.5, 1.0]


def test_jsonl_read_and_append_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "records.jsonl"
    assert read_jsonl(path) == []

    rec1 = {"id": "1", "val": 42}
    rec2 = {"id": "2", "val": "hello"}
    append_jsonl(path, rec1)
    append_jsonl(path, rec2)

    rows = read_jsonl(path)
    assert rows == [rec1, rec2]


def test_benchmark_safety_settings_configured() -> None:
    from evals._api import BENCHMARK_SAFETY_SETTINGS

    assert isinstance(BENCHMARK_SAFETY_SETTINGS, list)
    assert len(BENCHMARK_SAFETY_SETTINGS) >= 4
    for setting in BENCHMARK_SAFETY_SETTINGS:
        assert "type" in setting
        assert setting["threshold"] == "block_only_high"
