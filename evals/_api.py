"""Shared HTTP retry, latency measurement, and JSONL persistence for evals scripts."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from gemini_decision_router.cli import add_tiers_argument

__all__ = [
    "BENCHMARK_SAFETY_SETTINGS",
    "NON_RETRYABLE_STATUS_CODES",
    "add_tiers_argument",
    "append_jsonl",
    "call_with_retry_and_latency",
    "is_retryable_error",
    "read_jsonl",
]

# HTTP status codes that must not be retried because re-sending the same request cannot succeed
NON_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({400, 401, 403, 404, 405, 422})

# Permissive safety settings for benchmark evaluation (prevents false-positive blocks on benign code/debugging tasks)
BENCHMARK_SAFETY_SETTINGS: list[dict[str, str]] = [
    {
        "type": "dangerous_content",
        "threshold": "block_only_high",
    },
    {
        "type": "hate_speech",
        "threshold": "block_only_high",
    },
    {
        "type": "harassment",
        "threshold": "block_only_high",
    },
    {
        "type": "sexually_explicit",
        "threshold": "block_only_high",
    },
]


def is_retryable_error(exc: Exception) -> bool:
    """Return False if `exc` is a programmer error or a client error (e.g. 400 Bad Request)."""
    if isinstance(exc, (ValueError, TypeError, KeyError, AssertionError, NotImplementedError)):
        return False

    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return not (isinstance(code, int) and code in NON_RETRYABLE_STATUS_CODES)


async def call_with_retry_and_latency[T](
    coro_fn: Callable[[], Coroutine[Any, Any, T]],
    *,
    max_attempts: int = 4,
    base_delay: float = 1.5,
    sleep_fn: Callable[[float], Coroutine[Any, Any, None]] = asyncio.sleep,
) -> tuple[T, float, int]:
    """Execute `coro_fn()` with exponential backoff and record ONLY the winning attempt's latency in ms.

    Returns `(result, latency_ms, attempt_number)`.
    Raises non-retryable exceptions immediately on the first attempt.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        t0 = time.perf_counter()
        try:
            result = await coro_fn()
        except Exception as exc:
            last_exc = exc
            if not is_retryable_error(exc) or attempt == max_attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            await sleep_fn(delay)
        else:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return result, latency_ms, attempt

    assert last_exc is not None
    raise last_exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of parsed dictionaries. Returns empty list if path does not exist."""
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append a single JSON record to `path`, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
