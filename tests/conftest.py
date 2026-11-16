"""Shared pytest fixtures and test configuration for gemini-decision-router tests."""

from __future__ import annotations

import pytest

from evals.sandbox import SandboxBackend, detect_available_backend


@pytest.fixture(scope="module")
def real_backend() -> SandboxBackend:
    """Return available container sandbox backend ('container' or 'docker'), skipping if unavailable."""
    try:
        return detect_available_backend()
    except RuntimeError as exc:
        pytest.skip(f"no sandbox backend available: {exc}")
