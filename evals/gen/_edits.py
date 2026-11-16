"""Shared source-code editing helpers for item generators."""

from __future__ import annotations

from collections.abc import Iterable


def apply_edits(source: str, edits: Iterable[tuple[str, str]]) -> str:
    """Apply exact substring replacements to `source`, asserting each anchor occurs exactly once."""
    for old, new in edits:
        count = source.count(old)
        if count != 1:
            raise AssertionError(f"edit anchor must occur exactly once (found {count}): {old!r}")
        source = source.replace(old, new)
    return source
