"""Token usage of a single API call, normalized for billing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from gemini_decision_router.envelope import extract_modality_tokens

__all__ = ["Usage"]


@dataclass(frozen=True)
class Usage:
    """Token counts for one call.

    In the Interactions API, `usage.total_output_tokens` EXCLUDES thinking tokens, which are
    reported separately in `usage.total_thought_tokens`. Both are billed at the output rate,
    so cost math must use `billed_output` (= output + thought), never `output` alone.
    """

    input: int = 0
    output: int = 0
    thought: int = 0
    by_modality: Mapping[str, int] = field(default_factory=dict, hash=False)

    @property
    def billed_output(self) -> int:
        """Tokens billed at the output rate: visible output plus thinking."""
        return self.output + self.thought

    @classmethod
    def from_response(cls, response: Any) -> Usage:
        """Read usage from an Interaction or Decision response (missing fields count as 0)."""
        usage = getattr(response, "usage", None)
        return cls(
            input=int(getattr(usage, "total_input_tokens", 0) or 0),
            output=int(getattr(usage, "total_output_tokens", 0) or 0),
            thought=int(getattr(usage, "total_thought_tokens", 0) or 0),
            by_modality=extract_modality_tokens(usage),
        )
