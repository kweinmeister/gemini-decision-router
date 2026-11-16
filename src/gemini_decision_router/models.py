"""Candidate model tiers, valid thinking_level guards, and PAVA isotonic projection."""

from __future__ import annotations

from dataclasses import dataclass, field

from gemini_decision_router.ladder import (
    ThinkingLevel,
    TierLadder,
    TierSpec,
    default_ladder,
)

# Tier ids are defined by the active ladder (bundled `default_tiers.toml` or a user TOML file).
TierID = str

__all__ = [
    "RouteDecision",
    "ThinkingLevel",
    "TierID",
    "TierLadder",
    "TierSpec",
    "pava_isotonic_increasing",
    "validate_thinking_level",
]


def validate_thinking_level(
    model: str, thinking_level: str, *, ladder: TierLadder | None = None
) -> ThinkingLevel:
    """Validate that `thinking_level` is supported by `model` (prevents HTTP 400 on minimal).

    Supported levels come from the ladder's `[models]` catalog (see `default_tiers.toml`).
    """
    return (ladder or default_ladder()).validate_thinking_level(model, thinking_level)


def pava_isotonic_increasing(values: list[float] | tuple[float, ...]) -> list[float]:
    """Project `values` onto the monotone non-decreasing cone (p_0 <= p_1 <= ... <= p_{K-1})
    using the Pool-Adjacent-Violators Algorithm (PAVA) under L2 loss.

    Ensures isolated `noul` solvability estimates never violate capability monotonicity.
    """
    if not values:
        return []
    # Each block is [sum_of_values, count]
    blocks: list[list[float]] = []
    for v in values:
        clamped = min(0.99, max(0.01, float(v)))
        blocks.append([clamped, 1.0])
        while len(blocks) >= 2 and (blocks[-2][0] / blocks[-2][1]) > (
            blocks[-1][0] / blocks[-1][1]
        ):
            last_sum, last_cnt = blocks.pop()
            blocks[-1][0] += last_sum
            blocks[-1][1] += last_cnt
    result: list[float] = []
    for block_sum, block_cnt in blocks:
        mean_val = block_sum / block_cnt
        result.extend([mean_val] * int(block_cnt))
    return result


@dataclass
class RouteDecision:
    """Structured output of a routing decision."""

    selected_tier_id: TierID
    model: str
    thinking_level: ThinkingLevel
    policy_name: str
    # P(selected tier suffices) under the policy's distribution; 0.0 on fallback.
    confidence: float
    router_latency_ms: float
    router_input_tokens: int
    router_compute_units: float
    estimated_downstream_compute_units: float
    raw_probabilities: dict[str, float] = field(default_factory=dict)
    solvability_probabilities: dict[TierID, float] = field(default_factory=dict)
    expected_complexity_score: float | None = None
    needs_precision_noul: float | None = None
    fallback_triggered: bool = False
    fallback_reason: str | None = None
