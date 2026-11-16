"""Relative compute-unit math for length-aware routing and efficiency tracking."""

from __future__ import annotations

from gemini_decision_router.ladder import ComputeWeight, TierLadder, default_ladder
from gemini_decision_router.usage import Usage

__all__ = [
    "ComputeWeight",
    "calculate_compute_units",
    "estimate_tier_compute_units",
]


def calculate_compute_units(
    model: str,
    usage: Usage,
    *,
    ladder: TierLadder | None = None,
) -> float:
    """Compute relative compute units (in millions of reference-input-token equivalents).

    Thinking tokens are weighted at the output rate (`usage.billed_output`).
    """
    lad = ladder or default_ladder()
    return lad.compute_weight(model).units(usage.input, usage.billed_output)


def estimate_tier_compute_units(
    tier_id: str,
    known_input_tokens: int,
    *,
    slice_name: str = "default",
    expected_output_tokens_table: dict[str, dict[str, float]] | None = None,
    ladder: TierLadder | None = None,
) -> float:
    """Compute input-length-aware expected compute units: u_hat_{i,t} = (L_i * w_in + O_hat_{slice,t} * w_out) / 1e6."""
    lad = ladder or default_ladder()
    weight = lad.compute_weight(lad.tier(tier_id).model)
    expected_out = lad.expected_output_tokens_for(tier_id, slice_name, expected_output_tokens_table)
    return weight.units(known_input_tokens, expected_out)
