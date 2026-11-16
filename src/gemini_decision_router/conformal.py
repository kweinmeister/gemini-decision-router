"""Finite-sample Conformal Risk Control (CRC) for ordered model ladders.

Implements Angelopoulos et al. (ICLR 2024) Conformal Risk Control over a
1D monotone cumulative-probability threshold ``theta in (0, 1]`` on ``choice``
tier distributions, with optional importance weights for traffic-mix shifts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from gemini_decision_router.models import TierID

__all__ = [
    "ConformalCalibrationResult",
    "calibrate_conformal_cdf_threshold",
    "select_tier_from_cdf_threshold",
]


@dataclass(frozen=True)
class ConformalCalibrationResult:
    """Result of fitting a 1D Conformal Risk Control CDF threshold."""

    threshold: float
    target_alpha: float
    empirical_risk: float
    finite_sample_bound: float
    n_effective: float


def select_tier_from_cdf_threshold(
    probabilities: dict[str, float],
    ordered_tier_ids: Sequence[TierID],
    threshold: float,
) -> tuple[TierID, float]:
    """Pick the cheapest tier where cumulative probability mass reaches ``threshold``."""
    if not ordered_tier_ids:
        raise ValueError("ordered_tier_ids must not be empty")
    cum_prob = 0.0
    for tier_id in ordered_tier_ids:
        cum_prob += float(probabilities.get(tier_id, 0.0))
        if cum_prob >= threshold:
            return tier_id, min(1.0, max(0.0, cum_prob))
    return ordered_tier_ids[-1], 1.0


def calibrate_conformal_cdf_threshold(
    probabilities_list: Sequence[dict[str, float]],
    tier_accuracies_list: Sequence[dict[TierID, float]],
    ordered_tier_ids: Sequence[TierID],
    *,
    target_alpha: float = 0.015,
    weights: Sequence[float] | None = None,
    upper_bound_b: float = 1.0,
    grid_size: int = 200,
) -> ConformalCalibrationResult:
    """Find the smallest ``theta`` guaranteeing ``E[regret] <= target_alpha``.

    Per-item routing regret is ``L_i(theta) = max_t Y_{i,t} - Y_{i, pi_theta(x_i)}``
    (clamped to ``[0, upper_bound_b]``). Under exchangeability and monotone-decreasing
    expected error along ``ordered_tier_ids``, Angelopoulos et al. (2024) Theorem 1
    guarantees ``(sum(w_i * L_i(theta)) + B) / (sum(w_i) + 1) <= alpha`` yields
    ``E[L_{n+1}(theta_hat)] <= alpha``.
    """
    n = len(probabilities_list)
    if n == 0 or len(tier_accuracies_list) != n:
        raise ValueError(
            "probabilities_list and tier_accuracies_list must be non-empty and equal length"
        )
    if not ordered_tier_ids:
        raise ValueError("ordered_tier_ids must not be empty")
    if not 0.0 < target_alpha <= 1.0:
        raise ValueError(f"target_alpha must be in (0, 1], got {target_alpha}")

    if weights is None:
        norm_weights = [1.0] * n
        n_eff = float(n)
    else:
        if len(weights) != n:
            raise ValueError(f"weights length ({len(weights)}) must match items ({n})")
        if any(float(w) < 0.0 for w in weights):
            raise ValueError("weights must be non-negative")
        w_sum = sum(float(w) for w in weights)
        if w_sum <= 0.0:
            raise ValueError("weights must have positive sum")
        # Scale weights so sum(w_i) == n (or Kish effective sample size when non-uniform)
        sum_sq = sum(float(w) ** 2 for w in weights)
        n_eff = (w_sum * w_sum) / sum_sq
        scale = n_eff / w_sum
        norm_weights = [float(w) * scale for w in weights]

    precomputed: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
    for probs, accs, w in zip(probabilities_list, tier_accuracies_list, norm_weights, strict=True):
        best_acc = max(float(accs.get(tid, 0.0)) for tid in ordered_tier_ids)
        cum_prob = 0.0
        cum_probs: list[float] = []
        w_regrets: list[float] = []
        for tid in ordered_tier_ids:
            cum_prob += float(probs.get(tid, 0.0))
            cum_probs.append(cum_prob)
            regret = max(0.0, min(upper_bound_b, best_acc - float(accs.get(tid, 0.0))))
            w_regrets.append(w * regret)
        precomputed.append((tuple(cum_probs), tuple(w_regrets)))

    last_idx = len(ordered_tier_ids) - 1
    candidates = [i / grid_size for i in range(1, grid_size + 1)]
    best_result: ConformalCalibrationResult | None = None
    last_emp_risk = 0.0
    last_conformal_bound = upper_bound_b / (n_eff + 1.0)

    for theta in candidates:
        weighted_loss = 0.0
        for item_cum_probs, item_w_regrets in precomputed:
            chosen_idx = last_idx
            for idx, cp in enumerate(item_cum_probs):
                if cp >= theta:
                    chosen_idx = idx
                    break
            weighted_loss += item_w_regrets[chosen_idx]

        last_emp_risk = weighted_loss / n_eff
        last_conformal_bound = (weighted_loss + upper_bound_b) / (n_eff + 1.0)
        if last_conformal_bound <= target_alpha:
            best_result = ConformalCalibrationResult(
                threshold=theta,
                target_alpha=target_alpha,
                empirical_risk=last_emp_risk,
                finite_sample_bound=last_conformal_bound,
                n_effective=n_eff,
            )
            break

    if best_result is None:
        # Finite-sample floor B / (n_eff + 1) exceeds target_alpha; fall back to theta = 1.0
        return ConformalCalibrationResult(
            threshold=1.0,
            target_alpha=target_alpha,
            empirical_risk=last_emp_risk,
            finite_sample_bound=last_conformal_bound,
            n_effective=n_eff,
        )
    return best_result
