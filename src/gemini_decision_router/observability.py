"""Production observability, calibration tracking (ECE/Brier/LogLoss), and drift metrics for DecisionRouter."""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from typing import Any

from gemini_decision_router.compute import (
    calculate_compute_units,
    estimate_tier_compute_units,
)
from gemini_decision_router.ladder import TierLadder, default_ladder
from gemini_decision_router.models import RouteDecision, TierID
from gemini_decision_router.usage import Usage


def compute_calibration_metrics(
    predictions: list[tuple[float, float]],
    *,
    num_bins: int = 10,
) -> dict[str, Any]:
    """Compute Brier score, Log-Loss, Expected Calibration Error (ECE), and per-bin reliability summaries.

    Args:
        predictions: List of `(predicted_probability, empirical_outcome)` pairs in `[0, 1]`.
        num_bins: Number of equal-width probability bins in `[0, 1]` (default: 10).
    """
    if num_bins <= 0:
        raise ValueError(f"num_bins must be positive, got {num_bins}")
    if not predictions:
        return {"brier_score": 0.0, "log_loss": 0.0, "ece": 0.0, "count": 0.0, "bins": []}

    m = len(predictions)
    brier_sum = 0.0
    logloss_sum = 0.0
    bins: list[list[tuple[float, float]]] = [[] for _ in range(num_bins)]

    for p_raw, y_raw in predictions:
        p = min(1.0, max(0.0, float(p_raw)))
        y = min(1.0, max(0.0, float(y_raw)))
        brier_sum += (p - y) ** 2
        p_ll = min(1.0 - 1e-6, max(1e-6, p))
        logloss_sum += -(y * math.log(p_ll) + (1.0 - y) * math.log(1.0 - p_ll))
        b_idx = min(num_bins - 1, max(0, int(p * num_bins)))
        bins[b_idx].append((p, y))

    ece = 0.0
    bin_summaries: list[dict[str, float | int]] = []
    for idx, bucket in enumerate(bins):
        if not bucket:
            continue
        mean_p = sum(x[0] for x in bucket) / len(bucket)
        mean_y = sum(x[1] for x in bucket) / len(bucket)
        ece += (len(bucket) / m) * abs(mean_p - mean_y)
        bin_summaries.append(
            {
                "bin_idx": idx,
                "mean_pred": round(mean_p, 4),
                "mean_emp": round(mean_y, 4),
                "count": len(bucket),
            }
        )

    return {
        "brier_score": brier_sum / m,
        "log_loss": logloss_sum / m,
        "ece": ece,
        "count": float(m),
        "bins": bin_summaries,
    }


def jensen_shannon_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """Compute symmetric Jensen-Shannon divergence (base 2, bounded in `[0, 1]`) between two discrete distributions.

    Useful for monitoring router tier-distribution drift over time or evaluating whether
    multimodal input compression (`resolution='low'`, Page-1 PDF slicing) shifts router posteriors.
    """
    keys = sorted(set(p.keys()) | set(q.keys()))
    if not keys:
        return 0.0
    p_vec = [max(1e-12, float(p.get(k, 0.0))) for k in keys]
    q_vec = [max(1e-12, float(q.get(k, 0.0))) for k in keys]
    sp = sum(p_vec)
    sq = sum(q_vec)
    p_vec = [x / sp for x in p_vec]
    q_vec = [x / sq for x in q_vec]
    m_vec = [0.5 * (pi + qi) for pi, qi in zip(p_vec, q_vec, strict=False)]

    kl_pm = sum(pi * math.log2(pi / mi) for pi, mi in zip(p_vec, m_vec, strict=False))
    kl_qm = sum(qi * math.log2(qi / mi) for qi, mi in zip(q_vec, m_vec, strict=False))
    return max(0.0, 0.5 * (kl_pm + kl_qm))


def spearman_rank_correlation(xs: list[float], ys: list[float]) -> float:
    """Compute Spearman rank correlation rho (with average ranks for ties)."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return 0.0

    def _ranks(vals: list[float]) -> list[float]:
        indexed = sorted(enumerate(vals), key=lambda pair: pair[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
                j += 1
            avg_rank = 0.5 * (i + j) + 1.0
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx = _ranks(xs)
    ry = _ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=False))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in rx))
    den_y = math.sqrt(sum((b - my) ** 2 for b in ry))
    if den_x < 1e-12 or den_y < 1e-12:
        return 0.0
    return min(1.0, max(-1.0, num / (den_x * den_y)))


@dataclass
class RouteObservation:
    """A single recorded routing event and optional downstream outcome for observability."""

    decision: RouteDecision
    known_input_tokens: int = 100
    success: bool | None = None
    realized_downstream_compute_units: float | None = None
    realized_downstream_latency_ms: float | None = None


@dataclass
class RouterObservabilitySummary:
    """Aggregated production telemetry for a DecisionRouter instance."""

    total_requests: int
    tier_counts: dict[str, int]
    tier_shares: dict[str, float]
    fallback_rate: float
    mean_router_latency_ms: float
    p95_router_latency_ms: float
    total_router_compute_units: float
    total_estimated_compute_units: float
    baseline_frontier_compute_units: float
    estimated_compute_savings_pct: float
    calibration: dict[str, Any] = field(default_factory=dict)


class RouterObserver:
    """Lightweight in-memory telemetry & calibration observer for `DecisionRouter`.

    Tracks tier distribution, router overhead latency (mean & p95), fallback rate,
    compute savings relative to a static baseline tier (default: the ladder's `router.baseline_tier`),
    and live calibration (ECE, Brier score, Log-Loss) when downstream outcomes are recorded.
    """

    def __init__(
        self,
        *,
        baseline_tier_id: TierID | None = None,
        ladder: TierLadder | None = None,
        max_observations: int | None = None,
    ) -> None:
        if max_observations is not None and max_observations <= 0:
            raise ValueError(f"max_observations must be positive, got {max_observations}")
        self.ladder = ladder or default_ladder()
        self.baseline_tier_id = baseline_tier_id or self.ladder.baseline_tier_id
        self.ladder.tier(self.baseline_tier_id)  # fail fast on unknown tier ids
        self.observations: collections.deque[RouteObservation] = collections.deque(
            maxlen=max_observations
        )

    def record_decision(
        self,
        decision: RouteDecision,
        *,
        known_input_tokens: int = 100,
    ) -> RouteObservation:
        """Record a routing decision emitted by `DecisionRouter`."""
        obs = RouteObservation(decision=decision, known_input_tokens=known_input_tokens)
        self.observations.append(obs)
        return obs

    def record_outcome(
        self,
        decision: RouteDecision,
        *,
        success: bool,
        usage: Usage | None = None,
        downstream_latency_ms: float | None = None,
        known_input_tokens: int | None = None,
    ) -> RouteObservation:
        """Attach a verified downstream outcome (and optional token usage) to a routing decision.

        Updates the observation that `DecisionRouter.route()` already recorded for `decision`,
        so each request is counted once; a decision that was never recorded is appended
        (with `known_input_tokens`, default 100). `usage` (e.g. `Usage.from_response(interaction)`)
        sets the realized downstream compute units, with thinking tokens weighted at the output rate.
        """
        obs = next((o for o in reversed(self.observations) if o.decision is decision), None)
        if obs is None:
            obs = self.record_decision(
                decision,
                known_input_tokens=100 if known_input_tokens is None else known_input_tokens,
            )
        elif known_input_tokens is not None:
            obs.known_input_tokens = known_input_tokens
        obs.success = success
        if usage is not None:
            obs.realized_downstream_compute_units = calculate_compute_units(
                decision.model, usage, ladder=self.ladder
            )
        if downstream_latency_ms is not None:
            obs.realized_downstream_latency_ms = downstream_latency_ms
        return obs

    def summary(self, *, num_calibration_bins: int = 10) -> RouterObservabilitySummary:
        """Compute an aggregated telemetry and calibration report across all recorded decisions."""
        n = len(self.observations)
        ordered_ids = self.ladder.ordered_tier_ids
        if n == 0:
            return RouterObservabilitySummary(
                total_requests=0,
                tier_counts=dict.fromkeys(ordered_ids, 0),
                tier_shares=dict.fromkeys(ordered_ids, 0.0),
                fallback_rate=0.0,
                mean_router_latency_ms=0.0,
                p95_router_latency_ms=0.0,
                total_router_compute_units=0.0,
                total_estimated_compute_units=0.0,
                baseline_frontier_compute_units=0.0,
                estimated_compute_savings_pct=0.0,
                calibration=compute_calibration_metrics([], num_bins=num_calibration_bins),
            )

        counts: dict[str, int] = collections.Counter(
            obs.decision.selected_tier_id for obs in self.observations
        )
        tier_counts = {tid: counts.get(tid, 0) for tid in ordered_ids}
        tier_shares = {tid: round(tier_counts[tid] / n, 4) for tid in ordered_ids}
        fallbacks = sum(1 for obs in self.observations if obs.decision.fallback_triggered)

        latencies = sorted(obs.decision.router_latency_ms for obs in self.observations)
        mean_lat = sum(latencies) / n
        p95_lat = latencies[min(n - 1, int(0.95 * n))]

        total_router_units = sum(obs.decision.router_compute_units for obs in self.observations)
        total_est_units = sum(
            obs.decision.router_compute_units
            + (
                obs.realized_downstream_compute_units
                if obs.realized_downstream_compute_units is not None
                else obs.decision.estimated_downstream_compute_units
            )
            for obs in self.observations
        )

        baseline_units = sum(
            estimate_tier_compute_units(
                tier_id=self.baseline_tier_id,
                known_input_tokens=obs.known_input_tokens,
                ladder=self.ladder,
            )
            for obs in self.observations
        )
        savings_pct = (
            100.0 * (baseline_units - total_est_units) / baseline_units
            if baseline_units > 0
            else 0.0
        )

        cal_pairs: list[tuple[float, float]] = []
        for obs in self.observations:
            # Fallback decisions carry no router probability (confidence 0.0) and would only
            # distort calibration; they are reported via `fallback_rate` instead.
            if obs.success is None or obs.decision.fallback_triggered:
                continue
            # `confidence` is P(selected tier suffices) for every policy
            cal_pairs.append((float(obs.decision.confidence), 1.0 if obs.success else 0.0))

        return RouterObservabilitySummary(
            total_requests=n,
            tier_counts=tier_counts,
            tier_shares=tier_shares,
            fallback_rate=round(fallbacks / n, 4),
            mean_router_latency_ms=round(mean_lat, 2),
            p95_router_latency_ms=round(p95_lat, 2),
            total_router_compute_units=round(total_router_units, 6),
            total_estimated_compute_units=round(total_est_units, 6),
            baseline_frontier_compute_units=round(baseline_units, 6),
            estimated_compute_savings_pct=round(savings_pct, 2),
            calibration=compute_calibration_metrics(cal_pairs, num_bins=num_calibration_bins),
        )
