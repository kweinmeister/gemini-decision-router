"""Post-hoc probability calibration (Platt scaling) for zero-shot router probabilities."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from gemini_decision_router.models import TierID

__all__ = [
    "PlattParams",
    "fit_per_tier_platt_scaling",
    "fit_platt_scaling",
]


def _clamp_prob(prob: float) -> float:
    return min(0.999, max(0.001, float(prob)))


def _logit(prob: float) -> float:
    p = _clamp_prob(prob)
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        exp_neg = math.exp(-min(z, 60.0))
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(max(z, -60.0))
    return exp_pos / (1.0 + exp_pos)


@dataclass(frozen=True)
class PlattParams:
    """2-parameter Platt scaling: ``sigmoid(a * logit(p) + b)``."""

    a: float = 1.0
    b: float = 0.0

    def apply(self, raw_prob: float) -> float:
        """Transform a raw probability into a calibrated probability in ``[0.01, 0.99]``."""
        calibrated = _sigmoid(self.a * _logit(raw_prob) + self.b)
        return min(0.99, max(0.01, calibrated))


def fit_platt_scaling(
    raw_probs: Sequence[float],
    labels: Sequence[float],
    *,
    reg: float = 1e-2,
    max_iter: int = 50,
    tol: float = 1e-7,
) -> PlattParams:
    """Fit L2-regularized 2-parameter Platt scaling via damped Newton-Raphson.

    Uses Platt's Laplace target smoothing when binary labels are provided so
    separable calibration folds do not diverge to infinite slopes.
    """
    n = len(raw_probs)
    if n == 0 or len(labels) != n:
        raise ValueError(
            f"raw_probs ({n}) and labels ({len(labels)}) must be non-empty and equal length"
        )

    xs = [_logit(p) for p in raw_probs]
    ys_raw = [min(1.0, max(0.0, float(y))) for y in labels]

    n_pos = sum(1.0 for y in ys_raw if y >= 0.5)
    n_neg = n - n_pos
    hi_target = (n_pos + 1.0) / (n_pos + 2.0)
    lo_target = 1.0 / (n_neg + 2.0)
    ys = [lo_target + y * (hi_target - lo_target) for y in ys_raw]

    a = 1.0
    b = 0.0
    for _ in range(max_iter):
        g_a = reg * (a - 1.0)
        g_b = reg * b
        h_aa = reg
        h_ab = 0.0
        h_bb = reg

        for x, y in zip(xs, ys, strict=True):
            p = _sigmoid(a * x + b)
            diff = p - y
            w = max(1e-9, p * (1.0 - p))
            g_a += diff * x
            g_b += diff
            h_aa += w * x * x
            h_ab += w * x
            h_bb += w

        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            break
        step_a = (h_bb * g_a - h_ab * g_b) / det
        step_b = (h_aa * g_b - h_ab * g_a) / det
        a -= step_a
        b -= step_b
        if max(abs(step_a), abs(step_b)) < tol:
            break

    if a <= 0.0:
        return PlattParams()
    return PlattParams(a=float(a), b=float(b))


def fit_per_tier_platt_scaling(
    probs_by_tier: Mapping[TierID, Sequence[float]],
    labels_by_tier: Mapping[TierID, Sequence[float]],
    *,
    reg: float = 1e-2,
) -> dict[TierID, PlattParams]:
    """Fit independent Platt calibration parameters for each tier in ``probs_by_tier``."""
    calibrated: dict[TierID, PlattParams] = {}
    for tier_id, probs in probs_by_tier.items():
        labels = labels_by_tier.get(tier_id)
        if labels is None or len(probs) == 0:
            calibrated[tier_id] = PlattParams()
            continue
        calibrated[tier_id] = fit_platt_scaling(probs, labels, reg=reg)
    return calibrated
