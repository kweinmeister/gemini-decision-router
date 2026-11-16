"""Core DecisionRouter implementation using POST /v1beta/decisions (gemini-decision-1)."""

from __future__ import annotations

import bisect
import dataclasses
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from google import genai
from google.genai import errors as genai_errors

from gemini_decision_router.calibration import PlattParams
from gemini_decision_router.compute import (
    calculate_compute_units,
    estimate_tier_compute_units,
)
from gemini_decision_router.conformal import select_tier_from_cdf_threshold
from gemini_decision_router.envelope import (
    ImageResolution,
    PdfRouterMode,
    prepare_router_envelope,
)
from gemini_decision_router.ladder import TierLadder, default_ladder
from gemini_decision_router.models import (
    RouteDecision,
    TierID,
    pava_isotonic_increasing,
)
from gemini_decision_router.observability import RouterObserver
from gemini_decision_router.usage import Usage

PolicyMode = Literal[
    "score_argmax",
    "score_tail",
    "choice_argmax",
    "choice_tail",
    "choice_crc",
    "noul_eu",
]

SAFETY_ERROR_CODES = frozenset(
    {
        "safety",
        "prohibited_content",
        "blocklist",
        "image_safety",
        "content_blocked",
    }
)


@dataclass
class RouterPolicyConfig:
    """Hyperparameters and policy configuration for DecisionRouter."""

    policy: PolicyMode = "score_tail"
    slice_name: str = "default"
    # Ladder of tiers/models/compute weights; None -> default_ladder() (bundled TOML or $GEMINI_ROUTER_TIERS)
    ladder: TierLadder | None = None
    # None -> ladder's `router.fallback_tier`
    fallback_tier_id: TierID | None = None
    # Multimodal compression settings for the router pass
    compress_media_for_router: bool = True
    router_image_resolution: ImageResolution | None = "low"
    router_pdf_mode: PdfRouterMode = "page1_pdf"
    # Optional empirical tier cost ordering (e.g. from training folds); None -> ladder.ordered_tier_ids
    tier_order: tuple[TierID, ...] | None = None
    # Thresholds for `score_tail` (and multi-cutoff `choice_tail`) policy (len(tiers) - 1 values tunable via CV);
    # None -> ladder's `router.score_cutoffs` for `score_tail`, or `choice_cdf_threshold` for `choice_tail`
    score_cutoffs: tuple[float, ...] | None = None
    score_frontier_tail_threshold: float = 0.35
    needs_precision_threshold: float = 0.65
    # Cumulative probability threshold for standalone `choice_tail` policy when `score_cutoffs` is None
    choice_cdf_threshold: float = 0.70
    # Optional calibrated threshold from `calibrate_conformal_cdf_threshold` for `choice_crc`
    conformal_cdf_threshold: float | None = None
    # Compute trade-off lambda for `noul_eu` policy: utility = p_hat_{i,t} - lambda * u_hat_{i,t}
    eu_lambda: float = 6.0
    # Optional per-tier Platt scaling applied before PAVA in `noul_eu`
    noul_platt_params: dict[TierID, PlattParams] | None = None
    expected_output_tokens_table: dict[str, dict[str, float]] | None = None

    def __post_init__(self) -> None:
        if self.tier_order is not None:
            _ = self.resolved_tier_order

    @property
    def resolved_ladder(self) -> TierLadder:
        return self.ladder or default_ladder()

    @property
    def resolved_tier_order(self) -> tuple[TierID, ...]:
        lad = self.resolved_ladder
        if self.tier_order is None:
            return lad.ordered_tier_ids
        order = tuple(self.tier_order)
        if len(order) != len(lad) or set(order) != set(lad.ordered_tier_ids):
            raise ValueError(
                f"tier_order {order!r} must be a permutation of ladder tiers {lad.ordered_tier_ids!r}"
            )
        return order

    @property
    def resolved_fallback_tier_id(self) -> TierID:
        return self.fallback_tier_id or self.resolved_ladder.fallback_tier_id

    @property
    def resolved_score_cutoffs(self) -> tuple[float, ...]:
        lad = self.resolved_ladder
        cutoffs = tuple(self.score_cutoffs) if self.score_cutoffs is not None else lad.score_cutoffs
        if len(cutoffs) != len(lad) - 1:
            raise ValueError(
                f"score_cutoffs needs {len(lad) - 1} values for a {len(lad)}-tier ladder, got {len(cutoffs)}"
            )
        return cutoffs

    def estimate_compute_units(self, tier_id: TierID, known_input_tokens: int) -> float:
        """Length-aware expected compute units for running `tier_id` under this config."""
        return estimate_tier_compute_units(
            tier_id=tier_id,
            known_input_tokens=known_input_tokens,
            slice_name=self.slice_name,
            expected_output_tokens_table=self.expected_output_tokens_table,
            ladder=self.resolved_ladder,
        )


def build_score_questions(ladder: TierLadder | None = None) -> dict[str, dict[str, Any]]:
    """Build the N-level ordered `score` question (one level per tier) + `needs_precision` `noul` probe.

    Note: While the launch-day AI Studio Markdown docs show `questions` as a `list` with a `"question"`
    field, `google-genai==3.2.0` (`_gaos/types/decisions/`) requires `questions` to be a
    `dict[str, Question]` where each question uses `"instructions"`.
    """
    return {
        "complexity_tier": {
            "type": "score",
            "instructions": (
                "What level of reasoning depth and model capability is required to solve "
                "this user request accurately on the first attempt?"
            ),
            "criteria": [t.capability_true_description for t in (ladder or default_ladder()).tiers],
        },
        "needs_precision": {
            "type": "noul",
            "instructions": (
                "Does a correct response require exact numerical calculation, formal proof, "
                "or multi-step code verification?"
            ),
            "criteria": {
                "true_description": (
                    "Requires exact arithmetic, multi-step code execution correctness, or "
                    "formal deduction where an approximate answer fails."
                ),
                "false_description": (
                    "Allows qualitative summary, direct lookup, or straightforward extraction."
                ),
            },
        },
    }


def build_choice_questions(ladder: TierLadder | None = None) -> dict[str, dict[str, Any]]:
    """Build the N-option `choice` question over capability-described tiers."""
    return {
        "target_tier": {
            "type": "choice",
            "instructions": (
                "Which capability tier is the minimum tier required to solve this user "
                "request accurately on the first attempt?"
            ),
            "criteria": {
                t.tier_id: t.capability_true_description for t in (ladder or default_ladder()).tiers
            },
        }
    }


def solvability_question_name(tier_id: TierID) -> str:
    """Name of the per-tier `noul` solvability probe for `tier_id` (`can_<tier_id>_solve`)."""
    return f"can_{tier_id}_solve"


def build_noul_solvability_questions(ladder: TierLadder | None = None) -> dict[str, dict[str, Any]]:
    """Build one cumulative `noul` solvability probe per tier (`can_<tier_id>_solve`).

    Unlike `score` / `choice` (which ask for the *minimum* capable tier, so each tier's `can_solve`
    describes only its marginal band), each `noul` probe asks an independent cumulative solvability
    question where higher tiers strictly subsume all lower tiers.
    """
    tiers = (ladder or default_ladder()).tiers
    max_rank = max(0, len(tiers) - 1)
    questions: dict[str, dict[str, Any]] = {}
    for t in tiers:
        cumulative_true = (
            t.capability_true_description
            if t.rank == 0
            else (
                f"Solvable at or below tier rank {t.rank} (handles all simpler tasks from lower "
                f"tiers, up through: {t.capability_true_description})"
            )
        )
        questions[solvability_question_name(t.tier_id)] = {
            "type": "noul",
            "instructions": (
                f"Can tier {t.tier_id} (rank {t.rank} of {max_rank}, where higher tiers strictly "
                "subsume all lower tiers) solve this user request accurately and completely on "
                "the first attempt when given the full user input?"
            ),
            "criteria": {
                "true_description": cumulative_true,
                "false_description": t.capability_false_description,
            },
        }
    return questions


def selected_tier_confidence(ordered_probs: Sequence[float], selected_idx: int) -> float:
    """P(selected tier suffices): probability mass at or below `selected_idx` (cheapest first).

    `score` levels and `choice` options both ask for the *minimum* capable tier, so (assuming
    capability is monotone in the ladder) any tier at or above that minimum also solves it.
    """
    return min(1.0, max(0.0, sum(ordered_probs[: selected_idx + 1])))


def _select_by_cutoffs_and_frontier_tail(
    ordered_probs: Sequence[float],
    expected_val: float,
    ordered_ids: tuple[TierID, ...],
    config: RouterPolicyConfig,
) -> int:
    """Select tier index via `score_cutoffs` and escalate to frontier if tail mass exceeds threshold."""
    lad = config.resolved_ladder
    pos_of = {tid: pos for pos, tid in enumerate(ordered_ids)}
    idx = bisect.bisect_right(config.resolved_score_cutoffs, expected_val)
    frontier_idxs = [pos_of[tid] for tid in lad.frontier_tier_ids]
    frontier_tail_mass = sum(ordered_probs[i] for i in frontier_idxs)
    min_frontier_idx = min(frontier_idxs)
    if frontier_tail_mass >= config.score_frontier_tail_threshold and idx < min_frontier_idx:
        idx = min_frontier_idx
    return idx


def select_tier_from_score_distribution(
    score_val: float,
    probabilities: dict[str, float],
    needs_precision_noul: float | None,
    config: RouterPolicyConfig,
) -> tuple[TierID, float]:
    """Evaluate either `score_argmax` or `score_tail` from a `complexity_tier` answer.

    Returns `(tier_id, confidence)`; confidence is P(selected tier suffices)
    (see `selected_tier_confidence`).
    """
    lad = config.resolved_ladder
    ordered_ids = config.resolved_tier_order
    pos_of = {tid: pos for pos, tid in enumerate(ordered_ids)}
    probs_by_idx = [float(probabilities.get(str(lad.index(tid)), 0.0)) for tid in ordered_ids]

    if config.policy == "score_argmax":
        best_idx = max(range(len(probs_by_idx)), key=lambda i: probs_by_idx[i])
        return ordered_ids[best_idx], selected_tier_confidence(probs_by_idx, best_idx)

    # `score_tail` policy: use continuous expected score + frontier tail mass + precision floor
    effective_score = (
        score_val
        if ordered_ids == lad.ordered_tier_ids
        else sum(pos * p for pos, p in enumerate(probs_by_idx))
    )
    idx = _select_by_cutoffs_and_frontier_tail(probs_by_idx, effective_score, ordered_ids, config)

    # Precision floor: if needs_precision >= threshold, route to at least `precision_floor_tier`
    if (
        needs_precision_noul is not None
        and needs_precision_noul >= config.needs_precision_threshold
    ):
        idx = max(idx, pos_of[lad.precision_floor_tier_id])

    return ordered_ids[idx], selected_tier_confidence(probs_by_idx, idx)


def select_tier_from_choice_distribution(
    probabilities: dict[str, float],
    config: RouterPolicyConfig,
    *,
    raw_choice: str | None = None,
) -> tuple[TierID, float]:
    """Evaluate either `choice_argmax` or `choice_tail` from a `target_tier` choice answer.

    Returns `(tier_id, confidence)`; confidence is P(selected tier suffices)
    (see `selected_tier_confidence`).
    """
    ordered_ids = config.resolved_tier_order
    ordered_probs = [float(probabilities.get(tid, 0.0)) for tid in ordered_ids]

    if config.policy == "choice_argmax":
        if raw_choice is not None and raw_choice in ordered_ids:
            best_idx = ordered_ids.index(raw_choice)
        else:
            best_idx = max(range(len(ordered_probs)), key=lambda i: ordered_probs[i])
        return ordered_ids[best_idx], selected_tier_confidence(ordered_probs, best_idx)

    if config.policy == "choice_crc":
        threshold = (
            config.conformal_cdf_threshold
            if config.conformal_cdf_threshold is not None
            else (
                config.resolved_ladder.conformal_cdf_threshold
                if config.resolved_ladder.conformal_cdf_threshold is not None
                else config.choice_cdf_threshold
            )
        )
        return select_tier_from_cdf_threshold(probabilities, ordered_ids, threshold)

    # When `score_cutoffs` is explicitly provided (e.g. out-of-fold CV tuning), apply the exact
    # same multi-cutoff + frontier-tail cost-sensitive selector used by `score_tail`.
    if config.score_cutoffs is not None:
        exp_rank = sum(pos * p for pos, p in enumerate(ordered_probs))
        idx = _select_by_cutoffs_and_frontier_tail(ordered_probs, exp_rank, ordered_ids, config)
        return ordered_ids[idx], selected_tier_confidence(ordered_probs, idx)

    # Standalone default `choice_tail`: smallest tier k where cumulative probability >= choice_cdf_threshold
    return select_tier_from_cdf_threshold(probabilities, ordered_ids, config.choice_cdf_threshold)


def select_tier_from_noul_eu(
    raw_tier_nouls: dict[TierID, float],
    known_input_tokens: int,
    config: RouterPolicyConfig,
) -> tuple[TierID, dict[TierID, float], float]:
    """Apply PAVA isotonic projection to raw `noul` probabilities and maximize expected utility.

    Returns `(tier_id, projected_solvability_by_tier, confidence)`; confidence is the
    projected P(selected tier solves the request).
    """
    ordered_ids = config.resolved_tier_order
    raw_vec = [float(raw_tier_nouls.get(tid, 0.5)) for tid in ordered_ids]
    if config.noul_platt_params:
        raw_vec = [
            config.noul_platt_params[tid].apply(val) if tid in config.noul_platt_params else val
            for tid, val in zip(ordered_ids, raw_vec, strict=True)
        ]
    monotone_vec = pava_isotonic_increasing(raw_vec)
    projected_map: dict[TierID, float] = {
        tid: monotone_vec[idx] for idx, tid in enumerate(ordered_ids)
    }

    best_tid: TierID = ordered_ids[0]
    best_utility = float("-inf")
    for tid in ordered_ids:
        est_units = config.estimate_compute_units(tid, known_input_tokens)
        utility = projected_map[tid] - (config.eu_lambda * est_units)
        if utility > best_utility:
            best_utility = utility
            best_tid = tid

    return best_tid, projected_map, projected_map[best_tid]


# ---------------------------------------------------------------------------
# Policy families: which questions to ask, and how their answers become a tier
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TierSelection:
    """A policy's chosen tier plus the answer signals it was derived from."""

    tier_id: TierID
    # P(selected tier suffices); see `selected_tier_confidence`
    confidence: float
    raw_probabilities: dict[str, float] = field(default_factory=dict)
    solvability_probabilities: dict[TierID, float] = field(default_factory=dict)
    expected_complexity_score: float | None = None
    needs_precision_noul: float | None = None


@dataclass(frozen=True)
class PolicyFamily:
    """The Decisions API questions for a group of policies, and how their answers become a tier.

    `select(answers, known_input_tokens, config)` receives answers indexed by question name.
    If `required_answer` is set and absent from the response, the router falls back to
    `config`'s fallback tier instead of calling `select`.
    """

    build_questions: Callable[[TierLadder | None], dict[str, dict[str, Any]]]
    select: Callable[[dict[str, Any], int, RouterPolicyConfig], TierSelection]
    required_answer: str | None = None


def answers_by_name(raw_answers: Any) -> dict[str, Any]:
    """Index Decisions API answers by question name (accepts a dict, a list of answers, or None)."""
    if not raw_answers:
        return {}
    if isinstance(raw_answers, dict):
        return dict(raw_answers)
    return {getattr(a, "name", ""): a for a in raw_answers}


def _answer_probabilities(answer: Any) -> dict[str, float]:
    return {str(k): float(v) for k, v in (getattr(answer, "probabilities", {}) or {}).items()}


def _select_from_score_answers(
    answers: dict[str, Any], known_input_tokens: int, config: RouterPolicyConfig
) -> TierSelection:
    comp_ans = answers["complexity_tier"]
    expected_score = float(getattr(comp_ans, "score", 0.0) or 0.0)
    raw_probs = _answer_probabilities(comp_ans)
    needs_prec: float | None = None
    if "needs_precision" in answers:
        needs_prec = float(getattr(answers["needs_precision"], "noul", 0.0) or 0.0)
    tier_id, conf = select_tier_from_score_distribution(
        expected_score, raw_probs, needs_prec, config
    )
    return TierSelection(
        tier_id,
        conf,
        raw_probabilities=raw_probs,
        expected_complexity_score=expected_score,
        needs_precision_noul=needs_prec,
    )


def _select_from_choice_answers(
    answers: dict[str, Any], known_input_tokens: int, config: RouterPolicyConfig
) -> TierSelection:
    target_ans = answers["target_tier"]
    raw_probs = _answer_probabilities(target_ans)
    raw_choice_val = getattr(target_ans, "choice", None)
    raw_choice = str(raw_choice_val) if raw_choice_val is not None else None
    tier_id, conf = select_tier_from_choice_distribution(raw_probs, config, raw_choice=raw_choice)
    return TierSelection(tier_id, conf, raw_probabilities=raw_probs)


def _select_from_noul_answers(
    answers: dict[str, Any], known_input_tokens: int, config: RouterPolicyConfig
) -> TierSelection:
    # A missing (or null) per-tier probe counts as an uninformative 0.5
    raw_nouls: dict[TierID, float] = {}
    for tid in config.resolved_ladder.ordered_tier_ids:
        noul = getattr(answers.get(solvability_question_name(tid)), "noul", None)
        raw_nouls[tid] = 0.5 if noul is None else float(noul)
    tier_id, projected, conf = select_tier_from_noul_eu(raw_nouls, known_input_tokens, config)
    return TierSelection(
        tier_id, conf, raw_probabilities=dict(raw_nouls), solvability_probabilities=projected
    )


_SCORE_FAMILY = PolicyFamily(build_score_questions, _select_from_score_answers, "complexity_tier")
_CHOICE_FAMILY = PolicyFamily(build_choice_questions, _select_from_choice_answers, "target_tier")
_NOUL_FAMILY = PolicyFamily(build_noul_solvability_questions, _select_from_noul_answers)

# Every `PolicyMode` maps to the family that implements it (checked by the test suite).
POLICY_FAMILIES: dict[PolicyMode, PolicyFamily] = {
    "score_argmax": _SCORE_FAMILY,
    "score_tail": _SCORE_FAMILY,
    "choice_argmax": _CHOICE_FAMILY,
    "choice_tail": _CHOICE_FAMILY,
    "choice_crc": _CHOICE_FAMILY,
    "noul_eu": _NOUL_FAMILY,
}


def _is_safety_api_error(exc: Exception) -> tuple[bool, str]:
    """Detect whether an exception from `client.decisions.create` is a safety/content block."""
    msg = str(exc).lower()
    for code in SAFETY_ERROR_CODES:
        if code in msg:
            return True, f"http_error:{code}"
    if (
        isinstance(exc, genai_errors.APIError)
        and getattr(exc, "code", None) == 400
        and ("block" in msg or "safety" in msg or "prohibited" in msg)
    ):
        return True, "http_400_safety"
    return False, ""


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("true", "1")


def create_genai_client(
    api_key: str | None = None,
    *,
    enterprise: bool | None = None,
    vertexai: bool | None = None,
    project: str | None = None,
    location: str | None = None,
    api_version: str = "v1beta",
) -> genai.Client:
    """Build a configured `genai.Client` supporting AI Studio and Vertex AI / Agent Platform (ADC).

    Precedence:
    1. Explicit `api_key` argument, or `GEMINI_API_KEY`, or `GOOGLE_API_KEY`.
    2. Vertex AI / Enterprise ADC: If `vertexai=True` / `GOOGLE_GENAI_USE_VERTEXAI=true`,
       `enterprise=True` / `GOOGLE_GENAI_USE_ENTERPRISE=true`, or `GOOGLE_CLOUD_PROJECT` is set
       without an API key.
    3. Location defaults to 'global' (or `GOOGLE_CLOUD_LOCATION`) when Vertex/Enterprise is active.
    """
    resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    use_vertex = vertexai if vertexai is not None else _env_truthy("GOOGLE_GENAI_USE_VERTEXAI")
    use_enterprise = (
        enterprise if enterprise is not None else _env_truthy("GOOGLE_GENAI_USE_ENTERPRISE")
    )
    if (
        not use_vertex
        and not use_enterprise
        and not resolved_key
        and os.environ.get("GOOGLE_CLOUD_PROJECT")
    ):
        use_enterprise = True

    is_cloud = use_vertex or use_enterprise
    resolved_project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    resolved_location = (
        location or os.environ.get("GOOGLE_CLOUD_LOCATION") or ("global" if is_cloud else None)
    )

    client_kwargs: dict[str, Any] = {"http_options": {"api_version": api_version}}

    if is_cloud:
        return _build_cloud_client(
            api_version,
            use_vertex=use_vertex,
            location=resolved_location,
            project=resolved_project,
            api_key=resolved_key,
        )

    if resolved_key:
        client_kwargs["api_key"] = resolved_key
        return genai.Client(**client_kwargs)

    raise OSError(
        "Missing credentials: set GEMINI_API_KEY or GOOGLE_API_KEY in your environment, "
        "or configure Vertex AI / Agent Platform (GOOGLE_GENAI_USE_VERTEXAI=true or "
        "GOOGLE_GENAI_USE_ENTERPRISE=true and GOOGLE_CLOUD_PROJECT)."
    )


def _build_cloud_client(
    api_version: str,
    *,
    use_vertex: bool,
    location: str | None,
    project: str | None,
    api_key: str | None = None,
) -> genai.Client:
    """Build a Vertex AI or Agent Platform (`enterprise=True`) GenAI client."""
    kwargs: dict[str, Any] = {"http_options": {"api_version": api_version}}
    if use_vertex:
        kwargs["vertexai"] = True
    else:
        kwargs["enterprise"] = True
    if location:
        kwargs["location"] = location
    if project:
        kwargs["project"] = project
    if api_key:
        kwargs["api_key"] = api_key
    return genai.Client(**kwargs)


def create_downstream_genai_client(
    api_key: str | None = None,
    *,
    api_version: str = "v1beta",
) -> genai.Client | None:
    """Build a separate Vertex AI / Enterprise ADC client for downstream calls in hybrid mode.

    When both an API key (used by `decisions.create`) and Vertex/Enterprise env vars are set,
    downstream `interactions.create` calls can route through Vertex AI / Agent Platform via ADC.
    """
    has_key = bool(api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    use_vertex = _env_truthy("GOOGLE_GENAI_USE_VERTEXAI")
    use_enterprise = _env_truthy("GOOGLE_GENAI_USE_ENTERPRISE")
    if not (has_key and (use_vertex or use_enterprise)):
        return None
    return _build_cloud_client(
        api_version,
        use_vertex=use_vertex,
        location=os.environ.get("GOOGLE_CLOUD_LOCATION") or "global",
        project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
    )


class DecisionRouter:
    """Probabilistic model router powered by `gemini-decision-1` (`POST /v1beta/decisions`)."""

    def __init__(
        self,
        config: RouterPolicyConfig | None = None,
        *,
        api_key: str | None = None,
        client: Any = None,
        downstream_client: Any = None,
        observer: RouterObserver | None = None,
        ladder: TierLadder | None = None,
    ) -> None:
        self.config = config or RouterPolicyConfig()
        if ladder is not None:
            if config is not None and config.ladder is not None and config.ladder is not ladder:
                raise ValueError(
                    "Pass ladder via either RouterPolicyConfig(ladder=...) or DecisionRouter(ladder=...), not both."
                )
            self.config = dataclasses.replace(self.config, ladder=ladder)
        self.ladder = self.config.resolved_ladder
        # Fail fast on config/ladder mismatches (unknown policy or fallback tier, wrong cutoff count).
        if self.config.policy not in POLICY_FAMILIES:
            raise ValueError(
                f"Unknown router policy: {self.config.policy!r} "
                f"(expected one of {list(POLICY_FAMILIES)})"
            )
        self.ladder.tier(self.config.resolved_fallback_tier_id)
        _ = self.config.resolved_score_cutoffs
        _ = self.config.resolved_tier_order
        self._questions = self._family.build_questions(self.ladder)
        self.observer = observer
        if client is not None:
            self.client = client
            self.downstream_client = downstream_client if downstream_client is not None else client
        else:
            self.client = create_genai_client(api_key=api_key)
            self.downstream_client = (
                downstream_client
                if downstream_client is not None
                else (create_downstream_genai_client(api_key=api_key) or self.client)
            )

    @property
    def _family(self) -> PolicyFamily:
        return POLICY_FAMILIES[self.config.policy]

    def _decision_for(
        self,
        selection: TierSelection | None,
        *,
        latency_ms: float,
        known_input_tokens: int,
        usage: Usage,
        fallback_reason: str | None = None,
    ) -> RouteDecision:
        """Assemble a `RouteDecision`; `selection=None` routes to the fallback tier (confidence 0)."""
        fallback_triggered = selection is None
        sel = (
            selection
            if selection is not None
            else TierSelection(self.config.resolved_fallback_tier_id, 0.0)
        )
        tier_spec = self.ladder.tier(sel.tier_id)
        return RouteDecision(
            selected_tier_id=tier_spec.tier_id,
            model=tier_spec.model,
            thinking_level=tier_spec.thinking_level,
            policy_name=self.config.policy,
            confidence=sel.confidence,
            router_latency_ms=latency_ms,
            router_input_tokens=usage.input,
            router_compute_units=calculate_compute_units(
                self.ladder.decision_model,
                usage,
                ladder=self.ladder,
            ),
            estimated_downstream_compute_units=self.config.estimate_compute_units(
                tier_spec.tier_id, known_input_tokens
            ),
            raw_probabilities=sel.raw_probabilities,
            solvability_probabilities=sel.solvability_probabilities,
            expected_complexity_score=sel.expected_complexity_score,
            needs_precision_noul=sel.needs_precision_noul,
            fallback_triggered=fallback_triggered,
            fallback_reason=fallback_reason,
        )

    def _parse_response_with_tokens(
        self,
        response: Any,
        latency_ms: float,
        known_input_tokens: int,
        *,
        is_uncompressed: bool = False,
    ) -> tuple[RouteDecision, int]:
        family = self._family
        usage = Usage.from_response(response)
        effective_in_tokens = (
            usage.input if (is_uncompressed and usage.input > 0) else known_input_tokens
        )
        answers = answers_by_name(getattr(response, "answers", None))

        # Check in-band prompt_feedback safety block or missing answers
        prompt_fb = getattr(response, "prompt_feedback", None)
        block_reason = getattr(prompt_fb, "block_reason", None) if prompt_fb else None
        reason: str | None = None
        if block_reason:
            reason = f"in_band_block:{block_reason}"
        elif not answers:
            reason = "empty_answers"
        elif family.required_answer and answers.get(family.required_answer) is None:
            reason = f"missing_{family.required_answer}"

        selection = None if reason else family.select(answers, effective_in_tokens, self.config)
        return (
            self._decision_for(
                selection,
                latency_ms=latency_ms,
                known_input_tokens=effective_in_tokens,
                usage=usage,
                fallback_reason=reason,
            ),
            effective_in_tokens,
        )

    def _prepare_route_request(
        self, input_payload: str | list[dict[str, Any]]
    ) -> tuple[str | list[dict[str, Any]], dict[str, dict[str, Any]], int, bool]:
        """Prepare compressed router payload, question dictionary, fallback token estimate, and uncompressed flag."""
        router_input, est_in_tokens, is_uncompressed = prepare_router_envelope(
            input_payload,
            compress_media=self.config.compress_media_for_router,
            image_resolution=self.config.router_image_resolution,
            pdf_mode=self.config.router_pdf_mode,
        )
        return router_input, self._questions, est_in_tokens, is_uncompressed

    def _fail_route(self, exc: Exception, t0: float, known_in_tokens: int) -> RouteDecision:
        """Convert safety HTTP errors into a finalized fallback `RouteDecision`, or re-raise unexpected errors."""
        latency_ms = (time.perf_counter() - t0) * 1000.0
        is_safety, reason = _is_safety_api_error(exc)
        if not is_safety:
            raise exc
        decision = self._decision_for(
            None,
            latency_ms=latency_ms,
            known_input_tokens=known_in_tokens,
            usage=Usage(),
            fallback_reason=reason,
        )
        return self._finalize_decision(decision, known_in_tokens)

    def _complete_route(
        self, resp: Any, t0: float, est_in_tokens: int, is_uncompressed: bool
    ) -> RouteDecision:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        decision, effective_in_tokens = self._parse_response_with_tokens(
            resp, latency_ms, est_in_tokens, is_uncompressed=is_uncompressed
        )
        return self._finalize_decision(decision, effective_in_tokens)

    def _finalize_decision(self, decision: RouteDecision, known_in_tokens: int) -> RouteDecision:
        if self.observer is not None:
            self.observer.record_decision(decision, known_input_tokens=known_in_tokens)
        return decision

    def route(self, input_payload: str | list[dict[str, Any]]) -> RouteDecision:
        """Route a user request synchronously using `gemini-decision-1`."""
        router_input, questions, est_in_tokens, is_uncompressed = self._prepare_route_request(
            input_payload
        )
        t0 = time.perf_counter()
        try:
            resp = self.client.decisions.create(
                model=self.ladder.decision_model,
                input=cast(Any, router_input),
                questions=cast(Any, questions),
            )
        except Exception as exc:
            return self._fail_route(exc, t0, est_in_tokens)
        return self._complete_route(resp, t0, est_in_tokens, is_uncompressed)

    async def aroute(self, input_payload: str | list[dict[str, Any]]) -> RouteDecision:
        """Route a user request asynchronously using `client.aio.decisions.create`."""
        router_input, questions, est_in_tokens, is_uncompressed = self._prepare_route_request(
            input_payload
        )
        t0 = time.perf_counter()
        try:
            resp = await self.client.aio.decisions.create(
                model=self.ladder.decision_model,
                input=cast(Any, router_input),
                questions=cast(Any, questions),
            )
        except Exception as exc:
            return self._fail_route(exc, t0, est_in_tokens)
        return self._complete_route(resp, t0, est_in_tokens, is_uncompressed)

    def _build_execution_kwargs(
        self,
        decision: RouteDecision,
        input_payload: str | list[dict[str, Any]],
        *,
        system_instruction: str | None = None,
        safety_settings: list[dict[str, str]] | None = None,
        generation_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        gen_config: dict[str, Any] = {"thinking_level": decision.thinking_level}
        if generation_config:
            gen_config.update(generation_config)
        kwargs: dict[str, Any] = {
            "model": decision.model,
            "input": input_payload,
            "generation_config": gen_config,
        }
        if safety_settings is not None:
            kwargs["safety_settings"] = safety_settings
        if system_instruction:
            kwargs["system_instruction"] = system_instruction
        return kwargs

    def route_and_execute(
        self,
        input_payload: str | list[dict[str, Any]],
        *,
        system_instruction: str | None = None,
        safety_settings: list[dict[str, str]] | None = None,
        generation_config: dict[str, Any] | None = None,
        on_decision: Callable[[RouteDecision], None] | None = None,
    ) -> tuple[RouteDecision, Any]:
        """Route `input_payload` with compressed media, then invoke the chosen tier with FULL media."""
        decision = self.route(input_payload)
        if on_decision is not None:
            on_decision(decision)
        kwargs = self._build_execution_kwargs(
            decision,
            input_payload,
            system_instruction=system_instruction,
            safety_settings=safety_settings,
            generation_config=generation_config,
        )
        interaction = self.downstream_client.interactions.create(**kwargs)
        return decision, interaction

    async def aroute_and_execute(
        self,
        input_payload: str | list[dict[str, Any]],
        *,
        system_instruction: str | None = None,
        safety_settings: list[dict[str, str]] | None = None,
        generation_config: dict[str, Any] | None = None,
        on_decision: Callable[[RouteDecision], None] | None = None,
    ) -> tuple[RouteDecision, Any]:
        """Async version of :meth:`route_and_execute`."""
        decision = await self.aroute(input_payload)
        if on_decision is not None:
            on_decision(decision)
        kwargs = self._build_execution_kwargs(
            decision,
            input_payload,
            system_instruction=system_instruction,
            safety_settings=safety_settings,
            generation_config=generation_config,
        )
        interaction = await self.downstream_client.aio.interactions.create(**kwargs)
        return decision, interaction
