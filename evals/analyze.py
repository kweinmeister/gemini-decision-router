"""Stage 3: 5-fold stratified CV, Leave-One-Replicate-Out (LORO) Oracle, ECE/Brier/LogLoss, Q4 fidelity, and paired bootstrap."""

from __future__ import annotations

import argparse
import bisect
import collections
import dataclasses
import functools
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import BaseModel, ConfigDict, ValidationError

from evals._api import add_tiers_argument, read_jsonl
from evals._metrics import (
    build_net_value_summary,
    compute_loro_oracle_for_item,
    compute_static_pareto_hull,
    interpolate_hull_min_x_at_accuracy,
    paired_bootstrap_ci,
    profile_net_value_block,
    stratified_k_folds,
    summarize_oof_arm,
    weighted_quantile,
)
from gemini_decision_router.calibration import fit_per_tier_platt_scaling
from gemini_decision_router.conformal import calibrate_conformal_cdf_threshold
from gemini_decision_router.ladder import TierLadder, default_ladder, resolve_ladder
from gemini_decision_router.models import TierID
from gemini_decision_router.observability import (
    compute_calibration_metrics,
    jensen_shannon_divergence,
    spearman_rank_correlation,
)
from gemini_decision_router.router import (
    RouterPolicyConfig,
    select_tier_from_choice_distribution,
    select_tier_from_noul_eu,
    select_tier_from_score_distribution,
    solvability_question_name,
)

__all__ = [
    "DEFAULT_EU_LAMBDA",
    "DEFAULT_LAMBDA_GRID",
    "REPO_ROOT",
    "TRAFFIC_WEIGHT_PROFILES",
    "build_net_value_summary",
    "compute_calibration_metrics",
    "compute_loro_oracle_for_item",
    "compute_static_pareto_hull",
    "fit_expected_output_tokens_table",
    "fit_training_tier_mean_cost",
    "interpolate_hull_min_x_at_accuracy",
    "paired_bootstrap_ci",
    "profile_net_value_block",
    "run_analysis",
    "stratified_k_folds",
    "summarize_oof_arm",
    "weighted_quantile",
]

REPO_ROOT = Path(__file__).resolve().parent.parent

# Realistic production traffic weights: 60% s1, 25% s2, 5% s4, 5% s5 (10% multimodal), 5% s3
TRAFFIC_WEIGHT_PROFILES: dict[str, dict[str, float]] = {
    "balanced": {
        "s1_trivial_text": 0.20,
        "s2_everyday_code": 0.20,
        "s3_frontier_reasoning": 0.20,
        "s4_visual_and_pdf": 0.20,
        "s5_spoken_audio": 0.20,
    },
    "realistic_skew": {
        "s1_trivial_text": 0.60,
        "s2_everyday_code": 0.25,
        "s3_frontier_reasoning": 0.05,
        "s4_visual_and_pdf": 0.05,
        "s5_spoken_audio": 0.05,
    },
}

# Spans zero-regret (2.00 units/correct) through balanced (0.17 units/correct) to ultra-frugal (0.003 units/correct)
DEFAULT_LAMBDA_GRID: tuple[float, ...] = (
    0.0,
    0.5,
    1.5,
    3.0,
    6.0,
    12.0,
    25.0,
    50.0,
    120.0,
    300.0,
)
DEFAULT_EU_LAMBDA: float = 6.0


def fit_expected_output_tokens_table(
    train_ids: set[str],
    matrix_by_item_tier: dict[str, dict[str, list[dict[str, Any]]]],
    item_slice_map: dict[str, str],
    tier_ids: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Fit mean weighted output+thought token counts per `(slice, tier_id)` strictly on the training fold."""
    sums: dict[str, dict[str, float]] = collections.defaultdict(
        lambda: collections.defaultdict(float)
    )
    counts: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: collections.defaultdict(int)
    )
    for iid in train_ids:
        sl = item_slice_map[iid]
        for tid in tier_ids:
            for rep in matrix_by_item_tier[iid][tid]:
                toks = float(rep.get("output_tokens", 0)) + float(rep.get("thought_tokens", 0))
                sums[sl][tid] += toks
                counts[sl][tid] += 1
                sums["default"][tid] += toks
                counts["default"][tid] += 1

    return {
        sl: {tid: (tmap[tid] / max(1, counts[sl][tid])) for tid in tier_ids}
        for sl, tmap in sums.items()
    }


def fit_training_tier_mean_cost(
    train_ids: set[str],
    matrix_by_item_tier: dict[str, dict[str, list[dict[str, Any]]]],
    cost_key: str,
    tier_ids: Sequence[str],
) -> dict[str, float]:
    """Compute measured mean downstream cost per tier strictly on the training fold."""
    return {
        tid: (
            sum(float(rep[cost_key]) for iid in train_ids for rep in matrix_by_item_tier[iid][tid])
            / max(1, sum(len(matrix_by_item_tier[iid][tid]) for iid in train_ids))
        )
        for tid in tier_ids
    }


def _item_downstream_metrics(
    matrix_by_item_tier: dict[str, dict[str, list[dict[str, Any]]]],
    iid: str,
    tid: str,
    cost_key: str,
) -> tuple[float, float, float]:
    """Return `(mean_pass_rate, mean_downstream_cost, mean_downstream_latency_ms)` across replicates."""
    reps = matrix_by_item_tier[iid][tid]
    if not reps:
        return 0.0, 0.0, 0.0
    acc = sum(1.0 if r["passed"] else 0.0 for r in reps) / len(reps)
    cost = sum(float(r[cost_key]) for r in reps) / len(reps)
    lat = sum(float(r.get("latency_ms", 0.0)) for r in reps) / len(reps)
    return acc, cost, lat


def _select_heuristic_tier(
    toks: int,
    modality: str,
    tok_low: int,
    tok_high: int | None,
    multimodal_min_frac: float,
    *,
    tier_ids: Sequence[str],
) -> TierID:
    """Ladder-size-agnostic token + modality heuristic covering all `0 .. len(tier_ids) - 1` ranks."""
    n = len(tier_ids)
    if n <= 1:
        return tier_ids[0]
    if tok_high is None:
        idx = 0 if toks < tok_low else min(n - 1, 1)
    elif n == 2:
        idx = 0 if toks < tok_low else 1
    else:
        ratio = max(1.05, float(tok_high) / max(1.0, float(tok_low)))
        cutoffs = [tok_low * (ratio ** (k / (n - 2))) for k in range(n - 1)]
        idx = bisect.bisect_right(cutoffs, toks)
    if modality in ("image", "pdf", "audio", "video"):
        min_mm_idx = min(n - 1, round(multimodal_min_frac * (n - 1)))
        idx = max(idx, min_mm_idx)
    return tier_ids[min(n - 1, idx)]


def _build_score_cutoff_candidates(
    ladder: TierLadder,
) -> list[tuple[tuple[float, ...], float, float]]:
    """Build a compact, decoupled cutoff grid so mid-tier and frontier thresholds tune independently."""
    n_cut = len(ladder.ordered_tier_ids) - 1
    candidates: list[tuple[tuple[float, ...], float, float]] = [
        (ladder.score_cutoffs, tail, prec) for tail, prec in ((0.35, 0.65), (0.45, 0.85))
    ]
    if n_cut <= 0:
        return candidates
    if n_cut == 1:
        cut_sets = [(c,) for c in (0.35, 0.55, 0.80, 1.05)]
    elif n_cut == 2:
        cut_sets = [
            (c0, c1) for c0 in (0.45, 0.75, 1.15) for c1 in (1.35, 1.75, 2.15) if c0 + 0.15 < c1
        ]
    else:
        cut_sets = []
        for c0 in (0.50, 0.80, 1.30, 1.90):
            for mid_shift in (0.0, 0.55, 0.85):
                mids = tuple(round(k + 0.5 + mid_shift, 3) for k in range(1, n_cut - 1))
                for c_top in (n_cut - 0.75, n_cut - 0.55, n_cut - 0.15, n_cut + 0.25):
                    tup = (c0, *mids, round(c_top, 3))
                    if all(tup[i] + 0.08 < tup[i + 1] for i in range(len(tup) - 1)):
                        cut_sets.append(tup)
    for cuts in dict.fromkeys(cut_sets):
        for tail, prec in ((0.25, 0.60), (0.45, 0.75), (0.75, 0.95), (1.01, 1.01)):
            candidates.append((cuts, tail, prec))
    return list(dict.fromkeys(candidates))


@dataclasses.dataclass(frozen=True)
class _ItemRouterSignals:
    """Pre-extracted per-item router signals."""

    known_tokens: int
    modality: str
    score_val: float
    score_probs: dict[str, float]
    needs_precision: float
    score_cost: float
    score_lat: float
    choice_probs: dict[str, float]
    choice_raw_winner: str | None
    choice_cost: float
    choice_lat: float
    noul_raw: dict[str, float]
    noul_cost: float
    noul_lat: float
    fl_choice_tier: str | None
    fl_tier_probs: dict[str, float]
    fl_solv_probs: dict[str, float]
    fl_cost: float
    fl_lat: float


def _extract_item_signals(
    r_rec: dict[str, Any], tier_ids: Sequence[str], cost_key: str
) -> _ItemRouterSignals:
    sc_rec = r_rec["decision_score"]
    sc_ans = sc_rec["answers"]["complexity_tier"]
    ch_rec = r_rec["decision_choice"]
    ch_ans = ch_rec["answers"]["target_tier"]
    nl_rec = r_rec["decision_noul"]
    nl_ans = nl_rec["answers"]
    fl_rec = r_rec["flash_lite_json"]
    fl_parsed = fl_rec["parsed"]

    # Use API-tokenized full-payload input count when present (e.g. from full_media_score on s4 or
    # uncompressed router input_tokens on s1/s2/s3/s5), falling back to recorded known_input_tokens.
    ablation = r_rec.get("q4_ablation", {})
    if (
        "full_media_score" in ablation
        and int(ablation["full_media_score"].get("input_tokens", 0)) > 0
    ):
        known_tokens = int(ablation["full_media_score"]["input_tokens"])
    elif r_rec.get("slice") != "s4_visual_and_pdf" and int(sc_rec.get("input_tokens", 0)) > 0:
        known_tokens = int(sc_rec["input_tokens"])
    else:
        known_tokens = int(r_rec.get("known_input_tokens", 100))

    raw_ch = ch_ans.get("choice")
    fl_choice = fl_parsed.get("choice_tier") or fl_parsed.get("target_tier")
    return _ItemRouterSignals(
        known_tokens=known_tokens,
        modality=str(r_rec.get("modality", "text")),
        score_val=float(sc_ans["score"]),
        score_probs=dict(sc_ans["probabilities"]),
        needs_precision=float(sc_rec["answers"].get("needs_precision", {}).get("noul", 0.0)),
        score_cost=float(sc_rec[cost_key]),
        score_lat=float(sc_rec.get("latency_ms", 0.0)),
        choice_probs=dict(ch_ans["probabilities"]),
        choice_raw_winner=str(raw_ch) if raw_ch is not None else None,
        choice_cost=float(ch_rec[cost_key]),
        choice_lat=float(ch_rec.get("latency_ms", 0.0)),
        noul_raw={tid: float(nl_ans[solvability_question_name(tid)]["noul"]) for tid in tier_ids},
        noul_cost=float(nl_rec[cost_key]),
        noul_lat=float(nl_rec.get("latency_ms", 0.0)),
        fl_choice_tier=str(fl_choice) if fl_choice is not None else None,
        fl_tier_probs=dict(fl_parsed["tier_probabilities"]),
        fl_solv_probs={tid: float(fl_parsed["solvability_probabilities"][tid]) for tid in tier_ids},
        fl_cost=float(fl_rec[cost_key]),
        fl_lat=float(fl_rec.get("latency_ms", 0.0)),
    )


class _CascadeOutcome(NamedTuple):
    selected_tier: str
    accuracy: float
    router_compute_units: float
    downstream_compute_units: float
    router_latency_ms: float
    downstream_latency_ms: float


def _cascade_outcome(
    *,
    iid: str,
    verifier_rec: dict[str, Any],
    matrix_by_item_tier: dict[str, dict[str, list[dict[str, Any]]]],
    cost_key: str,
    cheap_tier_id: str,
    escalation_tier_id: str,
    accept_threshold: float,
    prefill_tid: str | None = None,
    prefill_p_cheap: float = 0.0,
    prefill_cost: float = 0.0,
    prefill_lat: float = 0.0,
) -> _CascadeOutcome:
    """Compute selected tier, accuracy, and split cost/latency for pure or hybrid cascades."""
    if prefill_tid is not None and prefill_tid != cheap_tier_id:
        p_acc, p_cost, p_lat = _item_downstream_metrics(
            matrix_by_item_tier, iid, prefill_tid, cost_key
        )
        return _CascadeOutcome(prefill_tid, p_acc, prefill_cost, p_cost, prefill_lat, p_lat)

    c_acc, c_cost, c_lat = _item_downstream_metrics(
        matrix_by_item_tier, iid, cheap_tier_id, cost_key
    )
    if prefill_tid is not None and prefill_p_cheap >= accept_threshold:
        return _CascadeOutcome(cheap_tier_id, c_acc, prefill_cost, c_cost, prefill_lat, c_lat)

    v_block = verifier_rec["verifier"]
    v_answers = v_block.get("answers", {})
    v_ans = v_answers.get("draft_solves_request", v_answers.get("draft_sufficient", {}))
    v_conf = float(v_ans.get("noul", v_ans.get("probabilities", {}).get("1", 0.0)))
    v_cost = float(v_block.get(cost_key, v_block.get("compute_units", 0.0)))
    v_lat = float(v_block.get("latency_ms", 0.0))

    if v_conf >= accept_threshold:
        return _CascadeOutcome(
            cheap_tier_id,
            c_acc,
            prefill_cost + v_cost,
            c_cost,
            prefill_lat + v_lat,
            c_lat,
        )

    e_acc, e_cost, e_lat = _item_downstream_metrics(
        matrix_by_item_tier, iid, escalation_tier_id, cost_key
    )
    return _CascadeOutcome(
        escalation_tier_id,
        e_acc,
        prefill_cost + v_cost,
        c_cost + e_cost,
        prefill_lat + v_lat,
        c_lat + e_lat,
    )


def _evaluate_arm_across_folds(
    folds: list[tuple[list[str], list[str]]],
    items_by_id: dict[str, dict[str, Any]],
    matrix_by_item_tier: dict[str, dict[str, list[dict[str, Any]]]],
    router_by_item: dict[str, dict[str, Any]],
    eu_lambda: float,
    *,
    ladder: TierLadder,
    cost_key: str = "compute_units",
    weight_profile: dict[str, float] | None = None,
    cascade_verifiers_by_item: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Run 5-fold stratified CV for `eu_lambda` and optional `weight_profile`."""
    tier_ids = ladder.ordered_tier_ids
    profile = weight_profile or TRAFFIC_WEIGHT_PROFILES["balanced"]
    cfg = functools.partial(RouterPolicyConfig, ladder=ladder)
    item_slice_map = {iid: str(it["slice"]) for iid, it in items_by_id.items()}
    signals = {
        iid: _extract_item_signals(r_rec, tier_ids, cost_key)
        for iid, r_rec in router_by_item.items()
    }
    downstream_cache: dict[tuple[str, str], tuple[float, float, float]] = {
        (iid, tid): _item_downstream_metrics(matrix_by_item_tier, iid, tid, cost_key)
        for iid, tier_map in matrix_by_item_tier.items()
        for tid in tier_map
    }
    oof_results: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)

    score_cutoff_candidates = _build_score_cutoff_candidates(ladder)
    # Exact same multi-cutoff + frontier-tail candidate optimizer shared across score_tail,
    # choice_tail, and flash_lite_tail (without the score-only `needs_precision` coordinate).
    choice_tail_candidates = list(
        dict.fromkeys((cuts, tail) for cuts, tail, _ in score_cutoff_candidates)
    )
    lambda_mult_candidates = [0.25, 0.5, 1.0, 1.5, 2.5, 4.0]
    heuristic_candidates: list[tuple[int, int | None, float]] = [
        (60, 2500, 0.67),
        (60, 4000, 0.34),
        (90, 8000, 0.34),
        (120, None, 0.34),
        (180, None, 0.0),
        (400, None, 0.0),
    ]

    for train_ids_list, test_ids_list in folds:
        train_ids = set(train_ids_list)
        out_tok_table = fit_expected_output_tokens_table(
            train_ids, matrix_by_item_tier, item_slice_map, tier_ids
        )
        train_tier_cost = fit_training_tier_mean_cost(
            train_ids, matrix_by_item_tier, cost_key, tier_ids
        )
        # Empirical monotonic tier ladder ordered strictly by measured training-fold mean cost
        empirical_tier_ids = tuple(
            sorted(tier_ids, key=lambda tid: (train_tier_cost[tid], ladder.index(tid)))
        )

        train_slice_counts: dict[str, int] = collections.Counter(
            item_slice_map[iid] for iid in train_ids_list
        )
        train_item_weights = {
            iid: profile.get(item_slice_map[iid], 0.20)
            / max(1, train_slice_counts[item_slice_map[iid]])
            for iid in train_ids_list
        }
        total_train_w = max(1e-9, sum(train_item_weights.values()))

        def _score_train_utility(
            tier_selector_fn: Any,
            tr_ids: list[str] = train_ids_list,
            w_map: dict[str, float] = train_item_weights,
            w_norm: float = total_train_w,
            ds_cache: dict[tuple[str, str], tuple[float, float, float]] = downstream_cache,
        ) -> float:
            total_u = 0.0
            for iid in tr_ids:
                tid, r_cost = tier_selector_fn(iid)
                acc, d_cost, _ = ds_cache[(iid, tid)]
                total_u += w_map[iid] * (acc - eu_lambda * (r_cost + d_cost))
            return total_u / w_norm

        # 1. Tune `decision_score_tail_cv` on training fold
        best_score_params = max(
            score_cutoff_candidates,
            key=lambda params, emp=empirical_tier_ids: _score_train_utility(
                lambda iid, p=params, eo=emp: (
                    select_tier_from_score_distribution(
                        signals[iid].score_val,
                        signals[iid].score_probs,
                        signals[iid].needs_precision,
                        cfg(
                            policy="score_tail",
                            tier_order=eo,
                            score_cutoffs=p[0],
                            score_frontier_tail_threshold=p[1],
                            needs_precision_threshold=p[2],
                        ),
                    )[0],
                    signals[iid].score_cost,
                )
            ),
        )

        # 2. Tune `decision_choice_tail_cv` on training fold (shared tail optimizer)
        best_choice_params = max(
            choice_tail_candidates,
            key=lambda params, emp=empirical_tier_ids: _score_train_utility(
                lambda iid, p=params, eo=emp: (
                    select_tier_from_choice_distribution(
                        signals[iid].choice_probs,
                        cfg(
                            policy="choice_tail",
                            tier_order=eo,
                            score_cutoffs=p[0],
                            score_frontier_tail_threshold=p[1],
                        ),
                    )[0],
                    signals[iid].choice_cost,
                )
            ),
        )

        # 3. Tune `decision_noul_eu_cv` on training fold
        best_noul_mult = max(
            lambda_mult_candidates,
            key=lambda mult, tok_tbl=out_tok_table, emp=empirical_tier_ids: _score_train_utility(
                lambda iid, m=mult, tbl=tok_tbl, eo=emp: (
                    select_tier_from_noul_eu(
                        signals[iid].noul_raw,
                        signals[iid].known_tokens,
                        cfg(
                            policy="noul_eu",
                            tier_order=eo,
                            slice_name=item_slice_map[iid],
                            eu_lambda=eu_lambda * m,
                            expected_output_tokens_table=tbl,
                        ),
                    )[0],
                    signals[iid].noul_cost,
                )
            ),
        )

        # 4. Tune `flash_lite_tail_cv` on training fold (exact same tail optimizer)
        best_fl_params = max(
            choice_tail_candidates,
            key=lambda params, emp=empirical_tier_ids: _score_train_utility(
                lambda iid, p=params, eo=emp: (
                    select_tier_from_choice_distribution(
                        signals[iid].fl_tier_probs,
                        cfg(
                            policy="choice_tail",
                            tier_order=eo,
                            score_cutoffs=p[0],
                            score_frontier_tail_threshold=p[1],
                        ),
                    )[0],
                    signals[iid].fl_cost,
                )
            ),
        )

        # 5. Tune `flash_lite_noul_eu_cv` on training fold
        best_fl_mult = max(
            lambda_mult_candidates,
            key=lambda mult, tok_tbl=out_tok_table, emp=empirical_tier_ids: _score_train_utility(
                lambda iid, m=mult, tbl=tok_tbl, eo=emp: (
                    select_tier_from_noul_eu(
                        signals[iid].fl_solv_probs,
                        signals[iid].known_tokens,
                        cfg(
                            policy="noul_eu",
                            tier_order=eo,
                            slice_name=item_slice_map[iid],
                            eu_lambda=eu_lambda * m,
                            expected_output_tokens_table=tbl,
                        ),
                    )[0],
                    signals[iid].fl_cost,
                )
            ),
        )

        # 6. Tune `token_modality_heuristic` on training fold
        best_heur_params = max(
            heuristic_candidates,
            key=lambda hp, emp=empirical_tier_ids: _score_train_utility(
                lambda iid, h=hp, eo=emp: (
                    _select_heuristic_tier(
                        signals[iid].known_tokens,
                        signals[iid].modality,
                        *h,
                        tier_ids=eo,
                    ),
                    0.0,
                )
            ),
        )

        # 7. Fit Conformal Risk Control threshold (`choice_crc`, target_alpha=0.015) on training fold
        train_choice_probs = [signals[iid].choice_probs for iid in train_ids_list]
        train_tier_accs = [
            {tid: downstream_cache[(iid, tid)][0] for tid in empirical_tier_ids}
            for iid in train_ids_list
        ]
        crc_fit = calibrate_conformal_cdf_threshold(
            train_choice_probs,
            train_tier_accs,
            empirical_tier_ids,
            target_alpha=0.015,
            weights=[train_item_weights[iid] for iid in train_ids_list],
        )

        # 8. Fit per-tier Platt scaling strictly on training fold for `decision_noul_platt_eu_cv`
        platt_probs_by_tier = {
            tid: [signals[iid].noul_raw.get(tid, 0.5) for iid in train_ids_list]
            for tid in empirical_tier_ids
        }
        platt_labels_by_tier = {
            tid: [downstream_cache[(iid, tid)][0] for iid in train_ids_list]
            for tid in empirical_tier_ids
        }
        fold_platt_params = fit_per_tier_platt_scaling(platt_probs_by_tier, platt_labels_by_tier)
        best_noul_platt_mult = max(
            lambda_mult_candidates,
            key=lambda mult, tok_tbl=out_tok_table, emp=empirical_tier_ids, pp=fold_platt_params: (
                _score_train_utility(
                    lambda iid, m=mult, tbl=tok_tbl, eo=emp, platt=pp: (
                        select_tier_from_noul_eu(
                            signals[iid].noul_raw,
                            signals[iid].known_tokens,
                            cfg(
                                policy="noul_eu",
                                tier_order=eo,
                                slice_name=item_slice_map[iid],
                                eu_lambda=eu_lambda * m,
                                noul_platt_params=platt,
                                expected_output_tokens_table=tbl,
                            ),
                        )[0],
                        signals[iid].noul_cost,
                    )
                )
            ),
        )

        # 9. Optional Cascade & Hybrid Cascade-Router threshold tuning on training fold
        has_cascade = bool(
            cascade_verifiers_by_item
            and all(iid in cascade_verifiers_by_item for iid in train_ids_list)
            and all(iid in cascade_verifiers_by_item for iid in test_ids_list)
        )
        cascade_thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]
        best_pure_cascade_thresh = 0.70
        best_hybrid_cascade_thresh = 0.70
        if has_cascade and cascade_verifiers_by_item is not None:
            cheap_tid = tier_ids[0]
            esc_tid = tier_ids[min(1, len(tier_ids) - 1)]

            def _score_cascade_thresh(
                thresh: float,
                *,
                hybrid: bool,
                tr_ids: list[str] = train_ids_list,
                eo: tuple[str, ...] = empirical_tier_ids,
                noul_mult: float = best_noul_mult,
                tok_tbl: Any = out_tok_table,
                c_tid: str = cheap_tid,
                e_tid: str = esc_tid,
                w_map: dict[str, float] = train_item_weights,
                w_norm: float = total_train_w,
                verifiers: dict[str, dict[str, Any]] = cascade_verifiers_by_item,
            ) -> float:
                total_u = 0.0
                for iid in tr_ids:
                    prefill_tid: str | None = None
                    p_cheap = 0.0
                    p_cost = 0.0
                    p_lat = 0.0
                    if hybrid:
                        sig = signals[iid]
                        prefill_tid, _, p_cheap = select_tier_from_noul_eu(
                            sig.noul_raw,
                            sig.known_tokens,
                            cfg(
                                policy="noul_eu",
                                tier_order=eo,
                                slice_name=item_slice_map[iid],
                                eu_lambda=eu_lambda * noul_mult,
                                expected_output_tokens_table=tok_tbl,
                            ),
                        )
                        p_cost = sig.noul_cost
                        p_lat = sig.noul_lat
                    outcome = _cascade_outcome(
                        iid=iid,
                        verifier_rec=verifiers[iid],
                        matrix_by_item_tier=matrix_by_item_tier,
                        cost_key=cost_key,
                        cheap_tier_id=c_tid,
                        escalation_tier_id=e_tid,
                        accept_threshold=thresh,
                        prefill_tid=prefill_tid,
                        prefill_p_cheap=p_cheap,
                        prefill_cost=p_cost,
                        prefill_lat=p_lat,
                    )
                    total_cost = outcome.router_compute_units + outcome.downstream_compute_units
                    total_u += w_map[iid] * (outcome.accuracy - eu_lambda * total_cost)
                return total_u / w_norm

            best_pure_cascade_thresh = max(
                cascade_thresholds,
                key=lambda t: _score_cascade_thresh(t, hybrid=False),
            )
            best_hybrid_cascade_thresh = max(
                cascade_thresholds,
                key=lambda t: _score_cascade_thresh(t, hybrid=True),
            )

        # Evaluate all arms out-of-fold on `test_ids_list`
        for iid in test_ids_list:
            sig = signals[iid]
            sl = item_slice_map[iid]

            def _record(
                arm_name: str,
                tid: str,
                r_units: float,
                r_lat: float,
                cur_iid: str = iid,
                sl_val: str = sl,
            ) -> None:
                acc, d_units, d_lat = _item_downstream_metrics(
                    matrix_by_item_tier, cur_iid, tid, cost_key
                )
                oof_results[arm_name][cur_iid] = {
                    "slice": sl_val,
                    "selected_tier": tid,
                    "accuracy": acc,
                    "router_compute_units": r_units,
                    "downstream_compute_units": d_units,
                    "total_compute_units": r_units + d_units,
                    "router_latency_ms": r_lat,
                    "downstream_latency_ms": d_lat,
                    "total_latency_ms": r_lat + d_lat,
                }

            for tid in tier_ids:
                _record(f"static_{tid}", tid, 0.0, 0.0)
            extra_static_tiers = sorted(
                {
                    tid
                    for tid in matrix_by_item_tier.get(iid, {})
                    if tid not in tier_ids
                    and all(tid in matrix_by_item_tier.get(i, {}) for i in items_by_id)
                }
            )
            for tid in extra_static_tiers:
                _record(f"static_{tid}", tid, 0.0, 0.0)

            h_tid = _select_heuristic_tier(
                sig.known_tokens,
                sig.modality,
                *best_heur_params,
                tier_ids=empirical_tier_ids,
            )
            _record("token_modality_heuristic", h_tid, 0.0, 0.0)

            sc_arg_tid, _ = select_tier_from_score_distribution(
                sig.score_val,
                sig.score_probs,
                sig.needs_precision,
                cfg(policy="score_argmax"),
            )
            _record("decision_score_argmax", sc_arg_tid, sig.score_cost, sig.score_lat)

            sc_tail_tid, _ = select_tier_from_score_distribution(
                sig.score_val,
                sig.score_probs,
                sig.needs_precision,
                cfg(
                    policy="score_tail",
                    tier_order=empirical_tier_ids,
                    score_cutoffs=best_score_params[0],
                    score_frontier_tail_threshold=best_score_params[1],
                    needs_precision_threshold=best_score_params[2],
                ),
            )
            _record("decision_score_tail_cv", sc_tail_tid, sig.score_cost, sig.score_lat)

            ch_arg_tid, _ = select_tier_from_choice_distribution(
                sig.choice_probs,
                cfg(policy="choice_argmax"),
                raw_choice=sig.choice_raw_winner,
            )
            _record("decision_choice_argmax", ch_arg_tid, sig.choice_cost, sig.choice_lat)

            ch_tail_tid, _ = select_tier_from_choice_distribution(
                sig.choice_probs,
                cfg(
                    policy="choice_tail",
                    tier_order=empirical_tier_ids,
                    score_cutoffs=best_choice_params[0],
                    score_frontier_tail_threshold=best_choice_params[1],
                ),
            )
            _record("decision_choice_tail_cv", ch_tail_tid, sig.choice_cost, sig.choice_lat)

            ch_crc_tid, _ = select_tier_from_choice_distribution(
                sig.choice_probs,
                cfg(
                    policy="choice_crc",
                    tier_order=empirical_tier_ids,
                    conformal_cdf_threshold=crc_fit.threshold,
                ),
            )
            _record("decision_choice_crc_cv", ch_crc_tid, sig.choice_cost, sig.choice_lat)

            noul_tid, _, noul_conf = select_tier_from_noul_eu(
                sig.noul_raw,
                sig.known_tokens,
                cfg(
                    policy="noul_eu",
                    tier_order=empirical_tier_ids,
                    slice_name=sl,
                    eu_lambda=eu_lambda * best_noul_mult,
                    expected_output_tokens_table=out_tok_table,
                ),
            )
            _record("decision_noul_eu_cv", noul_tid, sig.noul_cost, sig.noul_lat)

            noul_platt_tid, _, _ = select_tier_from_noul_eu(
                sig.noul_raw,
                sig.known_tokens,
                cfg(
                    policy="noul_eu",
                    tier_order=empirical_tier_ids,
                    slice_name=sl,
                    eu_lambda=eu_lambda * best_noul_platt_mult,
                    noul_platt_params=fold_platt_params,
                    expected_output_tokens_table=out_tok_table,
                ),
            )
            _record("decision_noul_platt_eu_cv", noul_platt_tid, sig.noul_cost, sig.noul_lat)

            fl_arg_tid = sig.fl_choice_tier or ladder.fallback_tier_id
            if fl_arg_tid not in ladder.tier_by_id:
                fl_arg_tid = ladder.fallback_tier_id
            _record("flash_lite_argmax", fl_arg_tid, sig.fl_cost, sig.fl_lat)

            fl_tail_tid, _ = select_tier_from_choice_distribution(
                sig.fl_tier_probs,
                cfg(
                    policy="choice_tail",
                    tier_order=empirical_tier_ids,
                    score_cutoffs=best_fl_params[0],
                    score_frontier_tail_threshold=best_fl_params[1],
                ),
            )
            _record("flash_lite_tail_cv", fl_tail_tid, sig.fl_cost, sig.fl_lat)

            fl_noul_tid, _, _ = select_tier_from_noul_eu(
                sig.fl_solv_probs,
                sig.known_tokens,
                cfg(
                    policy="noul_eu",
                    tier_order=empirical_tier_ids,
                    slice_name=sl,
                    eu_lambda=eu_lambda * best_fl_mult,
                    expected_output_tokens_table=out_tok_table,
                ),
            )
            _record("flash_lite_noul_eu_cv", fl_noul_tid, sig.fl_cost, sig.fl_lat)

            if has_cascade and cascade_verifiers_by_item is not None:
                cheap_tid = tier_ids[0]
                esc_tid = tier_ids[min(1, len(tier_ids) - 1)]
                v_rec = cascade_verifiers_by_item[iid]
                for arm_name, is_hybrid, thresh in (
                    ("pure_cascade_t0_t1", False, best_pure_cascade_thresh),
                    ("hybrid_cascade_router", True, best_hybrid_cascade_thresh),
                ):
                    outcome = _cascade_outcome(
                        iid=iid,
                        verifier_rec=v_rec,
                        matrix_by_item_tier=matrix_by_item_tier,
                        cost_key=cost_key,
                        cheap_tier_id=cheap_tid,
                        escalation_tier_id=esc_tid,
                        accept_threshold=thresh,
                        prefill_tid=noul_tid if is_hybrid else None,
                        prefill_p_cheap=noul_conf if is_hybrid else 0.0,
                        prefill_cost=sig.noul_cost if is_hybrid else 0.0,
                        prefill_lat=sig.noul_lat if is_hybrid else 0.0,
                    )
                    oof_results[arm_name][iid] = {
                        "slice": sl,
                        "selected_tier": outcome.selected_tier,
                        "accuracy": outcome.accuracy,
                        "router_compute_units": outcome.router_compute_units,
                        "downstream_compute_units": outcome.downstream_compute_units,
                        "total_compute_units": (
                            outcome.router_compute_units + outcome.downstream_compute_units
                        ),
                        "router_latency_ms": outcome.router_latency_ms,
                        "downstream_latency_ms": outcome.downstream_latency_ms,
                        "total_latency_ms": outcome.router_latency_ms
                        + outcome.downstream_latency_ms,
                    }

            if len(matrix_by_item_tier[iid][tier_ids[0]]) >= 2:
                loro_acc, loro_cost, loro_lat = compute_loro_oracle_for_item(
                    matrix_by_item_tier[iid],
                    cost_key=cost_key,
                    eu_lambda=eu_lambda,
                    training_tier_mean_cost=train_tier_cost,
                    tier_ids=empirical_tier_ids,
                )
            else:
                loro_acc, loro_cost, loro_lat = _item_downstream_metrics(
                    matrix_by_item_tier, iid, ladder.baseline_tier_id, cost_key
                )
            oof_results["loro_oracle"][iid] = {
                "slice": sl,
                "selected_tier": "oracle",
                "accuracy": loro_acc,
                "router_compute_units": 0.0,
                "downstream_compute_units": loro_cost,
                "total_compute_units": loro_cost,
                "router_latency_ms": 0.0,
                "downstream_latency_ms": loro_lat,
                "total_latency_ms": loro_lat,
            }

    return oof_results


def _compute_bootstrap_table(
    oof_map: dict[str, dict[str, dict[str, Any]]],
    pairs: Sequence[tuple[str, str]],
    *,
    weight_profile: dict[str, float] | None = None,
) -> dict[str, Any]:
    all_eval_ids = sorted(oof_map["decision_score_tail_cv"].keys())
    weights: list[float] | None = None
    if weight_profile is not None:
        slice_counts = collections.Counter(
            str(oof_map["decision_score_tail_cv"][iid]["slice"]) for iid in all_eval_ids
        )
        weights = [
            weight_profile.get(str(oof_map["decision_score_tail_cv"][iid]["slice"]), 0.20)
            / max(1, slice_counts[str(oof_map["decision_score_tail_cv"][iid]["slice"])])
            for iid in all_eval_ids
        ]

    out: dict[str, Any] = {}
    for cand_arm, ref_arm in pairs:
        acc_diffs = [
            float(oof_map[cand_arm][iid]["accuracy"]) - float(oof_map[ref_arm][iid]["accuracy"])
            for iid in all_eval_ids
        ]
        units_diffs = [
            (
                float(oof_map[cand_arm][iid]["total_compute_units"])
                - float(oof_map[ref_arm][iid]["total_compute_units"])
            )
            * 1000.0
            for iid in all_eval_ids
        ]
        lat_diffs = [
            float(oof_map[cand_arm][iid]["total_latency_ms"])
            - float(oof_map[ref_arm][iid]["total_latency_ms"])
            for iid in all_eval_ids
        ]
        acc_m, acc_lo, acc_hi = paired_bootstrap_ci(acc_diffs, weights=weights)
        units_m, units_lo, units_hi = paired_bootstrap_ci(units_diffs, weights=weights)
        lat_m, lat_lo, lat_hi = paired_bootstrap_ci(lat_diffs, weights=weights)
        out[f"{cand_arm}_vs_{ref_arm}"] = {
            "accuracy_diff_mean": round(acc_m, 4),
            "accuracy_diff_ci95": [round(acc_lo, 4), round(acc_hi, 4)],
            "compute_units_per_1k_diff_mean": round(units_m, 4),
            "compute_units_per_1k_diff_ci95": [round(units_lo, 4), round(units_hi, 4)],
            "total_latency_diff_mean_ms": round(lat_m, 2),
            "total_latency_diff_ci95_ms": [round(lat_lo, 2), round(lat_hi, 2)],
        }
    return out


@dataclasses.dataclass
class _Q4FidelityBucket:
    js_divs: list[float] = dataclasses.field(default_factory=list)
    top1_matches: list[float] = dataclasses.field(default_factory=list)
    ref_scores: list[float] = dataclasses.field(default_factory=list)
    comp_scores: list[float] = dataclasses.field(default_factory=list)
    acc_regrets: list[float] = dataclasses.field(default_factory=list)
    token_savings: list[float] = dataclasses.field(default_factory=list)

    def add_pair(
        self,
        *,
        iid: str,
        ref_rec: dict[str, Any],
        comp_rec: dict[str, Any],
        matrix_by_item_tier: dict[str, dict[str, list[dict[str, Any]]]],
        tier_ids: Sequence[str],
        cost_key: str,
    ) -> None:
        ref_ans = ref_rec["answers"]["complexity_tier"]
        comp_ans = comp_rec["answers"]["complexity_tier"]
        self.js_divs.append(
            jensen_shannon_divergence(ref_ans["probabilities"], comp_ans["probabilities"])
        )
        top_ref = max(ref_ans["probabilities"].items(), key=lambda kv: float(kv[1]))[0]
        top_comp = max(comp_ans["probabilities"].items(), key=lambda kv: float(kv[1]))[0]
        self.top1_matches.append(1.0 if top_comp == top_ref else 0.0)
        self.ref_scores.append(float(ref_ans["score"]))
        self.comp_scores.append(float(comp_ans["score"]))
        acc_ref, _, _ = _item_downstream_metrics(
            matrix_by_item_tier, iid, tier_ids[int(top_ref)], cost_key
        )
        acc_comp, _, _ = _item_downstream_metrics(
            matrix_by_item_tier, iid, tier_ids[int(top_comp)], cost_key
        )
        self.acc_regrets.append(acc_ref - acc_comp)
        ref_tok = max(1, int(ref_rec.get("input_tokens", 1)))
        comp_tok = int(comp_rec.get("input_tokens", ref_tok))
        self.token_savings.append((ref_tok - comp_tok) / ref_tok)

    def summarize(self, count_key: str) -> dict[str, Any]:
        n = max(1, len(self.js_divs))
        return {
            count_key: len(self.js_divs),
            "top_tier_agreement_rate": round(sum(self.top1_matches) / n, 4),
            "mean_js_divergence_bits": round(sum(self.js_divs) / n, 4),
            "spearman_rho_expected_score": round(
                spearman_rank_correlation(self.ref_scores, self.comp_scores), 4
            ),
            "mean_downstream_accuracy_regret": round(sum(self.acc_regrets) / n, 4),
            "mean_router_input_token_savings_pct": round(100.0 * sum(self.token_savings) / n, 2),
        }


def _check_data_matches_ladder(
    eval_ids: list[str],
    matrix_by_item_tier: dict[str, dict[str, list[dict[str, Any]]]],
    router_by_item: dict[str, dict[str, Any]],
    ladder: TierLadder,
) -> None:
    """Fail fast if the matrix or router telemetry was collected under a different ladder."""
    ladder_ids = set(ladder.ordered_tier_ids)
    missing_cells = [
        (iid, tid)
        for iid in eval_ids
        for tid in ladder.ordered_tier_ids
        if not matrix_by_item_tier[iid].get(tid)
    ]
    stale_router = [
        iid
        for iid in eval_ids
        if set(router_by_item[iid]["flash_lite_json"]["parsed"]["tier_probabilities"]) != ladder_ids
    ]
    if missing_cells or stale_router:
        raise ValueError(
            f"Data does not match ladder {ladder.source} (tiers {list(ladder.ordered_tier_ids)}): "
            f"{len(missing_cells)} (item, tier) cells missing from the matrix (e.g. {missing_cells[:3]}); "
            f"{len(stale_router)} router records collected with other tiers (e.g. {stale_router[:3]}). "
            "Pass the --tiers file used for routers/collect_matrix."
        )


class _PermissiveRow(BaseModel):
    model_config = ConfigDict(extra="allow")


class _MatrixRowSchema(_PermissiveRow):
    item_id: str
    tier_id: str
    passed: bool
    compute_units: float
    latency_ms: float


class _RouterArmTelemetrySchema(_PermissiveRow):
    compute_units: float
    latency_ms: float
    answers: dict[str, Any]


class _NoulTelemetrySchema(_RouterArmTelemetrySchema):
    pava_projected: dict[str, float]


class _FlashLiteJsonSchema(_PermissiveRow):
    compute_units: float
    latency_ms: float
    parsed: dict[str, Any]


class _RouterRowSchema(_PermissiveRow):
    item_id: str
    decision_score: _RouterArmTelemetrySchema
    decision_choice: _RouterArmTelemetrySchema
    decision_noul: _NoulTelemetrySchema
    flash_lite_json: _FlashLiteJsonSchema


class _CascadeVerifierRowSchema(_PermissiveRow):
    item_id: str
    verifier: _RouterArmTelemetrySchema


def _validate_jsonl_rows(
    rows: list[dict[str, Any]],
    schema: type[BaseModel],
    source_path: Path,
) -> list[dict[str, Any]]:
    """Validate loaded JSONL records at the `evals` boundary for clear schema mismatch errors."""
    for idx, row in enumerate(rows, start=1):
        try:
            schema.model_validate(row)
        except ValidationError as exc:
            raise ValueError(
                f"{source_path} row {idx} does not match expected {schema.__name__}: {exc}"
            ) from exc
    return rows


def run_analysis(
    dataset_path: Path,
    matrix_path: Path,
    router_path: Path,
    out_summary_path: Path,
    *,
    ladder: TierLadder | None = None,
    cascade_verifiers_path: Path | None = None,
) -> dict[str, Any]:
    """Run the full 5-fold stratified CV evaluation, LORO Oracle, Q3 calibration, and Q4 fidelity check."""
    lad = ladder or default_ladder()
    tier_ids = lad.ordered_tier_ids
    items = read_jsonl(dataset_path)
    items_by_id = {str(it["id"]): it for it in items}

    matrix_rows = _validate_jsonl_rows(read_jsonl(matrix_path), _MatrixRowSchema, matrix_path)
    matrix_by_item_tier: dict[str, dict[str, list[dict[str, Any]]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for rec in matrix_rows:
        matrix_by_item_tier[rec["item_id"]][rec["tier_id"]].append(rec)

    router_rows = _validate_jsonl_rows(read_jsonl(router_path), _RouterRowSchema, router_path)
    router_by_item = {rec["item_id"]: rec for rec in router_rows}
    cascade_verifiers_by_item: dict[str, dict[str, Any]] | None = None
    if cascade_verifiers_path is not None and cascade_verifiers_path.exists():
        verifier_rows = _validate_jsonl_rows(
            read_jsonl(cascade_verifiers_path),
            _CascadeVerifierRowSchema,
            cascade_verifiers_path,
        )
        cascade_verifiers_by_item = {str(rec["item_id"]): rec for rec in verifier_rows}

    eval_items = [
        it
        for it in items
        if str(it["id"]) in router_by_item and str(it["id"]) in matrix_by_item_tier
    ]
    _check_data_matches_ladder(
        [str(it["id"]) for it in eval_items], matrix_by_item_tier, router_by_item, lad
    )
    folds = stratified_k_folds(eval_items, k=5, seed=20261111)
    oracle_available = bool(
        eval_items
        and min(len(matrix_by_item_tier[str(it["id"])][tier_ids[0]]) for it in eval_items) >= 2
    )

    bootstrap_pairs = list(
        dict.fromkeys(
            [
                ("decision_score_argmax", f"static_{lad.baseline_tier_id}"),
                ("decision_score_argmax", "flash_lite_argmax"),
                ("decision_score_tail_cv", f"static_{lad.baseline_tier_id}"),
                ("decision_score_tail_cv", f"static_{lad.precision_floor_tier_id}"),
                ("decision_score_tail_cv", "token_modality_heuristic"),
                ("decision_score_tail_cv", "flash_lite_argmax"),
                ("decision_score_tail_cv", "flash_lite_tail_cv"),
                ("decision_noul_eu_cv", "flash_lite_noul_eu_cv"),
                ("decision_score_tail_cv", "decision_choice_tail_cv"),
            ]
        )
    )
    sweep_arm_keys: tuple[str, ...] = (
        "decision_score_tail_cv",
        "decision_choice_tail_cv",
        "decision_choice_crc_cv",
        "decision_noul_eu_cv",
        "decision_noul_platt_eu_cv",
        "flash_lite_tail_cv",
        "flash_lite_noul_eu_cv",
        "loro_oracle",
    )
    if cascade_verifiers_by_item:
        sweep_arm_keys = (
            *sweep_arm_keys,
            "pure_cascade_t0_t1",
            "hybrid_cascade_router",
        )

    default_cost_key = "compute_units"
    oof_balanced = _evaluate_arm_across_folds(
        folds,
        items_by_id,
        matrix_by_item_tier,
        router_by_item,
        eu_lambda=DEFAULT_EU_LAMBDA,
        ladder=lad,
        cost_key=default_cost_key,
        weight_profile=TRAFFIC_WEIGHT_PROFILES["balanced"],
        cascade_verifiers_by_item=cascade_verifiers_by_item,
    )
    oof_skew = _evaluate_arm_across_folds(
        folds,
        items_by_id,
        matrix_by_item_tier,
        router_by_item,
        eu_lambda=DEFAULT_EU_LAMBDA,
        ladder=lad,
        cost_key=default_cost_key,
        weight_profile=TRAFFIC_WEIGHT_PROFILES["realistic_skew"],
        cascade_verifiers_by_item=cascade_verifiers_by_item,
    )

    balanced_arms = {
        arm: {
            "arm_name": arm,
            **summarize_oof_arm(imap, TRAFFIC_WEIGHT_PROFILES["balanced"]),
        }
        for arm, imap in oof_balanced.items()
    }
    skewed_arms = {
        arm: {
            "arm_name": arm,
            **summarize_oof_arm(imap, TRAFFIC_WEIGHT_PROFILES["realistic_skew"]),
        }
        for arm, imap in oof_skew.items()
    }

    bootstrap_balanced = _compute_bootstrap_table(oof_balanced, bootstrap_pairs)
    bootstrap_skew = _compute_bootstrap_table(
        oof_skew,
        bootstrap_pairs,
        weight_profile=TRAFFIC_WEIGHT_PROFILES["realistic_skew"],
    )

    pareto_sweep: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    pareto_sweep_skew: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for lam in DEFAULT_LAMBDA_GRID:
        oof_lam_bal = _evaluate_arm_across_folds(
            folds,
            items_by_id,
            matrix_by_item_tier,
            router_by_item,
            eu_lambda=lam,
            ladder=lad,
            cost_key=default_cost_key,
            weight_profile=TRAFFIC_WEIGHT_PROFILES["balanced"],
            cascade_verifiers_by_item=cascade_verifiers_by_item,
        )
        oof_lam_skew = _evaluate_arm_across_folds(
            folds,
            items_by_id,
            matrix_by_item_tier,
            router_by_item,
            eu_lambda=lam,
            ladder=lad,
            cost_key=default_cost_key,
            weight_profile=TRAFFIC_WEIGHT_PROFILES["realistic_skew"],
            cascade_verifiers_by_item=cascade_verifiers_by_item,
        )
        for arm_key in sweep_arm_keys:
            pareto_sweep[arm_key].append(
                {
                    "eu_lambda": lam,
                    **summarize_oof_arm(oof_lam_bal[arm_key], TRAFFIC_WEIGHT_PROFILES["balanced"]),
                }
            )
            pareto_sweep_skew[arm_key].append(
                {
                    "eu_lambda": lam,
                    **summarize_oof_arm(
                        oof_lam_skew[arm_key], TRAFFIC_WEIGHT_PROFILES["realistic_skew"]
                    ),
                }
            )

    net_value_summary = build_net_value_summary(
        oof_default=oof_balanced,
        balanced_arms=balanced_arms,
        skewed_arms=skewed_arms,
        pareto_sweep=pareto_sweep,
        tier_ids=tier_ids,
        oracle_available=oracle_available,
        oof_skew=oof_skew,
        pareto_sweep_skew=pareto_sweep_skew,
        baseline_tier_id=lad.baseline_tier_id,
    )

    # Per-slice latency analysis (mean, p50, p95)
    oof_ref = oof_balanced
    latency_by_slice: dict[str, dict[str, float]] = collections.defaultdict(dict)
    latency_percentiles_by_slice: dict[str, dict[str, dict[str, float]]] = collections.defaultdict(
        dict
    )
    latency_arms = dict.fromkeys(
        [
            f"static_{tier_ids[0]}",
            f"static_{lad.fallback_tier_id}",
            f"static_{lad.baseline_tier_id}",
            "decision_score_tail_cv",
            "decision_noul_eu_cv",
            "flash_lite_tail_cv",
        ]
    )
    for arm_key in latency_arms:
        by_sl: dict[str, list[float]] = collections.defaultdict(list)
        for rec in oof_ref[arm_key].values():
            by_sl[str(rec["slice"])].append(float(rec["total_latency_ms"]))
        for sl, vals in sorted(by_sl.items()):
            mean_v = round(sum(vals) / max(1, len(vals)), 2)
            vw = [(v, 1.0) for v in vals]
            latency_by_slice[sl][arm_key] = mean_v
            latency_percentiles_by_slice[sl][arm_key] = {
                "mean_ms": mean_v,
                "p50_ms": round(weighted_quantile(vw, 0.50), 2),
                "p95_ms": round(weighted_quantile(vw, 0.95), 2),
            }

    # Q3 Calibration over all N x len(tiers) (item, tier) predictions + 5-bin per-tier ECE
    raw_pairs: list[tuple[float, float]] = []
    pava_pairs: list[tuple[float, float]] = []
    fl_raw_pairs: list[tuple[float, float]] = []
    fl_pava_pairs: list[tuple[float, float]] = []
    per_tier_decision_pairs: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    per_tier_fl_pairs: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)

    for iid in sorted(oof_ref["decision_score_tail_cv"].keys()):
        r_rec = router_by_item[iid]
        noul_ans = r_rec["decision_noul"]["answers"]
        pava_map = r_rec["decision_noul"]["pava_projected"]
        fl_parsed = r_rec["flash_lite_json"]["parsed"]
        fl_solv = fl_parsed.get("solvability_probabilities", {})
        fl_pava = fl_parsed.get("pava_projected", fl_solv)

        for tid in tier_ids:
            reps = matrix_by_item_tier[iid][tid]
            emp_acc = sum(1.0 if x["passed"] else 0.0 for x in reps) / max(1, len(reps))
            p_raw = float(noul_ans[solvability_question_name(tid)]["noul"])
            p_iso = float(pava_map[tid])
            raw_pairs.append((p_raw, emp_acc))
            pava_pairs.append((p_iso, emp_acc))
            per_tier_decision_pairs[tid].append((p_iso, emp_acc))

            if tid in fl_solv:
                fp_raw = float(fl_solv[tid])
                fp_iso = float(fl_pava.get(tid, fp_raw))
                fl_raw_pairs.append((fp_raw, emp_acc))
                fl_pava_pairs.append((fp_iso, emp_acc))
                per_tier_fl_pairs[tid].append((fp_iso, emp_acc))

    q3_calibration = {
        "pooled_pair_count": len(raw_pairs),
        "raw_pooled_10bin": compute_calibration_metrics(raw_pairs, num_bins=10),
        "pava_pooled_10bin": compute_calibration_metrics(pava_pairs, num_bins=10),
        "flash_lite_raw_pooled_10bin": compute_calibration_metrics(fl_raw_pairs, num_bins=10),
        "flash_lite_pava_pooled_10bin": compute_calibration_metrics(fl_pava_pairs, num_bins=10),
        "per_tier_5bin_decision_pava": {
            tid: compute_calibration_metrics(per_tier_decision_pairs[tid], num_bins=5)
            for tid in tier_ids
        },
        "per_tier_5bin_flash_lite_pava": {
            tid: compute_calibration_metrics(per_tier_fl_pairs[tid], num_bins=5) for tid in tier_ids
        },
    }

    # Q4 Multimodal compression fidelity across all 30 visual items in s4_visual_and_pdf
    img_bucket = _Q4FidelityBucket()
    pdf_bucket = _Q4FidelityBucket()

    for iid, r_rec in router_by_item.items():
        if r_rec["slice"] != "s4_visual_and_pdf" or iid not in matrix_by_item_tier:
            continue
        ablation = r_rec.get("q4_ablation", {})
        if "full_media_score" not in ablation:
            continue

        if r_rec["modality"] == "image":
            img_bucket.add_pair(
                iid=iid,
                ref_rec=ablation["full_media_score"],
                comp_rec=r_rec["decision_score"],
                matrix_by_item_tier=matrix_by_item_tier,
                tier_ids=tier_ids,
                cost_key=default_cost_key,
            )
        elif r_rec["modality"] == "pdf":
            pdf_bucket.add_pair(
                iid=iid,
                ref_rec=ablation["full_media_score"],
                comp_rec=r_rec["decision_score"],
                matrix_by_item_tier=matrix_by_item_tier,
                tier_ids=tier_ids,
                cost_key=default_cost_key,
            )
            if "pdf_page1_img_high" in ablation and "pdf_page1_img_low" in ablation:
                img_bucket.add_pair(
                    iid=iid,
                    ref_rec=ablation["pdf_page1_img_high"],
                    comp_rec=ablation["pdf_page1_img_low"],
                    matrix_by_item_tier=matrix_by_item_tier,
                    tier_ids=tier_ids,
                    cost_key=default_cost_key,
                )

    img_summary = img_bucket.summarize("visual_items_evaluated")
    pdf_summary = pdf_bucket.summarize("pdf_items_evaluated")
    q4_summary = {
        "image_resolution_low_vs_high_all_30": img_summary,
        "pdf_page1_envelope_vs_full_doc_15": pdf_summary,
        "visual_items_evaluated": img_summary["visual_items_evaluated"],
        "mean_js_divergence_bits": img_summary["mean_js_divergence_bits"],
        "top_tier_agreement_rate": img_summary["top_tier_agreement_rate"],
    }

    summary = {
        "items_evaluated": len(eval_items),
        "ladder": lad.to_summary(),
        "balanced_equal_weights": balanced_arms,
        "production_skew_60_25_10_5": skewed_arms,
        "net_value_summary": net_value_summary,
        "paired_bootstrap_10k_ci95": bootstrap_balanced,
        "paired_bootstrap_10k_ci95_skew": bootstrap_skew,
        "pareto_lambda_sweep": dict(pareto_sweep),
        "pareto_lambda_sweep_skew": dict(pareto_sweep_skew),
        "latency_by_slice_ms": dict(latency_by_slice),
        "latency_percentiles_by_slice_ms": dict(latency_percentiles_by_slice),
        "q3_calibration": q3_calibration,
        "q4_multimodal_compression": q4_summary,
    }
    out_summary_path.parent.mkdir(parents=True, exist_ok=True)
    out_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze evals matrix and router telemetry.")
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "data/eval.jsonl")
    parser.add_argument(
        "--matrix", type=Path, default=REPO_ROOT / "results/downstream_matrix.jsonl"
    )
    parser.add_argument(
        "--routers", type=Path, default=REPO_ROOT / "results/router_telemetry.jsonl"
    )
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results/summary.json")
    parser.add_argument(
        "--cascade-verifiers",
        type=Path,
        default=None,
        help="Optional cascade verifiers JSONL from `python -m evals.routers --cascade-matrix ...`.",
    )
    add_tiers_argument(parser)
    args = parser.parse_args()

    res = run_analysis(
        args.dataset,
        args.matrix,
        args.routers,
        args.output,
        ladder=resolve_ladder(args.tiers),
        cascade_verifiers_path=args.cascade_verifiers,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
