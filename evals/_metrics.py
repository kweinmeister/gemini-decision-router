"""Statistical resampling, LORO Oracle, Pareto convex-hull geometry, and Net-Value metrics."""

from __future__ import annotations

import collections
import random
from collections.abc import Sequence
from typing import Any


def stratified_k_folds(
    items: list[dict[str, Any]],
    k: int = 5,
    seed: int = 20261111,
) -> list[tuple[list[str], list[str]]]:
    """Split `items` into `k` disjoint stratified folds by `slice`.

    Returns a list of `k` tuples `(train_item_ids, test_item_ids)`.
    """
    rng = random.Random(seed)
    by_slice: dict[str, list[str]] = collections.defaultdict(list)
    for it in items:
        by_slice[str(it["slice"])].append(str(it["id"]))

    fold_test_buckets: list[list[str]] = [[] for _ in range(k)]
    for slice_name in sorted(by_slice.keys()):
        ids = list(by_slice[slice_name])
        rng.shuffle(ids)
        for idx, item_id in enumerate(ids):
            fold_test_buckets[idx % k].append(item_id)

    all_ids = [str(it["id"]) for it in items]
    folds: list[tuple[list[str], list[str]]] = []
    for f_idx in range(k):
        test_set = set(fold_test_buckets[f_idx])
        train_ids = [iid for iid in all_ids if iid not in test_set]
        test_ids = list(fold_test_buckets[f_idx])
        folds.append((train_ids, test_ids))
    return folds


def compute_loro_oracle_for_item(
    replicates_by_tier: dict[str, list[dict[str, Any]]],
    cost_key: str = "compute_units",
    *,
    eu_lambda: float = 25.0,
    training_tier_mean_cost: dict[str, float] | None = None,
    tier_ids: Sequence[str] | None = None,
) -> tuple[float, float, float]:
    """Compute unbiased Leave-One-Replicate-Out (LORO) Oracle accuracy, cost, and latency across k replicates."""
    ids = tuple(tier_ids) if tier_ids is not None else tuple(replicates_by_tier)
    num_reps = len(replicates_by_tier[ids[0]])
    if num_reps < 2:
        raise ValueError("LORO Oracle requires at least 2 replicates per (item, tier).")

    held_out_passes: list[float] = []
    held_out_costs: list[float] = []
    held_out_latencies: list[float] = []

    for r in range(num_reps):
        best_tid = ids[0]
        best_key: tuple[float, float, float] = (float("-inf"), float("-inf"), float("-inf"))
        for tid in ids:
            reps = replicates_by_tier[tid]
            sel_reps = [reps[j] for j in range(num_reps) if j != r]
            sel_acc = sum(1.0 if x["passed"] else 0.0 for x in sel_reps) / len(sel_reps)
            sel_cost = sum(float(x[cost_key]) for x in sel_reps) / len(sel_reps)
            utility = sel_acc - (eu_lambda * sel_cost)
            tie_cost = (
                float(training_tier_mean_cost[tid])
                if training_tier_mean_cost and tid in training_tier_mean_cost
                else sel_cost
            )
            cand_key = (utility, -tie_cost, -sel_cost)
            if cand_key > best_key:
                best_key = cand_key
                best_tid = tid

        eval_rep = replicates_by_tier[best_tid][r]
        held_out_passes.append(1.0 if eval_rep["passed"] else 0.0)
        held_out_costs.append(float(eval_rep[cost_key]))
        held_out_latencies.append(float(eval_rep.get("latency_ms", 0.0)))

    return (
        sum(held_out_passes) / len(held_out_passes),
        sum(held_out_costs) / len(held_out_costs),
        sum(held_out_latencies) / len(held_out_latencies),
    )


def paired_bootstrap_ci(
    diffs: list[float],
    *,
    weights: Sequence[float] | None = None,
    n_boot: int = 10000,
    seed: int = 20261111,
) -> tuple[float, float, float]:
    """Compute `(mean_diff, ci_2.5%, ci_97.5%)` over paired item-level differences.

    When `weights` is provided, computes the traffic-weighted mean and resamples items with
    replacement weighted by `weights` (item-clustered bootstrap under realistic traffic skew).
    """
    if not diffs:
        return 0.0, 0.0, 0.0
    n = len(diffs)
    rng = random.Random(seed)
    if weights is None:
        obs_mean = sum(diffs) / n
        boot_means = [sum(rng.choices(diffs, k=n)) / n for _ in range(n_boot)]
    else:
        w_list = [max(0.0, float(w)) for w in weights]
        w_tot = sum(w_list)
        if w_tot <= 0.0:
            return 0.0, 0.0, 0.0
        obs_mean = sum(d * w for d, w in zip(diffs, w_list, strict=True)) / w_tot
        boot_means = [sum(rng.choices(diffs, weights=w_list, k=n)) / n for _ in range(n_boot)]
    boot_means.sort()
    lo = boot_means[int(0.025 * n_boot)]
    hi = boot_means[int(0.975 * n_boot)]
    return obs_mean, lo, hi


def weighted_quantile(values_and_weights: Sequence[tuple[float, float]], q: float) -> float:
    """Compute weighted quantile `q` in `[0.0, 1.0]` using midpoint cumulative weights."""
    pts = sorted((float(v), float(w)) for v, w in values_and_weights if w > 0.0)
    if not pts:
        return 0.0
    if len(pts) == 1:
        return pts[0][0]
    total_w = sum(w for _, w in pts)
    target = max(0.0, min(1.0, q)) * total_w
    cum = 0.0
    mids: list[float] = []
    for _, w in pts:
        mids.append(cum + 0.5 * w)
        cum += w
    if target <= mids[0]:
        return pts[0][0]
    if target >= mids[-1]:
        return pts[-1][0]
    for i in range(len(pts) - 1):
        if mids[i] <= target <= mids[i + 1]:
            span = mids[i + 1] - mids[i]
            frac = 0.0 if span <= 1e-12 else (target - mids[i]) / span
            return pts[i][0] + frac * (pts[i + 1][0] - pts[i][0])
    return pts[-1][0]


def summarize_oof_arm(
    item_map: dict[str, dict[str, Any]],
    weight_profile: dict[str, float],
) -> dict[str, Any]:
    """Aggregate item-level out-of-fold metrics under a slice traffic-weight profile."""
    by_slice: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for rec in item_map.values():
        by_slice[str(rec["slice"])].append(rec)

    weighted_acc = 0.0
    weighted_units = 0.0
    weighted_router_units = 0.0
    weighted_downstream_units = 0.0
    weighted_router_lat = 0.0
    weighted_total_lat = 0.0
    total_w = 0.0
    tier_weights: dict[str, float] = collections.defaultdict(float)
    router_lat_samples: list[tuple[float, float]] = []
    total_lat_samples: list[tuple[float, float]] = []

    for sl, w in weight_profile.items():
        rows = by_slice.get(sl, [])
        if not rows or w <= 0.0:
            continue
        m = len(rows)
        item_w = w / m
        weighted_acc += w * (sum(float(x["accuracy"]) for x in rows) / m)
        weighted_units += w * (sum(float(x["total_compute_units"]) for x in rows) / m)
        weighted_router_units += w * (sum(float(x["router_compute_units"]) for x in rows) / m)
        weighted_downstream_units += w * (
            sum(float(x["downstream_compute_units"]) for x in rows) / m
        )
        weighted_router_lat += w * (sum(float(x["router_latency_ms"]) for x in rows) / m)
        weighted_total_lat += w * (sum(float(x["total_latency_ms"]) for x in rows) / m)
        total_w += w
        for x in rows:
            tier_weights[str(x["selected_tier"])] += item_w
            router_lat_samples.append((float(x["router_latency_ms"]), item_w))
            total_lat_samples.append((float(x["total_latency_ms"]), item_w))

    norm = max(1e-9, total_w)
    return {
        "mean_accuracy": round(weighted_acc / norm, 4),
        "mean_compute_units": round(weighted_units / norm, 7),
        "compute_units_per_1k": round((weighted_units / norm) * 1000.0, 4),
        "router_compute_units_per_1k": round((weighted_router_units / norm) * 1000.0, 4),
        "downstream_compute_units_per_1k": round((weighted_downstream_units / norm) * 1000.0, 4),
        "mean_router_latency_ms": round(weighted_router_lat / norm, 2),
        "p50_router_latency_ms": round(weighted_quantile(router_lat_samples, 0.50), 2),
        "p95_router_latency_ms": round(weighted_quantile(router_lat_samples, 0.95), 2),
        "mean_total_latency_ms": round(weighted_total_lat / norm, 2),
        "p50_total_latency_ms": round(weighted_quantile(total_lat_samples, 0.50), 2),
        "p95_total_latency_ms": round(weighted_quantile(total_lat_samples, 0.95), 2),
        "tier_selection_rates": {
            tid: round(wt / norm, 4) for tid, wt in sorted(tier_weights.items())
        },
    }


def compute_static_pareto_hull(
    static_points: Sequence[dict[str, Any]],
    *,
    x_key: str = "compute_units_per_1k",
    y_key: str = "mean_accuracy",
) -> list[dict[str, Any]]:
    """Compute the upper-left concave Pareto hull of zero-router static tiers."""
    ordered = sorted(static_points, key=lambda p: (float(p[x_key]), -float(p[y_key])))
    undominated: list[dict[str, Any]] = []
    best_y = float("-inf")
    for pt in ordered:
        y = float(pt[y_key])
        if y > best_y + 1e-12:
            undominated.append(pt)
            best_y = y

    hull: list[dict[str, Any]] = []
    for pt in undominated:
        while len(hull) >= 2:
            x0, y0 = float(hull[-2][x_key]), float(hull[-2][y_key])
            x1, y1 = float(hull[-1][x_key]), float(hull[-1][y_key])
            x2, y2 = float(pt[x_key]), float(pt[y_key])
            if (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0) >= -1e-12:
                hull.pop()
            else:
                break
        hull.append(pt)
    return hull


def interpolate_hull_min_x_at_accuracy(
    hull: Sequence[dict[str, Any]],
    target_accuracy: float,
    *,
    x_key: str = "compute_units_per_1k",
    y_key: str = "mean_accuracy",
) -> float | None:
    """Minimum `x_key` (cost or latency) on `hull` achieving accuracy `>= target_accuracy`."""
    if not hull:
        return None
    y_min = float(hull[0][y_key])
    y_max = float(hull[-1][y_key])
    if target_accuracy <= y_min + 1e-9:
        return float(hull[0][x_key])
    if target_accuracy > y_max + 1e-9:
        return None
    for i in range(len(hull) - 1):
        y0, y1 = float(hull[i][y_key]), float(hull[i + 1][y_key])
        if y0 - 1e-9 <= target_accuracy <= y1 + 1e-9:
            x0, x1 = float(hull[i][x_key]), float(hull[i + 1][x_key])
            span = y1 - y0
            frac = 0.0 if span <= 1e-12 else max(0.0, min(1.0, (target_accuracy - y0) / span))
            return x0 + frac * (x1 - x0)
    return float(hull[-1][x_key])


def profile_net_value_block(
    arms_summary: dict[str, dict[str, Any]],
    tier_ids: Sequence[str],
) -> dict[str, Any]:
    """Compare every routed arm against static tiers (hull & single-model) and LLM-router baselines."""
    static_arms = [
        arms_summary[f"static_{tid}"] for tid in tier_ids if f"static_{tid}" in arms_summary
    ]
    compute_hull = compute_static_pareto_hull(
        static_arms, x_key="compute_units_per_1k", y_key="mean_accuracy"
    )
    lat_hull = compute_static_pareto_hull(
        static_arms, x_key="mean_total_latency_ms", y_key="mean_accuracy"
    )

    vs_no_router: dict[str, dict[str, Any]] = {}
    for arm_name, st in arms_summary.items():
        if arm_name.startswith("static_") or arm_name == "loro_oracle":
            continue
        acc = float(st["mean_accuracy"])
        units = float(st["compute_units_per_1k"])
        lat = float(st["mean_total_latency_ms"])

        hull_units = interpolate_hull_min_x_at_accuracy(
            compute_hull, acc, x_key="compute_units_per_1k", y_key="mean_accuracy"
        )
        hull_lat = interpolate_hull_min_x_at_accuracy(
            lat_hull, acc, x_key="mean_total_latency_ms", y_key="mean_accuracy"
        )
        matching_statics = [s for s in static_arms if float(s["mean_accuracy"]) + 1e-9 >= acc]
        cheapest_static = (
            min(matching_statics, key=lambda s: float(s["compute_units_per_1k"]))
            if matching_statics
            else None
        )
        fastest_static = (
            min(matching_statics, key=lambda s: float(s["mean_total_latency_ms"]))
            if matching_statics
            else None
        )

        hull_saved = None if hull_units is None else round(hull_units - units, 4)
        vs_no_router[arm_name] = {
            "mean_accuracy": acc,
            "compute_units_per_1k": units,
            "router_compute_units_per_1k": st["router_compute_units_per_1k"],
            "static_hull_compute_units_per_1k": None
            if hull_units is None
            else round(hull_units, 4),
            "net_compute_saved_vs_static_hull_per_1k": hull_saved,
            "pareto_dominates_no_router": bool(hull_saved is not None and hull_saved > 0.0),
            "cheapest_matching_static_arm": (
                None if cheapest_static is None else cheapest_static["arm_name"]
            ),
            "net_compute_saved_vs_single_static_per_1k": (
                None
                if cheapest_static is None
                else round(float(cheapest_static["compute_units_per_1k"]) - units, 4)
            ),
            "net_latency_saved_vs_static_hull_ms": (
                None if hull_lat is None else round(hull_lat - lat, 2)
            ),
            "net_latency_saved_vs_single_static_ms": (
                None
                if fastest_static is None
                else round(float(fastest_static["mean_total_latency_ms"]) - lat, 2)
            ),
        }

    decision_vs_flash_lite: dict[str, dict[str, float]] = {}
    for dec_arm, fl_arm in (
        ("decision_score_argmax", "flash_lite_argmax"),
        ("decision_score_tail_cv", "flash_lite_tail_cv"),
        ("decision_noul_eu_cv", "flash_lite_noul_eu_cv"),
    ):
        if dec_arm in arms_summary and fl_arm in arms_summary:
            d_st, f_st = arms_summary[dec_arm], arms_summary[fl_arm]
            decision_vs_flash_lite[f"{dec_arm}_vs_{fl_arm}"] = {
                "accuracy_diff": round(
                    float(d_st["mean_accuracy"]) - float(f_st["mean_accuracy"]), 4
                ),
                "total_compute_saved_per_1k": round(
                    float(f_st["compute_units_per_1k"]) - float(d_st["compute_units_per_1k"]), 4
                ),
                "router_tax_saved_per_1k": round(
                    float(f_st["router_compute_units_per_1k"])
                    - float(d_st["router_compute_units_per_1k"]),
                    4,
                ),
                "mean_router_latency_saved_ms": round(
                    float(f_st["mean_router_latency_ms"]) - float(d_st["mean_router_latency_ms"]),
                    2,
                ),
                "p50_router_latency_saved_ms": round(
                    float(f_st["p50_router_latency_ms"]) - float(d_st["p50_router_latency_ms"]), 2
                ),
                "p95_router_latency_saved_ms": round(
                    float(f_st["p95_router_latency_ms"]) - float(d_st["p95_router_latency_ms"]), 2
                ),
            }

    return {
        "static_compute_hull": [
            {
                "arm_name": p["arm_name"],
                "mean_accuracy": p["mean_accuracy"],
                "compute_units_per_1k": p["compute_units_per_1k"],
            }
            for p in compute_hull
        ],
        "vs_no_router": vs_no_router,
        "decision_vs_flash_lite": decision_vs_flash_lite,
    }


def _compute_utility_sweep(
    arms_summary: dict[str, dict[str, Any]],
    pareto_sweep: dict[str, list[dict[str, Any]]],
    tier_ids: Sequence[str],
    oracle_available: bool,
) -> list[dict[str, Any]]:
    static_names = [f"static_{tid}" for tid in tier_ids if f"static_{tid}" in arms_summary]
    decision_cv_arms = ("decision_score_tail_cv", "decision_choice_tail_cv", "decision_noul_eu_cv")
    rows: list[dict[str, Any]] = []
    sweep_len = len(pareto_sweep.get("decision_score_tail_cv", []))
    for idx in range(sweep_len):
        lam = float(pareto_sweep["decision_score_tail_cv"][idx]["eu_lambda"])
        if lam <= 0.0:
            continue
        best_static_name, best_static_u = max(
            (
                (
                    s_name,
                    float(arms_summary[s_name]["mean_accuracy"])
                    - lam * float(arms_summary[s_name]["mean_compute_units"]),
                )
                for s_name in static_names
            ),
            key=lambda kv: kv[1],
        )
        cand_decision_pts: list[tuple[str, float]] = [
            (
                arm,
                float(pareto_sweep[arm][idx]["mean_accuracy"])
                - lam * float(pareto_sweep[arm][idx]["mean_compute_units"]),
            )
            for arm in decision_cv_arms
            if arm in pareto_sweep
        ]
        cand_decision_pts.extend(
            (
                fixed_arm,
                float(arms_summary[fixed_arm]["mean_accuracy"])
                - lam * float(arms_summary[fixed_arm]["mean_compute_units"]),
            )
            for fixed_arm in ("decision_score_argmax", "decision_choice_argmax")
            if fixed_arm in arms_summary
        )
        best_dec_name, best_dec_u = max(cand_decision_pts, key=lambda kv: kv[1])
        net_val_per_1k = ((best_dec_u - best_static_u) / lam) * 1000.0

        oracle_pt = pareto_sweep["loro_oracle"][idx]
        oracle_u = float(oracle_pt["mean_accuracy"]) - lam * float(oracle_pt["mean_compute_units"])
        headroom_pct: float | None = None
        if oracle_available and oracle_u - best_static_u > 1e-6:
            headroom_pct = round(
                100.0 * (best_dec_u - best_static_u) / (oracle_u - best_static_u), 2
            )

        rows.append(
            {
                "eu_lambda": lam,
                "unit_tradeoff_per_correct": round(1.0 / lam, 4),
                "best_static_arm": best_static_name,
                "best_static_utility": round(best_static_u, 5),
                "best_decision_arm": best_dec_name,
                "best_decision_utility": round(best_dec_u, 5),
                "net_value_vs_best_static_per_1k_units": round(net_val_per_1k, 4),
                "oracle_available": oracle_available,
                "oracle_headroom_captured_pct": headroom_pct,
            }
        )
    return rows


def build_net_value_summary(
    *,
    oof_default: dict[str, dict[str, dict[str, Any]]],
    balanced_arms: dict[str, dict[str, Any]],
    skewed_arms: dict[str, dict[str, Any]],
    pareto_sweep: dict[str, list[dict[str, Any]]],
    tier_ids: Sequence[str],
    oracle_available: bool,
    oof_skew: dict[str, dict[str, dict[str, Any]]] | None = None,
    pareto_sweep_skew: dict[str, list[dict[str, Any]]] | None = None,
    baseline_tier_id: str | None = None,
) -> dict[str, Any]:
    """Synthesize end-to-end net value and serial-latency break-even curves."""
    utility_by_lambda = _compute_utility_sweep(
        balanced_arms, pareto_sweep, tier_ids, oracle_available
    )
    utility_by_lambda_skew = (
        _compute_utility_sweep(skewed_arms, pareto_sweep_skew, tier_ids, oracle_available)
        if pareto_sweep_skew is not None
        else []
    )

    base_tid = baseline_tier_id or tier_ids[-1]
    base_static_key = f"static_{base_tid}"
    oof_for_curve = oof_skew if oof_skew is not None else oof_default

    # Sweep routine traffic share (s1 + s2 combined weight, where 0.85 == 60% s1 + 25% s2 skew)
    # to trace both cost net-value and the continuous serial-latency break-even curve.
    breakeven_sweep: list[dict[str, Any]] = []
    tracked_lat_arms = (
        base_static_key,
        "decision_score_tail_cv",
        "decision_noul_eu_cv",
        "flash_lite_tail_cv",
    )
    for routine_share in (0.0, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95, 1.0):
        rem = 1.0 - routine_share
        prof = {
            "s1_trivial_text": round((0.60 / 0.85) * routine_share, 6),
            "s2_everyday_code": round((0.25 / 0.85) * routine_share, 6),
            "s3_frontier_reasoning": round(rem / 3.0, 6),
            "s4_visual_and_pdf": round(rem / 3.0, 6),
            "s5_spoken_audio": round(rem / 3.0, 6),
        }
        prof_arms = {
            arm: {"arm_name": arm, **summarize_oof_arm(imap, prof)}
            for arm, imap in oof_for_curve.items()
        }
        prof_nv = profile_net_value_block(prof_arms, tier_ids)["vs_no_router"]
        base_lat = float(prof_arms[base_static_key]["mean_total_latency_ms"])
        lat_by_arm = {
            arm: {
                "mean_ms": prof_arms[arm]["mean_total_latency_ms"],
                "p50_ms": prof_arms[arm]["p50_total_latency_ms"],
                "p95_ms": prof_arms[arm]["p95_total_latency_ms"],
            }
            for arm in tracked_lat_arms
            if arm in prof_arms
        }
        breakeven_sweep.append(
            {
                "routine_s1_s2_share": routine_share,
                "decision_score_argmax_net_compute_saved_vs_hull_per_1k": prof_nv[
                    "decision_score_argmax"
                ]["net_compute_saved_vs_static_hull_per_1k"],
                "decision_score_tail_cv_net_compute_saved_vs_hull_per_1k": prof_nv[
                    "decision_score_tail_cv"
                ]["net_compute_saved_vs_static_hull_per_1k"],
                "decision_score_tail_cv_latency_saved_vs_baseline_ms": round(
                    base_lat - float(prof_arms["decision_score_tail_cv"]["mean_total_latency_ms"]),
                    2,
                ),
                "flash_lite_tail_cv_latency_saved_vs_baseline_ms": round(
                    base_lat - float(prof_arms["flash_lite_tail_cv"]["mean_total_latency_ms"]),
                    2,
                ),
                "latency_by_arm_ms": lat_by_arm,
            }
        )

    return {
        "balanced_equal_weights": profile_net_value_block(balanced_arms, tier_ids),
        "production_skew_60_25_10_5": profile_net_value_block(skewed_arms, tier_ids),
        "utility_and_headroom_by_lambda": utility_by_lambda,
        "utility_and_headroom_by_lambda_skew": utility_by_lambda_skew,
        "easy_traffic_breakeven_sweep": breakeven_sweep,
    }
