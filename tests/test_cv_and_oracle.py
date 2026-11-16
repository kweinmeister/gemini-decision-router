"""Unit tests for 5-fold stratified CV, Leave-One-Replicate-Out (LORO) Oracle, Spearman rho, and ECE/Brier."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest

from evals.analyze import (
    _cascade_outcome,
    _check_data_matches_ladder,
    _select_heuristic_tier,
    compute_calibration_metrics,
    compute_loro_oracle_for_item,
    compute_static_pareto_hull,
    fit_training_tier_mean_cost,
    interpolate_hull_min_x_at_accuracy,
    jensen_shannon_divergence,
    paired_bootstrap_ci,
    spearman_rank_correlation,
    stratified_k_folds,
    summarize_oof_arm,
    weighted_quantile,
)
from evals.plots import plot_latency_breakeven, plot_slice_latency_percentiles
from gemini_decision_router.ladder import default_ladder

TIER_IDS = default_ladder().ordered_tier_ids


def _naive_in_sample_oracle_for_test(
    replicates_by_tier: dict[str, list[dict[str, Any]]],
    cost_key: str = "compute_units",
    eu_lambda: float = 25.0,
) -> float:
    """In-sample oracle helper (used only in tests to quantify post-hoc selection bias vs LORO)."""
    best_acc = 0.0
    best_u = float("-inf")
    for tid in TIER_IDS:
        reps = replicates_by_tier[tid]
        acc = sum(1.0 if x["passed"] else 0.0 for x in reps) / len(reps)
        cost = sum(float(x[cost_key]) for x in reps) / len(reps)
        u = acc - eu_lambda * cost
        if u > best_u:
            best_u = u
            best_acc = acc
    return best_acc


def test_stratified_5_folds_disjoint_and_balanced() -> None:
    items = [
        {"id": f"item_{sl}_{i}", "slice": sl}
        for sl in ("s1", "s2", "s3", "s4", "s5")
        for i in range(30)
    ]
    folds = stratified_k_folds(items, k=5, seed=42)
    assert len(folds) == 5

    all_test_seen: list[str] = []
    for train_ids, test_ids in folds:
        assert len(train_ids) == 120
        assert len(test_ids) == 30
        assert set(train_ids).isdisjoint(set(test_ids))
        all_test_seen.extend(test_ids)

    assert len(set(all_test_seen)) == 150


def test_loro_oracle_eliminates_optimistic_posthoc_bias() -> None:
    """When all tiers have true pass probability p=0.5 across k=3 replicates,
    naive in-sample selection picks whichever tier happened to get 3/3 or 2/3 and scores it on the
    SAME replicates (yielding >0.80 instead of the true 0.50).
    Leave-One-Replicate-Out (LORO) Oracle evaluates on the held-out replicate and stays unbiased (~0.50).
    """
    rng = random.Random(12345)
    n_trials = 600
    loro_accs: list[float] = []
    naive_accs: list[float] = []

    for _ in range(n_trials):
        replicates_by_tier = {}
        for idx, tid in enumerate(TIER_IDS):
            reps = [
                {
                    "passed": (rng.random() < 0.5),
                    "compute_units": 0.001 * (idx + 1),
                    "latency_ms": 200.0 * (idx + 1),
                }
                for _ in range(3)
            ]
            replicates_by_tier[tid] = reps

        loro_acc, _, _ = compute_loro_oracle_for_item(
            replicates_by_tier, eu_lambda=25.0, tier_ids=TIER_IDS
        )
        naive_acc = _naive_in_sample_oracle_for_test(replicates_by_tier, eu_lambda=25.0)
        loro_accs.append(loro_acc)
        naive_accs.append(naive_acc)

    mean_loro = sum(loro_accs) / n_trials
    mean_naive = sum(naive_accs) / n_trials

    # True pass rate of every tier is 0.50; LORO is unbiased (~0.50), naive is inflated (>0.75)
    assert abs(mean_loro - 0.50) < 0.03
    assert mean_naive > 0.75


def test_calibration_spearman_and_js_divergence() -> None:
    perfect_pairs = [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)] * 25
    cal = compute_calibration_metrics(perfect_pairs, num_bins=10)
    assert cal["ece"] == pytest.approx(0.0, abs=1e-9)
    assert cal["brier_score"] == pytest.approx(0.0, abs=1e-9)
    assert len(cal["bins"]) == 3

    p = {"0": 0.5, "1": 0.5, "2": 0.0}
    q = {"0": 0.5, "1": 0.5, "2": 0.0}
    assert jensen_shannon_divergence(p, q) == pytest.approx(0.0, abs=1e-9)
    assert spearman_rank_correlation(
        [1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]
    ) == pytest.approx(1.0)


def _matrix_and_router(
    item_ids: list[str], matrix_tiers: tuple[str, ...], router_tiers: tuple[str, ...]
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, dict[str, Any]]]:
    matrix = {
        iid: {
            tid: [{"passed": True, "compute_units": 0.001 * (k + 1)}]
            for k, tid in enumerate(matrix_tiers)
        }
        for iid in item_ids
    }
    router = {
        iid: {
            "flash_lite_json": {
                "parsed": {
                    "tier_probabilities": {tid: 1.0 / len(router_tiers) for tid in router_tiers}
                }
            }
        }
        for iid in item_ids
    }
    return matrix, router


def test_check_data_matches_ladder_fails_fast_on_stale_data() -> None:
    lad = default_ladder()
    ids = ["i1", "i2"]

    # Matching data passes; extra matrix tiers outside the ladder are ignored
    matrix, router = _matrix_and_router(ids, (*TIER_IDS, "t4_carbon_high"), TIER_IDS)
    _check_data_matches_ladder(ids, matrix, router, lad)

    # A missing (item, tier) cell is reported
    del matrix["i2"][TIER_IDS[2]]
    with pytest.raises(ValueError, match="1 \\(item, tier\\) cells missing"):
        _check_data_matches_ladder(ids, matrix, router, lad)

    # Router telemetry collected under a 5-tier ladder is reported
    matrix, router = _matrix_and_router(ids, TIER_IDS, (*TIER_IDS, "t4_carbon_high"))
    with pytest.raises(ValueError, match="2 router records collected with other tiers"):
        _check_data_matches_ladder(ids, matrix, router, lad)


def test_fit_training_tier_mean_cost_uses_only_training_items() -> None:
    matrix = {
        "train_a": {"lo": [{"c": 0.001}, {"c": 0.003}], "hi": [{"c": 0.010}]},
        "train_b": {"lo": [{"c": 0.002}], "hi": [{"c": 0.030}]},
        "test_x": {"lo": [{"c": 9.0}], "hi": [{"c": 9.0}]},
    }
    costs = fit_training_tier_mean_cost({"train_a", "train_b"}, matrix, "c", ("lo", "hi"))
    assert costs == {"lo": pytest.approx(0.002), "hi": pytest.approx(0.020)}


def test_paired_bootstrap_ci_properties() -> None:
    # Empty diffs return zeros
    assert paired_bootstrap_ci([]) == (0.0, 0.0, 0.0)

    # Constant diffs return exact constant for mean and bounds
    const_diffs = [2.5] * 20
    m, lo, hi = paired_bootstrap_ci(const_diffs, n_boot=500)
    assert m == pytest.approx(2.5)
    assert lo == pytest.approx(2.5)
    assert hi == pytest.approx(2.5)

    # Monotonicity of CI bounds on varying data
    diffs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    m, lo, hi = paired_bootstrap_ci(diffs, n_boot=1000, seed=42)
    assert lo < m < hi
    assert lo >= min(diffs)
    assert hi <= max(diffs)


def test_static_pareto_hull_prunes_dominated_and_interior_tiers() -> None:
    # t_med_dom is strictly dominated (same 0.90 acc as t_low at higher compute).
    # t_mid_convex (40 units, 0.95) sits below the 50/50 coin-flip mix of t_low (10, 0.90)
    # and t_high (50, 1.00), which gets 0.95 accuracy at 30 units!
    statics = [
        {"arm_name": "static_t0", "compute_units_per_1k": 2.0, "mean_accuracy": 0.60},
        {
            "arm_name": "static_t_low",
            "compute_units_per_1k": 10.0,
            "mean_accuracy": 0.90,
        },
        {
            "arm_name": "static_t_med_dom",
            "compute_units_per_1k": 25.0,
            "mean_accuracy": 0.90,
        },
        {
            "arm_name": "static_t_mid_convex",
            "compute_units_per_1k": 40.0,
            "mean_accuracy": 0.95,
        },
        {
            "arm_name": "static_t_high",
            "compute_units_per_1k": 50.0,
            "mean_accuracy": 1.00,
        },
    ]
    hull = compute_static_pareto_hull(statics)
    assert [p["arm_name"] for p in hull] == [
        "static_t0",
        "static_t_low",
        "static_t_high",
    ]

    # At 0.95 accuracy, optimal no-router mix of t_low and t_high uses 30.0 units/1k:
    # - A router using 22/1k at 0.95 accuracy has +8/1k positive net value
    # - A router using 34/1k at 0.95 accuracy has -4/1k negative net value
    hull_at_95 = interpolate_hull_min_x_at_accuracy(hull, 0.95)
    assert hull_at_95 == pytest.approx(30.0)
    assert interpolate_hull_min_x_at_accuracy(hull, 1.00) == pytest.approx(50.0)
    assert interpolate_hull_min_x_at_accuracy(hull, 0.50) == pytest.approx(2.0)


def test_weighted_quantile_and_summarize_oof_arm_latency_and_rates() -> None:
    assert weighted_quantile([], 0.5) == 0.0
    assert weighted_quantile([(10.0, 1.0)], 0.95) == pytest.approx(10.0)
    pts = [(100.0, 1.0), (200.0, 1.0), (300.0, 1.0), (1000.0, 1.0)]
    assert weighted_quantile(pts, 0.50) == pytest.approx(250.0)
    assert weighted_quantile(pts, 0.95) > 800.0

    item_map = {
        "i1": {
            "slice": "s1",
            "selected_tier": "t0",
            "accuracy": 1.0,
            "router_compute_units": 0.0001,
            "downstream_compute_units": 0.0019,
            "total_compute_units": 0.0020,
            "router_latency_ms": 350.0,
            "downstream_latency_ms": 2000.0,
            "total_latency_ms": 2350.0,
        },
        "i2": {
            "slice": "s2",
            "selected_tier": "t3",
            "accuracy": 1.0,
            "router_compute_units": 0.0002,
            "downstream_compute_units": 0.0498,
            "total_compute_units": 0.0500,
            "router_latency_ms": 450.0,
            "downstream_latency_ms": 60000.0,
            "total_latency_ms": 60450.0,
        },
    }
    st = summarize_oof_arm(item_map, {"s1": 0.75, "s2": 0.25})
    assert st["router_compute_units_per_1k"] == pytest.approx(
        (0.75 * 0.0001 + 0.25 * 0.0002) * 1000
    )
    assert st["downstream_compute_units_per_1k"] == pytest.approx(
        (0.75 * 0.0019 + 0.25 * 0.0498) * 1000
    )
    assert st["tier_selection_rates"] == {"t0": 0.75, "t3": 0.25}
    assert st["p50_total_latency_ms"] < st["p95_total_latency_ms"]


def test_heuristic_scales_across_4_tier_and_5_tier_ladders() -> None:
    tiers4 = ("t0", "t1", "t2", "t3")
    assert _select_heuristic_tier(40, "text", 60, 4000, 0.34, tier_ids=tiers4) == "t0"
    assert _select_heuristic_tier(20_000, "text", 60, 4000, 0.34, tier_ids=tiers4) == "t3"
    assert _select_heuristic_tier(20_000, "text", 60, None, 0.34, tier_ids=tiers4) == "t1"

    # On a 5-tier ladder, intermediate rank 3 ("t3") and frontier ("t4") are both reachable
    tiers5 = ("t0", "t1", "t2", "t3", "t4")
    reached = {
        _select_heuristic_tier(tok, "text", 100, 10_000, 0.0, tier_ids=tiers5)
        for tok in (50, 200, 600, 3_000, 25_000)
    }
    assert reached == {"t0", "t1", "t2", "t3", "t4"}


def test_calibration_metrics_preserve_extreme_probabilities_without_brier_distortion() -> None:
    cal = compute_calibration_metrics([(0.0, 0.0), (1.0, 1.0)] * 10, num_bins=10)
    assert cal["brier_score"] == pytest.approx(0.0, abs=1e-12)
    assert cal["ece"] == pytest.approx(0.0, abs=1e-12)
    assert 0.0 <= cal["log_loss"] < 1e-5


def test_weighted_paired_bootstrap_reflects_traffic_skew() -> None:
    diffs = [10.0, 10.0, -2.0, -2.0]
    weights = [0.45, 0.45, 0.05, 0.05]
    m, lo, hi = paired_bootstrap_ci(diffs, weights=weights, n_boot=1000, seed=42)
    assert m == pytest.approx(8.8)
    assert lo > 0.0
    assert lo <= m <= hi


def test_latency_breakeven_and_percentile_plots_render(tmp_path: Path) -> None:
    arms = [
        {
            "arm_name": "static_t0",
            "mean_accuracy": 0.55,
            "mean_total_latency_ms": 1800.0,
            "p50_total_latency_ms": 1500.0,
            "p95_total_latency_ms": 2600.0,
        },
        {
            "arm_name": "static_t3",
            "mean_accuracy": 0.96,
            "mean_total_latency_ms": 14000.0,
            "p50_total_latency_ms": 11000.0,
            "p95_total_latency_ms": 24000.0,
        },
        {
            "arm_name": "decision_score_tail_cv",
            "mean_accuracy": 0.95,
            "mean_total_latency_ms": 4800.0,
            "p50_total_latency_ms": 2400.0,
            "p95_total_latency_ms": 15000.0,
        },
    ]
    sweep = [
        {
            "routine_s1_s2_share": r,
            "latency_by_arm_ms": {
                "static_t3": {"mean_ms": 14000.0, "p50_ms": 11000.0, "p95_ms": 24000.0},
                "decision_score_tail_cv": {
                    "mean_ms": 14400.0 - 11000.0 * r,
                    "p50_ms": 2400.0,
                    "p95_ms": 15000.0,
                },
            },
        }
        for r in (0.0, 0.5, 0.85, 1.0)
    ]
    out_curve = tmp_path / "lat_curve.png"
    plot_latency_breakeven(
        arms,
        {"s1_trivial_text": {"static_t3": 6000.0, "decision_score_tail_cv": 2200.0}},
        out_curve,
        target_arms=["static_t3", "decision_score_tail_cv"],
        breakeven_sweep=sweep,
    )
    assert out_curve.exists() and out_curve.stat().st_size > 1000

    out_pct = tmp_path / "lat_pct.png"
    plot_slice_latency_percentiles(
        {
            "s1_trivial_text": {
                "static_t3": {"mean_ms": 6000.0, "p50_ms": 5400.0, "p95_ms": 9200.0},
                "decision_score_tail_cv": {
                    "mean_ms": 2200.0,
                    "p50_ms": 1900.0,
                    "p95_ms": 3400.0,
                },
            }
        },
        out_pct,
        target_arms=["static_t3", "decision_score_tail_cv"],
    )
    assert out_pct.exists() and out_pct.stat().st_size > 1000


@pytest.mark.parametrize(
    (
        "verifier_conf",
        "prefill_tid",
        "prefill_p_cheap",
        "prefill_cost",
        "prefill_lat",
        "expected_tier",
        "expected_r_u",
        "expected_d_u",
        "expected_r_lat",
        "expected_d_lat",
    ),
    [
        # Pure cascade accepts when verifier >= threshold
        (0.92, None, 0.0, 0.0, 0.0, "t0", 0.002, 0.01, 50.0, 100.0),
        # Pure cascade escalates when verifier < threshold and pays both t0 + t1 downstream
        (0.40, None, 0.0, 0.0, 0.0, "t1", 0.002, 0.06, 50.0, 400.0),
        # Hybrid cascade skips t0 & verifier when prefill routes directly to t2
        (0.40, "t2", 0.20, 0.001, 40.0, "t2", 0.001, 0.20, 40.0, 800.0),
        # Hybrid cascade accepts t0 without running verifier when prefill confidence >= threshold
        (0.40, "t0", 0.88, 0.001, 40.0, "t0", 0.001, 0.01, 40.0, 100.0),
    ],
)
def test_cascade_outcome_pure_and_hybrid(
    verifier_conf: float,
    prefill_tid: str | None,
    prefill_p_cheap: float,
    prefill_cost: float,
    prefill_lat: float,
    expected_tier: str,
    expected_r_u: float,
    expected_d_u: float,
    expected_r_lat: float,
    expected_d_lat: float,
) -> None:
    matrix_by_item_tier = {
        "item_1": {
            "t0": [{"passed": True, "compute_units": 0.01, "latency_ms": 100.0}],
            "t1": [{"passed": True, "compute_units": 0.05, "latency_ms": 300.0}],
            "t2": [{"passed": True, "compute_units": 0.20, "latency_ms": 800.0}],
        }
    }
    verifier_rec = {
        "verifier": {
            "answers": {"draft_solves_request": {"noul": verifier_conf}},
            "compute_units": 0.002,
            "latency_ms": 50.0,
        }
    }
    outcome = _cascade_outcome(
        iid="item_1",
        verifier_rec=verifier_rec,
        matrix_by_item_tier=matrix_by_item_tier,
        cost_key="compute_units",
        cheap_tier_id="t0",
        escalation_tier_id="t1",
        accept_threshold=0.70,
        prefill_tid=prefill_tid,
        prefill_p_cheap=prefill_p_cheap,
        prefill_cost=prefill_cost,
        prefill_lat=prefill_lat,
    )
    assert (outcome.selected_tier, outcome.accuracy) == (expected_tier, 1.0)
    assert outcome.router_compute_units == pytest.approx(expected_r_u)
    assert outcome.downstream_compute_units == pytest.approx(expected_d_u)
    assert outcome.router_latency_ms == pytest.approx(expected_r_lat)
    assert outcome.downstream_latency_ms == pytest.approx(expected_d_lat)


def test_run_cascade_verifiers_writes_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio
    import json
    from types import SimpleNamespace

    from evals import routers as eval_routers

    dataset_path = tmp_path / "eval.jsonl"
    matrix_path = tmp_path / "matrix.jsonl"
    out_path = tmp_path / "verifiers.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "i1",
                "slice": "s1_trivial_text",
                "modality": "text",
                "prompt": "What is 2+2?",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    matrix_path.write_text(
        json.dumps(
            {
                "item_id": "i1",
                "tier_id": TIER_IDS[0],
                "replicate": 1,
                "response_text": "4",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    fake_resp = SimpleNamespace(
        answers={
            "draft_solves_request": SimpleNamespace(
                name="draft_solves_request",
                noul=0.9,
                probabilities={"0": 0.1, "1": 0.9},
            )
        },
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=2,
            total_tokens=12,
            thought_tokens=0,
        ),
    )

    async def _fake_eval(client: Any, payload: Any, questions: Any, *, model: str) -> Any:
        return fake_resp, 25.0, 1

    monkeypatch.setattr(eval_routers, "create_genai_client", lambda: object())
    monkeypatch.setattr(eval_routers, "disable_sdk_auto_retries", lambda c: None)
    monkeypatch.setattr(eval_routers, "_eval_decision_schema", _fake_eval)

    asyncio.run(
        eval_routers.run_cascade_verifiers(
            dataset_path, matrix_path, out_path, cheap_tier_id=TIER_IDS[0]
        )
    )
    lines = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 1
    assert lines[0]["item_id"] == "i1"
    assert lines[0]["verifier"]["answers"]["draft_solves_request"]["noul"] == pytest.approx(0.9)

    empty_matrix = tmp_path / "empty_matrix.jsonl"
    empty_matrix.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="No downstream drafts"):
        asyncio.run(
            eval_routers.run_cascade_verifiers(
                dataset_path, empty_matrix, out_path, cheap_tier_id=TIER_IDS[0]
            )
        )


def test_validate_jsonl_rows_rejects_missing_compute_units(tmp_path: Path) -> None:
    from evals.analyze import _MatrixRowSchema, _RouterRowSchema, _validate_jsonl_rows

    with pytest.raises(ValueError, match="compute_units"):
        _validate_jsonl_rows(
            [{"item_id": "i1", "tier_id": "t0", "passed": True, "latency_ms": 10.0}],
            _MatrixRowSchema,
            tmp_path / "matrix.jsonl",
        )

    with pytest.raises(ValueError, match="pava_projected"):
        _validate_jsonl_rows(
            [
                {
                    "item_id": "i1",
                    "decision_score": {"compute_units": 0.1, "latency_ms": 1.0, "answers": {}},
                    "decision_choice": {"compute_units": 0.1, "latency_ms": 1.0, "answers": {}},
                    "decision_noul": {"compute_units": 0.1, "latency_ms": 1.0, "answers": {}},
                    "flash_lite_json": {"compute_units": 0.1, "latency_ms": 1.0, "parsed": {}},
                }
            ],
            _RouterRowSchema,
            tmp_path / "routers.jsonl",
        )


def test_evaluate_arm_across_folds_pure_cascade_escalates_to_t1_not_fallback() -> None:
    from evals.analyze import _evaluate_arm_across_folds

    ladder = default_ladder()
    tids = ladder.ordered_tier_ids
    item_ids = [f"item_{i}" for i in range(4)]
    folds = [
        ([item_ids[0], item_ids[1]], [item_ids[2], item_ids[3]]),
        ([item_ids[2], item_ids[3]], [item_ids[0], item_ids[1]]),
    ]
    items_by_id = {
        iid: {"id": iid, "slice": "s1_trivial_text", "modality": "text", "prompt": "hi"}
        for iid in item_ids
    }

    matrix_by_item_tier: dict[str, dict[str, list[dict[str, Any]]]] = {
        iid: {
            tid: [
                {
                    "passed": tid != tids[0],
                    "compute_units": 0.001 * (idx + 1),
                    "latency_ms": 100.0 * (idx + 1),
                }
            ]
            for idx, tid in enumerate(tids)
        }
        for iid in item_ids
    }
    router_by_item = {
        iid: {
            "modality": "text",
            "known_input_tokens": 50,
            "decision_score": {
                "compute_units": 0.0001,
                "latency_ms": 20.0,
                "answers": {
                    "complexity_tier": {"score": 1.0, "probabilities": {"0": 0.2, "1": 0.8}},
                    "needs_precision": {"noul": 0.1},
                },
            },
            "decision_choice": {
                "compute_units": 0.0001,
                "latency_ms": 20.0,
                "answers": {"target_tier": {"choice": tids[1], "probabilities": {tids[1]: 1.0}}},
            },
            "decision_noul": {
                "compute_units": 0.0001,
                "latency_ms": 20.0,
                "answers": {
                    f"can_{tid}_solve": {"noul": 0.2 if tid == tids[0] else 0.95} for tid in tids
                },
                "pava_projected": {tid: (0.2 if tid == tids[0] else 0.95) for tid in tids},
            },
            "flash_lite_json": {
                "compute_units": 0.0001,
                "latency_ms": 20.0,
                "parsed": {
                    "choice_tier": tids[1],
                    "tier_probabilities": {tid: (1.0 if tid == tids[1] else 0.0) for tid in tids},
                    "solvability_probabilities": {tid: 0.9 for tid in tids},
                },
            },
        }
        for iid in item_ids
    }
    verifiers_by_item = {
        iid: {
            "item_id": iid,
            "verifier": {
                "compute_units": 0.0002,
                "latency_ms": 30.0,
                "answers": {"draft_solves_request": {"noul": 0.1}},
            },
        }
        for iid in item_ids
    }

    oof = _evaluate_arm_across_folds(
        folds,
        items_by_id,
        matrix_by_item_tier,
        router_by_item,
        eu_lambda=25.0,
        ladder=ladder,
        cascade_verifiers_by_item=verifiers_by_item,
    )
    assert all(rec["selected_tier"] == tids[1] for rec in oof["pure_cascade_t0_t1"].values())


def test_run_analysis_with_cascade_verifiers_includes_pareto_sweep(tmp_path: Path) -> None:
    import json

    from evals.analyze import run_analysis

    ladder = default_ladder()
    tids = ladder.ordered_tier_ids
    item_ids = [f"item_{i}" for i in range(10)]
    dataset_path = tmp_path / "eval.jsonl"
    matrix_path = tmp_path / "matrix.jsonl"
    routers_path = tmp_path / "routers.jsonl"
    verifiers_path = tmp_path / "verifiers.jsonl"
    summary_path = tmp_path / "summary.json"

    dataset_path.write_text(
        "\n".join(
            json.dumps({"id": iid, "slice": "s1_trivial_text", "modality": "text", "prompt": "hi"})
            for iid in item_ids
        )
        + "\n",
        encoding="utf-8",
    )
    matrix_rows = [
        {
            "item_id": iid,
            "tier_id": tid,
            "passed": tid != tids[0],
            "compute_units": 0.001 * (idx + 1),
            "latency_ms": 100.0 * (idx + 1),
        }
        for iid in item_ids
        for idx, tid in enumerate(tids)
    ]
    matrix_path.write_text("\n".join(json.dumps(r) for r in matrix_rows) + "\n", encoding="utf-8")
    router_rows = [
        {
            "item_id": iid,
            "slice": "s1_trivial_text",
            "modality": "text",
            "known_input_tokens": 50,
            "decision_score": {
                "compute_units": 0.0001,
                "latency_ms": 20.0,
                "answers": {
                    "complexity_tier": {"score": 1.0, "probabilities": {"0": 0.2, "1": 0.8}},
                    "needs_precision": {"noul": 0.1},
                },
            },
            "decision_choice": {
                "compute_units": 0.0001,
                "latency_ms": 20.0,
                "answers": {"target_tier": {"choice": tids[1], "probabilities": {tids[1]: 1.0}}},
            },
            "decision_noul": {
                "compute_units": 0.0001,
                "latency_ms": 20.0,
                "answers": {
                    f"can_{tid}_solve": {"noul": 0.2 if tid == tids[0] else 0.95} for tid in tids
                },
                "pava_projected": {tid: (0.2 if tid == tids[0] else 0.95) for tid in tids},
            },
            "flash_lite_json": {
                "compute_units": 0.0001,
                "latency_ms": 20.0,
                "parsed": {
                    "choice_tier": tids[1],
                    "tier_probabilities": {tid: (1.0 if tid == tids[1] else 0.0) for tid in tids},
                    "solvability_probabilities": {tid: 0.9 for tid in tids},
                },
            },
        }
        for iid in item_ids
    ]
    routers_path.write_text("\n".join(json.dumps(r) for r in router_rows) + "\n", encoding="utf-8")
    verifier_rows = [
        {
            "item_id": iid,
            "verifier": {
                "compute_units": 0.0002,
                "latency_ms": 30.0,
                "answers": {"draft_solves_request": {"noul": 0.1}},
            },
        }
        for iid in item_ids
    ]
    verifiers_path.write_text(
        "\n".join(json.dumps(r) for r in verifier_rows) + "\n", encoding="utf-8"
    )

    summary = run_analysis(
        dataset_path,
        matrix_path,
        routers_path,
        summary_path,
        ladder=ladder,
        cascade_verifiers_path=verifiers_path,
    )
    assert "pure_cascade_t0_t1" in summary["pareto_lambda_sweep"]
    assert "hybrid_cascade_router" in summary["pareto_lambda_sweep"]
