"""Matplotlib chart generators for Pareto accuracy-cost curves, latency break-even, and reliability diagrams."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt

from evals.analyze import REPO_ROOT, compute_calibration_metrics, compute_static_pareto_hull

matplotlib.use("Agg")


def plot_pareto_frontier(
    arm_points: list[dict[str, Any]],
    out_path: Path,
    *,
    title: str = "Accuracy vs. Total Compute Units per 1,000 Requests",
) -> None:
    """Render a Pareto accuracy-vs-compute scatter and curve diagram with the static no-router hull."""
    fig, ax = plt.subplots(figsize=(9.5, 6), dpi=200)

    static_pts = [pt for pt in arm_points if str(pt.get("arm_name", "")).startswith("static_")]
    if len(static_pts) >= 2:
        normalized_statics = [
            {
                **pt,
                "compute_units_per_1k": float(
                    pt.get("compute_units_per_1k", float(pt["mean_compute_units"]) * 1000.0)
                ),
            }
            for pt in static_pts
        ]
        hull = compute_static_pareto_hull(
            normalized_statics, x_key="compute_units_per_1k", y_key="mean_accuracy"
        )
        ax.plot(
            [float(p["compute_units_per_1k"]) for p in hull],
            [float(p["mean_accuracy"]) * 100.0 for p in hull],
            linestyle="--",
            linewidth=1.8,
            color="#555555",
            alpha=0.85,
            label="Static No-Router Hull (above/left = +Net Value)",
        )

    for pt in arm_points:
        units_per_1k = float(
            pt.get("compute_units_per_1k", float(pt["mean_compute_units"]) * 1000.0)
        )
        acc_pct = float(pt["mean_accuracy"]) * 100.0
        label = str(pt["arm_name"])
        marker = "D" if "decision" in label else ("s" if "static" in label else "o")
        ax.scatter(units_per_1k, acc_pct, s=75, marker=marker)
        ax.annotate(
            label,
            (units_per_1k, acc_pct),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )

    if len(static_pts) >= 2:
        ax.legend(loc="lower right", fontsize=8.5)

    ax.set_xlabel("Total Compute Units per 1,000 Requests (Router + Downstream)", fontsize=11)
    ax.set_ylabel("Benchmark Accuracy (%)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_latency_breakeven(
    arm_points: list[dict[str, Any]],
    latency_by_slice: dict[str, dict[str, float]],
    out_path: Path,
    *,
    title: str = "End-to-End Serial Latency Pareto & Traffic-Skew Break-Even (ms)",
    target_arms: Sequence[str] | None = None,
    breakeven_sweep: Sequence[dict[str, Any]] | None = None,
) -> None:
    """Render (left) Accuracy vs E2E Latency with [p50, p95] spans and (right) Serial-Latency Break-Even Curve.

    When `breakeven_sweep` (`net_value_summary["easy_traffic_breakeven_sweep"]`) is supplied, the right
    panel traces continuous E2E serial latency across 0%..100% routine (`s1+s2`) traffic share; otherwise
    it falls back to per-slice grouped latency bars from `latency_by_slice`.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.8), dpi=200)
    fig.suptitle(title, fontsize=12, fontweight="bold")

    static_pts = [pt for pt in arm_points if str(pt.get("arm_name", "")).startswith("static_")]
    if len(static_pts) >= 2:
        hull = compute_static_pareto_hull(
            static_pts, x_key="mean_total_latency_ms", y_key="mean_accuracy"
        )
        ax1.plot(
            [float(p["mean_total_latency_ms"]) for p in hull],
            [float(p["mean_accuracy"]) * 100.0 for p in hull],
            linestyle="--",
            linewidth=1.6,
            color="#555555",
            alpha=0.8,
            label="Static No-Router Hull",
        )

    for pt in arm_points:
        lat_ms = float(pt.get("mean_total_latency_ms", 0.0))
        acc_pct = float(pt["mean_accuracy"]) * 100.0
        label = str(pt["arm_name"])
        marker = "D" if "decision" in label else ("s" if "static" in label else "o")
        if "p50_total_latency_ms" in pt and "p95_total_latency_ms" in pt:
            p50 = float(pt["p50_total_latency_ms"])
            p95 = float(pt["p95_total_latency_ms"])
            ax1.hlines(acc_pct, p50, p95, colors="#999999", alpha=0.35, linewidth=1.4)
        ax1.scatter(lat_ms, acc_pct, s=65, marker=marker)
        ax1.annotate(
            label, (lat_ms, acc_pct), textcoords="offset points", xytext=(5, 3), fontsize=7.5
        )

    if len(static_pts) >= 2:
        ax1.legend(loc="lower right", fontsize=8)
    ax1.set_xlabel("Mean End-to-End Latency (ms; whiskers = p50..p95)", fontsize=10)
    ax1.set_ylabel("Benchmark Accuracy (%)", fontsize=10)
    ax1.set_title("Accuracy vs. End-to-End Serial Latency", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)

    if breakeven_sweep:
        xs_pct = [float(row["routine_s1_s2_share"]) * 100.0 for row in breakeven_sweep]
        first_arms = breakeven_sweep[0].get("latency_by_arm_ms", {})
        curve_arms = (
            [a for a in target_arms if a in first_arms]
            if target_arms is not None
            else list(first_arms.keys())
        )
        for arm in curve_arms:
            means = [float(row["latency_by_arm_ms"][arm]["mean_ms"]) for row in breakeven_sweep]
            ls = "--" if arm.startswith("static_") else ("-" if "decision" in arm else "-.")
            mk = "s" if arm.startswith("static_") else ("D" if "decision" in arm else "o")
            ax2.plot(
                xs_pct, means, marker=mk, linestyle=ls, linewidth=2.0, markersize=4.5, label=arm
            )
        ax2.axvline(
            85.0,
            linestyle=":",
            color="#333333",
            linewidth=1.4,
            label="Production Skew (85% s1+s2)",
        )
        ax2.set_xlim(0.0, 100.0)
        ax2.set_xlabel("Routine Traffic Share (s1 + s2, %)", fontsize=10)
        ax2.set_ylabel("Mean End-to-End Serial Latency (ms)", fontsize=10)
        ax2.set_title("Serial-Latency Break-Even vs. Routine Share", fontsize=11)
        ax2.legend(fontsize=8)
        ax2.grid(True, linestyle="--", alpha=0.5)
    else:
        slices = sorted(latency_by_slice.keys())
        arms = (
            list(target_arms)
            if target_arms is not None
            else list(dict.fromkeys(arm for sl in slices for arm in latency_by_slice[sl]))
        )
        width = min(0.25, 0.8 / max(1, len(arms)))
        for idx, arm in enumerate(arms):
            vals = [float(latency_by_slice[sl].get(arm, 0.0)) for sl in slices]
            xs = [i + (idx - (len(arms) - 1) / 2) * width for i in range(len(slices))]
            ax2.bar(xs, vals, width=width, label=arm)

        ax2.set_xticks(range(len(slices)))
        ax2.set_xticklabels([s.replace("_", "\n") for s in slices], fontsize=8)
        ax2.set_ylabel("Mean End-to-End Latency (ms)", fontsize=10)
        ax2.set_title("Per-Slice E2E Latency Break-Even", fontsize=11)
        ax2.legend(fontsize=8)
        ax2.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_slice_latency_percentiles(
    latency_percentiles_by_slice: dict[str, dict[str, dict[str, float]]],
    out_path: Path,
    *,
    target_arms: Sequence[str] | None = None,
    title: str = "Per-Slice End-to-End Latency: Median (p50), Mean, and Tail (p95)",
) -> None:
    """Render grouped per-slice E2E latency bars (p50 height with p95 upper whiskers and mean markers)."""
    slices = sorted(latency_percentiles_by_slice.keys())
    if not slices:
        return
    arms = (
        [a for a in target_arms if any(a in latency_percentiles_by_slice[sl] for sl in slices)]
        if target_arms is not None
        else list(dict.fromkeys(arm for sl in slices for arm in latency_percentiles_by_slice[sl]))
    )
    if not arms:
        return

    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=200)
    width = min(0.24, 0.8 / max(1, len(arms)))
    for idx, arm in enumerate(arms):
        xs = [i + (idx - (len(arms) - 1) / 2) * width for i in range(len(slices))]
        p50s = [
            float(latency_percentiles_by_slice[sl].get(arm, {}).get("p50_ms", 0.0)) for sl in slices
        ]
        p95s = [
            float(latency_percentiles_by_slice[sl].get(arm, {}).get("p95_ms", 0.0)) for sl in slices
        ]
        means = [
            float(latency_percentiles_by_slice[sl].get(arm, {}).get("mean_ms", 0.0))
            for sl in slices
        ]
        upper_err = [max(0.0, hi - med) for med, hi in zip(p50s, p95s, strict=False)]
        ax.bar(
            xs,
            p50s,
            width=width,
            yerr=[[0.0] * len(slices), upper_err],
            capsize=3,
            alpha=0.85,
            label=f"{arm} (bar=p50, cap=p95)",
        )
        ax.scatter(xs, means, color="#111111", marker="d", s=22, zorder=4)

    ax.set_xticks(range(len(slices)))
    ax.set_xticklabels([s.replace("_", "\n") for s in slices], fontsize=8.5)
    ax.set_ylabel("End-to-End Serial Latency (ms)", fontsize=10)
    ax.set_title(title, fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_reliability_diagram(
    predictions: list[tuple[float, float]] | None,
    out_path: Path,
    *,
    num_bins: int = 10,
    cal_metrics: dict[str, Any] | None = None,
    comparison_cal_metrics: dict[str, Any] | None = None,
    title: str = "Solvability Calibration: gemini-decision-1 vs. flash-lite-json",
) -> None:
    """Render a 10-bin reliability diagram using `compute_calibration_metrics` bins (zero duplicated binning)."""
    primary = (
        cal_metrics
        if cal_metrics is not None
        else compute_calibration_metrics(predictions or [], num_bins=num_bins)
    )
    bins = primary.get("bins", [])
    xs = [float(b["mean_pred"]) for b in bins]
    ys = [float(b["mean_emp"]) for b in bins]
    counts = [int(b["count"]) for b in bins]

    fig, ax = plt.subplots(figsize=(6.8, 6.5), dpi=200)
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", label="Perfect Calibration")
    ax.plot(
        xs,
        ys,
        marker="o",
        linewidth=2,
        color="#1f77b4",
        label=f"gemini-decision-1 (ECE={primary.get('ece', 0.0):.4f})",
    )
    for x, y, c in zip(xs, ys, counts, strict=False):
        ax.annotate(f"n={c}", (x, y), textcoords="offset points", xytext=(4, -10), fontsize=8)

    if comparison_cal_metrics and comparison_cal_metrics.get("bins"):
        c_bins = comparison_cal_metrics["bins"]
        c_xs = [float(b["mean_pred"]) for b in c_bins]
        c_ys = [float(b["mean_emp"]) for b in c_bins]
        ax.plot(
            c_xs,
            c_ys,
            marker="s",
            linewidth=2,
            linestyle="-.",
            color="#d62728",
            label=f"flash-lite-json (ECE={comparison_cal_metrics.get('ece', 0.0):.4f})",
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Predicted Solvability Probability", fontsize=11)
    ax.set_ylabel("Empirical Downstream Pass Rate", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render evals charts from summary.json.")
    parser.add_argument("--summary", type=Path, default=REPO_ROOT / "results/summary.json")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results/plots")
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ladder_info: dict[str, Any] = summary.get("ladder", {})
    if "balanced_equal_weights" in summary:
        plot_pareto_frontier(
            list(summary["balanced_equal_weights"].values()),
            out_dir / "pareto_frontier.png",
            title="Accuracy vs. Compute Units per 1,000 Requests (Balanced)",
        )
    if "production_skew_60_25_10_5" in summary:
        plot_pareto_frontier(
            list(summary["production_skew_60_25_10_5"].values()),
            out_dir / "pareto_frontier_skew.png",
            title="Accuracy vs. Compute Units per 1,000 Requests (60/25/10/5 Skew)",
        )

    lat_arm_map = summary.get("production_skew_60_25_10_5") or summary.get("balanced_equal_weights")
    if lat_arm_map:
        baseline = ladder_info.get("baseline_tier")
        target_arms = [
            *([f"static_{baseline}"] if baseline else []),
            "decision_score_tail_cv",
            "decision_noul_eu_cv",
            "flash_lite_tail_cv",
        ]
        plot_latency_breakeven(
            list(lat_arm_map.values()),
            summary.get("latency_by_slice_ms", {}),
            out_dir / "latency_pareto_and_breakeven.png",
            target_arms=target_arms,
            breakeven_sweep=summary.get("net_value_summary", {}).get(
                "easy_traffic_breakeven_sweep"
            ),
        )
        if "latency_percentiles_by_slice_ms" in summary:
            plot_slice_latency_percentiles(
                summary["latency_percentiles_by_slice_ms"],
                out_dir / "latency_by_slice_p50_p95.png",
                target_arms=target_arms,
            )

    q3 = summary.get("q3_calibration", {})
    cal_primary = q3.get("pava_pooled_10bin") or q3.get("pava_750_10bin")
    cal_comp = q3.get("flash_lite_pava_pooled_10bin") or q3.get("flash_lite_pava_750_10bin")
    if cal_primary is not None:
        plot_reliability_diagram(
            None,
            out_dir / "calibration_reliability.png",
            cal_metrics=cal_primary,
            comparison_cal_metrics=cal_comp,
        )


if __name__ == "__main__":
    main()
