"""Deterministic generator for the evals splits, their multimodal assets, and `evals/DATASET.md`.

Three splits are written to `data/`:

- `pilot.jsonl` (5 items per slice, seed `20261101`): pre-flight checks only.
- `eval.jsonl` (30 items per slice, seed `20261111`): the evaluation set.
- `calibration.jsonl` (21 items, seed `20261121`): every capability-gated subtype at least once,
  used to confirm tier separation before spending quota on the full benchmark.

Every item draws from its own `random.Random(f"{seed}:{item_id}")`, so editing one generator never
changes the items another one produces. The capability-gated generators live in `evals.gen`;
this module keeps the easy items (all of s1, the even-indexed halves of s4 and s5) and the split
wiring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from evals.gen import (
    item_rng as _item_rng,
)
from evals.gen import (
    s1_trivial,
    s2_specs,
    s3_repair,
    s3_timeline,
    s3_vm,
    s4_vision,
    s5_dialogue,
)

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
ASSETS_DIR = DATA_DIR / "assets"
IMG_DIR = ASSETS_DIR / "images"
PDF_DIR = ASSETS_DIR / "pdfs"
AUDIO_DIR = ASSETS_DIR / "audio"
DATASET_MD_PATH = REPO_ROOT / "evals" / "DATASET.md"

PILOT_SEED = 20261101
BENCH_SEED = 20261111
CALIB_SEED = 20261121

ENGLISH_TTS_VOICES = ("Samantha", "Daniel", "Karen", "Moira", "Rishi")


def _asset_path(kind: str, item_id: str, ext: str) -> tuple[Path, str]:
    """Return the `(absolute, dataset-relative)` paths of an item's asset, creating its directory."""
    rel = f"data/assets/{kind}/{item_id}.{ext}"
    path = REPO_ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    return path, rel


def _clear_assets(prefix: str) -> None:
    """Delete previously generated assets of a split so a rebuild leaves no stale files behind."""
    for asset_dir in (IMG_DIR, PDF_DIR, AUDIO_DIR):
        if asset_dir.is_dir():
            for stale in asset_dir.glob(f"{prefix}_*"):
                stale.unlink()


def _s2_item(seed: int, prefix: str, i: int) -> dict[str, Any]:
    """Item `i` of `s2_everyday_code`: spec variant `i % 5`, graded by its asserts plus a hidden differential suite."""
    item_id = f"{prefix}_s2_{i:02d}"
    return s2_specs.make_s2_item(_item_rng(seed, item_id), item_id, i)


def _make_s2_items(seed: int, count: int, prefix: str) -> list[dict[str, Any]]:
    """Generate `s2_everyday_code` items with boundary-condition-rich single-function engineering specs."""
    return [_s2_item(seed, prefix, i) for i in range(count)]


# The s3 families rotate by item index; `i // 3` is the item's index within its family.
_S3_FAMILIES: tuple[Callable[[random.Random, str, int], dict[str, Any]], ...] = (
    s3_repair.make_repair_item,
    s3_timeline.make_timeline_item,
    s3_vm.make_vm_item,
)


def _s3_item(seed: int, prefix: str, i: int) -> dict[str, Any]:
    """Item `i` of `s3_frontier_reasoning`: symptom-only repair, long-context timeline, or VM spec (`i % 3`)."""
    item_id = f"{prefix}_s3_{i:02d}"
    family = _S3_FAMILIES[i % len(_S3_FAMILIES)]
    return family(_item_rng(seed, item_id), item_id, i // len(_S3_FAMILIES))


def _make_s3_items(seed: int, count: int, prefix: str) -> list[dict[str, Any]]:
    """Generate `s3_frontier_reasoning` items from the three capability-gated families in `evals.gen`."""
    return [_s3_item(seed, prefix, i) for i in range(count)]


def _generate_chart_image(
    img_path: Path,
    title: str,
    quarters: list[str],
    alpha_rev: list[float],
    alpha_cost: list[float],
    beta_util: list[int],
    beta_p99_ms: list[int],
    footnote_code: int,
    util_thresh: int,
    lat_thresh: int,
    penalty_per_q: float,
    reserve_haircut: float,
) -> None:
    """Render a high-resolution (2400x1600) dual-panel financial & SLA telemetry audit chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), dpi=200)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax1.plot(quarters, alpha_rev, marker="o", linewidth=2, label="Alpha Gross Rev ($M)")
    ax1.plot(quarters, alpha_cost, marker="s", linewidth=2, label="Alpha Op Cost ($M)")
    for q, r, c in zip(quarters, alpha_rev, alpha_cost, strict=False):
        ax1.annotate(f"{r:.1f}", (q, r), textcoords="offset points", xytext=(0, 6), fontsize=8.0)
        ax1.annotate(f"{c:.1f}", (q, c), textcoords="offset points", xytext=(0, -12), fontsize=8.0)
    ax1.set_title("Panel A — Region Alpha Revenue & Cost ($M)", fontsize=10.5)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)

    x_pos = list(range(len(quarters)))
    bars = ax2.bar(x_pos, beta_util, color="#2b5c8f", width=0.55, label="Cluster Util (%)")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(quarters)
    ax2.set_ylim(0, 115)
    ax2.set_title("Panel B — Cluster Utilization (%) & p99 Latency (ms)", fontsize=10.5)
    for bar, u_val, l_val in zip(bars, beta_util, beta_p99_ms, strict=False):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            u_val + 1.5,
            f"U:{u_val}%\nL:{l_val}ms",
            ha="center",
            fontsize=6.5,
        )

    fine_text = (
        f"AUDIT FOOTNOTE #{footnote_code} — RECONCILIATION PROTOCOL\n"
        f"1. Compute Alpha Net Margin = Rev - Cost for each quarter Q1..Q6,\n"
        f"   EXCLUDING any quarter where Alpha Op Cost > 38.0M.\n"
        f"2. Count SLA Breach Quarters in Panel B where BOTH:\n"
        f"   Cluster Util >= {util_thresh}% AND p99 Latency > {lat_thresh}ms.\n"
        f"3. Subtract ({penalty_per_q:.1f}M * breach_count) and subtract Reserve = {reserve_haircut:.1f}M."
    )
    ax2.text(
        0.97,
        0.03,
        fine_text,
        transform=ax2.transAxes,
        fontsize=5.8,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "#fff8dc", "alpha": 0.94},
    )
    plt.tight_layout()
    fig.savefig(img_path, dpi=200)
    plt.close(fig)


def _generate_multipage_pdf(
    pdf_path: Path,
    doc_title: str,
    total_pages: int,
    cluster_id: str,
    primary_zone: str,
    headline_budget_mwh: int,
    subclusters: list[tuple[str, int, str, int]],
    min_eff_pct: int,
    credit_per_qual_sc: int,
    surcharge_mwh: int,
) -> None:
    """Create a multi-page PDF with Page-1 headline summary AND multi-page cross-table reconciliation across Pages 2, 3, and final Appendix."""
    target_page_idx = total_pages - 1
    with PdfPages(pdf_path) as pdf:
        for p in range(total_pages):
            header = f"{doc_title} — Page {p + 1} of {total_pages}"
            if p == 0:
                body = (
                    f"Executive Summary — Cluster {cluster_id}\n\n"
                    f"- Primary Availability Zone: {primary_zone}\n"
                    f"- Headline Allocated Budget (MWh): {headline_budget_mwh}\n"
                    f"- Total Report Length: {total_pages} pages\n\n"
                    f"Note for Auditors: Metered gross loads and operational statuses appear in Table 2A on Page 2, "
                    f"thermal efficiency ratings appear in Table 3B on Page 3, and the final reconciliation rules "
                    f"appear in Appendix Schedule C on Page {target_page_idx + 1}."
                )
            elif p == 1:
                lines = [
                    f"Table 2A — Sub-Cluster Gross Metered Consumption ({cluster_id})",
                    "(Status codes: ACTIVE, STANDBY_EXEMPT, DECOMMISSIONED)",
                    "",
                ]
                for sc_name, sc_val, status, _eff in subclusters:
                    lines.append(f"  * {sc_name}: {sc_val} MWh | Status = {status}")
                body = "\n".join(lines)
            elif p == 2:
                lines = [
                    f"Table 3B — Sub-Cluster Thermal Efficiency Audit ({cluster_id})",
                    f"(Minimum Qualifying Efficiency Floor for Renewable Rebate = {min_eff_pct}%)",
                    "",
                ]
                for sc_name, _sc_val, _status, eff in subclusters:
                    lines.append(f"  * {sc_name}: Thermal Efficiency = {eff}%")
                body = "\n".join(lines)
            elif p == target_page_idx:
                body = (
                    f"Appendix Schedule C — Multi-Table Reconciliation Protocol ({cluster_id})\n\n"
                    f"1. Active Billed MWh (`active_subcluster_mwh`):\n"
                    f"   Sum gross MWh from Table 2A ONLY for sub-clusters with Status == ACTIVE\n"
                    f"   (exclude both DECOMMISSIONED and STANDBY_EXEMPT).\n\n"
                    f"2. Qualifying Renewable Credit:\n"
                    f"   Each sub-cluster that has Status == ACTIVE on Page 2 AND Thermal Efficiency >= {min_eff_pct}%\n"
                    f"   on Page 3 earns a credit of {credit_per_qual_sc} MWh.\n\n"
                    f"3. Reconciled Net Billed MWh (`reconciled_net_mwh`):\n"
                    f"   active_subcluster_mwh - (qualifying_active_count * {credit_per_qual_sc}) + {surcharge_mwh}"
                )
            else:
                body = (
                    f"Section {p + 1}: Harmonic distortion and transformer impedance logs for {cluster_id}.\n"
                    f"No auxiliary breaker trips recorded during this interval."
                )

            wrapped_body = "\n".join(
                textwrap.fill(line, width=76) if line else "" for line in body.splitlines()
            )
            fig = plt.figure(figsize=(8.5, 11.0))
            try:
                fig.text(0.08, 0.93, header, fontsize=13, fontweight="bold", va="top")
                fig.text(0.08, 0.87, wrapped_body, fontsize=11, va="top")
                pdf.savefig(fig)
            finally:
                plt.close(fig)


def _easy_chart_item(
    rng: random.Random, item_id: str, i: int, img_abs: Path, img_rel: str
) -> dict[str, Any]:
    """Render the labelled dual-panel audit chart and ask for one printed value (Panel A, Q6 revenue)."""
    quarters = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
    alpha_rev = [round(rng.uniform(52.0, 96.0), 1) for _ in range(6)]
    # Guarantee some costs above and below 38.0M
    alpha_cost = [
        round(rng.uniform(24.0, 36.5), 1),
        round(rng.uniform(39.0, 44.5), 1),
        round(rng.uniform(25.0, 35.5), 1),
        round(rng.uniform(38.5, 43.0), 1),
        round(rng.uniform(22.0, 34.0), 1),
        round(rng.uniform(26.0, 37.0), 1),
    ]
    beta_util = [rng.randint(68, 96) for _ in range(6)]
    beta_p99_ms = [rng.randint(140, 290) for _ in range(6)]
    fn_code = rng.randint(100, 999)
    util_thresh = 80
    lat_thresh = 210
    penalty_per_q = round(rng.uniform(3.5, 7.5), 1)
    reserve_haircut = round(rng.uniform(2.0, 6.5), 1)

    _generate_chart_image(
        img_abs,
        title=f"Division {i + 1} Telemetry & Financial Audit",
        quarters=quarters,
        alpha_rev=alpha_rev,
        alpha_cost=alpha_cost,
        beta_util=beta_util,
        beta_p99_ms=beta_p99_ms,
        footnote_code=fn_code,
        util_thresh=util_thresh,
        lat_thresh=lat_thresh,
        penalty_per_q=penalty_per_q,
        reserve_haircut=reserve_haircut,
    )
    prompt = (
        "Inspect Panel A of the attached chart ('Region Alpha Revenue & Cost ($M)'). "
        "What is the exact Q6 Gross Revenue (in $M) for Region Alpha? "
        "End your response with `FINAL_ANSWER: <number>`."
    )
    return {
        "id": item_id,
        "slice": "s4_visual_and_pdf",
        "modality": "image",
        "subtype": "chart_headline_lookup",
        "source_provenance": "High-res Matplotlib dual-panel financial/telemetry chart generator",
        "prompt": prompt,
        "attachments": [img_rel],
        "grader_type": "numeric_tolerance",
        "grader_spec": {"expected": alpha_rev[-1], "abs_tol": 0.15, "rel_tol": 1e-2},
    }


_PDF_ZONES = ("us-east1-b", "europe-west4-a", "asia-northeast1-c", "us-central1-f")


def _easy_pdf_item(
    rng: random.Random, item_id: str, j: int, pdf_abs: Path, pdf_rel: str
) -> dict[str, Any]:
    """Render the text-layer multi-page energy audit and ask for its Page-1 executive summary fields."""
    total_pages = 5 + (j % 3)
    cluster_id = f"CL-{rng.randint(100, 999)}"
    zone = _PDF_ZONES[j % len(_PDF_ZONES)]
    headline_budget = rng.randint(4000, 9500)
    min_eff_pct = 88
    sc_vals = [rng.randint(550, 1450) for _ in range(6)]
    subclusters = [
        ("SC-1", sc_vals[0], "ACTIVE", 91),
        ("SC-2", sc_vals[1], "STANDBY_EXEMPT", 94),
        ("SC-3", sc_vals[2], "ACTIVE", 84),
        ("SC-4", sc_vals[3], "DECOMMISSIONED", 92),
        ("SC-5", sc_vals[4], "ACTIVE", 89),
        ("SC-6", sc_vals[5], "ACTIVE", 86),
    ]
    credit_per_qual_sc = rng.randint(110, 240)
    surcharge_mwh = rng.randint(45, 160)

    _generate_multipage_pdf(
        pdf_abs,
        doc_title=f"Infrastructure Energy Audit ({item_id})",
        total_pages=total_pages,
        cluster_id=cluster_id,
        primary_zone=zone,
        headline_budget_mwh=headline_budget,
        subclusters=subclusters,
        min_eff_pct=min_eff_pct,
        credit_per_qual_sc=credit_per_qual_sc,
        surcharge_mwh=surcharge_mwh,
    )
    prompt = (
        "Read the Executive Summary on Page 1 of the attached PDF and return ONLY a JSON object "
        "with keys `cluster_id` (string), `primary_zone` (string), and `headline_budget_mwh` (int)."
    )
    return {
        "id": item_id,
        "slice": "s4_visual_and_pdf",
        "modality": "pdf",
        "subtype": "pdf_page1_executive_lookup",
        "source_provenance": "Programmatic multi-page operational & financial reconciliation report",
        "prompt": prompt,
        "attachments": [pdf_rel],
        "grader_type": "exact_json",
        "grader_spec": {
            "expected": {
                "cluster_id": cluster_id,
                "primary_zone": zone,
                "headline_budget_mwh": headline_budget,
            }
        },
    }


def _s4_image_item(seed: int, prefix: str, i: int) -> dict[str, Any]:
    """Image item `i`: even = labelled-chart lookup; odd = perception-gated image (`hard_index = i // 2`)."""
    item_id = f"{prefix}_s4_img_{i:02d}"
    img_abs, img_rel = _asset_path("images", item_id, "png")
    rng = _item_rng(seed, item_id)
    if i % 2 == 0:
        return _easy_chart_item(rng, item_id, i, img_abs, img_rel)
    return s4_vision.make_hard_image_item(rng, item_id, i // 2, img_abs, img_rel)


def _s4_pdf_item(seed: int, prefix: str, j: int) -> dict[str, Any]:
    """PDF item `j`: even = Page-1 text-layer lookup; odd = scanned errata audit packet (`hard_index = j // 2`)."""
    item_id = f"{prefix}_s4_pdf_{j:02d}"
    pdf_abs, pdf_rel = _asset_path("pdfs", item_id, "pdf")
    rng = _item_rng(seed, item_id)
    if j % 2 == 0:
        return _easy_pdf_item(rng, item_id, j, pdf_abs, pdf_rel)
    return s4_vision.make_hard_pdf_item(rng, item_id, j // 2, pdf_abs, pdf_rel)


def _make_s4_items(seed: int, count: int, prefix: str) -> list[dict[str, Any]]:
    """Generate `s4_visual_and_pdf` items: half simple lookups, half perception-gated images and scanned PDFs."""
    n_imgs = (count + 1) // 2
    items = [_s4_image_item(seed, prefix, i) for i in range(n_imgs)]
    items.extend(_s4_pdf_item(seed, prefix, j) for j in range(count - n_imgs))
    return items


_DISPATCH_SECTORS = ("North", "South", "East", "West", "Central")
_DISPATCH_PRIORITIES = ("Low", "Medium", "High", "Critical")


def _easy_dispatch_item(
    rng: random.Random, item_id: str, i: int, wav_abs: Path, wav_rel: str
) -> dict[str, Any]:
    """Synthesize a single-speaker dispatch notice (stated twice) and ask for its three fields."""
    voice = ENGLISH_TTS_VOICES[i % len(ENGLISH_TTS_VOICES)]
    unit_num = rng.randint(11, 89)
    sector = _DISPATCH_SECTORS[i % len(_DISPATCH_SECTORS)]
    priority = _DISPATCH_PRIORITIES[(i // 2) % len(_DISPATCH_PRIORITIES)]
    spoken_script = (
        f"Dispatch notice for field unit {unit_num}. "
        f"Proceed to sector {sector} and set your status priority to {priority}. "
        f"Repeating: unit {unit_num}, sector {sector}, priority {priority}."
    )
    s5_dialogue.say_to_wav(spoken_script, voice, wav_abs)
    prompt = (
        "Listen to the attached spoken dispatch audio recording and return ONLY a JSON object "
        "with keys `unit_id` (int), `sector` (string), and `priority` (string)."
    )
    return {
        "id": item_id,
        "slice": "s5_spoken_audio",
        "modality": "audio",
        "subtype": "spoken_dispatch_json_extraction",
        "source_provenance": f"Synthesized 16kHz spoken English dispatch clip via macOS TTS voice={voice}",
        "prompt": prompt,
        "attachments": [wav_rel],
        "grader_type": "exact_json",
        "grader_spec": {
            "expected": {
                "unit_id": unit_num,
                "sector": sector,
                "priority": priority,
            }
        },
    }


def _s5_item(seed: int, prefix: str, i: int) -> dict[str, Any]:
    """Audio item `i`: even = spoken dispatch lookup; odd = three-speaker handoff call (`hard_index = i // 2`)."""
    item_id = f"{prefix}_s5_aud_{i:02d}"
    wav_abs, wav_rel = _asset_path("audio", item_id, "wav")
    rng = _item_rng(seed, item_id)
    if i % 2 == 0:
        return _easy_dispatch_item(rng, item_id, i, wav_abs, wav_rel)
    return s5_dialogue.make_hard_audio_item(rng, item_id, i // 2, wav_abs, wav_rel)


def _make_s5_items(seed: int, count: int, prefix: str) -> list[dict[str, Any]]:
    """Generate `s5_spoken_audio` items with real spoken English WAV files (zero transcript leakage in text prompt)."""
    return [_s5_item(seed, prefix, i) for i in range(count)]


def _write_jsonl(out_path: Path, items: list[dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def build_split(
    seed: int, items_per_slice: int, prefix: str, out_path: Path
) -> list[dict[str, Any]]:
    """Build a complete stratified split (`pilot` or `main`) and write it to `out_path`."""
    _clear_assets(prefix)
    all_items: list[dict[str, Any]] = []
    all_items.extend(dict(it) for it in s1_trivial.make_s1_items(seed, items_per_slice, prefix))
    all_items.extend(_make_s2_items(seed, items_per_slice, prefix))
    all_items.extend(_make_s3_items(seed, items_per_slice, prefix))
    all_items.extend(_make_s4_items(seed, items_per_slice, prefix))
    all_items.extend(_make_s5_items(seed, items_per_slice, prefix))
    _write_jsonl(out_path, all_items)
    return all_items


def build_calib_split(
    seed: int,
    prefix: str,
    out_path: Path,
    *,
    s2_count: int = 5,
    s3_count: int = 6,
    s4_image_count: int = 6,
    s4_pdf_count: int = 2,
    s5_count: int = 2,
) -> list[dict[str, Any]]:
    """Build the hard-only calibration split and write it to `out_path`.

    Item indexes keep the parity convention of the full splits (odd s4/s5 indexes are hard), so the
    default 21 items cover: the 5 s2 spec variants; 2 items per s3 family; s4 images with
    `hard_index` 0-5 (digraph, chart max-gap month, scatter, digraph, chart months-above,
    scatter); 2 scanned PDFs; and 2 three-speaker handoff calls.
    """
    _clear_assets(prefix)
    items: list[dict[str, Any]] = []
    items.extend(_s2_item(seed, prefix, i) for i in range(s2_count))
    items.extend(_s3_item(seed, prefix, i) for i in range(s3_count))
    items.extend(_s4_image_item(seed, prefix, 1 + 2 * k) for k in range(s4_image_count))
    items.extend(_s4_pdf_item(seed, prefix, 1 + 2 * k) for k in range(s4_pdf_count))
    items.extend(_s5_item(seed, prefix, 1 + 2 * k) for k in range(s5_count))
    _write_jsonl(out_path, items)
    return items


# (slice, modality, task subtypes, graders) for the DATASET.md slice summary.
_SLICE_SUMMARY: tuple[tuple[str, str, str, str], ...] = (
    (
        "s1_trivial_text",
        "Text",
        "Incident JSON extraction, IFEval bullet/lexical rules, ISO-8601 date normalization",
        "`exact_json`, `ifeval_constraints`, `canonical_string`",
    ),
    (
        "s2_everyday_code",
        "Code",
        "5 single-function specs (RFC-style quoted-row tokenizer, sliding-window alert "
        "deduplication, nested macro validation, span-capped interval merge, SemVer 2.0 "
        "pre-release resolution). Each prompt states every tested behaviour; grading runs the "
        "hand-written asserts plus a hidden differential suite of ~64 reference-generated cases",
        "`python_unit_tests` (air-gapped container)",
    ),
    (
        "s3_frontier_reasoning",
        "Text + Code",
        "3 capability-gated families in rotation: symptom-only repair of an original cron "
        "evaluator with 3 injected bugs (1 unreported) and a hidden differential suite; state "
        "tracking over a ~17K-token operations change log with ID-only corrections/voids and "
        "replayed entries; implementing a long-spec stack VM graded on generated programs "
        "against a reference interpreter",
        "`python_unit_tests`, `exact_json`",
    ),
    (
        "s4_visual_and_pdf",
        "Image + PDF",
        "Images first, then PDFs. Even indexes are lookups (one printed value on a labelled "
        "dual-panel chart; the Page-1 summary of a text-layer PDF). Odd indexes are "
        "perception-gated: weighted digraph shortest path with antiparallel arcs, unlabeled "
        "three-series line chart read against gridlines, hollow-marker count inside a dashed "
        "region, and a scanned (text-layer-free) audit packet with an errata page",
        "`numeric_tolerance`, `exact_json`",
    ),
    (
        "s5_spoken_audio",
        "Audio (`.wav`)",
        "Even indexes: single-speaker spoken dispatch JSON extraction. Odd indexes: ~1-minute "
        "three-speaker shift-handoff calls with self-corrections, supervisor recounts, "
        "conditional transfers, and auditor suggestions that only take effect when the "
        "supervisor gives or confirms them (16 kHz PCM WAV via `/usr/bin/say`)",
        "`exact_json`",
    ),
)


def write_dataset_md(
    pilot_items: list[dict[str, Any]],
    bench_items: list[dict[str, Any]],
    calib_items: list[dict[str, Any]],
) -> None:
    """Write `evals/DATASET.md` with slice and subtype overviews plus complete per-item provenance."""
    splits = [
        ("eval", bench_items),
        ("pilot", pilot_items),
        ("calibration", calib_items),
    ]
    n_total = sum(len(items) for _label, items in splits)
    num_slices = 5
    eval_per_slice = len(bench_items) // num_slices
    pilot_per_slice = len(pilot_items) // num_slices

    def _count(items: list[dict[str, Any]], **match: str) -> int:
        return sum(1 for it in items if all(it[k] == v for k, v in match.items()))

    lines = [
        "# Evaluation Dataset Provenance",
        "",
        "This document records the provenance, contamination properties, and deterministic verification",
        f"method for all **{len(bench_items)} evaluation items** (`data/eval.jsonl`, {eval_per_slice} per slice), "
        f"**{len(pilot_items)} disjoint pilot items** (`data/pilot.jsonl`, {pilot_per_slice} per slice), and "
        f"**{len(calib_items)} hard-only calibration items** (`data/calibration.jsonl`).",
        "All items are original and generated by `evals/build_dataset.py` (`uv run python -m evals.build_dataset`).",
        "",
        "## 1. Split Separation & Determinism",
        f"- **`data/pilot.jsonl` ($N = {len(pilot_items)}$, {pilot_per_slice} per slice, seed `{PILOT_SEED}`):** Used strictly for pre-flight verification (checking tier separation, `Usage` token accounting, and container sandbox execution). Pilot items are **never** included in the {len(bench_items)}-item evaluation or 5-fold cross-validation.",
        f"- **`data/eval.jsonl` ($N = {len(bench_items)}$, {eval_per_slice} per slice, seed `{BENCH_SEED}`):** Evaluated across all 5 default-ladder `(model, thinking_level)` tiers with $k = 3$ independent replicates per cell (${len(bench_items)} \\times 5 \\times 3 = {len(bench_items) * 5 * 3:,}$ downstream calls).",
        f"- **`data/calibration.jsonl` ($N = {len(calib_items)}$, seed `{CALIB_SEED}`):** Only the capability-gated subtypes, used once per tier to confirm that the hard items separate the tiers before the full benchmark is run. Never used for training or evaluation.",
        '- **Per-item seeding:** every item draws from its own `random.Random(f"{seed}:{item_id}")`, so changing one generator never changes another generator\'s items.',
        "",
        "## 2. Slice Summary",
        "",
        "| Slice ID | Count (Main / Pilot / Calib) | Modality | Task Subtypes | Deterministic Grader |",
        "| :--- | :---: | :---: | :--- | :--- |",
    ]
    for slice_id, modality, subtypes, graders in _SLICE_SUMMARY:
        counts = " / ".join(str(_count(items, slice=slice_id)) for _label, items in splits)
        lines.append(f"| `{slice_id}` | {counts} | {modality} | {subtypes} | {graders} |")

    lines += [
        "",
        "## 3. Subtype Counts",
        "",
        "| Slice | Task Subtype | Grader | Main | Pilot | Calib |",
        "| :--- | :--- | :--- | :---: | :---: | :---: |",
    ]
    seen: dict[tuple[str, str, str], None] = {}
    for _label, items in splits:
        for it in items:
            seen.setdefault((it["slice"], it["subtype"], it["grader_type"]), None)
    for slice_id, subtype, grader in sorted(seen):
        counts = " | ".join(
            str(_count(items, slice=slice_id, subtype=subtype)) for _label, items in splits
        )
        lines.append(f"| `{slice_id}` | `{subtype}` | `{grader}` | {counts} |")

    lines += [
        "",
        f"## 4. Complete Per-Item Provenance Table (All {n_total} Items)",
        "",
        "| Item ID | Split | Slice | Modality | Task Subtype | Grader | Provenance & License |",
        "| :--- | :---: | :--- | :---: | :--- | :---: | :--- |",
    ]
    for split_label, item_list in [
        ("pilot", pilot_items),
        ("eval", bench_items),
        ("calibration", calib_items),
    ]:
        for it in item_list:
            lines.append(
                f"| `{it['id']}` | `{split_label}` | `{it['slice']}` | `{it['modality']}` | "
                f"`{it['subtype']}` | `{it['grader_type']}` | {it['source_provenance']} |"
            )
    lines.append("")
    DATASET_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def assert_unique_item_fingerprints(*splits: list[dict[str, Any]]) -> None:
    """Verify zero duplicate item IDs or `(prompt, attachment_sha256s)` inputs across and within splits."""
    all_items = [it for sp in splits for it in sp]
    ids = [str(it["id"]) for it in all_items]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate item ids across splits")

    seen_inputs: dict[tuple[str, tuple[str, ...]], str] = {}
    for it in all_items:
        iid = str(it["id"])
        att_hashes = tuple(
            hashlib.sha256((REPO_ROOT / rel).read_bytes()).hexdigest()
            for rel in it.get("attachments", [])
        )
        fp = (str(it["prompt"]), att_hashes)
        prev = seen_inputs.get(fp)
        if prev is not None:
            raise RuntimeError(
                f"Duplicate item input fingerprint between {prev!r} and {iid!r} "
                f"(slice={it['slice']!r}, subtype={it['subtype']!r})"
            )
        seen_inputs[fp] = iid


def _positive_int(raw: str) -> int:
    val = int(raw)
    if val <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {raw!r}")
    return val


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate stratified pilot, evaluation, and calibration splits."
    )
    parser.add_argument(
        "--pilot-n",
        type=_positive_int,
        default=5,
        help="Items per slice for data/pilot.jsonl (default: 5).",
    )
    parser.add_argument(
        "--eval-n",
        type=_positive_int,
        default=30,
        help="Items per slice for data/eval.jsonl (default: 30).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    pilot = build_split(
        seed=PILOT_SEED,
        items_per_slice=args.pilot_n,
        prefix="pilot",
        out_path=DATA_DIR / "pilot.jsonl",
    )
    bench = build_split(
        seed=BENCH_SEED,
        items_per_slice=args.eval_n,
        prefix="bench",
        out_path=DATA_DIR / "eval.jsonl",
    )
    calib = build_calib_split(
        seed=CALIB_SEED, prefix="calib", out_path=DATA_DIR / "calibration.jsonl"
    )
    assert_unique_item_fingerprints(pilot, bench, calib)
    write_dataset_md(pilot, bench, calib)
    print(
        f"Generated {len(pilot)} pilot, {len(bench)} benchmark, and {len(calib)} calibration items, "
        "plus evals/DATASET.md."
    )


if __name__ == "__main__":
    main()
