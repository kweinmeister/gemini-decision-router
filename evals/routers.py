"""Stage 2: Collect router telemetry for gemini-decision-1 (score/choice/noul), flash-lite-json, and Q4 media ablations."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, cast

from google import genai

from evals._api import (
    BENCHMARK_SAFETY_SETTINGS,
    add_tiers_argument,
    append_jsonl,
    call_with_retry_and_latency,
    read_jsonl,
)
from evals.collect_matrix import build_item_payload, disable_sdk_auto_retries
from gemini_decision_router.cascade import build_verification_questions, format_verification_input
from gemini_decision_router.compute import calculate_compute_units
from gemini_decision_router.envelope import (
    compress_blocks_for_router,
    estimate_input_tokens,
    extract_interaction_text,
)
from gemini_decision_router.ladder import TierLadder, default_ladder, resolve_ladder
from gemini_decision_router.models import pava_isotonic_increasing
from gemini_decision_router.router import (
    answers_by_name,
    build_choice_questions,
    build_noul_solvability_questions,
    build_score_questions,
    create_genai_client,
    solvability_question_name,
)
from gemini_decision_router.usage import Usage

REPO_ROOT = Path(__file__).resolve().parent.parent


def flash_lite_router_schema(ladder: TierLadder) -> dict[str, Any]:
    """Structured Outputs schema for the prompted LLM-router baseline over `ladder`'s tiers."""
    ids = list(ladder.ordered_tier_ids)
    tier_prob_props = {
        tid: {"type": "number", "description": f"Probability in [0, 1] for {tid}."} for tid in ids
    }
    return {
        "type": "object",
        "properties": {
            "choice_tier": {
                "type": "string",
                "enum": list(ids),
            },
            "complexity_score": {
                "type": "number",
                "description": f"Continuous complexity score from 0.0 (trivial) to {len(ids) - 1:.1f} (frontier).",
            },
            "needs_precision": {
                "type": "boolean",
            },
            "tier_probabilities": {
                "type": "object",
                "description": f"Closed-world probability distribution across the {len(ids)} tiers summing to 1.0.",
                "properties": tier_prob_props,
                "required": list(ids),
            },
            "solvability_probabilities": {
                "type": "object",
                "description": "Independent probability P(solve) in [0.01, 0.99] that each tier solves the request.",
                "properties": tier_prob_props,
                "required": list(ids),
            },
        },
        "required": [
            "choice_tier",
            "complexity_score",
            "needs_precision",
            "tier_probabilities",
            "solvability_probabilities",
        ],
    }


def flash_lite_router_system_prompt(ladder: TierLadder) -> str:
    """System prompt for the prompted LLM-router baseline, listing `ladder`'s tiers."""
    return (
        "You are a fast, calibrated model router. Inspect the user request and estimate:\n"
        "1. `choice_tier`: minimum required capability tier.\n"
        "2. `tier_probabilities`: probability that each tier is the minimum required tier (summing to 1.0).\n"
        "3. `solvability_probabilities`: cumulative probability P(solve) in [0.01, 0.99] that each tier will "
        "solve the request on the first attempt when given the full user input (non-decreasing with rank, "
        "as higher tiers strictly subsume all lower tiers).\n"
        "Candidate tiers (cheapest -> most capable):\n"
        + "\n".join(
            f"- {t.tier_id} (rank {t.rank}): {t.capability_true_description}" for t in ladder.tiers
        )
        + "\nRespond strictly with the requested JSON object."
    )


async def _eval_decision_schema(
    client: genai.Client,
    payload: str | list[dict[str, Any]],
    questions: dict[str, dict[str, Any]],
    *,
    model: str,
    max_attempts: int = 4,
) -> tuple[Any, float, int]:
    """Call `client.aio.decisions.create` with SDK auto-retry disabled and record ONLY the winning attempt's latency."""
    return await call_with_retry_and_latency(
        lambda: client.aio.decisions.create(
            model=model,
            input=cast(Any, payload),
            questions=cast(Any, questions),
        ),
        max_attempts=max_attempts,
    )


async def _eval_flash_lite_json(
    client: genai.Client,
    payload: str | list[dict[str, Any]],
    *,
    ladder: TierLadder,
    max_attempts: int = 4,
    safety_settings: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Evaluate the prompted Structured Outputs router baseline on the ladder's cheapest tier.

    With the bundled ladder that is `gemini-3.5-flash-lite` with `minimal` thinking.
    """
    router_tier = ladder.tiers[0]
    ids = ladder.ordered_tier_ids

    gen_config: dict[str, Any] = {"thinking_level": router_tier.thinking_level}
    call_kwargs: dict[str, Any] = {
        "model": router_tier.model,
        "input": payload,
        "system_instruction": flash_lite_router_system_prompt(ladder),
        "response_mime_type": "application/json",
        "response_format": flash_lite_router_schema(ladder),
        "generation_config": gen_config,
    }
    if safety_settings:
        call_kwargs["safety_settings"] = safety_settings

    interaction, latency_ms, attempts = await call_with_retry_and_latency(
        lambda: client.aio.interactions.create(**call_kwargs),
        max_attempts=max_attempts,
    )

    usage = Usage.from_response(interaction)
    raw_text = extract_interaction_text(interaction)

    n_tiers = len(ids)
    default_tier_probs = dict.fromkeys(ids, 1.0 / n_tiers)
    # Monotone prior P(solve) ramp from 0.35 (cheapest) to 0.92 (most capable)
    default_solv_probs = {
        tid: 0.35 + (0.92 - 0.35) * (idx / max(1, n_tiers - 1)) for idx, tid in enumerate(ids)
    }
    try:
        parsed = json.loads(raw_text)
        # Normalize tier_probabilities to sum to 1.0
        raw_tp = parsed.get("tier_probabilities") or default_tier_probs
        tp_vals = {tid: max(1e-6, float(raw_tp.get(tid, 1.0 / n_tiers))) for tid in ids}
        tp_sum = sum(tp_vals.values())
        parsed["tier_probabilities"] = {tid: v / tp_sum for tid, v in tp_vals.items()}
        # Clamp solvability_probabilities to [0.01, 0.99] and compute PAVA projection
        raw_sp = parsed.get("solvability_probabilities") or default_solv_probs
        sp_vec = [min(0.99, max(0.01, float(raw_sp.get(tid, 0.5)))) for tid in ids]
        parsed["solvability_probabilities"] = {tid: sp_vec[idx] for idx, tid in enumerate(ids)}
        parsed["pava_projected"] = dict(zip(ids, pava_isotonic_increasing(sp_vec), strict=False))
        if "choice_tier" not in parsed and "target_tier" in parsed:
            parsed["choice_tier"] = parsed["target_tier"]
    except Exception:
        parsed = {
            "choice_tier": ladder.fallback_tier_id,
            "complexity_score": float(ladder.index(ladder.fallback_tier_id)),
            "needs_precision": True,
            "tier_probabilities": default_tier_probs,
            "solvability_probabilities": default_solv_probs,
            "pava_projected": default_solv_probs,
        }

    return {
        "router_family": "flash_lite_json",
        "model": router_tier.model,
        "thinking_level": router_tier.thinking_level,
        "latency_ms": round(latency_ms, 2),
        "attempts": attempts,
        "input_tokens": usage.input,
        "input_tokens_by_modality": dict(usage.by_modality),
        "output_tokens": usage.output,
        "thought_tokens": usage.thought,
        "compute_units": calculate_compute_units(router_tier.model, usage, ladder=ladder),
        "parsed": parsed,
    }


def _serialize_decision_resp(
    resp: Any, latency_ms: float, attempts: int = 1, *, ladder: TierLadder
) -> dict[str, Any]:
    usage = Usage.from_response(resp)
    ans_dict: dict[str, Any] = {}
    for name, a in answers_by_name(getattr(resp, "answers", None)).items():
        entry: dict[str, Any] = {"type": getattr(a, "type", "")}
        if getattr(a, "noul", None) is not None:
            entry["noul"] = float(a.noul)
        if getattr(a, "choice", None) is not None:
            entry["choice"] = str(a.choice)
        if getattr(a, "score", None) is not None:
            entry["score"] = float(a.score)
        if getattr(a, "probabilities", None) is not None:
            entry["probabilities"] = {str(k): float(v) for k, v in a.probabilities.items()}
        if getattr(a, "confidence", None) is not None:
            entry["confidence"] = float(a.confidence)
        ans_dict[str(name)] = entry

    return {
        "latency_ms": round(latency_ms, 2),
        "attempts": attempts,
        "input_tokens": usage.input,
        "input_tokens_by_modality": dict(usage.by_modality),
        "output_tokens": usage.output,
        "compute_units": calculate_compute_units(ladder.decision_model, usage, ladder=ladder),
        "answers": ans_dict,
    }


async def _decide_serialized(
    client: genai.Client,
    payload: str | list[dict[str, Any]],
    questions: dict[str, dict[str, Any]],
    *,
    ladder: TierLadder,
) -> dict[str, Any]:
    resp, latency_ms, attempts = await _eval_decision_schema(
        client, payload, questions, model=ladder.decision_model
    )
    return _serialize_decision_resp(resp, latency_ms, attempts, ladder=ladder)


async def _evaluate_router_item(
    client: genai.Client,
    item: dict[str, Any],
    *,
    ladder: TierLadder,
    q_score: dict[str, dict[str, Any]],
    q_choice: dict[str, dict[str, Any]],
    q_noul: dict[str, dict[str, Any]],
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    """Run all router arms (plus Q4 ablation for visual items) for a single dataset item."""
    full_payload = build_item_payload(item, REPO_ROOT)
    est_in_tokens = estimate_input_tokens(full_payload)
    compressed_payload = compress_blocks_for_router(
        full_payload,
        image_resolution="low",
        pdf_mode="page1_pdf",
    )

    # Execute each HTTP call sequentially inside the item semaphore so total in-flight
    # requests never exceed `concurrency` (prevents burst fan-out from inflating latency).
    async with sem:
        decision_score = await _decide_serialized(
            client, compressed_payload, q_score, ladder=ladder
        )
        decision_choice = await _decide_serialized(
            client, compressed_payload, q_choice, ladder=ladder
        )
        decision_noul = await _decide_serialized(client, compressed_payload, q_noul, ladder=ladder)
        flash_lite_rec = await _eval_flash_lite_json(
            client,
            compressed_payload,
            ladder=ladder,
            safety_settings=BENCHMARK_SAFETY_SETTINGS,
        )

        # Q4 Multimodal Compression Ablation on s4_visual_and_pdf items
        q4_ablation: dict[str, Any] = {}
        if item["slice"] == "s4_visual_and_pdf":
            q4_ablation["full_media_score"] = await _decide_serialized(
                client, full_payload, q_score, ladder=ladder
            )
            if item["modality"] == "pdf":
                pdf_img_high = compress_blocks_for_router(
                    full_payload, image_resolution="high", pdf_mode="page1_image"
                )
                pdf_img_low = compress_blocks_for_router(
                    full_payload, image_resolution="low", pdf_mode="page1_image"
                )
                q4_ablation["pdf_page1_img_high"] = await _decide_serialized(
                    client, pdf_img_high, q_score, ladder=ladder
                )
                q4_ablation["pdf_page1_img_low"] = await _decide_serialized(
                    client, pdf_img_low, q_score, ladder=ladder
                )

    # Prefer exact API-tokenized full-payload input token count L_i over local heuristic
    api_full_in_tokens = (
        int(q4_ablation["full_media_score"].get("input_tokens", 0))
        if "full_media_score" in q4_ablation
        else int(decision_score.get("input_tokens", 0))
    )
    known_in_tokens = api_full_in_tokens if api_full_in_tokens > 0 else est_in_tokens

    raw_noul_vec = [
        float(decision_noul["answers"].get(solvability_question_name(tid), {}).get("noul", 0.5))
        for tid in ladder.ordered_tier_ids
    ]
    decision_noul["pava_projected"] = dict(
        zip(ladder.ordered_tier_ids, pava_isotonic_increasing(raw_noul_vec), strict=False)
    )

    return {
        "item_id": item["id"],
        "slice": item["slice"],
        "modality": item["modality"],
        "known_input_tokens": known_in_tokens,
        "estimated_input_tokens": est_in_tokens,
        "decision_score": decision_score,
        "decision_choice": decision_choice,
        "decision_noul": decision_noul,
        "flash_lite_json": flash_lite_rec,
        "q4_ablation": q4_ablation,
    }


async def run_routers(
    dataset_path: Path,
    output_path: Path,
    *,
    concurrency: int = 2,
    ladder: TierLadder | None = None,
) -> None:
    lad = ladder or default_ladder()
    client = create_genai_client()
    disable_sdk_auto_retries(client)
    items = read_jsonl(dataset_path)
    completed_ids: set[str] = {rec["item_id"] for rec in read_jsonl(output_path)}

    # Warm-up calls for both endpoints before recording latency
    warm_tier = lad.tiers[0]
    try:
        await client.aio.decisions.create(
            model=lad.decision_model,
            input="Warmup check.",
            questions={"ok": {"type": "noul", "instructions": "Is this a warmup?"}},
        )
        await client.aio.interactions.create(
            model=warm_tier.model,
            input="Warmup check.",
            generation_config={"thinking_level": warm_tier.thinking_level},
            safety_settings=BENCHMARK_SAFETY_SETTINGS,
        )
    except Exception:
        pass

    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    q_score = build_score_questions(lad)
    q_choice = build_choice_questions(lad)
    q_noul = build_noul_solvability_questions(lad)

    total_items = len(items)
    completed_count = len(completed_ids)
    if completed_count > 0:
        print(f"Resuming router collection: {completed_count}/{total_items} already completed.")

    async def _process_item(item: dict[str, Any]) -> None:
        nonlocal completed_count
        if item["id"] in completed_ids:
            return
        record = await _evaluate_router_item(
            client,
            item,
            ladder=lad,
            q_score=q_score,
            q_choice=q_choice,
            q_noul=q_noul,
            sem=sem,
        )
        async with write_lock:
            append_jsonl(output_path, record)
            completed_count += 1
            print(
                f"[routers {completed_count}/{total_items}] {item['id']} slice={item['slice']} modality={item['modality']}",
                flush=True,
            )

    await asyncio.gather(*(_process_item(it) for it in items))


async def run_cascade_verifiers(
    dataset_path: Path,
    matrix_path: Path,
    output_path: Path,
    *,
    cheap_tier_id: str | None = None,
    concurrency: int = 2,
    ladder: TierLadder | None = None,
) -> None:
    """Collect post-generation `gemini-decision-1` verification scores on cached cheap-tier drafts."""
    lad = ladder or default_ladder()
    target_tier = cheap_tier_id or lad.ordered_tier_ids[0]
    lad.tier(target_tier)

    items = read_jsonl(dataset_path)
    cheap_rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(matrix_path):
        if row.get("tier_id") != target_tier:
            continue
        iid = str(row["item_id"])
        prev = cheap_rows.get(iid)
        if prev is None or int(row.get("replicate", 1)) < int(prev.get("replicate", 1)):
            cheap_rows[iid] = row

    done_ids: set[str] = set()
    if output_path.exists():
        for rec in read_jsonl(output_path):
            done_ids.add(str(rec["item_id"]))

    remaining = [
        it for it in items if str(it["id"]) not in done_ids and str(it["id"]) in cheap_rows
    ]
    if not cheap_rows:
        raise ValueError(
            f"No downstream drafts for cheap tier {target_tier!r} found in {matrix_path}"
        )
    if not remaining:
        print(
            f"All {len(items)} cascade verifiers for {target_tier} already collected at {output_path}"
        )
        return

    client = create_genai_client()
    disable_sdk_auto_retries(client)
    sem = asyncio.Semaphore(max(1, concurrency))
    lock = asyncio.Lock()
    verify_qs = build_verification_questions()

    async def _process_one(item: dict[str, Any]) -> None:
        async with sem:
            iid = str(item["id"])
            cheap_row = cheap_rows[iid]
            payload = build_item_payload(item, REPO_ROOT)
            verify_payload = format_verification_input(
                payload, str(cheap_row.get("response_text", ""))
            )
            resp, lat, att = await _eval_decision_schema(
                client, verify_payload, verify_qs, model=lad.decision_model
            )
            record = {
                "item_id": iid,
                "slice": item["slice"],
                "modality": item["modality"],
                "cheap_tier_id": target_tier,
                "replicate": int(cheap_row.get("replicate", 1)),
                "verifier": _serialize_decision_resp(resp, lat, att, ladder=lad),
            }
            async with lock:
                append_jsonl(output_path, record)

    await asyncio.gather(*(_process_one(it) for it in remaining))


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect router telemetry across all arms.")
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "data/eval.jsonl")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results/router_telemetry.jsonl")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument(
        "--cascade-matrix",
        type=Path,
        default=None,
        help="Optional downstream matrix JSONL; when set, runs post-generation cascade verifiers on cached cheap-tier drafts.",
    )
    parser.add_argument(
        "--cascade-output",
        type=Path,
        default=REPO_ROOT / "results/cascade_verifiers.jsonl",
    )
    parser.add_argument("--cheap-tier", type=str, default=None)
    add_tiers_argument(parser)
    args = parser.parse_args()

    ladder = resolve_ladder(args.tiers)
    if args.cascade_matrix is not None:
        asyncio.run(
            run_cascade_verifiers(
                args.dataset,
                args.cascade_matrix,
                args.cascade_output,
                cheap_tier_id=args.cheap_tier,
                concurrency=args.concurrency,
                ladder=ladder,
            )
        )
        return

    asyncio.run(
        run_routers(
            args.dataset,
            args.output,
            concurrency=args.concurrency,
            ladder=ladder,
        )
    )


if __name__ == "__main__":
    main()
