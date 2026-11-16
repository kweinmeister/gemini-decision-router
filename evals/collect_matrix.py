"""Stage 1: Collect downstream ground-truth execution matrix (150 items x N ladder tiers x k=3 replicates)."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
from pathlib import Path
from typing import Any, Literal

from google import genai

from evals._api import (
    BENCHMARK_SAFETY_SETTINGS,
    add_tiers_argument,
    append_jsonl,
    call_with_retry_and_latency,
    read_jsonl,
)
from evals.graders import grade_item
from evals.sandbox import SandboxBackend
from gemini_decision_router.compute import calculate_compute_units
from gemini_decision_router.envelope import extract_interaction_text, load_media_block
from gemini_decision_router.ladder import TierLadder, TierSpec, default_ladder, resolve_ladder
from gemini_decision_router.models import validate_thinking_level
from gemini_decision_router.router import create_genai_client
from gemini_decision_router.usage import Usage

REPO_ROOT = Path(__file__).resolve().parent.parent


def disable_sdk_auto_retries(client: genai.Client) -> None:
    """Disable Speakeasy internal auto-retry sleep so single-attempt HTTP timers are unpolluted."""
    for sub in (
        getattr(client, "interactions", None),
        getattr(client, "decisions", None),
        getattr(getattr(client, "aio", None), "interactions", None),
        getattr(getattr(client, "aio", None), "decisions", None),
    ):
        sdk_cfg = getattr(sub, "sdk_configuration", None)
        if sdk_cfg is not None and hasattr(sdk_cfg, "retry_config"):
            sdk_cfg.retry_config = None


def build_item_payload(item: dict[str, Any], allowed_root: Path) -> str | list[dict[str, Any]]:
    """Construct the full-resolution Interactions/Decisions API input payload for a dataset item."""
    attachments = item.get("attachments") or []
    if not attachments:
        return str(item["prompt"])
    blocks: list[dict[str, Any]] = [
        load_media_block(rel_path, allowed_root) for rel_path in attachments
    ]
    blocks.append({"type": "text", "text": str(item["prompt"])})
    return blocks


def build_matrix_record(
    item: dict[str, Any],
    tier: TierSpec,
    replicate: int,
    *,
    response_text: str,
    usage: Usage,
    latency_ms: float,
    attempts: int,
    passed: bool,
    grade_reason: str,
    interaction_status: str,
    ladder: TierLadder,
) -> dict[str, Any]:
    """One matrix row (pure, no I/O), with relative `compute_units`.

    Thinking tokens are weighted at the output rate (`Usage.billed_output`).
    """
    return {
        "item_id": item["id"],
        "slice": item["slice"],
        "modality": item["modality"],
        "tier_id": tier.tier_id,
        "model": tier.model,
        "thinking_level": tier.thinking_level,
        "replicate": replicate,
        "passed": passed,
        "grade_reason": grade_reason,
        "latency_ms": round(latency_ms, 2),
        "attempts": attempts,
        "input_tokens": usage.input,
        "input_tokens_by_modality": dict(usage.by_modality),
        "output_tokens": usage.output,
        "thought_tokens": usage.thought,
        "compute_units": calculate_compute_units(tier.model, usage, ladder=ladder),
        # Surfaces truncated / budget-exhausted responses (e.g. 0 output tokens after max thinking).
        "interaction_status": interaction_status,
        "empty_response": not response_text.strip(),
        "response_text": response_text,
    }


async def _call_single_attempt_with_backoff(
    client: genai.Client,
    model: str,
    thinking_level: str,
    payload: str | list[dict[str, Any]],
    *,
    max_attempts: int = 4,
    max_output_tokens: int | None = None,
    ladder: TierLadder | None = None,
    safety_settings: list[dict[str, str]] | None = None,
) -> tuple[Any, float, int]:
    """Execute `client.aio.interactions.create` and record ONLY the winning HTTP attempt's latency_ms."""
    validate_thinking_level(model, thinking_level, ladder=ladder)
    gen_config: dict[str, Any] = {"thinking_level": thinking_level}
    if max_output_tokens is not None:
        gen_config["max_output_tokens"] = max_output_tokens
    applied_safety = safety_settings if safety_settings is not None else BENCHMARK_SAFETY_SETTINGS
    call_kwargs: dict[str, Any] = {
        "model": model,
        "input": payload,
        "generation_config": gen_config,
    }
    if applied_safety:
        call_kwargs["safety_settings"] = applied_safety
    return await call_with_retry_and_latency(
        lambda: client.aio.interactions.create(**call_kwargs),
        max_attempts=max_attempts,
    )


async def run_collection(
    dataset_path: Path,
    output_path: Path,
    *,
    replicates: int = 3,
    concurrency: int = 2,
    sandbox_backend: SandboxBackend | Literal["auto"] = "auto",
    ladder: TierLadder | None = None,
    only_tier: str | None = None,
    tier_id_override: str | None = None,
    max_output_tokens: int | None = None,
) -> None:
    lad = ladder or default_ladder()
    tiers_to_run = [t for t in lad.tiers if only_tier is None or t.tier_id == only_tier]
    if not tiers_to_run:
        raise ValueError(f"Unknown --only-tier {only_tier!r}")
    if tier_id_override is not None:
        if len(tiers_to_run) != 1:
            raise ValueError("--tier-id-override requires --only-tier matching exactly 1 tier")
        tiers_to_run = [dataclasses.replace(tiers_to_run[0], tier_id=tier_id_override)]

    client = create_genai_client()
    disable_sdk_auto_retries(client)

    items = read_jsonl(dataset_path)
    target_tier_ids = {t.tier_id for t in tiers_to_run}
    completed_keys: set[tuple[str, str, int]] = {
        (rec["item_id"], rec["tier_id"], int(rec["replicate"]))
        for rec in read_jsonl(output_path)
        if rec["tier_id"] in target_tier_ids
    }

    # Warm-up call to establish TLS/HTTP2 session pool before recording latency
    warm_tier = tiers_to_run[0]
    with contextlib.suppress(Exception):
        await client.aio.interactions.create(
            model=warm_tier.model,
            input="Reply with OK.",
            generation_config={"thinking_level": warm_tier.thinking_level},
            safety_settings=BENCHMARK_SAFETY_SETTINGS,
        )

    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_calls = len(items) * len(tiers_to_run) * replicates
    completed_count = len(completed_keys)
    if completed_count > 0:
        print(f"Resuming matrix collection: {completed_count}/{total_calls} already completed.")

    async def _tracked_worker(item: dict[str, Any], tier: TierSpec, rep: int) -> None:
        nonlocal completed_count
        key = (item["id"], tier.tier_id, rep)
        if key in completed_keys:
            return
        payload = build_item_payload(item, REPO_ROOT)
        async with sem:
            interaction, latency_ms, attempts = await _call_single_attempt_with_backoff(
                client,
                tier.model,
                tier.thinking_level,
                payload,
                max_output_tokens=max_output_tokens,
                ladder=lad,
            )
        response_text = extract_interaction_text(interaction)
        passed, grade_reason = await asyncio.to_thread(
            grade_item,
            item,
            response_text,
            sandbox_backend=sandbox_backend,
        )
        record = build_matrix_record(
            item,
            tier,
            rep,
            response_text=response_text,
            usage=Usage.from_response(interaction),
            latency_ms=latency_ms,
            attempts=attempts,
            passed=passed,
            grade_reason=grade_reason,
            interaction_status=str(getattr(interaction, "status", "") or ""),
            ladder=lad,
        )
        async with write_lock:
            append_jsonl(output_path, record)
            completed_count += 1
            status_str = "PASS" if passed else f"FAIL ({grade_reason})"
            print(
                f"[matrix {completed_count}/{total_calls}] {item['id']} {tier.tier_id} rep={rep} -> {status_str} ({latency_ms:.0f}ms)",
                flush=True,
            )

    await asyncio.gather(
        *(
            _tracked_worker(item, tier, rep)
            for item in items
            for tier in tiers_to_run
            for rep in range(replicates)
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect downstream ground-truth execution matrix."
    )
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "data/eval.jsonl")
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "results/downstream_matrix.jsonl"
    )
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument(
        "--sandbox-backend", choices=["auto", "container", "docker"], default="auto"
    )
    parser.add_argument("--only-tier", type=str, default=None)
    parser.add_argument("--tier-id-override", type=str, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    add_tiers_argument(parser)
    args = parser.parse_args()

    asyncio.run(
        run_collection(
            args.dataset,
            args.output,
            replicates=args.replicates,
            concurrency=args.concurrency,
            sandbox_backend=args.sandbox_backend,
            ladder=resolve_ladder(args.tiers),
            only_tier=args.only_tier,
            tier_id_override=args.tier_id_override,
            max_output_tokens=args.max_output_tokens,
        )
    )


if __name__ == "__main__":
    main()
