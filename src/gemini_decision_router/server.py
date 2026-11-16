"""Thin FastAPI HTTP wrapper around `DecisionRouter` for Cloud Run deployment."""

import dataclasses
import json
import os
import re
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from gemini_decision_router.envelope import ImageResolution, extract_interaction_text
from gemini_decision_router.ladder import TierLadder, default_ladder
from gemini_decision_router.models import RouteDecision
from gemini_decision_router.observability import RouterObserver
from gemini_decision_router.router import DecisionRouter, PolicyMode, RouterPolicyConfig

_TRACE_ID_RE = re.compile(r"^[0-9a-fA-F]{32}$")


@dataclasses.dataclass(frozen=True)
class ServerLimits:
    """Configurable request-size limits aligned with Gemini inline & Cloud Run bounds."""

    max_text_chars: int = 4_000_000  # ~1M tokens
    max_media_b64_chars: int = 28_000_000  # ~20 MiB raw inline media (<32 MiB Cloud Run HTTP/1.1)
    max_blocks: int = 64
    max_system_instruction_chars: int = 128_000

    def __post_init__(self) -> None:
        for field_name in (
            "max_text_chars",
            "max_media_b64_chars",
            "max_blocks",
            "max_system_instruction_chars",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be > 0")

    @classmethod
    def from_env(cls) -> ServerLimits:
        """Load limits from `GEMINI_ROUTER_MAX_*` environment variables."""
        defaults = cls()
        return cls(
            max_text_chars=int(
                os.environ.get("GEMINI_ROUTER_MAX_TEXT_CHARS", defaults.max_text_chars)
            ),
            max_media_b64_chars=int(
                os.environ.get("GEMINI_ROUTER_MAX_MEDIA_B64_CHARS", defaults.max_media_b64_chars)
            ),
            max_blocks=int(os.environ.get("GEMINI_ROUTER_MAX_BLOCKS", defaults.max_blocks)),
            max_system_instruction_chars=int(
                os.environ.get(
                    "GEMINI_ROUTER_MAX_SYSTEM_INSTRUCTION_CHARS",
                    defaults.max_system_instruction_chars,
                )
            ),
        )


def _validate_request_limits(req: RouteRequest, limits: ServerLimits) -> None:
    """Enforce `ServerLimits` on text, multimodal blocks, and system instructions."""
    if isinstance(req.input, str):
        if len(req.input) > limits.max_text_chars:
            raise HTTPException(
                status_code=422,
                detail=f"Input text exceeds max length of {limits.max_text_chars} chars.",
            )
    else:
        if len(req.input) > limits.max_blocks:
            raise HTTPException(
                status_code=422,
                detail=f"Input exceeds max of {limits.max_blocks} content blocks.",
            )
        for block in req.input:
            if isinstance(block, TextBlock):
                if len(block.text) > limits.max_text_chars:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Text block exceeds max length of {limits.max_text_chars} chars.",
                    )
            elif len(block.data) > limits.max_media_b64_chars:
                raise HTTPException(
                    status_code=422,
                    detail=f"Media block exceeds max length of {limits.max_media_b64_chars} base64 chars.",
                )
    if (
        isinstance(req, RouteAndExecuteRequest)
        and req.system_instruction is not None
        and len(req.system_instruction) > limits.max_system_instruction_chars
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                f"system_instruction exceeds max length of "
                f"{limits.max_system_instruction_chars} chars."
            ),
        )


_NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _StrictBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextBlock(_StrictBlock):
    """Text content block."""

    type: Literal["text"]
    text: _NonEmptyStr


class ImageBlock(_StrictBlock):
    """Inline base64 image content block."""

    type: Literal["image"]
    data: _NonEmptyStr
    mime_type: _NonEmptyStr
    resolution: ImageResolution | None = None


class VideoBlock(_StrictBlock):
    """Inline base64 video content block."""

    type: Literal["video"]
    data: _NonEmptyStr
    mime_type: _NonEmptyStr
    resolution: ImageResolution | None = None


class AudioBlock(_StrictBlock):
    """Inline base64 audio content block."""

    type: Literal["audio"]
    data: _NonEmptyStr
    mime_type: _NonEmptyStr


class DocumentBlock(_StrictBlock):
    """Inline base64 PDF document content block."""

    type: Literal["document"]
    data: _NonEmptyStr
    mime_type: Literal["application/pdf"] = "application/pdf"


ContentBlock = Annotated[
    TextBlock | ImageBlock | VideoBlock | AudioBlock | DocumentBlock,
    Field(discriminator="type"),
]


class RouteRequest(BaseModel):
    """Request body for `POST /v1/route`."""

    model_config = ConfigDict(extra="forbid")

    input: _NonEmptyStr | Annotated[list[ContentBlock], Field(min_length=1)]
    policy: PolicyMode = "score_tail"
    compress_media_for_router: bool = True
    conformal_cdf_threshold: Annotated[float, Field(gt=0.0, le=1.0, allow_inf_nan=False)] | None = (
        None
    )


class RouteAndExecuteRequest(RouteRequest):
    """Request body for `POST /v1/route-and-execute`."""

    system_instruction: str | None = None
    max_output_tokens: Annotated[int, Field(gt=0)] | None = None


def _to_router_payload(raw_input: str | list[ContentBlock]) -> str | list[dict[str, Any]]:
    if isinstance(raw_input, str):
        return raw_input
    return [block.model_dump(exclude_none=True) for block in raw_input]


def _emit_route_log(
    request: Request,
    decision: RouteDecision,
    raw_input: str | list[ContentBlock],
    *,
    event: str,
) -> None:
    """Emit single-line structured JSON telemetry to stdout for Cloud Logging (no prompt/media PII)."""
    entry: dict[str, Any] = {
        "severity": "WARNING" if decision.fallback_triggered else "INFO",
        "message": event,
        "selected_tier_id": decision.selected_tier_id,
        "model": decision.model,
        "thinking_level": decision.thinking_level,
        "policy_name": decision.policy_name,
        "confidence": decision.confidence,
        "router_latency_ms": round(decision.router_latency_ms, 2),
        "router_input_tokens": decision.router_input_tokens,
        "router_compute_units": decision.router_compute_units,
        "estimated_downstream_compute_units": decision.estimated_downstream_compute_units,
        "fallback_triggered": decision.fallback_triggered,
        "fallback_reason": decision.fallback_reason,
        "input_type": "text" if isinstance(raw_input, str) else "multimodal",
    }
    if not isinstance(raw_input, str):
        entry["block_types"] = [b.type for b in raw_input]

    trace_header = request.headers.get("X-Cloud-Trace-Context", "")
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if trace_header and project_id:
        trace_id = trace_header.split("/", 1)[0]
        if _TRACE_ID_RE.fullmatch(trace_id):
            entry["logging.googleapis.com/trace"] = f"projects/{project_id}/traces/{trace_id}"

    print(json.dumps(entry, separators=(",", ":")), file=sys.stdout, flush=True)


def _router_for_request(
    base_router: DecisionRouter,
    req: RouteRequest,
    observer: RouterObserver,
) -> DecisionRouter:
    """Reuse `base_router`'s client, ladder, and observer while applying per-request policy."""
    ladder = base_router.ladder
    target_crc = (
        req.conformal_cdf_threshold
        if req.conformal_cdf_threshold is not None
        else (
            ladder.conformal_cdf_threshold
            if ladder.conformal_cdf_threshold is not None
            else base_router.config.conformal_cdf_threshold
        )
    )
    if (
        req.policy == base_router.config.policy
        and req.compress_media_for_router == base_router.config.compress_media_for_router
        and target_crc == base_router.config.conformal_cdf_threshold
    ):
        return base_router

    cfg = dataclasses.replace(
        base_router.config,
        policy=req.policy,
        compress_media_for_router=req.compress_media_for_router,
        conformal_cdf_threshold=target_crc,
        ladder=None,
    )
    return DecisionRouter(
        config=cfg,
        client=base_router.client,
        downstream_client=base_router.downstream_client,
        observer=observer,
        ladder=ladder,
    )


def create_app(
    *,
    router: DecisionRouter | None = None,
    observer: RouterObserver | None = None,
    ladder: TierLadder | None = None,
    max_observations: int = 1000,
    limits: ServerLimits | None = None,
) -> FastAPI:
    """Construct the FastAPI application with optional dependency injection for testing."""
    resolved_limits = limits or ServerLimits.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        resolved_ladder = ladder or (router.ladder if router is not None else default_ladder())
        obs = observer or (
            router.observer
            if router is not None and router.observer is not None
            else RouterObserver(ladder=resolved_ladder, max_observations=max_observations)
        )
        base_router = router or DecisionRouter(
            config=RouterPolicyConfig(ladder=resolved_ladder),
            observer=obs,
        )
        app.state.ladder = resolved_ladder
        app.state.observer = obs
        app.state.router = base_router
        yield

    app = FastAPI(
        title="gemini-decision-router",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.exception_handler(ValueError)
    async def _value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/tiers")
    async def get_tiers() -> dict[str, Any]:
        return app.state.ladder.to_summary()

    @app.get("/v1/metrics")
    async def get_metrics() -> dict[str, Any]:
        return dataclasses.asdict(app.state.observer.summary())

    @app.post("/v1/route")
    async def route_prompt(req: RouteRequest, request: Request) -> dict[str, Any]:
        _validate_request_limits(req, resolved_limits)
        req_router = _router_for_request(app.state.router, req, app.state.observer)
        decision = await req_router.aroute(_to_router_payload(req.input))
        _emit_route_log(request, decision, req.input, event="route_decision")
        return dataclasses.asdict(decision)

    @app.post("/v1/route-and-execute")
    async def route_and_execute(req: RouteAndExecuteRequest, request: Request) -> dict[str, Any]:
        _validate_request_limits(req, resolved_limits)
        req_router = _router_for_request(app.state.router, req, app.state.observer)
        gen_config = (
            {"max_output_tokens": req.max_output_tokens}
            if req.max_output_tokens is not None
            else None
        )
        decision, interaction = await req_router.aroute_and_execute(
            _to_router_payload(req.input),
            system_instruction=req.system_instruction,
            generation_config=gen_config,
            on_decision=lambda d: _emit_route_log(request, d, req.input, event="route_and_execute"),
        )
        return {
            "decision": dataclasses.asdict(decision),
            "downstream_output": extract_interaction_text(interaction),
        }

    return app


app = create_app()
