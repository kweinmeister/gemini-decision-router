"""Two-stage model cascades and Hybrid Cascade-Routing using Decisions API verification."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from gemini_decision_router.compute import calculate_compute_units
from gemini_decision_router.envelope import extract_interaction_text, prepare_router_envelope
from gemini_decision_router.models import RouteDecision, TierID
from gemini_decision_router.router import DecisionRouter
from gemini_decision_router.usage import Usage

__all__ = [
    "CascadeConfig",
    "CascadeResult",
    "HybridCascadeRouter",
    "build_verification_questions",
    "format_verification_input",
]


def build_verification_questions() -> dict[str, dict[str, Any]]:
    """Build the post-generation ``noul`` audit question for ``gemini-decision-1``."""
    return {
        "draft_solves_request": {
            "type": "noul",
            "instructions": (
                "Does the candidate draft response accurately, completely, and "
                "safely satisfy the user's request without logical, factual, or "
                "syntax errors?"
            ),
            "criteria": {
                "true_description": (
                    "The draft directly answers the prompt, follows all constraints, "
                    "and has no visible bugs, missing steps, or contradictions."
                ),
                "false_description": (
                    "The draft is incomplete, evasive, fails a constraint, or contains "
                    "a likely reasoning, math, or code error."
                ),
            },
        }
    }


@dataclass(frozen=True)
class CascadeConfig:
    """Configuration for pure 2-stage cascades and Hybrid Cascade-Routing."""

    cheap_tier_id: TierID = "t0_flash_lite_min"
    escalation_tier_id: TierID = "t1_hydrogen_low"
    accept_threshold: float = 0.70
    use_prefill_skip: bool = True
    prefill_accept_threshold: float = 0.75


@dataclass
class CascadeResult:
    """Outcome and accounting for a cascade or hybrid cascade-route execution."""

    final_tier_id: TierID
    final_model: str
    final_thinking_level: str
    interaction: Any
    prefill_decision: RouteDecision | None = None
    prefill_skipped_cheap: bool = False
    escalated: bool = False
    verifier_confidence: float | None = None
    total_compute_units: float = 0.0
    total_latency_ms: float = 0.0


def format_verification_input(
    input_payload: str | list[dict[str, Any]],
    draft_text: str,
) -> list[dict[str, Any]]:
    """Append the candidate draft to the compressed prompt payload for verification."""
    compressed, _, _ = prepare_router_envelope(input_payload, compress_media=True)
    if isinstance(compressed, str):
        blocks: list[dict[str, Any]] = [{"type": "text", "text": compressed}]
    else:
        blocks = [dict(b) for b in compressed]
    blocks.append(
        {
            "type": "text",
            "text": f"\n\n[Candidate draft response]\n{draft_text}",
        }
    )
    return blocks


class HybridCascadeRouter:
    """Executes either a pure 2-stage cascade or a Hybrid Cascade-Router.

    In hybrid mode (``use_prefill_skip=True``), a prefill ``DecisionRouter``
    call first routes clearly hard prompts straight to their target tier without
    wasting ``cheap_tier_id`` tokens, accepts high-confidence ``cheap_tier_id``
    prompts directly, and verifies borderline ``cheap_tier_id`` drafts with
    ``gemini-decision-1`` before escalating.
    """

    def __init__(
        self,
        router: DecisionRouter,
        config: CascadeConfig | None = None,
    ) -> None:
        self.router = router
        self.config = config or CascadeConfig()
        # Validate referenced tier IDs eagerly against the active ladder.
        self.router.ladder.tier(self.config.cheap_tier_id)
        self.router.ladder.tier(self.config.escalation_tier_id)

    def _invoke_tier(
        self,
        tier_id: TierID,
        input_payload: str | list[dict[str, Any]],
        *,
        system_instruction: str | None = None,
        safety_settings: list[dict[str, str]] | None = None,
        generation_config: dict[str, Any] | None = None,
    ) -> tuple[Any, float, float]:
        spec = self.router.ladder.tier(tier_id)
        kwargs = self.router.build_execution_kwargs(
            tier_id,
            input_payload,
            system_instruction=system_instruction,
            safety_settings=safety_settings,
            generation_config=generation_config,
        )
        t0 = time.perf_counter()
        interaction = self.router.downstream_client.interactions.create(**kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        usage = Usage.from_response(interaction)
        units = calculate_compute_units(spec.model, usage, ladder=self.router.ladder)
        return interaction, units, latency_ms

    def _verify_draft(
        self,
        input_payload: str | list[dict[str, Any]],
        draft_text: str,
    ) -> tuple[float, float, float]:
        ver_input = format_verification_input(input_payload, draft_text)
        return self.router.evaluate_noul_probe(
            ver_input,
            "draft_solves_request",
            build_verification_questions(),
        )

    def run(
        self,
        input_payload: str | list[dict[str, Any]],
        *,
        system_instruction: str | None = None,
        safety_settings: list[dict[str, str]] | None = None,
        generation_config: dict[str, Any] | None = None,
    ) -> CascadeResult:
        """Execute the cascade or hybrid cascade-router end-to-end."""
        total_units = 0.0
        total_ms = 0.0
        prefill_decision: RouteDecision | None = None

        if self.config.use_prefill_skip:
            prefill_decision = self.router.route(input_payload)
            total_units += prefill_decision.router_compute_units
            total_ms += prefill_decision.router_latency_ms

            if prefill_decision.selected_tier_id != self.config.cheap_tier_id:
                interaction, units, lat = self._invoke_tier(
                    prefill_decision.selected_tier_id,
                    input_payload,
                    system_instruction=system_instruction,
                    safety_settings=safety_settings,
                    generation_config=generation_config,
                )
                return CascadeResult(
                    final_tier_id=prefill_decision.selected_tier_id,
                    final_model=prefill_decision.model,
                    final_thinking_level=prefill_decision.thinking_level,
                    interaction=interaction,
                    prefill_decision=prefill_decision,
                    prefill_skipped_cheap=True,
                    total_compute_units=total_units + units,
                    total_latency_ms=total_ms + lat,
                )

        # Run cheap tier draft
        draft_interaction, draft_units, draft_ms = self._invoke_tier(
            self.config.cheap_tier_id,
            input_payload,
            system_instruction=system_instruction,
            safety_settings=safety_settings,
            generation_config=generation_config,
        )
        total_units += draft_units
        total_ms += draft_ms
        cheap_spec = self.router.ladder.tier(self.config.cheap_tier_id)

        # High-confidence cheap prefill skips post-generation verification
        if (
            prefill_decision is not None
            and prefill_decision.confidence >= self.config.prefill_accept_threshold
        ):
            return CascadeResult(
                final_tier_id=cheap_spec.tier_id,
                final_model=cheap_spec.model,
                final_thinking_level=cheap_spec.thinking_level,
                interaction=draft_interaction,
                prefill_decision=prefill_decision,
                total_compute_units=total_units,
                total_latency_ms=total_ms,
            )

        draft_text = extract_interaction_text(draft_interaction)
        ver_prob, ver_units, ver_ms = self._verify_draft(input_payload, draft_text)
        total_units += ver_units
        total_ms += ver_ms

        if ver_prob >= self.config.accept_threshold:
            return CascadeResult(
                final_tier_id=cheap_spec.tier_id,
                final_model=cheap_spec.model,
                final_thinking_level=cheap_spec.thinking_level,
                interaction=draft_interaction,
                prefill_decision=prefill_decision,
                verifier_confidence=ver_prob,
                total_compute_units=total_units,
                total_latency_ms=total_ms,
            )

        esc_spec = self.router.ladder.tier(self.config.escalation_tier_id)
        esc_interaction, esc_units, esc_ms = self._invoke_tier(
            esc_spec.tier_id,
            input_payload,
            system_instruction=system_instruction,
            safety_settings=safety_settings,
            generation_config=generation_config,
        )
        return CascadeResult(
            final_tier_id=esc_spec.tier_id,
            final_model=esc_spec.model,
            final_thinking_level=esc_spec.thinking_level,
            interaction=esc_interaction,
            prefill_decision=prefill_decision,
            escalated=True,
            verifier_confidence=ver_prob,
            total_compute_units=total_units + esc_units,
            total_latency_ms=total_ms + esc_ms,
        )
