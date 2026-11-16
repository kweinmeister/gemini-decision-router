"""Calibrated multimodal model and thinking-level router using Gemini Decisions API."""

import warnings

from .calibration import (
    PlattParams,
    fit_per_tier_platt_scaling,
    fit_platt_scaling,
)
from .cascade import (
    CascadeConfig,
    CascadeResult,
    HybridCascadeRouter,
    build_verification_questions,
)
from .compute import (
    calculate_compute_units,
    estimate_tier_compute_units,
)
from .conformal import (
    ConformalCalibrationResult,
    calibrate_conformal_cdf_threshold,
    select_tier_from_cdf_threshold,
)
from .envelope import (
    ImageResolution,
    PdfRouterMode,
    compress_blocks_for_router,
    estimate_input_tokens,
    extract_interaction_text,
    extract_modality_tokens,
    load_media_block,
    resolve_safe_attachment_path,
)
from .ladder import (
    ComputeWeight,
    ModelSpec,
    TierLadder,
    default_ladder,
)
from .models import (
    RouteDecision,
    TierID,
    TierSpec,
    pava_isotonic_increasing,
    validate_thinking_level,
)
from .observability import (
    RouteObservation,
    RouterObservabilitySummary,
    RouterObserver,
    compute_calibration_metrics,
    jensen_shannon_divergence,
    spearman_rank_correlation,
)
from .router import DecisionRouter, RouterPolicyConfig, create_genai_client
from .usage import Usage

# Suppress google-genai client SDK UserWarning when forward-passing newly introduced
# thinking levels (such as 'xhigh') supported by the backend Gemini API.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*is not a valid ThinkingLevel.*",
)

__all__ = [
    "CascadeConfig",
    "CascadeResult",
    "ComputeWeight",
    "ConformalCalibrationResult",
    "DecisionRouter",
    "HybridCascadeRouter",
    "ImageResolution",
    "ModelSpec",
    "PdfRouterMode",
    "PlattParams",
    "RouteDecision",
    "RouteObservation",
    "RouterObservabilitySummary",
    "RouterObserver",
    "RouterPolicyConfig",
    "TierID",
    "TierLadder",
    "TierSpec",
    "Usage",
    "build_verification_questions",
    "calculate_compute_units",
    "calibrate_conformal_cdf_threshold",
    "compress_blocks_for_router",
    "compute_calibration_metrics",
    "create_genai_client",
    "default_ladder",
    "estimate_input_tokens",
    "estimate_tier_compute_units",
    "extract_interaction_text",
    "extract_modality_tokens",
    "fit_per_tier_platt_scaling",
    "fit_platt_scaling",
    "jensen_shannon_divergence",
    "load_media_block",
    "pava_isotonic_increasing",
    "resolve_safe_attachment_path",
    "select_tier_from_cdf_threshold",
    "spearman_rank_correlation",
    "validate_thinking_level",
]
