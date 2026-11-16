"""Configurable routing ladder: tiers, model catalog, and compute weights loaded from TOML.

The bundled `default_tiers.toml` defines the default ladder. Users can supply their own
file via `TierLadder.from_toml(path)`, `gdr route --tiers path.toml`, or the
`GEMINI_ROUTER_TIERS` environment variable (which replaces the default ladder process-wide).
"""

import functools
import os
import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

ThinkingLevel = Literal["minimal", "low", "medium", "high", "xhigh"]

LADDER_ENV_VAR = "GEMINI_ROUTER_TIERS"
DEFAULT_LADDER_RESOURCE = "default_tiers.toml"


@dataclass(frozen=True)
class ComputeWeight:
    """Unitless relative compute weight per input and output (including thinking) token."""

    input: float
    output: float

    def units(self, input_tokens: float, output_tokens: float) -> float:
        """Compute relative compute units (in millions of reference-input-token equivalents)."""
        return (input_tokens * self.input + output_tokens * self.output) / 1_000_000.0


@dataclass(frozen=True)
class TierSpec:
    """A candidate (model, thinking_level) target in the routing ladder."""

    tier_id: str
    rank: int
    model: str
    thinking_level: ThinkingLevel
    capability_true_description: str
    capability_false_description: str


@dataclass(frozen=True)
class ModelSpec:
    """A model entry in the ladder catalog: supported thinking levels and compute weights."""

    name: str
    thinking_levels: frozenset[ThinkingLevel]
    compute_weight: ComputeWeight


# ---------------------------------------------------------------------------
# Pydantic schema for the TOML file (parsing + validation only)
# ---------------------------------------------------------------------------


_NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
_NonNegativeFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
_ConformalThresholdFloat = Annotated[float, Field(gt=0.0, le=1.0, allow_inf_nan=False)]


class _ComputeWeightCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: _NonNegativeFloat
    output: _NonNegativeFloat


class _ModelCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    thinking_levels: list[ThinkingLevel] = Field(default_factory=list)
    compute_weight: _ComputeWeightCfg


class _TierCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: Annotated[str, Field(min_length=1)]
    model: str
    thinking_level: ThinkingLevel
    can_solve: _NonEmptyStr
    cannot_solve: _NonEmptyStr
    expected_output_tokens: dict[str, _NonNegativeFloat]

    @model_validator(mode="after")
    def _has_default_tokens(self) -> _TierCfg:
        if "default" not in self.expected_output_tokens:
            raise ValueError(f"tier {self.id!r}: expected_output_tokens.default is required")
        return self


class _RouterCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision_model: str = "gemini-decision-1"
    fallback_tier: str | None = None
    frontier_tiers: list[str] | None = None
    precision_floor_tier: str | None = None
    baseline_tier: str | None = None
    score_cutoffs: list[_FiniteFloat] | None = None
    conformal_cdf_threshold: _ConformalThresholdFloat | None = None


class _LadderFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    router: _RouterCfg = Field(default_factory=_RouterCfg)
    models: Annotated[dict[str, _ModelCfg], Field(min_length=1)]
    tiers: Annotated[list[_TierCfg], Field(min_length=1)]

    @model_validator(mode="after")
    def _cross_validate(self) -> _LadderFile:
        ids = [t.id for t in self.tiers]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate tier ids: {dupes}")

        for t in self.tiers:
            m = self.models.get(t.model)
            if m is None:
                raise ValueError(f"tier {t.id!r}: model {t.model!r} is not defined in [models]")
            if t.thinking_level not in m.thinking_levels:
                raise ValueError(
                    f"tier {t.id!r}: model {t.model!r} does not support "
                    f"thinking_level={t.thinking_level!r}; allowed: {sorted(m.thinking_levels)}"
                )

        r = self.router
        if r.decision_model not in self.models:
            raise ValueError(
                f"router.decision_model {r.decision_model!r} is not defined in [models]"
            )

        id_set = set(ids)
        for label, tid in (
            ("fallback_tier", r.fallback_tier),
            ("precision_floor_tier", r.precision_floor_tier),
            ("baseline_tier", r.baseline_tier),
        ):
            if tid is not None and tid not in id_set:
                raise ValueError(f"router.{label} {tid!r} is not a defined tier id")
        for tid in r.frontier_tiers or []:
            if tid not in id_set:
                raise ValueError(f"router.frontier_tiers entry {tid!r} is not a defined tier id")

        if r.score_cutoffs is not None:
            if len(r.score_cutoffs) != len(self.tiers) - 1:
                raise ValueError(
                    f"router.score_cutoffs needs {len(self.tiers) - 1} values for "
                    f"{len(self.tiers)} tiers, got {len(r.score_cutoffs)}"
                )
            if any(b <= a for a, b in zip(r.score_cutoffs, r.score_cutoffs[1:], strict=False)):
                raise ValueError("router.score_cutoffs must be strictly increasing")
        return self


# ---------------------------------------------------------------------------
# Resolved, immutable ladder used at runtime
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TierLadder:
    """An ordered (cheapest -> most capable) set of tiers plus model catalog and compute weights."""

    tiers: tuple[TierSpec, ...]
    models: MappingProxyType[str, ModelSpec]
    decision_model: str
    fallback_tier_id: str
    frontier_tier_ids: tuple[str, ...]
    precision_floor_tier_id: str
    baseline_tier_id: str
    score_cutoffs: tuple[float, ...]
    expected_output_tokens: MappingProxyType[str, MappingProxyType[str, float]]
    conformal_cdf_threshold: float | None = None
    source: str = "<dict>"
    _index: MappingProxyType[str, int] = field(init=False, repr=False, compare=False)
    _ordered_tier_ids: tuple[str, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_index", MappingProxyType({t.tier_id: i for i, t in enumerate(self.tiers)})
        )
        object.__setattr__(self, "_ordered_tier_ids", tuple(t.tier_id for t in self.tiers))

    # --- construction -------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source: str = "<dict>") -> TierLadder:
        """Validate a parsed TOML/JSON mapping and build a `TierLadder`."""
        cfg = _LadderFile.model_validate(data)
        n = len(cfg.tiers)
        ids = [t.id for t in cfg.tiers]
        r = cfg.router

        models = {
            name: ModelSpec(
                name=name,
                thinking_levels=frozenset(m.thinking_levels),
                compute_weight=ComputeWeight(
                    input=m.compute_weight.input,
                    output=m.compute_weight.output,
                ),
            )
            for name, m in cfg.models.items()
        }
        tiers = tuple(
            TierSpec(
                tier_id=t.id,
                rank=rank,
                model=t.model,
                thinking_level=t.thinking_level,
                capability_true_description=t.can_solve,
                capability_false_description=t.cannot_solve,
            )
            for rank, t in enumerate(cfg.tiers)
        )

        slices = sorted({s for t in cfg.tiers for s in t.expected_output_tokens})
        exp_tokens = {
            sl: MappingProxyType(
                {
                    t.id: float(
                        t.expected_output_tokens.get(sl, t.expected_output_tokens["default"])
                    )
                    for t in cfg.tiers
                }
            )
            for sl in slices
        }

        return cls(
            tiers=tiers,
            models=MappingProxyType(models),
            decision_model=r.decision_model,
            fallback_tier_id=r.fallback_tier or ids[n // 2],
            frontier_tier_ids=tuple(r.frontier_tiers or [ids[-1]]),
            precision_floor_tier_id=r.precision_floor_tier or ids[min(1, n - 1)],
            baseline_tier_id=r.baseline_tier or ids[-1],
            score_cutoffs=tuple(r.score_cutoffs or default_score_cutoffs(n)),
            expected_output_tokens=MappingProxyType(exp_tokens),
            conformal_cdf_threshold=r.conformal_cdf_threshold,
            source=source,
        )

    @classmethod
    def from_toml(cls, path: str | os.PathLike[str]) -> TierLadder:
        """Load and validate a ladder from a TOML file."""
        p = Path(path)
        with p.open("rb") as fh:
            return cls.from_dict(tomllib.load(fh), source=str(p))

    @classmethod
    def bundled(cls) -> TierLadder:
        """Load the ladder shipped with the package (`default_tiers.toml`)."""
        return _load_bundled_ladder()

    # --- lookups ------------------------------------------------------------

    @property
    def ordered_tier_ids(self) -> tuple[str, ...]:
        return self._ordered_tier_ids

    @property
    def tier_by_id(self) -> dict[str, TierSpec]:
        return {t.tier_id: t for t in self.tiers}

    def __len__(self) -> int:
        return len(self.tiers)

    def tier(self, tier_id: str) -> TierSpec:
        try:
            return self.tiers[self._index[tier_id]]
        except KeyError:
            raise KeyError(
                f"Unknown tier {tier_id!r}; ladder {self.source} defines {list(self._index)}"
            ) from None

    def index(self, tier_id: str) -> int:
        return self.tier(tier_id).rank

    def compute_weight(self, model: str) -> ComputeWeight:
        spec = self.models.get(model)
        if spec is None:
            raise ValueError(f"Unknown model {model!r} in ladder {self.source}")
        return spec.compute_weight

    def validate_thinking_level(self, model: str, thinking_level: str) -> ThinkingLevel:
        """Validate that `thinking_level` is supported by `model` (prevents HTTP 400 on minimal)."""
        spec = self.models.get(model)
        if spec is None:
            raise ValueError(f"Unsupported downstream model: {model!r}")
        if thinking_level not in spec.thinking_levels:
            raise ValueError(
                f"Model {model!r} does not support thinking_level={thinking_level!r}; "
                f"allowed levels: {sorted(spec.thinking_levels)}"
            )
        return thinking_level

    def expected_output_tokens_table(self) -> dict[str, dict[str, float]]:
        """Plain-dict copy of the slice -> tier -> expected output tokens priors."""
        return {sl: dict(m) for sl, m in self.expected_output_tokens.items()}

    def expected_output_tokens_for(
        self,
        tier_id: str,
        slice_name: str = "default",
        table: dict[str, dict[str, float]] | None = None,
    ) -> float:
        """Look up expected output tokens, preferring a fitted `table` over the ladder priors."""
        prior = self.expected_output_tokens
        if table:
            slice_map = table.get(slice_name) or table.get("default") or {}
            if tier_id in slice_map:
                return float(slice_map[tier_id])
        slice_prior = prior.get(slice_name) or prior["default"]
        return float(slice_prior.get(tier_id, prior["default"][tier_id]))

    def to_summary(self) -> dict[str, Any]:
        """JSON-serializable view for CLI inspection."""
        return {
            "source": self.source,
            "decision_model": self.decision_model,
            "fallback_tier": self.fallback_tier_id,
            "frontier_tiers": list(self.frontier_tier_ids),
            "precision_floor_tier": self.precision_floor_tier_id,
            "baseline_tier": self.baseline_tier_id,
            "score_cutoffs": list(self.score_cutoffs),
            "tiers": [
                {
                    "rank": t.rank,
                    "id": t.tier_id,
                    "model": t.model,
                    "thinking_level": t.thinking_level,
                }
                for t in self.tiers
            ],
        }


def default_score_cutoffs(num_tiers: int) -> tuple[float, ...]:
    """Evenly spaced midpoints between adjacent score levels: (0.5, 1.5, ..., N-1.5)."""
    return tuple(k + 0.5 for k in range(max(0, num_tiers - 1)))


@functools.cache
def _load_bundled_ladder() -> TierLadder:
    ref = resources.files("gemini_decision_router").joinpath(DEFAULT_LADDER_RESOURCE)
    with ref.open("rb") as fh:
        return TierLadder.from_dict(tomllib.load(fh), source=f"<bundled {DEFAULT_LADDER_RESOURCE}>")


@functools.cache
def default_ladder() -> TierLadder:
    """Process-wide default ladder: `$GEMINI_ROUTER_TIERS` if set, else the bundled default.

    Resolved lazily on first call (never at import time, so a bad env path can't break
    `import gemini_decision_router`) and cached afterwards. Call `default_ladder.cache_clear()`
    after changing the environment variable.
    """
    env_path = os.environ.get(LADDER_ENV_VAR)
    if env_path:
        return TierLadder.from_toml(env_path)
    return TierLadder.bundled()


def resolve_ladder(path: str | os.PathLike[str] | None) -> TierLadder:
    """Return a `TierLadder` from `path` if given, else `default_ladder()`."""
    return TierLadder.from_toml(path) if path else default_ladder()
