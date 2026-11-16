# `gemini-decision-router`

A calibrated multimodal model and `thinking_level` router for a configurable ladder of Gemini models, built on Google's [Gemini Decisions API](https://aistudio.google.com/docs/decisions-api.md) (`gemini-decision-1` via `POST /v1beta/decisions`).

`gemini-decision-router` evaluates a prompt and any image, PDF, audio, or video attachments in a single zero-decoding prefill pass (no output tokens; see [`default_tiers.toml`](src/gemini_decision_router/default_tiers.toml)) and routes the request to the right model and thinking level. That avoids the multi-token decoding latency of generative router calls and the poor calibration of LLM JSON confidence scores.

## Features

### Configurable routing ladder

Tiers, the model catalog, thinking levels, and relative compute weights (`compute_weight`) are configured in [`default_tiers.toml`](src/gemini_decision_router/default_tiers.toml). The bundled default is a 5-tier ladder:

| Tier ID             | Model                   | Thinking level |
| ------------------- | ----------------------- | -------------- |
| `t0_flash_lite_min` | `gemini-3.5-flash-lite` | `minimal`      |
| `t1_hydrogen_low`   | `gemini-4-hydrogen`     | `low`          |
| `t2_hydrogen_med`   | `gemini-4-hydrogen`     | `medium`       |
| `t3_hydrogen_high`  | `gemini-4-hydrogen`     | `high`         |
| `t4_hydrogen_xhigh` | `gemini-4-hydrogen`     | `xhigh`        |

`gemini-3.8-flash`, `gemini-4-argon`, and `gemini-4-carbon` also have `compute_weight` entries in the catalog for custom ladders; see [Custom tiers](#7-custom-tiers-toml). On load, `TierLadder` checks that every model supports its tier's thinking level, every model defines `compute_weight`, and `score_cutoffs` has one fewer entry than the number of tiers.

### Router-pass media compression

For the routing pass, `envelope.py` clones image and video blocks at low resolution and slices multi-page PDFs to Page 1 plus a page-count and byte-size metadata note to substantially reduce router input tokens. The selected downstream model still receives the untouched, full-resolution payload.

### Calibrated routing policies

`router.py` supports three Decisions API primitives (`score`, `choice`, `noul`) exposed as six `RouterPolicyConfig.policy` modes:

| Policy                   | How it selects a tier                                                                                                                                                                                      | Key knobs (`RouterPolicyConfig` / TOML)                                                         | Best for                                                                             |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `score_tail` _(default)_ | Bins the expected rubric `score` via `score_cutoffs`, escalates to the lowest-ranked `frontier_tiers` if their combined tail mass is high, and applies `precision_floor_tier` when `needs_precision` fires | `score_cutoffs`, `score_frontier_tail_threshold` (`0.35`), `needs_precision_threshold` (`0.65`) | Production default — guards against under-routing hard/ambiguous prompts (tail risk) |
| `choice_tail`            | Picks the lowest-ranked tier whose cumulative `choice` probability reaches the target threshold (or bins the expected rank if `score_cutoffs` is set)                                                      | `choice_cdf_threshold` (`0.70`), optional `score_cutoffs`                                       | Simpler single-question tail-risk control without a separate `needs_precision` probe |
| `choice_crc`             | Uses a 1D Conformal Risk Control cumulative `choice` threshold calibrated against an accuracy-regret budget $\alpha$ via `calibrate_conformal_cdf_threshold`                                               | `conformal_cdf_threshold` (falls back to `[router].conformal_cdf_threshold` or `choice_cdf_threshold`) | Finite-sample accuracy-drop guarantees once you have graded eval examples            |
| `noul_eu`                | Runs parallel `noul` (yes/no) solvability probes per tier, optionally calibrates via `fit_per_tier_platt_scaling`, projects via PAVA (`pava_isotonic_increasing`), and maximizes $\hat{p}_{i,t} - \lambda \hat{u}_{i,t}$ | `eu_lambda` (`6.0`), `noul_platt_params`, `expected_output_tokens_table`                        | Input-length-aware cost/compute vs. accuracy trade-offs (e.g., very long prompts)    |
| `score_argmax`           | Picks the single highest-probability tier from the `score` distribution (ignores tail mass and `needs_precision`)                                                                                          | —                                                                                               | Baselines or ablations                                                               |
| `choice_argmax`          | Picks the single highest-probability tier from the `choice` distribution                                                                                                                                   | —                                                                                               | Baselines or ablations                                                               |

Every policy sets `RouteDecision.confidence` to the estimated probability that the selected tier suffices: cumulative mass at or below the selected tier for `score_*` and `choice_*`, or the PAVA-projected solvability for `noul_eu`. Safety or parse fallbacks report `0.0` with `fallback_triggered=True`.

### Observability and thinking-aware compute accounting

`RouterObserver` in `observability.py` tracks live tier shares, mean and p95 router latency, safety fallback rate, estimated compute savings against a static baseline tier, Expected Calibration Error, Brier score, Log-Loss, and Jensen-Shannon distribution drift.

Because the Interactions API reports thinking tokens separately from visible output tokens, `Usage.from_response(...)` captures both counts and `calculate_compute_units` in `compute.py` weights their sum at the output compute weight.

> [!NOTE]
> **Relative compute weights:** The unitless `compute_weight` values in [`default_tiers.toml`](src/gemini_decision_router/default_tiers.toml) (`1.0` ≈ one `gemini-4-hydrogen` input token) are approximate defaults for `noul_eu` trade-offs and `RouterObserver` savings percentages. They are not billing rates or currency quotes. If you want to customize weights or track actual spend from token counts, refer to [Gemini API Pricing](https://aistudio.google.com/docs/pricing).

## Installation

From a fresh clone (requires Python 3.14+ and [uv](https://docs.astral.sh/uv/)):

```bash
git clone https://github.com/kweinmeister/gemini-decision-router.git
cd gemini-decision-router
uv sync

# Set your Gemini API key (or copy .env.example to .env)
export GEMINI_API_KEY="your-api-key"
```

Or install as a library dependency in an existing project:

```bash
uv add gemini-decision-router
# or: pip install gemini-decision-router
```

## Quickstart

### 1. Python SDK (`DecisionRouter`)

```python
from gemini_decision_router import DecisionRouter, RouterObserver, RouterPolicyConfig

# Optional: attach a lightweight telemetry & calibration observer
observer = RouterObserver(baseline_tier_id="t4_hydrogen_xhigh")

router = DecisionRouter(
    config=RouterPolicyConfig(
        policy="score_tail",  # "score_tail" | "choice_tail" | "choice_crc" | "noul_eu" | "score_argmax" | "choice_argmax"
        compress_media_for_router=True,
    ),
    observer=observer,
)

# 1. Route only (returns a structured RouteDecision)
decision = router.route("Prove that every finite integral domain is a field.")
print(decision.selected_tier_id, decision.model, decision.thinking_level)
# -> t3_hydrogen_high gemini-4-hydrogen high

# 2. Route and immediately execute on the chosen downstream model via Interactions API
decision, interaction = router.route_and_execute(
    "Extract the ISO-8601 date from: 'Rollout completed on November 11, 2026.'"
)
print(interaction.output_text)
```

### 2. Multimodal routing (images, PDFs, audio, video)

```python
from gemini_decision_router import DecisionRouter, load_media_block

router = DecisionRouter()

# Load local attachment safely (enforces canonical directory bounds)
pdf_block = load_media_block("reports/q4_audit.pdf", allowed_root=".")
payload = [
    pdf_block,
    {"type": "text", "text": "Cross-reference Page 1 with Appendix C and compute net billed MWh."},
]

# The router automatically slices q4_audit.pdf to Page 1 + metadata for the decision pass,
# then sends the full multi-page PDF to the selected downstream model!
decision, interaction = router.route_and_execute(payload)
```

### 3. Production observability and calibration (`RouterObserver`)

```python
from gemini_decision_router import (
    DecisionRouter,
    RouterObserver,
    Usage,
    compute_calibration_metrics,
    jensen_shannon_divergence,
)

observer = RouterObserver(baseline_tier_id="t4_hydrogen_xhigh")
router = DecisionRouter(observer=observer)

dec, interaction = router.route_and_execute(
    "Write a Python function to merge overlapping intervals."
)
# Optionally record whether the downstream call succeeded (e.g. unit test / user thumbs-up).
# This updates the observation route() already recorded (each request is counted once);
# `usage` sets the realized downstream compute units, weighting thinking tokens at the output rate.
observer.record_outcome(
    dec,
    success=True,
    usage=Usage.from_response(interaction),  # or Usage(input=45, output=110, thought=64)
)

summary = observer.summary()
print(f"Requests routed: {summary.total_requests}")
print(f"Tier shares:     {summary.tier_shares}")
print(
    f"Compute savings: {summary.estimated_compute_savings_pct:.1f}% vs {observer.baseline_tier_id}"
)
print(
    f"Live ECE:        {summary.calibration['ece']:.4f} (Brier={summary.calibration['brier_score']:.4f})"
)
```

Calibration pairs each recorded outcome with `decision.confidence`; fallback decisions are excluded from calibration and reported via `fallback_rate` instead.

### 4. Hybrid Cascade-Routing (pure cascade or prefill-skip)

```python
from gemini_decision_router import CascadeConfig, DecisionRouter, HybridCascadeRouter

cascade = HybridCascadeRouter(
    DecisionRouter(),
    config=CascadeConfig(
        cheap_tier_id="t0_flash_lite_min",
        escalation_tier_id="t1_hydrogen_low",
        use_prefill_skip=True,  # False for a pure 2-stage cascade without prefill routing
        prefill_accept_threshold=0.75,
        accept_threshold=0.70,
    ),
)
result = cascade.run("Summarize this incident timeline and identify the root cause.")
print(result.final_tier_id, result.prefill_skipped_cheap, result.escalated)
```

### 5. Conformal Risk Control (`choice_crc`) and Platt calibration

```python
from gemini_decision_router import (
    DecisionRouter,
    RouterPolicyConfig,
    calibrate_conformal_cdf_threshold,
    default_ladder,
    fit_per_tier_platt_scaling,
)

ladder = default_ladder()

# Fit a finite-sample Conformal Risk Control threshold guaranteeing E[regret] <= target_alpha
crc = calibrate_conformal_cdf_threshold(
    choice_probs_list,
    tier_accuracies_list,
    ladder.ordered_tier_ids,
    target_alpha=0.015,
)
crc_router = DecisionRouter(
    config=RouterPolicyConfig(policy="choice_crc", conformal_cdf_threshold=crc.threshold)
)

# Optional: fit per-tier Platt scaling on labeled noul solvability pairs before PAVA in noul_eu
platt = fit_per_tier_platt_scaling(noul_pairs_by_tier)
eu_router = DecisionRouter(config=RouterPolicyConfig(policy="noul_eu", noul_platt_params=platt))
```

### 6. Command-line interface (`gdr`)

```bash
# Inspect routing decision as JSON
uv run gdr route "Write a Python function to merge overlapping intervals." --policy score_tail

# Route with a calibrated Conformal Risk Control threshold
uv run gdr route "Prove Cauchy's theorem for finite groups." \
  --policy choice_crc \
  --conformal-cdf-threshold 0.82

# Route a multimodal prompt with an image/PDF/audio attachment and execute downstream
uv run gdr route "Summarize the key anomalies in this chart." \
  --attach path/to/chart.png \
  --execute

# Validate and print the active tier ladder
uv run gdr tiers --tiers my_tiers.toml
```

### 7. Custom tiers (TOML)

Copy [`default_tiers.toml`](src/gemini_decision_router/default_tiers.toml), edit the `[models]` catalog and `[[tiers]]` list from cheapest to most capable, then load it via the SDK, CLI, or `GEMINI_ROUTER_TIERS`:

```python
from gemini_decision_router import DecisionRouter, TierLadder

ladder = TierLadder.from_toml("my_tiers.toml")
router = DecisionRouter(ladder=ladder)
```

```bash
uv run gdr route "..." --tiers my_tiers.toml       # CLI flag
export GEMINI_ROUTER_TIERS=my_tiers.toml            # process-wide default (also used by the evals)
```

`default_ladder()` returns `$GEMINI_ROUTER_TIERS` when set, or the bundled file otherwise. It resolves lazily on first use so a bad path never breaks `import gemini_decision_router`, and caches the result; call `default_ladder.cache_clear()` after changing the environment variable at runtime.

A minimal tier entry:

```toml
[[tiers]]
id = "t4_carbon_high"
model = "gemini-4-carbon"          # must exist in [models] with "high" in thinking_levels
thinking_level = "high"
can_solve = "Solvable by a flagship frontier reasoning model given maximum deliberation budget."
cannot_solve = "Ill-posed, contradictory, or missing information required to produce a verifiable answer."
expected_output_tokens = { default = 2800 }   # compute prior; per-slice keys optional
```

`can_solve` (paired with `cannot_solve` for `solvability` binary choice questions) becomes the Decisions API criteria. `[router]` sets `fallback_tier`, `frontier_tiers` for `score_tail` escalation, `precision_floor_tier`, `baseline_tier`, optional `score_cutoffs`, and optional `conformal_cdf_threshold` (for `choice_crc`). Any omitted `[router]` field falls back to a default derived from the tier list.

## Benchmarks and reproducible evaluation (`evals/`)

`evals/` contains the pre-registered 5-slice multimodal benchmark harness comparing `gemini-decision-1` (`score`, `choice`, `choice_crc`, PAVA-projected `noul_eu`, and Platt-calibrated `noul_eu`) plus pure and hybrid cascades (`--cascade-matrix` / `--cascade-verifiers`) against prompted `gemini-3.5-flash-lite` Structured Outputs (`flash-lite-json`), static single-tier baselines, and an unbiased Leave-One-Replicate-Out (`LORO`) Oracle.

- [`evals/README.md`](evals/README.md): step-by-step instructions to generate the local benchmark dataset and run the pilot and full evaluation matrix.
- [`evals/DATASET.md`](evals/DATASET.md): per-slice provenance and deterministic grading specifications for the generated benchmark splits.

## HTTP Server & Cloud Run Deployment

[`server.py`](src/gemini_decision_router/server.py) provides a thin FastAPI wrapper around `DecisionRouter` for local serving or Cloud Run (`Dockerfile`, `.dockerignore`, `.gcloudignore`, and `cloudbuild.yaml`).

### Run locally

```bash
uv sync --extra server
uv run uvicorn gemini_decision_router.server:app --host 0.0.0.0 --port 8080
```

Endpoints:

- `GET /health` — liveness check (`{"status": "ok"}`)
- `GET /v1/tiers` — active tier ladder summary
- `GET /v1/metrics` — aggregated in-memory `RouterObserver` metrics
- `POST /v1/route` — returns the `RouteDecision` (`{"input": "..."}` or multimodal `ContentBlock` list, optional `policy`, `compress_media_for_router`, `conformal_cdf_threshold`)
- `POST /v1/route-and-execute` — routes the request and invokes the selected model (`{"input": "...", "max_output_tokens": 256}`, optional `system_instruction`)

### Deploy to Cloud Run

Store `GEMINI_API_KEY` in Secret Manager (`gemini-api-key`), then deploy directly from source:

```bash
gcloud run deploy gemini-decision-router \
  --source . \
  --region us-central1 \
  --project "$PROJECT_ID" \
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest \
  --set-env-vars GOOGLE_CLOUD_PROJECT="$PROJECT_ID" \
  --no-allow-unauthenticated
```

Or build and deploy via [`cloudbuild.yaml`](cloudbuild.yaml):

```bash
gcloud builds submit --config cloudbuild.yaml --project "$PROJECT_ID" .
```

Test an authenticated service:

```bash
SERVICE_URL=$(gcloud run services describe gemini-decision-router --region us-central1 --project "$PROJECT_ID" --format 'value(status.url)')
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H "Content-Type: application/json" \
  -d '{"input": "Explain TCP vs UDP in two sentences."}' \
  "$SERVICE_URL/v1/route-and-execute"
```

### Structured Cloud Logging

Each `/v1/route` and `/v1/route-and-execute` call emits single-line JSON to `stdout` with `severity`, `selected_tier_id`, `model`, `thinking_level`, `policy_name`, `confidence`, `router_latency_ms`, token/compute-unit estimates, fallback flags, `input_type`, and `logging.googleapis.com/trace` (when `GOOGLE_CLOUD_PROJECT` and `X-Cloud-Trace-Context` are present). Raw prompt text, `downstream_output`, and media bytes are never logged.
