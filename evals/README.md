# Reproducible evaluation suite

The `evals/` harness benchmarks `gemini-decision-1` as a multimodal model and thinking-level router across five task slices: 150 evaluation items across 5 tiers and 3 replicates, plus a 25-item pilot split and a 21-item hard calibration split. See [DATASET.md](DATASET.md) for per-slice and per-item provenance.

## 1. Install extras and container sandbox

```bash
uv sync --extra evals

# Start air-gapped container runtime for grading s2_everyday_code & s3_frontier_reasoning
container system status || container system start
container image pull python:3.14-slim
```

## 2. Build datasets and multimodal assets

```bash
uv run python -m evals.build_dataset
```

This writes the three JSONL splits under `data/`, their multimodal assets under `data/assets/`, and the provenance table in [DATASET.md](DATASET.md):

| Output                   | Contents                                                                           |
| ------------------------ | ---------------------------------------------------------------------------------- |
| `data/pilot.jsonl`       | 25 pilot items, 5 per slice                                                        |
| `data/eval.jsonl`        | 150 evaluation items, 30 per slice                                                 |
| `data/calibration.jsonl` | 21 items covering every capability-gated subtype                                   |
| `data/assets/images/`    | Matplotlib charts, scatter plots, and weighted directed graphs                     |
| `data/assets/pdfs/`      | Multi-page programmatic PDF reports, including raster-only scans with errata pages |
| `data/assets/audio/`     | 16 kHz spoken English WAV clips, including three-person shift handoffs             |

Each item uses its own seeded RNG, so rebuilding is byte-for-byte reproducible and changing one generator leaves other items untouched. Capability-gated generators live in [gen/](gen/).

## 3. Check tier separation on the calibration split

Run every tier once on the 21 hard items before spending quota on the full suite:

```bash
uv run python -m evals.collect_matrix \
  --dataset data/calibration.jsonl \
  --output results/calib_matrix.jsonl \
  --replicates 1 \
  --sandbox-backend container

# Pass rate per tier
uv run python -c "import collections, json; rows = [json.loads(l) for l in open('results/calib_matrix.jsonl')]; c = collections.defaultdict(list); [c[r['tier_id']].append(r['passed']) for r in rows]; print({t: f'{sum(v)}/{len(v)}' for t, v in c.items()})"
```

The calibration split is never used for training or evaluation.

## 4. Run the 25-item pilot

```bash
uv run python -m evals.routers \
  --dataset data/pilot.jsonl \
  --output results/pilot_router_telemetry.jsonl

uv run python -m evals.collect_matrix \
  --dataset data/pilot.jsonl \
  --output results/pilot_matrix.jsonl \
  --replicates 1 \
  --sandbox-backend container

uv run python -m evals.analyze \
  --dataset data/pilot.jsonl \
  --matrix results/pilot_matrix.jsonl \
  --routers results/pilot_router_telemetry.jsonl \
  --output results/pilot_summary.json
```

## 5. Run the 150-item evaluation

```bash
# Stage 2: Collect router telemetry (score, choice, noul, flash-lite-json, and Q4 media ablations)
uv run python -m evals.routers \
  --dataset data/eval.jsonl \
  --output results/router_telemetry.jsonl \
  --concurrency 2

# Stage 1: Collect 150 x 5 x 3 = 2,250 downstream calls (resumable JSONL)
uv run python -m evals.collect_matrix \
  --dataset data/eval.jsonl \
  --output results/downstream_matrix.jsonl \
  --replicates 3 \
  --concurrency 2 \
  --sandbox-backend container

# Stage 3: Run 5-fold stratified CV, LORO Oracle, ECE/Brier/LogLoss, Q4 fidelity, and 10k paired bootstrap
uv run python -m evals.analyze \
  --dataset data/eval.jsonl \
  --matrix results/downstream_matrix.jsonl \
  --routers results/router_telemetry.jsonl \
  --output results/summary.json

# Render Pareto, Latency Break-Even, and Calibration Reliability plots
uv run python -m evals.plots \
  --summary results/summary.json \
  --out-dir results/plots
```

## 6. Custom tier ladders and compute accounting

The `routers`, `collect_matrix`, and `analyze` commands all accept `--tiers my_tiers.toml`, defaulting to `$GEMINI_ROUTER_TIERS` or [default_tiers.toml](../src/gemini_decision_router/default_tiers.toml). Pass the same ladder file to all three stages: `analyze` exits with an error if the matrix is missing any ladder tier or if router telemetry used a different tier list. Extra matrix tiers outside the active ladder are ignored, so a matrix collected with extra tiers can still be analyzed under a smaller ladder. The summary saves the ladder it used, and `plots` reads its baseline tier from there.

Each matrix and telemetry row includes a relative `compute_units` field derived from each model's `compute_weight` (`input` and `output`). Thinking tokens are weighted at the output rate, which accounts for most of the compute on medium and high thinking tiers.
