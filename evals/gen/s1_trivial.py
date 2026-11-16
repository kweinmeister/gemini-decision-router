"""Trivial text item generator for `s1_trivial_text`."""

from __future__ import annotations

from evals.gen._schema import DatasetItem, item_rng

_DEPARTMENTS = (
    "Payments",
    "CoreInfra",
    "Search",
    "TrustSafety",
    "CloudStorage",
    "MLPlatform",
    "IdentityAuth",
    "EdgeRouting",
)
_REGIONS = (
    "us-east1",
    "europe-west4",
    "asia-northeast1",
    "us-central1",
    "us-west1",
    "europe-north1",
)
_TOPICS: tuple[tuple[str, list[str]], ...] = (
    ("database connection pooling", ["latency", "socket"]),
    ("immutable container images", ["digest", "rollback"]),
    ("exponential backoff jitter", ["retry", "thundering"]),
    ("structured JSON logging", ["trace", "parser"]),
    ("content-addressable caching", ["hash", "hit"]),
    ("write-ahead transaction logs", ["checkpoint", "recovery"]),
    ("read-replica lag monitoring", ["quorum", "stale"]),
    ("token-bucket rate limiting", ["burst", "refill"]),
    ("blue-green traffic shifting", ["canary", "drain"]),
    ("mutual TLS certificate rotation", ["handshake", "expiry"]),
    ("columnar parquet compression", ["predicate", "stripe"]),
    ("dead-letter message queues", ["poison", "replay"]),
    ("consistent hash ring rebalancing", ["virtual", "partition"]),
    ("gRPC deadline propagation", ["timeout", "upstream"]),
    ("idempotency key deduplication", ["ledger", "duplicate"]),
    ("ephemeral build sandboxes", ["isolation", "teardown"]),
    ("bloom filter membership checks", ["false", "memory"]),
    ("zero-downtime schema migrations", ["backfill", "dual"]),
    ("distributed lock leases", ["fencing", "heartbeat"]),
    ("HTTP conditional ETag caching", ["revalidation", "bandwidth"]),
)
_FORBIDDEN_WORDS = (
    "obviously",
    "simply",
    "merely",
    "effortlessly",
    "seamlessly",
    "magically",
    "trivially",
)
_END_TOKENS = ("[END]", "[DONE]", "[EOF]", "[COMPLETE]", "[SUMMARY_END]")
_DATE_EVENTS = (
    "firmware rollout",
    "TLS root certificate cutover",
    "ledger schema migration",
    "regional failover drill",
    "telemetry pipeline upgrade",
    "storage cluster rebalance",
    "kernel security patch",
    "billing batch reconciliation",
)
_MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def make_s1_items(seed: int, count: int, prefix: str) -> list[DatasetItem]:
    """Generate `s1_trivial_text` items: JSON extraction, IFEval constraints, and ISO-8601 date normalization."""
    items: list[DatasetItem] = []
    for i in range(count):
        item_id = f"{prefix}_s1_{i:02d}"
        rng = item_rng(seed, item_id)
        mode = i % 3
        if mode == 0:
            dept = rng.choice(_DEPARTMENTS)
            reg = rng.choice(_REGIONS)
            ticket_num = rng.randint(10000, 99999)
            sev = f"SEV-{rng.randint(1, 4)}"
            duration_min = rng.randint(12, 240)
            prompt = (
                f"Extract the incident fields from the note below and return ONLY a valid JSON object "
                f"with keys `ticket_id` (int), `department` (string), `region` (string), `severity` (string), "
                f"and `duration_minutes` (int).\n\n"
                f"Incident Note: On-call engineer paged for ticket #{ticket_num} in the {dept} service "
                f"running in {reg}. Classified as {sev} and mitigated after {duration_min} minutes."
            )
            items.append(
                {
                    "id": item_id,
                    "slice": "s1_trivial_text",
                    "modality": "text",
                    "subtype": "json_field_extraction",
                    "source_provenance": "Parameterized incident extraction template",
                    "prompt": prompt,
                    "attachments": [],
                    "grader_type": "exact_json",
                    "grader_spec": {
                        "expected": {
                            "ticket_id": ticket_num,
                            "department": dept,
                            "region": reg,
                            "severity": sev,
                            "duration_minutes": duration_min,
                        }
                    },
                }
            )
        elif mode == 1:
            topic_name, req_kw = rng.choice(_TOPICS)
            bullets = rng.randint(3, 6)
            forbidden_word = rng.choice(_FORBIDDEN_WORDS)
            min_words = rng.choice([25, 30, 35])
            max_words = min_words + rng.choice([55, 65, 75])
            suffix = rng.choice(_END_TOKENS)
            prompt = (
                f"Write a concise summary about {topic_name} obeying ALL of the following rules:\n"
                f"1. Format the entire response as {bullets} markdown bullet points (each line starting with `- `).\n"
                f"2. Include the exact words `{req_kw[0]}` and `{req_kw[1]}` somewhere in the text.\n"
                f"3. Do NOT use the word `{forbidden_word}`.\n"
                f"4. Keep the total response between {min_words} and {max_words} words.\n"
                f"5. End the final line with the exact token `{suffix}`."
            )
            items.append(
                {
                    "id": item_id,
                    "slice": "s1_trivial_text",
                    "modality": "text",
                    "subtype": "ifeval_single_turn_constraints",
                    "source_provenance": "google/IFEval-style deterministic rule template",
                    "prompt": prompt,
                    "attachments": [],
                    "grader_type": "ifeval_constraints",
                    "grader_spec": {
                        "constraints": {
                            "exact_bullet_count": bullets,
                            "required_keywords": req_kw,
                            "forbidden_words": [forbidden_word],
                            "min_words": min_words,
                            "max_words": max_words,
                            "ends_with": suffix,
                        }
                    },
                }
            )
        else:
            year = rng.randint(2018, 2034)
            month = rng.randint(1, 12)
            day = rng.randint(1, 28)
            event_label = rng.choice(_DATE_EVENTS)
            rel_tag = f"v{rng.randint(1, 9)}.{rng.randint(0, 24)}.{rng.randint(0, 15)}"
            iso_date = f"{year:04d}-{month:02d}-{day:02d}"
            prompt = (
                f"Convert the release date in the sentence below to strict ISO-8601 format (`YYYY-MM-DD`) "
                f"and end your response with `FINAL_ANSWER: <YYYY-MM-DD>`.\n\n"
                f"Sentence: The {event_label} ({rel_tag}) completed on {_MONTH_NAMES[month - 1]} {day}, {year}."
            )
            items.append(
                {
                    "id": item_id,
                    "slice": "s1_trivial_text",
                    "modality": "text",
                    "subtype": "canonical_date_normalization",
                    "source_provenance": "Deterministic ISO-8601 normalization benchmark",
                    "prompt": prompt,
                    "attachments": [],
                    "grader_type": "canonical_string",
                    "grader_spec": {"expected": iso_date},
                }
            )
    return items
