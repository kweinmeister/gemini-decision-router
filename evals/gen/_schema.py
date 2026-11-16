"""Shared dataset-item contract and per-item RNG helper for `evals.gen`."""

import random
from typing import Any, Literal, TypedDict

SliceName = Literal[
    "s1_trivial_text",
    "s2_everyday_code",
    "s3_frontier_reasoning",
    "s4_visual_and_pdf",
    "s5_spoken_audio",
]
Modality = Literal["text", "code", "image", "pdf", "audio"]


class DatasetItem(TypedDict):
    """Canonical schema returned by every slice generator in `evals.gen` and `evals.build_dataset`."""

    id: str
    slice: SliceName
    modality: Modality
    subtype: str
    source_provenance: str
    prompt: str
    attachments: list[str]
    grader_type: str
    grader_spec: dict[str, Any]


def item_rng(seed: int, item_id: str) -> random.Random:
    """Return an item's deterministic RNG (`random.Random(f'{seed}:{item_id}')`)."""
    return random.Random(f"{seed}:{item_id}")
