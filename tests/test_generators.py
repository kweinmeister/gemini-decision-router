"""In-repo correctness and determinism tests for capability-gated dataset generators."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, get_args

import pytest

from evals.build_dataset import (
    BENCH_SEED,
    CALIB_SEED,
    PILOT_SEED,
    assert_unique_item_fingerprints,
    build_calib_split,
    build_split,
)
from evals.gen import DatasetItem, Modality, SliceName, s2_specs, s3_repair, s3_timeline, s3_vm
from evals.gen._edits import apply_edits
from evals.sandbox import SandboxBackend


def test_apply_edits_replaces_exact_once_anchors() -> None:
    source = "def foo():\n    return 1\n"
    res = apply_edits(source, [("return 1", "return 2")])
    assert res == "def foo():\n    return 2\n"


def test_apply_edits_raises_on_missing_anchor() -> None:
    source = "def foo():\n    return 1\n"
    with pytest.raises(AssertionError, match="found 0"):
        apply_edits(source, [("return 999", "return 2")])


def test_apply_edits_raises_on_duplicate_anchor() -> None:
    source = "x = 1\nx = 1\n"
    with pytest.raises(AssertionError, match="found 2"):
        apply_edits(source, [("x = 1", "x = 2")])


@pytest.mark.parametrize("variant", [0, 1, 2, 3, 4])
def test_s2_generator_is_deterministic_and_schema_compliant(variant: int) -> None:
    item_id = f"test_s2_{variant:02d}"
    b1 = s2_specs.build_s2_item(random.Random(20261101 + variant), item_id, variant)
    b2 = s2_specs.build_s2_item(random.Random(20261101 + variant), item_id, variant)

    assert b1.item == b2.item
    assert b1.reference_source == b2.reference_source
    it = b1.item
    assert it["id"] == item_id
    assert it["slice"] == "s2_everyday_code"
    assert it["modality"] == "code"
    assert len(it["prompt"]) > 50
    assert len(b1.reference_source) > 50
    assert it["grader_type"] == "python_unit_tests"
    assert "test_code" in it["grader_spec"]


def test_s3_repair_is_deterministic_and_schema_compliant() -> None:
    it1 = s3_repair.make_repair_item(random.Random(42), "test_s3_repair_00", 0)
    it2 = s3_repair.make_repair_item(random.Random(42), "test_s3_repair_00", 0)

    assert it1 == it2
    assert it1["id"] == "test_s3_repair_00"
    assert it1["slice"] == "s3_frontier_reasoning"
    assert it1["grader_type"] == "python_unit_tests"
    assert "test_code" in it1["grader_spec"]


def test_s3_vm_is_deterministic_and_schema_compliant() -> None:
    it1 = s3_vm.make_vm_item(random.Random(42), "test_s3_vm_00", 0)
    it2 = s3_vm.make_vm_item(random.Random(42), "test_s3_vm_00", 0)

    assert it1 == it2
    assert it1["id"] == "test_s3_vm_00"
    assert it1["slice"] == "s3_frontier_reasoning"
    assert it1["grader_type"] == "python_unit_tests"
    assert "test_code" in it1["grader_spec"]


def test_s3_timeline_is_deterministic_and_schema_compliant() -> None:
    it1 = s3_timeline.make_timeline_item(random.Random(42), "test_s3_timeline_00", 0)
    it2 = s3_timeline.make_timeline_item(random.Random(42), "test_s3_timeline_00", 0)

    assert it1 == it2
    assert it1["id"] == "test_s3_timeline_00"
    assert it1["slice"] == "s3_frontier_reasoning"
    assert it1["grader_type"] == "exact_json"
    assert isinstance(it1["grader_spec"]["expected"], dict)


_REQUIRED_ITEM_KEYS = set(DatasetItem.__annotations__)
_VALID_SLICES = set(get_args(SliceName))
_VALID_MODALITIES = set(get_args(Modality))


def _assert_dataset_item_conformance(item: dict[str, Any]) -> None:
    assert set(item) == _REQUIRED_ITEM_KEYS
    for str_key in ("id", "subtype", "source_provenance", "prompt", "grader_type"):
        assert isinstance(item[str_key], str) and item[str_key].strip()
    assert item["slice"] in _VALID_SLICES
    assert item["modality"] in _VALID_MODALITIES
    assert isinstance(item["attachments"], list)
    assert all(isinstance(att, str) and att for att in item["attachments"])
    assert isinstance(item["grader_spec"], dict)


def test_generated_splits_have_zero_duplicate_fingerprints_or_text_prompts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib
    import wave

    from evals.gen import s5_dialogue

    def fake_say_to_wav(text: str, voice: str, out_path: Path) -> None:
        digest = hashlib.sha256(f"{voice}:{text}".encode()).digest()
        pcm = (digest * 100)[:3200]
        with wave.open(str(out_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(s5_dialogue.SAMPLE_RATE_HZ)
            wav.writeframes(pcm)

    monkeypatch.setattr(s5_dialogue, "say_to_wav", fake_say_to_wav)

    pilot = build_split(
        seed=PILOT_SEED,
        items_per_slice=2,
        prefix="pilot",
        out_path=tmp_path / "pilot.jsonl",
    )
    bench = build_split(
        seed=BENCH_SEED,
        items_per_slice=2,
        prefix="bench",
        out_path=tmp_path / "eval.jsonl",
    )
    calib = build_calib_split(
        seed=CALIB_SEED,
        prefix="calib",
        out_path=tmp_path / "calibration.jsonl",
        s2_count=1,
        s3_count=1,
        s4_image_count=1,
        s4_pdf_count=1,
        s5_count=1,
    )
    assert len(pilot) == 10
    assert len(bench) == 10
    assert len(calib) == 5

    assert_unique_item_fingerprints(pilot, bench, calib)

    # All non-attachment text/code items must have globally distinct prompt strings across splits
    text_code_prompts = [
        str(it["prompt"]).strip() for it in (*pilot, *bench, *calib) if not it.get("attachments")
    ]
    assert len(text_code_prompts) == len(set(text_code_prompts))

    for it in (*pilot, *bench, *calib):
        _assert_dataset_item_conformance(dict(it))


@pytest.mark.slow
@pytest.mark.parametrize("variant", [0, 1, 2, 3, 4])
def test_s2_golden_reference_passes_grader(variant: int, real_backend: SandboxBackend) -> None:
    from evals.graders import grade_item

    item_id = f"test_s2_grade_{variant:02d}"
    b = s2_specs.build_s2_item(random.Random(20261101 + variant), item_id, variant)
    passed, logs = grade_item(b.item, b.reference_source, sandbox_backend=real_backend)
    assert passed, f"Golden reference failed for variant {variant}:\n{logs}"


def test_build_dataset_cli_parses_default_and_custom_sizes() -> None:
    """build_parser defaults to pilot_n=5, eval_n=30 and rejects non-positive sizes."""
    from evals.build_dataset import build_parser

    parser = build_parser()
    defaults = parser.parse_args([])
    assert defaults.pilot_n == 5
    assert defaults.eval_n == 30

    custom = parser.parse_args(["--pilot-n", "2", "--eval-n", "10"])
    assert custom.pilot_n == 2
    assert custom.eval_n == 10

    with pytest.raises(SystemExit):
        parser.parse_args(["--pilot-n", "0"])

    with pytest.raises(SystemExit):
        parser.parse_args(["--eval-n", "-1"])


def test_build_dataset_main_respects_custom_sizes_and_writes_dynamic_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """main(['--pilot-n', '2', '--eval-n', '4']) passes custom counts and writes dynamic DATASET.md."""
    from evals import build_dataset

    recorded_calls: list[tuple[str, int]] = []

    def fake_build_split(
        seed: int, items_per_slice: int, prefix: str, out_path: Path
    ) -> list[dict[str, object]]:
        recorded_calls.append((prefix, items_per_slice))
        slices = [
            "s1_trivial_text",
            "s2_everyday_code",
            "s3_frontier_reasoning",
            "s4_visual_and_pdf",
            "s5_spoken_audio",
        ]
        return [
            {
                "id": f"{prefix}_{sl}_{idx}",
                "slice": sl,
                "subtype": "stub",
                "modality": "text",
                "prompt": f"{prefix} {sl} {idx}",
                "grader_type": "exact_json",
                "grader_spec": {},
                "source_provenance": "synthetic",
            }
            for sl in slices
            for idx in range(items_per_slice)
        ]

    def fake_build_calib(seed: int, prefix: str, out_path: Path) -> list[dict[str, object]]:
        return [
            {
                "id": "calib_stub_0",
                "slice": "s1_trivial_text",
                "subtype": "stub",
                "modality": "text",
                "prompt": "calib stub",
                "grader_type": "exact_json",
                "grader_spec": {},
                "source_provenance": "synthetic",
            }
        ]

    md_path = tmp_path / "DATASET.md"
    monkeypatch.setattr(build_dataset, "DATA_DIR", tmp_path)
    monkeypatch.setattr(build_dataset, "DATASET_MD_PATH", md_path)
    monkeypatch.setattr(build_dataset, "build_split", fake_build_split)
    monkeypatch.setattr(build_dataset, "build_calib_split", fake_build_calib)

    build_dataset.main(["--pilot-n", "2", "--eval-n", "4"])

    assert recorded_calls == [("pilot", 2), ("bench", 4)]
    md_text = md_path.read_text(encoding="utf-8")
    assert "**20 evaluation items** (`data/eval.jsonl`, 4 per slice)" in md_text
    assert "**10 disjoint pilot items** (`data/pilot.jsonl`, 2 per slice)" in md_text
    assert "$N = 10$, 2 per slice" in md_text
    assert "$N = 20$, 4 per slice" in md_text
