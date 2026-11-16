"""Unit tests for multimodal input envelope compression and path-traversal guards."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image
from pypdf import PdfReader, PdfWriter

import gemini_decision_router.envelope as envelope_mod
from gemini_decision_router.envelope import (
    ImageResolution,
    _read_and_encode_file,
    compress_blocks_for_router,
    estimate_input_tokens,
    extract_interaction_text,
    extract_modality_tokens,
    load_media_block,
    payload_has_compressible_media,
    prepare_router_envelope,
    render_pdf_first_page_png_bytes,
    resolve_safe_attachment_path,
    slice_pdf_first_page_bytes,
)


def _create_sample_pdf_bytes(num_pages: int = 4) -> bytes:
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_resolve_safe_attachment_path_blocks_traversal(tmp_path: Path) -> None:
    allowed = tmp_path / "workspace"
    allowed.mkdir()
    inside_file = allowed / "chart.png"
    inside_file.write_bytes(b"pngdata")

    outside_file = tmp_path / "secret.txt"
    outside_file.write_text("secret")

    assert resolve_safe_attachment_path("chart.png", allowed) == inside_file.resolve()

    with pytest.raises(PermissionError, match="outside allowed directory"):
        resolve_safe_attachment_path("../secret.txt", allowed)

    with pytest.raises(PermissionError, match="outside allowed directory"):
        resolve_safe_attachment_path(outside_file, allowed)


def test_compress_blocks_non_mutating_and_slices_pdf() -> None:
    pdf_4p = _create_sample_pdf_bytes(4)
    b64_pdf = base64.b64encode(pdf_4p).decode("ascii")

    original_blocks = [
        {"type": "image", "mime_type": "image/png", "data": "AAAA"},
        {"type": "audio", "mime_type": "audio/wav", "data": "BBBB"},
        {"type": "document", "mime_type": "application/pdf", "data": b64_pdf},
        {"type": "text", "text": "Analyze all attachments."},
    ]

    compressed = compress_blocks_for_router(
        original_blocks,
        image_resolution="low",
        pdf_mode="page1_pdf",
    )
    assert isinstance(compressed, list)

    # Original payload must remain untouched for downstream full-resolution model execution
    assert "resolution" not in original_blocks[0]
    orig_pdf_decoded = base64.b64decode(original_blocks[2]["data"])
    assert len(PdfReader(io.BytesIO(orig_pdf_decoded)).pages) == 4

    # Compressed payload has resolution="low" on image, no resolution on audio, and 1-page PDF + metadata
    assert compressed[0]["resolution"] == "low"
    assert "resolution" not in compressed[1]
    assert compressed[2]["type"] == "text"
    assert "total_pages=4" in compressed[2]["text"]

    comp_pdf_decoded = base64.b64decode(compressed[3]["data"])
    assert len(PdfReader(io.BytesIO(comp_pdf_decoded)).pages) == 1


def test_load_media_block_caching_and_immutability(tmp_path: Path) -> None:
    _read_and_encode_file.cache_clear()
    root = tmp_path / "assets"
    root.mkdir()
    img_file = root / "test.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\nfakeimagebytes")

    block1 = load_media_block("test.png", root, resolution="low")
    block2 = load_media_block("test.png", root, resolution="high")

    # Cache must hit for the underlying read and base64 encode
    assert _read_and_encode_file.cache_info().hits >= 1
    # Independent dictionaries returned so mutations don't leak
    assert block1 is not block2
    assert block1["resolution"] == "low"
    assert block2["resolution"] == "high"
    assert block1["data"] == block2["data"]


def test_resolve_safe_attachment_path_missing_dir_and_symlink_escape(tmp_path: Path) -> None:
    allowed = tmp_path / "workspace"
    allowed.mkdir()
    subdir = allowed / "nested"
    subdir.mkdir()

    outside = tmp_path / "outside.png"
    outside.write_bytes(b"secret-image")
    escaped_link = allowed / "escape.png"
    escaped_link.symlink_to(outside)

    with pytest.raises(FileNotFoundError, match="Attachment file not found"):
        resolve_safe_attachment_path("does_not_exist.png", allowed)

    with pytest.raises(FileNotFoundError, match="Attachment file not found"):
        resolve_safe_attachment_path("nested", allowed)

    with pytest.raises(PermissionError, match="outside allowed directory"):
        resolve_safe_attachment_path("escape.png", allowed)


@pytest.mark.parametrize(
    ("filename", "expected_type", "expected_mime", "resolution"),
    [
        ("clip.mp4", "video", "video/mp4", "low"),
        ("clip.mp4", "video", "video/mp4", None),
        ("audio.wav", "audio", "audio/wav", None),
        ("audio.mp3", "audio", "audio/mp3", None),
        ("photo.jpg", "image", "image/jpeg", "high"),
        ("doc.pdf", "document", "application/pdf", None),
    ],
)
def test_load_media_block_supported_modalities(
    tmp_path: Path,
    filename: str,
    expected_type: str,
    expected_mime: str,
    resolution: ImageResolution | None,
) -> None:
    _read_and_encode_file.cache_clear()
    asset = tmp_path / filename
    asset.write_bytes(b"sample-bytes")

    block = load_media_block(filename, tmp_path, resolution=resolution)

    assert block["type"] == expected_type
    assert block["mime_type"] == expected_mime
    if expected_type in ("image", "video") and resolution is not None:
        assert block["resolution"] == resolution
    else:
        assert "resolution" not in block


def test_load_media_block_rejects_unsupported_mime(tmp_path: Path) -> None:
    _read_and_encode_file.cache_clear()
    unknown = tmp_path / "payload.bin"
    unknown.write_bytes(b"\x00\x01\x02")

    with pytest.raises(ValueError, match="Unsupported or generic MIME type"):
        load_media_block("payload.bin", tmp_path)


def test_compress_blocks_variants_and_invalid_pdf_mode() -> None:
    # String payloads pass through unchanged
    assert compress_blocks_for_router("plain text") == "plain text"

    # 1-page PDF returns original bytes unchanged from slice helper
    pdf_1p = _create_sample_pdf_bytes(1)
    sliced_bytes, count = slice_pdf_first_page_bytes(pdf_1p)
    assert count == 1
    assert sliced_bytes == pdf_1p

    pdf_2p = _create_sample_pdf_bytes(2)
    b64_pdf = base64.b64encode(pdf_2p).decode("ascii")
    payload = [
        {"type": "video", "mime_type": "video/mp4", "data": "VID"},
        {"type": "document", "mime_type": "application/pdf", "data": b64_pdf},
    ]

    # pdf_mode="full" leaves document untouched while still setting video resolution
    full_out = compress_blocks_for_router(payload, image_resolution="low", pdf_mode="full")
    assert isinstance(full_out, list)
    assert full_out[0]["resolution"] == "low"
    assert full_out[1]["type"] == "document"
    assert full_out[1]["data"] == b64_pdf

    # pdf_mode="page1_image" renders page 1 to a PNG image block + metadata text block
    img_out = compress_blocks_for_router(payload, image_resolution="low", pdf_mode="page1_image")
    assert isinstance(img_out, list)
    assert img_out[1]["type"] == "text"
    assert "total_pages=2" in img_out[1]["text"]
    assert img_out[2]["type"] == "image"
    assert img_out[2]["mime_type"] == "image/png"
    assert img_out[2]["resolution"] == "low"

    with pytest.raises(ValueError, match="Unknown pdf_mode"):
        compress_blocks_for_router(payload, pdf_mode="bogus")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def test_payload_has_compressible_media_edge_cases() -> None:
    assert payload_has_compressible_media("just text") is False
    assert payload_has_compressible_media([{"type": "audio", "data": "AA"}]) is False
    assert (
        payload_has_compressible_media(
            [{"type": "image", "resolution": "low", "data": "AA"}],
            image_resolution="low",
        )
        is False
    )
    assert (
        payload_has_compressible_media(
            [{"type": "video", "resolution": "high", "data": "AA"}],
            image_resolution="low",
        )
        is True
    )
    assert (
        payload_has_compressible_media(
            [{"type": "document", "mime_type": "application/pdf", "data": "AA"}],
            pdf_mode="page1_pdf",
        )
        is True
    )
    assert (
        payload_has_compressible_media(
            [{"type": "document", "mime_type": "application/pdf", "data": "AA"}],
            pdf_mode="full",
        )
        is False
    )


def test_estimate_input_tokens_across_modalities_and_corrupt_pdf() -> None:
    assert estimate_input_tokens("") == 1
    assert estimate_input_tokens("a" * 40) == 10

    pdf_2p = base64.b64encode(_create_sample_pdf_bytes(2)).decode("ascii")
    corrupt_pdf = base64.b64encode(b"not-a-pdf").decode("ascii")
    audio_b64 = base64.b64encode(b"\x00" * 64_000).decode("ascii")

    blocks = [
        {"type": "text", "text": "hello world!"},  # 3
        {"type": "image", "resolution": "low"},  # 64
        {"type": "image", "resolution": "ultra_high"},  # 1536
        {"type": "document", "data": pdf_2p},  # 258 * 2 = 516
        {"type": "document", "data": corrupt_pdf},  # 512 fallback
        {"type": "audio", "data": audio_b64},  # 64000 bytes -> 2s * 25 = 50
        {"type": "video"},  # 1024
        {"type": "custom_unknown"},  # 128
    ]
    assert estimate_input_tokens(blocks) == 3 + 64 + 1536 + 516 + 512 + 50 + 1024 + 128


def test_extract_interaction_text_and_modality_tokens() -> None:
    # 1. Direct output_text takes precedence
    assert extract_interaction_text(SimpleNamespace(output_text="  hello  ")) == "hello"

    # 2. Falls back to steps (model_output) + outputs
    from_steps = SimpleNamespace(
        output_text=None,
        steps=[
            SimpleNamespace(
                type="model_output",
                content=[SimpleNamespace(type="text", text="from step")],
            )
        ],
        outputs=[SimpleNamespace(type="text", text="from output")],
    )
    assert extract_interaction_text(from_steps) == "from step\nfrom output"

    # 3. Empty object returns empty string
    assert extract_interaction_text(SimpleNamespace()) == ""

    # 4. Modality tokens extraction + missing modality fallback
    usage = SimpleNamespace(
        input_tokens_by_modality=[
            SimpleNamespace(modality="text", tokens=100),
            SimpleNamespace(modality="image", tokens=64),
            SimpleNamespace(modality=None, tokens=10),
        ]
    )
    assert extract_modality_tokens(usage) == {"text": 100, "image": 64, "unknown": 10}
    assert extract_modality_tokens(SimpleNamespace(input_tokens_by_modality=None)) == {}


def test_prepare_router_envelope_uncompressed_list_passthrough() -> None:
    text_blocks = [{"type": "text", "text": "hello world"}]
    out, est_tokens, is_uncompressed = prepare_router_envelope(text_blocks)
    assert out is text_blocks
    assert est_tokens == 2
    assert is_uncompressed is True

    img_blocks = [{"type": "image", "mime_type": "image/png", "data": "AAAA"}]
    out2, est2, is_uncompressed2 = prepare_router_envelope(img_blocks, compress_media=False)
    assert out2 is img_blocks
    assert est2 == 258
    assert is_uncompressed2 is True


def test_estimate_input_tokens_handles_none_fields_safely() -> None:
    assert estimate_input_tokens([{"type": "text", "text": None}]) == 1
    assert estimate_input_tokens([{"type": "audio", "data": None}]) == 128


@pytest.mark.parametrize(
    ("data", "expected_match"),
    [
        ("not!valid!b64", "Invalid base64"),
        (None, "Invalid base64"),
        (base64.b64encode(b"not-a-pdf").decode("ascii"), "Invalid PDF"),
    ],
)
def test_compress_blocks_for_router_invalid_pdf_payload_raises_value_error(
    data: Any,
    expected_match: str,
) -> None:
    with pytest.raises(ValueError, match=expected_match):
        compress_blocks_for_router(
            [{"type": "document", "mime_type": "application/pdf", "data": data}],
            pdf_mode="page1_pdf",
        )


def test_render_pdf_first_page_png_bytes_caps_longest_edge_to_max_pixels() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=20_000, height=10_000)
    huge_buf = io.BytesIO()
    writer.write(huge_buf)

    png_bytes, total_pages = render_pdf_first_page_png_bytes(huge_buf.getvalue(), max_pixels=512)

    assert total_pages == 1
    with Image.open(io.BytesIO(png_bytes)) as img:
        assert max(img.size) <= 512


def test_read_and_encode_file_oversized_attachment_raises_value_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    big_file = tmp_path / "huge.png"
    big_file.write_bytes(b"0" * 100)
    monkeypatch.setenv("GEMINI_ROUTER_MAX_ATTACHMENT_BYTES", "50")
    _read_and_encode_file.cache_clear()
    try:
        with pytest.raises(ValueError, match="exceeds limit"):
            _read_and_encode_file(str(big_file.resolve()))
    finally:
        _read_and_encode_file.cache_clear()


def test_load_media_block_quicktime_mov_returns_video_block(tmp_path: Path) -> None:
    mov_file = tmp_path / "clip.mov"
    mov_file.write_bytes(b"fake-mov")

    mov_block = load_media_block(mov_file, allowed_root=tmp_path)

    assert mov_block["type"] == "video"
    assert mov_block["mime_type"] == "video/quicktime"


@pytest.mark.parametrize(
    "internal_exc",
    [
        IndexError("broken page tree"),
        KeyError("/Pages"),
    ],
)
def test_slice_pdf_first_page_bytes_malformed_xref_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
    internal_exc: Exception,
) -> None:
    def _broken_reader(*_args: object, **_kwargs: object) -> object:
        raise internal_exc

    monkeypatch.setattr(envelope_mod, "PdfReader", _broken_reader)

    with pytest.raises(ValueError, match="Invalid PDF document payload") as exc_info:
        slice_pdf_first_page_bytes(b"%PDF-1.4 broken")

    assert exc_info.value.__cause__ is internal_exc
