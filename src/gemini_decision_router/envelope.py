"""Multimodal input envelope helpers: safe path resolution, image low-res cloning, and PDF page-1 slicing."""

from __future__ import annotations

import base64
import binascii
import functools
import io
import mimetypes
import os
from pathlib import Path
from typing import Any, Literal

import pypdfium2 as pdfium
from pypdf import PdfReader, PdfWriter
from pypdf.errors import PdfReadError, PdfStreamError

DEFAULT_MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024  # Gemini inline media limit (20 MiB)
DEFAULT_MAX_PDF_RENDER_PIXELS = 4096

ImageResolution = Literal["low", "medium", "high", "ultra_high"]
PdfRouterMode = Literal["full", "page1_pdf", "page1_image", "page1_image_low"]

SUPPORTED_IMAGE_MIMES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/heic",
        "image/heif",
        "image/gif",
        "image/bmp",
        "image/tiff",
    }
)

SUPPORTED_AUDIO_MIMES = frozenset(
    {
        "audio/wav",
        "audio/mp3",
        "audio/mpeg",
        "audio/aiff",
        "audio/aac",
        "audio/ogg",
        "audio/flac",
        "audio/m4a",
        "audio/opus",
        "audio/webm",
        "audio/l16",
        "audio/alaw",
        "audio/mulaw",
    }
)

SUPPORTED_VIDEO_MIMES = frozenset(
    {
        "video/mp4",
        "video/mpeg",
        "video/mov",
        "video/quicktime",
        "video/avi",
        "video/webm",
        "video/wmv",
        "video/x-flv",
        "video/3gpp",
    }
)


def resolve_safe_attachment_path(file_path: str | Path, allowed_root: str | Path) -> Path:
    """Canonicalize `file_path` and verify it resides strictly inside `allowed_root`."""
    root = Path(allowed_root).resolve()
    raw = Path(file_path)
    candidate = (root / raw).resolve() if not raw.is_absolute() else raw.resolve()
    if not candidate.is_relative_to(root):
        raise PermissionError(
            f"Attachment path {str(file_path)!r} resolves outside allowed directory {str(root)!r}"
        )
    if not candidate.is_file():
        raise FileNotFoundError(f"Attachment file not found: {candidate}")
    return candidate


def _decode_b64_bytes(encoded: str) -> bytes:
    """Decode base64 media payload, raising `ValueError` on malformed input."""
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError, TypeError) as exc:
        raise ValueError("Invalid base64 media payload.") from exc


@functools.lru_cache(maxsize=128)
def _read_and_encode_file(
    resolved_path: str,
    max_bytes: int = DEFAULT_MAX_ATTACHMENT_BYTES,
) -> tuple[str, str]:
    """Read file and return `(mime_type, base64_encoded_str)`, cached in memory."""
    p = Path(resolved_path)
    effective_max = int(os.environ.get("GEMINI_ROUTER_MAX_ATTACHMENT_BYTES", max_bytes))
    size_bytes = p.stat().st_size
    if size_bytes > effective_max:
        raise ValueError(
            f"Attachment {p.name} ({size_bytes} bytes) exceeds limit of {effective_max} bytes."
        )
    mime_type, _ = mimetypes.guess_type(p.name)
    suf = p.suffix.lower()
    if suf == ".wav":
        mime_type = "audio/wav"
    elif suf == ".mp3":
        mime_type = "audio/mp3"
    elif suf == ".pdf":
        mime_type = "application/pdf"
    elif suf in (".jpg", ".jpeg"):
        mime_type = "image/jpeg"
    elif suf == ".png":
        mime_type = "image/png"
    elif suf == ".webp":
        mime_type = "image/webp"

    if not mime_type or mime_type == "application/octet-stream":
        raise ValueError(f"Unsupported or generic MIME type for {p.name}")

    encoded = base64.b64encode(p.read_bytes()).decode("ascii")
    return mime_type, encoded


def load_media_block(
    file_path: str | Path,
    allowed_root: str | Path,
    *,
    resolution: ImageResolution | None = None,
) -> dict[str, Any]:
    """Load a local image, audio, video, or PDF file into an Interactions/Decisions API Content block."""
    safe_path = resolve_safe_attachment_path(file_path, allowed_root)
    mime_type, encoded = _read_and_encode_file(str(safe_path))

    if mime_type in SUPPORTED_IMAGE_MIMES:
        block: dict[str, Any] = {"type": "image", "data": encoded, "mime_type": mime_type}
        if resolution:
            block["resolution"] = resolution
        return block
    if mime_type in SUPPORTED_VIDEO_MIMES:
        block = {"type": "video", "data": encoded, "mime_type": mime_type}
        if resolution:
            block["resolution"] = resolution
        return block
    if mime_type in SUPPORTED_AUDIO_MIMES:
        return {"type": "audio", "data": encoded, "mime_type": mime_type}
    if mime_type == "application/pdf":
        return {"type": "document", "data": encoded, "mime_type": "application/pdf"}

    raise ValueError(f"Unsupported media MIME type: {mime_type!r}")


def slice_pdf_first_page_bytes(pdf_bytes: bytes) -> tuple[bytes, int]:
    """Extract only page 1 of a PDF document and return `(page1_pdf_bytes, total_page_count)`."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes), strict=False)
        total_pages = len(reader.pages)
        if total_pages <= 1:
            return pdf_bytes, total_pages
        writer = PdfWriter()
        writer.add_page(reader.pages[0])
        buf = io.BytesIO()
        writer.write(buf)
        return buf.getvalue(), total_pages
    except (PdfReadError, PdfStreamError, ValueError, RuntimeError, IndexError, KeyError) as exc:
        raise ValueError("Invalid PDF document payload.") from exc


def render_pdf_first_page_png_bytes(
    pdf_bytes: bytes,
    dpi: int = 150,
    *,
    max_pixels: int = DEFAULT_MAX_PDF_RENDER_PIXELS,
) -> tuple[bytes, int]:
    """Render page 1 of a PDF to PNG bytes, capping longest edge at `max_pixels`."""
    try:
        doc = pdfium.PdfDocument(pdf_bytes)
        try:
            total_pages = len(doc)
            if total_pages == 0:
                raise ValueError("Cannot render page 1 of an empty PDF document.")
            page = doc[0]
            try:
                width_pt, height_pt = page.get_size()
                longest_pt = max(float(width_pt), float(height_pt), 1.0)
                scale = min(dpi / 72.0, max_pixels / longest_pt)
                bitmap = page.render(scale=scale)
                try:
                    pil_image = bitmap.to_pil()
                finally:
                    bitmap.close()
            finally:
                page.close()
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            return buf.getvalue(), total_pages
        finally:
            doc.close()
    except pdfium.PdfiumError as exc:
        raise ValueError("Invalid PDF document payload.") from exc


def payload_has_compressible_media(
    input_payload: str | list[dict[str, Any]],
    *,
    image_resolution: ImageResolution | None = "low",
    pdf_mode: PdfRouterMode = "page1_pdf",
) -> bool:
    """Return True if `compress_blocks_for_router` will downsample or slice any block in `input_payload`."""
    if isinstance(input_payload, str):
        return False
    for raw_block in input_payload:
        b_type = raw_block.get("type")
        if b_type in ("image", "video") and image_resolution is not None:
            if raw_block.get("resolution") != image_resolution:
                return True
        elif (
            b_type == "document"
            and raw_block.get("mime_type") == "application/pdf"
            and "data" in raw_block
            and pdf_mode != "full"
        ):
            return True
    return False


_IMAGE_RESOLUTION_TOKENS: dict[str, int] = {
    "low": 64,
    "medium": 258,
    "high": 768,
    "ultra_high": 1536,
}


def _pdf_page_count(pdf_bytes: bytes) -> int:
    reader = PdfReader(io.BytesIO(pdf_bytes), strict=False)
    return len(reader.pages)


def _pdf_metadata_block(total_pages: int, file_bytes: int) -> dict[str, Any]:
    return {
        "type": "text",
        "text": f"[PDF Metadata: total_pages={total_pages}, file_bytes={file_bytes}]\n",
    }


def _estimate_block_tokens(block: dict[str, Any]) -> int:
    """Estimate prefill tokens for a single Interactions/Decisions API content block."""
    b_type = block.get("type")
    if b_type == "text":
        return max(1, len(block.get("text") or "") // 4)
    if b_type == "image":
        res = block.get("resolution", "medium")
        return _IMAGE_RESOLUTION_TOKENS.get(res, 258)
    if b_type == "document" and "data" in block:
        try:
            pdf_bytes = _decode_b64_bytes(block["data"])
            return 258 * _pdf_page_count(pdf_bytes)
        except Exception:
            return 512
    if b_type == "audio" and isinstance(block.get("data"), str) and block["data"]:
        # ~25 tokens/sec; rough estimate from byte length (~32KB/sec for 16kHz 16-bit mono WAV)
        raw_bytes = len(block["data"]) * 3 // 4
        seconds = max(1.0, raw_bytes / 32000.0)
        return int(seconds * 25)
    if b_type == "video":
        return 1024
    return 128


def _compress_pdf_block(
    block: dict[str, Any],
    *,
    pdf_mode: PdfRouterMode,
    image_resolution: ImageResolution | None,
) -> tuple[list[dict[str, Any]], int]:
    """Compress a PDF `document` block according to `pdf_mode`; returns `(blocks, total_pages)`."""
    if pdf_mode not in ("page1_pdf", "page1_image", "page1_image_low"):
        raise ValueError(f"Unknown pdf_mode: {pdf_mode!r}")

    pdf_bytes = _decode_b64_bytes(block["data"])
    if pdf_mode == "page1_pdf":
        page1_bytes, total_pages = slice_pdf_first_page_bytes(pdf_bytes)
        compressed_doc = dict(block)
        compressed_doc["data"] = base64.b64encode(page1_bytes).decode("ascii")
        return [_pdf_metadata_block(total_pages, len(pdf_bytes)), compressed_doc], total_pages

    png_bytes, total_pages = render_pdf_first_page_png_bytes(pdf_bytes)
    img_block = {
        "type": "image",
        "mime_type": "image/png",
        "data": base64.b64encode(png_bytes).decode("ascii"),
        "resolution": image_resolution or "low",
    }
    return [_pdf_metadata_block(total_pages, len(pdf_bytes)), img_block], total_pages


def compress_blocks_for_router(
    input_payload: str | list[dict[str, Any]],
    *,
    image_resolution: ImageResolution | None = "low",
    pdf_mode: PdfRouterMode = "page1_pdf",
) -> str | list[dict[str, Any]]:
    """Return a non-mutating copy of `input_payload` compressed for the router pass.

    - `image_resolution`: Sets `resolution` on `image` and `video` blocks (`"low"` by default).
      Note: `document` and `audio` blocks do NOT accept `resolution` in the Decisions API.
    - `pdf_mode`:
      - `"full"`: Leave PDF `document` blocks untouched.
      - `"page1_pdf"`: Slice multi-page PDFs to page 1 + prepend a text block with `[PDF Metadata: pages=N, bytes=B]`.
      - `"page1_image"` (alias `"page1_image_low"`): Render page 1 of the PDF to a PNG `image` block at `image_resolution` + metadata text block.
    """
    router_input, _, _ = prepare_router_envelope(
        input_payload,
        compress_media=True,
        image_resolution=image_resolution,
        pdf_mode=pdf_mode,
    )
    return router_input if isinstance(router_input, str) else [dict(b) for b in router_input]


def estimate_input_tokens(input_payload: str | list[dict[str, Any]]) -> int:
    """Fast local prefill token estimator L_i for length-aware compute estimation before calling downstream models."""
    if isinstance(input_payload, str):
        return max(1, len(input_payload) // 4)
    return max(1, sum(_estimate_block_tokens(b) for b in input_payload))


def prepare_router_envelope(
    input_payload: str | list[dict[str, Any]],
    *,
    compress_media: bool = True,
    image_resolution: ImageResolution | None = "low",
    pdf_mode: PdfRouterMode = "page1_pdf",
) -> tuple[str | list[dict[str, Any]], int, bool]:
    """Single-pass router payload preparation: compress media and estimate input tokens together.

    Returns `(router_input, estimated_input_tokens, is_uncompressed)` without decoding or parsing
    PDF payloads twice.
    """
    if isinstance(input_payload, str):
        return input_payload, max(1, len(input_payload) // 4), True

    if compress_media and pdf_mode not in ("full", "page1_pdf", "page1_image", "page1_image_low"):
        raise ValueError(f"Unknown pdf_mode: {pdf_mode!r}")

    out_blocks: list[dict[str, Any]] = []
    est_tokens = 0
    compressed = False

    for raw_block in input_payload:
        b = dict(raw_block)
        b_type = b.get("type")

        if (
            compress_media
            and b_type == "document"
            and b.get("mime_type") == "application/pdf"
            and "data" in b
            and pdf_mode != "full"
        ):
            pdf_blocks, total_pages = _compress_pdf_block(
                b, pdf_mode=pdf_mode, image_resolution=image_resolution
            )
            est_tokens += 258 * total_pages
            out_blocks.extend(pdf_blocks)
            compressed = True
            continue

        est_tokens += _estimate_block_tokens(b)
        if (
            b_type in ("image", "video")
            and compress_media
            and image_resolution is not None
            and b.get("resolution") != image_resolution
        ):
            b["resolution"] = image_resolution
            compressed = True
        out_blocks.append(b)

    if not compressed:
        return input_payload, max(1, est_tokens), True
    return out_blocks, max(1, est_tokens), False


def extract_interaction_text(interaction: Any) -> str:
    """Extract concatenated text from a `google-genai` Interaction object (`output_text`, `steps`, or `outputs`)."""
    direct = getattr(interaction, "output_text", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    parts: list[str] = [
        getattr(block, "text", "") or ""
        for step in getattr(interaction, "steps", []) or []
        if getattr(step, "type", "") == "model_output"
        for block in getattr(step, "content", []) or []
        if getattr(block, "type", "") == "text"
    ]
    parts.extend(
        getattr(out, "text", "") or ""
        for out in getattr(interaction, "outputs", []) or []
        if getattr(out, "type", "") == "text"
    )
    return "\n".join(parts).strip()


def extract_modality_tokens(usage: Any) -> dict[str, int]:
    """Extract `input_tokens_by_modality` from an Interaction/Decision `Usage` object."""
    by_mod: dict[str, int] = {}
    for item in getattr(usage, "input_tokens_by_modality", None) or []:
        mod = str(getattr(item, "modality", "") or "unknown")
        tok = int(getattr(item, "tokens", 0) or 0)
        by_mod[mod] = by_mod.get(mod, 0) + tok
    return by_mod
