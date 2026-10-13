"""PDF: one segment per page, from the text layer (pypdfium2, BSD/Apache).

Pages without a text layer are counted and reported in the note; OCR is a separate feature
that uses `render_page_png`.
"""

from __future__ import annotations

import struct
import zlib

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from lanternkb.extract.base import API_VERSION, ExtractError, ExtractResult, Limits, Segment

FPDF_ERR_PASSWORD = getattr(pdfium_c, "FPDF_ERR_PASSWORD", 4)


def _open(data: bytes) -> pdfium.PdfDocument:
    try:
        return pdfium.PdfDocument(data)
    except pdfium.PdfiumError as exc:
        if getattr(exc, "err_code", None) == FPDF_ERR_PASSWORD:
            raise ExtractError("pdf-encrypted") from exc
        raise ExtractError("pdf-unreadable") from exc
    except Exception as exc:  # malformed input must never crash the worker
        raise ExtractError("pdf-unreadable") from exc


class PdfExtractor:
    api_version = API_VERSION
    id = "extractor.pdf"

    def extract(self, data: bytes, path: str, limits: Limits) -> ExtractResult:
        try:
            pdf = _open(data)
        except ExtractError as exc:
            if exc.reason == "pdf-encrypted":
                return ExtractResult(skip="pdf-encrypted")
            raise
        try:
            return self._pages(pdf, limits)
        except ExtractError:
            raise
        except pdfium.PdfiumError as exc:
            raise ExtractError("pdf-unreadable") from exc
        finally:
            pdf.close()

    def _pages(self, pdf: pdfium.PdfDocument, limits: Limits) -> ExtractResult:
        total = len(pdf)
        budget = limits.output_mb * 1024 * 1024
        used = 0
        segments: list[Segment] = []
        empty = 0
        stopped_at: int | None = None
        for index in range(total):
            page = pdf[index]
            try:
                textpage = page.get_textpage()
                try:
                    text = textpage.get_text_range()
                finally:
                    textpage.close()
            finally:
                page.close()
            text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
            if not text:
                empty += 1
                continue
            size = len(text.encode("utf-8"))
            if used + size > budget:
                stopped_at = index
                break
            used += size
            lines = text.count("\n") + 1
            segments.append(Segment(text, f"page {index + 1}", 1, lines))
        if stopped_at is not None:
            note = (
                f"output limit of {limits.output_mb} MB reached at page {stopped_at + 1} "
                f"of {total}; the rest was not indexed"
            )
            return ExtractResult(segments, note=note)
        if total == 0 or empty == total:
            return ExtractResult(skip="pdf-no-text-layer")
        note = f"{empty} of {total} pages had no text layer (scanned?)" if empty else None
        return ExtractResult(segments, note=note)


def _png(width: int, height: int, rgb: bytes) -> bytes:
    def chunk(kind: bytes, body: bytes) -> bytes:
        crc = zlib.crc32(kind + body) & 0xFFFFFFFF
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)

    stride = width * 3
    raw = b"".join(b"\x00" + rgb[y * stride : (y + 1) * stride] for y in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def render_page_png(data: bytes, page_index: int, scale: float = 2.0) -> bytes:
    """Render one page (0-based) to PNG bytes, for the OCR extractor. No Pillow needed."""
    pdf = _open(data)
    try:
        if not 0 <= page_index < len(pdf):
            raise ExtractError("pdf-unreadable")
        page = pdf[page_index]
        try:
            bitmap = page.render(scale=scale, rev_byteorder=True, maybe_alpha=False)
            try:
                width, height = bitmap.width, bitmap.height
                buf = bytes(bitmap.buffer)
                step, stride = bitmap.n_channels, bitmap.stride
                rows = []
                for y in range(height):
                    row = buf[y * stride : y * stride + width * step]
                    if step != 3:
                        row = b"".join(row[i : i + 3] for i in range(0, len(row), step))
                    rows.append(row)
                rgb = b"".join(rows)
                return _png(width, height, rgb)
            finally:
                bitmap.close()
        finally:
            page.close()
    except pdfium.PdfiumError as exc:
        raise ExtractError("pdf-unreadable") from exc
    finally:
        pdf.close()
