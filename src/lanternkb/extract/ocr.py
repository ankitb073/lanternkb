"""Images: read text with macOS Vision, else Tesseract, else skip. Workers whose output may
reach an open client use Tesseract only (`CPU_ONLY_ENV`).

Everything runs on this machine. Vision takes the bytes directly; Tesseract gets them on
stdin, so no temporary file is needed for either.
"""

from __future__ import annotations

import os
import shutil

from lanternkb.extract.base import (
    API_VERSION,
    CPU_ONLY_ENV,
    ExtractError,
    ExtractResult,
    Limits,
    Segment,
)
from lanternkb.extract.tools import run_tool

LOW_CONFIDENCE = 0.5
LINE_TOLERANCE = 0.5  # fraction of a line's height within which two boxes share a line


class _Unavailable(Exception):
    pass


def _order_lines(boxes: list[tuple[str, float, float, float, float]]) -> list[str]:
    """Reading order from (text, x, y_top, height, width) boxes: top to bottom, then left to right.

    y_top grows downward. Boxes whose vertical centres are within half a line height share a line.
    """
    items = sorted(boxes, key=lambda b: (b[2] + b[3] / 2, b[1]))
    rows: list[list[tuple[str, float, float, float, float]]] = []
    for b in items:
        centre = b[2] + b[3] / 2
        if rows:
            last = rows[-1][-1]
            last_centre = last[2] + last[3] / 2
            if abs(centre - last_centre) <= LINE_TOLERANCE * max(b[3], last[3]):
                rows[-1].append(b)
                continue
        rows.append([b])
    return [" ".join(t for t, *_ in sorted(r, key=lambda b: b[1])) for r in rows]


def _vision(data: bytes) -> tuple[str, float]:
    try:
        import Vision
        from Foundation import NSData
    except ImportError as e:
        raise _Unavailable from e
    nsdata = NSData.dataWithBytes_length_(data, len(data))
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(nsdata, None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)
    ok, err = handler.performRequests_error_([request], None)
    if not ok:
        raise ExtractError("image-unreadable")
    boxes = []
    confs = []
    for obs in request.results() or []:
        cands = obs.topCandidates_(1)
        if not cands:
            continue
        text = str(cands[0].string()).strip()
        if not text:
            continue
        bb = obs.boundingBox()  # normalised, origin bottom-left
        x, y, w, h = bb.origin.x, bb.origin.y, bb.size.width, bb.size.height
        boxes.append((text, x, 1.0 - (y + h), h, w))
        confs.append(float(cands[0].confidence()))
    if not boxes:
        return "", 1.0
    return "\n".join(_order_lines(boxes)), sum(confs) / len(confs)


def _tesseract(data: bytes, timeout: int) -> tuple[str, float]:
    exe = shutil.which("tesseract")
    if not exe:
        raise _Unavailable
    proc = run_tool([exe, "stdin", "stdout", "tsv"], timeout, input=data)
    if proc.returncode != 0:
        raise ExtractError("image-unreadable")
    lines: dict[tuple[str, str, str], list[str]] = {}
    confs: list[float] = []
    for row in proc.stdout.decode("utf-8", errors="replace").splitlines()[1:]:
        f = row.split("\t")
        if len(f) < 12 or f[0] != "5" or not f[11].strip():
            continue
        lines.setdefault((f[2], f[3], f[4]), []).append(f[11].strip())
        confs.append(max(float(f[10]), 0.0) / 100.0)
    if not lines:
        return "", 1.0
    text = "\n".join(" ".join(words) for words in lines.values())
    return text, sum(confs) / len(confs)


def ocr_png(data: bytes, timeout: int = 60) -> tuple[str, float]:
    """OCR one image (any format the engine reads). Returns (text, mean confidence 0..1).

    Raises ExtractError("image-unreadable") for corrupt data and ExtractError("ocr-unavailable")
    when no engine is installed.
    """
    if os.environ.get(CPU_ONLY_ENV) != "1":  # Vision needs the shared GPU caches
        try:
            return _vision(data)
        except _Unavailable:
            pass
    try:
        return _tesseract(data, timeout)
    except _Unavailable:
        raise ExtractError("ocr-unavailable") from None


class ImageExtractor:
    api_version = API_VERSION
    id = "extractor.images"

    def extract(self, data: bytes, path: str, limits: Limits) -> ExtractResult:
        try:
            text, conf = ocr_png(data, limits.cpu_seconds)
        except ExtractError as e:
            if e.reason == "ocr-unavailable":
                return ExtractResult(skip="ocr-unavailable")
            raise
        text = text.strip()
        if not text:
            return ExtractResult(skip="no-text-found")
        note = "low OCR confidence" if conf < LOW_CONFIDENCE else None
        cap = limits.output_mb * 1024 * 1024
        raw = text.encode("utf-8")
        if len(raw) > cap:
            text = raw[:cap].decode("utf-8", errors="ignore")
            note = "; ".join(filter(None, [note, "output truncated"]))
        n = text.count("\n") + 1
        return ExtractResult([Segment(text, "image", 1, n)], note=note)
