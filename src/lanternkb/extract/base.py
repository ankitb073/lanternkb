"""The extractor interface.

An extractor turns a file's bytes into text segments with locations. It never opens paths,
writes files or uses the network; it runs inside a sandboxed worker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

API_VERSION = 1
# Set for media workers whose output may reach an open client: they have no GPU caches, so
# images go to Tesseract and audio runs on the CPU (`extract.pool.OPEN_MEDIA_PROFILE`).
CPU_ONLY_ENV = "LANTERNKB_MEDIA_CPU_ONLY"


@dataclass(frozen=True)
class Limits:
    cpu_seconds: int = 30
    memory_mb: int = 512
    output_mb: int = 20
    # The most text, in characters, this file may yield (the stricter of output_mb and the
    # user's max_text_chars). More is refused as `too-large-text`; an extractor that can stop
    # early should return what fits, with a note.
    max_chars: int = 20_000_000


@dataclass(frozen=True)
class Segment:
    text: str
    location: str  # "lines 3-40", "page 3", "slide 2", "Sheet1!A1:F40", "00:01:12"
    start_line: int = 1  # 1-based, within this segment's text
    end_line: int = 1


@dataclass(frozen=True)
class ExtractResult:
    segments: list[Segment] = field(default_factory=list)
    note: str | None = None  # quality note for the user, e.g. "4 of 10 pages had no text layer"
    skip: str | None = None  # a reason to skip the whole file, e.g. "pdf-encrypted"


class ExtractError(Exception):
    """A file this extractor cannot read; `reason` becomes the skip reason."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class Extractor(Protocol):
    api_version: int
    id: str  # the feature id this implements, e.g. "extractor.pdf"

    def extract(self, data: bytes, path: str, limits: Limits) -> ExtractResult: ...


def utf16_without_bom(data: bytes, sniff: int = 8192) -> str | None:
    """ "utf-16-le" or "utf-16-be" for text written as UTF-16 without a byte order mark (some
    Windows tools do), else None: in the first `sniff` bytes, almost every other byte is
    zero, on one side only, and the bytes between are text; and the rest of the file has
    its zero bytes on the same side."""
    head = data[: sniff - sniff % 2]
    if len(head) < 4 or len(data) % 2:
        return None  # UTF-16 has an even number of bytes
    even, odd = head[0::2], head[1::2]
    for enc, zeros, chars in (("utf-16-le", odd, even), ("utf-16-be", even, odd)):
        if (zeros.count(0) >= 0.9 * len(zeros) and chars.count(0) <= 0.02 * len(chars)
                and sum(32 <= c < 127 or c in (9, 10, 13) for c in chars)
                >= 0.9 * len(chars)):  # fmt: skip
            # and the rest of the file the same way, a window at a time: text appended in
            # another encoding is not UTF-16
            side = 1 if enc == "utf-16-le" else 0
            for i in range(len(head), len(data), 512):
                window = data[i : i + 512]
                if window[side::2].count(0) < 0.8 * len(window[side::2]):
                    return None
            return enc
    return None


def decode_text(data: bytes) -> str:
    """Bytes to text: UTF-8 (with or without BOM), UTF-16 with a BOM or without one (by its
    zero bytes, `utf16_without_bom`), else Windows-1252."""
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", errors="replace")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    if (enc := utf16_without_bom(data)) is not None:
        return data.decode(enc, errors="replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")
