"""Plain text, Markdown, code and configuration: decode the bytes, one segment per file.
Web pages, rich text and subtitles are turned into their readable text first
(`extract.markup`, `extract.rtf`)."""

from __future__ import annotations

from lanternkb.extract.base import (
    API_VERSION,
    ExtractError,
    ExtractResult,
    Limits,
    Segment,
    decode_text,
    utf16_without_bom,
)

BINARY_SNIFF = 8192
MARKUP = frozenset({".html", ".htm", ".xhtml"})
SUBTITLES = frozenset({".srt", ".vtt"})


class TextExtractor:
    api_version = API_VERSION

    def __init__(self, feature_id: str):
        self.id = feature_id

    def extract(self, data: bytes, path: str, limits: Limits) -> ExtractResult:
        suffix = path[path.rfind(".") :].lower() if "." in path.rsplit("/", 1)[-1] else ""
        if suffix == ".rtf":
            from lanternkb.extract.rtf import rtf_to_text

            return _one(rtf_to_text(data))
        utf16 = data.startswith((b"\xff\xfe", b"\xfe\xff")) or utf16_without_bom(data)
        if not utf16 and b"\x00" in data[:BINARY_SNIFF]:
            raise ExtractError("not-text")
        text = decode_text(data)
        if self.id == "extractor.text" and suffix in MARKUP:
            from lanternkb.extract.markup import html_to_text

            return _one(html_to_text(text))
        if self.id == "extractor.text" and suffix in SUBTITLES:
            from lanternkb.extract.markup import subtitle_segments

            return ExtractResult(subtitle_segments(text))
        lines = text.count("\n") + 1
        return ExtractResult([Segment(text, f"lines 1-{lines}", 1, lines)])


def _one(text: str) -> ExtractResult:
    lines = text.count("\n") + 1
    return ExtractResult([Segment(text, f"lines 1-{lines}", 1, lines)] if text else [])
