"""lanternkb-epub: an example plugin that indexes EPUB books (feature `extractor.epub`).

It is the worked example in docs/plugins.md. An EPUB is a ZIP of XHTML chapters:
META-INF/container.xml points to the package document (OPF), whose spine lists the
chapters in reading order. Each chapter becomes one segment, located as "chapter N".

The extractor runs inside lanternkb's sandboxed extraction workers: it is given the file's
bytes, never a path to open, and has no network and nowhere to write.
"""

from __future__ import annotations

import posixpath
from collections.abc import Mapping
from html.parser import HTMLParser
from typing import Any
from urllib.parse import unquote

from lanternkb.extract.archive import SafeZip
from lanternkb.extract.base import (
    API_VERSION,
    ExtractError,
    ExtractResult,
    Limits,
    Segment,
    decode_text,
)

OPF = "{http://www.idpf.org/2007/opf}"
CONTAINER = "{urn:oasis:names:tc:opendocument:xmlns:container}"
BLOCKS = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section",
          "blockquote", "pre", "dt", "dd"}  # fmt: skip


def create(options: Mapping[str, Any]) -> EpubExtractor:
    """The plugin's factory: lanternkb calls it with this feature's own options."""
    return EpubExtractor(include_front_matter=bool(options.get("include_front_matter", True)))


class _Text(HTMLParser):
    """Visible text of an XHTML chapter, one line per block; scripts and styles dropped."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = []
        self._line: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in BLOCKS:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        elif tag in BLOCKS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._line.append(data)

    def _flush(self) -> None:
        line = " ".join("".join(self._line).split())
        if line:
            self.lines.append(line)
        self._line = []

    def text(self) -> str:
        self._flush()
        return "\n".join(self.lines)


class EpubExtractor:
    api_version = API_VERSION
    id = "extractor.epub"

    def __init__(self, include_front_matter: bool = True):
        self.include_front_matter = include_front_matter

    def extract(self, data: bytes, path: str, limits: Limits) -> ExtractResult:
        book = SafeZip(data, limits, unreadable="epub-unreadable")
        if not book.has("META-INF/container.xml"):
            raise ExtractError("epub-unreadable")
        if book.has("META-INF/encryption.xml"):
            raise ExtractError("epub-encrypted")  # DRM: the chapters are not readable text
        root = book.xml("META-INF/container.xml").find(f".//{CONTAINER}rootfile")
        opf_path = root.get("full-path", "") if root is not None else ""
        if not book.has(opf_path):
            raise ExtractError("epub-unreadable")
        opf = book.xml(opf_path)
        base = posixpath.dirname(opf_path)
        items = {i.get("id"): i for i in opf.iter(f"{OPF}item")}
        spine = [r.get("idref") for r in opf.iter(f"{OPF}itemref")
                 if self.include_front_matter or r.get("linear", "yes") != "no"]  # fmt: skip
        spine = list(dict.fromkeys(spine))  # a chapter listed twice is read once
        segments, budget = [], limits.max_chars
        for n, idref in enumerate(spine, start=1):
            item = items.get(idref)
            if item is None or "html" not in (item.get("media-type") or ""):
                continue
            name = posixpath.normpath(posixpath.join(base, unquote(item.get("href", ""))))
            if not book.has(name):
                continue
            parser = _Text()
            parser.feed(decode_text(book.read(name, allow_doctype=True)))
            text = parser.text()
            if not text:
                continue
            budget -= len(text)
            if budget < 0:
                note = f"text limit reached after {len(segments)} chapters; the rest skipped"
                if not segments:
                    return ExtractResult(skip="too-large-text")
                return ExtractResult(segments=segments, note=note)
            segments.append(Segment(text, f"chapter {n}", 1, text.count("\n") + 1))
        if not segments:
            return ExtractResult(skip="epub-no-text")
        return ExtractResult(segments=segments)
