"""Safe ZIP archives for extractors (built-in and plugins): Office files, EPUB, ...

`SafeZip` refuses, as a skipped file with a reason, an archive that could exhaust memory or
time: more than MAX_MEMBERS members, a declared or actual uncompressed total above
TOTAL_FACTOR x the feature's output limit, or a large member compressed more than
BOMB_RATIO:1. Its XML parser refuses any DOCTYPE (entity expansion and external entities)
and UTF-16 input, which could hide one. Every refusal, and every damaged member or
malformed XML, raises `ExtractError`, so an extractor can simply let it propagate.
"""

from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
import zipfile
import zlib
from collections.abc import Iterator
from typing import IO

from lanternkb.extract.base import ExtractError, Limits

MAX_MEMBERS = 10_000
BOMB_RATIO = 100
BOMB_MIN_SIZE = 1024 * 1024
TOTAL_FACTOR = 20  # total uncompressed bytes <= TOTAL_FACTOR * output_mb MB
# What a damaged member raises while it is opened or read.
_BROKEN = (zipfile.BadZipFile, zlib.error, EOFError, OSError, RuntimeError, NotImplementedError,
           ValueError)  # fmt: skip


class ZipStream:
    """A member's byte stream that counts every byte against the archive-wide cap and
    refuses a DOCTYPE, even one split across reads."""

    def __init__(self, raw: IO[bytes], archive: SafeZip, allow_doctype: bool = False):
        self._raw = raw
        self._archive = archive
        self._tail = b""
        self._first = True
        self._allow_doctype = allow_doctype

    def read(self, size: int = -1) -> bytes:
        try:
            chunk = self._raw.read(size if size and size > 0 else 1 << 20)
        except _BROKEN as e:  # a bad checksum, a truncated or encrypted member
            raise ExtractError(self._archive.unreadable) from e
        self._archive.consume(len(chunk))
        if self._allow_doctype:  # read as HTML (html.parser expands no entities)
            return chunk
        if self._first and chunk[:2] in (b"\xff\xfe", b"\xfe\xff") or b"\x00" in chunk[:64]:
            raise ExtractError(self._archive.unreadable)  # UTF-16 could hide a DOCTYPE
        self._first = False
        if b"<!DOCTYPE" in self._tail + chunk:  # the C parser exposes no hook for it
            raise ExtractError("xml-doctype")
        self._tail = chunk[-8:]
        return chunk


class SafeZip:
    """A ZIP archive opened with limits; `unreadable` is the skip reason for a broken one."""

    def __init__(self, data: bytes, limits: Limits, unreadable: str = "archive-unreadable"):
        self.unreadable = unreadable
        if not zipfile.is_zipfile(io.BytesIO(data)):
            raise ExtractError(unreadable)
        try:
            self.zip = zipfile.ZipFile(io.BytesIO(data))
            infos = self.zip.infolist()
        except (zipfile.BadZipFile, OSError, ValueError) as exc:
            raise ExtractError(unreadable) from exc
        self.cap = TOTAL_FACTOR * limits.output_mb * 1024 * 1024
        self.read_total = 0
        if len(infos) > MAX_MEMBERS:
            raise ExtractError("archive-bomb")
        declared = 0
        for info in infos:
            declared += info.file_size
            ratio = info.file_size / max(info.compress_size, 1)
            if info.file_size > BOMB_MIN_SIZE and ratio > BOMB_RATIO:
                raise ExtractError("archive-bomb")
        if declared > self.cap:
            raise ExtractError("archive-bomb")
        self.names = {info.filename for info in infos}

    def consume(self, n: int) -> None:
        self.read_total += n
        if self.read_total > self.cap:
            raise ExtractError("archive-bomb")

    def has(self, name: str) -> bool:
        return name in self.names

    def names_matching(self, pattern: str) -> list[str]:
        rx = re.compile(pattern)
        return sorted(n for n in self.names if rx.fullmatch(n))

    def stream(self, name: str, allow_doctype: bool = False) -> ZipStream:
        try:
            return ZipStream(self.zip.open(name), self, allow_doctype)
        except (KeyError, *_BROKEN) as e:
            raise ExtractError(self.unreadable) from e

    def read(self, name: str, allow_doctype: bool = False) -> bytes:
        """A whole member, through the same size checks. allow_doctype=True only for content
        parsed as HTML (html.parser), never for XML parsed with ElementTree."""
        out, s = [], self.stream(name, allow_doctype)
        while chunk := s.read(1 << 20):
            out.append(chunk)
        return b"".join(out)

    def xml(self, name: str) -> ET.Element:
        try:
            return ET.parse(self.stream(name), ET.XMLParser()).getroot()
        except ET.ParseError as e:
            raise ExtractError(self.unreadable) from e

    def iterparse(self, name: str, events: tuple[str, ...]) -> Iterator[tuple[str, ET.Element]]:
        try:
            yield from ET.iterparse(self.stream(name), events=events, parser=ET.XMLParser())
        except ET.ParseError as e:
            raise ExtractError(self.unreadable) from e
