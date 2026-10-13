"""Structure-aware chunking (moved from the phase 1 baselines unchanged; phase 5 tunes it).

Markdown: split at headings, then pack paragraphs up to MAX_CHARS. Python: one chunk per
top-level function or class (plus the module preamble), with long ones split by lines.
Plain text: pack paragraphs. Every chunk records its file and line range.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, replace
from pathlib import Path

from lanternkb.text import split_identifier

# Chosen on dev in phase 5; phase 1 used 1,500 and 60.
MAX_CHARS = 800  # about 150-250 tokens
CODE_MAX_LINES = 32
# A line longer than MAX_CHARS (a paragraph exported as one line, minified code) is split
# at spaces, so no chunk outgrows what the embedder reads. Off only to reproduce phase 1.
SPLIT_LONG_LINES = True


def _split_line(line: str) -> list[str]:
    """Pieces of at most MAX_CHARS, broken at the last space before the limit if any."""
    if not SPLIT_LONG_LINES or len(line) <= MAX_CHARS:
        return [line]
    out = []
    while len(line) > MAX_CHARS:
        cut = line.rfind(" ", MAX_CHARS // 2, MAX_CHARS + 1)
        cut = cut if cut > 0 else MAX_CHARS
        out.append(line[:cut])
        line = line[cut:].lstrip(" ")
    return [*out, line] if line else out


@dataclass(frozen=True)
class Chunk:
    path: str
    start: int  # 1-based line numbers, inclusive
    end: int
    text: str
    location: str | None = None  # "lines 3-40", "page 3, lines 1-20", "slide 2", ...
    # Holds a piece of a line cut by `_split_line`: its text has a line start or end the
    # scanned text did not, so it is scanned again before it is stored.
    split: bool = False


def _pack(path: str, lines: list[str], first_line: int) -> list[Chunk]:
    """Pack blank-line-separated paragraphs into chunks of at most MAX_CHARS."""
    chunks: list[Chunk] = []
    buf: list[str] = []
    start = first_line
    size = 0
    cut = False  # buf holds a piece of a cut line
    for i, whole in enumerate(lines):
        lineno = first_line + i
        pieces = _split_line(whole)
        for j, line in enumerate(pieces):
            cut_here = j > 0 or not line.strip() or size > MAX_CHARS
            if size + len(line) > MAX_CHARS and buf and cut_here:
                end = lineno if j > 0 else lineno - 1
                chunks.append(Chunk(path, start, end, "\n".join(buf).strip(), split=cut))
                buf, size, start, cut = [], 0, lineno, False
            buf.append(line)
            size += len(line) + 1
            cut = cut or len(pieces) > 1
    if "".join(buf).strip():
        chunks.append(Chunk(path, start, first_line + len(lines) - 1, "\n".join(buf).strip(),
                            split=cut))  # fmt: skip
    return chunks


def chunk_markdown(path: str, text: str) -> list[Chunk]:
    lines = text.splitlines()
    sections: list[tuple[int, int]] = []
    start = 0
    for i, line in enumerate(lines):
        if re.match(r"#{1,6}\s", line) and i > start:
            sections.append((start, i))
            start = i
    sections.append((start, len(lines)))
    out: list[Chunk] = []
    for a, b in sections:
        out += _pack(path, lines[a:b], a + 1)
    return out


def chunk_python(path: str, text: str) -> list[Chunk]:
    lines = text.splitlines()
    try:
        tree = ast.parse(text)
    except Exception:  # SyntaxError, but also MemoryError/RecursionError on hostile input
        return _pack(path, lines, 1)
    spans: list[tuple[int, int]] = []
    prev_end = 0
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            first = min([node.lineno, *(d.lineno for d in node.decorator_list)])
            if first - 1 > prev_end:
                spans.append((prev_end + 1, first - 1))
            spans.append((first, node.end_lineno or first))
            prev_end = node.end_lineno or first
    if prev_end < len(lines):
        spans.append((prev_end + 1, len(lines)))
    out: list[Chunk] = []
    for a, b in spans:
        for s in range(a, b + 1, CODE_MAX_LINES):
            e = min(b, s + CODE_MAX_LINES - 1)
            body = "\n".join(lines[s - 1 : e]).strip()
            if body and SPLIT_LONG_LINES and len(body) > 2 * MAX_CHARS:
                out += _pack(path, lines[s - 1 : e], s)  # very long lines: pack by size
            elif body:
                out.append(Chunk(path, s, e, body))
    return out


def chunk_text(rel: str, text: str) -> list[Chunk]:
    """Chunk already-extracted (and redacted) text, choosing the chunker by file type."""
    if rel.endswith(".py"):
        return chunk_python(rel, text)
    if rel.endswith(".md"):
        return chunk_markdown(rel, text)
    return _pack(rel, text.splitlines(), 1)


def chunk_file(corpus_dir: Path, rel: str) -> list[Chunk]:
    return chunk_text(rel, (corpus_dir / rel).read_text(encoding="utf-8", errors="replace"))


def corpus_files(corpus_dir: Path) -> list[str]:
    return sorted(
        p.relative_to(corpus_dir).as_posix()
        for p in corpus_dir.rglob("*")
        if p.is_file() and p.suffix in {".md", ".py", ".txt"}
    )


def split_identifiers(text: str) -> str:
    """Words from code identifiers: `parseConfigV2 _merge_url` -> `parse config v2 merge url`."""
    words: list[str] = []
    for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        parts = split_identifier(tok)
        if len(parts) > 1:
            words += parts
    return " ".join(words)


def chunk_segments(rel: str, segments: list[dict[str, object]]) -> list[Chunk]:
    """Chunks from extracted segments, each tagged with where it came from.

    A single whole-file segment (text and code) is chunked by structure, as before. Paged or
    sheeted content (PDF pages, slides, sheet ranges, audio spans) is packed per segment.
    """
    if len(segments) == 1 and str(segments[0]["location"]).startswith("lines "):
        return [replace(c, location=f"lines {c.start}-{c.end}")
                for c in chunk_text(rel, str(segments[0]["text"]))]  # fmt: skip
    out: list[Chunk] = []
    for seg in segments:
        loc = str(seg["location"])
        packed = _pack(rel, str(seg["text"]).splitlines(), 1)
        for c in packed:
            where = loc if len(packed) == 1 else f"{loc}, lines {c.start}-{c.end}"
            out.append(replace(c, location=where))
    return out
