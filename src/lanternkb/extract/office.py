"""Office Open XML (.docx, .pptx, .xlsx) with the standard library only.

The zip is the threat: caps on member count, uncompressed size and compression ratio are
checked before parsing, every read is counted, `extractall` is never used, and any DOCTYPE is
refused (OOXML never needs one, and refusing it blocks entity-expansion tricks).
"""

from __future__ import annotations

import posixpath
import xml.etree.ElementTree as ET
import zipfile
import zlib

from lanternkb.extract.archive import (  # noqa: F401 - limits re-exported for callers
    BOMB_MIN_SIZE,
    BOMB_RATIO,
    MAX_MEMBERS,
    TOTAL_FACTOR,
    SafeZip,
)
from lanternkb.extract.base import API_VERSION, ExtractError, ExtractResult, Limits, Segment

XLSX_BLOCK_ROWS = 200

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"


class _Budget:
    """Counts output bytes; `add` returns False once the cap is passed."""

    def __init__(self, limit_bytes: int):
        self.limit = limit_bytes
        self.used = 0
        self.exceeded = False

    def add(self, text: str) -> bool:
        size = len(text.encode("utf-8"))
        if self.used + size > self.limit:
            self.exceeded = True
            return False
        self.used += size
        return True


def _Package(data: bytes, limits: Limits) -> SafeZip:  # noqa: N802 - kept for readability
    return SafeZip(data, limits, unreadable="office-unreadable")


def _relationships(pkg: _Package, rels_name: str, base_dir: str) -> dict[str, tuple[str, str]]:
    """Relationship id -> (type, normalised target path)."""
    if not pkg.has(rels_name):
        return {}
    out: dict[str, tuple[str, str]] = {}
    for rel in pkg.xml(rels_name).iter(f"{REL}Relationship"):
        if rel.get("TargetMode") == "External":
            continue
        target = rel.get("Target", "")
        if target.startswith("/"):
            path = posixpath.normpath(target.lstrip("/"))
        else:
            path = posixpath.normpath(posixpath.join(base_dir, target))
        out[rel.get("Id", "")] = (rel.get("Type", ""), path)
    return out


def _rels_name(part: str) -> str:
    return posixpath.join(posixpath.dirname(part), "_rels", posixpath.basename(part) + ".rels")


def _segment(text: str, location: str) -> Segment:
    return Segment(text, location, 1, text.count("\n") + 1)


# --------------------------------------------------------------------------- docx


def _w_paragraph(p: ET.Element) -> str:
    parts: list[str] = []
    for el in p.iter():
        if el.tag == f"{W}t" and el.text:
            parts.append(el.text)
        elif el.tag == f"{W}tab":
            parts.append("\t")
        elif el.tag in (f"{W}br", f"{W}cr"):
            parts.append("\n")
    return "".join(parts)


def _w_blocks(parent: ET.Element) -> list[str]:
    lines: list[str] = []
    for child in parent:
        if child.tag == f"{W}p":
            text = _w_paragraph(child).strip()
            if text:
                lines.append(text)
        elif child.tag == f"{W}tbl":
            for row in child.findall(f"{W}tr"):
                cells = [" ".join(_w_blocks(cell)).strip() for cell in row if cell.tag == f"{W}tc"]
                if any(cells):
                    lines.append("\t".join(cells))
        elif child.tag in (f"{W}sdt", f"{W}ins"):
            inner = child.find(f"{W}sdtContent") if child.tag == f"{W}sdt" else child
            if inner is not None:
                lines.extend(_w_blocks(inner))
        else:  # containers: w:footnote, w:comment, w:hdr, ...
            lines.extend(_w_blocks(child))
    return lines


def _docx(pkg: _Package, budget: _Budget) -> list[Segment]:
    segments: list[Segment] = []
    body = pkg.xml("word/document.xml").find(f"{W}body")
    main = _w_blocks(body) if body is not None else []
    groups = [
        ("document", ["word/document.xml"]),
        ("header", pkg.names_matching(r"word/header\d*\.xml")),
        ("footer", pkg.names_matching(r"word/footer\d*\.xml")),
        ("footnotes", ["word/footnotes.xml"]),
        ("endnotes", ["word/endnotes.xml"]),
        ("comments", ["word/comments.xml"]),
    ]
    for location, parts in groups:
        if location == "document":
            lines = main
        else:
            lines = []
            seen: set[str] = set()
            for part in parts:
                if not pkg.has(part):
                    continue
                for line in _w_blocks(pkg.xml(part)):
                    if line not in seen:
                        seen.add(line)
                        lines.append(line)
        text = "\n".join(lines)
        if not text:
            continue
        if not budget.add(text):
            break
        segments.append(_segment(text, location))
    return segments


# --------------------------------------------------------------------------- pptx


def _a_paragraphs(root: ET.Element, skip_placeholders: bool = False) -> list[str]:
    lines: list[str] = []
    skipped = {"sldNum", "sldImg", "hdr", "ftr", "dt"}

    def walk(node: ET.Element) -> None:
        for child in node:
            if skip_placeholders and child.tag == f"{P}sp":
                ph = child.find(f"{P}nvSpPr/{P}nvPr/{P}ph")
                if ph is not None and ph.get("type") in skipped:
                    continue
            if child.tag == f"{A}p":
                parts = []
                for el in child.iter():
                    if el.tag == f"{A}t" and el.text:
                        parts.append(el.text)
                    elif el.tag == f"{A}br":
                        parts.append("\n")
                text = "".join(parts).strip()
                if text:
                    lines.append(text)
            else:
                walk(child)

    walk(root)
    return lines


def _pptx(pkg: _Package, budget: _Budget) -> list[Segment]:
    pres = pkg.xml("ppt/presentation.xml")
    rels = _relationships(pkg, "ppt/_rels/presentation.xml.rels", "ppt")
    segments: list[Segment] = []
    ids = [el.get(f"{R}id", "") for el in pres.iter(f"{P}sldId")]
    for n, rid in enumerate(ids, 1):
        if rid not in rels or not pkg.has(rels[rid][1]):
            continue
        slide_part = rels[rid][1]
        lines = _a_paragraphs(pkg.xml(slide_part))
        slide_rels = _relationships(pkg, _rels_name(slide_part), posixpath.dirname(slide_part))
        for rtype, target in slide_rels.values():
            if rtype.endswith("/notesSlide") and pkg.has(target):
                notes = _a_paragraphs(pkg.xml(target), skip_placeholders=True)
                if notes:
                    lines += ["Notes:", *notes]
                break
        if not lines:
            continue
        text = "\n".join(lines)
        if not budget.add(text):
            break
        segments.append(_segment(text, f"slide {n}"))
    return segments


# --------------------------------------------------------------------------- xlsx


def _col_index(ref: str) -> int:
    n = 0
    for ch in ref:
        if not ch.isalpha():
            break
        n = n * 26 + ord(ch.upper()) - 64
    return n


def _col_letters(n: int) -> str:
    out = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        out = chr(65 + rem) + out
    return out


def _shared_strings(pkg: _Package, part: str) -> list[str]:
    if not pkg.has(part):
        return []
    out: list[str] = []
    for _event, el in pkg.iterparse(part, ("end",)):
        if el.tag == f"{S}si":
            pieces = [t.text or "" for t in el.iter(f"{S}t") if not _in_phonetic(el, t)]
            out.append("".join(pieces))
            el.clear()
    return out


def _in_phonetic(si: ET.Element, t: ET.Element) -> bool:
    return any(t in rph.iter(f"{S}t") for rph in si.iter(f"{S}rPh"))


def _cell_value(c: ET.Element, shared: list[str]) -> str:
    kind = c.get("t", "n")
    if kind == "inlineStr":
        return "".join(t.text or "" for t in c.iter(f"{S}t"))
    v = c.find(f"{S}v")
    raw = v.text if v is not None and v.text is not None else ""
    if kind == "s":
        try:
            return shared[int(raw)]
        except (ValueError, IndexError):
            return ""
    if kind == "b":
        return "TRUE" if raw.strip() == "1" else "FALSE"
    return raw


def _sheet_blocks(
    pkg: _Package, part: str, shared: list[str], name: str, hidden: bool, budget: _Budget
) -> tuple[list[Segment], bool]:
    """Segments for one sheet; the flag says whether the output cap was hit."""
    suffix = " (hidden)" if hidden else ""
    segments: list[Segment] = []
    rows: list[tuple[int, list[str]]] = []
    next_row = 1

    def flush() -> bool:
        if not rows:
            return True
        width = max(len(cells) for _, cells in rows)
        text = "\n".join("\t".join(cells).rstrip("\t") for _, cells in rows)
        location = f"{name}!A{rows[0][0]}:{_col_letters(width)}{rows[-1][0]}{suffix}"
        rows.clear()
        if not budget.add(text):
            return False
        segments.append(_segment(text, location))
        return True

    for _event, el in pkg.iterparse(part, ("end",)):
        if el.tag != f"{S}row":
            continue
        row_no = int(el.get("r") or next_row)
        next_row = row_no + 1
        cells: dict[int, str] = {}
        auto_col = 1
        for c in el.findall(f"{S}c"):
            col = _col_index(c.get("r") or "") or auto_col
            auto_col = col + 1
            value = _cell_value(c, shared)
            if value != "":
                cells[col] = value
        el.clear()
        if not cells:
            continue
        rows.append((row_no, [cells.get(i, "") for i in range(1, max(cells) + 1)]))
        if len(rows) >= XLSX_BLOCK_ROWS and not flush():
            return segments, True
    return segments, not flush()


def _xlsx(pkg: _Package, budget: _Budget) -> tuple[list[Segment], bool]:
    wb = pkg.xml("xl/workbook.xml")
    rels = _relationships(pkg, "xl/_rels/workbook.xml.rels", "xl")
    shared_part = next(
        (t for ty, t in rels.values() if ty.endswith("/sharedStrings")), "xl/sharedStrings.xml"
    )
    shared = _shared_strings(pkg, shared_part)
    segments: list[Segment] = []
    for sheet in wb.iter(f"{S}sheet"):
        rid = sheet.get(f"{R}id", "")
        if rid not in rels or not pkg.has(rels[rid][1]):
            continue
        hidden = sheet.get("state", "visible") != "visible"
        got, capped = _sheet_blocks(
            pkg, rels[rid][1], shared, sheet.get("name", "Sheet"), hidden, budget
        )
        segments.extend(got)
        if capped:
            return segments, True
    return segments, False


# --------------------------------------------------------------------------- entry


class OfficeExtractor:
    api_version = API_VERSION
    id = "extractor.office"

    def extract(self, data: bytes, path: str, limits: Limits) -> ExtractResult:
        try:
            return self._extract(data, limits)
        except ExtractError as e:
            if e.reason in ("archive-bomb", "xml-doctype"):  # a refusal, not a broken file
                return ExtractResult(skip=e.reason)
            raise
        except (
            zipfile.BadZipFile,
            zlib.error,
            EOFError,
            ET.ParseError,
            KeyError,
            NotImplementedError,
            RuntimeError,
            ValueError,
            OSError,
        ) as exc:
            raise ExtractError("office-unreadable") from exc

    def _extract(self, data: bytes, limits: Limits) -> ExtractResult:
        pkg = _Package(data, limits)
        if not pkg.has("[Content_Types].xml"):
            raise ExtractError("office-unknown-format")
        budget = _Budget(limits.output_mb * 1024 * 1024)
        capped = False
        if pkg.has("word/document.xml"):
            segments = _docx(pkg, budget)
        elif pkg.has("ppt/presentation.xml"):
            segments = _pptx(pkg, budget)
        elif pkg.has("xl/workbook.xml"):
            segments, capped = _xlsx(pkg, budget)
        else:
            raise ExtractError("office-unknown-format")
        note = None
        if budget.exceeded or capped:
            note = f"output limit of {limits.output_mb} MB reached; the rest was not indexed"
        return ExtractResult(segments, note=note)
