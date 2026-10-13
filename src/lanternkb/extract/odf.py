"""OpenDocument (.odt, .ods, .odp: LibreOffice, OpenOffice, Collabora, Pages exports) with
the standard library only, through the same guarded ZIP reader as Office files: caps on
members, sizes and compression ratio, and no DOCTYPE.

- **Text (.odt):** paragraphs, headings, lists and tables in order; headers and footers;
  footnotes and comments after the body. Tracked deletions are left out.
- **Spreadsheets (.ods):** each sheet's rows as tab-separated lines, in blocks of 200 rows
  located like Excel's (`Budget!A1:D200`), so the secret scan's table pass sees each sheet.
- **Presentations (.odp):** each slide's text, then its speaker notes.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
import zlib

from lanternkb.extract.archive import SafeZip
from lanternkb.extract.base import API_VERSION, ExtractError, ExtractResult, Limits, Segment
from lanternkb.extract.office import _Budget, _col_letters, _segment

OFFICE = "{urn:oasis:names:tc:opendocument:xmlns:office:1.0}"
TEXT = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
TABLE = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}"
DRAW = "{urn:oasis:names:tc:opendocument:xmlns:drawing:1.0}"
PRES = "{urn:oasis:names:tc:opendocument:xmlns:presentation:1.0}"
STYLE = "{urn:oasis:names:tc:opendocument:xmlns:style:1.0}"
BLOCK_ROWS = 200
MAX_REPEAT = 100  # a repeated non-empty cell or row is written at most this many times
MAX_COLUMNS = 1024
MIMETYPES = {
    "application/vnd.oasis.opendocument.text": "text",
    "application/vnd.oasis.opendocument.text-template": "text",
    "application/vnd.oasis.opendocument.spreadsheet": "spreadsheet",
    "application/vnd.oasis.opendocument.spreadsheet-template": "spreadsheet",
    "application/vnd.oasis.opendocument.presentation": "presentation",
    "application/vnd.oasis.opendocument.presentation-template": "presentation",
}
_CELLS = (f"{TABLE}table-cell", f"{TABLE}covered-table-cell")
_NESTED = (f"{DRAW}frame", f"{DRAW}text-box", f"{TEXT}p", f"{TEXT}h", f"{TEXT}list")
_ROW_GROUPS = (f"{TABLE}table-header-rows", f"{TABLE}table-row-group", f"{TABLE}table-rows")
_SKIPPED = {f"{TEXT}tracked-changes", f"{TEXT}sequence-decls", f"{TEXT}variable-decls",
            f"{TEXT}user-field-decls", f"{OFFICE}forms", f"{TEXT}note-citation",
            f"{TABLE}shapes", f"{OFFICE}binary-data", f"{TEXT}script"}  # fmt: skip


def _inline(el: ET.Element, notes: list[str] | None = None) -> str:
    """The text of a paragraph or heading: spans, links, spaces, tabs and line breaks.
    Footnotes and comments go to `notes` instead of breaking the sentence."""
    parts = [el.text or ""]
    for child in el:
        tag = child.tag
        if tag == f"{TEXT}s":
            parts.append(" " * _count(child.get(f"{TEXT}c")))
        elif tag == f"{TEXT}tab":
            parts.append("\t")
        elif tag == f"{TEXT}line-break":
            parts.append("\n")
        elif tag in (f"{TEXT}note", f"{OFFICE}annotation"):
            if notes is not None:
                body = child.find(f"{TEXT}note-body") if tag == f"{TEXT}note" else child
                if body is not None:
                    text = " ".join(_paragraphs(body))
                    if text:
                        notes.append(text)
        elif tag in _SKIPPED or tag == f"{OFFICE}annotation-end":
            pass
        elif tag in _NESTED:  # a text box anchored in the paragraph: its own lines
            inner = (
                _paragraphs(child, notes)
                if tag != f"{TEXT}p" and tag != f"{TEXT}h"
                else [_inline(child, notes)]
            )
            parts.append("\n" + "\n".join(inner) + "\n")
        else:
            parts.append(_inline(child, notes))
        parts.append(child.tail or "")
    return "".join(parts)


def _count(value: str | None) -> int:
    """`text:c`: a number of spaces, 1 if missing or not a number, at most 100."""
    try:
        return max(1, min(int(value or 1), 100))
    except ValueError:
        return 1


def _one_line(text: str) -> str:
    return " ".join(text.split("\n")) if "\n" in text else text


def _rows(table: ET.Element) -> list[ET.Element]:
    """A table's own rows, in header, group and body containers, not those of a table
    nested in one of its cells."""
    out: list[ET.Element] = []
    for child in table:
        if child.tag == f"{TABLE}table-row":
            out.append(child)
        elif child.tag in _ROW_GROUPS:
            out.extend(_rows(child))
    return out


def _paragraphs(parent: ET.Element, notes: list[str] | None = None) -> list[str]:
    """Lines of a container in document order: paragraphs, headings, lists, sections,
    frames and tables (a row as tab-separated cells)."""
    lines: list[str] = []
    for child in parent:
        tag = child.tag
        if tag in (f"{TEXT}p", f"{TEXT}h"):
            text = _inline(child, notes).strip()
            if text:
                lines.append(text)
        elif tag == f"{TABLE}table":
            for row in _rows(child):
                cells = [_one_line(" ".join(_paragraphs(c, notes))) for c in row
                         if c.tag in _CELLS]  # fmt: skip
                if any(cells):
                    lines.append("\t".join(cells).rstrip("\t"))
        elif tag in _SKIPPED:
            continue
        else:  # lists, list items, sections, frames, text boxes, index bodies, ...
            lines.extend(_paragraphs(child, notes))
    return lines


def _text_document(pkg: SafeZip, budget: _Budget) -> list[Segment]:
    body = pkg.xml("content.xml").find(f"{OFFICE}body/{OFFICE}text")
    segments: list[Segment] = []
    notes: list[str] = []
    parts = [("document", _paragraphs(body, notes) if body is not None else [])]
    if pkg.has("styles.xml"):
        styles = pkg.xml("styles.xml")
        for kind in ("header", "footer"):
            seen: dict[str, None] = {}
            for variant in ("", "-left", "-first"):
                for el in styles.iter(f"{STYLE}{kind}{variant}"):
                    seen.update(dict.fromkeys(_paragraphs(el)))
            parts.append((kind, list(seen)))
    parts.append(("notes and comments", notes))
    for location, lines in parts:
        text = "\n".join(lines)
        if not text:
            continue
        if not budget.add(text):
            break
        segments.append(_segment(text, location))
    return segments


def _cell_text(cell: ET.Element, comments: list[str] | None = None) -> str:
    """A cell's value on one line (a line break would split the row for the table pass);
    its comment goes to `comments`, never into the row."""
    for note in cell.findall(f"{OFFICE}annotation"):
        if comments is not None and (text := " ".join(_paragraphs(note))):
            comments.append(text)
        cell.remove(note)
    text = _one_line(" ".join(_paragraphs(cell)))
    if text:
        return text
    for attr in ("value", "date-value", "time-value", "boolean-value", "string-value"):
        v = cell.get(f"{OFFICE}{attr}")
        if v:
            return v
    return ""


def _repeat(el: ET.Element, attr: str) -> int:
    try:
        return max(1, int(el.get(f"{TABLE}{attr}", "1")))
    except ValueError:
        return 1


def _spreadsheet(pkg: SafeZip, budget: _Budget) -> tuple[list[Segment], bool]:
    """Streamed row by row: a large sheet is never held whole, and every row counts against
    the output limit as it is made, so repeats cannot build a huge block first."""
    segments: list[Segment] = []
    lines: list[str] = []
    first = last = width = 0
    name, row_no, depth = "Sheet", 0, 0

    def flush() -> None:
        nonlocal width
        if lines:
            text = "\n".join(lines)
            segments.append(_segment(text, f"{name}!A{first}:{_col_letters(width)}{last}"))
            lines.clear()
            width = 0

    def add(line: str, n: int, cols: int) -> bool:
        """One line for row `n`; False once the output limit is reached."""
        nonlocal first, last, width
        if not budget.add(line + "\n"):
            return False
        if not lines:
            first = n
        lines.append(line)
        last, width = n, max(width, cols)
        if len(lines) >= BLOCK_ROWS:
            flush()
        return True

    for event, el in pkg.iterparse("content.xml", ("start", "end")):
        if el.tag == f"{TABLE}table":
            if event == "start":
                depth += 1
                if depth == 1:  # a sheet; a table inside a cell is that cell's text
                    flush()
                    name, row_no = el.get(f"{TABLE}name", "Sheet"), 0
            else:
                depth -= 1
                if depth == 0:
                    flush()
                    el.clear()
            continue
        if event != "end" or el.tag != f"{TABLE}table-row" or depth != 1:
            continue
        # Sparse: empty runs only move the column, so a tiny file cannot ask for millions of
        # cells; columns past MAX_COLUMNS are dropped.
        values: dict[int, str] = {}
        comments: list[str] = []
        col = 0
        for cell in el:
            if cell.tag not in _CELLS:
                continue
            value = _cell_text(cell, comments)
            n = _repeat(cell, "number-columns-repeated")
            if value:
                for k in range(min(n, MAX_REPEAT)):
                    if col + k < MAX_COLUMNS:
                        values[col + k] = value
            col += n
            if col >= MAX_COLUMNS:
                break
        n = _repeat(el, "number-rows-repeated")
        el.clear()
        if values:
            line = "\t".join(values.get(k, "") for k in range(max(values) + 1))
            for _ in range(min(n, MAX_REPEAT)):
                row_no += 1
                if not add(line, row_no, max(values) + 1):
                    flush()
                    return segments, True
            row_no += max(0, n - MAX_REPEAT)
        else:
            row_no += n
        if comments:  # a line of its own after the row: the table pass skips it
            note = "Comment: " + " / ".join(comments).replace("\t", " ")
            if not add(note, row_no, 1):
                flush()
                return segments, True
    flush()
    return segments, False


def _presentation(pkg: SafeZip, budget: _Budget) -> list[Segment]:
    body = pkg.xml("content.xml").find(f"{OFFICE}body/{OFFICE}presentation")
    segments: list[Segment] = []
    if body is None:
        return segments
    for n, page in enumerate(body.iter(f"{DRAW}page"), 1):
        notes_el = page.find(f"{PRES}notes")
        lines = [ln for child in page if child is not notes_el for ln in _paragraphs(child)]
        if child_lines := (_paragraphs(notes_el) if notes_el is not None else []):
            lines += ["Notes:", *child_lines]
        if not lines:
            continue
        text = "\n".join(lines)
        if not budget.add(text):
            break
        segments.append(_segment(text, f"slide {n}"))
    return segments


class OpenDocumentExtractor:
    api_version = API_VERSION
    id = "extractor.opendocument"

    def extract(self, data: bytes, path: str, limits: Limits) -> ExtractResult:
        try:
            return self._extract(data, path, limits)
        except ExtractError as e:
            if e.reason in ("archive-bomb", "xml-doctype", "odf-encrypted"):
                return ExtractResult(skip=e.reason)  # a refusal, not a broken file
            raise
        except (zipfile.BadZipFile, zlib.error, EOFError, ET.ParseError, KeyError,
                NotImplementedError, RuntimeError, ValueError, OSError) as exc:  # fmt: skip
            raise ExtractError("odf-unreadable") from exc

    def _extract(self, data: bytes, path: str, limits: Limits) -> ExtractResult:
        pkg = SafeZip(data, limits, unreadable="odf-unreadable")
        # A password-protected document: its manifest says so. Read as bytes, not parsed:
        # OpenOffice.org's manifests carry a DOCTYPE, which the XML reader refuses.
        manifest = (
            pkg.read("META-INF/manifest.xml", allow_doctype=True)
            if pkg.has("META-INF/manifest.xml")
            else b""
        )
        if b"encryption-data" in manifest or (pkg.has("encrypted-package")
                                               and not pkg.has("content.xml")):  # fmt: skip
            raise ExtractError("odf-encrypted")
        if not pkg.has("content.xml"):
            raise ExtractError("odf-unknown-format")
        kind = None
        if pkg.has("mimetype"):
            kind = MIMETYPES.get(pkg.read("mimetype")[:200].decode("ascii", "replace").strip())
        if kind is None:
            kind = {
                ".odt": "text",
                ".ott": "text",
                ".ods": "spreadsheet",
                ".ots": "spreadsheet",
                ".odp": "presentation",
                ".otp": "presentation",
            }.get(  # fmt: skip
                path[path.rfind(".") :].lower()
            )
        if kind is None:
            raise ExtractError("odf-unknown-format")
        budget = _Budget(limits.output_mb * 1024 * 1024)
        capped = False
        if kind == "text":
            segments = _text_document(pkg, budget)
        elif kind == "spreadsheet":
            segments, capped = _spreadsheet(pkg, budget)
        else:
            segments = _presentation(pkg, budget)
        note = None
        if budget.exceeded or capped:
            note = f"output limit of {limits.output_mb} MB reached; the rest was not indexed"
        return ExtractResult(segments, note=note)
