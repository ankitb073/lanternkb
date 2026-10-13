"""Decoding views: the text with one encoding undone, and a map back to original offsets.

The rules read raw text, so a secret inside an encoding goes unseen: an XML document that
splits it across elements, a folded line, an escape (the third held-out secrets test:
47% of such secrets caught). Each view undoes one encoding; the scanner runs its
rules (and the table pass, and base64 decoding) on the view and maps findings back, so
redaction covers every original fragment. A view only adds findings.

All views here are round 3 of the 2026-10-03 rules: a candidate, off by default until a
held-out split shows it keeps more secrets out (`scan.rules.OPT_IN`).
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Callable

View = tuple[str, Callable[[int], int]]


def edit_view(text: str, edits: list[tuple[int, int, str]]) -> View | None:
    """Apply (start, end, replacement) edits, sorted and not overlapping; None if none.
    A view offset inside a replacement maps to the start of what it replaced."""
    if not edits:
        return None
    pieces: list[str] = []
    seg_view: list[int] = []
    seg_orig: list[int] = []
    linear: list[bool] = []
    pos = vpos = 0
    for a, b, rep in edits:
        if a < pos:
            continue
        pieces.append(text[pos:a])
        seg_view.append(vpos)
        seg_orig.append(pos)
        linear.append(True)
        vpos += a - pos
        if rep:
            pieces.append(rep)
            seg_view.append(vpos)
            seg_orig.append(a)
            linear.append(False)
            vpos += len(rep)
        pos = b
    pieces.append(text[pos:])
    seg_view.append(vpos)
    seg_orig.append(pos)
    linear.append(True)

    def to_orig(i: int) -> int:
        k = bisect_right(seg_view, i) - 1
        return seg_orig[k] + (i - seg_view[k]) if linear[k] else seg_orig[k]

    return "".join(pieces), to_orig


def window_views(text: str, edits: list[tuple[int, int, str]], lines: int = 2,
                 merge: int = 4096) -> list[View] | None:  # fmt: skip
    """Views of only the lines near the edits (sorted), each scanned as a text of its own: a
    view that changes a few places costs a scan of those places, not of the whole text, and
    no rule can match across text the view leaves out. Windows fewer than
    `merge` characters apart are one window, with the text between them, so a scan's fixed
    cost is not paid per edit."""
    if not edits:
        return None
    starts = [0] + [m.end() for m in re.finditer("\n", text)]
    spans: list[list[int]] = []  # first and last line of each window
    for a, b, _ in edits:
        i = bisect_right(starts, a) - 1
        j = bisect_right(starts, max(a, b - 1)) - 1
        lo, hi = max(0, i - lines), min(len(starts) - 1, j + lines)
        if spans and starts[lo] <= starts[spans[-1][1]] + merge:
            spans[-1][1] = max(spans[-1][1], hi)
        else:
            spans.append([lo, hi])
    out: list[View] = []
    k = 0
    for first, last in spans:
        lo = starts[first]
        hi = starts[last + 1] if last + 1 < len(starts) else len(text)
        mine = []
        while k < len(edits) and edits[k][0] < hi:
            a, b, rep = edits[k]
            mine.append((a - lo, b - lo, rep))
            k += 1
        if (view := edit_view(text[lo:hi], mine)) is not None:
            sub, f = view
            out.append((sub, lambda i, f=f, lo=lo: lo + f(i)))
    return out


# -- separators: extracted text where markup split or joined tokens ----------------------------

SEPARATOR = "\u2063"  # U+2063 INVISIBLE SEPARATOR, put by extractors at inline boundaries


_SEP_RUN = re.compile(r"[A-Za-z0-9_+/=-]+(?:\u2063[A-Za-z0-9_+/=-]+)+")


def _around(text: str, a: int, b: int, margin: int = 256) -> tuple[int, int]:
    """A span widened by up to `margin` characters each way, so a view of one run costs the
    same however long its line is. The window keeps whole lines where a line break is
    near, so rules anchored to a line's start or end still apply; in a longer line it cuts
    at a space, else between two pieces of a run, and never inside a token."""
    lo, hi = max(0, a - margin), min(len(text), b + margin)
    if lo > 0:
        for ch in ("\n", " ", SEPARATOR):
            if (cut := text.find(ch, lo, a)) != -1:
                lo = cut + 1
                break
        else:  # no cut: never start inside a token, whose tail could match a rule
            while lo < a and _TOKEN_CHAR.match(text, lo - 1):
                lo += 1
    if hi < len(text):
        for ch in ("\n", " ", SEPARATOR):
            if (cut := text.rfind(ch, b, hi)) != -1:
                hi = cut
                break
        else:
            while hi > b and _TOKEN_CHAR.match(text, hi):
                hi -= 1
    return lo, hi


_TOKEN_CHAR = re.compile(r"[A-Za-z0-9_+/=.-]")


def separator_join_view(text: str) -> list[View] | None:
    """Extractors mark a boundary between two inline elements whose texts touch (an HTML
    span, a Word run) with an invisible separator: the scan reads it as a break, so
    "Access key ID" and "AKIA..." are two words. These views read a run of such texts with
    each stretch of up to `JOIN_SPAN` neighbouring boundaries joined and the others read as
    spaces, so a token split by markup is whole whatever touches it on either side ("Token"
    + "AKIA" + "XYZ..." + "Copy" + "Revoke"), and the run joined whole.
    A stretch's view is its pieces and a few hundred characters around them, so a long run
    costs in proportion to its length. Past `JOIN_CHARS` characters of views in all (a long
    page dense with inline elements), a few combined readings in merged windows instead."""
    if SEPARATOR not in text:
        return None
    runs, readings = [], []  # readings: (window start, end, the separators joined)
    for run in _SEP_RUN.finditer(text):
        seps = [run.start() + i for i, c in enumerate(run.group(0)) if c == SEPARATOR]
        runs.append(seps)
        firsts, lasts = [run.start()] + [i + 1 for i in seps], [*seps, run.end()]  # pieces
        readings.append((*_around(text, run.start(), run.end()), seps))
        for a in range(len(seps)):
            for b in range(a + 1, min(a + JOIN_SPAN, len(seps)) + 1):
                if b - a < len(seps):  # separator a joins pieces a and a + 1
                    readings.append((*_around(text, firsts[a], lasts[b]), seps[a:b]))
    out: list[View] = []
    if len(readings) <= JOIN_VIEWS and sum(hi - lo for lo, hi, _ in readings) <= JOIN_CHARS:
        for lo, hi, chosen in readings:
            view = edit_view(text[lo:hi], [(i - lo, i - lo + 1, "") for i in chosen])
            if view is not None:
                sub, f = view  # the separators left read as spaces, as in the scan
                out.append((sub.replace(SEPARATOR, " "), lambda k, f=f, lo=lo: lo + f(k)))
        # and every separator joined at once, in merged windows: a secret split on both
        # sides of punctuation spans two runs (`Hun<b>ter2</b>!x<b>yz9q</b>`), which no
        # single run's reading holds
        if len(runs) > 1:
            everything = [(i, i + 1, "") for seps in runs for i in seps]
            out += [(sub.replace(SEPARATOR, " "), f)
                    for sub, f in window_views(text, everything) or []]  # fmt: skip
        return out or None
    # Many runs (a long page dense with inline elements): readings across the text, in
    # merged windows each: every run with its first and last few boundaries kept and the
    # rest joined, so up to three (two past 2 MB) touching texts on either side stay apart.
    keep = 4 if len(text) <= 2_000_000 else 3
    for left in range(keep):
        for right in range(keep):
            edits = [(i, i + 1, "") for seps in runs for i in seps[left : len(seps) - right]]
            for sub, f in window_views(text, edits) or []:
                out.append((sub.replace(SEPARATOR, " "), f))
    return out or None


JOIN_SPAN = 6  # a token split into up to seven pieces
JOIN_VIEWS = 20_000  # views of single runs, at most
JOIN_CHARS = 6_000_000  # characters in those views, at most (about a second of scanning)


# -- markup: XML and HTML text with tags removed, cells as tabs, rows as lines -----------------

_TAG = re.compile(r"<(/?)([A-Za-z][\w:.-]{0,60})((?:\s[^<>]{0,400}?)?)(/?)>", re.S)
_INLINE = frozenset("span r t run b i u em strong code font a sub sup data is v s mark small tspan "
                    "wbr abbr kbd samp var q cite".split())  # fmt: skip
_PROPS = frozenset("rpr ppr sectpr tblpr trpr tcpr rfonts lang sz color".split())
_CELLS = frozenset("td th cell c table-cell tc".split())
_ROWS = frozenset("tr row table-row".split())
_BLOCKS = frozenset("p div br li h1 h2 h3 h4 h5 h6 h section article table tbody thead tfoot "
                    "ul ol si title header footer body html head line-break pre blockquote "
                    "dt dd".split())  # fmt: skip
_GAP = re.compile(r">([ \t\r\n]+)<")
_ENTITY = re.compile(r"&(?:#\d{1,7}|#[xX][0-9A-Fa-f]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});")
# invisible characters an entity may put inside a token (`Pass&shy;word`): dropped
_INVISIBLE = frozenset("\u00ad\u200b\u200c\u200d\u2060\ufeff")
_MARKUP = (".xml", ".html", ".htm", ".xhtml", ".fodt", ".fods", ".fodp", ".svg", ".rss",
           ".atom", ".xaml", ".resx", ".config")  # fmt: skip


def _entity(m: re.Match[str]) -> str:
    """Every HTML entity, named or numbered (six names were not enough)."""
    import html

    out = html.unescape(m.group(0))
    if out in _INVISIBLE:
        return ""
    return " " if out.isspace() else out  # &nbsp;, &ensp;, &#160;: a space to the rules


def _comments(text: str) -> list[tuple[int, int]]:
    """`<!-- -->` spans, found in one pass: a regex would be quadratic on many `<!--`
    without an end. An unclosed comment runs to the end, as in a browser."""
    out, pos = [], 0
    while (start := text.find("<!--", pos)) != -1:
        end = text.find("-->", start + 4)
        end = len(text) if end == -1 else end + 3
        out.append((start, end))
        pos = end
    return out


def markup_view(text: str, path: str) -> View | None:
    """XML or HTML as text: whitespace between tags dropped; inline and unknown tags removed,
    so a secret split across them is whole; cells as tabs and rows as lines, so the table
    pass reads a sheet's or a page's table (a paragraph inside a cell is a space); other
    block tags as line breaks; comments and run properties removed; entities decoded
    (`rk&#x5f;live` is `rk_live`)."""
    head = text.lstrip()[:200].lower()
    if not (
        path.lower().endswith(_MARKUP) or head.startswith(("<?xml", "<!doctype html", "<html"))
    ):
        return None
    comments = _comments(text)
    starts = [a for a, _ in comments]
    # Whitespace between tags is one space: dropped, it glued pretty-printed values into
    # one run that was redacted whole.
    edits = [(m.start(1), m.end(1), " ") for m in _GAP.finditer(text)]
    edits += [(a, b, "") for a, b in comments]
    depth = 0  # inside how many cells
    for m in _TAG.finditer(text):
        k = bisect_right(starts, m.start()) - 1
        if k >= 0 and m.start() < comments[k][1]:
            continue  # inside a comment, which goes whole
        closing, name, self_closing = m.group(1), m.group(2), m.group(4)
        local = name.rsplit(":", 1)[-1].lower()
        if local in _CELLS and not self_closing:
            depth = max(0, depth - 1) if closing else depth + 1
            rep = "\t" if closing else ""
        elif local in _ROWS:
            rep = "\n" if closing else ""
        elif local in ("s", "space"):
            rep = " " if self_closing else ""
        elif local == "tab":
            rep = "\t"
        elif local in _BLOCKS:
            rep = " " if depth else "\n"
        else:  # inline, properties and anything unknown
            rep = ""
        edits.append((m.start(), m.end(), rep))
    edits += [(m.start(), m.end(), _entity(m)) for m in _ENTITY.finditer(text)]
    edits.sort()
    # overlapping edits (a gap inside a comment) keep the first; edit_view skips the rest
    return edit_view(text, edits)


# -- folded lines -----------------------------------------------------------------------------

_FOLDED = re.compile(r"^(?:dn:|BEGIN:V(?:CALENDAR|CARD|EVENT|TODO))", re.M)
_FOLD = re.compile(r"\r?\n[ \t]")


def unfold_view(text: str) -> View | None:
    """LDIF, iCalendar and vCard fold long lines: a line break then one space or tab
    continues the line. A base64 value (LDIF `::`) or a token folded over lines is whole
    in the view."""
    if not _FOLDED.search(text[:20_000]):
        return None
    edits = [(m.start(), m.end(), "") for m in _FOLD.finditer(text)
             if not _MARKER_END.search(text, max(0, m.start() - 80), m.start())]  # fmt: skip
    return window_views(text, edits)


_FLOWED_MARK = re.compile(r"(?i)format\s*=\s*\"?flowed")
_DELSP = re.compile(r"(?i)delsp\s*=\s*\"?yes")
_FLOWED = re.compile(r" (\r?\n)(?!\r?\n)")


def flowed_view(text: str) -> list[View] | None:
    """Email in `format=flowed` breaks a long line after a space (RFC 3676); the view joins
    such lines both ways: keeping the space (it separated two words) and dropping it (a
    client broke one token there, as `DelSp=yes` says, and some do without saying so). With
    `DelSp=yes`, only dropped. Joined words only ever drop a space, never invent a token
   ."""
    if not _FLOWED_MARK.search(text):
        return None
    breaks = list(_FLOWED.finditer(text))
    dropped = window_views(text, [(m.start(), m.end(), "") for m in breaks]) or []
    if _DELSP.search(text):
        return dropped or None
    kept = window_views(text, [(m.start(1), m.end(1), "") for m in breaks]) or []
    return (kept + dropped) or None


# A long line ending in a token-like run, continued on the next line: a value wrapped by a
# PDF extractor or a narrow column. The run must look like a secret (letters and digits).
# The continuation starts with a letter or digit (not `---`, `- item` or `+`), and a run
# ending in base64 padding is complete, not wrapped (a YAML value before `---`)
_WRAPPED = re.compile(
    r"(?<![A-Za-z0-9_+/=-])([A-Za-z0-9_+/=-]{15,}+(?<![=]))\r?\n"
    r"([A-Za-z0-9][A-Za-z0-9_+/=-]{0,199})"
)
_MARKERS = re.compile(r"\[REDACTED:[a-z0-9._-]+\]")


# where text is prose a long token can wrap: notes, mail, documents' extracted text (not
# files without an extension: configs and netrc); every round 3 gain on the spent
# splits came from .txt and .eml
_PROSE = frozenset({".txt", ".text", ".eml", ".mbox", ".md", ".markdown", ".rst", ".org",
                    ".pdf", ".docx", ".odt", ".rtf"})  # fmt: skip
_MARKER_END = re.compile(r"\[REDACTED:[a-z0-9._-]+\]$")


def wrap_view(text: str, path: str = "") -> View | None:
    """Prose only: in a table a line is a row, and a row ending in a password is not a
    value wrapped onto the next row."""
    suffix = path[path.rfind(".") :].lower() if "." in path.rsplit("/", 1)[-1] else ""
    if suffix not in _PROSE:
        return None  # config files and logs: a line's end is not a wrap
    edits = []
    for m in _WRAPPED.finditer(text):
        run = m.group(1) + m.group(2)
        line_start = text.rfind("\n", 0, m.start()) + 1
        nl = m.end(1)
        next_end = text.find("\n", nl + 1)
        line, after = text[line_start:nl], text[nl + 1 : next_end if next_end != -1 else None]
        # rows of a table (both lines hold the same delimiter) are not a wrapped value; the
        # length is measured without markers, so the second scan joins the same lines
        rows = any(d in line and d in after for d in "\t,;|")
        if (not rows and len(_MARKERS.sub("", line)) >= 40 and any(c.isdigit() for c in run)
                and any(c.isalpha() for c in run)):  # fmt: skip
            edits.append((nl, m.start(2), ""))
    return window_views(text, edits)


# Round 6: a token with a vendor's prefix wrapped right after the prefix or a few characters
# (a PDF's line ended just after it began), which `_WRAPPED` leaves alone below 15 characters
_VENDOR_PREFIX = (r"(?:sk-proj-|sk-ant-|sk-svcacct-|sk-|[sr]k_live_|[sr]k_test_|pk_live_|whsec_|"
                  r"xox[abposr]-|xapp-|glpat-|glptt-|gh[pousr]_|github_pat_|hf_|npm_|pypi-|AKIA|"
                  r"ASIA|AIza|SG\.|ya29\.|shp(?:at|ss|ca)_|do[op]_v1_|sq0(?:atp|csp)-|lin_api_|"
                  r"\d{8,10}:AA)")  # fmt: skip
_VENDOR = re.compile(_VENDOR_PREFIX)
_PREFIXED_WRAP = re.compile(
    r"(?<![A-Za-z0-9_+/=.:-])(" + _VENDOR_PREFIX + r"[A-Za-z0-9_+/=.:-]{0,199})(\r?\n)"
    r"(?=[A-Za-z0-9_+/=.:-])"
)
# the shortest whole token of a vendor (a head this long may be complete); others 20
_VENDOR_MIN = {"hf_": 37, "ghp_": 40, "gho_": 40, "ghu_": 40, "ghs_": 40, "ghr_": 40,
               "github_pat_": 82, "glpat-": 26, "glptt-": 46, "npm_": 40, "AKIA": 20, "ASIA": 20,
               "AIza": 39, "sk-proj-": 100, "sk-ant-": 100, "sk-svcacct-": 100, "SG.": 69,
               "pypi-": 80, "sk_live_": 32, "rk_live_": 32, "sk_test_": 32, "rk_test_": 32,
               "xoxb-": 40, "xoxp-": 40, "xoxa-": 40}  # fmt: skip


def _complete(head: str) -> bool:
    """Whether a token head is already as long as its vendor's shortest whole token."""
    best = max((p for p in _VENDOR_MIN if head.startswith(p)), key=len, default="")
    return len(head) >= _VENDOR_MIN.get(best, 46 if head[:1].isdigit() else 20)


def prefix_wrap_view(text: str, path: str = "") -> list[View] | None:
    """Prose only, as `wrap_view`: a line ending in a token that starts with a vendor's
    prefix (`hf_`, `xoxb-`, `sk-proj-`, `glpat-`, a Telegram bot id...) joined to the next
    line, so a token broken anywhere after its prefix is whole."""
    suffix = path[path.rfind(".") :].lower() if "." in path.rsplit("/", 1)[-1] else ""
    if suffix not in _PROSE:
        return None
    edits = []
    for m in _PREFIXED_WRAP.finditer(text):
        nxt = text[m.end(2) : m.end(2) + 64].split(None, 1)[0] if m.end(2) < len(text) else ""
        if _PLAIN_WORD.match(nxt) and _complete(m.group(1)):
            continue  # a whole token, then a new line of prose ("Login Secret: ...")
        edits.append((m.start(2), m.end(2), ""))
    return window_views(text, edits)


_PLAIN_WORD = re.compile(r"[A-Z]?[a-z]+[.,;:!?)]*$")


_LATEX = frozenset({".tex", ".sty", ".cls", ".bib", ".ltx", ".dtx", ".bbl"})
_LATEX_ESC = re.compile(r"\{\\_\}|\\textunderscore(?:\{\}|[ \t]?)|\\([_#%&$])|\\-|"
                        r"\\allowbreak(?:\{\}|[ \t]?)")  # fmt: skip


def latex_view(text: str, path: str = "") -> list[View] | None:
    """LaTeX sources with the escapes a token picks up decoded: `\\_` (and `{\\_}`,
    `\\textunderscore`) as `_`, `\\#` `\\%` `\\&` `\\$` as themselves, and the break hints
    `\\-` and `\\allowbreak` removed, so `gho\\_...` reads as `gho_...`."""
    suffix = path[path.rfind(".") :].lower() if "." in path.rsplit("/", 1)[-1] else ""
    if suffix not in _LATEX or "\\" not in text:
        return None
    edits = []
    for m in _LATEX_ESC.finditer(text):
        token = m.group(0)
        rep = m.group(1) or ("_" if "_" in token or "underscore" in token else "")
        edits.append((m.start(), m.end(), rep))
    return window_views(text, edits)


# -- escapes ----------------------------------------------------------------------------------

_JSON_U = re.compile(r"\\u00([2-7][0-9A-Fa-f])|\\(/)")


def json_escape_view(text: str) -> View | None:
    """JSON `\\u005f` escapes of printable ASCII, and `\\/` (PHP escapes every `/`), decoded."""
    edits = [(m.start(), m.end(), "/" if m.group(2) else chr(int(m.group(1), 16)))
             for m in _JSON_U.finditer(text)]  # fmt: skip
    return window_views(text, edits)


_RTF_HEX = re.compile(r"\\'([2-7][0-9A-Fa-f])")


def rtf_view(text: str) -> View | None:
    """RTF `\\'5f` escapes of printable ASCII decoded."""
    if not text.lstrip().startswith("{\\rtf"):
        return None
    edits = [(m.start(), m.end(), chr(int(m.group(1), 16))) for m in _RTF_HEX.finditer(text)]
    return edit_view(text, edits)


def nul_view(text: str) -> View | None:
    """NUL-separated text (a process's environment, `find -print0`) as lines: same length."""
    if "\x00" not in text:
        return None
    return text.replace("\x00", "\n"), lambda i: i


# -- registry exports: hex values ---------------------------------------------------------------

_REG_VALUE = re.compile(r'(?m)^(?:"[^"\n]{0,200}"|@)=hex(?:\(([0-9a-fA-F]{1,2})\))?:'
                        r"((?:[0-9a-fA-F]{2},?(?:\\\r?\n[ \t]*)?){4,})")  # fmt: skip


def registry_values(text: str) -> list[tuple[int, int, str]]:
    """(start, end, decoded text) of hex values in a Windows registry export: REG_SZ,
    REG_EXPAND_SZ and REG_MULTI_SZ (`hex(1)`, `hex(2)`, `hex(7)`) are UTF-16; others bytes."""
    if "hex" not in text or not text.lstrip("﻿").startswith(("Windows Registry", "REGEDIT")):
        return []
    out = []
    for m in _REG_VALUE.finditer(text):
        data = bytes.fromhex("".join(re.findall(r"[0-9a-fA-F]{2}", m.group(2))))
        kind = (m.group(1) or "3").lower()
        utf16 = kind in ("1", "2", "7") or (
            len(data) >= 8 and data[1::2].count(0) * 2 >= len(data) // 2
        )
        decoded = data.decode("utf-16-le", errors="replace") if utf16 else data.decode("latin-1")
        out.append((m.start(2), m.end(2), decoded.replace("\x00", "\n")))
    return out


# -- round 4 (2026-10-03): indexed formats the fourth held-out test showed missing ---------

_RUN = "A-Za-z0-9_+/=-"
_HEAD = re.compile(rf"[{_RUN}]{{6,}}$")
_TAIL = re.compile(rf"[{_RUN}]+")


def joins(before: str, after: str) -> bool:
    """Whether a line break between `before` and `after` (leading space stripped) looks like
    one inside a token: a run of six or more token characters ends the line, and the next
    starts with a run that has a digit, a symbol or a capital after its first letter. So
    `...BEU7t` + `1RCUI...` joins and `summary_v3.docx` + `see thread above` does not."""
    before = before[-256:].rstrip("\r")  # a long line costs no more
    if not _HEAD.search(before) or (m := _TAIL.match(after, 0, 256)) is None:
        return False
    tail = m.group(0)
    return any(c.isdigit() or c in "_+/=-" for c in tail) or any(c.isupper() for c in tail[1:])


_YAML = (".yaml", ".yml")
_BLOCK = re.compile(r"(?:^|:|-)[ \t]*[>|][+-]?\d?[ \t]*(?:#.*)?$")


_COMMENT = re.compile(r"^[ \t]*#")
# a key's value or a list item that starts with a quote; a line that starts with one (what
# closes a folded value, or what a redaction left) never opens one
_OPENS = re.compile(
    r"""^[ \t]*(?:-[ \t]+(?:[^\s#'"][^#'"]*?:[ \t]+)?|[^\s#'"-][^#'"]*?:[ \t]+)(["'])"""
)


def _closes(line: str, start: int, q: str) -> bool:
    """Whether a YAML scalar quoted with `q` closes in `line` from `start` (`\\"` and `''`
    are escapes, not ends)."""
    i = start
    while (i := line.find(q, i)) >= 0:
        j = i
        while j > start and line[j - 1] == "\\":
            j -= 1
        if q == '"' and (i - j) % 2:
            i += 1  # an escaped quote
        elif q == "'" and line.startswith("''", i):
            i += 2
        else:
            return True
    return False


def yaml_fold_view(text: str, path: str) -> View | None:
    """YAML: a value folded over lines inside a block scalar (`>`, `|`) or an unclosed quoted
    scalar, joined without the line break and indentation, so a token split there is whole.
    A quote opens only where a value starts with one, never in a comment."""
    if not path.lower().endswith(_YAML):
        return None
    lines = text.split("\n")
    offs, pos = [], 0
    for line in lines:
        offs.append(pos)
        pos += len(line) + 1
    edits = []
    block = -1  # the indentation that keeps a block scalar open, or -1
    quote = ""  # the quote of a scalar still open at the end of the line
    for i, raw in enumerate(lines[:-1]):
        line = raw.rstrip("\r")
        indent = len(line) - len(line.lstrip(" "))
        nxt = lines[i + 1]
        if block >= 0 and line.strip() and indent < block:
            block = -1
        if block < 0:
            if quote:
                if _closes(line, 0, quote):
                    quote = ""
            elif not _COMMENT.match(line):
                if _BLOCK.search(line.rstrip()):
                    block = len(nxt) - len(nxt.lstrip(" ")) if nxt.strip() else indent + 1
                    continue
                if (m := _OPENS.match(line)) and not _closes(line, m.end(), m.group(1)):
                    quote = m.group(1)
        # a block scalar's last line never joins the key after it
        in_block = block >= 0 and len(nxt) - len(nxt.lstrip(" ")) >= block
        if (in_block or (block < 0 and quote)) and joins(line, nxt.lstrip(" ")):
            edits.append((offs[i] + len(line), offs[i + 1] + len(nxt) - len(nxt.lstrip(" ")), ""))
    return edit_view(text, edits)


_PUNCT = re.compile(
    r"(?<![A-Za-z0-9_/+=-])([A-Za-z0-9_/+=-]{10,})([,.;:!?)\]}>~])(?=\s|$|[,.;:!?)\]}>~'\"])"
)


def punct_view(text: str) -> list[View] | None:
    """A space before punctuation that ends a token (`token: Meadow-lantern716,`), so rules
    whose values must end at a space or a quote see the value end. Only the changed lines
    are scanned (`window_views`)."""
    marks = [(m.start(), m.end()) for m in _MARKERS.finditer(text)]
    firsts = [a for a, _ in marks]
    edits = []
    for m in _PUNCT.finditer(text):
        run = m.group(1)
        if not any(c.isdigit() for c in run) or (len(run) < 20 and run.islower()):
            continue  # a word or a lower-case name (`batch_000123`) ending a sentence
        k = bisect_right(firsts, m.start(2)) - 1
        if k >= 0 and marks[k][0] < m.start(2) < marks[k][1]:
            continue  # inside a marker: it stays whole
        edits.append((m.start(2), m.start(2), " "))
    return window_views(text, edits)


_EMPHASIS = re.compile(
    r"(?<![A-Za-z0-9_/+=-])([A-Za-z0-9_/+=-]{10,})([*`|])(?=\s|$|[*`|,.;:!?)\]}>~'\"])"
)


def emphasis_view(text: str) -> list[View] | None:
    """Round 6: a space before Markdown emphasis or code marks, or a table bar, that end a
    token (`**sk_live_...**`, `` `token` ``), as `punct_view` does for punctuation."""
    marks = [(m.start(), m.end()) for m in _MARKERS.finditer(text)]
    firsts = [a for a, _ in marks]
    edits = []
    for m in _EMPHASIS.finditer(text):
        run = m.group(1)
        vendor = _VENDOR.match(run) is not None  # hf_ tokens may hold no digit
        if not vendor and (not any(c.isdigit() for c in run) or (len(run) < 20 and run.islower())):
            continue
        k = bisect_right(firsts, m.start(2)) - 1
        if k >= 0 and marks[k][0] < m.start(2) < marks[k][1]:
            continue
        edits.append((m.start(2), m.start(2), " "))
        k = m.start(1)  # and after an opening mark: rules want a space or quote before a token
        if k > 0 and text[k - 1] in "*`|" and (k < 2 or text[k - 2] in " \t\n*`|([{"):
            edits.append((k, k, " "))
    return window_views(text, sorted(edits))


def json_backslash_view(text: str, path: str) -> list[View] | None:
    """JSON: an escaped backslash or quote (`\\\\`, `\\"`) removed, so a password the rules'
    value class stops at is whole (`*Wil\\\\low57443+`, `Pixe\\"l1003+`)."""
    if not path.lower().endswith((".json", ".jsonl", ".ipynb")) or "\\" not in text:
        return None
    return window_views(text, [(m.start(), m.end(), "") for m in re.finditer(r'\\[\\"]', text)])
