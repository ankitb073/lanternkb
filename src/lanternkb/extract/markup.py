"""Web pages and subtitles to plain text, with the standard library only.

- **HTML:** the readable text of a page, without its markup. Scripts, styles and other
  non-text parts are dropped; block elements become line breaks and table cells tabs, so a
  page reads (and is scanned for secrets) the way it looks.
- **Subtitles (`.srt`, `.vtt`):** the spoken text in cue order, without numbers, timings or
  styling, in segments located at their first cue's start time (`00:01:12`).
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

from lanternkb.extract.base import Segment

# Elements whose content is not text a reader sees.
_SKIP = frozenset("script style noscript template svg math canvas iframe object embed".split())
_FOREIGN = frozenset({"svg", "math"})  # a browser leaves these at the next HTML block
_BLOCK = frozenset("p div br li ul ol dl dt dd h1 h2 h3 h4 h5 h6 table thead tbody tfoot "
                   "section article header footer nav aside blockquote pre figure figcaption "
                   "form fieldset legend address main details summary hr caption "
                   "option label body".split())  # fmt: skip
# HTML elements that end an unclosed <svg> or <math>, as a browser's parser does
_BREAKOUT = frozenset("p div h1 h2 h3 h4 h5 h6 ul ol li dl table section article main header "
                      "footer nav aside blockquote pre body form".split())  # fmt: skip
_CELL = frozenset({"td", "th"})
_VOID = frozenset("area base br col embed hr img input link meta source track wbr".split())
_HEAD = frozenset("title meta link style script base noscript template".split())
# Between two inline elements a browser may draw no space: "Access key ID" + "AKIA..." are
# two words, but "AKIA" + "<span>XYZ...</span>" is one token. A mark at every inline
# boundary becomes U+2063 (invisible separator) where both sides are token characters, and
# nothing elsewhere: the scan reads it as a break, and its join view (`scan.views`) reads
# the text without it, so a secret is found either way.
_EDGE = "\x00"
SEPARATOR = "\u2063"
_EDGE_SPACE = re.compile(r"(?<=[A-Za-z0-9_+/=-])\x00+(?=[A-Za-z0-9_+/=-])")
_NO_EDGE = frozenset({"wbr"})  # a break point inside a word: never a boundary


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip: list[str] = []  # open elements that are not shown, innermost last
        self.skip_open: dict[str, int] = {}  # how many of each are open (no list scans)
        self.pre = 0  # depth inside <pre>, where whitespace is kept
        self.title: list[str] = []
        self.in_title = False
        self.titled = False  # the page's first <title> only
        self.in_head = False  # ends at </head>, <body>, or the first body element or text
        self.cells = 0  # cells so far in the current table row
        self.in_cell = 0  # inside a cell, blocks are spaces: the row stays one line

    def _newline(self) -> None:
        self.parts.append(" " if self.in_cell else "\n")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "head":
            self.in_head = True
            return
        if self.in_head and tag not in _HEAD:
            self.in_head = False  # `</head>` and `<body>` may both be left out
        if tag == "title" and not self.skip and not self.titled:
            self.in_title = True
            return
        if self.skip and self.skip[-1] in _FOREIGN and tag in _BREAKOUT:
            while self.skip and self.skip[-1] in _FOREIGN:  # an unclosed <svg> ends here
                self._pop_skip()
        if tag in _SKIP:
            if tag not in _VOID:
                self.skip.append(tag)
                self.skip_open[tag] = self.skip_open.get(tag, 0) + 1
            return
        if self.skip:
            return
        if tag == "pre":
            self.pre += 1
        if tag == "table":
            self.in_cell = 0  # `</td>` may be left out: a table never inherits a cell
        if tag in _BLOCK:
            self._newline()
        elif tag in _CELL:
            if self.cells:
                self.parts.append("\t")  # between a row's cells, empty ones kept in place
            self.cells += 1
            self.in_cell = 1  # a new cell closes the last, whose `</td>` is optional
        elif tag == "tr":  # rows on consecutive lines, as the table pass reads them
            self.cells, self.in_cell = 0, 0
            if self.parts and not self.parts[-1].endswith("\n"):
                self.parts.append("\n")
        elif tag not in _NO_EDGE:
            self.parts.append(_EDGE)
        if tag == "li":
            self.parts.append("- ")
        if tag == "img":
            alt = next((v for k, v in attrs if k == "alt" and v), None)
            if alt:
                self.parts.append(f" {alt} ")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP and tag not in _VOID:
            return  # `<script/>` opens nothing
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag == "head":
            self.in_head = False
            return
        if tag == "title" and self.in_title:
            self.in_title, self.titled = False, True
            return
        if self.skip:
            if self.skip_open.get(tag):  # close it, and anything left open inside it
                while self._pop_skip() != tag:
                    pass
            return
        if tag in _SKIP:
            return  # a stray end tag: it hides nothing
        if tag == "pre" and self.pre:
            self.pre -= 1
        if tag in _CELL or tag == "table":
            self.in_cell = 0
            if tag == "table":
                self._newline()
        elif tag in _BLOCK:
            self._newline()
        elif tag == "tr":
            self.parts.append("\n")
            self.in_cell = 0
        elif tag not in _NO_EDGE:
            self.parts.append(_EDGE)

    def _pop_skip(self) -> str:
        tag = self.skip.pop()
        self.skip_open[tag] -= 1
        return tag

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title.append(data)
            return
        if self.skip:
            return
        if self.in_head:
            if not data.strip():
                return
            self.in_head = False  # text means the body has begun
        self.parts.append(data if self.pre else re.sub(r"\s+", " ", data))


def html_to_text(html: str) -> str:
    """The page's readable text: its title first, then its body as lines."""
    p = _Text()
    p.feed(html)
    p.close()
    body = _EDGE_SPACE.sub(SEPARATOR, "".join(p.parts)).replace(_EDGE, "")
    lines = [re.sub(r" *\t *", "\t", ln).strip(" ") for ln in body.split("\n")]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    title = re.sub(r"\s+", " ", "".join(p.title)).strip()
    if title and not text.startswith(title):
        text = f"{title}\n\n{text}" if text else title
    return text


# -- subtitles --------------------------------------------------------------------------------

_STAMP = r"(?:\d{1,3}:)?\d{1,2}:\d{2}(?:[.,]\d{1,3})?"
_TIMING = re.compile(rf"^\s*({_STAMP})\s*-->\s*({_STAMP})")
# <i>, </i>, <c.red>, <v Bob>, <00:01.000>, {\an8}; not "a<b and c>d" in the text
_CUE_TAG = re.compile(
    r"</?(?:[ibu]|ruby|rt)>|</?(?:c|v|lang|font|span)(?:[.\s][^<>]{0,100})?>"
    r"|<\d[\d:.]{0,15}>|\{\\[^}]{0,200}\}"
)
SUBTITLE_SEGMENT_CHARS = 1200


def _clock(stamp: str) -> str:
    """`00:01:12,500` or `01:12.500` as `00:01:12`."""
    parts = stamp.replace(",", ".").split(":")
    if len(parts) == 2:
        parts.insert(0, "0")
    h, m, s = int(parts[0]), int(parts[1]), int(float(parts[2]))
    return f"{h:02d}:{m:02d}:{s:02d}"


def subtitle_segments(text: str) -> list[Segment]:
    """Cue text in order, in segments of about `SUBTITLE_SEGMENT_CHARS` characters, each
    located at the start time of its first cue. A line repeated by the next cue (rolling
    captions) is kept once."""
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n"))
    cues: list[tuple[str, str]] = []
    last = ""
    for block in blocks:
        lines = block.strip("\n").split("\n")
        timing = next((i for i, ln in enumerate(lines) if _TIMING.match(ln)), None)
        if timing is None:
            continue  # the WEBVTT header, NOTE, STYLE and REGION blocks, or stray text
        start = _clock(_TIMING.match(lines[timing]).group(1))  # type: ignore[union-attr]
        words = []
        for ln in lines[timing + 1 :]:
            ln = re.sub(r"\s+", " ", _CUE_TAG.sub("", ln)).strip()
            if ln and ln != last:
                words.append(ln)
                last = ln
        if words:
            cues.append((start, " ".join(words)))
    segments: list[Segment] = []
    group: list[str] = []
    first = ""
    size = 0
    for start, line in cues:
        if group and size + len(line) > SUBTITLE_SEGMENT_CHARS:
            body = "\n".join(group)
            segments.append(Segment(body, first, 1, body.count("\n") + 1))
            group, size = [], 0
        if not group:
            first = start
        group.append(line)
        size += len(line) + 1
    if group:
        body = "\n".join(group)
        segments.append(Segment(body, first, 1, body.count("\n") + 1))
    return segments
