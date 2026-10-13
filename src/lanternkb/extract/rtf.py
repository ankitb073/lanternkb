"""Rich Text Format (`.rtf`, TextEdit's default) to plain text, with the standard library.

A small tokenizer, not a renderer: groups, control words, hex and Unicode escapes. Text in
destinations that are not part of the document (font and colour tables, style sheets, the
info block, pictures, embedded objects, field instructions) is dropped; everything else is
kept in order, with paragraphs, line breaks, tabs and table cells as line breaks and tabs.
"""

from __future__ import annotations

import codecs
import re

from lanternkb.extract.base import ExtractError

# Destinations whose content is not document text.
_IGNORE = frozenset(
    """fonttbl colortbl stylesheet info pict object objdata objclass datastore
    themedata colorschememapping latentstyles listtable listoverridetable rsidtbl generator
    xmlnstbl mmathPr fldinst bkmkstart bkmkend filetbl revtbl pgdsctbl operator panose
    falt fontemb fontfile listtext pntext pntxta pntxtb shppict nonshppict blipuid
    passwordhash protusertbl userprops docvar wgrffmtfilter sp sn sv""".split()
)
# Starred destinations that hold document text: comments and text boxes (as DOCX and ODT
# keep them); every other starred destination is dropped.
_KEEP_STARRED = frozenset({"annotation", "shpinst", "shptxt"})
# Between visible groups a renderer draws no space: `{\b AKIA...}{\i Copy}` is two words,
# but Word also splits one token into runs (`AKIA{\insrsid1 XYZ...}`). A mark at each
# visible group boundary becomes U+2063 (invisible separator) where both sides are token
# characters: the scan reads it as a break and its join view without it, so a secret is
# found either way. Dropped groups (bookmarks, fields' instructions) leave no
# mark, so a token they interrupt reads whole.
_EDGE = "\x00"
_SEPARATOR = "\u2063"
_EDGE_SPACE = re.compile(r"(?<=[A-Za-z0-9_+/=-])\x00+(?=[A-Za-z0-9_+/=-])")
_TEXT = {"par": "\n", "line": "\n", "sect": "\n\n", "page": "\n\n", "row": "\n",
         "cell": "\t", "tab": "\t", "emdash": "\u2014", "endash": "\u2013",
         "bullet": "\u2022", "lquote": "\u2018", "rquote": "\u2019",
         "ldblquote": "\u201c", "rdblquote": "\u201d", "emspace": " ", "enspace": " ",
         "qmspace": " ", "nestcell": "\t", "nestrow": "\n"}  # fmt: skip
# `\` before a line break is a paragraph mark (TextEdit writes one per line)
_SYMBOL = {"~": "\u00a0", "_": "\u2011", "-": "", "\\": "\\", "{": "{", "}": "}",
           "\n": "\n", "\r": "\n"}  # fmt: skip
_TOKEN = re.compile(
    r"\\([a-zA-Z]{1,32})(-?\d{1,10})? ?"  # a control word with its parameter
    r"|\\'([0-9a-fA-F]{2})"  # a byte in the document's code page
    r"|\\(.)"  # a control symbol
    r"|([{}])"
    r"|([^\\{}\r\n]+)"  # text
    r"|[\r\n]+",  # line breaks in the source are not text
    re.S,
)
MAX_DEPTH = 512


def _codec(page: int) -> str:
    name = {10000: "mac_roman", 437: "cp437", 850: "cp850"}.get(page, f"cp{page}")
    try:
        codecs.lookup(name)
    except LookupError:
        return "cp1252"
    return name


def rtf_to_text(data: bytes) -> str:
    """The document text of an RTF file. Raises ExtractError if it is not RTF."""
    data = data.removeprefix(b"\xef\xbb\xbf")  # a UTF-8 byte order mark some tools add
    if not data.lstrip()[:5] == b"{\\rtf":
        raise ExtractError("not-rtf")
    src = data.decode("latin-1")  # one character per byte; hex escapes carry the rest
    out: list[str] = []
    notes: list[str] = []  # comments: after the text, never between a label and its value
    pending = bytearray()  # code-page bytes waiting to be decoded together (multi-byte pages)
    codec = "cp1252"
    # per group: (ignored, characters to skip after \uN, tracked deletion, a comment, its
    # parent's boundary still pending)
    stack: list[tuple[bool, int, bool, bool, bool]] = []
    ignored, uc, deleted, note = False, 1, False, False
    skip = 0  # fallback characters still to skip after a \uN
    star = False  # the last token was \*: the next control word names a destination
    first_word = False  # the next token is the first in its group
    edge = False  # a visible group opened: mark its boundary before its first text

    def emit(piece: str) -> None:
        nonlocal edge
        if note:
            notes.append(piece)
            return
        if edge:
            out.append(_EDGE)
            edge = False
        out.append(piece)

    def flush() -> None:
        if pending:
            emit(pending.decode(codec, "replace"))
            pending.clear()

    pos, n = 0, len(src)
    while pos < n:
        m = _TOKEN.match(src, pos)
        if m is None:  # a lone backslash at the end
            break
        pos = m.end()
        word, param, hexbyte, symbol, brace, text = m.groups()
        starting, was_star = first_word, star
        first_word, star = False, False
        if brace == "{":
            if len(stack) >= MAX_DEPTH:
                raise ExtractError("rtf-too-deep")
            flush()
            stack.append((ignored, uc, deleted, note, edge))
            first_word, skip, edge = True, 0, not ignored
            continue
        if brace == "}":
            flush()
            closed_visible, closed_note = not ignored, note
            ignored, uc, deleted, note, parent_edge = (
                stack.pop() if stack else (ignored, uc, deleted, note, False)
            )
            skip = 0
            if closed_visible and not ignored and not closed_note:
                out.append(_EDGE)
                edge = False
            else:  # a dropped group leaves its parent's boundary as it was
                edge = parent_edge
            if closed_note and not note:
                notes.append("\n")
            continue
        if symbol is not None:
            if symbol == "*":
                star = True
                first_word = starting  # \* comes first; the destination word follows
                continue
            if not ignored and not deleted and symbol in _SYMBOL:
                flush()
                emit(_SYMBOL[symbol])
            continue
        if hexbyte is not None:
            if skip:
                skip -= 1
            elif not ignored and not deleted:
                pending.append(int(hexbyte, 16))
            continue
        if word is not None:
            if was_star and starting and word == "annotation":
                note = True  # a comment: kept, after the text
            elif (was_star and starting and word not in _KEEP_STARRED) or (
                    starting and word in _IGNORE):  # fmt: skip
                ignored, edge = True, False
                continue
            if word == "bin" and param:  # raw binary data follows: skip it
                pos += max(0, int(param))
                continue
            if word == "ansicpg" and param:
                flush()
                codec = _codec(int(param))
            elif word == "mac":
                codec = "mac_roman"
            elif word == "uc" and param is not None:
                uc = max(0, int(param))
            elif word == "deleted":  # tracked deletion, as DOCX and ODT leave out
                deleted = param is None or int(param) != 0
            elif word == "plain":  # character formatting back to the default
                deleted = False
            elif word == "u" and param is not None:
                skip = uc
                code = int(param)
                code = code + 65536 if code < 0 else code
                if not ignored and not deleted and 0 <= code <= 0x10FFFF:
                    flush()
                    emit(chr(code))
            elif not ignored and not deleted and word in _TEXT:
                flush()
                emit(_TEXT[word])
            continue
        if text is not None and not ignored and not deleted:
            if skip:
                cut = min(skip, len(text))
                text, skip = text[cut:], skip - cut
            if text and text.isascii():
                flush()
                emit(text)
            elif text:  # raw 8-bit text is in the document's code page too
                pending.extend(text.encode("latin-1"))
        elif text is not None and skip:
            skip = max(0, skip - len(text))
    flush()
    text = "".join(out) + ("\n\n" + "".join(notes) if "".join(notes).strip() else "")
    joined = _EDGE_SPACE.sub(_SEPARATOR, text).replace(_EDGE, "")
    # surrogate pairs written as two \uN escapes become one character
    joined = joined.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    lines = [ln.rstrip() for ln in joined.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
