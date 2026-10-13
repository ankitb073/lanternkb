"""Small text helpers shared by indexing and evaluation."""

from __future__ import annotations

import re

# Every character `str.splitlines` breaks on. The chunker stores lines joined by "\n", so
# text is scanned in that same form: a rule ending in `$` must see a CRLF line end just as
# it sees "\n".
BREAKS = re.compile("\r\n|[\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029]")


def one_break(text: str) -> str:
    """Every line break as a plain newline, as the chunker will store the text."""
    return BREAKS.sub("\n", text)


def split_identifier(name: str) -> list[str]:
    """`parseConfigV2` -> [parse, config, v2]; `HTTP2Client` -> [http2, client];
    `_merge_url` -> [merge, url]. Digits stay with the word before them."""
    words: list[str] = []
    for part in re.split(r"[_\W]+", name):
        words += re.findall(r"[A-Z]?[a-z]+\d*|[A-Z]+\d*(?=[A-Z][a-z]|\b)|[A-Z]+\d*|\d+", part)
    return [w.lower() for w in words if w]


def split_identifier_v1(name: str) -> list[str]:
    """The phase 1 splitter, frozen for the evaluation: the phase 1 queries and baselines
    were made with it (`parseConfigV2` -> [parse, config, v, 2])."""
    words: list[str] = []
    for part in re.split(r"[_\W]+", name):
        words += re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+\d*|[A-Z]+\d*|\d+", part)
    return [w.lower() for w in words if w]
