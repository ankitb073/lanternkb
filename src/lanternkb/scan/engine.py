"""The scanning engine and redaction.

For each rule: a cheap keyword check on the lower-cased text, then the regex over the whole
text (so multi-line keys match), then the secret group, minimum entropy and allowlists. A
second pass decodes base64 runs and scans the decoded text; a hit there covers the whole
encoded run.

Findings hold offsets, never the matched value, and their repr shows only rule and line,
so they are safe to log.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import json
import math
import re
import string
import time
from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import accumulate
from typing import Any

from lanternkb.scan import views as V
from lanternkb.scan.rules import STRUCTURAL, Allowlist, Rule, RuleSet, load_rules

B64_RUN = re.compile(r"(?<![A-Za-z0-9+/_-])[A-Za-z0-9+/_-]{20,}={0,2}")
DECODE_DEPTH = 2
VIEW_MAX = 2_000_000  # decoding views (round 3) only on texts up to this many characters
# windowed views that undo what an extractor did, needed at any size
_ANY_SIZE = frozenset({"separator-join-view"})
PRINTABLE_SHARE = 0.9


@dataclass(frozen=True)
class Finding:
    rule: str
    start: int  # offsets of the secret span within the scanned text
    end: int
    line: int  # 1-based line of start
    action: str  # "redact" | "skip-file"
    encoded: bool = False  # found inside a base64-encoded run

    def __repr__(self) -> str:  # never include the value
        enc = ", encoded" if self.encoded else ""
        return f"Finding({self.rule}, line {self.line}, {self.action}{enc})"


def shannon(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def _line_starts(text: str) -> list[int]:
    starts = [0]
    starts += [m.end() for m in re.finditer("\n", text)]
    return starts


def _allowed(al: Allowlist, secret: str, match: str, line: str, path: str,
             as_words: bool = False) -> bool:  # fmt: skip
    target = {"secret": secret, "match": match, "line": line}.get(al.target, secret)
    checks = []
    if al.regexes:
        checks.append(any(r.search(target) for r in al.regexes))
    if al.stopwords:
        checks.append(_has_stopword(secret, al.stopwords, as_words))
    if al.paths:
        checks.append(any(p.search(path) for p in al.paths))
    if not checks:
        return False
    return all(checks) if al.condition == "AND" else any(checks)


def _has_stopword(secret: str, words: Iterable[str], as_words: bool) -> bool:
    """gitleaks: a stopword anywhere in the secret allowlists it, so a random token that
    happens to hold `doc_` or `base` is dropped. With `as_words` (round 6, the
    `stopword-words` pass), a short stopword (under five letters, which a random token holds
    by chance) counts only where it reads as a word of the value; longer ones anywhere."""
    low = secret.lower()
    if not as_words or not _random_looking(secret):
        return any(w in low for w in words)  # a name or prose: gitleaks' reading
    for w in words:
        if len(w.rstrip("-._")) >= 5:
            if w in low:
                return True
            continue
        i = low.find(w)
        while i != -1:
            if _word_at(secret, i, len(w)):
                return True
            i = low.find(w, i + 1)
    return False


def _random_looking(s: str) -> bool:
    """How a generated token reads: upper case, lower case and digits together, with most
    letters outside words of three or more (`Ministral3ForCausalLM` is a name)."""
    if not (any(c.isupper() for c in s) and any(c.islower() for c in s)
            and any(c.isdigit() for c in s)):  # fmt: skip
        return False
    letters = sum(c.isalpha() for c in s)
    in_words = sum(len(w) for w in _WORD_RUN.findall(s))
    return in_words < 0.6 * letters


_WORD_RUN = re.compile(r"[A-Z]?[a-z]{3,}")  # (_WORDISH, below, is the table pass's)


def _word_at(s: str, i: int, n: int) -> bool:
    """Whether s[i:i+n] is a word of s: lower case, upper case or capitalised; at the start,
    after a character that is not a letter or digit, or capitalised after a lower-case
    letter (camelCase); at the end, before a character that is not a letter or digit, or
    before a capital (camelCase). A delimiter the stopword ends with is its own end."""
    core = s[i : i + n].rstrip("-._")
    if not core or not (core.islower() or core.isupper()
                        or (core[0].isupper() and core[1:].islower())):  # fmt: skip
        return False
    camel = core[0].isupper() and not core.isupper()
    prev = s[i - 1] if i > 0 else ""
    if prev.isalnum() and not (camel and (prev.islower() or prev.isdigit())):
        return False  # inside a run of letters (Qwen3Next: a word after a digit)
    end = i + n
    if len(core) < n or end >= len(s):
        return True
    for suffix in ("", "s", "es", "ed", "er", "ing"):  # Types, cached
        tail = s[end : end + len(suffix)]
        if tail.lower() != suffix or (suffix and tail.isupper() != core.isupper()):
            continue
        after = s[end + len(suffix) : end + len(suffix) + 1]
        if not after.isalnum() or (after.isupper() and not core.isupper()):
            return True
    return False


class ScanTimeout(Exception):
    """A file took longer than the scan budget. Its text must not be indexed."""


class _Cover:
    """Whether a span lies inside one of a fixed set of spans, in O(log n): the maximum end
    over spans sorted by start (a plain `any(...)` per span was quadratic)."""

    def __init__(self, spans: Iterable[tuple[int, int]]):
        ordered = sorted(spans)
        self.starts = [s for s, _ in ordered]
        self.reach = list(accumulate((e for _, e in ordered), max))

    def __call__(self, s: int, e: int) -> bool:
        k = bisect_right(self.starts, s)
        return k > 0 and self.reach[k - 1] >= e


class _GrowingCover:
    """Like _Cover, for a set that grows while it is asked: every start is known up front,
    so a prefix-maximum Fenwick tree answers and grows in O(log n)."""

    def __init__(self, starts: Iterable[int]):
        self.keys = sorted(set(starts))
        self.tree = [-1] * (len(self.keys) + 1)

    def add(self, s: int, e: int) -> None:
        i = bisect_left(self.keys, s) + 1
        while i < len(self.tree):
            self.tree[i] = max(self.tree[i], e)
            i += i & -i

    def __call__(self, s: int, e: int) -> bool:
        i, best = bisect_right(self.keys, s), -1
        while i > 0:
            best = max(best, self.tree[i])
            i -= i & -i
        return best >= e


# The redaction safety net: a finding is widened to the whole run of token-like ASCII
# characters around it, so no tail of a secret survives when a rule stops early (at a `.`,
# `,`, `)` or `&` inside the value, say). It stops at whitespace, quotes, opening brackets,
# `=` and `:` on the left, `@` and quotes on the right, and any non-ASCII character (so CJK
# prose is kept). Closing brackets are crossed: losing a `)` beats leaking a password tail.
_LEFT = frozenset(string.ascii_letters + string.digits + "+/_-.~")
_RIGHT = frozenset(string.ascii_letters + string.digits + "+/=_-.~!#$%*?^&|,:)]}>")


# In a delimited file the delimiter ends a cell: crossing it would redact the next cell
# and break the row's shape for the second scan.
_RIGHT_CELL = _RIGHT - {",", "|"}
_DELIMITED = (".csv", ".psv")


def _widen(text: str, s: int, e: int, right: frozenset[str] = _RIGHT) -> tuple[int, int]:
    if e <= s:
        return s, e
    while s > 0 and text[s - 1] in _LEFT:
        s -= 1
    while e < len(text) and text[e] in right:
        e += 1
    return s, e


class Scanner:
    def __init__(self, ruleset: RuleSet | None = None, budget_s: float = 20.0):
        self.ruleset = ruleset or load_rules()
        self.rules: tuple[Rule, ...] = self.ruleset.rules
        self.hash = self.ruleset.hash
        self.ids = frozenset(r.id for r in self.rules) | frozenset(STRUCTURAL)
        self.budget_s = budget_s
        self._deadline = float("inf")
        self._redacted = False
        self._marked: str | None = None
        # an action override for the table pass; other passes report the rule
        # that matched, which carries its own action
        self._table_action = dict(self.ruleset.pass_actions).get("csv-secret-column", "redact")

    def scan(self, text: str, path: str = "", _depth: int = 0, *, redacted: bool = False,
             breaks: tuple[int, ...] = ()) -> list[Finding]:  # fmt: skip
        """All findings in text, sorted by position. path is relative, POSIX.

        `redacted` is for scanning redacted output again: a match made only of
        this rule set's own markers is then not a finding.
        `breaks` are offsets where an extractor's parts start (pages, sheets), for the
        table pass: a new table may start there.
        Raises ScanTimeout if the file takes longer than the budget; callers skip the file.
        """
        if _depth == 0:
            self._deadline = time.monotonic() + self.budget_s
            self._redacted = redacted
            # Extractors' invisible separators (U+2063, between touching inline texts) read
            # as spaces in every pass, one character for one, so a key that touches the next
            # word still ends where the rules expect; the join view alone reads
            # the text without them.
            self._marked = text if V.SEPARATOR in text else None
            if self._marked is not None:
                text = text.replace(V.SEPARATOR, " ")
        # the table pass handles cells in a delimited file; without it, widening on keeps
        # catching the next cell's secret (phase 2's rules relied on that)
        table_on = "csv-secret-column" in self.ruleset.passes
        right = _RIGHT_CELL if table_on and path.lower().endswith(_DELIMITED) else _RIGHT
        starts = _line_starts(text)
        found = [
            Finding(rule.id, *_widen(text, s, e, right), bisect_right(starts, s), rule.action)
            for rule, s, e in self._match(text, path)
        ]
        views = []
        if _depth == 0 and (view := normalised_view(text)) is not None:
            views.append(view)
        qp = _depth == 0 and "quoted-printable" in self.ruleset.passes
        if qp and (view := qp_view(text)) is not None:
            views.append(view)
        cells = _depth == 0 and "csv-multiline-cell" in self.ruleset.passes
        if cells and (view := csv_cell_view(text, path)) is not None:
            views.append(view)
        hits = [
            (rule, *_widen(text, to_orig(s), to_orig(e - 1) + 1, right))
            for vtext, to_orig in views
            for rule, s, e in self._match(vtext, path)
        ]
        if hits:  # each kept unless inside a finding so far, in order
            cover = _GrowingCover([f.start for f in found] + [s for _, s, _ in hits])
            for f in found:
                cover.add(f.start, f.end)
            for rule, os_, oe in hits:
                if not cover(os_, oe):
                    found.append(Finding(rule.id, os_, oe, bisect_right(starts, os_), rule.action))
                    cover.add(os_, oe)
        if _depth == 0 and "csv-secret-column" in self.ruleset.passes:
            cover = _Cover((f.start, f.end) for f in found)
            redacted_ = self._redacted
            ids = self.ids if redacted_ else None  # judge what is left beside markers
            for s, e in table_secrets(text, self._deadline, path, ids, breaks):
                if cover(s, e) or (redacted_ and _only_markers(text[s:e], self.ids)):
                    continue  # a marker in a secret column is not a finding
                found.append(Finding("csv-secret-column", s, e, bisect_right(starts, s),
                                     self._table_action))  # fmt: skip
        if _depth == 0 and "kubernetes-secret-data" in self.ruleset.passes:
            cover = _Cover((f.start, f.end) for f in found)
            for s, e in k8s_secret_values(text):
                if cover(s, e) or (self._redacted and _only_markers(text[s:e], self.ids)):
                    continue
                found.append(Finding("kubernetes-secret-data", s, e, bisect_right(starts, s),
                                     "redact"))  # fmt: skip
        if _depth == 0 and "kubernetes-secret-json" in self.ruleset.passes:
            cover = _Cover((f.start, f.end) for f in found)
            for s, e in k8s_secret_json(text, self._deadline):
                if cover(s, e) or (self._redacted and _only_markers(text[s:e], self.ids)):
                    continue
                found.append(Finding("kubernetes-secret-json", s, e, bisect_right(starts, s),
                                     "redact"))  # fmt: skip
        if _depth == 0:  # round 3: decoding views, the rules and table pass run on each
            found += self._deep_views(text, path, starts, found, right)
        if _depth == 0 and "registry-hex" in self.ruleset.passes:
            found += self._registry(text, path, starts, found)
        if _depth < DECODE_DEPTH:
            found += self._decoded(text, path, starts, found, _depth)
            if "base64-block" in self.ruleset.passes:
                found += self._decoded_blocks(text, path, starts, found, _depth)
        if _depth == 0 and "hexdump" in self.ruleset.passes:
            found += self._hexdumps(text, path, starts, found)
        found.sort(key=lambda f: (f.start, f.end, f.rule))
        return found

    def _deep_views(self, text: str, path: str, starts: list[int], found: list[Finding],
                    right: frozenset[str]) -> list[Finding]:  # fmt: skip
        """Findings in decoding views (`scan.views`), mapped back to the original text."""
        on = self.ruleset.passes
        makers = [("markup-view", lambda: V.markup_view(text, path)),
                  ("unfold-view", lambda: V.unfold_view(text)),
                  ("flowed-view", lambda: V.flowed_view(text)),
                  ("wrap-view", lambda: V.wrap_view(text, path)),
                  ("json-escape-view", lambda: V.json_escape_view(text)),
                  ("rtf-view", lambda: V.rtf_view(text)),
                  ("nul-view", lambda: V.nul_view(text)),
                  ("separator-join-view", lambda: V.separator_join_view(self._marked or "")),
                  ("yaml-fold-view", lambda: V.yaml_fold_view(text, path)),
                  ("punct-view", lambda: V.punct_view(text)),
                  ("json-backslash-view", lambda: V.json_backslash_view(text, path)),
                  ("latex-view", lambda: V.latex_view(text, path)),
                  ("emphasis-view", lambda: V.emphasis_view(text)),
                  ("prefix-wrap-view", lambda: V.prefix_wrap_view(text, path))]  # fmt: skip
        out: list[Finding] = []
        big = len(text) > VIEW_MAX  # each view scans the text again: large files skip most
        for name, make in makers:
            if big and name not in _ANY_SIZE:
                continue
            if name not in on or (made := make()) is None:
                continue
            # one cover per view: its windows are apart, so they cannot cover each other
            cover = _Cover((f.start, f.end) for f in [*found, *out])
            for vtext, to_orig in made if isinstance(made, list) else [made]:
                if time.monotonic() > self._deadline:
                    raise ScanTimeout(name)
                out += self._view_findings(name, text, path, starts, cover, right, vtext,
                                           to_orig)  # fmt: skip
        return out

    def _view_findings(self, name: str, text: str, path: str, starts: list[int],
                       cover: _Cover, right: frozenset[str], vtext: str,
                       to_orig: Callable[[int], int]) -> list[Finding]:  # fmt: skip
        """One view's findings, mapped back to the text, outside what is found already."""
        out: list[Finding] = []
        # a view that joins lines would join a table's rows: no table pass on it (and spaces
        # before punctuation, or a window of lines, would change a table)
        tables = name not in ("wrap-view", "unfold-view", "flowed-view", "yaml-fold-view",
                              "punct-view", "json-backslash-view",
                              "separator-join-view", "prefix-wrap-view")  # fmt: skip
        # the join view glues neighbouring texts to a token: widening there would take them
        # in (a label and a row of buttons redacted with the key, #38); widening on the text
        # itself stops where they meet
        widen = name != "separator-join-view"
        for f in self._scan_view(vtext, path, right, tables, widen):
            if f.end <= f.start:
                continue  # a path-only rule: found on the text itself
            os_, oe = _widen(text, to_orig(f.start), to_orig(f.end - 1) + 1, right)
            if oe > os_ and not cover(os_, oe):
                out.append(Finding(f.rule, os_, oe, bisect_right(starts, os_), f.action,
                                   f.encoded))  # fmt: skip
        return out

    def _scan_view(self, vtext: str, path: str, right: frozenset[str] = _RIGHT,
                   tables: bool = True, widen: bool = True) -> list[Finding]:  # fmt: skip
        """The rules, the table pass and base64 decoding on a view's text. Without `widen`,
        a match keeps its own span in the view, and is widened only once mapped back."""
        got = [Finding(rule.id, *(_widen(vtext, s, e, right) if widen else (s, e)), 0,
                       rule.action)
               for rule, s, e in self._match(vtext, path)]  # fmt: skip
        sheet = path.lower().endswith((".xlsx", ".xlsm", ".xls", ".ods", ".numbers"))
        if tables and not sheet and "csv-secret-column" in self.ruleset.passes:
            ids = self.ids if self._redacted else None
            got += [Finding("csv-secret-column", s, e, 0, self._table_action)
                    for s, e in table_secrets(vtext, self._deadline, path, ids)]  # fmt: skip
        return got + self._decoded(vtext, path, _line_starts(vtext), got, 1)

    def _registry(self, text: str, path: str, starts: list[int],
                  found: list[Finding]) -> list[Finding]:  # fmt: skip
        """Hex values of a registry export, decoded and scanned; a finding covers the value."""
        out = []
        cover = _Cover((f.start, f.end) for f in found)
        for s, e, decoded in V.registry_values(text):
            if cover(s, e):
                continue
            inner = self.scan(decoded, path, 1)
            if inner:
                action = "skip-file" if any(f.action == "skip-file" for f in inner) else "redact"
                out.append(Finding(inner[0].rule, s, e, bisect_right(starts, s), action, True))
        return out

    def _match(self, text: str, path: str) -> list[tuple[Rule, int, int]]:
        """(rule, start, end) of every secret span the rules accept in text."""
        lower = text.lower()
        starts = _line_starts(text)
        on = self.ruleset.passes
        out: list[tuple[Rule, int, int]] = []
        for rule in self.rules:
            if time.monotonic() > self._deadline:
                raise ScanTimeout(rule.id)
            if rule.path is not None and not rule.path.search(path):
                continue
            if rule.regex is None:  # a path-only rule: the whole file is the finding
                out.append((rule, 0, 0))
                continue
            if rule.keywords and not any(k in lower for k in rule.keywords):
                continue
            resume = "allowlist-resume" in on
            done: set[int] = set()  # match starts handled: a match from a start is always the same
            got: dict[tuple[int, int], None] = {}  # one finding per span
            for first in rule.regex.finditer(text):
                todo = [first]
                while todo:
                    m = todo.pop()
                    if resume:
                        if m.start() in done:
                            continue  # each start once: no chain is walked twice
                        done.add(m.start())
                    span = self._secret_span(rule, m)
                    if span is None:
                        continue
                    s, e = span
                    if self._redacted and _only_markers(text[s:e], self.ids):
                        continue
                    if self._passes(rule, text[s:e], m, text, starts, path):
                        got[(s, e)] = None
                    elif resume and s > m.start():
                        # A rejected value can hide the next one: in `"msg":"auth","api_key":
                        # "..."` the match is `auth` with the value `api_key`. Also look at the
                        # matches that start inside the rejected one, from its value on; the
                        # usual matches go on as without this.
                        todo += reversed(self._starting_in(rule.regex, text, s, m.end()))
            out += [(rule, s, e) for s, e in got]
        return out

    def _starting_in(self, rx: re.Pattern[str], text: str, a: int, b: int) -> list[re.Match[str]]:
        """Non-overlapping matches of `rx` that start in [a, b), left to right."""
        got, pos = [], a
        while pos < b and (m := rx.search(text, pos)) is not None and m.start() < b:
            if time.monotonic() > self._deadline:
                raise ScanTimeout(rx.pattern[:40])
            got.append(m)
            pos = m.end() if m.end() > m.start() else m.end() + 1
        return got

    @staticmethod
    def _secret_span(rule: Rule, m: re.Match[str]) -> tuple[int, int] | None:
        """The part to redact: the rule's secret group; else the only group that matched;
        else, when several groups matched, the whole match (redacting just the first group
        could leave the rest of the secret behind)."""
        if rule.secret_group:
            g = rule.secret_group
        else:
            hit = [i for i in range(1, m.re.groups + 1) if m.group(i)]
            g = hit[0] if len(hit) == 1 else 0
        s, e = m.span(g)
        return (s, e) if e > s else None

    def _passes(
        self, rule: Rule, secret: str, m: re.Match[str], text: str, starts: list[int], path: str
    ) -> bool:
        if rule.entropy:
            if shannon(secret) <= rule.entropy:
                return False
            if rule.id.startswith("generic") and not any(c.isdigit() for c in secret):
                return False
        g = self.ruleset.global_allow
        if any(r.search(secret) for r in g.regexes):
            return False
        low = secret.lower()
        if any(w in low for w in g.stopwords):
            return False
        if rule.allowlists:
            i = bisect_right(starts, m.start()) - 1
            j = text.find("\n", m.start())
            line = text[starts[i] : j if j != -1 else len(text)]
            words = "stopword-words" in self.ruleset.passes
            if any(_allowed(a, secret, m.group(0), line, path, words) for a in rule.allowlists):
                return False
        return True

    def _decoded(
        self, text: str, path: str, starts: list[int], found: list[Finding], depth: int
    ) -> list[Finding]:
        """Findings for base64 runs whose decoded text contains a secret."""
        out = []
        cover = _Cover((f.start, f.end) for f in found)
        for n, m in enumerate(B64_RUN.finditer(text)):
            if n % 256 == 0 and time.monotonic() > self._deadline:
                raise ScanTimeout("base64")
            s, e = m.span()
            if cover(s, e):
                continue  # already covered by a direct finding
            decoded = _b64_text(m.group(0))
            if decoded is None:
                continue
            inner = self.scan(decoded, path, depth + 1)
            if inner:
                action = "skip-file" if any(f.action == "skip-file" for f in inner) else "redact"
                out.append(Finding(inner[0].rule, s, e, bisect_right(starts, s), action, True))
        return out

    def _decoded_blocks(
        self, text: str, path: str, starts: list[int], found: list[Finding], depth: int
    ) -> list[Finding]:
        """Base64 wrapped over several lines (a key file encoded whole, an email body at 76
        characters), decoded as one: line by line, no single run holds the secret."""
        out = []
        cover = _Cover((f.start, f.end) for f in found)
        for s, e in _b64_blocks(text):
            if time.monotonic() > self._deadline:
                raise ScanTimeout("base64-block")
            if cover(s, e):
                continue
            decoded = _b64_text("".join(text[s:e].split()))
            if decoded is None:
                continue
            inner = self.scan(decoded, path, depth + 1)
            if inner:
                action = "skip-file" if any(f.action == "skip-file" for f in inner) else "redact"
                out.append(Finding(inner[0].rule, s, e, bisect_right(starts, s), action, True))
        return out

    def _hexdumps(
        self, text: str, path: str, starts: list[int], found: list[Finding]
    ) -> list[Finding]:
        """Hexdumps (xxd, hexdump -C, od-like): the bytes rebuilt from the hex columns and
        scanned; a finding covers every dump line its bytes came from."""
        out = []
        cover = _Cover((f.start, f.end) for f in found)
        for lines in _hex_blocks(text):
            if time.monotonic() > self._deadline:
                raise ScanTimeout("hexdump")
            data = bytearray()
            owner: list[int] = []  # for each byte, the index of its line in `lines`
            for k, (_s, _e, hexpart) in enumerate(lines):
                bs = bytes.fromhex(hexpart)
                data += bs
                owner += [k] * len(bs)
            decoded = data.decode("latin-1")
            for f in self.scan(decoded, path, DECODE_DEPTH):
                a = lines[owner[min(f.start, len(owner) - 1)]][0]
                b = lines[owner[min(max(f.end - 1, f.start), len(owner) - 1)]][1]
                if not cover(a, b):
                    out.append(Finding(f.rule, a, b, bisect_right(starts, a), f.action, True))
        return out


# Pieces deleted in the normalised view: a backslash before a quote (JSON inside JSON,
# notebooks), the seam between two string literals joined by an operator or a line
# continuation ("abc" + "def", 'abc' \\n 'def'), and a backslash-newline inside a string.
_ESCAPED_QUOTE = r"""\\(?=["'])"""
# Possessive `*+` throughout: with plain `*` a quote followed by a long run of spaces
# backtracks quadratically (found by the phase 2 review).
_STRING_SEAM = r"""(["'])[ \t]*+(?:(?:\+|\.|&|\|\|)[ \t]*+(?:\\?[ \t]*+\r?\n[ \t]*+)?|\\?[ \t]*+\r?\n[ \t]*+)\1"""  # noqa: E501
_CONTINUATION = r"""\\\r?\n[ \t]*+"""
_VIEW_DELETE = re.compile(f"{_ESCAPED_QUOTE}|{_STRING_SEAM}|{_CONTINUATION}")


def normalised_view(text: str):  # type: ignore[no-untyped-def]
    """The text with escapes and string seams removed, and a map back to original offsets.

    Returns None when there is nothing to remove. Scanning the view finds secrets that are
    split across string literals or hidden behind escaped quotes; each finding maps back to
    an original span that covers every fragment, so redaction removes all of them.
    """
    cuts = [(m.start(), m.end()) for m in _VIEW_DELETE.finditer(text)]
    if not cuts:
        return None
    pieces, seg_view, seg_orig = [], [], []
    pos = vpos = 0
    for a, b in cuts:
        pieces.append(text[pos:a])
        seg_view.append(vpos)
        seg_orig.append(pos)
        vpos += a - pos
        pos = b
    pieces.append(text[pos:])
    seg_view.append(vpos)
    seg_orig.append(pos)

    def to_orig(i: int) -> int:
        k = bisect_right(seg_view, i) - 1
        return seg_orig[k] + (i - seg_view[k])

    return "".join(pieces), to_orig


# A soft break may carry transport padding before the line end (RFC 2045), and decoders
# accept lower-case hex.
_QP_SOFT = re.compile(r"=[ \t]*\r?\n")
_QP_HEX = re.compile(r"=([0-9A-Fa-f]{2})")
# Soft breaks that count as evidence: at the end of a line of 40 or more characters, as
# wrapping makes them; `KEY=` lines of an .env file are short.
_QP_WRAPPED = re.compile(r"(?m)^[^\n]{39,}=[ \t]*\r?\n")


def qp_view(text: str):  # type: ignore[no-untyped-def]
    """Quoted-printable text decoded, with a map back to original offsets, or None.

    Email bodies wrap long lines with a soft break ("=" at the end of a line) and write "="
    as "=3D", which splits or disguises a secret (phase 2: 0 of 8 found in such bodies).
    Only for text that says it is quoted-printable, or has several soft breaks ending long
    lines (one or two look like base64 padding; short ones like `.env` lines); only
    printable ASCII escapes are decoded, so offsets stay simple.
    """
    if "quoted-printable" not in text.lower() and len(_QP_WRAPPED.findall(text)) < 3:
        return None
    pieces: list[str] = []
    seg_view: list[int] = []
    seg_orig: list[int] = []
    pos = vpos = 0
    for m in re.finditer(f"{_QP_SOFT.pattern}|{_QP_HEX.pattern}", text):
        if m.group(1) is not None:
            ch = chr(int(m.group(1), 16))
            if not (" " <= ch <= "~"):
                continue  # not printable ASCII: left as it is
        else:
            ch = ""
        pieces.append(text[pos : m.start()])
        seg_view.append(vpos)
        seg_orig.append(pos)
        vpos += m.start() - pos
        if ch:
            pieces.append(ch)
            seg_view.append(vpos)
            seg_orig.append(m.start())
            vpos += 1
        pos = m.end()
    if not pieces:
        return None
    pieces.append(text[pos:])
    seg_view.append(vpos)
    seg_orig.append(pos)

    def to_orig(i: int) -> int:
        k = bisect_right(seg_view, i) - 1
        return seg_orig[k] + (i - seg_view[k])

    return "".join(pieces), to_orig


# A column whose header names a secret (an export of credentials as CSV, TSV or a sheet):
# its cells are secrets, though no keyword sits on their lines (phase 2: 10 of 22 found).
# Each delimiter is its own pass, so a table in one
# never ends or hides a table in another. Being strict about what is a table keeps code,
# prose and data out; cells under a password column are then judged generously, because a
# leak costs more than a false alarm (which only makes a file local-only).
_PASSWORD_WORDS = (
    r"passwords?|passwd|pwd|pw|passphrase|passwort|kennwort|mot de passe|contrase(?:ñ|n)a"
    r"|senha|wachtwoord|l(?:ö|o)senord|salasana|has(?:ł|l)o|parola|пароль|パスワード|密码|비밀번호"
    r"|(?:ş|s)ifre|heslo|passord|adgangskode|密碼|mật khẩu|kata sandi|jelsz(?:ó|o)"
    r"|parol(?:ă|a)|lozinka|geslo|slapta(?:ž|z)odis|parool|κωδικός"
    r"|network security key|wi-?fi (?:code|key)|psk|pre-?shared key"
)
_PREFIX = r"(?:[a-z0-9]+[_ .-]){0,4}"  # four words at most: a name, not a sentence
_PREFIX_WEAK = r"(?:[a-z0-9]+[_ .-]){0,2}"  # `credentials` ends sentences: two words
_SUFFIX = r"(?:[_ .-]?\d{1,4})?"
_PASSWORD_COLUMN = re.compile(f"{_PREFIX}(?:{_PASSWORD_WORDS}){_SUFFIX}")
_STRONG_COLUMN = re.compile(
    f"{_PREFIX}(?:secret|passcode|totp|otp[_ -]?(?:secret|seed|url|uri)|otpauth|apikey"
    r"|recovery[_ -]?(?:keys?|codes?)|backup[_ -]?codes?|seed[_ -]?phrase|mnemonic"
    r"|totp[_ -]?(?:key|secret|seed)"
    r"|client[_ -]?secret"
    r"|(?:api|access|private|secret|signing|master|encryption|wpa2?|wifi|wi-fi|network|wep)"
    r"[_ -]?key|(?:auth|access|refresh|api|bearer|session|bot|oauth|personal[_ -]access)"
    f"[_ -]?token){_SUFFIX}"
)
_WEAK_COLUMN = re.compile(f"{_PREFIX_WEAK}(?:tokens?|credentials?|creds){_SUFFIX}")
# `Pass` is a password beside an account column, else a test or exam result
_PASS_COLUMN = re.compile(f"{_PREFIX}pass{_SUFFIX}")
_ACCOUNT_COLUMN = re.compile(
    f"{_PREFIX}(?:user(?: ?name)?|login|e-?mail|site|web ?site|url|uri|account|host(?: ?name)?"
    r"|server)"
)
_HEADER_HINT = re.compile(
    r"(?i)pass|pwd|pw|secret|token|key|cred|totp|otp|\bpin\b|kennw|mot de passe|contrase"
    r"|senha|wachtw"
    r"|senord|salasana|hasł|haslo|parol|пароль|パスワード|密码|비밀번호|ifre|heslo|kode"
    r"|密碼|khẩu|sandi|jelsz|lozinka|geslo|odis|κωδικ|psk|code|mnemonic|seed"
)
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_TRAILER = re.compile(r"[(\[][^()\[\]]*[)\]]$")  # `(old)`, `[old]`; names are capped first
_LOOSE_NAME = re.compile(r"\w[\w .\-/#()?%&':*+]{0,63}")
_STRICT_NAME = re.compile(r"[A-Za-z_][\w .\-]{0,47}")
_CODE_LEAD = re.compile(r"(?:(?:--|#|//|;)[ \t]*|[^\n]*?(?:\"\"\"|\'\'\'))?")
_NUMERIC = re.compile(r"[$€£]?[\d.,%+-]*\d[\d.,%+-]*")  # a number, amount or percentage
_LABEL = re.compile(r"[A-Za-z][\w .\-]{0,39}")  # a field's label in a label-value export
_CODE_ANY = re.compile(r"[={}]")  # in any file, a cell with these is not a column name
_CODE_ONLY = re.compile(  # in a code file, neither are these (not elsewhere)
    r"[\[\]]|^\s*(?:(?:select|insert|update|delete|create|alter|SELECT|INSERT|UPDATE|DELETE"
    r"|CREATE|ALTER)\s|(?:int|long|short|char|byte|bool|boolean|float|double|string|String"
    r"|void|var|let|const|final|static|private|public|protected|def|func|function|fn|return"
    r"|import|export)\s+[\w$]|from\s+[\w.]+\s+import\s)"
)
_DELIMITERS = (",", "\t", ";", "|")
_PLACEHOLDER = re.compile(
    r"^(?:(?i:\*+|x+|\.+|:?-+:?|n/?a|none|null|nil|empty|unset|not set|unknown|redacted"
    r"|hidden|tbd|todo|change[_ -]?me|replace[_ -]?me|example.*|<.*>|\{\{.*\}\})"
    r"|\$\{\w+\}|\$[A-Z_][A-Z0-9_]*)$"  # ${VAR} or $VAR in capitals, not $unshine2024
)
# Words that sit in a password column without being passwords
_NOT_A_PASSWORD = re.compile(
    r"(?i)^(?:required|optional|unchanged|same(?: as (?:above|before))?|see (?:the )?(?:notes?"
    r"|above|below|vault|email|wiki|keychain|1password|bitwarden|lastpass|password manager)"
    r"|(?:stored )?in (?:the )?(?:vault|keychain)|vault|keychain|sso"
    r"|saml|oauth2?|ldap|kerberos|not (?:required|needed|set|used|applicable|shared)|blank"
    r"|yes|no|true|false|reset|expired|locked|disabled|enabled|active|inactive|changed"
    r"|rotated|pending|ask(?: (?:me|admin|it|ops))?|1password|bitwarden|lastpass"
    r"|keepass(?:xc)?|dashlane|(?:google |microsoft )?authenticator(?: app)?|sms|email"
    r"|hardware (?:key|token)|yubikey|passkey|biometrics?|face ?id|touch ?id|default"
    r"|generated|random|temporary|temp|forgotten|forgot|unknown)$"
)
_SYMBOL = re.compile(r"[0-9!@#$%^&*+=?~]")
# A marker and what is glued to its right: redaction widens over `,` and `:`, so in
# redacted output the next cells' text can follow a marker directly
_MARKER_SPILL = re.compile(r"\[REDACTED:([a-z0-9._-]+)\](?:[^\sA-Za-z0-9]\S*)?")
_WORDISH = re.compile(r"[A-Za-z_.\-]+")
_STRIP = " \t\r\"'\ufeff"
_SHEETS = (".xlsx", ".xlsm", ".xls", ".ods", ".numbers")  # extracted text: no CSV quoting
# Exports, where a password column's words of letters are passwords at once; in documents
# (a packing list, a password-policy table) only beside a password with digits or symbols
_EXPORTS = (".csv", ".tsv", ".txt", ".tab", ".psv", *_SHEETS)
_TABLES = (".csv", ".tsv", ".tab", ".psv")  # files that are one table: it ends with the file
_CODE = (".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".java", ".kt", ".kts", ".go",
         ".rs", ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift", ".scala",
         ".sql", ".sh", ".bash", ".zsh", ".ps1", ".pl", ".lua", ".dart", ".groovy")  # fmt: skip
_MAX_QUOTED_LINES = 200  # a quoted cell spanning more lines is a stray quote
_MAX_GAP = 6  # in a document, lines without the delimiter a table crosses (a cell's lines)
# documents whose tables are extracted as tab-separated rows: like sheets, a row's further
# lines (line breaks inside a cell) have no tab, and do not end the table
_TABBED_DOCS = (".docx", ".odt", ".rtf", ".pages", ".pptx", ".odp")
_MAX_NAME = 200  # a longer cell is not a column name (and keeps the name regexes linear)


def _cells(text: str, a: int, b: int, d: str, quotes: bool = True, width: int = 0,
           deadline: float = float("inf")) -> tuple[list[tuple[int, int]], int, bool]:  # fmt: skip
    """The cells of the record starting at a, split at d outside double quotes (at every d,
    if not `quotes`). The record ends at the first line break outside quotes, or at b.
    Returns the cells' spans, where the record ends, and whether it ended inside a quote
    that looks stray: never closed, open for too many lines, or open across a line that
    has a whole row's delimiters (`width` cells). One pass: linear time."""
    nl = text.find("\n", a, b)
    line_end = b if nl == -1 else nl
    if not quotes or '"' not in text[a:line_end]:
        out, pos = [], a
        for part in text[a:line_end].split(d):
            out.append((pos, pos + len(part)))
            pos += len(part) + 1
        return out, line_end, False
    out, cell, inside, fresh, broken, i = [], a, False, True, 0, a
    while i < b:
        if (i - a) & 0xFFFF == 0xFFFF and time.monotonic() > deadline:
            raise ScanTimeout("csv-secret-column")
        c = text[i]
        if c == '"':
            if inside:
                if i + 1 < b and text[i + 1] == '"':
                    i += 2  # an escaped quote
                    continue
                inside = False
            elif fresh:
                inside = True  # a quote opens a cell only at its start
            fresh = False
        elif inside:
            if c == "\n":
                broken += 1
                nxt = text.find("\n", i + 1, b)
                ahead = text[i + 1 : b if nxt == -1 else nxt]
                if broken > _MAX_QUOTED_LINES or (width >= 2 and ahead.count(d) >= width - 1):
                    out.append((cell, i))
                    return out, i, True
        elif c == d:
            out.append((cell, i))
            cell, fresh = i + 1, True
        elif c == "\n":
            break
        elif c != " ":
            fresh = False
        i += 1
    out.append((cell, i))
    return out, i, inside


def _trim(text: str, a: int, b: int) -> tuple[int, int]:
    while a < b and text[a] in _STRIP:
        a += 1
    while b > a and text[b - 1] in _STRIP:
        b -= 1
    return a, b


def _column(raw: str) -> str:
    """A header cell as a name: no quotes, backticks, `*` or `:`, no trailing `(old)` or
    `[old]`, camelCase split, lower case; empty if too long to be a name."""
    name = raw.strip(_STRIP + "*:`")
    if len(name) > _MAX_NAME:
        return ""
    name = _TRAILER.sub("", name).strip(_STRIP + "*:`")
    return _CAMEL.sub(" ", name).lower().strip()


def _tokenish_name(raw: str, name: str) -> bool:
    """A cell that only splits into a secret-sounding name by accident: a token of twelve or
    more characters with three digits, or a name with two or more words holding digits."""
    word = raw.strip(_STRIP)
    if (
        len(word) >= 12
        and not any(c.isspace() for c in word)
        and (sum(map(str.isdigit, word)) >= 3 or len(_CAMEL.split(word)) >= 4)
    ):
        return True  # a token, or a random ID that splits into name-like words
    return sum(any(c.isdigit() for c in w) for w in name.split()) >= 2


def _kind(name: str, names: list[str]) -> str | None:
    """'password', 'strong' or 'weak' for a secret-naming column, else None. A name with
    alternatives (`Password / PIN`) is judged by its strongest part."""
    best = None
    for part in [name, *(p.strip() for p in name.split(" / ") if " / " in name)]:
        if _PASSWORD_COLUMN.fullmatch(part):
            return "password"
        if _PASS_COLUMN.fullmatch(part):
            if any(_ACCOUNT_COLUMN.fullmatch(n) for n in names):
                return "password"
            best = best or "weak"
        elif _STRONG_COLUMN.fullmatch(part):
            best = "strong"
        elif _WEAK_COLUMN.fullmatch(part):
            best = best or "weak"
    return best


def _header(line: str, d: str, quotes: bool, code: bool,
            deadline: float = float("inf")) -> tuple[dict[int, str], int] | None:  # fmt: skip
    """({column: kind}, width) when the line is a table header, split at d, naming a secret
    column: two or more named cells, the secret ones clean names, most others (all, in a
    code file) column-name-like, and nothing that reads as code."""
    if d not in line or (d != "\t" and "\t" in line) or not _HEADER_HINT.search(line):
        return None
    if "[REDACTED:" in line:  # a redacted row, never a header: both scans must agree
        return None
    if code:
        if line[:1] in " \t":  # indented: an argument list, not a CSV in a heredoc
            return None
        # a table in a comment (`-- user | password`) or opened with a string literal
        # (`USERS = """user,password`): the header is what follows
        lead = _CODE_LEAD.match(line).end()
        line = line[lead:]
    spans = _cells(line, 0, len(line), d, quotes, deadline=deadline)[0]
    if not 2 <= len(spans) <= 60:
        return None
    raw = [line[a:b] for a, b in spans]
    if any(len(r) > _MAX_NAME or _CODE_ANY.search(r) or r.count("(") != r.count(")")
           or "://" in r or "@" in r for r in raw):  # fmt: skip
        return None  # code, or a data row (a URL, an address)
    if code and any(_CODE_ONLY.search(r) for r in raw):
        return None
    names = [_column(r) for r in raw]
    if sum(1 for n in names if n) < 2:
        return None
    cols: dict[int, str] = {}
    for k, n in enumerate(names):
        if n and not _tokenish_name(raw[k], n) and (kind := _kind(n, names)) is not None:
            cols[k] = kind
    if not cols:
        return None
    others = [n for k, n in enumerate(names) if n and k not in cols]
    if code:
        if any(q in r for r in raw for q in "\"'") or not all(map(_STRICT_NAME.fullmatch, others)):
            return None
        if not lead and any(" " in n for n in names):
            return None  # prose in a docstring or comment, not a CSV header
    elif 2 * sum(bool(_LOOSE_NAME.fullmatch(n) and n.count(" ") <= 7) for n in others) < len(
        others
    ):
        return None
    return cols, len(spans)


def _names_a_column(value: str) -> bool:
    """A cell holding a secret column's name (`Password`, `API key`), not a value. Digits
    rule it out: `Xay59G5zfPW` splits on camelCase into words ending in `pw`."""
    return (len(value) <= 40 and not any(c.isdigit() for c in value)
            and _kind(_column(value), []) is not None)  # fmt: skip


def _secretish(run: str) -> bool:
    """A word that looks like a password or token: letters and digits or symbols."""
    return len(run) >= 6 and bool(_SYMBOL.search(run)) and any(c.isalpha() for c in run)


def _judge(value: str, kind: str, export: bool = True) -> str:
    """'yes', 'maybe' (letters only: a secret only if others in its column are) or 'no',
    for a cell under a password or other strong column."""
    if not value or _PLACEHOLDER.match(value):
        return "no"
    if _names_a_column(value):
        return "no"  # a column's name in a cell (`Password`, `API key`): a glossary, a menu
    note = len(value) > 64 or value.count(" ") > 7
    if note:  # a note, unless a word in it looks like a secret
        return "yes" if any(map(_secretish, value.split())) else "no"
    if kind == "password":  # generous: passphrases have spaces, letters-only passwords exist
        if _NOT_A_PASSWORD.match(value):
            return "no"
        if _SYMBOL.search(value):
            return "yes" if len(value) >= 4 else "no"
        return ("yes" if export else "maybe") if len(value) >= 6 else "no"
    if any(c.isspace() for c in value):
        return "yes" if any(map(_secretish, value.split())) else "no"
    if len(value) >= 6 and _SYMBOL.search(value):
        return "yes"
    return "maybe" if len(value) >= 10 and _WORDISH.fullmatch(value) else "no"


def _weak_runs(text: str, a: int, b: int) -> list[tuple[int, int]]:
    """Token-like runs in a cell under a weak column name: ten or more characters, letters
    with digits or symbols, not low in entropy (`SARS-CoV-2`, `RN BSN 2019` are not), not
    a URL. Runs split only at spaces and delimiters, so a token with `/` stays whole."""
    out = []
    for m in re.finditer(r"[^\s,;|]+", text[a:b]):
        run = m.group(0).strip(_STRIP)
        if (len(run) >= 10 and _secretish(run) and shannon(run) > 3.0 and "://" not in run
                and not _PLACEHOLDER.match(run)):  # fmt: skip
            s = a + m.start() + m.group(0).index(run)
            out.append((s, s + len(run)))
    return out


_RANK = {"weak": 0, "strong": 1, "password": 2}


def table_secrets(text: str, deadline: float = float("inf"), path: str = "",
                  ids: frozenset[str] | None = None,
                  breaks: tuple[int, ...] = ()) -> list[tuple[int, int]]:  # fmt: skip
    """Spans of secrets in cells under a secret-named column, in delimited tables.

    Each delimiter is a pass of its own. A table never ends on a guess:
    a header row inside it adds its columns to the table's (or replaces a title row's), so
    a wrong guess costs redaction in a file that has secrets anyway, never a leak. A table
    ends only where the text says so: at `breaks` (offsets where the next sheet starts), at
    the end of an export (csv, tsv), and in other files after four lines without the
    delimiter. In a sheet's text, lines without a tab (a cell's further lines) are skipped
    and quotes are literal. Rows with fewer cells (a sheet drops empty trailing cells) or
    extra ones (the secret's cell then reaches over the extras) are judged as they are.
    With `ids` (scanning redacted output), markers of those rules are ignored and what is
    left beside them is judged; a row a marker has shortened is judged only before the
    marker. Raises ScanTimeout past the deadline.
    """
    low = path.lower()
    sheet, code = low.endswith(_SHEETS), low.endswith(_CODE)
    quotes, export = not sheet, not low or low.endswith(_EXPORTS)
    whole = not low or low.endswith(_TABLES)  # a table ends only with the file
    lines = text.split("\n")
    offs, pos = [], 0
    for line in lines:
        offs.append(pos)
        pos += len(line) + 1
    ends = {bisect_right(offs, b) - 1 for b in breaks if 0 < b < pos}
    found: set[tuple[int, int]] = set()
    n, steps = len(lines), 0

    def tick() -> None:
        nonlocal steps
        steps += 1
        if steps & 0xFF == 1 and time.monotonic() > deadline:
            raise ScanTimeout("csv-secret-column")

    def value(a: int, b: int) -> tuple[int, int, str]:
        a, b = _trim(text, a, b)
        v = text[a:b]
        if ids is not None and "[REDACTED:" in v:  # what a redaction left in the cell
            v = _MARKER_SPILL.sub(lambda m: " " if m.group(1) in ids else m.group(0), v)
            v = v.strip(_STRIP)
        return a, b, v

    def count_named(line: str, d: str) -> int:
        return sum(1 for c in line.split(d) if c.strip(_STRIP))

    def starts_block(i: int, d: str) -> bool:
        """A header starts a block of rows: the first line, after a blank line or a line
        without the delimiter, where a sheet starts, or under a title row with fewer named
        cells. A data row in the middle of a large CSV is not one."""
        if i == 0 or i in ends:
            return True
        before = lines[i - 1]
        return (d not in before or not before.strip(d + _STRIP)
                or count_named(before, d) < count_named(lines[i], d))  # fmt: skip

    for d in _DELIMITERS:
        if d not in text:
            continue
        if whole:  # label and value pairs in a row (Enpass: Title,Field,Value,Field,Value)
            first = {c.strip(_STRIP).lower() for c in lines[0].split(d)}
            labelled = {"field", "value"} <= first  # the file says it is label and value
            for j, line in enumerate(lines):
                tick()
                if d not in line or not _HEADER_HINT.search(line):
                    continue
                spans = _cells(text, offs[j], offs[j] + len(line), d, quotes)[0]
                raw = [text[a:b].strip(_STRIP) for a, b in spans]
                named = [_column(c) if _LABEL.fullmatch(c) else "" for c in raw]
                if not labelled and not any(_ACCOUNT_COLUMN.fullmatch(named[k]) and raw[k + 1]
                           and not _ACCOUNT_COLUMN.fullmatch(named[k + 1])
                           and _kind(named[k + 1], []) is None
                           for k in range(len(raw) - 1)):  # fmt: skip
                    continue  # a label-value row names the account: `Username, ann`
                for k in range(len(spans) - 1):
                    label = text[spans[k][0] : spans[k][1]].strip(_STRIP)
                    if not _LABEL.fullmatch(label):  # exactly a label: `Password`, `PIN`
                        continue
                    name = _column(label)
                    kind = _kind(name, []) if len(name.split()) <= 2 else None
                    if name in ("pin", "pin code", "passcode"):
                        kind = "password"
                    if kind in ("password", "strong"):
                        a, b, v = value(*spans[k + 1])
                        if "://" in v:
                            continue  # a URL beside a title like `Mail password`
                        if _secretish(v) or (kind == "password" and _judge(v, kind) == "yes"):
                            found.add((a, b))
        i = 0
        while i < n:
            tick()
            head = _header(lines[i], d, quotes, code, deadline) if starts_block(i, d) else None
            if head is None:
                i += 1
                continue
            kinds, width = dict(head[0]), head[1]
            tabbed = d == "\t" and low.endswith(_TABBED_DOCS)
            named_at = [k for k, c in enumerate(lines[i].split(d)) if c.strip(_STRIP)]
            head_named = len(named_at)
            # a title row (`Passwords, Updated 2026-01-01`): two named cells at most, the
            # first a secret name; any other header is merged into, never replaced
            title = head_named <= 2 and named_at[:1] == [min(kinds)]
            cells: dict[int, list[tuple[int, int, str]]] = {}
            j, gap, rows = i + 1, 0, 0
            while j < n and j not in ends:
                tick()
                line = lines[j]
                blank = not line.strip(d + _STRIP)
                if blank or (d not in line and not (quotes and line.lstrip().startswith('"'))):
                    gap += 1  # blank, only delimiters, or no delimiter
                    if not (sheet or whole or tabbed) and gap > _MAX_GAP:
                        break
                    j += 1
                    continue
                gap = 0
                far = min(j + _MAX_QUOTED_LINES, n - 1)
                spans, end, stray = _cells(text, offs[j], offs[far] + len(lines[far]), d,
                                           quotes, width, deadline)  # fmt: skip
                if stray:  # read the line alone, quotes literal
                    spans, end, _ = _cells(text, offs[j], offs[j] + len(line), d, False)
                new = (_header(line, d, quotes, code, deadline)
                       if end <= offs[j] + len(line) else None)  # fmt: skip
                if new is not None and any(k < len(spans) and _secretish(value(*spans[k])[2])
                                           for k in kinds):  # fmt: skip
                    new = None  # a data row holding a secret, though it reads like a header
                named = sum(1 for a, b in spans if text[a:b].strip(_STRIP))
                if new is not None and any(_NUMERIC.fullmatch(text[a:b].strip(_STRIP))
                                           for a, b in spans):  # fmt: skip
                    new = None  # a data row of numbers, not another table's header
                if new is not None and rows == 0 and named > head_named and title:
                    kinds, width = dict(new[0]), new[1]  # the first header was a title row
                    head_named = named
                    j = bisect_right(offs, end)
                    continue
                if new is not None:  # another table: its columns join this one's
                    for k, kind in new[0].items():
                        if _RANK[kind] > _RANK.get(kinds.get(k, ""), -1):
                            kinds[k] = kind
                rows += 1
                extra = max(0, len(spans) - width)
                limit = len(spans)
                if ids is not None and len(spans) < width:  # widened over a delimiter
                    marked = [k for k, (a, b) in enumerate(spans) if "[REDACTED:" in text[a:b]]
                    limit = marked[0] if marked else limit
                for k in kinds:
                    if k < limit:
                        last_k = min(k + extra, len(spans) - 1)
                        cells.setdefault(k, []).append(value(spans[k][0], spans[last_k][1]))
                j = bisect_right(offs, end)  # the width stays: both scans parse rows alike
            for k, got in cells.items():
                kind = "strong" if code and kinds[k] == "password" else kinds[k]
                if kind == "weak":
                    for a, b, v in got:
                        if ids is not None and v != text[a:b]:  # judged beside markers
                            if _weak_runs(v, 0, len(v)):
                                found.add((a, b))
                        else:
                            found.update(_weak_runs(text, a, b))
                    continue
                judged = [(a, b, _judge(v, kind, export)) for a, b, v in got]
                sure = any(c == "yes" for _, _, c in judged)
                found.update((a, b) for a, b, c in judged if c == "yes" or (c == "maybe" and sure))
            i = max(j, i + 1)
    return sorted(found)


_B64_LINE = re.compile(r"[ \t]*[A-Za-z0-9+/_-]{16,}={0,2}[ \t]*")


def _b64_blocks(text: str) -> list[tuple[int, int]]:
    """Spans of two or more consecutive lines made only of base64, the first of them at
    least 40 characters (wrapped base64, not a column of short ids)."""
    out: list[tuple[int, int]] = []
    pos, run_start, run_end, run_n, first_len = 0, -1, -1, 0, 0
    for line in text.split("\n"):
        end = pos + len(line)
        if _B64_LINE.fullmatch(line):
            if run_n == 0:
                run_start, first_len = pos, len(line.strip())
            run_end, run_n = end, run_n + 1
        else:
            if run_n >= 2 and first_len >= 40:
                out.append((run_start, run_end))
            run_n = 0
        pos = end + 1
    if run_n >= 2 and first_len >= 40:
        out.append((run_start, run_end))
    return out


# An offset, then hex bytes (in pairs, single-spaced or in groups), maybe an ASCII column.
_HEX_LINE = re.compile(
    r"[0-9A-Fa-f]{6,16}:?[ \t]+((?:[0-9A-Fa-f]{2}(?:[ \t]{1,2}|$)|[0-9A-Fa-f]{4}[ \t]){1,20})"
)


def _hex_blocks(text: str) -> list[list[tuple[int, int, str]]]:
    """Runs of three or more hexdump lines: (line start, line end, the hex digits)."""
    blocks: list[list[tuple[int, int, str]]] = []
    cur: list[tuple[int, int, str]] = []
    pos = 0
    for line in text.split("\n"):
        end = pos + len(line)
        m = _HEX_LINE.match(line)
        if m:
            cur.append((pos, end, "".join(m.group(1).split())))
        elif line.strip() != "*":  # "*": hexdump's "same as above"
            if len(cur) >= 3:
                blocks.append(cur)
            cur = []
        pos = end + 1
    if len(cur) >= 3:
        blocks.append(cur)
    return blocks


_K8S_DOC = re.compile(r"^---[ \t]*(?:#.*)?$", re.M)
_K8S_KIND = re.compile(r"""^kind:[ \t]*["']?Secret["']?[ \t]*(?:#.*)?$""", re.M)
_K8S_DATA = re.compile(r"^([ \t]*)(?:data|stringData):[ \t]*(?:#.*)?$", re.M)
_K8S_ENTRY = re.compile(r"""[\w.-]+[ \t]*:[ \t]*(.*?)[ \t]*$""")
_K8S_EMPTY = re.compile(r"""^(?:""|''|\{\{.*\}\}|["']\{\{.*\}\}["'])?$""")


def k8s_secret_values(text: str) -> list[tuple[int, int]]:
    """Every value under `data:` or `stringData:` in a Kubernetes `kind: Secret` document
   . The vendored rule catches only a document's first value."""
    if "Secret" not in text or "ata:" not in text:
        return []
    out: list[tuple[int, int]] = []
    bounds = [0, *(m.end() for m in _K8S_DOC.finditer(text)), len(text)]
    for a, b in zip(bounds, bounds[1:], strict=False):
        if not _K8S_KIND.search(text, a, b):
            continue
        for d in _K8S_DATA.finditer(text, a, b):
            base = len(d.group(1).expandtabs())
            block = -1  # the indentation of a key whose value is a block scalar
            pos = d.end() + 1
            while pos < b:
                end = text.find("\n", pos, b)
                end = b if end < 0 else end
                line = text[pos:end].rstrip("\r")
                stripped = line.strip()
                indent = len(line) - len(line.lstrip(" \t"))
                if stripped and indent <= base:
                    break
                if stripped and not stripped.startswith("#"):
                    if 0 <= block < indent:
                        out.append((pos + indent, pos + len(line.rstrip())))
                    elif (m := _K8S_ENTRY.match(line, indent)) is not None:
                        v = m.group(1)
                        block = indent if v[:1] in ("|", ">") else -1
                        if block < 0 and not _K8S_EMPTY.match(v) and not v.startswith("#"):
                            out.append((pos + m.start(1), pos + m.end(1)))
                pos = end + 1
    return out


K8S_JSON_MAX = 5_000_000  # characters: a larger file is not searched for JSON Secrets
_K8S_KIND_JSON = re.compile(r'"kind"\s*:\s*"Secret"')
_JSON_STRING = re.compile(r'"(?:[^"\\\n]|\\.)*"')


class _Obj(dict):  # type: ignore[type-arg]
    """A JSON object that keeps every pair, duplicate keys included (json keeps the last)."""

    pairs: list[tuple[str, Any]]


def _keep_pairs(items: list[tuple[str, Any]]) -> _Obj:
    obj = _Obj(items)
    obj.pairs = items
    return obj


def k8s_secret_json(text: str, deadline: float = float("inf")) -> list[tuple[int, int]]:
    """Every value under `data` or `stringData` of a Kubernetes Secret written as JSON (round
    6): a whole file (`kubectl get secret -o json`, a List), a document among others (NDJSON,
    a paste in notes or a log, the last-applied copy in a YAML file), or one inside a string
    (kubectl's last-applied-configuration annotation). Each value is found at every place it
    occurs as a JSON string, plain or escaped inside another string."""
    if "Secret" not in text or "ata" not in text or len(text) > K8S_JSON_MAX:
        return []  # (an escaped copy reads `\"Secret\"`: no quotes around the word)
    values: set[str] = set()
    regions = []
    for doc, start, end in _json_docs(text, deadline):
        _secret_values(doc, values, deadline)
        regions.append((start, end))
    if not values:
        return []
    plain = {json.dumps(v, ensure_ascii=False)[1:-1] for v in values}
    out: list[tuple[int, int]] = []
    # string literals are paired from each document's own start, so a stray quote before it
    # on the line cannot shift them
    for start, end in regions:
        for n, m in enumerate(_JSON_STRING.finditer(text, start, end)):
            if n % 4096 == 0 and time.monotonic() > deadline:
                raise ScanTimeout("kubernetes-secret-json")
            a, b = m.start() + 1, m.end() - 1
            inner = text[a:b]
            if inner in plain or ("\\" in inner and _decoded(m.group(0)) in values):
                out.append((a, b))
            elif '\\"' in inner and _holds(m.group(0), values):
                out.append((a, b))  # JSON inside a string (kubectl's annotation): all of it
    return sorted(set(out))


def _holds(literal: str, values: set[str]) -> bool:
    """Whether a string holding JSON has one of `values` among its own strings, however
    its writer escaped them (kubectl, in Go, writes `&` as `\\u0026`)."""
    inner = _decoded(literal)
    return inner is not None and any(
        _decoded(m.group(0)) in values for m in _JSON_STRING.finditer(inner))  # fmt: skip


def _decoded(literal: str) -> str | None:
    try:
        value = json.loads(literal)
    except ValueError:
        return None
    return value if isinstance(value, str) else None


_JSONC_COMMA = re.compile(r",(\s*[}\]])")
_JSONC_LINE_COMMENT = re.compile(r"^[ \t]*//[^\n]*$", re.M)


def _json_docs(text: str, deadline: float) -> Iterable[tuple[Any, int, int]]:
    """The whole text as JSON, else each JSON object that holds a `"kind": "Secret"`: from
    the nearest opening brace before it that parses into an object reaching past it. Each
    comes with its span. An object that breaks past the match stops the walk (the ones
    around it break too), and parsing stops at four times the text's length."""
    for attempt in (text, _JSONC_COMMA.sub(r"\1", _JSONC_LINE_COMMENT.sub("", text))):
        try:  # as JSON, then as JSONC (comment lines and trailing commas)
            yield json.loads(attempt, object_pairs_hook=_keep_pairs), 0, len(text)
            return
        except (ValueError, RecursionError):
            pass
    decoder = json.JSONDecoder(object_pairs_hook=_keep_pairs)
    budget = 4 * len(text) + 100_000
    done = -1
    for m in _K8S_KIND_JSON.finditer(text):
        if m.start() < done:
            continue  # inside an object already read
        pos = m.start()
        for _ in range(64):
            pos = text.rfind("{", 0, pos)
            if pos < 0 or m.start() - pos > 200_000 or budget < 0:
                break
            if time.monotonic() > deadline:
                raise ScanTimeout("kubernetes-secret-json")
            try:
                doc, end = decoder.raw_decode(text, pos)
            except json.JSONDecodeError as e:
                budget -= max(e.pos - pos, 1)
                if e.pos > m.start():
                    break
                continue
            except RecursionError:
                continue
            budget -= end - pos
            if end > m.start():
                yield doc, pos, end
                done = end
                break


def _secret_values(doc: Any, values: set[str], deadline: float) -> None:
    """Values (four characters or more) of every Secret in a parsed JSON document, and of
    Secrets in strings that are JSON themselves (an annotation), two levels deep."""
    stack: list[tuple[Any, int]] = [(doc, 0)]
    seen = 0
    while stack:
        node, depth = stack.pop()
        seen += 1
        if seen % 4096 == 0 and time.monotonic() > deadline:
            raise ScanTimeout("kubernetes-secret-json")
        if isinstance(node, dict):
            if node.get("kind") == "Secret":
                for field in ("data", "stringData"):
                    obj = node.get(field)
                    pairs = getattr(obj, "pairs", None)
                    if pairs is None:
                        pairs = list(obj.items()) if isinstance(obj, dict) else []
                    values.update(v for _, v in pairs if isinstance(v, str) and len(v) >= 4)
            stack += [(v, depth) for v in node.values()]
        elif isinstance(node, list):
            stack += [(v, depth) for v in node]
        elif (isinstance(node, str) and depth < 2 and '"Secret"' in node
              and node.lstrip().startswith("{")):  # fmt: skip
            with contextlib.suppress(ValueError, RecursionError):
                stack.append((json.loads(node, object_pairs_hook=_keep_pairs), depth + 1))


def csv_cell_view(text: str, path: str):  # type: ignore[no-untyped-def]
    """A CSV or TSV file with line breaks inside quoted cells removed where they split a
    token (`views.joins`), and a map back to original offsets, or None. A token wrapped
    inside a cell is then seen whole."""
    if not path.lower().endswith((".csv", ".tsv")) or '"' not in text:
        return None
    cuts: list[tuple[int, int]] = []
    inside = False
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            if inside and i + 1 < n and text[i + 1] == '"':
                i += 2  # an escaped quote
                continue
            inside = not inside
        elif inside and c in "\r\n":
            j = i
            while j < n and text[j] in "\r\n \t":
                j += 1
            # only a break inside a token: joining prose lines would make tokens of words
            if V.joins(text[max(0, i - 256) : i], text[j : j + 256]):
                cuts.append((i, j))
            i = j
            continue
        i += 1
    if not cuts:
        return None
    pieces, seg_view, seg_orig = [], [], []
    pos = vpos = 0
    for a, b in cuts:
        pieces.append(text[pos:a])
        seg_view.append(vpos)
        seg_orig.append(pos)
        vpos += a - pos
        pos = b
    pieces.append(text[pos:])
    seg_view.append(vpos)
    seg_orig.append(pos)

    def to_orig(k: int) -> int:
        idx = bisect_right(seg_view, k) - 1
        return seg_orig[idx] + (k - seg_view[idx])

    return "".join(pieces), to_orig


def _b64_text(run: str) -> str | None:
    """Decode a base64 (standard or URL-safe) run into mostly printable text, else None."""
    raw = run.rstrip("=")
    padded = raw + "=" * (-len(raw) % 4)
    for alt in (None, b"-_"):
        try:
            data = base64.b64decode(padded, altchars=alt, validate=True)
        except (binascii.Error, ValueError):
            continue
        try:
            s = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if s and sum(c.isprintable() or c in "\n\r\t" for c in s) / len(s) >= PRINTABLE_SHARE:
            return s
    return None


REDACTION = "[REDACTED:{rule}]"


MARKER = re.compile(r"\[REDACTED:[a-z0-9._-]+\]")
_BARE_TAG = re.compile(r"</?[A-Za-z][\w:.-]{0,60}/?>")


def _only_markers(secret: str, ids: frozenset[str]) -> bool:
    """Whether a matched secret is nothing but redaction markers of known rules (and quotes
    or spaces), so scanning redacted text again finds nothing new. A marker in front of a
    real secret leaves letters or digits behind and still counts, and so does text that
    only looks like a marker (`[REDACTED:hunter2secret9]`)."""
    names = [m.group(0)[10:-1] for m in MARKER.finditer(secret)]
    # A bare tag a redaction left beside its marker (`[REDACTED:x]</emphasis>`) is markup,
    # not a secret; a tag with attributes could hold one, so it still counts.
    rest = _BARE_TAG.sub("", MARKER.sub("", secret))
    return bool(names) and all(n in ids for n in names) and not any(c.isalnum() for c in rest)


def redact(text: str, findings: list[Finding]) -> str:
    """Replace every redact-finding span with `[REDACTED:<rule>]`, keeping line breaks.

    Overlapping spans merge; the merged span is named after its most specific rule
    (a provider rule rather than a `generic-*` one).
    """
    spans = sorted((f.start, f.end, f.rule) for f in findings if f.end > f.start)
    merged: list[tuple[int, int, list[str]]] = []
    for s, e, rule in spans:
        if merged and s <= merged[-1][1]:
            ms, me, rules = merged[-1]
            merged[-1] = (ms, max(me, e), [*rules, rule])
        else:
            merged.append((s, e, [rule]))
    out, pos = [], 0
    for s, e, rules in merged:
        name = next((r for r in rules if not r.startswith("generic")), rules[0])
        out.append(text[pos:s])
        out.append(REDACTION.format(rule=name) + "\n" * text.count("\n", s, e))
        pos = e
    out.append(text[pos:])
    return "".join(out)
