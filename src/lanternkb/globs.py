"""Gitignore-style globs over POSIX relative paths.

- A pattern without `/` (other than a trailing one) matches at any depth: `*.pem` matches a
  file name anywhere, and `.ssh/` matches a folder named `.ssh` anywhere, with everything in it.
- A pattern with a `/` in it, or starting with `/`, is anchored at the root: `finance/**`,
  `/finance/`, `Library/Keychains/`.
- A trailing `/` means "this folder and everything under it".
- `**` crosses folders, `*` and `?` do not, `[...]` is a character class (`[!...]` negates),
  `{a,b}` is either alternative.
- Matching is case-sensitive unless `ignore_case=True`. macOS volumes are usually
  case-insensitive, so the deny list always ignores case (a `.SSH` folder is `.ssh`).
"""

from __future__ import annotations

import re
from functools import lru_cache


class GlobError(ValueError):
    pass


def _translate(body: str) -> str:
    out, i = [], 0
    while i < len(body):
        c = body[i]
        if body.startswith("**/", i):
            out.append("(?:[^/]*/)*")
            i += 3
        elif body.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "[":
            j = body.find("]", i + 2)
            if j == -1:
                raise GlobError(f"unclosed [ in {body!r}")
            inner = body[i + 1 : j]
            if inner.startswith("!"):
                inner = "^" + inner[1:]
            out.append("[" + inner.replace("\\", "\\\\").replace("/", "") + "]")
            i = j + 1
        elif c == "{":
            j = body.find("}", i + 1)
            if j == -1:
                raise GlobError(f"unclosed {{ in {body!r}")
            alts = body[i + 1 : j].split(",")
            out.append("(?:" + "|".join(_translate(a) for a in alts) + ")")
            i = j + 1
        else:
            out.append(re.escape(c))
            i += 1
    return "".join(out)


@lru_cache(maxsize=4096)
def _compile(pattern: str, ignore_case: bool) -> tuple[re.Pattern[str], bool]:
    anchored = pattern.startswith("/")
    pattern = pattern.lstrip("/")
    anywhere = not anchored and "/" not in pattern.rstrip("/")
    body = pattern.rstrip("/") + ("/**" if pattern.endswith("/") else "")
    try:
        rx = re.compile(_translate(body) + r"\Z", re.IGNORECASE if ignore_case else 0)
    except re.error as e:
        raise GlobError(f"bad pattern {pattern!r}: {e}") from e
    return rx, anywhere


def validate(pattern: str) -> None:
    """Raise GlobError if the pattern cannot be used."""
    if not pattern or not pattern.strip("/"):
        raise GlobError("empty pattern")
    _compile(pattern, False)


def match(pattern: str, rel_path: str, ignore_case: bool = False) -> bool:
    """Whether a POSIX relative path (no leading `/`) matches the pattern."""
    rx, anywhere = _compile(pattern, ignore_case)
    if anywhere:
        parts = rel_path.split("/")
        if pattern.endswith("/"):  # a folder name anywhere: any component but the last
            return any(rx.match(p + "/x") for p in parts[:-1])
        return any(rx.match(p) for p in parts)
    return rx.match(rel_path) is not None


def match_anywhere(pattern: str, path: str, ignore_case: bool = False) -> bool:
    """Whether the pattern matches the path or any trailing part of it.

    Used for the sensitive deny list, which must hold whatever the root is:
    `Library/Keychains/` has to match `/Users/someone/Library/Keychains/x` as well.
    """
    parts = path.strip("/").split("/")
    return any(match(pattern, "/".join(parts[i:]), ignore_case) for i in range(len(parts)))


class PatternSet:
    """Many patterns compiled into a few regexes, for the hot path of a sync.

    `suffix=True` gives `match_anywhere` semantics (the pattern may match any trailing part of
    an absolute path); `suffix=False` gives `match` semantics on a root-relative path.
    `first()` returns the first pattern (in list order) that matches, exactly as checking them
    one by one would; the combined regexes only decide quickly that nothing matches.
    """

    def __init__(self, patterns: tuple[str, ...], ignore_case: bool, suffix: bool):
        self.patterns = patterns
        self.ignore_case = ignore_case
        self.suffix = suffix
        flags = re.IGNORECASE if ignore_case else 0
        comp, folder, anchored = [], [], []
        for p in patterns:
            _compile(p, ignore_case)  # validates
            is_anchored = p.startswith("/")
            body = p.lstrip("/")
            if not is_anchored and "/" not in body.rstrip("/"):
                if body.endswith("/"):
                    folder.append(_translate(body.rstrip("/") + "/**"))
                else:
                    comp.append(_translate(body))
            else:
                anchored.append(
                    _translate(body.rstrip("/") + ("/**" if body.endswith("/") else ""))
                )
        self._comp = re.compile(f"(?:{'|'.join(comp)})\\Z", flags) if comp else None
        self._folder = re.compile(f"(?:{'|'.join(folder)})\\Z", flags) if folder else None
        if anchored and suffix:
            self._anch = re.compile(f"(?:^|/)(?:{'|'.join(anchored)})\\Z", flags)
        elif anchored:
            self._anch = re.compile(f"(?:{'|'.join(anchored)})\\Z", flags)
        else:
            self._anch = None

    def _hit(self, path: str) -> bool:
        parts = path.strip("/").split("/")
        if self._comp is not None and any(self._comp.match(c) for c in parts):
            return True
        if self._folder is not None and any(self._folder.match(c + "/x") for c in parts[:-1]):
            return True
        if self._anch is not None:
            joined = "/".join(parts)
            return bool(self._anch.search(joined) if self.suffix else self._anch.match(joined))
        return False

    def first(self, path: str) -> str | None:
        if not self._hit(path):
            return None
        check = match_anywhere if self.suffix else match
        for p in self.patterns:
            if check(p, path, self.ignore_case):
                return p
        return None
