"""`--redact`: output people can paste into a bug report.

Your home folder becomes `~`, and every other path part, and every bare file name, becomes
a short hash made with a key that is new for each run, so a hash cannot be looked up in a
list of likely names, while the same name gives the same hash within one report. Names
the index and config know (roots, files, folders) are found whole, even when they contain
spaces. lanternkb's own file and folder names stay readable.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from collections.abc import Iterable
from pathlib import Path

# Kept as they are: lanternkb's own files and folders, and the usual places they live.
KEEP = frozenset({"~", ".config", ".local", "share", "lanternkb", "config.toml", "index.db",
                  "sync.log", "models", "Library", "Logs", "LaunchAgents", "usr", "bin", "opt",
                  "homebrew", "tmp", "private", "var", "folders"})  # fmt: skip
# A path: a quoted one whole (tracebacks quote them), or parts joined by "/", where a middle
# part may hold single spaces ("Backup Disk/"); the last part stops at a space.
_PART = r"[^\s:,;'\"()\[\]{}<>/]"
_PATH = re.compile(rf"""'(?:~|/)[^'\n]*'|"(?:~|/)[^"\n]*"|"""
                   rf"(?:~|/)(?:{_PART}+(?: {_PART}+)*/)*{_PART}*")  # fmt: skip
# A file name standing alone, as in "error on notes.md": a name with a dot and a common
# document, media or code extension (feature ids such as extractor.images do not match).
_EXTS = (
    "md|markdown|txt|text|rst|org|csv|tsv|json|jsonl|yaml|yml|toml|ini|cfg|conf|env|xml|"
    "html?|xhtml|pdf|docx?|xlsx?|pptx?|odt|ods|odp|rtf|pages|numbers|key|epub|eml|mbox|ics|vcf|"
    "srt|vtt|adoc|asciidoc|bib|mdx|qmd|rmd|ott|ots|otp|ps1|psm1|bat|cmd|dart|scala|ex|exs|erl|"
    "hs|clj|cljs|kts|groovy|proto|graphql|zig|nim|jl|vim|el|ml|fs|sol|nix|vue|svelte|astro|"
    "png|jpe?g|gif|heic|tiff?|bmp|webp|m4a|mp3|wav|aiff?|flac|ogg|opus|mp4|mov|py|ipynb|"
    "js|ts|tsx|jsx|go|rs|java|kt|swift|c|cc|cpp|h|hpp|rb|php|sh|zsh|sql|log|db|sqlite|zip"
)


def _all_exts() -> str:
    """The list above, plus every extension a built-in feature handles, so a file
    of any indexed type is hidden."""
    try:
        from lanternkb.features import builtin_manifests

        more = {x.lstrip(".") for m in builtin_manifests() for x in m.suffixes}
    except Exception:  # noqa: BLE001 - the fixed list still works
        more = set()
    known = set(_EXTS.split("|"))
    return "|".join([_EXTS, *sorted(re.escape(e) for e in more - known)])


_NAME = re.compile(rf"(?<![\w/.~-])[^\s/:,;'\"()\[\]{{}}<>]+\.(?:{_all_exts()})(?![\w.-])", re.I)
_MARK = re.compile("(\\d+)")  # a known name, held while paths are rewritten


class Redactor:
    def __init__(self, known: Iterable[str] = (), keep: Iterable[str] = ()) -> None:
        # Kept as they are: exact names such as feature ids (extractor.text), never a prefix,
        # so a file of the user's named extractor.md is still hidden.
        self.keep = frozenset(keep)
        self.key = secrets.token_bytes(16)
        home = str(Path.home())
        self._home = re.compile(re.escape(home) + r"(?![\w.-])") if home != "/" else None
        names: set[str] = set()
        for k in known:
            k = self._tilde(k)
            if k and k != "~":
                names.add(k)
                names.update(p for p in k.split("/") if " " in p.strip())
        # Longest first, so a whole path wins over one of its parts.
        self.names = sorted((n for n in names if n.strip() and n not in KEEP),
                            key=len, reverse=True)  # fmt: skip
        self._index = {n: i for i, n in enumerate(self.names)}
        self._known = (re.compile("|".join(map(re.escape, self.names)))
                       if self.names else None)  # fmt: skip

    def _tilde(self, text: str) -> str:
        return self._home.sub("~", text) if self._home else text

    def tag(self, name: str) -> str:
        return "h" + hmac.new(self.key, name.encode(), hashlib.sha256).hexdigest()[:6]

    def _part(self, part: str) -> str:
        if not part or part in KEEP or part.startswith(".lanternkb"):
            return part
        held = _MARK.fullmatch(part)
        if held:
            name = self.names[int(held.group(1))]
            if "/" in name:
                return "/".join(self._part(p) for p in name.split("/"))
            return self._part(name) if name in KEEP else self.tag(name)
        return self.tag(_MARK.sub(lambda m: self.names[int(m.group(1))], part))

    def _path(self, m: re.Match[str]) -> str:
        path = m.group(0)
        quote = path[0] if path[0] in "'\"" else ""
        body = path[1:-1] if quote else path
        return quote + "/".join(self._part(p) for p in body.split("/")) + quote

    def __call__(self, text: str) -> str:
        text = self._tilde(text)
        if self._known is not None:
            text = self._known.sub(lambda m: f"\ue000{self._index[m.group(0)]}\ue001", text)
        text = _PATH.sub(self._path, text)
        text = _NAME.sub(lambda m: m.group(0) if m.group(0) in self.keep
                         else self._part(m.group(0)), text)  # fmt: skip
        return _MARK.sub(lambda m: self._part(m.group(0)), text)


def known_names(config_path: Path) -> list[str]:
    """What the config and index know: the config and index paths, the roots, and every
    stored file's path and relative path (best effort: a missing or unreadable index gives
    fewer names)."""
    import sqlite3

    out = [str(config_path)]
    try:
        from lanternkb.config import load

        cfg = load(config_path)
    except Exception:  # noqa: BLE001 - redaction still works on what the regexes find
        return out
    out += [str(cfg.index_path)] + [str(r.path) for r in cfg.roots]
    try:
        con = sqlite3.connect(f"file:{cfg.index_path}?mode=ro", uri=True)
        try:
            for path, rel in con.execute("SELECT path, rel FROM files"):
                out += [path, rel or ""]  # a sync warning names the file by its rel
        finally:
            con.close()
    except sqlite3.Error:
        pass
    return out


class RedactingStream:
    """A stream that redacts everything written to it."""

    def __init__(self, stream: object, redactor: Redactor):
        self.stream = stream
        self.redactor = redactor

    def write(self, text: str) -> int:
        return self.stream.write(self.redactor(text))  # type: ignore[attr-defined, no-any-return]

    def flush(self) -> None:
        self.stream.flush()  # type: ignore[attr-defined]

    def __getattr__(self, name: str) -> object:
        return getattr(self.stream, name)
