"""Filesystem helpers for the path policy."""

from __future__ import annotations

import contextlib
import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=8192)
def _entries(folder: str) -> tuple[str, ...]:
    try:
        return tuple(os.listdir(folder))
    except OSError:
        return ()


def canonical_case(path: Path) -> Path:
    """The path with each existing component spelt as it is on disk.

    On a case-insensitive volume `/Users/me/notes/.SSH/x` and `/Users/me/Notes/.ssh/x` are the
    same file; comparing roots, deny patterns and tiers needs one spelling. Components that
    do not exist are kept as given.
    """
    p = Path(os.path.abspath(path))
    out = Path(p.anchor)
    for part in p.parts[1:]:
        names = _entries(str(out))
        if part in names:
            out = out / part
            continue
        folded = [n for n in names if n.casefold() == part.casefold()]
        out = out / (folded[0] if len(folded) == 1 else part)
    return out


def clear_cache() -> None:
    _entries.cache_clear()


def private_dirs(path: Path) -> None:
    """Create `path` and any missing parent folders, each one only its owner can enter.
    (`mkdir(parents=True, mode=...)` gives the mode to the last folder only.) Folders that
    already exist keep their mode."""
    missing = []
    p = path
    while not p.exists():
        missing.append(p)
        if p.parent == p:
            break
        p = p.parent
    for folder in reversed(missing):
        with contextlib.suppress(FileExistsError):  # made meanwhile by another process
            folder.mkdir(mode=0o700)
