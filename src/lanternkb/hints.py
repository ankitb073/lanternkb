"""Commands lanternkb suggests, written as the person running it would type them: plain
`lanternkb` when that is the one on their PATH, `uv run lanternkb` from a checkout, else
its full path; with `--config` when they gave one (found in a fresh-clone run)."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import sys
from functools import cache
from pathlib import Path

_config: str | None = None
_COMMANDS = ("scan", "explain", "features", "roots", "models", "setup", "sync", "search",
             "recent", "serve", "install-schedule", "mcp-setup", "doctor", "ui", "status",
             "skipped")  # fmt: skip
# only `lanternkb <subcommand>`: text that merely names lanternkb is left alone
_WORD = re.compile(r"(?<![\w/.-])lanternkb(?= (?:" + "|".join(_COMMANDS) + r")\b)")


def use_config(path: Path | None) -> None:
    """Remember a config given with --config, so suggestions carry it."""
    global _config
    _config = shlex.quote(str(path.expanduser())) if path is not None else None


@cache
def _program() -> str:
    script = Path(sys.argv[0]) if sys.argv and sys.argv[0] else None
    found = shutil.which("lanternkb")
    if script and script.name == "lanternkb" and found and (
            Path(found).resolve() == script.resolve()):  # fmt: skip
        return "lanternkb"
    project = Path(sys.prefix).parent
    if os.environ.get("UV") and (project / "pyproject.toml").is_file():
        try:
            Path.cwd().relative_to(project)
            return "uv run lanternkb"
        except ValueError:
            return f"uv run --project {shlex.quote(str(project))} lanternkb"
    if script and script.name == "lanternkb" and script.exists():
        return shlex.quote(str(script.resolve()))
    return "lanternkb"


def shown(text: str) -> str:
    """`text` with each `lanternkb <command>` written as this person would run it."""
    if "lanternkb" not in text:
        return text
    program = _program() + (f" --config {_config}" if _config else "")
    return _WORD.sub(lambda m: program, text)
