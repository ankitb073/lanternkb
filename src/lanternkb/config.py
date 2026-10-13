"""Configuration: roots and their tiers, extra deny globs, tier rules, limits, scanner options.

The file lives at `~/.config/lanternkb/config.toml` by default. Code that loads it always
takes an explicit path, so tests never touch a real configuration.

    index = "~/.local/share/lanternkb/index.db"
    deny = ["*.secret"]                  # added to the built-in deny list, never replacing it

    [[roots]]
    path = "~/notes"
    tier = "open"                        # default: "local-only"
    secrets_make_local_only = true       # default: true

    [[tier_rules]]                       # later rules win
    pattern = "finance/"
    tier = "local-only"

    [limits]
    max_file_bytes = 10000000
    max_text_chars = 5000000

    [features."extractor.code"]          # per feature: enabled, tier (manifest defaults otherwise)
    enabled = false

    [scanner]
    disabled_rules = []
    enabled_rules = []                   # opt-in rules and passes (scan.rules.OPT_IN)
    [scanner.actions]                    # per-rule action override: "redact" or "skip-file"
    "generic-api-key" = "redact"
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from lanternkb import globs
from lanternkb.fsutil import canonical_case
from lanternkb.tiers import DEFAULT_TIER, PathTierRule, Tier, stricter

DEFAULT_CONFIG = Path("~/.config/lanternkb/config.toml")
DEFAULT_INDEX = Path("~/.local/share/lanternkb/index.db")
ACTIONS = ("redact", "skip-file")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Root:
    path: Path  # absolute and resolved
    tier: Tier = DEFAULT_TIER
    secrets_make_local_only: bool = True


@dataclass(frozen=True)
class Limits:
    max_file_bytes: int = 10_000_000
    max_text_chars: int = 5_000_000
    workers: int = 0  # extraction workers; 0 = one per core, at most 8


@dataclass(frozen=True)
class ScannerOptions:
    disabled_rules: tuple[str, ...] = ()
    actions: dict[str, str] = field(default_factory=dict)
    enabled_rules: tuple[str, ...] = ()  # opt-in rules (scan.rules.OPT_IN) switched on


@dataclass(frozen=True)
class Config:
    roots: tuple[Root, ...]
    index_path: Path = DEFAULT_INDEX
    deny: tuple[str, ...] = ()
    tier_rules: tuple[PathTierRule, ...] = ()
    limits: Limits = Limits()
    scanner: ScannerOptions = ScannerOptions()
    features: dict[str, dict[str, Any]] = field(default_factory=dict)
    source: Path | None = None  # the config file this came from, if any


def _path(value: Any, base: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"expected a path string, got {value!r}")
    p = Path(os.path.expanduser(value))
    return (p if p.is_absolute() else base / p).resolve()


def _tier(value: Any) -> Tier:
    try:
        return Tier(value)
    except ValueError as e:
        raise ConfigError(f"unknown tier {value!r}; use 'open' or 'local-only'") from e


_KEYS = {
    "": {"index", "deny", "roots", "tier_rules", "limits", "scanner", "features"},
    "features.*": {"enabled", "tier", "options"},
    "roots": {"path", "tier", "secrets_make_local_only"},
    "tier_rules": {"pattern", "tier"},
    "limits": {"max_file_bytes", "max_text_chars", "workers"},
    "scanner": {"disabled_rules", "enabled_rules", "actions"},
}
# First components that mark a pattern as an absolute path rather than a root-relative one.
_SYSTEM_TOP = {"users", "home", "volumes", "private", "var", "tmp", "opt", "etc", "system",
               "applications", "library", "mnt", "media", "root"}  # fmt: skip


def _check_keys(where: str, table: dict[str, Any]) -> None:
    unknown = set(table) - _KEYS[where]
    if unknown:
        raise ConfigError(f"unknown key(s) {sorted(unknown)} in {where or 'the top level'}; "
                          f"allowed: {sorted(_KEYS[where])}")  # fmt: skip


def _looks_absolute(pattern: str) -> bool:
    if pattern.startswith("~"):
        return True
    first = pattern.lstrip("/").split("/", 1)[0].casefold()
    return pattern.startswith("/") and first in _SYSTEM_TOP


def _tier_pattern(value: Any) -> str:
    pattern = str(value)
    if _looks_absolute(pattern):
        raise ConfigError(f"tier rule {pattern!r} looks like an absolute path; tier rules are "
                          "relative to their root (write 'health/', not '~/notes/health/')")  # fmt: skip  # noqa: E501
    try:
        globs.validate(pattern)
    except globs.GlobError as e:
        raise ConfigError(str(e)) from e
    return pattern


def _deny_pattern(value: Any) -> str:
    """Deny patterns may be absolute (`~/notes/private/`): kept absolute, matched as such."""
    pattern = str(value)
    if _looks_absolute(pattern):
        pattern = "/" + os.path.expanduser(pattern).lstrip("/")
        return "abs:" + pattern
    try:
        globs.validate(pattern)
    except globs.GlobError as e:
        raise ConfigError(str(e)) from e
    return pattern


def _merge_roots(roots: list[Root]) -> tuple[Root, ...]:
    """One entry per folder (same path or same inode); the stricter settings win."""
    merged: dict[object, Root] = {}
    for r in roots:
        try:
            st = r.path.stat()
            key: object = (st.st_dev, st.st_ino)
        except OSError:
            key = str(r.path).casefold()
        if key in merged:
            old = merged[key]
            merged[key] = Root(old.path, stricter(old.tier, r.tier),
                               old.secrets_make_local_only or r.secrets_make_local_only)  # fmt: skip  # noqa: E501
        else:
            merged[key] = r
    return tuple(merged.values())


def parse(data: dict[str, Any], base: Path) -> Config:
    """Build a Config from parsed TOML. Relative paths resolve against base.

    Unknown keys, bad patterns and absolute tier rules are errors, never silently ignored:
    a typo in a restriction must not quietly leave files open.
    """
    _check_keys("", data)
    roots = []
    for r in data.get("roots", []):
        _check_keys("roots", r)
        roots.append(
            Root(
                canonical_case(_path(r.get("path"), base)),
                _tier(r.get("tier", DEFAULT_TIER)),
                bool(r.get("secrets_make_local_only", True)),
            )
        )
    for t in data.get("tier_rules", []):
        _check_keys("tier_rules", t)
    rules = tuple(
        PathTierRule(_tier_pattern(t["pattern"]), _tier(t["tier"]))
        for t in data.get("tier_rules", [])
    )
    lim = data.get("limits", {})
    _check_keys("limits", lim)
    limits = Limits(
        int(lim.get("max_file_bytes", Limits.max_file_bytes)),
        int(lim.get("max_text_chars", Limits.max_text_chars)),
        int(lim.get("workers", 0)),
    )
    sc = data.get("scanner", {})
    _check_keys("scanner", sc)
    actions = {str(k): str(v) for k, v in sc.get("actions", {}).items()}
    bad = {k: v for k, v in actions.items() if v not in ACTIONS}
    if bad:
        raise ConfigError(f"unknown scanner actions {bad}; use one of {ACTIONS}")
    features: dict[str, dict[str, Any]] = {}
    for fid, table in data.get("features", {}).items():
        if not isinstance(table, dict):
            raise ConfigError(f"[features.{fid!r}] must be a table")
        _check_keys("features.*", table)
        entry: dict[str, Any] = {}
        if "enabled" in table:
            if not isinstance(table["enabled"], bool):
                raise ConfigError(f"features.{fid}.enabled must be true or false")
            entry["enabled"] = table["enabled"]
        if "tier" in table:
            entry["tier"] = _tier(table["tier"])
        if "options" in table:
            entry["options"] = _options(fid, table["options"])
        features[str(fid)] = entry
    index = _path(data["index"], base) if "index" in data else DEFAULT_INDEX.expanduser()
    return Config(
        roots=_merge_roots(roots),
        index_path=index,
        deny=tuple(_deny_pattern(d) for d in data.get("deny", [])),
        tier_rules=rules,
        limits=limits,
        scanner=ScannerOptions(
            tuple(sc.get("disabled_rules", [])), actions, tuple(sc.get("enabled_rules", []))
        ),  # fmt: skip
        features=features,
    )


def _options(fid: str, value: Any) -> dict[str, Any]:
    """A plugin's own settings: a table of text, numbers, booleans, lists and tables."""

    def ok(v: Any) -> bool:
        if isinstance(v, (str, int, float, bool)):
            return True
        if isinstance(v, list):
            return all(ok(x) for x in v)
        if isinstance(v, dict):
            return all(isinstance(k, str) and ok(x) for k, x in v.items())
        return False

    if not isinstance(value, dict) or not ok(value):
        raise ConfigError(f"features.{fid}.options must be a table of text, numbers, booleans, "
                          "lists and tables (no dates)")  # fmt: skip
    return dict(value)


def load(path: Path) -> Config:
    """Load a config file. Relative paths inside it resolve against its folder."""
    with path.open("rb") as f:
        data = tomllib.load(f)
    return replace(parse(data, path.parent.resolve()), source=path.resolve())
