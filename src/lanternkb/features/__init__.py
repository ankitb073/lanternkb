"""Features: everything the user can switch on or off, described by a manifest.

A manifest (TOML) says what a feature is, what it may do (shown to the user before they
enable it), the limits the host enforces on it, and, for extractors, which files it handles.
Manifests are validated strictly; a bad one is an error, never silently ignored.
Plugins add manifests through entry points (`lanternkb.plugins`).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from lanternkb.tiers import Tier

API_VERSION = 1
KINDS = ("extractor", "embedder", "scanner", "tier_rule", "reranker")
CATEGORIES = ("documents", "code", "images", "audio", "safety", "search")

_SCHEMA = {
    "": {"feature", "capabilities", "limits", "handles"},
    "feature": {"id", "title", "category", "kind", "api_version", "default_enabled",
                "default_tier", "required", "description", "version"},
    "capabilities": {"reads", "network", "os_permissions"},
    "limits": {"cpu_seconds", "memory_mb", "output_mb"},
    "handles": {"suffixes", "names"},
}  # fmt: skip


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class Limits:
    cpu_seconds: int = 30
    memory_mb: int = 512
    output_mb: int = 20


@dataclass(frozen=True)
class Manifest:
    id: str
    title: str
    category: str
    kind: str
    api_version: int
    default_enabled: bool
    default_tier: Tier
    required: bool
    description: str
    reads: tuple[str, ...]
    network: bool
    os_permissions: tuple[str, ...]
    limits: Limits
    suffixes: frozenset[str]  # lower-case, with the dot
    names: frozenset[str]  # exact file names, e.g. "Makefile"
    # An extractor's output version: raising it makes the next sync read that feature's
    # files again (what it extracts changed), and only those.
    version: int = 1

    def handles(self, path: Path) -> bool:
        return path.suffix.lower() in self.suffixes or path.name in self.names


def _check(where: str, table: dict[str, Any], source: str) -> None:
    unknown = set(table) - _SCHEMA[where]
    if unknown:
        raise ManifestError(f"{source}: unknown key(s) {sorted(unknown)} in [{where or 'top'}]")


def parse_manifest(data: dict[str, Any], source: str = "<manifest>") -> Manifest:
    _check("", data, source)
    f = data.get("feature")
    if not isinstance(f, dict):
        raise ManifestError(f"{source}: missing [feature]")
    _check("feature", f, source)
    for key in ("id", "title", "category", "kind", "api_version"):
        if key not in f:
            raise ManifestError(f"{source}: [feature] needs {key!r}")
    if f["kind"] not in KINDS:
        raise ManifestError(f"{source}: kind {f['kind']!r} not one of {KINDS}")
    if f["category"] not in CATEGORIES:
        raise ManifestError(f"{source}: category {f['category']!r} not one of {CATEGORIES}")
    if f["api_version"] != API_VERSION:
        raise ManifestError(f"{source}: api_version {f['api_version']} is not {API_VERSION}")
    caps = data.get("capabilities", {})
    _check("capabilities", caps, source)
    if caps.get("network", False):
        raise ManifestError(f"{source}: network access is not allowed for features in the core")
    lim = data.get("limits", {})
    _check("limits", lim, source)
    limits = Limits(**{k: int(v) for k, v in lim.items()})
    if min(limits.cpu_seconds, limits.memory_mb, limits.output_mb) <= 0:
        raise ManifestError(f"{source}: limits must be positive")
    handles = data.get("handles", {})
    _check("handles", handles, source)
    suffixes = frozenset(s.lower() for s in handles.get("suffixes", []))
    if any(not s.startswith(".") for s in suffixes):
        raise ManifestError(f"{source}: suffixes must start with '.'")
    if f["kind"] == "extractor" and not (suffixes or handles.get("names")):
        raise ManifestError(f"{source}: an extractor must declare [handles]")
    try:
        tier = Tier(f.get("default_tier", Tier.LOCAL_ONLY))
    except ValueError as e:
        raise ManifestError(f"{source}: bad default_tier") from e
    required = bool(f.get("required", False))
    version = f.get("version", 1)
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ManifestError(f"{source}: version must be a positive integer")
    return Manifest(
        id=str(f["id"]),
        title=str(f["title"]),
        category=f["category"],
        kind=f["kind"],
        api_version=f["api_version"],
        default_enabled=bool(f.get("default_enabled", False)) or required,
        default_tier=tier,
        required=required,
        description=str(f.get("description", "")),
        reads=tuple(caps.get("reads", [])),
        network=False,
        os_permissions=tuple(caps.get("os_permissions", [])),
        limits=limits,
        suffixes=suffixes,
        names=frozenset(handles.get("names", [])),
        version=version,
    )


def builtin_manifests() -> list[Manifest]:
    base = resources.files("lanternkb.features") / "builtin"
    out = []
    for entry in sorted(base.iterdir(), key=lambda e: e.name):
        if entry.name.endswith(".toml"):
            out.append(parse_manifest(tomllib.loads(entry.read_text()), entry.name))
    return out


@dataclass(frozen=True)
class FeatureState:
    enabled: bool
    tier: Tier


class Registry:
    """All known features and the user's settings for them."""

    def __init__(self, manifests: list[Manifest] | None = None,
                 settings: dict[str, dict[str, Any]] | None = None):  # fmt: skip
        self.settings = settings or {}
        if manifests is None:  # built-in features plus installed plugins (all off until enabled)
            from lanternkb.plugins import discover

            found = discover()
            if set(self.settings) - {*found.manifests, *(m.id for m in builtin_manifests())}:
                found = discover(refresh=True)  # installed since a long-lived process looked
            manifests = [*builtin_manifests(), *found.manifests.values()]
        self.manifests: dict[str, Manifest] = {}
        for m in manifests:
            if m.id in self.manifests:
                raise ManifestError(f"duplicate feature id {m.id!r}")
            self.manifests[m.id] = m
        self._check_overlaps()
        # Features the config names but no installed plugin provides (uninstalled, or
        # refused): treated as off, so their data is hidden, never an error for readers.
        # Syncing refuses until the config is fixed (`check_known`), so nothing is deleted.
        self.unknown = tuple(sorted(set(self.settings) - set(self.manifests)))

    def check_known(self) -> None:
        """Raise, with the reason, if the config names features no plugin provides."""
        if not self.unknown:
            return
        from lanternkb.plugins import discover

        found = discover()
        why = []
        for fid in self.unknown:
            reason = found.refused(fid)
            why.append(f"{fid} (its plugin was refused: {reason})" if reason
                       else f"{fid} (no installed plugin provides it)")  # fmt: skip
        raise ManifestError(f"config names unknown feature(s): {'; '.join(why)}. Reinstall "
                            "the plugin, or purge the feature and remove its table from "
                            "config.toml")  # fmt: skip

    def _check_overlaps(self) -> None:
        seen: dict[str, str] = {}
        for m in self.manifests.values():
            for key in [*m.suffixes, *m.names]:
                if key in seen:
                    raise ManifestError(f"{key!r} is claimed by {seen[key]} and {m.id}")
                seen[key] = m.id

    def options(self, feature_id: str) -> dict[str, Any]:
        """The feature's own `options` table from the config (plugins see only theirs)."""
        return dict(self.settings.get(feature_id, {}).get("options", {}))

    def state(self, feature_id: str) -> FeatureState:
        if feature_id not in self.manifests:  # unknown: off (fail closed)
            return FeatureState(False, Tier.LOCAL_ONLY)
        m = self.manifests[feature_id]
        s = self.settings.get(feature_id, {})
        enabled = True if m.required else bool(s.get("enabled", m.default_enabled))
        return FeatureState(enabled, Tier(s.get("tier", m.default_tier)))

    def extractor_for(self, path: Path) -> Manifest | None:
        """The extractor feature that handles this file, enabled or not."""
        for m in self.manifests.values():
            if m.kind == "extractor" and m.handles(path):
                return m
        return None
