"""Plugins: features from other packages.

A plugin is a Python package that declares an entry point in the group
`lanternkb.features`, naming its factory, and ships its feature manifest as a file named
`lanternkb_feature.toml` inside that top-level package:

    [project.entry-points."lanternkb.features"]
    epub = "lanternkb_epub:create"

`create(options)` returns the implementation for the manifest's kind (an extractor, an
embedder or a tier rule), given only that feature's `options` table from config.toml.

Rules:
- **Discovery reads no plugin code.** The manifest is read as a file, found without
  importing the package, so an extractor plugin's code only ever runs inside the
  sandboxed extraction workers.
- **Strict validation.** A plugin with a bad manifest, an unsupported kind, a malformed or
  taken id, a file type another feature handles, or `required = true` is refused with a
  reason (`lanternkb doctor` lists it) and never half-loaded. Two plugins that clash with
  each other are both refused. Limits above `MAX_LIMITS` are lowered to it.
- **Off until the user turns it on.** Installing a plugin enables nothing, whatever its
  manifest says.
- Embedder and tier-rule plugins run in the main process with the user's privileges:
  enabling one means trusting its author. The settings page, `features list`, `features
  enable` and `doctor` say so. Installing any Python package already trusts its author
  (a package can run code at interpreter start, through a `.pth` file).
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import re
import tomllib
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from lanternkb.features import Limits, Manifest, ManifestError, builtin_manifests, parse_manifest

GROUP = "lanternkb.features"
MANIFEST_FILE = "lanternkb_feature.toml"
SUPPORTED_KINDS = ("extractor", "embedder", "tier_rule")
IN_PROCESS_KINDS = ("embedder", "tier_rule")  # run with the user's privileges
# Plugin ids: the kind, a dot, then lower-case letters, digits and dashes. Ids name vector
# tables, config tables and links, so the form is strict (and maps one-to-one to a name).
ID_PATTERN = re.compile(r"(extractor|embedder|tier_rule)\.[a-z0-9][a-z0-9-]{0,47}")
# The most a plugin's manifest may ask for; larger requests are lowered to these, so a hang
# costs at most two minutes and the memory watchdog stays meaningful.
MAX_LIMITS = Limits(cpu_seconds=120, memory_mb=2048, output_mb=50)
MAX_MANIFEST_BYTES = 64_000


@dataclass(frozen=True)
class PluginInfo:
    entry_point: str  # name = value, as declared
    dist: str
    version: str
    feature: str | None = None
    kind: str | None = None
    error: str | None = None  # why it was refused; None when loaded
    limits: Limits | None = None  # as enforced (lowered to MAX_LIMITS if it asked for more)


@dataclass(frozen=True)
class Discovery:
    manifests: dict[str, Manifest]
    entry_points: dict[str, importlib.metadata.EntryPoint]
    infos: list[PluginInfo]

    def refused(self, feature_id: str) -> str | None:
        """Why the plugin that declares this feature was refused, if one was."""
        return next((i.error for i in self.infos if i.feature == feature_id and i.error), None)


_cache: Discovery | None = None


def discover(refresh: bool = False) -> Discovery:
    """Every installed plugin's manifest (validated), without importing plugin code.

    Plugins that clash with each other (the same id, or the same file type) are all
    refused: which one would win must never depend on package names or install order."""
    global _cache
    if _cache is not None and not refresh:
        return _cache
    builtin = builtin_manifests()
    builtin_ids = {m.id for m in builtin}
    builtin_keys = {k for m in builtin for k in [*m.suffixes, *m.names]}
    parsed: list[tuple[importlib.metadata.EntryPoint, PluginInfo, Manifest | None]] = []
    for ep in sorted(importlib.metadata.entry_points(group=GROUP), key=lambda e: e.name):
        dist = ep.dist.metadata["Name"] if ep.dist else "?"
        version = ep.dist.version if ep.dist else "?"
        info = PluginInfo(f"{ep.name} = {ep.value}", dist, version)
        m = None
        try:
            m = _manifest_for(ep)
            info = replace(info, feature=m.id, kind=m.kind)
            _check_plugin(m, builtin_ids, builtin_keys)
        except Exception as e:  # noqa: BLE001 - refuse this plugin, keep the others
            info = replace(info, error=f"{type(e).__name__}: {e}"[:300])
            m = None
        parsed.append((ep, info, m))
    ids = Counter(m.id for _ep, _i, m in parsed if m)
    keys = Counter(k for _ep, _i, m in parsed if m and m.kind == "extractor"
                   for k in {*m.suffixes, *m.names})  # fmt: skip
    manifests: dict[str, Manifest] = {}
    eps: dict[str, importlib.metadata.EntryPoint] = {}
    infos: list[PluginInfo] = []
    for ep, info, m in parsed:
        if m is not None:
            shared = sorted(k for k in {*m.suffixes, *m.names} if keys[k] > 1)
            if ids[m.id] > 1:
                info = replace(info, error=f"ManifestError: two installed plugins declare "
                                           f"{m.id}; both are refused")  # fmt: skip
            elif m.kind == "extractor" and shared:
                info = replace(info, error=f"ManifestError: {', '.join(shared)} handled by "
                                           "two installed plugins; both are refused")  # fmt: skip
            else:
                limits = Limits(*(min(a, b) for a, b in zip(
                    (m.limits.cpu_seconds, m.limits.memory_mb, m.limits.output_mb),
                    (MAX_LIMITS.cpu_seconds, MAX_LIMITS.memory_mb, MAX_LIMITS.output_mb),
                    strict=True)))  # fmt: skip
                manifests[m.id] = replace(m, default_enabled=False, limits=limits)  # opt-in
                eps[m.id] = ep
                info = replace(info, limits=limits)
        infos.append(info)
    _cache = Discovery(manifests, eps, infos)
    return _cache


def _check_plugin(m: Manifest, builtin_ids: set[str], builtin_keys: set[str]) -> None:
    if m.kind not in SUPPORTED_KINDS:
        raise ManifestError(f"kind {m.kind!r} cannot be a plugin yet "
                            f"(supported: {', '.join(SUPPORTED_KINDS)})")  # fmt: skip
    if not m.id.startswith(m.kind + "."):
        raise ManifestError(f"id {m.id!r} must start with {m.kind!r}.")
    if not ID_PATTERN.fullmatch(m.id):
        raise ManifestError(f"id {m.id!r}: after the kind, use 1 to 48 lower-case letters, "
                            "digits and dashes")  # fmt: skip
    if m.id in builtin_ids:
        raise ManifestError(f"id {m.id!r} is already taken")
    if m.required:
        raise ManifestError("a plugin cannot be required")
    if m.kind == "extractor":
        clash = sorted(builtin_keys & {*m.suffixes, *m.names})
        if clash:
            raise ManifestError(f"handles {', '.join(clash)}, already handled")


def _manifest_for(ep: importlib.metadata.EntryPoint) -> Manifest:
    """Read the plugin's manifest file without executing any of its code."""
    module = ep.value.split(":", 1)[0].strip()
    if "." in module:
        # Finding a submodule would import its parent package (running its code), and
        # namespace packages would let one distribution's manifest speak for another's.
        raise ManifestError(f"the entry point must name a top-level package, not {module!r}")
    spec = importlib.util.find_spec(module)  # a top-level name: nothing is imported
    if spec is None or not spec.submodule_search_locations:
        raise ManifestError(f"{module!r} is not an installed package")
    path = Path(next(iter(spec.submodule_search_locations))) / MANIFEST_FILE
    if not path.is_file():
        raise ManifestError(f"no {MANIFEST_FILE} in the package {module!r}")
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ManifestError(f"{MANIFEST_FILE} is over {MAX_MANIFEST_BYTES} bytes")
    return parse_manifest(tomllib.loads(path.read_text()), f"{module}/{MANIFEST_FILE}")


def trust_note(feature_id: str) -> str | None:
    """What a user should know before enabling a plugin feature; None for built-ins."""
    found = discover()
    m = found.manifests.get(feature_id)
    if m is None:
        return None
    info = next(i for i in found.infos if i.feature == feature_id and not i.error)
    who = f"a plugin ({info.dist} {info.version})"
    if m.kind in IN_PROCESS_KINDS:
        return (f"{who} that runs inside lanternkb with your privileges: what it reads and "
                "whether it uses the network are its author's word, not enforced. Enable it "
                "only if you trust its author.")  # fmt: skip
    from lanternkb.extract.pool import sandbox_available

    where = ("in the sandbox (no network, no other files)" if sandbox_available()
             else "in a separate process (this system has no sandbox)")  # fmt: skip
    return f"{who} that runs {where}."


def create(feature_id: str, options: Mapping[str, Any] | None = None) -> Any:
    """Import a plugin and build its implementation. For extractors this runs only inside
    an extraction worker."""
    found = discover()
    if feature_id not in found.entry_points:
        raise KeyError(f"no plugin provides {feature_id}")
    factory: Callable[[Mapping[str, Any]], Any] = found.entry_points[feature_id].load()
    impl = factory(dict(options or {}))
    expected = _api_version(found.manifests[feature_id].kind)
    if getattr(impl, "api_version", None) != expected:
        raise ManifestError(f"{feature_id}: the implementation has api_version "
                            f"{getattr(impl, 'api_version', None)!r}, not {expected}")  # fmt: skip
    return impl


CREATE_TIMEOUT_S = 30.0  # building an in-process plugin


def create_with_deadline(feature_id: str, options: Mapping[str, Any] | None = None,
                         timeout_s: float | None = None) -> Any:  # fmt: skip
    """`create` on a worker thread with a deadline, reading the attributes lanternkb checks
    there too, so a plugin whose factory or attributes hang costs seconds, not the sync or
    the server. Raises TimeoutError after `timeout_s`; anything the plugin
    raises (SystemExit too) comes back as RuntimeError."""
    import threading

    timeout_s = CREATE_TIMEOUT_S if timeout_s is None else timeout_s
    box: dict[str, Any] = {}

    def build() -> None:
        try:
            impl = create(feature_id, options)
            for name in ("api_version", "name", "needs_text", "key", "dim", "model_id"):
                getattr(impl, name, None)  # a property that hangs hangs here, not later
            box["impl"] = impl
        except BaseException as e:  # noqa: BLE001 - re-raised below, as an ordinary error
            box["error"] = e

    t = threading.Thread(target=build, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise TimeoutError(f"plugin {feature_id} took over {timeout_s:.0f} s to load")
    if "error" in box:
        e = box["error"]
        if isinstance(e, Exception):
            raise e
        raise RuntimeError(f"plugin {feature_id} raised {type(e).__name__}") from e
    return box["impl"]


def _api_version(kind: str) -> int:
    if kind == "extractor":
        from lanternkb.extract.base import API_VERSION

        return API_VERSION
    if kind == "embedder":
        from lanternkb.embed import API_VERSION as EMBED_API

        return EMBED_API
    return 1  # tier rules: lanternkb.tiers.ContentTierRule
