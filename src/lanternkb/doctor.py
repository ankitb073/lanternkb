"""`lanternkb doctor`: is everything in order, and if not, what to do.

Each check is (status, what, advice). It reads configuration, metadata and the index's
health, never document text, and changes nothing.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

OK, WARN, FAIL = "ok", "warn", "fail"
Check = tuple[str, str, str]


def run(config_path: Path) -> list[Check]:
    from lanternkb.config import ConfigError, load

    out: list[Check] = []
    if not config_path.is_file():
        return [(FAIL, f"no config at {config_path}",
                 "add a folder: lanternkb roots add <folder> (creates the config)")]  # fmt: skip
    try:
        cfg = load(config_path)
    except (ConfigError, OSError, ValueError) as e:
        return [(FAIL, f"config cannot be read: {e}", "fix the file or restore it")]
    out.append((OK, f"config {config_path}", ""))
    for r in cfg.roots:
        if not r.path.is_dir():
            out.append((WARN, f"folder missing: {r.path}", "plugged out? its files stay "
                        "indexed until you remove the folder"))  # fmt: skip
    if not cfg.roots:
        out.append((WARN, "no folders to index", "lanternkb roots add <folder>"))
    out += _plugins(cfg)
    out += _scanner(cfg)
    out += _privacy(config_path, cfg.index_path)
    out += _index(cfg.index_path)
    out += _semantic(cfg)
    out += _reranking(cfg)
    out += _sandbox(cfg)
    return out


def _plugins(cfg: object) -> list[Check]:
    from lanternkb.features import ManifestError, Registry
    from lanternkb.ingest import _Unloadable, plugin_tier_rules
    from lanternkb.plugins import discover, trust_note

    out: list[Check] = []
    found = discover(refresh=True)
    registry = Registry(settings=cfg.features)  # type: ignore[attr-defined]
    try:
        registry.check_known()
    except ManifestError as e:
        out.append((FAIL, f"features: {e}", "syncing is stopped until this is fixed; search "
                    "still works, without that feature's data"))  # fmt: skip
    seen_by_workers = _worker_discovery()
    for info in found.infos:
        where = f"plugin {info.dist} {info.version} ({info.entry_point})"
        if info.error:
            out.append((FAIL, f"{where} refused: {info.error}", "ask its author, or uninstall it"))
            continue
        assert info.feature is not None
        on = registry.state(info.feature).enabled
        state = "on" if on else f"off (turn on: lanternkb features enable {info.feature})"
        lim = info.limits
        limits = f" Limits: {lim.cpu_seconds} s, {lim.memory_mb} MB." if lim else ""
        out.append((OK, f"{where}: {info.feature}, {state}. This is "
                    f"{trust_note(info.feature)}{limits}", ""))  # fmt: skip
        if (info.kind == "extractor" and seen_by_workers is not None
                and info.feature not in seen_by_workers):  # fmt: skip
            out.append((FAIL, f"{info.feature} is not visible to the extraction workers "
                        "(they run `python -I`, which ignores PYTHONPATH and user site "
                        "packages)", "install it into lanternkb's own environment"))  # fmt: skip
    for rule in plugin_tier_rules(registry):
        if isinstance(rule, _Unloadable):
            out.append((FAIL, f"tier rule {rule.feature} does not load ({rule.error}); every "
                        "file is local-only until it does", "reinstall or disable it"))  # fmt: skip
    return out


def _scanner(cfg: object) -> list[Check]:
    """Secret rules that did not load, and ids in `[scanner]` that name no rule."""
    from lanternkb.scan.rules import load_rules

    sc = cfg.scanner  # type: ignore[attr-defined]
    rules = load_rules(disabled=sc.disabled_rules, actions=sc.actions, enabled=sc.enabled_rules)
    return [(WARN, f"secret scanner: {rid}: {why}", "check the [scanner] table of the config")
            for rid, why in rules.failed]  # fmt: skip


def _worker_discovery() -> set[str] | None:
    """The plugin features an extraction worker's Python (`python -I`) can see."""
    import json
    import subprocess
    import sys

    code = ("import json; from lanternkb.plugins import discover; "
            "print(json.dumps(sorted(discover().manifests)))")  # fmt: skip
    try:
        out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True,
                             text=True, timeout=60)  # fmt: skip
        return set(json.loads(out.stdout)) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _privacy(config_path: Path, index: Path) -> list[Check]:
    """Who else could read the config or the index, and where copies of the index go."""
    import subprocess
    import sys

    from lanternkb.store import excluded_from_backups

    out: list[Check] = []
    for what, p in (("config", config_path), ("index", index), ("index folder", index.parent)):
        if p.exists() and p.stat().st_mode & 0o077:
            out.append((FAIL, f"the {what} {p} can be read by other users",
                        f"chmod {'700' if p.is_dir() else '600'} {p}"))  # fmt: skip
    if sys.platform == "darwin":
        try:
            vault = subprocess.run(["/usr/bin/fdesetup", "status"], capture_output=True,
                                   text=True, timeout=10).stdout  # fmt: skip
        except (OSError, subprocess.SubprocessError):
            vault = ""
        if "FileVault is On" in vault:
            out.append((OK, "FileVault is on (the index holds redacted text of your files)", ""))
        elif vault:
            out.append((WARN, "FileVault is off: the index holds the redacted text of every "
                        "indexed file", "turn on FileVault in System Settings, Privacy & "
                        "Security"))  # fmt: skip
        if index.exists() and not (excluded_from_backups(index)
                                   or excluded_from_backups(index.parent)):  # fmt: skip
            out.append((WARN, "Time Machine backs up the index, so purged text lives on in "
                        "old backups", f"tmutil addexclusion {index.parent}"))  # fmt: skip
    return out


def _index(path: Path) -> list[Check]:
    from lanternkb.store import SCHEMA_VERSION, Store

    if not path.exists():
        return [(WARN, f"no index yet at {path}", "lanternkb sync")]
    try:
        store = Store(path, readonly=True)
    except sqlite3.Error as e:
        return [(FAIL, f"index cannot be opened: {e}", "move it aside and run lanternkb sync")]
    out: list[Check] = []
    try:
        version = store.get_meta("schema_version")
        quick = store.con.execute("PRAGMA quick_check").fetchone()[0]
        if quick != "ok":
            out.append((FAIL, f"index damaged: {quick}", "move it aside and run lanternkb sync"))
        else:
            out.append((OK, f"index {path} (schema {version})", ""))
        if version is not None and int(version) < SCHEMA_VERSION:
            out.append((WARN, f"index schema {version} is older than {SCHEMA_VERSION}",
                        "the next sync upgrades it"))  # fmt: skip
        if store.get_meta("wal_pending") == "1":
            out.append((WARN, "old pages may remain in the index's -wal file",
                        "run lanternkb sync while no assistant is searching"))  # fmt: skip
        out += _orphans(store)
        if not store.vectors_ok:
            out.append((WARN, "the index predates version 7: keyword search only",
                        "lanternkb sync upgrades it"))  # fmt: skip
        else:
            for key in store.vector_models():
                pending = store.count_pending(key)
                if pending:
                    out.append((WARN, f"{pending:,} chunks still to embed ({key})",
                                "lanternkb sync finishes them"))  # fmt: skip
    finally:
        store.close()
    return out


def _orphans(store: object) -> list[Check]:
    """Data in the index from features nothing provides any more (an uninstalled plugin)."""
    from lanternkb.features import Registry

    known = set(Registry().manifests)
    stats = store.feature_stats()  # type: ignore[attr-defined]
    gone = {f for f, n in stats.items() if f not in known and (n.get("files") or n.get("chunks"))}
    if store.vectors_ok:  # type: ignore[attr-defined]
        gone |= {f"embedder.{k}" for k in store.vector_models()  # type: ignore[attr-defined]
                 if f"embedder.{k}" not in known}  # fmt: skip
    return [
        (
            WARN,
            f"the index still holds data from {f}, which is no longer installed "
            "(hidden from search)",
            f"lanternkb features purge {f}",
        )
        for f in sorted(gone)
    ]


def _reranking(cfg: object) -> list[Check]:
    from lanternkb import rerank
    from lanternkb.features import ManifestError, Registry

    try:
        key, why = rerank.availability(Registry(settings=cfg.features))  # type: ignore[attr-defined]
    except ManifestError:
        return []
    if key is None:
        return [(OK, "reranking is off (searches keep their own order)", "")]
    if why is None:
        return [(OK, f"reranking: {key}", "")] + _quick(cfg, key)
    if "on the CPU" in why:
        return [(OK, f"reranking is {why.split(':')[0]}", why.split(": ", 1)[1])]
    if "model missing" in why:
        return [(WARN, f"model for reranker {key} not downloaded (searches keep their own "
                 "order)", f"lanternkb models pull rerank:{key}")]  # fmt: skip
    return [(WARN, f"reranking {why}", "install lanternkb[embed]")]


def _quick(cfg: object, rerank_key: str) -> list[Check]:
    """Where torch runs the models, whether terminal searches can load the ONNX exports
    instead: about 0.5 to 2 s a search, against about 3 s."""
    import importlib.util

    from lanternkb import rerank
    from lanternkb.embed import EMBEDDERS, backend, enabled_embedders
    from lanternkb.features import Registry
    from lanternkb.models import model_path

    if backend() != "torch":
        return []
    keys = enabled_embedders(Registry(settings=cfg.features))  # type: ignore[attr-defined]
    spec = EMBEDDERS.get(keys[0]) if keys else None
    if spec is None or spec.onnx is None:
        return []
    missing = []
    if not (model_path(spec.onnx.repo) / spec.onnx.graph).is_file():
        missing.append(f"lanternkb models pull embed:{spec.key} --backend onnx")
    if not rerank.has_form(rerank.RERANKERS[rerank_key], "onnx"):
        missing.append(f"lanternkb models pull rerank:{rerank_key} --backend onnx")
    if any(importlib.util.find_spec(m) is None for m in ("onnxruntime", "tokenizers")):
        missing.insert(0, "install lanternkb[embed] again")
    if not missing:
        return [(OK, "terminal searches load the ONNX exports (quick)", "")]
    return [(OK, "terminal searches load torch each time (about 3 s a search)",
             "for about 1 s: " + "; ".join(missing))]  # fmt: skip


def _semantic(cfg: object) -> list[Check]:
    from lanternkb.embed import EMBEDDERS, backend, enabled_embedders, source
    from lanternkb.features import ManifestError, Registry
    from lanternkb.models import is_present

    try:
        keys = enabled_embedders(Registry(settings=cfg.features))  # type: ignore[attr-defined]
    except ManifestError:
        return []
    if not keys:
        return [(WARN, "semantic search is off", "lanternkb features enable "
                 "embedder.embeddinggemma-300m")]  # fmt: skip
    key = keys[0]
    out: list[Check] = []
    if len(keys) > 1:
        others = " ".join(f"embedder.{k}" for k in keys[1:])
        out.append((WARN, f"several embedders are on ({', '.join(keys)}); only {key} is used",
                    f"turn the others off to save memory and disk: lanternkb features "
                    f"disable {others}"))  # fmt: skip
    if key in EMBEDDERS:
        using = backend()
        src = source(EMBEDDERS[key], using) if using else None
        if using is None:
            out.append((WARN, "the embed extra is not installed", "install lanternkb[embed]"))
        elif src is None:
            out.append((WARN, f"{key} cannot run with ONNX Runtime",
                        "lanternkb features enable embedder.embeddinggemma-300m"))  # fmt: skip
        elif not is_present(src[0]):
            out.append((WARN, f"model for {key} not downloaded",
                        f"lanternkb models pull embed:{key}"))  # fmt: skip
    if not out:
        out.append((OK, f"semantic search: {key}", ""))
    return out


def _sandbox(cfg: object) -> list[Check]:
    from lanternkb.extract import pool

    if pool.sandbox_available():
        out: list[Check] = []
        # Workers may read Python's code folders; one inside an indexed folder would let
        # them read that folder's files too.
        for code in pool._code_dirs():
            for r in cfg.roots:  # type: ignore[attr-defined]
                root = str(r.path)
                if code == root or code.startswith(root + "/") or root.startswith(code + "/"):
                    out.append((WARN, f"extraction workers may read {code} (Python code), "
                                f"which overlaps the indexed folder {root}",
                                "install editable packages and plugins outside indexed "
                                "folders"))  # fmt: skip
        try:  # really start one: a profile this Python cannot run under stops every sync
            w = pool._Worker({"disabled_rules": [], "actions": {}}, True)
            applied, why = w.sandboxed, w.sandbox_error
            w.close()
        except Exception as e:  # noqa: BLE001 - reported, with what to send
            return [(FAIL, f"a sandboxed extraction worker cannot start here ({e})",
                     "a sync cannot extract until this is fixed: please report it, with "
                     "the path of the Python lanternkb runs on"), *out]  # fmt: skip
        if not applied:  # Linux: the worker could not sandbox itself
            problem = f"extraction workers could not sandbox themselves ({why})"
            return [(WARN, problem, "please report it, with `uname -r`"), *out]
        how = "Landlock and seccomp" if sys.platform.startswith("linux") else "sandbox-exec"
        return [(OK, f"extraction runs sandboxed ({how}: no network, no writes, no user "
                 "files)", ""), *out]  # fmt: skip
    return [(WARN, "no sandbox on this system: extractors run in plain worker processes",
             "on Linux, the sandbox needs Landlock (Linux 5.13 or later, enabled in the "
             "kernel's security modules)")]  # fmt: skip
