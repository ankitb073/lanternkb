"""The control API: the only place configuration changes are made.

Every change runs as one unit: take the lock; edit the TOML in place (comments survive) and
re-parse it with the strict config parser; open the index and take its write lock; mirror
the new state, re-apply tiers, add an audit row; write the config atomically (mode 600);
commit the index. Any failure rolls the index back and leaves the config file untouched.
The CLI and the settings UI are thin clients of this module. The MCP server must never
import it.
"""

from __future__ import annotations

import fcntl
import os
import sqlite3
import time
import tomllib
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

import tomlkit

from lanternkb import embed
from lanternkb.config import DEFAULT_CONFIG, Config, ConfigError, Root, parse
from lanternkb.features import Manifest, ManifestError, Registry
from lanternkb.fsutil import canonical_case, private_dirs
from lanternkb.ingest import IngestReport, embed_pending, mirror_features, retier, sync, tell
from lanternkb.store import OLD_INDEX, Store
from lanternkb.tiers import Tier

T = TypeVar("T")


class ControlError(ValueError):
    pass


@dataclass(frozen=True)
class FeatureInfo:
    manifest: Manifest
    enabled: bool  # what the config says
    tier: Tier
    files: int
    skipped: int
    chunks: int
    bytes: int
    applied: bool  # False if the index does not reflect the config yet (run a sync)


@dataclass(frozen=True)
class PurgePlan:
    feature: str
    files: int
    skipped: int
    chunks: int
    bytes: int
    done: bool  # False: a preview only; nothing was deleted
    wal_clean: bool = True  # False: a reader kept old pages in the WAL for now


class Control:
    def __init__(self, config_path: Path = DEFAULT_CONFIG):
        given = Path(os.path.expanduser(config_path))
        # Follow a symlinked config (dotfiles) so we write the real file and lock one place.
        self.config_path = given.resolve() if given.is_symlink() else given

    # -- reading ------------------------------------------------------------------------------

    def config(self) -> Config:
        try:
            if not self.config_path.exists():
                return replace(parse({}, self.config_path.parent), source=self.config_path)
            data = tomllib.loads(self.config_path.read_text())
            return replace(parse(data, self.config_path.parent), source=self.config_path.resolve())
        except (ConfigError, tomllib.TOMLDecodeError) as e:
            raise ControlError(f"{self.config_path}: {e}") from e

    def registry(self, cfg: Config | None = None) -> Registry:
        try:
            return Registry(settings=(cfg or self.config()).features)
        except ManifestError as e:
            raise ControlError(str(e)) from e

    def features(self) -> list[FeatureInfo]:
        cfg = self.config()
        reg = self.registry(cfg)

        def read(s: Store) -> tuple[dict[str, dict[str, int]], dict[str, tuple[bool, str]]]:
            return s.feature_stats(), s.feature_states()

        stats, index_state = self._read(cfg, read) or ({}, {})
        out = []
        for fid, m in sorted(reg.manifests.items(), key=lambda kv: (kv[1].category, kv[0])):
            st = reg.state(fid)
            n = stats.get(fid, {})
            counts = [n.get(k, 0) for k in ("files", "skipped", "chunks", "bytes")]
            applied = index_state.get(fid) == (st.enabled, st.tier.value)
            out.append(FeatureInfo(m, st.enabled, st.tier, *counts, applied))
        return out

    def status(self) -> dict[str, Any]:
        cfg = self.config()
        key, why = embed.availability(self.registry(cfg))
        semantic: dict[str, Any] = {"embedder": key, "available": key is not None and not why}
        if why:
            semantic["note"] = why
        from lanternkb import rerank

        rkey, rwhy = rerank.availability(self.registry(cfg))
        reranking: dict[str, Any] = {"reranker": rkey, "available": rkey is not None and not rwhy}
        if rwhy:
            reranking["note"] = rwhy

        def read(s: Store) -> dict[str, Any]:
            if key is not None and s.vectors_ok and key not in s.vector_models():
                semantic["pending"] = s.con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            rows = list(s.con.execute("SELECT status, reason, tier FROM files"))
            indexed = [r for r in rows if r[0] == "indexed"]
            skipped = [(r[1] or "").split(":")[0] for r in rows if r[0] == "skipped"]
            return {
                "index": str(cfg.index_path),
                "last_sync": s.get_meta("last_sync"),
                "rules_hash": s.get_meta("rules_hash"),
                "indexed": len(indexed),
                "by_tier": dict(Counter(r[2] for r in indexed)),
                "skipped": dict(Counter(skipped)),
                "chunks": s.con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
                "semantic_search": semantic,
                "reranking": reranking,
                "embeddings": _embedding_status(s),
                "wal_pending": s.get_meta("wal_pending") == "1",
            }

        return self._read(cfg, read) or {"index": str(cfg.index_path), "indexed": 0}

    def skipped(self) -> list[tuple[str, str]]:
        return self._read(self.config(), lambda s: s.skipped()) or []

    def notes(self) -> list[tuple[str, str]]:
        """Indexed files with an extraction quality note (for example, pages with no text)."""
        return self._read(self.config(), lambda s: s.notes()) or []

    def roots(self) -> tuple[Root, ...]:
        return self.config().roots

    # -- changing features --------------------------------------------------------------------

    def enable(self, feature_id: str) -> None:
        self._set_feature(feature_id, "enabled", True)

    def disable(self, feature_id: str) -> None:
        reg = self.registry()
        if feature_id in reg.manifests and reg.manifests[feature_id].required:
            raise ControlError(f"{feature_id} is required and cannot be switched off")
        self._set_feature(feature_id, "enabled", False)

    def set_feature_tier(self, feature_id: str, tier: str) -> None:
        self._set_feature(feature_id, "tier", self._tier(tier).value)

    def purge(self, feature_id: str, confirm: bool = False) -> PurgePlan:
        """Without confirm: say what would be deleted. With confirm: delete it, securely."""
        self._need_config()
        reg = self.registry()
        if feature_id not in reg.manifests and not self._has_data(feature_id):
            raise ControlError(f"unknown feature {feature_id!r}")
        # A feature nothing provides any more (an uninstalled plugin) can still be purged.
        if feature_id in reg.manifests and reg.manifests[feature_id].required:
            raise ControlError(f"{feature_id} is required; its data cannot be purged")
        with self._lock(wait_s=self.CHANGE_WAIT_S):
            cfg = self.config()
            if not confirm:
                st = (self._read(cfg, lambda s: s.feature_stats()) or {}).get(feature_id, {})
                return PurgePlan(feature_id, st.get("files", 0), st.get("skipped", 0),
                                 st.get("chunks", 0), st.get("bytes", 0), done=False)  # fmt: skip
            if not cfg.index_path.exists():
                return PurgePlan(feature_id, 0, 0, 0, 0, done=True)
            store = self._writer(cfg)
            try:
                st = store.purge_feature(feature_id)
                store.audit("control", str(self.config_path), f"purge:{feature_id}",
                            count=st.get("files", 0) + st.get("skipped", 0))  # fmt: skip
            except BaseException:
                store.close(commit=False)
                raise
            clean = self._close_and_checkpoint(store)
            return PurgePlan(feature_id, st.get("files", 0), st.get("skipped", 0),
                             st.get("chunks", 0), st.get("bytes", 0), True, clean)  # fmt: skip

    def _has_data(self, feature_id: str) -> bool:
        """The index holds files, chunks or vectors of this feature id."""

        def read(s: Store) -> bool:
            if feature_id.startswith("embedder.") and s.vectors_ok:
                return feature_id.split(".", 1)[1] in s.vector_models()
            return feature_id in s.feature_stats()

        return bool(self._read(self.config(), read))

    # -- changing roots -----------------------------------------------------------------------

    def add_root(self, path: str, tier: str = Tier.LOCAL_ONLY.value) -> None:
        p = canonical_case(Path(os.path.expanduser(path)).resolve())
        if not p.is_dir():
            raise ControlError(f"{path} is not a folder")
        tier = self._tier(tier).value
        with self._lock(wait_s=self.CHANGE_WAIT_S):
            doc = self._doc()
            roots = _root_tables(doc)
            if roots is None:
                roots = doc["roots"] = tomlkit.aot()
            if any(self._same(r["path"], p) for r in roots):
                raise ControlError(f"{p} is already a root")
            t = tomlkit.table()
            t["path"] = str(p)  # absolute, as it is on disk: never read relative to anything
            t["tier"] = tier
            roots.append(t)
            self._commit(doc, f"add-root:{tier}")

    def remove_root(self, path: str, force: bool = False) -> int:
        """Remove a root and delete its indexed data at once. Returns the files removed.

        Refused (unless force) when the root sits inside another root that would make its
        files more visible: removing `~/notes/health` (local-only) from inside `~/notes`
        (open) would turn the health notes open at the next sync.
        """
        self._need_config()
        gone = canonical_case(Path(os.path.expanduser(path)).resolve())
        with self._lock(wait_s=self.CHANGE_WAIT_S):
            cfg = self.config()
            target = next((r for r in cfg.roots if r.path == gone), None)
            if target is None:
                raise ControlError(f"{path} is not a root")
            outer = [r for r in cfg.roots if r is not target and gone.is_relative_to(r.path)]
            looser = [r for r in outer if r.tier == Tier.OPEN and target.tier == Tier.LOCAL_ONLY]
            if looser and not force:
                raise ControlError(
                    f"{gone} is inside the open root {looser[0].path}; removing it would make "
                    "its files open. Add a tier rule or a deny pattern instead, or use --force."
                )
            doc = self._doc()
            new, carry = tomlkit.aot(), []
            for r in _root_tables(doc) or []:
                if not self._same(r["path"], gone):
                    new.append(r)
                    continue
                # A comment written above the next table is stored at the end of this one:
                # keep it at the end of the table before, or, for the
                # first root, where the roots begin.
                tail = _trailing_comments(r)
                if len(new):
                    new[-1].value.body.extend(tail)
                else:
                    carry = tail
            at = next(i for i, (k, _v) in enumerate(doc.body) if k == "roots")
            if len(new):
                doc["roots"] = new
            else:
                del doc["roots"]
            doc.body[at:at] = carry

            def purge_root(s: Store) -> int:
                paths = [r[0] for r in s.con.execute("SELECT path FROM files WHERE root=?",
                                                      (str(gone),))]  # fmt: skip
                for fp in paths:
                    s.remove(fp)
                    s.con.execute("DELETE FROM audit WHERE path=?", (fp,))
                s.optimise_fts()
                s.audit("control", str(gone), "remove-root-data", count=len(paths))
                return len(paths)

            return self._commit(doc, "remove-root", extra=purge_root) or 0

    def set_root_tier(self, path: str, tier: str) -> None:
        self._need_config()
        tier = self._tier(tier).value
        with self._lock(wait_s=self.CHANGE_WAIT_S):
            doc = self._doc()
            for r in _root_tables(doc) or []:
                if self._same(r["path"], Path(os.path.expanduser(path))):
                    r["tier"] = tier
                    self._commit(doc, f"root-tier:{tier}")
                    return
            raise ControlError(f"{path} is not a root")

    # -- sync ---------------------------------------------------------------------------------

    EMBED_BATCH = 512  # chunks embedded per hold of the lock (a few seconds)
    EMBED_YIELD_S = 0.3  # pause between turns, so a waiting settings change gets the lock
    LOCK_WAIT_S = 60.0  # how long the embedding pass waits for a settings change to finish
    CHANGE_WAIT_S = 30.0  # how long a settings change waits for an embedding turn to end

    def sync(self, progress: Any = None) -> IngestReport:
        """Index changes (keyword search works when this returns), then embed what is
        pending in short turns of the lock, so settings changes are not refused for the
        whole of a long embedding run (cheap first, docs/efficiency.md). `progress(stage,
        done, total)` hears how far it is (`ingest.sync`, then "embedding" passages)."""
        self._need_config()
        with self._lock():
            cfg = self.config()
            if not cfg.roots:
                raise ControlError("no roots configured; add one with `lanternkb roots add`")
            registry = self.registry(cfg)
            try:
                registry.check_known()  # never sync away a missing plugin's files
            except ManifestError as e:
                raise ControlError(str(e)) from e
            store = self._writer(cfg)
            try:
                store.con.execute("DELETE FROM meta WHERE key='wal_pending'")
                report = sync(cfg, store, registry=registry, progress=progress)
            except BaseException:
                store.close(commit=False)  # never keep a half-indexed file
                raise
            report.wal_clean = self._close_and_checkpoint(store)
        self._embed(cfg, registry, report, progress)
        return report

    def _close_and_checkpoint(self, store: Store) -> bool:
        """Close, retrying the WAL checkpoint while a reader (an MCP server) holds a
        snapshot, so removed or redacted text does not stay in the -wal file.
        If a reader still blocks it, record `wal_pending`: the next sync tries again."""
        path = store.path
        if _close_cleanly(store, tries=2):  # each attempt waits up to Store.BUSY_MS
            return True
        marker = Store(path)
        marker.set_meta("wal_pending", "1")
        marker.commit()
        marker.close()
        return False

    def _embed(self, cfg: Config, registry: Registry, report: IngestReport,
               progress: Any = None) -> None:  # fmt: skip
        keys = embed.enabled_embedders(registry)
        if not keys:
            report.embed_note = "semantic search is off (no embedder feature is enabled)"
            return
        key = report.embedder = keys[0]
        several = f"several embedders are enabled; using {key}. " if len(keys) > 1 else ""
        state = self._read(cfg, lambda s: _embedding_state(s, key))
        if state is None or state == "done":
            report.embed_note = several.strip()
            return
        if state == "no-sqlite-vec":
            report.embed_note = several + "semantic search is unavailable: " + OLD_INDEX
            return
        embedder, why = embed.active_embedder(registry)  # loads the model: only when needed
        if embedder is None:
            report.embed_note = several + why
            report.pending = self._read(cfg, lambda s: _pending(s, key)) or 0
            return
        while True:
            part = IngestReport()
            try:
                with self._lock(wait_s=self.LOCK_WAIT_S):
                    store = self._writer(cfg)
                    try:
                        embed_pending(store, embedder, part, limit=self.EMBED_BATCH)
                    except BaseException:
                        store.close(commit=False)
                        raise
                    report.wal_clean = self._close_and_checkpoint(store) and report.wal_clean
            except ControlError:
                report.embed_note = several + "embedding paused: another sync or settings " \
                    "change is running; the next sync continues it"  # fmt: skip
                report.pending = self._read(cfg, lambda s: _pending(s, key)) or 0
                return
            report.embedded += part.embedded
            report.reused += part.reused
            report.failed += part.failed
            report.pending = part.pending
            done = report.embedded + report.reused + report.failed
            tell(progress, "embedding", done, done + part.pending)
            if part.embed_note or part.pending == 0 or part.embedded + part.reused == 0:
                report.embed_note = several + part.embed_note
                return
            time.sleep(self.EMBED_YIELD_S)  # let a waiting settings change go first

    # -- internals ----------------------------------------------------------------------------

    @staticmethod
    def _tier(tier: str) -> Tier:
        try:
            return Tier(tier)
        except ValueError as e:
            raise ControlError(f"unknown tier {tier!r}; use 'open' or 'local-only'") from e

    def _need_config(self) -> None:
        if not self.config_path.exists():
            raise ControlError(f"no config at {self.config_path}; start with `lanternkb roots add`"
                               " (or check the --config path)")  # fmt: skip

    def _doc(self) -> tomlkit.TOMLDocument:
        if self.config_path.exists():
            return tomlkit.parse(self.config_path.read_text())
        return tomlkit.document()

    def _set_feature(self, feature_id: str, key: str, value: object) -> None:
        self._need_config()
        if feature_id not in self.registry().manifests:
            raise ControlError(f"unknown feature {feature_id!r}")
        with self._lock(wait_s=self.CHANGE_WAIT_S):
            doc = self._doc()
            if "features" not in doc:
                doc["features"] = tomlkit.table(is_super_table=True)
            feats = doc["features"]
            if feature_id not in feats:
                feats[feature_id] = tomlkit.table()
            feats[feature_id][key] = value
            self._commit(doc, f"{key}={value}:{feature_id}")

    def _commit(self, doc: tomlkit.TOMLDocument, action: str,
                extra: Callable[[Store], T] | None = None) -> T | None:  # fmt: skip
        """Apply a config change to the index and the file as one unit (see module doc)."""
        text = tomlkit.dumps(doc)
        try:
            cfg = replace(parse(tomllib.loads(text), self.config_path.parent),
                          source=self.config_path.resolve())  # fmt: skip
            reg = Registry(settings=cfg.features)
        except (ConfigError, ManifestError, tomllib.TOMLDecodeError) as e:
            raise ControlError(f"change rejected, config left as it was: {e}") from e
        store = self._writer(cfg)
        try:
            mirror_features(reg, store)
            retier(cfg, store, reg)
            result = extra(store) if extra else None
            store.audit("control", str(self.config_path), action)
            _atomic_write(self.config_path, text)
        except BaseException as e:
            store.close(commit=False)
            if isinstance(e, sqlite3.Error):
                raise ControlError(f"index busy or unwritable, nothing changed: {e}") from e
            raise
        self._close_and_checkpoint(store)  # hidden or removed data must leave the -wal too
        return result

    def _writer(self, cfg: Config) -> Store:
        """Open the index for writing, holding its write lock, and check who owns it."""
        try:
            store = Store(cfg.index_path)
            store.begin_immediate()
        except sqlite3.Error as e:
            raise ControlError(f"index busy or unwritable, nothing changed: {e}") from e
        owner = store.get_meta("config_path")
        me = str(self.config_path.resolve())
        if owner is None:
            store.set_meta("config_path", me)
        elif owner != me:
            store.close(commit=False)
            raise ControlError(f"{cfg.index_path} belongs to the config {owner}, not {me}; "
                               "check the --config path")  # fmt: skip
        return store

    def _read(self, cfg: Config, fn: Callable[[Store], T]) -> T | None:
        if not cfg.index_path.exists():
            return None
        store = Store(cfg.index_path, readonly=True)
        try:
            return fn(store)
        finally:
            store.close()

    def _same(self, entry: object, p: Path) -> bool:
        """Whether a root entry in the file names p, reading it as the config parser does."""
        e = Path(os.path.expanduser(str(entry)))
        if not e.is_absolute():
            e = self.config_path.parent / e
        return canonical_case(e.resolve()) == canonical_case(p.resolve())

    @contextmanager
    def _lock(self, wait_s: float = 0.0) -> Iterator[None]:
        """One writer at a time, for config changes and syncs alike."""
        private_dirs(self.config_path.parent)
        lock = self.config_path.with_name(self.config_path.name + ".lock")
        fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            deadline = time.monotonic() + wait_s
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError as e:
                    if time.monotonic() >= deadline:
                        raise ControlError("another sync or settings change is running") from e
                    time.sleep(0.1)
            yield
        finally:
            os.close(fd)


def _root_tables(doc: tomlkit.TOMLDocument) -> Any:
    """The `[[roots]]` tables, or None. A one-line `roots = [{...}, ...]` array is rewritten
    as tables first, so it can be changed the same way."""
    from tomlkit.items import AoT

    roots = doc.get("roots")
    if roots is None or isinstance(roots, AoT):
        return roots
    tables = tomlkit.aot()
    for item in roots:
        t = tomlkit.table()
        for k, v in item.items():
            t[k] = v
        tables.append(t)
    doc["roots"] = tables
    return tables


def _trailing_comments(table: Any) -> list[Any]:
    """The comments (and blank lines between them) at the end of a TOML table's body, if
    there is at least one comment."""
    from tomlkit.items import Comment, Whitespace

    body = table.value.body
    i = len(body)
    while i > 0 and body[i - 1][0] is None and isinstance(body[i - 1][1], Comment | Whitespace):
        i -= 1
    tail = body[i:]
    return tail if any(isinstance(item, Comment) for _k, item in tail) else []


def _embedding_status(store: Store) -> dict[str, Any]:
    """Per embedder in the index: model, vectors stored, chunks pending and rejected."""
    if not store.vectors_ok:
        return {"error": OLD_INDEX}
    out: dict[str, Any] = {}
    for key, (model_id, dim) in store.vector_models().items():
        n = store.con.execute(f"SELECT COUNT(*) FROM {store.vec_table(key)}").fetchone()[0]
        out[key] = {"model": model_id, "dim": dim, "vectors": int(n),
                    "pending": store.count_pending(key), "failed": store.count_failed(key),
                    "complete": store.vectors_complete(key)}  # fmt: skip
    return out


def _embedding_state(store: Store, key: str) -> str:
    """'done', 'pending' (or a model change), or 'no-sqlite-vec', without loading a model."""
    if not store.vectors_ok:
        return "no-sqlite-vec"
    spec = embed.EMBEDDERS.get(key)
    have = store.vector_models().get(key)
    if spec is None or have != (embed.index_model_id(spec), spec.dim):
        return "pending"
    return "pending" if store.count_pending(key) else "done"


def _pending(store: Store, key: str) -> int:
    if not store.vectors_ok or key not in store.vector_models():
        return int(store.con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    return store.count_pending(key)


def _close_cleanly(store: Store, tries: int = 5) -> bool:
    """Close, retrying the WAL checkpoint briefly if a reader holds an old snapshot."""
    path = store.path
    if store.close():
        return True
    for _ in range(tries):
        time.sleep(0.2)
        s = Store(path)
        if s.close():
            return True
    return False


def _atomic_write(path: Path, text: str) -> None:
    """Write text to path atomically, readable only by the owner."""
    private_dirs(path.parent)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
