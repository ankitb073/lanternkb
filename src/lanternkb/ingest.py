"""The indexing pipeline: policy, read, scan, skip or redact, tier, chunk, store, embed.

Redaction happens before chunking and embedding, so only redacted text is ever stored,
tokenised or embedded. Logging carries paths, rule ids and counts only.
One bad file never stops a sync: it is recorded as skipped with an error reason.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any

from lanternkb import fsutil
from lanternkb.chunking import chunk_segments
from lanternkb.config import Config
from lanternkb.embed import BATCH_TIMEOUT_S, Embedder, with_deadline
from lanternkb.extract.pool import MEDIA_FEATURES, WorkerPool, default_workers
from lanternkb.extract.worker import Job, Result, process
from lanternkb.features import Registry
from lanternkb.plugins import discover as discover_plugins
from lanternkb.policy import Decision, PathPolicy
from lanternkb.scan.engine import Scanner, ScanTimeout, redact
from lanternkb.scan.rules import load_rules
from lanternkb.store import OLD_INDEX, FileRow, Store
from lanternkb.tiers import ContentTierRule, SecretsMakeLocalOnly, Tier, content_tier, stricter

log = logging.getLogger("lanternkb.ingest")
BINARY_SNIFF = 8192


@dataclass
class IngestReport:
    seen: int = 0
    indexed: int = 0
    hidden: int = 0  # kept but not visible: their feature is disabled
    unchanged: int = 0
    retiered: int = 0
    removed: int = 0
    errors: int = 0
    skipped: Counter[str] = field(default_factory=Counter)
    redacted_files: int = 0
    redactions: Counter[str] = field(default_factory=Counter)
    rules_hash: str = ""
    isolation: str = "none"  # "sandboxed" | "process" (no sandbox here) | "inline" | "none"
    wal_clean: bool = True  # False: a reader kept old pages in the WAL; the next sync retries
    embedder: str = ""  # the embedder used, or "" (keyword-only)
    embedded: int = 0  # chunk texts embedded this sync
    reused: int = 0  # chunks given an existing vector for identical input (no embedding)
    pending: int = 0  # chunks still without a vector (search falls back to keywords for them)
    failed: int = 0  # chunk inputs the embedder rejected (found by keywords only)
    embed_note: str = ""  # why semantic search is off or incomplete, if it is
    warnings: list[str] = field(default_factory=list)  # e.g. a tier rule that did not load


def mirror_features(registry: Registry, store: Store) -> None:
    """Copy every feature's state into the index, where readers take visibility from. A
    feature the index knows but no installed plugin provides is switched off there: its
    data is hidden until the plugin is back (fail closed), and kept for a purge."""
    for fid in registry.manifests:
        st = registry.state(fid)
        store.set_feature(fid, st.enabled, st.tier.value)
    for fid, (on, tier) in store.feature_states().items():
        if on and fid not in registry.manifests:
            store.set_feature(fid, False, tier)


def run_jobs(jobs: list[Job], scanner: Scanner, config: Config, inline: bool,
             report: IngestReport | None = None) -> Iterator[tuple[Job, Result]]:  # fmt: skip
    """Extract, scan and redact: in sandboxed worker processes, or in this process (tests,
    evaluation). Both run `worker.process`, so the results are the same."""
    if inline:
        if report is not None and jobs:
            report.isolation = "inline"
            log.warning("extracting in this process: no sandbox, no limits (tests or evaluation)")
        for job in jobs:
            yield job, process(job, scanner)
        return
    # Workers never mix: images and audio run on media workers (the
    # media sandbox profile); each plugin on workers of its own, so it never sees, or
    # tampers with the scanning of, another feature's files; and files whose path is
    # local-only on other workers than open ones, so a parser taken over by one file cannot
    # carry a local-only file's text into an open file's output.
    plugin_ids = set(discover_plugins().manifests)
    groups: dict[tuple[str, str], list[Job]] = {}
    for j in jobs:
        kind = ("media" if j.feature in MEDIA_FEATURES
                else f"plugin:{j.feature}" if j.feature in plugin_ids else "plain")  # fmt: skip
        groups.setdefault((kind, j.tier), []).append(j)
    n = config.limits.workers or default_workers()
    for (kind, tier), batch in sorted(groups.items()):
        is_media = kind == "media"

        # Output that could reach an open server, and every plugin's output, is scanned
        # again by a process that never parsed a file.
        def again(job: Job, kind: str = kind) -> bool:
            return kind.startswith("plugin:") or job.tier == "open"

        pool = WorkerPool(min(n, 2) if is_media else n, config.scanner.disabled_rules,
                          config.scanner.actions, media=is_media, rescan=again,
                          enabled_rules=config.scanner.enabled_rules,
                          open_tier=tier == "open")  # fmt: skip
        for job, result in pool.run(batch):
            if report is not None:
                report.isolation = "sandboxed" if pool.sandboxed else "process"
            if pool.rules_hash != scanner.hash:
                raise RuntimeError("worker rule set differs from the sync's; refusing to index")
            yield job, result


def sync(
    config: Config,
    store: Store,
    scanner: Scanner | None = None,
    embedder: Embedder | None = None,
    registry: Registry | None = None,
    isolate: bool | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> IngestReport:
    """Bring the index in line with the roots. Safe to run repeatedly.

    Extraction runs in sandboxed worker processes unless isolate=False, or a
    scanner object is passed in (which can only run in this process). `progress(stage,
    done, total)` hears how far it is: "checking" files (total 0: not known yet), then
    "reading" the ones that changed; it never sees a path.
    """
    inline = (scanner is not None) if isolate is None else not isolate
    scanner = scanner or Scanner(
        load_rules(
            disabled=config.scanner.disabled_rules,
            actions=config.scanner.actions,
            enabled=config.scanner.enabled_rules,
        )
    )
    fsutil.clear_cache()
    registry = registry or Registry(settings=config.features)
    registry.check_known()  # a missing plugin's files must not be synced away
    mirror_features(registry, store)
    policy = PathPolicy(config, registry)
    report = IngestReport(rules_hash=scanner.hash)
    report.warnings += [f"tier rule {r.feature} did not load ({r.error}): every file is "
                        "local-only until it does" for r in plugin_tier_rules(registry)
                        if isinstance(r, _Unloadable)]  # fmt: skip
    rules_changed = store.get_meta("rules_hash") != scanner.hash
    if rules_changed:
        log.info("rule set changed: re-scanning every file")
    if store.get_meta("chunk_layout") != chunk_layout():  # an older index: unknown layout
        log.info("chunking changed: re-reading every file")
        rules_changed = True
    versions = extractor_versions(registry)
    stale = stale_features(store, versions)
    if stale:
        log.info("extractors changed: re-reading their files (%s)", ", ".join(sorted(stale)))
    seen: set[str] = set()
    jobs: list[Job] = []
    decisions: dict[int, Decision] = {}
    for d in policy.walk():  # pass 1: decide (metadata only)
        report.seen += 1
        if report.seen % 256 == 0:
            tell(progress, "checking", report.seen, 0)
        path = str(d.path)
        seen.add(path)
        store.savepoint()
        try:
            job = _decide(d, path, config, store, scanner, report, rules_changed, registry,
                          len(jobs), stale)  # fmt: skip
            store.release()
        except Exception as e:  # one bad file must not stop the sync
            store.rollback_to()
            _error(store, report, d, path, e)
            continue
        if job is not None:
            jobs.append(job)
            decisions[job.id] = d
    tell(progress, "checking", report.seen, report.seen)
    for n, (job, result) in enumerate(run_jobs(jobs, scanner, config, inline, report), 1):
        tell(progress, "reading", n, len(jobs))  # pass 2: extract
        d = decisions[job.id]
        store.savepoint()
        try:
            _apply(d, str(d.path), result, store, scanner, embedder, report, registry, stale)
            store.release()
        except Exception as e:
            store.rollback_to()
            _error(store, report, d, str(d.path), e)
    for path in store.all_paths():  # includes files under roots that were removed
        if path not in seen:
            store.remove(path)
            store.audit("removed", None)  # counted, not named: the file is gone
            report.removed += 1
    report.warnings += [f"tier rule {r.feature} took too long; files it had not judged are "
                        "local-only until the next sync" for r in plugin_tier_rules(registry)
                        if isinstance(r, _Guarded) and (r.hung or r.over_budget)]  # fmt: skip
    store.set_meta("rules_hash", scanner.hash)
    store.set_meta("text_rules", text_rules_signature(registry))  # every file was read with them
    store.set_meta("chunk_layout", chunk_layout())
    store.set_meta("extract_versions", json.dumps(versions, sort_keys=True))
    store.set_meta("last_sync", _now_iso())
    if store.deleted_since_optimise:
        store.optimise_fts()  # rewrite the keyword index so deleted terms leave the file
    store.commit()  # cheap first: keyword search works from here, before any embedding
    if embedder is not None:
        embed_pending(store, embedder, report)
    log.info("sync: %d seen, %d indexed, %d unchanged, %d skipped, %d removed, %d errors, "
             "%d embedded, %d reused, %d pending",
             report.seen, report.indexed, report.unchanged, sum(report.skipped.values()),
             report.removed, report.errors, report.embedded, report.reused,
             report.pending)  # fmt: skip
    return report


def tell(progress: Callable[[str, int, int], None] | None, stage: str, done: int,
         total: int) -> None:  # fmt: skip
    """Report progress; a failing listener never stops the work."""
    if progress is not None:
        try:
            progress(stage, done, total)
        except Exception:  # noqa: BLE001 - progress is for people, not for correctness
            log.debug("progress listener failed")


def chunk_layout() -> str:
    """What decides a file's chunks and their hashes; a change means re-reading every file."""
    from lanternkb import chunking, embed

    return (f"chars={chunking.MAX_CHARS};code_lines={chunking.CODE_MAX_LINES};"
            f"long_lines={int(chunking.SPLIT_LONG_LINES)};"
            f"input={embed.INPUT}.{embed.INPUT_FORMAT};path_scan=1;idents=2")  # fmt: skip


def embed_pending(store: Store, embedder: Embedder, report: IngestReport,
                  batch: int = 256, limit: int | None = None) -> None:  # fmt: skip
    """Give every visible chunk a vector, committing after each batch so progress is kept.

    Identical inputs are embedded once: a chunk whose input already has a vector (an
    unchanged chunk in an edited file, or a copy elsewhere) reuses it. If a batch fails,
    its inputs are tried one by one: an input the model rejects is recorded and skipped
    from then on (it stays findable by keywords), so one bad input cannot stall the rest.
    If a plain probe fails too, the embedder itself is failing: the pass stops, never the
    sync, and the chunks stay pending. `limit` caps how many chunks are handled this time.
    """
    key = embedder.key
    report.embedder = key
    if not store.vectors_ok:
        report.embed_note = "semantic search is unavailable: " + OLD_INDEX
        return
    store.ensure_vectors(key, embedder.model_id, embedder.dim)
    store.commit()
    done = 0
    try:
        while limit is None or done < limit:
            rows = store.pending_chunks(key, batch if limit is None else min(batch, limit - done))
            if not rows:
                break
            have = store.vectors_for_hashes(key, sorted({h for _, h, _ in rows}))
            todo: dict[str, str] = {}
            for _cid, h, text in rows:
                if h not in have:
                    todo.setdefault(h, text)
            failed: list[str] = []
            if todo:
                got = _embed(embedder, list(todo.values()))
                if got is None:  # tried one by one
                    got = [_embed_one(embedder, t) for t in todo.values()]
                    if all(v is None for v in got) and _embed_one(embedder, "probe") is None:
                        raise _EmbedderFailing(_last_error.get("name", "ValueError"))
                for h, v in zip(todo, got, strict=True):
                    if v is None:
                        failed.append(h)
                    else:
                        have[h] = v
            put = [(cid, have[h]) for cid, h, _ in rows if h in have]
            store.put_vectors(key, put)
            if failed:
                store.record_failures(key, failed)
            store.commit()
            report.embedded += len(todo) - len(failed)
            report.failed += len(failed)
            report.reused += len(put) - (len(todo) - len(failed))
            done += len(rows)
    except Exception as e:  # noqa: BLE001 - never fail a sync over embeddings
        store.rollback()
        store.commit()
        name = e.cause if isinstance(e, _EmbedderFailing) else type(e).__name__
        report.embed_note = f"embedding stopped: {name}"
        log.warning("embedding stopped (%s); keyword search still covers every file", name)
    report.pending = store.mark_complete_if_done(key)
    store.commit()
    if report.failed:
        log.warning("%d chunk inputs could not be embedded; keyword search covers them",
                    report.failed)  # fmt: skip


class _EmbedderFailing(RuntimeError):
    def __init__(self, cause: str):
        super().__init__(cause)
        self.cause = cause  # the exception type's name, never a message (it may hold text)


_last_error: dict[str, str] = {}


def _embed(embedder: Embedder, texts: list[str]) -> list[bytes] | None:
    """Vectors for texts as float32 bytes, or None if the batch failed or was malformed."""
    import numpy as np

    try:
        vecs = with_deadline(embedder, "embed_documents", texts, BATCH_TIMEOUT_S,
                             as_array=True)  # fmt: skip
    except Exception as e:  # noqa: BLE001 - the caller retries one by one
        _last_error["name"] = type(e).__name__
        return None
    if vecs.shape != (len(texts), embedder.dim) or not np.isfinite(vecs).all():
        _last_error["name"] = "ValueError"  # malformed output
        return None
    return [v.tobytes() for v in vecs]


def _embed_one(embedder: Embedder, text: str) -> bytes | None:
    got = _embed(embedder, [text])
    return got[0] if got else None


def _error(store: Store, report: IngestReport, d: Decision, path: str, e: Exception) -> None:
    report.errors += 1
    log.warning("error on %s: %s", d.rel or path, type(e).__name__)
    root = str(d.root.path) if d.root else None
    rel = d.rel.as_posix() if d.rel else None
    _skip(store, report, path, f"error:{type(e).__name__}", root, rel, d.size, None,
          feature=d.feature)  # fmt: skip


def extractor_versions(registry: Registry) -> dict[str, int]:
    """Each extractor feature's output version (`version` in its manifest)."""
    return {fid: m.version for fid, m in registry.manifests.items() if m.kind == "extractor"}


def stale_features(store: Store, versions: dict[str, int]) -> frozenset[str]:
    """Extractors whose version differs from the one the index last synced with: their
    files are read again. An index from before versions were kept counts them all as 1."""
    try:
        before = json.loads(store.get_meta("extract_versions") or "{}")
    except ValueError:
        before = None
    if not isinstance(before, dict):
        return frozenset(versions)  # unreadable: read every extractor's files again, once
    return frozenset(f for f, v in versions.items() if before.get(f, 1) != v)


def _decide(d: Decision, path: str, config: Config, store: Store, scanner: Scanner,
            report: IngestReport, rules_changed: bool, registry: Registry,
            next_id: int, stale: frozenset[str] = frozenset()) -> Job | None:  # fmt: skip
    """Handle everything that needs no reading; return a Job for files that must be read."""
    root = str(d.root.path) if d.root else None
    rel = d.rel.as_posix() if d.rel else None
    if not d.allowed and (d.reason or "").startswith("feature-disabled:"):
        old = store.get_file(path)
        if old is not None and old.feature == d.feature:
            if old.status == "indexed" and (rules_changed or old.rules_hash != scanner.hash
                                            or old.feature in stale):  # fmt: skip
                # Scanned by an older rule set, or chunked by an older layout: it
                # must never come back as it is, and it cannot be read again while the
                # feature is off. Delete it; turning the feature on reads it afresh.
                _skip(store, report, path, d.reason or "", root, rel, d.size, None,
                      feature=d.feature)  # fmt: skip
                return None
            report.hidden += 1  # keep its rows, hidden; nothing is read
            return None
        _skip(store, report, path, d.reason or "", root, rel, d.size, None, feature=d.feature)
        return None
    if not d.allowed:
        _skip(store, report, path, d.reason or "skipped", root, rel, d.size, None,
              feature=d.feature)  # fmt: skip
        return None
    assert d.root is not None and d.rel is not None and d.tier is not None
    rules = _content_rules(d, registry)
    old = store.get_file(path)
    if (old and old.feature == d.feature and old.feature not in stale
            and _unchanged(old, d, rules_changed, scanner.hash)):  # fmt: skip
        if old.status == "skipped":  # a secret-file skip: no need to read it again
            report.skipped["secret"] += 1
            return None
        if not any(r.needs_text for r in rules):
            tier = content_tier(d.tier, d.rel, "", old.n_secrets, rules)
            if tier != Tier.LOCAL_ONLY and old.n_secrets and _path_secrets(scanner, d.rel):
                tier = Tier.LOCAL_ONLY
            if not _looser(tier, old):
                if (old.path_tier, old.tier) != (d.tier.value, tier.value):
                    store.set_tier(path, d.tier.value, tier.value)
                    report.retiered += 1
                else:
                    report.unchanged += 1
                return None
            # Open, but not read on an open worker: read it again, on one.
    assert d.feature is not None
    m = registry.manifests[d.feature]
    max_chars = min(config.limits.max_text_chars, m.limits.output_mb * 1_000_000)
    return Job(next_id, str(d.real_path or d.path), rel or "", d.feature, d.dev, d.ino,
               config.limits.max_file_bytes,
               {"cpu_seconds": m.limits.cpu_seconds, "memory_mb": m.limits.memory_mb,
                "output_mb": m.limits.output_mb}, max_chars,
               registry.options(d.feature), d.tier.value)  # fmt: skip


def _apply(d: Decision, path: str, r: Result, store: Store, scanner: Scanner,
           embedder: Embedder | None, report: IngestReport, registry: Registry | None = None,
           stale: frozenset[str] = frozenset()) -> None:  # fmt: skip
    """Write one extraction result. The text in `r` is already redacted."""
    root = str(d.root.path) if d.root else None
    rel = d.rel.as_posix() if d.rel else ""
    if r.skip:
        if r.skip.startswith("secret:"):
            first = next((f for f in r.findings if f["action"] == "skip-file"), None)
            store.audit("skip-file", path, r.skip[7:], first["line"] if first else None,
                        sum(1 for f in r.findings if f["action"] == "skip-file"))  # fmt: skip
            _skip(store, report, path, r.skip, root, rel, d.size, d.mtime_ns, scanner.hash,
                  feature=d.feature)  # fmt: skip
            log.info("skipped %s: %s", rel, r.skip[7:])
        else:
            old = store.get_file(path)
            if (old is not None and old.status == "indexed" and old.read_tier != Tier.OPEN.value
                    and old.feature not in stale  # text from a changed extractor goes
                    and (old.size, old.mtime_ns, old.rules_hash) == (d.size, d.mtime_ns,
                                                                     scanner.hash)):  # fmt: skip
                # Read again only to be shown as open, and that read failed (no Tesseract for
                # an open image, say): keep what it was, local-only.
                store.set_tier(path, old.path_tier or Tier.LOCAL_ONLY.value, Tier.LOCAL_ONLY.value)
                note = ("files that could not be read again on an open worker stay local-only "
                        f"({r.skip})")  # fmt: skip
                if note not in report.warnings:
                    report.warnings.append(note)
                return
            _skip(store, report, path, r.skip, root, rel, d.size, None, feature=d.feature)
        return
    assert d.tier is not None and d.rel is not None
    if r.findings:
        by_rule: dict[str, list[int]] = {}
        for f in r.findings:
            by_rule.setdefault(f["rule"], []).append(f["line"])
        for rule, lines in by_rule.items():
            store.audit("redact", path, rule, lines[0], len(lines))
            report.redactions[rule] += len(lines)
        report.redacted_files += 1
    chunks = chunk_segments(rel, r.segments)
    # A chunk holding a piece of a cut long line has line starts and ends the scanned text
    # did not (a `^password: x` match can appear at the cut): scan it as stored.
    late = 0
    for i, c in enumerate(chunks):
        if not c.split:
            continue
        try:
            found = scanner.scan(c.text, rel, redacted=True)
        except ScanTimeout:
            _skip(store, report, path, "scan-timeout", root, rel, d.size, None, feature=d.feature)
            return
        if found:
            chunks[i] = replace(c, text=redact(c.text, found))
            late += len(found)
            for f in found:
                store.audit("redact", path, f.rule, c.start, 1)
                report.redactions[f.rule] += 1
    if late and not r.findings:
        report.redacted_files += 1
    text = "\n".join(s["text"] for s in r.segments)
    in_path = _path_secrets(scanner, d.rel)
    n_secrets = r.n_secrets + in_path + late
    tier = content_tier(d.tier, d.rel, text, n_secrets, _content_rules(d, registry))
    if in_path:  # a path cannot be redacted: the file is only ever local-only
        tier = Tier.LOCAL_ONLY
    # Computed here, never taken from the worker: a worker's digest is 64 free characters
    # that an open server would show.
    digest = hashlib.sha256()
    for seg in r.segments:
        digest.update(seg["text"].encode())
    store.replace_file(path, root=root or "", rel=rel, size=d.size or 0,
                       mtime_ns=d.mtime_ns or 0, content_hash=digest.hexdigest(),
                       path_tier=d.tier.value, read_tier=d.tier.value,
                       tier=tier.value, n_secrets=n_secrets, rules_hash=scanner.hash,
                       chunks=chunks, feature=d.feature or "", note=r.note)  # fmt: skip
    report.indexed += 1


def _looser(new: Tier, old: FileRow) -> bool:
    """Whether showing a file as `new` needs a fresh read first: it would be open, but was
    not read on an open worker. Text read on a local-only worker was never scanned a second
    time, and that worker may have seen other local-only files.
    Rows from before schema 9 do not know; the next sync reads their open files again."""
    return new == Tier.OPEN and old.read_tier != Tier.OPEN.value


def _unchanged(old: FileRow, d: Decision, rules_changed: bool, rules_hash: str) -> bool:
    """Same size and time, and scanned by the current rule set (checked per file: a file
    kept hidden through a rule change must not count as scanned by the new rules)."""
    if rules_changed or old.rules_hash != rules_hash:
        return False
    if old.size != d.size or old.mtime_ns != d.mtime_ns:
        return False
    if old.status == "indexed":
        return True
    return old.status == "skipped" and (old.reason or "").startswith("secret:")


def _content_rules(d: Decision, registry: Registry | None = None) -> tuple[ContentTierRule, ...]:
    builtin = (SecretsMakeLocalOnly(),) if d.root and d.root.secrets_make_local_only else ()
    return builtin + (plugin_tier_rules(registry) if registry is not None else ())


@dataclass(frozen=True)
class _Unloadable:
    """Stands in for an enabled tier rule that failed to load. Rules only restrict, so
    leaving one out would loosen what the user restricted: instead every file is local-only
    until it loads again."""

    feature: str
    error: str
    api_version = 1
    name = "unloadable"
    needs_text = False

    def tier_for(self, rel_path: PurePosixPath, text: str, n_secrets: int) -> Tier:
        return Tier.LOCAL_ONLY


class _Guarded:
    """A tier-rule plugin with a deadline. It runs in lanternkb's own process, so
    a call that raises or takes longer than TIER_RULE_TIMEOUT_S makes that file local-only;
    after a timeout the rule is not called again during this sync (its thread cannot be
    stopped, only abandoned), and every later file is local-only too. Fail closed."""

    TIER_RULE_TIMEOUT_S = 2.0
    BUDGET_S = 120.0  # in all, per sync (registry): a slow rule cannot make a sync take hours
    # Rules whose call never returned, by feature id, for the life of this process: a new
    # registry (every sync, every settings change) must not start another thread on them
    #. Cleared when the abandoned call finally ends.
    _stuck: dict[str, threading.Thread] = {}

    def __init__(self, rule: Any, feature: str):
        self.rule, self.feature = rule, feature
        self.api_version = getattr(rule, "api_version", None)
        self.name = getattr(rule, "name", feature)
        self.needs_text = bool(getattr(rule, "needs_text", True))
        stuck = self._stuck.get(feature)
        if stuck is not None and not stuck.is_alive():
            del self._stuck[feature]
        self.hung = feature in self._stuck
        self.spent = 0.0
        self.over_budget = False

    def tier_for(self, rel_path: PurePosixPath, text: str, n_secrets: int) -> Any:
        if self.hung or self.over_budget:
            return Tier.LOCAL_ONLY
        if self.spent > self.BUDGET_S:
            self.over_budget = True
            log.warning("tier rule %s used its %.0f s for this sync; every later file is "
                        "local-only until the next sync", self.feature, self.BUDGET_S)  # fmt: skip
            return Tier.LOCAL_ONLY
        box: dict[str, Any] = {}

        def call() -> None:
            try:
                verdict = self.rule.tier_for(rel_path, text, n_secrets)
                # Judged here, inside the deadline: a verdict whose comparison raises or
                # hangs is the rule's failure, not the sync's.
                box["tier"] = Tier.LOCAL_ONLY if verdict and verdict != Tier.OPEN else None
            except BaseException as e:  # noqa: BLE001 - SystemExit too: reported, fail closed
                box["error"] = type(e).__name__

        t = threading.Thread(target=call, daemon=True)
        started = time.monotonic()
        t.start()
        t.join(self.TIER_RULE_TIMEOUT_S)
        self.spent += time.monotonic() - started
        if t.is_alive():
            self.hung = True
            self._stuck[self.feature] = t
            log.warning("tier rule %s took over %.0f s; every file is local-only until the "
                        "next sync", self.feature, self.TIER_RULE_TIMEOUT_S)  # fmt: skip
            return Tier.LOCAL_ONLY
        if "tier" not in box:  # raised, or the thread ended some other way
            log.warning("tier rule %s failed on a file (%s): it is local-only", self.feature,
                        box.get("error", "no answer"))  # fmt: skip
            return Tier.LOCAL_ONLY
        return box["tier"]


def plugin_tier_rules(registry: Registry) -> tuple[ContentTierRule, ...]:
    """Enabled tier-rule plugins (in-process, given only their own options), built once per
    registry object. One that fails to load, or is turned on but not installed, fails closed
    (`_Unloadable`) and is reported."""
    cached = getattr(registry, "_tier_rules", None)
    if cached is not None:
        return cached
    from lanternkb import plugins

    rules: list[ContentTierRule] = []
    for fid, m in registry.manifests.items():
        if m.kind != "tier_rule" or not registry.state(fid).enabled:
            continue
        try:
            rules.append(_Guarded(plugins.create_with_deadline(fid, registry.options(fid)),
                                  fid))  # fmt: skip
        except Exception as e:  # noqa: BLE001 - any failure, a hang included: fail closed
            log.warning("tier rule %s did not load: %s", fid, type(e).__name__)
            rules.append(_Unloadable(fid, type(e).__name__))
    # A tier rule the config turns on but no installed plugin provides (uninstalled, or
    # refused) cannot be built either: fail closed the same way.
    for fid in registry.unknown:
        if fid.startswith("tier_rule.") and registry.settings.get(fid, {}).get("enabled"):
            log.warning("tier rule %s is turned on but not installed", fid)
            rules.append(_Unloadable(fid, "not installed"))
    registry._tier_rules = tuple(rules)  # type: ignore[attr-defined]
    return registry._tier_rules  # type: ignore[attr-defined]


def text_rules_signature(registry: Registry) -> str:
    """Which tier-rule plugins (and options) are on: a text-reading rule's verdicts are only
    known for files read since this last changed."""
    on = sorted((fid, registry.options(fid)) for fid, m in registry.manifests.items()
                if m.kind == "tier_rule" and registry.state(fid).enabled)  # fmt: skip
    return json.dumps(on, sort_keys=True, default=str)


def _skip(store: Store, report: IngestReport, path: str, reason: str, root: str | None,
          rel: str | None, size: int | None, mtime_ns: int | None,
          rules_hash: str | None = None, feature: str | None = None) -> None:  # fmt: skip
    old = store.get_file(path)
    if old is None or old.status != "skipped" or old.reason != reason:
        store.record_skip(path, reason, root=root, rel=rel, size=size, mtime_ns=mtime_ns,
                          rules_hash=rules_hash, feature=feature)  # fmt: skip
        if not reason.startswith("secret:"):
            store.audit("skip-policy", path, reason)
    report.skipped[reason.split(":")[0]] += 1


def retier(config: Config, store: Store, registry: Registry | None = None) -> int:
    """Re-apply tiers to every indexed file without reading it (after a tier change).

    Uses the path policy (metadata only) and the stored secret count. Files the policy no
    longer allows are left for the next sync, which hides or removes them. A tier never
    becomes looser here: such a file stays local-only until the next sync reads it again
    (`_looser`). Returns how many files changed.
    """
    registry = registry or Registry(settings=config.features)
    policy = PathPolicy(config, registry)
    scanner: Scanner | None = None
    changed = 0
    # A text-reading rule's verdict cannot be recomputed without the text: never loosen a
    # file, and if those rules changed since the last sync read every file, hold every
    # file local-only until the next sync reads it again (fail closed).
    text_rules = any(r.needs_text for r in plugin_tier_rules(registry))
    unknown_verdicts = text_rules and store.get_meta("text_rules") != text_rules_signature(
        registry)  # fmt: skip
    for path, _rel, _root, n_secrets, _feature in store.indexed_rows():
        d = policy.check(Path(path), canonical=True)
        if not d.allowed or d.tier is None or d.rel is None:
            # Gone, unplugged, too big, newly linked, disabled... the next sync will sort it
            # out. Until then fail closed: never leave it more visible than it may be.
            old = store.get_file(path)
            if old is not None and old.tier != Tier.LOCAL_ONLY.value:
                store.set_tier(path, Tier.LOCAL_ONLY.value, Tier.LOCAL_ONLY.value)
                changed += 1
            continue
        tier = content_tier(d.tier, d.rel, "", n_secrets, _content_rules(d, registry))
        if tier != Tier.LOCAL_ONLY and n_secrets:
            opts = config.scanner
            scanner = scanner or Scanner(load_rules(disabled=opts.disabled_rules,
                                                    actions=opts.actions,
                                                    enabled=opts.enabled_rules))  # fmt: skip
            if _path_secrets(scanner, d.rel):
                tier = Tier.LOCAL_ONLY
        old = store.get_file(path)
        if text_rules and old is not None:
            tier = Tier.LOCAL_ONLY if unknown_verdicts else stricter(tier, Tier(old.tier))
        if old is not None and _looser(tier, old) and old.tier != Tier.OPEN.value:
            tier = Tier.LOCAL_ONLY  # the next sync reads it again, then shows it as open
        if old is not None and (old.path_tier, old.tier) != (d.tier.value, tier.value):
            store.set_tier(path, d.tier.value, tier.value)
            changed += 1
    return changed


def _path_secrets(scanner: Scanner, rel: PurePosixPath) -> int:
    """Secrets in a file's own path (a token used as a file or folder name). Counted with
    the file's secrets; the path cannot be redacted, so such a file is local-only. A scan
    that fails counts as a finding (fail closed)."""
    text = rel.as_posix()
    try:
        return len(scanner.scan(text, text))
    except Exception:  # noqa: BLE001 - ScanTimeout or a broken rule: fail closed
        return 1


def _now_iso() -> str:
    import datetime as dt

    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
