"""Search: keyword and vector retrieval, fused, behind the tier filter.

The filters are the security boundary and do not depend on the ranking: in SQL, and checked
again in Python on every row. Only features the index records as enabled are shown (unknown
or legacy rows fail closed), and a row's tier is the stricter of its file's and its
feature's. Vector neighbours are found first and filtered after, so the search widens until
enough visible results are found or the index is exhausted.

Ranking is configurable for the evaluation (`SearchConfig`); the default is the one chosen
on the development queries. Queries are never logged.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from lanternkb.chunking import split_identifiers
from lanternkb.fsutil import canonical_case
from lanternkb.rerank import TOP as RERANK_TOP
from lanternkb.store import FTS_TABLES, Store
from lanternkb.tiers import ServerTier, Tier, stricter, visible

log = logging.getLogger("lanternkb.search")
STOP = frozenset("a an and are as at be by for from how i in is it of on or the to what".split())

# Queries that look like an exact name: snake_case, camelCase, dotted.names, words with
# digits (v2, sha256, HM-53707) or ticket ids. Phase 1's routing idea.
EXACT = re.compile(r"_|[a-z][A-Z]|\w\.\w|\b(?=\w*\d)(?=\w*[A-Za-z])[\w-]+\b|\b[A-Z]{2,}-\d+")
# More exact shapes: --flags, KEY=VALUE settings, lower-case codes (ops-93970),
# and hyphenated codes mixing letters and digits (utf-8-sig, x86-64).
BROAD = re.compile(r"(^|\s)--?[A-Za-z][\w-]*|\b\w+=\S+|\b[a-z]{2,}-\d{2,}\b"
                   r"|\b(?=[\w-]*\d)(?=[\w-]*[A-Za-z])\w+(?:-\w+)+\b")  # fmt: skip
# Among exact-looking queries: codes (searched by keywords alone) as opposed to identifiers
# (dotted, snake_case, camelCase, leading underscores), which both lists find well.
CODE = re.compile(r"^(?:[A-Za-z]{1,8}-?\d[\w-]*|[A-Z]{2,}-\d+|--?[A-Za-z][\w-]*|\w+=\S+)$")
IDENT = re.compile(r"_|[a-z][A-Z]|\w\.\w")


@dataclass(frozen=True)
class SearchConfig:
    """How the keyword and vector lists are combined.

    Reciprocal rank fusion: score = sum over lists of weight / (rrf_k + rank). `exact` and
    `natural` are the (keyword, vector) weights for exact-looking and other queries; equal
    pairs mean no routing, and (1, 0) / (0, 1) mean pure routing. `mixed`, if set, is used
    instead of `exact` when an exact-looking name sits in a longer query (3 words or more).
    `top_bonus` adds QMD's bonus for being first (0.05) or second or third (0.02) in either
    list. `level` is
    whether lists are fused per chunk or per file (a file's best chunk in each list).
    """

    mode: str = "fused"  # "fused" | "keyword" | "vector"
    rrf_k: int = 60
    exact: tuple[float, float] = (1.0, 1.0)
    natural: tuple[float, float] = (1.0, 1.0)
    mixed: tuple[float, float] | None = None
    top_bonus: bool = False
    level: str = "file"  # "file" | "chunk"
    depth: int = 200  # candidates taken from each list
    bm25: tuple[float, float, float] = (1.0, 1.0, 1.0)  # text, path words, identifier words
    broad: bool = False  # also treat BROAD shapes as exact
    ident: tuple[float, float] | None = None  # weights for an identifier (not a code), if set

    def weights(self, query: str) -> tuple[float, float]:
        if not (EXACT.search(query) or (self.broad and BROAD.search(query))):
            return self.natural
        words = query.split()
        if self.mixed is not None and len(words) >= 3:
            return self.mixed
        if self.ident is not None:
            token = words[0] if len(words) == 1 else query
            if not CODE.match(token) and IDENT.search(token):
                return self.ident
        return self.exact


# Phase 5's choice: exact-looking queries to the keyword list,
# the rest to the vector list, a name inside a longer query to both.
PHASE5 = SearchConfig(exact=(1.0, 0.0), natural=(0.0, 1.0), mixed=(1.0, 1.0))
# The candidate chosen on development data: broader
# exact shapes, and identifiers searched by both lists. On the fresh blind set (300
# queries, used once; bench/phase9-routing-plan.md) it scored 0.829 against 0.820 for
# PHASE5 and failed fewer query types, so by the rule fixed in advance it is the default.
ROUTE9 = SearchConfig(exact=(1.0, 0.0), natural=(0.0, 1.0), mixed=(1.0, 1.0), broad=True,
                      ident=(1.0, 1.0))  # fmt: skip
DEFAULT = ROUTE9


def needs_vectors(query: str, config: SearchConfig = DEFAULT) -> bool:
    """Whether this query's routing uses the vector list: if not, no model is loaded. A query
    with no letters or digits is never searched, so it needs nothing."""
    if not _WORD.search(query[:MAX_QUERY_CHARS]):
        return False
    return config.mode == "vector" or (config.mode == "fused" and config.weights(query)[1] > 0)


@dataclass(frozen=True)
class Hit:
    path: str
    start_line: int
    end_line: int
    snippet: str
    tier: str
    location: str | None = None
    score: float = 0.0
    content_hash: str | None = None  # of the file when it was indexed
    indexed_at: str | None = None
    chunk_id: int = 0
    chunk_hash: str = ""

    @property
    def result_id(self) -> str:
        """An opaque handle for this chunk, from its content hash: it reveals nothing about
        other chunks (no sequence numbers) and stops working if the chunk's text changes.
        A short tag of the file's path follows, so the same text at the same place in two
        folders has two ids."""
        return f"r{self.chunk_hash[:16]}{path_tag(self.path)}"


def path_tag(path: str) -> str:
    """Four hex digits of a path's hash: tells apart the files of one result id."""
    return hashlib.sha256(path.encode("utf-8", "surrogateescape")).hexdigest()[:4]


@dataclass
class _Row:
    chunk: int
    path: str
    start: int
    end: int
    text: str
    tier: str
    feature: str
    location: str | None
    content_hash: str | None
    indexed_at: str | None
    status: str
    rel: str = ""
    chunk_hash: str = ""
    mtime_ns: int | None = None


@dataclass
class _Ranked:
    rows: dict[int, _Row] = field(default_factory=dict)
    keyword: list[int] = field(default_factory=list)  # chunk ids, best first
    vector: list[int] = field(default_factory=list)


MIN_FILES = 50  # candidate files gathered from each list (at least k), so big files cannot crowd
KEYWORD_CAP = 20_000  # most keyword rows read while looking for enough distinct files
MAX_QUERY_CHARS = 2_000
MAX_TERMS = 64
_WORD = re.compile(r"[^\W_]+")  # letters and digits in any script


def query_terms(query: str) -> list[str]:
    """The query's words and the words inside its identifiers, without stop words."""
    q = query[:MAX_QUERY_CHARS]
    words = _WORD.findall(q.lower()) + split_identifiers(q).split()
    terms = [w for w in dict.fromkeys(words) if w not in STOP][:MAX_TERMS]
    return terms or words[:1]  # a query of stop words only: search the first word anyway


def fts_query(query: str) -> str | None:
    """An OR of quoted terms. Terms are letters and digits only, so FTS5 syntax in a query
    is never interpreted; the table's tokenizer folds accents ("café" finds "cafe")."""
    return " OR ".join(f'"{t}"' for t in query_terms(query)) or None


def visible_tiers(server_tier: ServerTier) -> tuple[str, ...]:
    return tuple(t.value for t in Tier if visible(t, server_tier))


_COLS = ("c.id, f.path, c.start_line, c.end_line, c.text, f.tier, f.feature, c.location, "
         "f.content_hash, f.updated_at, f.status, COALESCE(f.rel, ''), "
         "COALESCE(c.hash, ''), f.mtime_ns")  # fmt: skip


@dataclass(frozen=True)
class Filters:
    """Optional limits on which files a search may return; they only remove candidates.
    `types`: file extensions without the dot (`pdf`, `md`); `after`/`before`: the file's
    modification time, in nanoseconds since the epoch (`after` inclusive, `before` not)."""

    types: tuple[str, ...] = ()
    after: int | None = None
    before: int | None = None

    def __post_init__(self) -> None:
        raw = [p.strip() for t in self.types for p in t.split(",")]
        if self.types and not any(raw):
            raise ValueError("a type is a file extension such as pdf or md")
        parts = [p.lower().lstrip(".") for p in raw if p]  # "pdf," is pdf
        if any(not p or not _TYPE.fullmatch(p) for p in parts):  # "." or "a b": an error
            raise ValueError("a type is a file extension such as pdf or md")
        object.__setattr__(self, "types", tuple(sorted(set(parts))))

    def ok(self, path: str, mtime_ns: int | None) -> bool:
        if self.types and not path.translate(_ASCII_LOWER).endswith(
                tuple(f".{t}" for t in self.types)):  # fmt: skip
            return False
        if self.after is not None and (mtime_ns is None or mtime_ns < self.after):
            return False
        return self.before is None or (mtime_ns is not None and mtime_ns < self.before)


_TYPE = re.compile(r"[a-z0-9_+-]{1,16}")
_AGO = re.compile(r"(\d{1,5})\s*([dwmy])")


def when(text: str, now: float | None = None) -> int:
    """A point in time for a filter, in nanoseconds since the epoch: `YYYY-MM-DD` (local
    midnight), an ISO date and time, or `30d`, `2w`, `6m` or `1y` before now."""
    import datetime as dt
    import time

    t = text.strip().lower()
    if m := _AGO.fullmatch(t):
        days = int(m.group(1)) * {"d": 1, "w": 7, "m": 30, "y": 365}[m.group(2)]
        if days > 200 * 365:
            raise ValueError(f"too far back: {text[:40]!r}")
        return int(((now if now is not None else time.time()) - days * 86400) * 1e9)
    try:
        moment = dt.datetime.fromisoformat(text.strip())
    except ValueError:
        raise ValueError(f"not a date: {text[:40]!r} (use YYYY-MM-DD or 30d)") from None
    if not 1900 <= moment.year <= 2200:  # nanoseconds since 1970 fit in 64 bits until 2262
        raise ValueError(f"a date between 1900 and 2200, not {text[:40]!r}")
    if moment.tzinfo is None:
        moment = moment.astimezone()  # the machine's local time
    return int(moment.timestamp() * 1e9)


def _like_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")


def _under(value: str, prefix: str) -> bool:
    """`value` is the folder or file `prefix`, or inside it. ASCII letters match in any case,
    as SQL's LIKE does, so the SQL filter and this check always agree."""
    v, p = value.translate(_ASCII_LOWER), prefix.translate(_ASCII_LOWER)
    return v == p or v.startswith(p + "/")


class _Filter:
    """The tier and feature filter (and an optional path prefix), as SQL with its
    parameters, as the vector search's partitions and tiers, and as a Python check on
    every row."""

    def __init__(self, store: Store, server_tier: ServerTier, path_prefix: str | None = None,
                 filters: Filters | None = None):  # fmt: skip
        self.server_tier = server_tier
        self.filters = filters or Filters()
        self.tiers = visible_tiers(server_tier)
        self.states = store.feature_states()
        self.features = sorted(i for i, (on, t) in self.states.items()
                               if on and visible(t, server_tier))  # fmt: skip
        self.prefix = (path_prefix or "").rstrip("/") or None
        if self.prefix and self.prefix.startswith("/"):  # spelt as the index stores paths:
            self.prefix = str(canonical_case(Path(os.path.realpath(self.prefix))))  # resolved
        only_open = "" if server_tier == ServerTier.ALL else f"AND tier = '{Tier.OPEN.value}'"
        marks = ",".join("?" * len(self.tiers))
        self.sql = (f"f.status = 'indexed' AND f.tier IN ({marks}) AND f.feature IN "
                    f"(SELECT id FROM features WHERE enabled = 1 {only_open})")  # fmt: skip
        self.params: tuple[str, ...] = self.tiers
        if self.prefix:  # the folder or file itself, or anything inside it
            like = " LIKE ? ESCAPE '\\'"
            self.sql += (f" AND (f.path{like} OR f.path{like} OR f.rel{like} OR "
                         f"f.rel{like})")  # fmt: skip
            exact, inside = _like_escape(self.prefix), _like_escape(self.prefix) + "/%"
            self.params += (exact, inside, exact, inside)
        f = self.filters
        if f.types:  # LIKE ignores ASCII case, as `Filters.ok` does
            self.sql += " AND (" + " OR ".join(["f.path LIKE ? ESCAPE '\\'"] * len(f.types)) + ")"
            self.params += tuple("%." + _like_escape(t) for t in f.types)
        if f.after is not None:
            self.sql += " AND f.mtime_ns >= ?"
            self.params += (f.after,)  # type: ignore[assignment]
        if f.before is not None:
            self.sql += " AND f.mtime_ns < ?"
            self.params += (f.before,)  # type: ignore[assignment]

    @property
    def restricts(self) -> bool:
        """Whether a path prefix or a filter narrows the search below its tier."""
        f = self.filters
        return bool(self.prefix or f.types or f.after is not None or f.before is not None)

    def ok(self, row: _Row) -> bool:
        if row.status != "indexed" or row.feature not in self.states:
            return False
        if self.prefix and not (_under(row.path, self.prefix) or _under(row.rel, self.prefix)):
            return False
        if not self.filters.ok(row.path, row.mtime_ns):
            return False
        on, feature_tier = self.states[row.feature]
        shown = visible(row.tier, self.server_tier) and visible(feature_tier, self.server_tier)
        return on and shown

    def tier_of(self, row: _Row) -> str:
        """The tier a result is shown with: the stricter of its file's and its feature's."""
        return stricter(Tier(row.tier), Tier(self.states[row.feature][1])).value


def _keyword(store: Store, query: str, flt: _Filter, cfg: SearchConfig, out: _Ranked,
             want_files: int) -> None:  # fmt: skip
    expr = fts_query(query)
    if expr is None:
        return
    # An open server ranks within the open table only, so its BM25 statistics never
    # depend on chunks it may not see; a local server ranks over both tables.
    tables = ("fts_open",) if flt.server_tier == ServerTier.OPEN else FTS_TABLES
    union = " UNION ALL ".join(f"SELECT rowid AS rid, bm25({t}, ?, ?, ?) AS score FROM {t} "
                               f"WHERE {t} MATCH ?" for t in tables)  # fmt: skip
    union_args = [a for _t in tables for a in (*cfg.bm25, expr)]
    limit = cfg.depth
    while True:  # widen until enough distinct files, so one big file cannot fill the list
        rows = [_Row(*r) for r in store.con.execute(
            f"SELECT {_COLS} FROM ({union}) m JOIN chunks c ON c.id = m.rid "
            f"JOIN files f ON f.id = c.file_id WHERE {flt.sql} ORDER BY m.score, m.rid LIMIT ?",
            (*union_args, *flt.params, limit))]  # fmt: skip
        enough = len({r.path for r in rows}) >= want_files
        if len(rows) < limit or enough or limit >= KEYWORD_CAP:
            break
        limit *= 4
    for row in rows:
        if flt.ok(row):
            out.rows[row.chunk] = row
            out.keyword.append(row.chunk)


def _vector(store: Store, embedder: Any, query: str, flt: _Filter, cfg: SearchConfig,
            out: _Ranked, want_files: int) -> bool:  # fmt: skip
    """Nearest chunks among what this server may see; False if the embedder failed."""
    import numpy as np

    try:
        from lanternkb.embed import QUERY_TIMEOUT_S, with_deadline

        vec = with_deadline(embedder, "embed_query", query[:MAX_QUERY_CHARS],
                            QUERY_TIMEOUT_S, as_array=True)  # fmt: skip
        dim = store.vector_models().get(embedder.key, (None, None))[1]
        if vec.ndim != 1 or (dim is not None and vec.shape[0] != dim) or not np.isfinite(
                vec).all():  # fmt: skip
            raise ValueError("query vector has the wrong shape or is not finite")
    except Exception as e:  # noqa: BLE001 - a busy or broken model must not stop a search
        log.warning("query embedding failed (%s); searching by keywords", type(e).__name__)
        return False
    only = None  # the filter's own chunk ids, built only when needed
    k = cfg.depth
    while True:
        try:
            near = store.knn(embedder.key, vec.tobytes(), k, flt.features, flt.tiers, only)
        except Exception as e:  # noqa: BLE001 - any vector failure: keywords only
            log.warning("vector search failed (%s); searching by keywords", type(e).__name__)
            return False
        ids = [cid for cid, _d in near]
        rows: dict[int, _Row] = {}
        for i in range(0, len(ids), 500):
            part = ids[i : i + 500]
            for r in store.con.execute(
                f"SELECT {_COLS} FROM chunks c JOIN files f ON f.id = c.file_id "
                f"WHERE c.id IN ({','.join('?' * len(part))}) AND {flt.sql}",
                (*part, *flt.params),
            ):
                row = _Row(*r)
                if flt.ok(row):
                    rows[row.chunk] = row
        ranked = [cid for cid in ids if cid in rows]
        files = {rows[c].path for c in ranked}
        if len(near) < k or len(files) >= want_files:
            break
        if flt.restricts and only is None:
            # A narrow filter or path left too few of the nearest chunks: search among its
            # own chunks instead, so it still finds its nearest. A broad one never pays for
            # the list.
            only = np.fromiter((cid for (cid,) in store.con.execute(
                f"SELECT c.id FROM chunks c JOIN files f ON f.id = c.file_id WHERE {flt.sql}",
                flt.params)), dtype=np.int64)  # fmt: skip
            if only.size == 0:
                ranked = []
                break
            k = cfg.depth
            continue
        if k >= store.MAX_K:
            break
        k = min(store.MAX_K, k * 4)  # a few files fill the nearest chunks: look further
    for cid in ranked:
        out.rows.setdefault(cid, rows[cid])
        out.vector.append(cid)
    return True


def _fuse(ranked: _Ranked, cfg: SearchConfig, weights: tuple[float, float]
          ) -> list[tuple[float, int]]:  # fmt: skip
    """(score, chunk id) best first; at file level, one entry per file."""
    scores: dict[object, float] = {}
    best: dict[object, tuple[float, int]] = {}  # unit -> (best contribution, chunk shown)
    for weight, ids in ((weights[0], ranked.keyword), (weights[1], ranked.vector)):
        if weight <= 0:
            continue
        rank_of: dict[object, int] = {}
        for cid in ids:
            unit = ranked.rows[cid].path if cfg.level == "file" else cid
            if unit not in rank_of:
                rank_of[unit] = len(rank_of) + 1
                part = weight / (cfg.rrf_k + rank_of[unit])
                if cfg.top_bonus:
                    part += 0.05 if rank_of[unit] == 1 else 0.02 if rank_of[unit] <= 3 else 0.0
                scores[unit] = scores.get(unit, 0.0) + part
                if unit not in best or part > best[unit][0]:
                    best[unit] = (part, cid)
    order = sorted(scores, key=lambda u: (-scores[u], best[u][1]))
    return [(scores[u], best[u][1]) for u in order]


_SNIP_WORD = re.compile(r"[^\W_]{3,}|[A-Za-z0-9_]{3,}")


def match_spans(text: str, query: str) -> list[tuple[int, int]]:
    """Where the query's words (two letters or more, the longest first) start a word of
    `text`, ignoring case: the parts a result shows marked."""
    words = sorted({w.lower() for w in re.findall(r"\w{2,}", query[:MAX_QUERY_CHARS])},
                   key=len, reverse=True)[:16]  # fmt: skip
    if not words:
        return []
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, words)) + ")", re.I)
    return [m.span() for m in pattern.finditer(text)]


def snippet(text: str, query: str, chars: int) -> str:
    """The part of a passage worth showing for a query: the window of `chars` characters
    with the most distinct query words (a little context before the first), cut at word
    boundaries, with "…" where text was left out. The start of the passage when no query
    word occurs in it (a match by meaning only)."""
    if len(text) <= chars:
        return text
    words = {w for w in _SNIP_WORD.findall(query.lower()) if w not in STOP}
    if not words:
        return text[:chars]
    lower = text.lower()
    hits = [(m.start(), m.group(0)) for m in re.finditer(
        "|".join(sorted((re.escape(w) for w in words), key=len, reverse=True)), lower)]  # fmt: skip
    if not hits:
        return text[:chars]
    best, best_score = 0, -1
    for start, _ in hits:
        lo = max(0, min(start - chars // 5, len(text) - chars))
        inside = {w for pos, w in hits if lo <= pos < lo + chars}
        score = len(inside) * 100 - lo // 1000  # most distinct words, then earliest
        if score > best_score:
            best, best_score = lo, score
    lo = best
    if lo > 0:  # start at a word
        space = text.find(" ", lo, lo + 30)
        lo = space + 1 if space != -1 else lo
    hi = min(len(text), lo + chars)
    if hi < len(text):
        space = text.rfind(" ", hi - 30, hi)
        hi = space if space > lo else hi
    return ("…" if lo > 0 else "") + text[lo:hi].strip() + ("…" if hi < len(text) else "")


def search(store: Store, query: str, server_tier: ServerTier, k: int = 8,
           snippet_chars: int = 300, embedder: Any = None, config: SearchConfig = DEFAULT,
           path_prefix: str | None = None, reranker: Any = None,
           visible: Any = None, filters: Filters | None = None) -> list[Hit]:  # fmt: skip
    """The k best files for a query, one hit (its best chunk) per file.

    Without an embedder, or before any vectors exist for it, this is keyword search. With a
    reranker, the top files are reordered (`lanternkb.rerank`). `visible(path)`, if given,
    drops files today's config hides (since the last sync) before anything else sees them,
    the reranker included. `filters` keep only some file types or modification
    times. The whole search reads one snapshot of the index.
    Queries are never logged.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if not _WORD.search(query[:MAX_QUERY_CHARS]):
        return []  # nothing searchable: no letters or digits
    own = not store.con.in_transaction
    if own:
        store.con.execute("BEGIN")
    try:
        return _search(store, query, server_tier, k, snippet_chars, embedder, config,
                       path_prefix, reranker, visible, filters)  # fmt: skip
    finally:
        if own:
            store.con.execute("COMMIT")


def comparable(embedder: Any, model: tuple[str, int], query: str) -> bool:
    """Whether the embedder's vector for this query may be compared with an index's vectors
    (`model`: their model id and size): the same model run the same way, or a built-in
    model's other backend for a query short enough that both give the same vector."""
    model_id, dim = model
    if dim != embedder.dim:
        return False
    if model_id == embedder.model_id:
        return True
    if model_id not in getattr(embedder, "query_ids", ()):
        return False
    try:
        return bool(embedder.fits_window(query[:MAX_QUERY_CHARS]))
    except Exception:  # noqa: BLE001 - a tokenizer failure: keywords only
        return False


def _search(store: Store, query: str, server_tier: ServerTier, k: int, snippet_chars: int,
            embedder: Any, config: SearchConfig, path_prefix: str | None,
            reranker: Any = None, visible: Any = None,
            filters: Filters | None = None) -> list[Hit]:  # fmt: skip
    flt = _Filter(store, server_tier, path_prefix, filters)
    if not flt.tiers:
        return []
    mode = config.mode
    weights = {"keyword": (1.0, 0.0), "vector": (0.0, 1.0)}.get(mode) or config.weights(query)
    model = store.vector_models().get(embedder.key) if embedder is not None else None
    use_vectors = (mode != "keyword" and store.vectors_ok and model is not None
                   and comparable(embedder, model, query))  # fmt: skip
    if mode == "vector" and not use_vectors:
        return []
    if mode == "fused":
        if not use_vectors:
            weights = (1.0, 0.0)  # keyword search
        elif not store.vectors_complete(embedder.key, open_only=server_tier == ServerTier.OPEN):
            # Embedding is not finished (cheap first): chunks without a vector can only be
            # found by keywords, so every query uses the keyword list until it is.
            weights = (max(weights[0], 1.0), weights[1])
    want = max(k, MIN_FILES)
    ranked = _Ranked()
    found = use_vectors and weights[1] > 0 and _vector(store, embedder, query, flt, config,
                                                       ranked, want)  # fmt: skip
    if use_vectors and weights[1] > 0 and not found and mode == "fused":
        weights = (max(weights[0], 1.0), 0.0)  # the embedder failed: keywords instead
    if weights[0] > 0:
        _keyword(store, query, flt, config, ranked, want)
    hits: list[Hit] = []
    rows: list[_Row] = []
    seen: set[str] = set()
    limit = max(k, RERANK_TOP) if reranker is not None else k
    for score, cid in _fuse(ranked, config, weights):
        row = ranked.rows[cid]
        if row.path in seen or not flt.ok(row):  # checked again, whatever the ranking did
            continue
        seen.add(row.path)
        if visible is not None and not visible(row.path):
            continue
        rows.append(row)
        hits.append(Hit(row.path, row.start, row.end, snippet(row.text, query, snippet_chars),
                        flt.tier_of(row), row.location, round(score, 6), row.content_hash,
                        row.indexed_at, row.chunk, row.chunk_hash))  # fmt: skip
        if len(hits) == limit:
            break
    if reranker is not None and len(hits) > 1:
        # Reordered, with the fused score, so a result's score still follows its place.
        hits = [
            h if score is None else replace(h, score=round(score, 6))
            for h, score in ((hits[i], s) for i, s in _rerank(reranker, query, rows))
        ]
    return hits[:k]


def _rerank(reranker: Any, query: str, rows: list[_Row]) -> list[tuple[int, float | None]]:
    """The new order of the top files and their fused scores (`lanternkb.rerank.fused`);
    their own order (and scores) if the reranker fails, hangs or answers in the wrong
    shape."""
    import numpy as np

    from lanternkb.embed import with_deadline
    from lanternkb.rerank import RRF_K, TIMEOUT_S, fused

    top = rows[:RERANK_TOP]
    pairs = [[query[:MAX_QUERY_CHARS], f"{r.rel}\n{r.text}"] for r in top]
    try:
        limit = getattr(reranker, "timeout_s", TIMEOUT_S)  # longer on the CPU
        scores = with_deadline(reranker, "score", pairs, limit, as_array=True)
        if scores.shape != (len(top),) or not np.isfinite(scores).all():
            raise ValueError("reranker scores have the wrong shape or are not finite")
    except Exception as e:  # noqa: BLE001 - the search's own order stands
        log.warning("reranking failed (%s); keeping the search's order", type(e).__name__)
        return [(i, None) for i in range(len(rows))]
    return fused(len(rows), [float(x) for x in scores], RRF_K)


RECENT = ("", "*")  # a query that asks for the files changed most recently
RECENT_EXAMINE = 20_000  # files a listing looks at, at most


def recent(store: Store, server_tier: ServerTier, k: int = 8, snippet_chars: int = 300,
           path_prefix: str | None = None, visible: Any = None,
           filters: Filters | None = None,
           capped: list[int] | None = None) -> list[Hit]:  # fmt: skip
    """The files changed most recently, newest first, among those this tier may see and the
    filters allow: what someone worked on lately. One hit per file, its first passage as
    the snippet; the score is the modification time, in seconds since the epoch. `visible`
    drops files today's config hides, as in `search`. If RECENT_EXAMINE files were looked
    at before k were found, the count is appended to `capped`."""
    if k < 1:
        raise ValueError("k must be at least 1")
    own = not store.con.in_transaction
    if own:
        store.con.execute("BEGIN")
    try:
        flt = _Filter(store, server_tier, path_prefix, filters)
        hits: list[Hit] = []
        # newest first, a page at a time by key (no OFFSET re-scans), pages doubling, and
        # at most RECENT_EXAMINE files looked at: files hidden since the sync are skipped
        # cheaply, and a listing never stalls
        batch, examined, after = max(4 * k, 32), 0, None
        while len(hits) < k:
            if examined >= RECENT_EXAMINE:
                if capped is not None:
                    capped.append(examined)  # the caller can say why the list is short
                break
            where = flt.sql + " AND f.mtime_ns IS NOT NULL"
            args: tuple[Any, ...] = flt.params
            if after is not None:  # by (time, row id): the index holds both, so no sort
                where += " AND (f.mtime_ns, f.id) < (?, ?)"
                args += after
            page = store.con.execute(
                f"SELECT f.id, f.mtime_ns FROM files f WHERE {where} "
                "ORDER BY f.mtime_ns DESC, f.id DESC LIMIT ?",
                (*args, batch)).fetchall()  # fmt: skip
            if not page:
                break
            after = (page[-1][1], page[-1][0])
            examined += len(page)
            batch = min(2 * batch, 4096)
            for file_id, _ in page:
                found = store.con.execute(
                    f"SELECT {_COLS} FROM chunks c JOIN files f ON f.id = c.file_id "
                    "WHERE f.id = ? ORDER BY c.ord LIMIT 1", (file_id,)).fetchone()  # fmt: skip
                if found is None:
                    continue  # nothing indexed from it (an empty file)
                r = _Row(*found)
                if not flt.ok(r) or (visible is not None and not visible(r.path)):
                    continue
                hits.append(Hit(r.path, r.start, r.end, snippet(r.text, "", snippet_chars),
                                flt.tier_of(r), r.location, round(r.mtime_ns / 1e9, 3),
                                r.content_hash, r.indexed_at, r.chunk, r.chunk_hash))  # fmt: skip
                if len(hits) == k:
                    break
        return hits
    finally:
        if own:
            store.con.execute("COMMIT")


def visible_chunks(store: Store, server_tier: ServerTier, where: str,
                   args: tuple[Any, ...]) -> list[Hit]:  # fmt: skip
    """Chunks matching an SQL condition on `c` (chunks) and `f` (files), restricted to what
    this server tier may see, in file order, read in one snapshot. Full text, no limit:
    callers cap what they return."""
    own = not store.con.in_transaction
    if own:
        store.con.execute("BEGIN")
    try:
        flt = _Filter(store, server_tier)
        rows = [_Row(*r) for r in store.con.execute(
            f"SELECT {_COLS} FROM chunks c JOIN files f ON f.id = c.file_id "
            f"WHERE ({where}) AND {flt.sql} ORDER BY f.path, c.ord",
            (*args, *flt.params))]  # fmt: skip
        return [Hit(r.path, r.start, r.end, r.text, flt.tier_of(r), r.location, 0.0,
                    r.content_hash, r.indexed_at, r.chunk, r.chunk_hash)
                for r in rows if flt.ok(r)]  # fmt: skip
    finally:
        if own:
            store.con.execute("COMMIT")
