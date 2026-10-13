"""The index: one SQLite file with files, chunks, a keyword index and an audit table.

Safety properties:
- the file (and its WAL and SHM) is created with mode 600, in a folder with mode 700;
- `PRAGMA secure_delete` zeroes freed pages, and the full-text table uses FTS5
  `secure-delete`, so text removed by a re-scan does not linger in the file;
- the audit table has no column for a matched value.
Only the sync job writes; readers open it read-only.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import os
import re
import sqlite3
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lanternkb.chunking import Chunk, split_identifiers
from lanternkb.embed import embed_input
from lanternkb.fsutil import private_dirs

SCHEMA_VERSION = 9
VECTOR_FORMAT = 8  # the schema version whose vector tables this code reads
# What a reader says while an older index waits for a sync to move its vectors.
OLD_INDEX = "the index predates version 8; the next sync upgrades it (keyword search until then)"
# The keyword index is split by who may see a chunk, so an open server's ranking statistics
# (BM25 document frequencies) come only from what it may see: `fts_open` holds chunks of
# open files in enabled open features, `fts_rest` everything else. Rows move between them
# when a file's tier or a feature's state changes.
FTS_TABLES = ("fts_open", "fts_rest")
OPEN_VISIBLE = ("f.status = 'indexed' AND f.tier = 'open' AND f.feature IN "
                "(SELECT id FROM features WHERE enabled = 1 AND tier = 'open')")  # fmt: skip
_FTS_COLUMNS = ("text, path_words, idents, content='', contentless_delete=1, "
                "tokenize='porter unicode61'")  # fmt: skip
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS files(
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,      -- absolute, as found
    root TEXT,
    rel TEXT,
    size INTEGER,
    mtime_ns INTEGER,
    content_hash TEXT,
    path_tier TEXT,                 -- tier from root and path rules
    tier TEXT,                      -- final tier after content rules
    n_secrets INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,           -- indexed | skipped
    reason TEXT,                    -- why skipped
    rules_hash TEXT,
    feature TEXT,                   -- the extractor feature that produced it (its namespace)
    note TEXT,                      -- extraction quality note, e.g. "4 of 10 pages had no text"
    read_tier TEXT,                 -- the tier of the worker that read it (schema 9)
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks(
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    ord INTEGER NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    text TEXT NOT NULL,
    location TEXT,                  -- "lines 3-40", "page 3", "slide 2", "Sheet1!A1:F40"
    hash TEXT                       -- sha256 of the embedding input: embedded once
);
CREATE INDEX IF NOT EXISTS chunks_by_hash ON chunks(hash);
CREATE INDEX IF NOT EXISTS chunks_by_file ON chunks(file_id);
CREATE INDEX IF NOT EXISTS files_by_mtime ON files(mtime_ns);  -- recent files
CREATE VIRTUAL TABLE IF NOT EXISTS fts_open USING fts5(
    text, path_words, idents, content='', contentless_delete=1, tokenize='porter unicode61');
CREATE VIRTUAL TABLE IF NOT EXISTS fts_rest USING fts5(
    text, path_words, idents, content='', contentless_delete=1, tokenize='porter unicode61');
CREATE TABLE IF NOT EXISTS vector_models(
    key TEXT PRIMARY KEY,           -- embedder feature key, e.g. "qwen3-0.6b"
    model_id TEXT NOT NULL,         -- model and input format: a change means a full re-embed
    dim INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embed_failures(
    hash TEXT NOT NULL,             -- an embedding input the model rejected (no text kept)
    model_id TEXT NOT NULL,
    PRIMARY KEY(hash, model_id)
);
CREATE TABLE IF NOT EXISTS features(
    id TEXT PRIMARY KEY,
    enabled INTEGER NOT NULL,       -- 0 hides the feature's rows from every reader at once
    tier TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit(
    id INTEGER PRIMARY KEY,
    time TEXT NOT NULL,
    kind TEXT NOT NULL,             -- redact | skip-file | skip-policy | removed
    path TEXT,
    rule TEXT,
    line INTEGER,
    count INTEGER
);
"""


def _load_vec(con: sqlite3.Connection) -> bool:
    """Load sqlite-vec, needed only to move an index's vectors from before version 7."""
    try:
        import sqlite_vec

        con.enable_load_extension(True)
        sqlite_vec.load(con)
        con.enable_load_extension(False)
        return True
    except (ImportError, AttributeError, sqlite3.Error):
        return False


@dataclass(frozen=True)
class _VecCache:
    version: str
    ids: Any  # numpy int64[n]: chunk ids
    feature: Any  # numpy int16[n]: index into features
    features: list[str]
    tier: Any  # numpy int16[n]: index into tiers
    tiers: list[str]
    bits: Any  # numpy uint64[n, dim/64] (or uint8[n, dim/8]): the sign bits


def sign_bits(vector: bytes) -> bytes:
    """One bit per dimension of a float32 vector, set where the value is positive, first
    dimension in the highest bit (numpy's packbits order)."""
    try:
        import numpy as np

        return bytes(np.packbits(np.frombuffer(vector, dtype=np.float32) > 0).tobytes())
    except ImportError:  # migrating an index without numpy installed
        from array import array

        values = array("f")
        values.frombytes(vector)
        out = bytearray((len(values) + 7) // 8)
        for i, x in enumerate(values):
            if x > 0:
                out[i >> 3] |= 0x80 >> (i & 7)
        return bytes(out)


def _stored_forms(vector: bytes) -> tuple[bytes, bytes]:
    """A float32 vector as stored: normalised to unit length (cosine distance is unchanged,
    and every value then fits float16), as float16 bytes, with the sign bits of those
    float16 values (so a reused vector gets the same bits as the original)."""
    try:
        import numpy as np

        a = np.frombuffer(vector, dtype=np.float32).astype(np.float64)
        norm = float(np.linalg.norm(a))
        if norm > 0 and np.isfinite(norm):
            a = a / norm
        half = a.astype(np.float16)
        return half.tobytes(), bytes(np.packbits(half > 0).tobytes())
    except ImportError:  # migrating an index without numpy installed
        import math
        import struct

        values = struct.unpack(f"<{len(vector) // 4}f", vector)
        norm = math.sqrt(sum(x * x for x in values))
        if norm > 0 and math.isfinite(norm):
            values = tuple(x / norm for x in values)
        half = struct.pack(f"<{len(values)}e", *values)
        return half, sign_bits(to_f32(half))


def to_f16(vector: bytes) -> bytes:
    """float32 bytes to float16 bytes (how vectors are stored)."""
    import struct

    n = len(vector) // 4
    return struct.pack(f"<{n}e", *struct.unpack(f"<{n}f", vector))


def to_f32(stored: bytes) -> bytes:
    """Stored float16 bytes back to float32 bytes (what embedders and callers use)."""
    import struct

    n = len(stored) // 2
    return struct.pack(f"<{n}f", *struct.unpack(f"<{n}e", stored))


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def path_words(rel: str) -> str:
    return re.sub(r"[/._-]+", " ", rel)


@dataclass(frozen=True)
class FileRow:
    id: int
    path: str
    size: int | None
    mtime_ns: int | None
    path_tier: str | None
    tier: str | None
    n_secrets: int
    status: str
    reason: str | None
    rules_hash: str | None
    feature: str | None
    # The tier of the worker that read it: only text read on an open worker, and scanned
    # a second time, may be shown as open. None: read before schema 9.
    read_tier: str | None = None


def _create_private(path: Path) -> None:
    """Create an empty file readable only by its owner, in a folder only the owner can enter,
    and keep it out of Time Machine backups (a purge must not live on in a backup)."""
    new_folder = not path.parent.exists()
    private_dirs(path.parent)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    # The folder only if lanternkb made it (its -wal and -shm come and go inside it); never a
    # folder of the user's, where only the index file itself is excluded.
    exclude_from_backups(path.parent if new_folder else path)


def exclude_from_backups(path: Path) -> None:
    """macOS: the sticky Time Machine exclusion `tmutil addexclusion` makes (an attribute on
    the item, no admin rights), written directly: tmutil takes a second. Nothing is sent
    anywhere. Failure is silent: `lanternkb doctor` reports the state."""
    if sys.platform != "darwin":
        return
    import ctypes
    import plistlib

    value = plistlib.dumps("com.apple.backupd", fmt=plistlib.FMT_BINARY)
    with contextlib.suppress(OSError, AttributeError):
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        libc.setxattr(str(path).encode(), BACKUP_EXCLUDE, value, len(value), 0, 0)


def excluded_from_backups(path: Path) -> bool | None:
    """Whether the item carries that exclusion (None where it means nothing: not macOS)."""
    if sys.platform != "darwin":
        return None
    import ctypes

    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    n = libc.getxattr(str(path).encode(), BACKUP_EXCLUDE, None, 0, 0, 0)
    return n > 0


BACKUP_EXCLUDE = b"com.apple.metadata:com_apple_backup_excludeItem"


class Store:
    BUSY_MS = 5000  # how long to wait for another writer before giving up

    def __init__(self, path: Path, readonly: bool = False):
        """Open the index. readonly=True is for readers (search, the MCP server): queries
        only, autocommit, so a reader never holds a snapshot that blocks a writer's
        checkpoint or keeps serving data that was hidden, re-tiered or purged."""
        self.path = path
        self.readonly = readonly
        self.deleted_since_optimise = False
        if readonly:
            if not path.exists():
                raise FileNotFoundError(path)
            self.con = sqlite3.connect(path, isolation_level=None)
            self.con.execute(f"PRAGMA busy_timeout={self.BUSY_MS}")
            self.con.execute("PRAGMA query_only=ON")
            # Vectors live in plain tables since version 7, in this format since 8 (decision
            # 0020); an older index is searched by keywords until a sync upgrades it.
            self._vectors_ok = self._version() >= VECTOR_FORMAT
            self._vec_cache: dict[str, _VecCache] = {}
            return
        if not path.exists():
            _create_private(path)
        elif path.stat().st_mode & 0o077:
            os.chmod(path, 0o600)
        self.con = sqlite3.connect(path, isolation_level=None)  # explicit transactions
        self._vectors_ok = True  # plain tables: no extension needed
        self._vec_cache = {}
        self.con.execute(f"PRAGMA busy_timeout={self.BUSY_MS}")
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA secure_delete=ON")
        self.con.execute("PRAGMA foreign_keys=ON")
        new = (
            self.con.execute("SELECT name FROM sqlite_master WHERE name='files'").fetchone() is None
        )
        self._migrate()
        self.con.executescript(SCHEMA)
        self._drop_orphan_vec_tables()
        self.deleted_since_optimise = False
        self.con.execute("BEGIN")
        if new:
            for t in FTS_TABLES:
                self.con.execute(f"INSERT INTO {t}({t}, rank) VALUES('secure-delete', 1)")
            self.set_meta("schema_version", str(SCHEMA_VERSION))
        self.commit()
        for suffix in ("-wal", "-shm"):
            side = Path(str(path) + suffix)
            if side.exists() and side.stat().st_mode & 0o077:
                os.chmod(side, 0o600)

    def _migrate(self) -> None:
        """Bring an older index up to SCHEMA_VERSION (run before the CREATE statements).

        Version 1 rows do not know their feature, so disable and purge could not reach them.
        They are dropped (securely) and the next sync rebuilds them.
        """
        cols = [r[1] for r in self.con.execute("PRAGMA table_info(files)")]
        if cols and "feature" not in cols:  # version 1 -> 2
            self.con.execute("BEGIN")
            self.con.execute("ALTER TABLE files ADD COLUMN feature TEXT")
            tables = {r[0] for r in self.con.execute("SELECT name FROM sqlite_master")}
            if "chunks_fts" in tables:
                self.con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all')")
            if "chunks" in tables:
                self.con.execute("DELETE FROM chunks")
            self.con.execute("DELETE FROM files")
            self.con.execute("DELETE FROM meta WHERE key='rules_hash'")
            self.con.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', '2')")
            self.con.execute("COMMIT")
        cols = [r[1] for r in self.con.execute("PRAGMA table_info(files)")]
        if cols and "note" not in cols:  # version 2 -> 3: additive
            self.con.execute("BEGIN")
            self.con.execute("ALTER TABLE files ADD COLUMN note TEXT")
            if [r for r in self.con.execute("PRAGMA table_info(chunks)")]:
                self.con.execute("ALTER TABLE chunks ADD COLUMN location TEXT")
            self.con.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', '3')")
            self.con.execute("COMMIT")
        cols = [r[1] for r in self.con.execute("PRAGMA table_info(chunks)")]
        if cols and "hash" not in cols:  # version 3 -> 4: chunk hashes, identifier column
            self.con.execute("BEGIN")
            self.con.execute("ALTER TABLE chunks ADD COLUMN hash TEXT")
            sql = "SELECT c.id, c.text, f.rel FROM chunks c JOIN files f ON f.id = c.file_id"
            rows = list(self.con.execute(sql))
            for cid, text, rel in rows:
                h = _hash(embed_input(rel or "", text))
                self.con.execute("UPDATE chunks SET hash=? WHERE id=?", (h, cid))
            self.con.execute("DROP TABLE IF EXISTS chunks_fts")
            self.con.execute(
                "CREATE VIRTUAL TABLE chunks_fts USING fts5(text, path_words, idents, "
                "content='', contentless_delete=1, tokenize='porter unicode61')"
            )
            self.con.execute("INSERT INTO chunks_fts(chunks_fts, rank) VALUES('secure-delete', 1)")
            for cid, text, rel in rows:
                self.con.execute(
                    "INSERT INTO chunks_fts(rowid, text, path_words, idents) VALUES (?,?,?,?)",
                    (cid, text, path_words(rel or ""), split_identifiers(text)),
                )
            self.con.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', '4')")
            self.con.execute("COMMIT")
        tables = {r[0] for r in self.con.execute("SELECT name FROM sqlite_master")}
        sql = "SELECT value FROM meta WHERE key='schema_version'"
        row = self.con.execute(sql).fetchone() if "meta" in tables else None
        if row is not None and int(row[0]) < 5:  # version 4 -> 5: vectors move in 6 -> 7
            self.con.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', '5')")
            row = (5,)
        if row is not None and int(row[0]) < 6:  # version 5 -> 6: keyword index by audience
            self.con.execute("BEGIN")
            self._migrate_fts_v6()
            self.con.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', '6')")
            self.con.execute("COMMIT")
            row = (6,)
        if row is not None and int(row[0]) < 7:  # version 6 -> 7: vectors in plain tables
            self.con.execute("BEGIN")
            self._migrate_vectors_v7()
            self.con.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', '7')")
            self.con.execute("COMMIT")
            row = (7,)
        if row is not None and int(row[0]) < 8:  # version 7 -> 8: one table name per key,
            self.con.execute("BEGIN")  # float16 throughout
            self._migrate_vectors_v8()
            self.con.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', '8')")
            self.con.execute("COMMIT")
            row = (8,)
        if row is not None and int(row[0]) < 9:  # version 8 -> 9: the tier each file was
            self.con.execute("BEGIN")  # read under (unknown for rows already there)
            if "read_tier" not in {r[1] for r in self.con.execute("PRAGMA table_info(files)")}:
                self.con.execute("ALTER TABLE files ADD COLUMN read_tier TEXT")
            self.con.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', '9')")
            self.con.execute("COMMIT")

    def _migrate_fts_v6(self) -> None:
        """Rebuild the keyword index as two tables from the stored (redacted) chunk text;
        the old table is dropped, and secure_delete zeroes its pages."""
        for t in FTS_TABLES:
            self.con.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS {t} USING fts5({_FTS_COLUMNS})")
            self.con.execute(f"INSERT INTO {t}({t}, rank) VALUES('secure-delete', 1)")
        names = {r[0] for r in self.con.execute("SELECT name FROM sqlite_master")}
        if "chunks" in names and "features" in names:
            sql = f"SELECT c.id FROM chunks c JOIN files f ON f.id = c.file_id WHERE {OPEN_VISIBLE}"
            open_ids = {r[0] for r in self.con.execute(sql)}
            rows = self.con.execute(
                "SELECT c.id, c.text, COALESCE(f.rel, '') FROM chunks c "
                "JOIN files f ON f.id = c.file_id").fetchall()  # fmt: skip
            for cid, text, rel in rows:
                self._fts_insert("fts_open" if cid in open_ids else "fts_rest", cid, text, rel)
        self.con.execute("DROP TABLE IF EXISTS chunks_fts")

    def _migrate_vectors_v7(self) -> None:
        """Copy each embedder's vectors out of sqlite-vec's tables (`vec_<key>` before
        version 5, `vec5_<key>` after) into plain tables, then zero and drop the old ones.
        Without sqlite-vec the old tables cannot be read: their stored blobs are zeroed, and
        the chunks are embedded again by the next sync."""
        if not self.con.execute("SELECT name FROM sqlite_master WHERE name='vector_models'"
                                ).fetchone():  # fmt: skip
            return
        tables = {r[0] for r in self.con.execute("SELECT name FROM sqlite_master")}
        readable = _load_vec(self.con)
        models = list(self.con.execute("SELECT key, model_id, dim FROM vector_models"))
        for key, _model, dim in models:
            name = re.sub(r"[^a-z0-9]", "_", key.lower())
            old = next((t for t in (f"vec5_{name}", f"vec_{name}") if t in tables), None)
            if old is None or not readable:
                if old is not None:
                    self._abandon_table(old)
                self.con.execute("DELETE FROM vector_models WHERE key=?", (key,))
                self._set_complete(key, False)
                continue
            self._create_vec(key)
            rows = list(self.con.execute(f"SELECT rowid, embedding FROM {old}"))
            self.put_vectors(key, rows)
            zeros = bytes(4 * int(dim))
            self.con.executemany(f"UPDATE {old} SET embedding = ? WHERE rowid = ?",
                                 [(zeros, cid) for cid, _v in rows])  # fmt: skip
            self.con.execute(f"DROP TABLE {old}")
        self.deleted_since_optimise = True

    def _migrate_vectors_v8(self) -> None:
        """Version 7 named tables `vec7_<key with non-alphanumerics as _>`, so two keys could
        share one, and some were written with float32 rows. Move each key's rows
        into its own table, re-stored as normalised float16; keys that shared a table cannot
        be told apart, so their vectors are erased and embedded again."""
        if not self.con.execute("SELECT name FROM sqlite_master WHERE name='vector_models'"
                                ).fetchone():  # fmt: skip
            return
        tables = {r[0] for r in self.con.execute("SELECT name FROM sqlite_master")}
        models = list(self.con.execute("SELECT key, model_id, dim FROM vector_models"))
        old_name = {key: "vec7_" + re.sub(r"[^a-z0-9]", "_", key.lower()) for key, _m, _d in models}
        shared = {n for n in old_name.values() if list(old_name.values()).count(n) > 1}
        done: set[str] = set()
        for key, _model, dim in models:
            old = old_name[key]
            if old not in tables:
                continue
            if old in shared:
                self.con.execute("DELETE FROM vector_models WHERE key=?", (key,))
                self._set_complete(key, False)
            else:
                self._create_vec(key)
                rows = [(cid, v if len(v) == 4 * int(dim) else to_f32(v)) for cid, v in
                        self.con.execute(f"SELECT id, embedding FROM {old}")]  # fmt: skip
                self.put_vectors(key, rows)
            if old not in done:
                ids = [r[0] for r in self.con.execute(f"SELECT id FROM {old}")]
                self._erase(old, int(dim) * 2, ids)  # zeroes either row width
                self.con.execute(f"DROP TABLE {old}")
                done.add(old)
        self.deleted_since_optimise = True

    @property
    def vectors_ok(self) -> bool:
        """Whether vectors can be searched. A reader opened on an older index looks again,
        so it gains them once a sync upgrades the index in place."""
        if not self._vectors_ok and self.readonly:
            self._vectors_ok = self._version() >= VECTOR_FORMAT
        return self._vectors_ok

    def _version(self) -> int:
        try:
            row = self.con.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        except sqlite3.OperationalError:
            return 0
        return int(row[0]) if row else 0

    # -- meta -------------------------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        row = self.con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.con.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))

    # -- files ------------------------------------------------------------------------------

    def get_file(self, path: str) -> FileRow | None:
        # A reader may be open on an index a sync has not upgraded yet.
        read_tier = "read_tier" if self._has_read_tier() else "NULL"
        row = self.con.execute(
            "SELECT id, path, size, mtime_ns, path_tier, tier, n_secrets, status, reason, "
            f"rules_hash, feature, {read_tier} FROM files WHERE path=?",
            (path,),
        ).fetchone()
        return FileRow(*row) if row else None

    def _has_read_tier(self) -> bool:
        if not getattr(self, "_read_tier_col", False):
            cols = {r[1] for r in self.con.execute("PRAGMA table_info(files)")}
            self._read_tier_col = "read_tier" in cols
        return self._read_tier_col

    def _drop_chunks(self, file_id: int) -> None:  # also drops the chunks' vectors
        ids = [r[0] for r in self.con.execute("SELECT id FROM chunks WHERE file_id=?", (file_id,))]
        if ids:
            self.deleted_since_optimise = True
            marks = ",".join("?" * len(ids))
            self._drop_vectors_for(ids)
            for t in FTS_TABLES:  # a row is in one of them; deleting a missing one is a no-op
                self.con.execute(f"DELETE FROM {t} WHERE rowid IN ({marks})", ids)
            self.con.execute(f"DELETE FROM chunks WHERE id IN ({marks})", ids)

    def _upsert_file(self, path: str, **cols: object) -> int:
        cols = {**cols, "updated_at": _now()}
        names = ", ".join(cols)
        marks = ", ".join("?" * len(cols))
        updates = ", ".join(f"{k}=excluded.{k}" for k in cols)
        self.con.execute(
            f"INSERT INTO files(path, {names}) VALUES (?, {marks}) "
            f"ON CONFLICT(path) DO UPDATE SET {updates}",
            (path, *cols.values()),
        )
        return self.con.execute("SELECT id FROM files WHERE path=?", (path,)).fetchone()[0]

    def record_skip(
        self,
        path: str,
        reason: str,
        *,
        root: str | None = None,
        rel: str | None = None,
        size: int | None = None,
        mtime_ns: int | None = None,
        rules_hash: str | None = None,
        feature: str | None = None,
    ) -> None:
        """Mark a path as skipped (with its reason) and delete anything indexed for it."""
        old = self.get_file(path)
        if old is not None:
            self._drop_chunks(old.id)
        self._upsert_file(path, root=root, rel=rel, size=size, mtime_ns=mtime_ns,
                          content_hash=None, path_tier=None, tier=None, n_secrets=0,
                          status="skipped", reason=reason, rules_hash=rules_hash,
                          feature=feature, note=None, read_tier=None)  # fmt: skip

    def replace_file(self, path: str, *, root: str, rel: str, size: int, mtime_ns: int,
                     content_hash: str, path_tier: str, tier: str, n_secrets: int,
                     rules_hash: str, chunks: list[Chunk], feature: str,
                     note: str | None = None, read_tier: str | None = None) -> int:  # fmt: skip
        """Replace a file's chunks. The caller has already redacted the text.

        Vectors of chunks whose text did not change are carried over, so an edit re-embeds
        only what changed.
        """
        old = self.get_file(path)
        keep: dict[str, dict[str, bytes]] = {}
        models = self.vector_models() if self.vectors_ok else {}
        if old is not None:
            keep = self._vectors_by_hash(old.id) if models else {}
            self._drop_chunks(old.id)
        fid = self._upsert_file(path, root=root, rel=rel, size=size, mtime_ns=mtime_ns,
                                content_hash=content_hash, path_tier=path_tier, tier=tier,
                                n_secrets=n_secrets, status="indexed", reason=None,
                                rules_hash=rules_hash, feature=feature,
                                note=note, read_tier=read_tier)  # fmt: skip
        is_open = self._is_open_visible(fid)
        table = "fts_open" if is_open else "fts_rest"
        for i, c in enumerate(chunks):
            h = _hash(embed_input(rel, c.text))
            cur = self.con.execute(
                "INSERT INTO chunks(file_id, ord, start_line, end_line, text, location, hash) "
                "VALUES (?,?,?,?,?,?,?)",
                (fid, i, c.start, c.end, c.text, c.location, h),
            )
            self._fts_insert(table, int(cur.lastrowid or 0), c.text, rel)
            for key in models:
                if h in keep.get(key, {}):
                    self.put_vectors(key, [(int(cur.lastrowid or 0), keep[key][h])])
                else:
                    self._set_complete(key, False, open_too=is_open)
        return fid

    # -- keyword index ------------------------------------------------------------------------

    def _fts_insert(self, table: str, cid: int, text: str, rel: str) -> None:
        self.con.execute(f"INSERT INTO {table}(rowid, text, path_words, idents) VALUES (?,?,?,?)",
                         (cid, text, path_words(rel), split_identifiers(text)))  # fmt: skip

    def _is_open_visible(self, file_id: int) -> bool:
        sql = f"SELECT 1 FROM files f WHERE f.id = ? AND {OPEN_VISIBLE}"
        return self.con.execute(sql, (file_id,)).fetchone() is not None

    def _fts_move(self, where: str, args: tuple[object, ...], to_open: bool) -> int:
        """Move the chunks matching `where` (on c and f) to the open or the rest table."""
        rows = self.con.execute(
            f"SELECT c.id, c.text, COALESCE(f.rel, '') FROM chunks c JOIN files f "
            f"ON f.id = c.file_id WHERE {where}", args).fetchall()  # fmt: skip
        src, dst = ("fts_rest", "fts_open") if to_open else ("fts_open", "fts_rest")
        for i in range(0, len(rows), 500):
            part = rows[i : i + 500]
            marks = ",".join("?" * len(part))
            self.con.execute(f"DELETE FROM {src} WHERE rowid IN ({marks})", [r[0] for r in part])
            for cid, text, rel in part:
                self._fts_insert(dst, cid, text, rel)
        if rows and not to_open:
            self.deleted_since_optimise = True
        return len(rows)

    # -- vectors ------------------------------------------------------------------------------
    # One plain table per embedder: rowid = chunk id, the file's feature and
    # tier, the float vector, and a one-bit-per-dimension copy (the signs) that search scans
    # in memory. Vectors are as sensitive as the text they came from: every deletion zeroes
    # them first, and the database runs with secure_delete.

    # Candidates re-scored exactly: 2% of the visible vectors, at least 4,000 (so search is
    # exact up to 4,000 visible) and at most 20,000 (about 100 ms of re-scoring). Measured
    # recall@10 on real, topic-clustered embeddings: 0.90 at 1%, 0.98 at 4%.
    SHORTLIST_MIN, SHORTLIST_SHARE, SHORTLIST_MAX = 4000, 0.02, 20_000

    @staticmethod
    def vec_table(key: str) -> str:
        """One table per embedder key; the hash keeps names apart when two keys differ only
        in characters the readable part drops (`qwen3-0.6b`, `qwen3-0-6b`)."""
        digest = hashlib.sha256(key.encode()).hexdigest()[:8]
        return f"vec8_{re.sub(r'[^a-z0-9]', '_', key.lower())}_{digest}"

    def vector_models(self) -> dict[str, tuple[str, int]]:
        """Embedder key -> (model id, dimension) for every vector table in the index."""
        try:
            rows = self.con.execute("SELECT key, model_id, dim FROM vector_models ORDER BY key")
            return {k: (m, int(d)) for k, m, d in rows}
        except sqlite3.OperationalError:  # an index older than version 4, opened read-only
            return {}

    def _create_vec(self, key: str) -> None:
        self.con.execute(f"CREATE TABLE IF NOT EXISTS {self.vec_table(key)}(id INTEGER PRIMARY "
                         "KEY, feature TEXT NOT NULL, tier TEXT NOT NULL, bits BLOB NOT NULL, "
                         "embedding BLOB NOT NULL)")  # fmt: skip

    def _drop_orphan_vec_tables(self) -> None:
        """Drop sqlite-vec tables left from before version 7 (their blobs were zeroed when
        they were abandoned); possible only while sqlite-vec loads."""
        rows = self.con.execute("SELECT name FROM sqlite_master WHERE type='table' "
                                "AND sql LIKE 'CREATE VIRTUAL TABLE%USING vec0%'")  # fmt: skip
        orphans = [r[0] for r in rows]
        if orphans and _load_vec(self.con):
            self.con.execute("BEGIN")
            for name in orphans:
                self.con.execute(f'DROP TABLE "{name}"')
            self.con.execute("COMMIT")

    def _bump_vectors(self, key: str) -> None:
        """Every change to an embedder's vectors moves this counter; readers reload their
        in-memory copy when it moves."""
        name = f"vectors_version:{key}"
        self.set_meta(name, str(int(self.get_meta(name) or 0) + 1))

    def ensure_vectors(self, key: str, model_id: str, dim: int) -> None:
        """Create the vector table for an embedder; a different model means starting over."""
        current = self.vector_models().get(key)
        if current == (model_id, dim):
            return
        if current is not None:
            self.drop_vectors(key)
        self._create_vec(key)
        self.con.execute("INSERT INTO vector_models VALUES (?,?,?,?)", (key, model_id, dim, _now()))
        self._set_complete(key, False)
        self._bump_vectors(key)

    def drop_vectors(self, key: str) -> None:
        """Delete all of an embedder's vectors: zeroed, deleted, then the table dropped."""
        model_id, dim = self.vector_models()[key]
        table = self.vec_table(key)
        ids = [r[0] for r in self.con.execute(f"SELECT rowid FROM {table}")]
        self._erase(table, dim, ids)
        self.con.execute(f"DROP TABLE {table}")
        self.con.execute("DELETE FROM vector_models WHERE key=?", (key,))
        self.con.execute("DELETE FROM embed_failures WHERE model_id=?", (model_id,))
        self._set_complete(key, False)
        self._bump_vectors(key)
        self.deleted_since_optimise = True

    def _erase(self, table: str, dim: int, ids: list[int]) -> None:
        zeros, zbits = bytes(2 * dim), bytes((dim + 7) // 8)
        for i in range(0, len(ids), 500):
            part = ids[i : i + 500]
            self.con.executemany(f"UPDATE {table} SET embedding = ?, bits = ? WHERE rowid = ?",
                                 [(zeros, zbits, cid) for cid in part])  # fmt: skip
            marks = ",".join("?" * len(part))
            self.con.execute(f"DELETE FROM {table} WHERE rowid IN ({marks})", part)

    def _abandon_table(self, table: str) -> None:
        """An sqlite-vec table that cannot be read (the extension does not load): its storage
        tables are ordinary tables, so zero every stored blob in them. The empty shell is
        dropped the next time sqlite-vec loads."""
        shadow = [r[0] for r in self.con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ? ESCAPE '\\'",
            (table.replace("_", "\\_") + "\\_%",))]  # fmt: skip
        for name in shadow:
            cols = [r[1] for r in self.con.execute(f'PRAGMA table_info("{name}")')]
            for col in cols:
                self.con.execute(f"UPDATE \"{name}\" SET \"{col}\" = zeroblob(length(\"{col}\")) "
                                 f"WHERE typeof(\"{col}\") = 'blob'")  # fmt: skip
        self.deleted_since_optimise = True

    def _drop_vectors_for(self, ids: list[int]) -> None:
        for key, (_model, dim) in self.vector_models().items():
            self._erase(self.vec_table(key), dim, ids)
            self._bump_vectors(key)

    def _vectors_by_hash(self, file_id: int) -> dict[str, dict[str, bytes]]:
        out: dict[str, dict[str, bytes]] = {}
        for key in self.vector_models():
            sql = (f"SELECT c.hash, v.embedding FROM chunks c JOIN {self.vec_table(key)} v "
                   "ON v.rowid = c.id WHERE c.file_id = ?")  # fmt: skip
            out[key] = {h: to_f32(v) for h, v in self.con.execute(sql, (file_id,))}
        return out

    def put_vectors(self, key: str, rows: list[tuple[int, bytes]]) -> None:
        """Store float32 vectors for chunks, with their file's feature and tier for filtering.
        Kept as float16 for re-scoring (half the size; ample precision for unit vectors) and
        as sign bits for the shortlist."""
        meta: dict[int, tuple[str, str]] = {}
        ids = [cid for cid, _v in rows]
        for i in range(0, len(ids), 500):
            part = ids[i : i + 500]
            meta.update((cid, (feature, tier)) for cid, feature, tier in self.con.execute(
                "SELECT c.id, COALESCE(f.feature, ''), COALESCE(f.tier, '') FROM chunks c "
                f"JOIN files f ON f.id = c.file_id WHERE c.id IN ({','.join('?' * len(part))})",
                part))  # fmt: skip
        sql = (f"INSERT INTO {self.vec_table(key)}(id, feature, tier, bits, embedding) "
               "VALUES (?,?,?,?,?)")  # fmt: skip
        stored = [(cid, *meta[cid], *_stored_forms(v)) for cid, v in rows if cid in meta]
        self.con.executemany(sql, [(cid, f, t, bits, half) for cid, f, t, half, bits in stored])
        if rows:
            self._bump_vectors(key)

    def _pending_where(self, key: str) -> str:
        """Chunks still to embed: no vector, in an enabled feature (a hidden feature's text
        is not read again), and not an input this model already rejected."""
        return (f"c.id NOT IN (SELECT rowid FROM {self.vec_table(key)}) "
                "AND f.feature IN (SELECT id FROM features WHERE enabled = 1) "
                "AND c.hash NOT IN "
                "(SELECT hash FROM embed_failures WHERE model_id = ?)")  # fmt: skip

    def pending_chunks(self, key: str, limit: int = 256) -> list[tuple[int, str, str]]:
        """(chunk id, hash, embedding input) of chunks with no vector yet for this embedder."""
        model_id = self.vector_models()[key][0]
        sql = (f"SELECT c.id, c.hash, c.text, f.rel FROM chunks c JOIN files f "
               f"ON f.id = c.file_id WHERE {self._pending_where(key)} "
               "ORDER BY c.id LIMIT ?")  # fmt: skip
        return [(cid, h, embed_input(rel, text)) for cid, h, text, rel in
                self.con.execute(sql, (model_id, limit))]  # fmt: skip

    def count_pending(self, key: str, open_only: bool = False) -> int:
        """Chunks still to embed; with open_only, only those an open server may see."""
        model_id = self.vector_models()[key][0]
        sql = (f"SELECT COUNT(*) FROM chunks c JOIN files f ON f.id = c.file_id "
               f"WHERE {self._pending_where(key)}"
               + (f" AND {OPEN_VISIBLE}" if open_only else ""))  # fmt: skip
        return int(self.con.execute(sql, (model_id,)).fetchone()[0])

    def record_failures(self, key: str, hashes: Iterable[str]) -> None:
        model_id = self.vector_models()[key][0]
        self.con.executemany("INSERT OR IGNORE INTO embed_failures VALUES (?, ?)",
                             [(h, model_id) for h in hashes])  # fmt: skip

    def count_failed(self, key: str) -> int:
        """Chunks whose input this embedder rejected: found by keywords only."""
        model_id = self.vector_models()[key][0]
        sql = ("SELECT COUNT(*) FROM chunks WHERE hash IN "
               "(SELECT hash FROM embed_failures WHERE model_id = ?)")  # fmt: skip
        return int(self.con.execute(sql, (model_id,)).fetchone()[0])

    def vectors_complete(self, key: str, open_only: bool = False) -> bool:
        """Whether every visible chunk has a vector (or was rejected by the model). Kept as a
        flag so a search does not count rows; cleared whenever a chunk without a vector
        appears, set by the embedding pass when nothing is pending. The open scope covers
        only what an open server may see, so private additions do not change its view."""
        name = "vectors_complete_open" if open_only else "vectors_complete"
        return self.get_meta(f"{name}:{key}") == "1"

    def _set_complete(self, key: str, done: bool, open_too: bool = True) -> None:
        names = ["vectors_complete", "vectors_complete_open"] if done or open_too else [
            "vectors_complete"]  # fmt: skip
        for name in names:
            if done:
                self.set_meta(f"{name}:{key}", "1")
            else:
                self.con.execute("DELETE FROM meta WHERE key=?", (f"{name}:{key}",))

    def mark_complete_if_done(self, key: str) -> int:
        """Set the completeness flags if nothing is pending; return the pending count."""
        pending = self.count_pending(key)
        if pending == 0:
            self._set_complete(key, True)
        else:
            self._set_complete(key, False, open_too=False)
            if self.count_pending(key, open_only=True) == 0:
                self.set_meta(f"vectors_complete_open:{key}", "1")
            else:
                self._set_complete(key, False, open_too=True)
        return pending

    def vectors_for_hashes(self, key: str, hashes: list[str]) -> dict[str, bytes]:
        """Existing vectors for identical text, reused instead of embedding it again."""
        out: dict[str, bytes] = {}
        for i in range(0, len(hashes), 500):
            part = hashes[i : i + 500]
            sql = (f"SELECT c.hash, v.embedding FROM chunks c JOIN {self.vec_table(key)} v "
                   f"ON v.rowid = c.id WHERE c.hash IN ({','.join('?' * len(part))})")  # fmt: skip
            out.update((h, to_f32(v)) for h, v in self.con.execute(sql, part))
        return out

    MAX_K = 4096  # the most neighbours a search asks for

    def knn(self, key: str, query: bytes, k: int, features: list[str],
            tiers: tuple[str, ...], only: Any = None) -> list[tuple[int, float]]:  # fmt: skip
        """The k nearest chunks by cosine distance, nearest first, among the given features
        and file tiers only (and, if given, among the chunk ids `only`: a search filter's
        matches, so a narrow filter still finds its nearest chunks). A Hamming
        scan over the in-memory sign bits, restricted to what the caller may see,
        shortlists SHORTLIST chunks; those are re-scored exactly with their float vectors
       ."""
        import numpy as np

        if not features or not tiers or k <= 0:
            return []
        k = min(k, self.MAX_K)
        c = self._vectors(key)
        if c.ids.size == 0:
            return []
        q = np.frombuffer(query, dtype=np.float32)
        feats = [c.features.index(f) for f in features if f in c.features]
        tcodes = [c.tiers.index(t) for t in tiers if t in c.tiers]
        mask = np.isin(c.feature, feats) & np.isin(c.tier, tcodes)
        if only is not None:
            mask &= np.isin(c.ids, only)
        visible = np.flatnonzero(mask)
        if visible.size == 0:
            return []
        n = max(
            k,
            min(
                self.SHORTLIST_MAX,
                max(self.SHORTLIST_MIN, int(self.SHORTLIST_SHARE * visible.size)),
            ),
        )
        if visible.size > n:
            qbits = np.frombuffer(sign_bits(query), dtype=np.uint8)
            if c.bits.dtype == np.uint64:
                qbits = qbits.view(np.uint64)
            dist = np.bitwise_count(c.bits[visible] ^ qbits).sum(axis=1, dtype=np.int64)
            visible = visible[np.argpartition(dist, n - 1)[:n]]
        ids = c.ids[visible]
        found: list[tuple[int, bytes]] = []
        for i in range(0, ids.size, 900):
            part = [int(x) for x in ids[i : i + 900]]
            found += self.con.execute(
                f"SELECT id, embedding FROM {self.vec_table(key)} WHERE id IN "
                f"({','.join('?' * len(part))})", part).fetchall()  # fmt: skip
        if not found:
            return []
        rid = np.array([r[0] for r in found], dtype=np.int64)
        mat = np.frombuffer(b"".join(r[1] for r in found), dtype=np.float16).reshape(
            len(found), -1).astype(np.float32)  # fmt: skip
        norms = np.linalg.norm(mat, axis=1) * float(np.linalg.norm(q))
        dist = 1.0 - (mat @ q) / np.where(norms == 0, 1.0, norms)
        order = np.lexsort((rid, dist))[:k]  # nearest first; ties by chunk id
        return [(int(rid[i]), float(dist[i])) for i in order]

    def _vectors(self, key: str) -> _VecCache:
        """The in-memory copy of an embedder's sign bits, feature and tier codes, reloaded
        when the vectors' version counter moves."""
        import numpy as np

        version = self.get_meta(f"vectors_version:{key}") or "0"
        cached = self._vec_cache.get(key)
        if cached is not None and cached.version == version:
            return cached
        table = self.vec_table(key)
        fidx: dict[str, int] = {}
        tidx: dict[str, int] = {}
        parts: list[tuple[Any, Any, Any, Any]] = []
        cur = self.con.execute(f"SELECT id, feature, tier, bits FROM {table} ORDER BY id")
        while batch := cur.fetchmany(50_000):  # bounded memory while loading
            parts.append((
                np.fromiter((r[0] for r in batch), dtype=np.int64, count=len(batch)),
                np.fromiter((fidx.setdefault(r[1], len(fidx)) for r in batch), dtype=np.int16,
                            count=len(batch)),
                np.fromiter((tidx.setdefault(r[2], len(tidx)) for r in batch), dtype=np.int16,
                            count=len(batch)),
                np.frombuffer(b"".join(r[3] for r in batch), dtype=np.uint8).reshape(
                    len(batch), -1),
            ))  # fmt: skip
        if parts:
            ids, fcode, tcode, bits = (np.concatenate([p[i] for p in parts]) for i in range(4))
        else:
            ids, fcode, tcode = (np.empty(0, dtype=t) for t in (np.int64, np.int16, np.int16))
            bits = np.empty((0, 8), dtype=np.uint8)
        features, tiers = list(fidx), list(tidx)
        if bits.shape[1] % 8 == 0:
            bits = bits.view(np.uint64)
        cache = _VecCache(version, ids, fcode, features, tcode, tiers, bits)
        self._vec_cache[key] = cache
        return cache

    def set_tier(self, path: str, path_tier: str, tier: str) -> None:
        row = self.con.execute("SELECT id FROM files WHERE path=?", (path,)).fetchone()
        was_open = row is not None and self._is_open_visible(row[0])
        self.con.execute("UPDATE files SET path_tier=?, tier=?, updated_at=? WHERE path=?",
                         (path_tier, tier, _now(), path))  # fmt: skip
        now_open = row is not None and self._is_open_visible(row[0])
        if row is not None and was_open != now_open:
            self._fts_move("c.file_id = ?", (row[0],), to_open=now_open)
            if now_open:
                self._recheck_open_completeness()
        models = self.vector_models()
        if not models:
            return
        sql = "SELECT c.id FROM chunks c JOIN files f ON f.id = c.file_id WHERE f.path=?"
        ids = [r[0] for r in self.con.execute(sql, (path,))]
        for key in models:
            self.con.executemany(f"UPDATE {self.vec_table(key)} SET tier = ? WHERE rowid = ?",
                                 [(tier, cid) for cid in ids])  # fmt: skip
            if ids:
                self._bump_vectors(key)

    def remove(self, path: str) -> None:
        """Forget a file that is gone: its rows, its text, and its name in the audit log
        (a file name can be sensitive too, and the file no longer exists)."""
        old = self.get_file(path)
        if old is not None:
            self._drop_chunks(old.id)
            self.con.execute("DELETE FROM files WHERE id=?", (old.id,))
        self.con.execute("DELETE FROM audit WHERE path=?", (path,))

    def all_paths(self) -> list[str]:
        return [r[0] for r in self.con.execute("SELECT path FROM files")]

    def paths_under(self, roots: Iterable[str]) -> list[str]:
        out = []
        for root in roots:
            out += [r[0] for r in self.con.execute(
                "SELECT path FROM files WHERE root=? OR path=?", (root, root))]  # fmt: skip
        return out

    def indexed_rows(self) -> list[tuple[str, str | None, str | None, int, str | None]]:
        """(path, rel, root, n_secrets, feature) of every indexed file, for re-tiering."""
        sql = "SELECT path, rel, root, n_secrets, feature FROM files WHERE status='indexed'"
        return list(self.con.execute(sql))

    # -- features -----------------------------------------------------------------------------

    def set_feature(self, feature_id: str, enabled: bool, tier: str) -> None:
        """Mirror a feature's state; readers take visibility from here, so it acts at once."""
        was_on, was_tier = self.feature_states().get(feature_id, (False, ""))
        self._set_feature_row(feature_id, enabled, tier)
        was_open, now_open = was_on and was_tier == "open", enabled and tier == "open"
        if was_open != now_open:  # its open files' keyword rows change audience
            self._fts_move("f.feature = ? AND f.status = 'indexed' AND f.tier = 'open'",
                           (feature_id,), to_open=now_open)  # fmt: skip
        if enabled and not was_on and self.vectors_ok:
            for key in self.vector_models():  # its chunks may never have been embedded
                if self.count_pending(key):
                    self._set_complete(key, False, open_too=False)
        if now_open and not was_open:
            self._recheck_open_completeness()

    def _recheck_open_completeness(self) -> None:
        """Chunks just became visible to an open server: if any lack a vector, its view of
        the index is no longer complete."""
        if not self.vectors_ok:
            return
        for key in self.vector_models():
            if self.count_pending(key, open_only=True):
                self.con.execute("DELETE FROM meta WHERE key=?", (f"vectors_complete_open:{key}",))

    def _set_feature_row(self, feature_id: str, enabled: bool, tier: str) -> None:
        self.con.execute(
            "INSERT INTO features(id, enabled, tier, updated_at) VALUES (?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET enabled=excluded.enabled, tier=excluded.tier, "
            "updated_at=excluded.updated_at",
            (feature_id, int(enabled), tier, _now()),
        )

    def feature_states(self) -> dict[str, tuple[bool, str]]:
        return {
            i: (bool(e), t)
            for i, e, t in self.con.execute("SELECT id, enabled, tier FROM features")
        }

    def feature_stats(self) -> dict[str, dict[str, int]]:
        """Per feature: indexed files, skipped files, chunks and stored text bytes."""
        out: dict[str, dict[str, int]] = {}
        for feature, status, n in self.con.execute(
            "SELECT feature, status, COUNT(*) FROM files WHERE feature IS NOT NULL "
            "GROUP BY feature, status"
        ):
            out.setdefault(feature, {"files": 0, "skipped": 0, "chunks": 0, "bytes": 0})
            out[feature]["files" if status == "indexed" else "skipped"] = n
        for feature, chunks, nbytes in self.con.execute(
            "SELECT f.feature, COUNT(c.id), COALESCE(SUM(LENGTH(c.text)), 0) "
            "FROM chunks c JOIN files f ON f.id = c.file_id WHERE f.feature IS NOT NULL "
            "GROUP BY f.feature"
        ):
            out.setdefault(feature, {"files": 0, "skipped": 0, "chunks": 0, "bytes": 0})
            out[feature]["chunks"], out[feature]["bytes"] = chunks, nbytes
        if self.vectors_ok:
            for key, (_model, dim) in self.vector_models().items():  # an embedder's vectors
                n = self.con.execute(f"SELECT COUNT(*) FROM {self.vec_table(key)}").fetchone()[0]
                out[f"embedder.{key}"] = {"files": 0, "skipped": 0, "chunks": int(n),
                                          "bytes": int(n) * 2 * dim}  # fmt: skip
        return out

    def purge_feature(self, feature_id: str) -> dict[str, int]:
        """Delete every file record, chunk and keyword-index entry of a feature, securely,
        and the audit rows that name those files (file names can be sensitive too).
        For an embedder, delete its vectors."""
        if feature_id.startswith("embedder."):
            model = self.vector_models().get(key := feature_id.split(".", 1)[1])
            if model is None:
                return {"files": 0, "skipped": 0, "chunks": 0, "bytes": 0}
            n = self.con.execute(f"SELECT COUNT(*) FROM {self.vec_table(key)}").fetchone()[0]
            self.drop_vectors(key)
            return {"files": 0, "skipped": 0, "chunks": int(n), "bytes": int(n) * 2 * model[1]}
        stats = self.feature_stats().get(feature_id, {"files": 0, "skipped": 0, "chunks": 0,
                                                       "bytes": 0})  # fmt: skip
        rows = list(self.con.execute("SELECT id, path FROM files WHERE feature=?", (feature_id,)))
        for fid, path in rows:
            self._drop_chunks(fid)
            self.con.execute("DELETE FROM audit WHERE path=?", (path,))
        self.con.execute("DELETE FROM files WHERE feature=?", (feature_id,))
        self.deleted_since_optimise = True
        self.optimise_fts()
        return stats

    def notes(self) -> list[tuple[str, str]]:
        """Indexed files with an extraction quality note (nothing silently half-read)."""
        sql = "SELECT path, note FROM files WHERE status='indexed' AND note IS NOT NULL"
        return list(self.con.execute(sql + " ORDER BY path"))

    def skipped(self) -> list[tuple[str, str]]:
        return list(self.con.execute(
            "SELECT path, reason FROM files WHERE status='skipped' ORDER BY path"))  # fmt: skip

    # -- audit ------------------------------------------------------------------------------

    def audit(self, kind: str, path: str | None, rule: str | None = None,
              line: int | None = None, count: int = 1) -> None:  # fmt: skip
        """Record what the safety layer did. There is deliberately no field for a value."""
        self.con.execute(
            "INSERT INTO audit(time, kind, path, rule, line, count) VALUES (?,?,?,?,?,?)",
            (_now(), kind, path, rule, line, count),
        )

    # -- lifecycle --------------------------------------------------------------------------

    # -- transactions -----------------------------------------------------------------------

    def savepoint(self) -> None:
        self.con.execute("SAVEPOINT one_file")

    def release(self) -> None:
        self.con.execute("RELEASE one_file")

    def rollback_to(self) -> None:
        self.con.execute("ROLLBACK TO one_file")
        self.con.execute("RELEASE one_file")

    def optimise_fts(self) -> None:
        """Rewrite the keyword index. FTS5 `secure-delete` removes a deleted row's postings
        but can leave the term strings in leaf pages; merging all segments drops them, and
        `secure_delete` zeroes the freed pages."""
        for t in FTS_TABLES:
            self.con.execute(f"INSERT INTO {t}({t}) VALUES('optimize')")
        self.deleted_since_optimise = False

    def commit(self) -> None:
        if self.con.in_transaction:
            self.con.execute("COMMIT")
        self.con.execute("BEGIN")

    def begin_immediate(self) -> None:
        """Take the write lock now (waiting up to the busy timeout), not at the first write."""
        if self.con.in_transaction:
            self.con.execute("COMMIT")
        self.con.execute("BEGIN IMMEDIATE")

    def rollback(self) -> None:
        if self.con.in_transaction:
            self.con.execute("ROLLBACK")

    def close(self, commit: bool = True) -> bool:
        """Commit (or roll back), fold the WAL into the main file and truncate it, then close.

        Pass commit=False when an error is propagating, so half-done work is never kept.
        Returns False if the WAL could not be truncated (a reader held a snapshot); old
        pages may then remain in the WAL until the next checkpoint.
        """
        if self.readonly:
            self.con.close()
            return True
        if not commit:
            self.rollback()
        elif self.deleted_since_optimise:
            self.optimise_fts()
        if self.con.in_transaction:
            self.con.execute("COMMIT")
        busy, frames, done = self.con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        self.con.close()
        return busy == 0 and frames == done
