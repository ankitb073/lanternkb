"""The extraction worker: read, extract, scan and redact one file at a time.

Run as `python -m lanternkb.extract.worker`, normally under `sandbox-exec` (no network, no
writes outside a private temporary folder, no reading of user files). It speaks JSON lines:
an `init` message, then one request per file, each preceded by the open file itself, sent
by the parent over a Unix socket. Only redacted text ever leaves this process.

`process()` is also called in-process by the inline mode (tests, evaluation), so both paths
share exactly the same logic.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import hashlib
import json
import os
import re
import resource
import socket
import stat as stat_
import sys
import threading
from dataclasses import asdict, dataclass, field
from typing import Any

from lanternkb.extract import get as get_extractor
from lanternkb.extract.base import ExtractError, ExtractResult, Limits, Segment
from lanternkb.scan.engine import Finding, Scanner, ScanTimeout, redact
from lanternkb.scan.rules import load_rules
from lanternkb.text import one_break


@dataclass
class Job:
    id: int
    path: str  # the real path the policy checked; opened without following links
    rel: str
    feature: str
    dev: int | None
    ino: int | None
    max_bytes: int
    limits: dict[str, int]  # cpu_seconds, memory_mb, output_mb (from the manifest)
    max_chars: int = 20_000_000  # the stricter of output_mb and the config's max_text_chars
    options: dict[str, Any] = field(default_factory=dict)  # a plugin's own config options
    tier: str = ""  # the path's tier: workers never mix open and local-only files


@dataclass
class Result:
    id: int
    skip: str | None = None
    note: str | None = None
    segments: list[dict[str, Any]] = field(default_factory=list)  # redacted
    findings: list[dict[str, Any]] = field(default_factory=list)  # rule, line, action
    n_secrets: int = 0
    digest: str = ""


def part_breaks(texts: list[str], locations: list[str]) -> tuple[int, ...]:
    """Offsets in `"\n".join(texts)` where another sheet starts (its location names another
    sheet: `Name!A1:D9`). The secret scan's table pass ends a table there, and nowhere else
    between an extractor's parts: pages, or blocks of rows of one sheet."""
    out, pos = [], 0
    for i, t in enumerate(texts):
        here, before = locations[i], locations[i - 1] if i else ""
        if i and "!" in here and here.split("!")[0] != before.split("!")[0]:
            out.append(pos)
        pos += len(t) + 1
    return tuple(out)


def open_checked(job: Job) -> int:
    """Open the checked target without following links or blocking, and confirm it is the
    file the policy checked (device and inode), a regular file with one link, under the cap.
    The pool calls this in the parent and hands the worker the open file: the sandbox lets
    workers read no user files at all."""
    try:
        fd = os.open(job.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as e:
        raise ExtractError("symlink-changed" if e.errno == errno.ELOOP else "unreadable") from e
    try:
        st = os.fstat(fd)
        if not stat_.S_ISREG(st.st_mode):
            raise ExtractError("not-a-file")
        if job.dev is not None and (st.st_dev, st.st_ino) != (job.dev, job.ino):
            raise ExtractError("changed-during-sync")
        if st.st_nlink > 1:
            raise ExtractError("hard-link")
        if st.st_size > job.max_bytes:
            raise ExtractError("too-large")
        # Where the opened file really is: a folder on the way swapped for a link after
        # the policy checked it would lead elsewhere. O_NOFOLLOW guards only
        # the last part of the path.
        real = _opened_path(fd)
        if real is not None and _same_place(real, job.path) is False:
            raise ExtractError("changed-during-sync")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _opened_path(fd: int) -> str | None:
    """The kernel's path for an open file (macOS F_GETPATH, Linux /proc), or None."""
    try:
        if sys.platform == "darwin":
            import fcntl

            raw = fcntl.fcntl(fd, fcntl.F_GETPATH, bytes(1024))
            return raw.rstrip(b"\0").decode("utf-8", "surrogateescape")
        return os.readlink(f"/proc/self/fd/{fd}")
    except (OSError, AttributeError, ValueError):
        return None


def _same_place(real: str, checked: str) -> bool | None:
    """Whether the opened file's real path is the checked one (case as on disk); None if
    that cannot be told."""
    from pathlib import Path

    from lanternkb.fsutil import canonical_case

    try:  # the checked path was resolved when it was checked: never resolve it again here
        return canonical_case(Path(real)) == canonical_case(Path(checked))
    except OSError:
        return None


def read_bytes(job: Job, fd: int | None = None) -> bytes:
    """The file's bytes, from an already checked open file (closed here) or by opening it."""
    if fd is None:
        fd = open_checked(job)
    with os.fdopen(fd, "rb") as f:
        data = f.read(job.max_bytes + 1)
    if len(data) > job.max_bytes:
        raise ExtractError("too-large")
    return data


MAX_LOCATION = 120  # longer locations (a sheet name is at most 31 characters) are replaced


# What an extractor may say about a file it skips: a short code. Free text could carry the
# file's content (a quoted line, a secret) past the scanner into the index and audit log,
# and `secret:` verdicts belong to the scanner alone.
REASON = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
MAX_NOTE = 300


def safe_reason(reason: object) -> str:
    r = str(reason)
    return r if REASON.fullmatch(r) else "extract-refused"


def _safe_note(note: object, scanner: Scanner, rel: str) -> str | None:
    """A quality note is text the user sees: scanned and redacted like content, and short."""
    if not note:
        return None
    text = str(note)[:MAX_NOTE]
    try:
        found = scanner.scan(text, rel)
    except ScanTimeout:
        return "note withheld"
    return redact(text, found) if found else text


_EXTRACTORS: dict[tuple[str, str], Any] = {}


def _extractor(feature: str, options: dict[str, Any]) -> Any:
    """One instance per feature and options for the worker's life (a plugin may load a
    model in its factory). A plugin that cannot be found or built skips the file with a
    reason, and the worker lives on."""
    key = (feature, json.dumps(options, sort_keys=True, default=str))
    if key not in _EXTRACTORS:
        from lanternkb.features import ManifestError

        try:
            _EXTRACTORS[key] = get_extractor(feature, options)
        except KeyError as e:
            raise ExtractError("plugin-not-found") from e
        except ManifestError as e:
            raise ExtractError("plugin-api-mismatch" if "api_version" in str(e)
                               else "plugin-refused") from e  # fmt: skip
        except Exception as e:  # noqa: BLE001 - a broken plugin factory
            raise ExtractError("plugin-failed-to-load") from e
    return _EXTRACTORS[key]


def process(job: Job, scanner: Scanner, fd: int | None = None) -> Result:
    """Read, extract, scan and redact one file. Never returns unredacted text.

    The segments are scanned as one text, so a secret that runs across a page, slide or row
    block is seen whole; findings are mapped back to each segment. Locations are content too
    (a sheet name can be anything) and are scanned and redacted the same way.
    """
    limits = Limits(**job.limits, max_chars=min(job.limits["output_mb"] * 1_000_000,
                                                 job.max_chars))  # fmt: skip
    try:
        data = read_bytes(job, fd)
        extracted = _extractor(job.feature, job.options).extract(data, job.rel, limits)
        segs = list(extracted.segments) if isinstance(extracted, ExtractResult) else None
    except ExtractError as e:
        return Result(job.id, skip=safe_reason(e.reason))
    except Exception:  # noqa: BLE001 - a parser bug costs this file, not the worker
        return Result(job.id, skip="extract-failed")
    raw = raw_view(data) if job.feature in DECODED else None
    if raw is not None:
        raw = one_break(raw)  # line breaks as the text is scanned
    del data
    if segs is None or not all(isinstance(x, Segment) and isinstance(x.text, str)
                               and isinstance(x.location, str) for x in segs):  # fmt: skip
        return Result(job.id, skip="extract-bad-output")
    note = _safe_note(extracted.note, scanner, job.rel)
    if extracted.skip:
        return Result(job.id, skip=safe_reason(extracted.skip), note=note)
    if sum(len(s.text) for s in segs) > limits.max_chars:
        return Result(job.id, skip="too-large-text")
    texts = [one_break(s.text) for s in segs]  # scanned as it will be stored
    starts, pos = [], 0
    for t in texts:
        starts.append(pos)
        pos += len(t) + 1
    try:
        found = scanner.scan("\n".join(texts), job.rel,
                             breaks=part_breaks(texts, [s.location for s in segs]))  # fmt: skip
        locs = [one_break(s.location) for s in segs]  # as stored, like the text
        loc_found = [scanner.scan(loc, job.rel) for loc in locs]
        if raw is not None and hidden_by_decoding(scanner, raw, "\n".join(texts), job.rel):
            return Result(job.id, skip="mixed-encoding", note=note)
    except ScanTimeout:
        return Result(job.id, skip="scan-timeout")
    out = Result(job.id, note=note)
    whole = [f.rule for f in [*found, *(f for fs in loc_found for f in fs)]
             if f.action == "skip-file"]  # fmt: skip
    per_seg: list[list[Finding]] = [[] for _ in segs]
    for f in found:
        for i, st in enumerate(starts):
            s, e = max(f.start, st), min(f.end, st + len(texts[i]))
            if s < e or (f.end == f.start and st <= f.start <= st + len(texts[i])):
                line = texts[i].count("\n", 0, s - st) + 1
                per_seg[i].append(Finding(f.rule, s - st, max(s, e) - st, line, f.action,
                                          f.encoded))  # fmt: skip
                out.findings.append({"rule": f.rule, "line": line, "action": f.action,
                                     "segment": i})  # fmt: skip
    locations = []
    for i in range(len(segs)):
        loc = redact(locs[i], loc_found[i]) if loc_found[i] else locs[i]
        for f in loc_found[i]:
            out.findings.append({"rule": f.rule, "line": 0, "action": f.action, "segment": i})
        locations.append(loc if len(loc) <= MAX_LOCATION else f"part {i + 1}")
    if whole:
        return Result(job.id, skip=f"secret:{whole[0]}", findings=out.findings,
                      n_secrets=len(out.findings))  # fmt: skip
    digest = hashlib.sha256()
    for i, seg in enumerate(segs):
        text = redact(texts[i], per_seg[i]) if per_seg[i] else texts[i]
        digest.update(text.encode())
        out.segments.append({"text": text, "location": locations[i],
                             "start_line": seg.start_line, "end_line": seg.end_line})  # fmt: skip
    out.n_secrets = len(out.findings)
    out.digest = digest.hexdigest()
    return out


DECODED = frozenset({"extractor.text", "extractor.code"})  # bytes decoded as text


def raw_view(data: bytes) -> str | None:
    """The raw bytes with zero bytes dropped, one character per byte, for a file whose
    decoding could hide part of it: one with zero bytes (UTF-16, or UTF-16
    pieces appended to other text) or a UTF-16 byte order mark. None for other files."""
    if b"\x00" not in data and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return None
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        data = data[2:]  # the byte order mark is not text
    return data.replace(b"\x00", b"").decode("latin-1")


def hidden_by_decoding(scanner: Scanner, raw: str, text: str, rel: str) -> bool:
    """Whether the raw bytes hold a secret the decoded text does not show as such: a file
    mixing UTF-16 with other text decodes part of itself into characters no rule can read
    (an ASCII line becomes CJK). Such a file is refused, never indexed half-read."""
    if raw == text:
        return False  # every byte read as itself: all-ASCII UTF-16, nothing can hide
    seen: set[str] = set()
    for f in scanner.scan(raw, rel):
        # the span's first line: a view may stretch it over a break into bytes that read
        # differently in the text (Japanese UTF-16BE)
        secret = raw[f.start : f.end].strip().split("\n", 1)[0].strip()
        if not secret or secret in seen:
            continue
        seen.add(secret)
        # hidden if the text shows it fewer times: one copy visible does not cover another
        # decoded into other characters
        if raw.count(secret) > text.count(secret):
            return True
    return False


# -- the worker process ------------------------------------------------------------------------

RETIRE_CPU = 1800  # seconds of CPU after which a worker hands over to a fresh one


def current_rss_mb() -> float:
    """Resident memory now (not the peak), in MB."""
    if sys.platform == "darwin":
        libproc = ctypes.CDLL(ctypes.util.find_library("proc"))
        buf = (ctypes.c_uint64 * 12)()  # struct proc_taskinfo starts with virtual, resident
        n = libproc.proc_pidinfo(os.getpid(), 4, 0, ctypes.byref(buf), ctypes.sizeof(buf))
        return buf[1] / 1e6 if n > 0 else 0.0
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6
    except OSError:
        return 0.0


class _Out:
    """Serialised writes to stdout (the watchdog thread may write too)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()

    def send(self, msg: dict[str, Any]) -> None:
        line = json.dumps(msg, ensure_ascii=False)
        with self.lock:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()


def main() -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    out = _Out()
    init = json.loads(sys.stdin.readline())["init"]
    # Retire voluntarily, between files, after RETIRE_CPU seconds of CPU; the hard limit is
    # only a backstop above that plus the longest per-file allowance (audio: 900 s).
    hard = RETIRE_CPU + 1200
    resource.setrlimit(resource.RLIMIT_CPU, (hard, hard + 5))
    scanner = Scanner(load_rules(disabled=tuple(init.get("disabled_rules", [])),
                                 actions=init.get("actions") or {},
                                 enabled=tuple(init.get("enabled_rules", []))))  # fmt: skip
    sandboxed, why = init.get("sandboxed"), None
    if init.get("linux_sandbox"):  # before any file arrives
        from lanternkb.extract import linux_sandbox

        box = init["linux_sandbox"]
        try:
            linux_sandbox.restrict(box["tmp"], box["reads"], box["execs"])
            sandboxed = True
        except OSError as e:  # the parent then reports `extraction process`, not sandboxed
            sandboxed, why = False, f"{type(e).__name__}: {e}"[:200]
    out.send({"ready": True, "rules_hash": scanner.hash, "sandboxed": sandboxed,
              "sandbox_error": why})  # fmt: skip
    files = socket.socket(fileno=init["files_fd"]) if init.get("files_fd") else None
    for line in sys.stdin:
        msg = json.loads(line)
        if "rescan" in msg:  # a scanner-only worker checking another worker's output
            try:
                breaks = tuple(int(b) for b in msg.get("breaks", []))
                found = sum(len(scanner.scan(t, msg.get("rel", ""), redacted=True,
                                             breaks=breaks if k == 0 else ()))
                            for k, t in enumerate(msg["rescan"]))  # fmt: skip
            except ScanTimeout:
                found = -1
            out.send({"found": found})
            continue
        job = Job(**msg)
        fd = None
        if files is not None:  # the parent sends the open file just before the job
            _msg, fds, _flags, _addr = socket.recv_fds(files, 1, 1)
            fd = fds[0] if fds else None
            if fd is None:
                break
        out.send(asdict(process(job, scanner, fd)))
        used = resource.getrusage(resource.RUSAGE_SELF)
        if used.ru_utime + used.ru_stime > RETIRE_CPU:
            break  # the pool sees the exit and starts a fresh worker


if __name__ == "__main__":
    main()
