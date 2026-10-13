"""A pool of sandboxed extraction workers.

Each worker is a long-lived `python -I -m lanternkb.extract.worker` process in its own
process group, started in its private temporary folder and, on macOS, under `sandbox-exec`
(deny by default): no network, no reading of user files (the parent opens each checked file
and hands the worker the open descriptor), no writes outside that folder, no services, no
signals to anything but itself. One thread per worker sends it jobs.

The parent, not the worker, enforces the limits on the whole process group (the worker and
any helper it starts, such as ffmpeg): a job past its time limit or memory limit gets the
group killed and the worker replaced. Replies are read with a deadline and a size cap and
checked before use; anything malformed costs that one file (`extract-crashed`), never the
sync.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import functools
import json
import os
import pwd
import queue
import select
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from lanternkb.extract.base import CPU_ONLY_ENV, ExtractError
from lanternkb.extract.worker import MAX_NOTE, REASON, Job, Result, open_checked, part_breaks
from lanternkb.text import one_break

SANDBOX_EXEC = "/usr/bin/sandbox-exec"
# Deny by default. Every extractor, built in or a plugin, gets only: running its
# own Python, reading the system and Python's code (never user files: the parent opens each
# file and hands it over), writing its private temp folder, reading system settings, and
# signalling itself. No network, no other processes, no IPC to other services (so nothing
# can ask another app to send data anywhere), no hardware. File metadata (stat) stays
# readable everywhere: Python resolves paths with it; it shows no file's contents.
_SYSTEM_READS = ('(subpath "/System") (subpath "/usr") (subpath "/private/var/db/timezone")'
                 ' (literal "/dev/null") (literal "/dev/urandom") (literal "/dev/random")'
                 ' (literal "/private/etc/localtime") (literal "/") (subpath "/dev/fd")'
                 ' (literal "/Library/Preferences/.GlobalPreferences.plist")'
                 ' (literal "/private/etc/passwd") (subpath "/private/etc/ssl")')  # fmt: skip
_COMMON = (
    "(version 1)(deny default)"
    "(allow file-read-metadata)"
    f'(allow file-read* {_SYSTEM_READS} (subpath (param "TMP")) READS)'
    '(allow file-write* (literal "/dev/null") (subpath (param "TMP")))'
    # System facts only: never kern.procargs2 or kern.proc (other processes' arguments and
    # environment, where tokens live), so no blanket sysctl-read.
    '(allow sysctl-read (sysctl-name-prefix "hw.") (sysctl-name-prefix "kern.os")'
    ' (sysctl-name "kern.hostname") (sysctl-name "kern.boottime") (sysctl-name "kern.version")'
    ' (sysctl-name "kern.argmax") (sysctl-name "kern.maxfilesperproc")'
    ' (sysctl-name "kern.usrstack64") (sysctl-name "sysctl.proc_translated")'
    ' (sysctl-name "kern.secure_kernel") (sysctl-name "vm.footprint_suspend")'
    ' (sysctl-name-prefix "machdep.cpu.") (sysctl-name "kern.iossupportversion")'
    ' (sysctl-name "kern.hv_vmm_present") (sysctl-name "security.mac.lockdown_mode_state"))'
    # Other processes' details (their arguments and environment included) stay hidden.
    "(deny process-info*)(allow process-info* (target self))"
    "(allow signal (target self))"
)
# Python itself, anywhere in its install: python.org's macOS build starts as bin/python3.x
# and re-launches as Resources/Python.app (a strict worker cannot fork, so an exec only
# ever replaces the worker).
PROFILE = _COMMON + '(allow process-exec (literal (param "PYTHON")) (subpath (param "PYHOME")))'
# Images (macOS Vision on the Neural Engine) and audio (ffmpeg, then MLX Whisper on the GPU)
# also need: helper processes and their libraries, the models folder, the GPU and Neural
# Engine, and three services found by testing: the Neural Engine, the directory helper that
# names the cache folder, and Metal's shader compiler. The compiler runs as its own system
# service and may write Metal's caches only through an extension the worker issues it, so
# the worker may issue one for those caches and nothing else. Only workers for those
# features get this profile; both features are off by default.
MEDIA_FEATURES = frozenset({"extractor.images", "extractor.audio"})
MEDIA_TOOLS = ("ffmpeg", "tesseract", "whisper-cli")  # the only programs media workers start
_CACHES = ('(subpath (param "METAL_FE")) (subpath (param "METAL")) (subpath (param "GPU_ARCHIVE"))'
           ' (subpath (param "E5_CACHE"))')  # fmt: skip
MEDIA_PROFILE = _COMMON + (
    "(allow process-fork)(allow process-exec* EXECS)(allow signal (target children))"
    '(allow file-read* (subpath "/opt/homebrew") (subpath "/Library/Apple")'
    f' (subpath (param "MODELS")) {_CACHES})'
    '(allow mach-lookup (global-name "com.apple.appleneuralengine")'
    ' (global-name "com.apple.bsd.dirhelper") (global-name "com.apple.MTLCompilerService")'
    ' (xpc-service-name "com.apple.MTLCompilerService"))'
    "(allow iokit-open)"
    f"(allow file-write* {_CACHES})"
    '(allow file-issue-extension (require-all (extension-class "com.apple.app-sandbox.read-write")'
    f" (require-any {_CACHES})))"
)
# Media workers whose output may be served to an open client share nothing with local-only
# ones: the GPU's and Neural Engine's caches, Metal's compiler and the Neural
# Engine service are one per user, and a local-only worker taken over by a hostile file
# could leave text there for an open one to pick up. So these workers get none of them:
# images are read with Tesseract (Vision needs those caches even on the CPU) and audio is
# transcribed on the CPU (MLX still opens the GPU device to list it, and never runs on it).
OPEN_MEDIA_PROFILE = _COMMON + (
    "(allow process-fork)(allow process-exec* EXECS)(allow signal (target children))"
    '(allow file-read* (subpath "/opt/homebrew") (subpath (param "MODELS")))'
    "(allow iokit-open)"
)


def _metal_caches() -> dict[str, str]:
    """The GPU's and Neural Engine's caches for this user (resolved: the sandbox matches real
    paths)."""
    try:
        out = subprocess.run(["getconf", "DARWIN_USER_CACHE_DIR"], capture_output=True,
                             text=True, timeout=5).stdout.strip()  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        out = ""
    base = os.path.realpath(out) if out else os.path.realpath(tempfile.gettempdir())
    names = {"METAL_FE": "com.apple.metalfe", "METAL": "com.apple.metal",
             "GPU_ARCHIVE": "com.apple.gpuarchiver"}  # fmt: skip
    out_ = {k: os.path.join(base, v) for k, v in names.items()}
    # The Neural Engine keeps compiled models per program in the home folder's caches.
    home = os.path.realpath(pwd.getpwuid(os.getuid()).pw_dir)
    out_["E5_CACHE"] = os.path.join(home, "Library", "Caches", "python",
                                    "com.apple.e5rt.e5bundlecache")  # fmt: skip
    return out_


@functools.cache
def _code_dirs() -> tuple[str, ...]:
    """The folders the worker's Python loads code from: its install, the virtual
    environment, and every entry on the sys.path of `python -I` (site-packages, and the
    source folders of editable installs, plugins included). A folder that contains the home
    folder is never allowed: that would be every user file."""
    try:
        out = subprocess.run([sys.executable, "-I", "-c", "import json, sys; "
                              "print(json.dumps(sys.path))"], capture_output=True, text=True,
                             timeout=60)  # fmt: skip
        paths = json.loads(out.stdout) if out.returncode == 0 else []
    except (OSError, subprocess.SubprocessError, ValueError):
        paths = []
    home = os.path.realpath(pwd.getpwuid(os.getuid()).pw_dir)
    found = set()
    for p in [*paths, sys.prefix, sys.base_prefix]:
        real = os.path.realpath(p) if p else ""
        if real and os.path.isdir(real) and not (home + os.sep).startswith(real.rstrip(os.sep)
                                                                          + os.sep):  # fmt: skip
            found.add(real)
    return tuple(d for d in sorted(found)
                 if not any(d.startswith(o + os.sep) for o in found if o != d))  # fmt: skip


def _sandbox_command(tmp: str, media: bool, open_tier: bool = False) -> list[str]:
    """sandbox-exec with the profile and its parameters (paths go in as parameters, never
    pasted into the profile's text). `open_tier` media workers get `OPEN_MEDIA_PROFILE`."""
    from lanternkb.models import models_dir

    params = {"TMP": tmp, "PYTHON": os.path.realpath(sys.executable),
              "PYHOME": os.path.realpath(sys.base_prefix)}  # fmt: skip
    reads = []
    for i, d in enumerate(_code_dirs()):
        params[f"READ_{i}"] = d
        reads.append(f'(subpath (param "READ_{i}"))')
    shared_gpu = media and not open_tier
    profile = (MEDIA_PROFILE if shared_gpu else OPEN_MEDIA_PROFILE if media
               else PROFILE).replace("READS", " ".join(reads))  # fmt: skip
    if media:
        if shared_gpu:
            params.update(_metal_caches())
        params["MODELS"] = os.path.realpath(models_dir())
        execs = ['(literal (param "PYTHON")) (subpath (param "PYHOME"))']
        for i, tool in enumerate(MEDIA_TOOLS):
            if found := shutil.which(tool):
                params[f"EXEC_{i}"] = os.path.realpath(found)
                execs.append(f'(literal (param "EXEC_{i}"))')
        profile = profile.replace("EXECS", " ".join(execs))
    args = [SANDBOX_EXEC]
    for k, v in params.items():
        args += ["-D", f"{k}={v}"]
    return [*args, "-p", profile]


POLL_S = 0.05
TMP_PREFIX = "lanternkb-worker-"  # + the parent's pid: a folder whose parent died is stale


def remove_stale_worker_folders() -> int:
    """Worker temp folders left by a sync that was killed (an audio file's copy and its
    decoded sound can sit there). Only ours, only when the process that made them is gone."""
    removed = 0
    base = Path(tempfile.gettempdir())
    for d in base.glob(f"{TMP_PREFIX}*"):
        pid = d.name[len(TMP_PREFIX) :].split("-", 1)[0]
        if not pid.isdigit() or int(pid) == os.getpid() or d.is_symlink():
            continue
        try:
            if d.stat().st_uid != os.getuid():
                continue
            os.kill(int(pid), 0)
            continue  # still running
        except ProcessLookupError:
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
        except (PermissionError, OSError):
            continue
    return removed


RESULT_FIELDS = {f.name for f in fields(Result)}


def sandbox_available() -> bool:
    """macOS: sandbox-exec. Linux: Landlock and seccomp, which workers apply to themselves
    (`linux_sandbox`)."""
    if sys.platform == "darwin":
        return os.path.exists(SANDBOX_EXEC)
    from lanternkb.extract import linux_sandbox

    return linux_sandbox.available()


# -- memory of a whole process group ----------------------------------------------------------


def family(pid: int) -> set[int]:
    """A worker's processes: its process group (it leads one) and all its descendants. A
    helper that left the group with setsid() is still a descendant."""
    if sys.platform == "darwin":
        lib = ctypes.CDLL(ctypes.util.find_library("proc"))
        buf = (ctypes.c_int * 4096)()

        def listed(fn: Any, arg: int) -> list[int]:
            n = fn(arg, ctypes.byref(buf), ctypes.sizeof(buf))  # a count of processes
            return [x for x in buf[: max(0, n)] if x]

        found = set(listed(lib.proc_listpgrppids, pid))
        todo = [pid]
        while todo:
            p = todo.pop()
            found.add(p)
            todo += [c for c in listed(lib.proc_listchildpids, p) if c not in found]
        return found
    parents: dict[int, int] = {}
    groups: dict[int, int] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as f:
                stat = f.read().rsplit(")", 1)[1].split()
            parents[int(entry)], groups[int(entry)] = int(stat[1]), int(stat[2])
        except (OSError, IndexError, ValueError):
            continue
    found = {p for p, g in groups.items() if g == pid} | {pid}
    grew = True
    while grew:
        more = {p for p, parent in parents.items() if parent in found} - found
        found |= more
        grew = bool(more)
    return found


def _sandbox_check() -> Any:
    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    fn = lib.sandbox_check
    fn.restype, fn.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    return fn


def processes_in(folder: str, sandboxed: bool) -> set[int]:
    """This user's processes whose working folder is `folder` or inside it: a worker's
    helpers, found by where they run rather than by who their parent is, so one that forked
    twice to leave the worker's family is still found. A worker's temp
    folder is private and new for each worker, so nothing else runs there; with `sandboxed`,
    macOS also requires the process to be sandboxed. A helper that changes folder is
    missed, which is the safe way to fail: nothing unrelated is ever killed."""
    folder = os.path.realpath(folder)
    inside = (folder, folder + os.sep)
    found = set()
    if sys.platform == "darwin":
        lib = ctypes.CDLL(ctypes.util.find_library("proc"))
        check = _sandbox_check() if sandboxed else None
        pids = (ctypes.c_int * 8192)()
        # PROC_UID_ONLY: this user's processes only. Returns bytes used.
        n = lib.proc_listpids(4, os.getuid(), ctypes.byref(pids), ctypes.sizeof(pids)) // 4
        info = ctypes.create_string_buffer(2352)  # struct proc_vnodepathinfo
        for pid in pids[: max(0, n)]:
            if pid <= 1 or pid == os.getpid():
                continue
            if lib.proc_pidinfo(pid, 9, 0, info, ctypes.sizeof(info)) != 2352:
                continue  # PROC_PIDVNODEPATHINFO: the working folder's path at offset 152
            # The kernel's path for the folder is already resolved.
            cwd = info.raw[152:1176].split(b"\0", 1)[0].decode(errors="replace")
            ours = cwd == inside[0] or cwd.startswith(inside[1])
            if ours and (check is None or check(pid, None, 0) == 1):
                found.add(pid)
        return found
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        try:
            if os.stat(f"/proc/{entry}").st_uid != os.getuid():
                continue
            cwd = os.path.realpath(os.readlink(f"/proc/{entry}/cwd"))
        except OSError:
            continue
        if cwd == inside[0] or cwd.startswith(inside[1]):
            found.add(int(entry))
    return found


def group_rss_mb(pid: int) -> float:
    """Resident memory of a worker's whole family (`family`), in MB."""
    total = 0
    if sys.platform == "darwin":
        lib = ctypes.CDLL(ctypes.util.find_library("proc"))
        info = (ctypes.c_uint64 * 12)()
        for p in family(pid):
            if lib.proc_pidinfo(p, 4, 0, ctypes.byref(info), ctypes.sizeof(info)) > 0:
                total += info[1]
        return total / 1e6
    page = os.sysconf("SC_PAGE_SIZE")
    for p in family(pid):
        try:
            with open(f"/proc/{p}/statm") as f:
                total += int(f.read().split()[1]) * page
        except (OSError, IndexError, ValueError):
            continue
    return total / 1e6


# -- one worker -------------------------------------------------------------------------------


def _exited_unreaped(pid: int) -> bool:
    """Whether a child has exited, without reaping it (waitid with WNOWAIT)."""
    if hasattr(os, "waitid"):
        try:
            return os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
        except ChildProcessError:
            return True
    # macOS: Python has no os.waitid there. P_PID 1; WEXITED 4, WNOHANG 1, WNOWAIT 0x20.
    libc = ctypes.CDLL(None, use_errno=True)
    info = ctypes.create_string_buffer(128)  # siginfo_t; si_pid is at offset 12
    if libc.waitid(1, pid, info, 4 | 1 | 0x20) != 0:
        return True  # not our child any more (already reaped)
    return int.from_bytes(info.raw[12:16], "little") != 0


class _Worker:
    def __init__(self, init: dict[str, Any], sandbox: bool, media: bool = False,
                 open_tier: bool = False):  # fmt: skip
        self.tmp = os.path.realpath(tempfile.mkdtemp(prefix=f"{TMP_PREFIX}{os.getpid()}-"))
        shared_gpu = media and not open_tier
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            # Media engines (Vision) find their model assets under the real home; the sandbox
            # still denies every write there. Other workers get a throwaway home.
            "HOME": pwd.getpwuid(os.getuid()).pw_dir if shared_gpu else self.tmp,
            "TMPDIR": self.tmp,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "LANG": "en_US.UTF-8",
        }
        for key in ("LANTERNKB_AUDIO_MODEL", "LANTERNKB_TEST_EXTRACTORS"):
            if key in os.environ:
                env[key] = os.environ[key]
        if media and not shared_gpu:
            env[CPU_ONLY_ENV] = "1"
        # Workers get a throwaway HOME, so resolve the models folder here, with the real one.
        from lanternkb.models import DIR_ENV, models_dir

        env[DIR_ENV] = str(models_dir())
        # -I: ignore PYTHON* variables and never put the current folder on sys.path.
        cmd = [sys.executable, "-I", "-m", "lanternkb.extract.worker"]
        self.sandboxed = sandbox and sandbox_available()
        linux = None
        if self.sandboxed and sys.platform == "darwin":
            cmd = [*_sandbox_command(self.tmp, media, open_tier), *cmd]
        elif self.sandboxed:  # Linux: the worker sandboxes itself before reading anything
            from lanternkb.models import models_dir

            tools = [os.path.realpath(t) for t in map(shutil.which, MEDIA_TOOLS) if t]
            reads, execs = [*_code_dirs()], [os.path.realpath(sys.executable)]
            if media:
                reads.append(os.path.realpath(models_dir()))
                execs += tools
            linux = {"tmp": self.tmp, "reads": reads, "execs": execs}
        # Files reach the worker as open descriptors over this socket (SCM_RIGHTS).
        self.files, child = socket.socketpair()
        # Worker stderr is dropped: a traceback could carry document text. For debugging
        # only, LANTERNKB_DEBUG_WORKER_STDERR names a file to append it to instead.
        debug = os.environ.get("LANTERNKB_DEBUG_WORKER_STDERR")
        err = open(debug, "ab") if debug else subprocess.DEVNULL  # noqa: SIM115
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=err, env=env, cwd=self.tmp,
                                         start_new_session=True,
                                         pass_fds=(child.fileno(),))  # fmt: skip
        finally:
            child_fd = child.fileno()
            child.close()
            if debug:
                err.close()  # type: ignore[union-attr]
        self.buf = b""
        self.base_mb = 0.0
        self._send({"init": {**init, "sandboxed": self.sandboxed, "files_fd": child_fd,
                             "linux_sandbox": linux}})  # fmt: skip
        ready = self._recv(time.monotonic() + 60, 1_000_000, None)
        if not isinstance(ready, dict) or not ready.get("ready"):
            self.close()
            raise RuntimeError("extraction worker failed to start")
        self.rules_hash = str(ready["rules_hash"])
        self.sandbox_error = str(ready.get("sandbox_error") or "")[:200]
        if linux is not None:  # only what the worker says it applied counts
            self.sandboxed = ready.get("sandboxed") is True
        self.base_mb = group_rss_mb(self.proc.pid)

    def _send(self, msg: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write((json.dumps(msg) + "\n").encode())
        self.proc.stdin.flush()

    def _recv(self, deadline: float, max_bytes: int, memory_mb: float | None) -> Any:
        """One JSON line, or a string naming why not: timeout, memory, closed, too-big, bad."""
        assert self.proc.stdout is not None
        fd = self.proc.stdout.fileno()
        while b"\n" not in self.buf:
            now = time.monotonic()
            if now >= deadline:
                return "timeout"
            if memory_mb is not None and group_rss_mb(self.proc.pid) > memory_mb:
                return "memory"
            ready, _, _ = select.select([fd], [], [], min(POLL_S, deadline - now))
            if not ready:
                continue
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                return "closed"
            self.buf += chunk
            if len(self.buf) > max_bytes:
                return "too-big"
        line, _, self.buf = self.buf.partition(b"\n")
        try:
            return json.loads(line)
        except ValueError:
            return "bad"

    def rescan(self, result: Result, rel: str, deadline_s: float) -> Result:
        """Scan another worker's (already redacted) output in this process, which has never
        parsed a file: a worker taken over by a hostile file could have skipped its own
        scan. A finding, a failure or a timeout refuses the file."""
        # The text as one, as the worker scanned it, so a secret split across two segments
        # is seen whole; then each location, and the note.
        texts = ["\n".join(seg["text"] for seg in result.segments)]
        texts += [seg["location"] for seg in result.segments]
        texts.append(result.note or "")
        breaks = part_breaks([seg["text"] for seg in result.segments],
                             [seg["location"] for seg in result.segments])  # fmt: skip
        try:
            self._send({"rescan": texts, "rel": rel, "breaks": list(breaks)})
        except OSError:
            return Result(result.id, skip="scan-timeout")
        reply = self._recv(time.monotonic() + deadline_s, 1000, None)
        if not isinstance(reply, dict) or not isinstance(reply.get("found"), int):
            self.kill()
            return Result(result.id, skip="scan-timeout")
        if reply["found"] < 0:
            return Result(result.id, skip="scan-timeout")
        return Result(result.id, skip="unredacted-output") if reply["found"] else result

    def run(self, job: Job) -> Result:
        try:
            fd = open_checked(job)
        except ExtractError as e:
            return Result(job.id, skip=e.reason)
        try:
            socket.send_fds(self.files, [b"f"], [fd])
            self._send(asdict(job))
        except OSError:
            return Result(job.id, skip="extract-crashed")
        finally:
            os.close(fd)
        lim = job.limits
        # Redacted text, as JSON, is at most a few times the text cap; anything bigger is wrong.
        max_bytes = 8 * min(lim["output_mb"] * 1_000_000, job.max_chars) + 1_000_000
        msg = self._recv(time.monotonic() + float(lim["cpu_seconds"]), max_bytes,
                         float(lim["memory_mb"]) + self.base_mb)  # fmt: skip
        if not isinstance(msg, dict):
            self.kill()
            reason = {"timeout": "extract-timeout", "memory": "extract-memory"}.get(str(msg))
            return Result(job.id, skip=reason or "extract-crashed")
        if set(msg) - RESULT_FIELDS or msg.get("id") != job.id or not _valid(msg):
            self.kill()
            return Result(job.id, skip="extract-crashed")
        for seg in msg.get("segments", []):  # stored, and scanned again, in this form
            seg["text"], seg["location"] = one_break(seg["text"]), one_break(seg["location"])
        return Result(**msg)

    @property
    def alive(self) -> bool:
        """Running, checked without reaping it: until it is reaped (even as a zombie) its pid,
        and so its group id, stay ours, and its group can be signalled safely."""
        return self.proc.returncode is None and not _exited_unreaped(self.proc.pid)

    def kill(self, helpers: set[int] | frozenset[int] = frozenset()) -> None:
        """Kill the worker and every helper it started: its process group and every
        descendant, including one that started its own session (and `helpers`, found
        earlier, while the worker still lived)."""
        # Not reaped yet (running, or a zombie): its pid, and so its group id, are still
        # ours. Once reaped they could be reused, so its group is then never signalled.
        unreaped = self.proc.returncode is None
        others = family(self.proc.pid) - {self.proc.pid} if unreaped else set()
        others |= set(helpers)
        if unreaped:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(self.proc.pid, signal.SIGKILL)
        for p in others:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(p, signal.SIGKILL)
        self.proc.wait()
        for _ in range(5):  # helpers no longer in its family, and anything they started
            late = processes_in(self.tmp, self.sandboxed)
            if not late:
                break
            for p in late:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(p, signal.SIGKILL)
            time.sleep(0.01)

    def close(self) -> None:
        # Killed, not asked to exit: an idle worker has nothing to finish, and this way its
        # helpers are found while it still holds its pid and group.
        self.kill()
        with contextlib.suppress(OSError):
            if self.proc.stdin:
                self.proc.stdin.close()
        self.files.close()
        shutil.rmtree(self.tmp, ignore_errors=True)


def _valid(msg: dict[str, Any]) -> bool:
    """Shape checks on a reply before anything from it is stored."""
    segments, findings = msg.get("segments", []), msg.get("findings", [])
    if not isinstance(segments, list) or not isinstance(findings, list):
        return False
    for seg in segments:
        if not (isinstance(seg, dict) and isinstance(seg.get("text"), str)
                and isinstance(seg.get("location"), str)):  # fmt: skip
            return False
    if not all(isinstance(msg.get(k), str | None) for k in ("skip", "note")):
        return False
    skip, note = msg.get("skip"), msg.get("note")
    if skip is not None and not REASON.fullmatch(skip.removeprefix("secret:")):
        return False  # free text could carry content past the scanner
    if note is not None and len(note) > MAX_NOTE:
        return False
    # Findings are stored in the audit log: a rule id, numbers and an action, nothing else
    #.
    for f in findings:
        if not (isinstance(f, dict) and set(f) <= {"rule", "line", "action", "segment"}
                and isinstance(f.get("rule"), str) and REASON.fullmatch(f["rule"])
                and f.get("action") in ("redact", "skip-file")
                and all(isinstance(f.get(k, 0), int) and not isinstance(f.get(k, 0), bool)
                        and f.get(k, 0) >= 0 for k in ("line", "segment"))):  # fmt: skip
            return False
    n = msg.get("n_secrets", 0)
    return isinstance(n, int) and not isinstance(n, bool) and n == len(findings)


# -- the pool ---------------------------------------------------------------------------------


class WorkerPool:
    """Runs jobs on N sandboxed workers and yields (job, result) as they finish."""

    def __init__(self, n: int, disabled_rules: tuple[str, ...] = (),
                 actions: dict[str, str] | None = None, sandbox: bool = True,
                 media: bool = False, rescan: Callable[[Job], bool] | None = None,
                 open_tier: bool = False, enabled_rules: tuple[str, ...] = ()):  # fmt: skip
        self.rescan = rescan  # which jobs' output a separate scanner process checks again
        self.open_tier = open_tier  # its output may reach an open client (`OPEN_MEDIA_PROFILE`)
        self.n = max(1, n)
        self.init = {"disabled_rules": list(disabled_rules), "actions": actions or {},
                     "enabled_rules": list(enabled_rules)}  # fmt: skip
        self.sandbox = sandbox
        self.media = media  # the media profile (images, audio); otherwise the strict one
        self.rules_hash: str | None = None
        self.sandboxed = False
        self.sandbox_error = ""  # why a Linux worker could not sandbox itself, if it could not

    def run(self, jobs: Iterable[Job]) -> Iterator[tuple[Job, Result]]:
        todo: queue.Queue[Job] = queue.Queue()
        done: queue.Queue[tuple[Job, Result] | BaseException] = queue.Queue()
        for job in jobs:
            todo.put(job)
        count = todo.qsize()
        if count == 0:
            return
        stop = threading.Event()
        remove_stale_worker_folders()
        first = self._start()  # learn the rules hash now
        self.rules_hash, self.sandboxed = first.rules_hash, first.sandboxed
        self.sandbox_error = first.sandbox_error

        def serve(worker: _Worker | None) -> None:
            w = s = None
            try:
                w = worker or self._start()
                while not stop.is_set():
                    try:
                        job = todo.get_nowait()
                    except queue.Empty:
                        break
                    if not w.alive:
                        w.close()
                        w = self._start()
                    try:
                        result = w.run(job)
                    except Exception:  # one job never takes the pool down
                        w.kill()
                        result = Result(job.id, skip="extract-crashed")
                    if (self.rescan is not None and (result.segments or result.note)
                            and self.rescan(job)):  # fmt: skip
                        if s is None or not s.alive:  # a scanner of its own, strict profile
                            s = _Worker(self.init, self.sandbox, False)
                        result = s.rescan(result, job.rel, float(job.limits["cpu_seconds"]))
                    done.put((job, result))
            except BaseException as e:  # a worker that cannot start at all
                done.put(e)
            finally:
                for x in (w, s):
                    if x is not None:
                        x.close()

        threads = [threading.Thread(target=serve, args=(first if i == 0 else None,), daemon=True)
                   for i in range(min(self.n, count))]  # fmt: skip
        for t in threads:
            t.start()
        try:
            for _ in range(count):
                item = done.get()
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop.set()  # the consumer stopped early or failed: no more jobs start
            for t in threads:
                t.join()

    def _start(self) -> _Worker:
        return _Worker(self.init, self.sandbox, self.media, self.open_tier)


def default_workers() -> int:
    return max(1, min(os.cpu_count() or 2, 8))
