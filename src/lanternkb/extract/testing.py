"""Misbehaving extractors, for proving that the worker's limits and sandbox hold.

Registered only when LANTERNKB_TEST_EXTRACTORS=1 is set (tests set it). Never used in a sync.
"""

from __future__ import annotations

import ctypes
import json
import os
import signal
import socket
import subprocess
import sys
import time

from lanternkb.extract.base import API_VERSION, ExtractResult, Limits, Segment


class _Base:
    api_version = API_VERSION

    def __init__(self, feature_id: str):
        self.id = feature_id


class Sleeper(_Base):
    def extract(self, data: bytes, path: str, limits: Limits) -> ExtractResult:
        time.sleep(3600)
        return ExtractResult()


class Crasher(_Base):
    def extract(self, data: bytes, path: str, limits: Limits) -> ExtractResult:
        os._exit(9)


class Hog(_Base):
    def extract(self, data: bytes, path: str, limits: Limits) -> ExtractResult:
        blocks = []
        while True:  # grow until the watchdog stops us
            blocks.append(bytearray(50_000_000))
            time.sleep(0.01)


class Writer(_Base):
    """Tries to write next to the file it was given; reports what happened."""

    def extract(self, data: bytes, path: str, limits: Limits) -> ExtractResult:
        target = data.decode().strip()
        try:
            with open(target, "w") as f:
                f.write("escaped")
            outcome = "write-allowed"
        except OSError as e:
            outcome = f"write-denied:{e.errno}"
        return ExtractResult([Segment(outcome, "lines 1-1")])


class Networker(_Base):
    def extract(self, data: bytes, path: str, limits: Limits) -> ExtractResult:
        try:
            socket.create_connection(("127.0.0.1", int(data.decode().strip())), timeout=2).close()
            outcome = "network-allowed"
        except OSError as e:
            outcome = f"network-denied:{type(e).__name__}"
        return ExtractResult([Segment(outcome, "lines 1-1")])


class Garbage(_Base):
    def extract(self, data: bytes, path: str, limits: Limits) -> ExtractResult:
        print("this is not json", flush=True)
        return ExtractResult([Segment("x", "lines 1-1")])


class Partial(_Base):
    def extract(self, data: bytes, path: str, limits: Limits) -> ExtractResult:
        sys.stdout.write('{"id": 0, "segments": [')
        sys.stdout.flush()
        time.sleep(3600)
        return ExtractResult()


class WrongId(_Base):
    def extract(self, data: bytes, path: str, limits: Limits) -> ExtractResult:
        print(json.dumps({"id": 999, "segments": [{"text": "forged", "location": "x"}]}),
              flush=True)  # fmt: skip
        return ExtractResult([Segment("x", "lines 1-1")])


class Spawner(_Base):
    """Starts a long-lived helper (like ffmpeg would) and then hangs."""

    def extract(self, data: bytes, path: str, limits: Limits) -> ExtractResult:
        try:
            seconds = int(data.decode().strip())
            subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({seconds})"])
        except OSError:
            return ExtractResult([Segment("spawn-denied", "lines 1-1")])
        time.sleep(3600)
        return ExtractResult()


class ChildHog(_Base):
    """A helper process that eats memory: counts against the job's limit."""

    def extract(self, data: bytes, path: str, limits: Limits) -> ExtractResult:
        subprocess.Popen([sys.executable, "-c",
                          "import time; x = bytearray(600_000_000); time.sleep(3600)"])  # fmt: skip
        time.sleep(3600)
        return ExtractResult()


class Prefs(_Base):
    def extract(self, data: bytes, path: str, limits: Limits) -> ExtractResult:
        try:
            r = subprocess.run(["/usr/bin/defaults", "write", data.decode().strip(), "k", "v"],
                               capture_output=True)  # fmt: skip
        except OSError:
            return ExtractResult([Segment("spawn-denied", "lines 1-1")])
        return ExtractResult([Segment(f"defaults-exit:{r.returncode}", "lines 1-1")])


class Signaller(_Base):
    def extract(self, data: bytes, path: str, limits: Limits) -> ExtractResult:
        try:
            os.kill(int(data.decode().strip()), signal.SIGTERM)
            outcome = "signal-allowed"
        except OSError as e:
            outcome = f"signal-denied:{e.errno}"
        return ExtractResult([Segment(outcome, "lines 1-1")])


class MachLookup(_Base):
    """Asks launchd for a system service by name (macOS)."""

    def extract(self, data: bytes, path: str, limits: Limits) -> ExtractResult:
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        port = ctypes.c_uint(0)
        bootstrap = ctypes.c_uint.in_dll(libc, "bootstrap_port")
        kr = libc.bootstrap_look_up(bootstrap, data.strip(), ctypes.byref(port))
        outcome = "mach-allowed" if kr == 0 else f"mach-denied:{kr}"
        return ExtractResult([Segment(outcome, "lines 1-1")])


class IOKitOpen(_Base):
    """Opens a connection to a kernel driver by class name (macOS)."""

    def extract(self, data: bytes, path: str, limits: Limits) -> ExtractResult:
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        iokit = ctypes.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
        iokit.IOServiceMatching.restype = ctypes.c_void_p
        iokit.IOServiceMatching.argtypes = [ctypes.c_char_p]
        iokit.IOServiceGetMatchingService.argtypes = [ctypes.c_uint, ctypes.c_void_p]
        iokit.IOServiceGetMatchingService.restype = ctypes.c_uint
        service = iokit.IOServiceGetMatchingService(0, iokit.IOServiceMatching(data.strip()))
        if not service:
            return ExtractResult([Segment("iokit-no-service", "lines 1-1")])
        conn = ctypes.c_uint(0)
        task = ctypes.c_uint.in_dll(libc, "mach_task_self_")
        kr = iokit.IOServiceOpen(service, task, 0, ctypes.byref(conn))
        outcome = "iokit-allowed" if kr == 0 else f"iokit-denied:{kr & 0xFFFFFFFF:#x}"
        return ExtractResult([Segment(outcome, "lines 1-1")])


class Reader(_Base):
    """Opens a file by path, as a hostile plugin would to copy it into its output."""

    def extract(self, data: bytes, path: str, limits: Limits) -> ExtractResult:
        try:
            with open(data.decode().strip(), "rb") as f:
                outcome = "read-allowed:" + f.read(40).decode("utf-8", "replace")
        except OSError as e:
            outcome = f"read-denied:{e.errno}"
        return ExtractResult([Segment(outcome, "lines 1-1")])


class Careless(_Base):
    """Says too much: quotes its input in skip reasons and notes (data: `skip:<text>`,
    `note:<text>`, `raise`, or `gen`)."""

    def extract(self, data: bytes, path: str, limits: Limits) -> ExtractResult:
        how, _, text = data.decode().partition(":")
        if how == "skip":
            return ExtractResult(skip=text)
        if how == "raise":
            raise ValueError(text)
        if how == "gen":
            return ExtractResult(Segment(t, "lines 1-1") for t in [text])  # type: ignore[arg-type]
        return ExtractResult([Segment("fine", "lines 1-1")], note=text)


class ProcArgs(_Base):
    """Reads another process's arguments and environment (data: its pid) through
    sysctl(KERN_PROCARGS2), as an exploited parser would to harvest tokens (macOS)."""

    def extract(self, data: bytes, path: str, limits: Limits) -> ExtractResult:
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        mib = (ctypes.c_int * 3)(1, 49, int(data.decode().strip()))  # CTL_KERN, KERN_PROCARGS2
        size = ctypes.c_size_t(1 << 20)
        buf = ctypes.create_string_buffer(size.value)
        rc = libc.sysctl(mib, 3, buf, ctypes.byref(size), None, ctypes.c_size_t(0))
        outcome = f"procargs-allowed:{size.value}" if rc == 0 else (
            f"procargs-denied:{ctypes.get_errno()}")  # fmt: skip
        return ExtractResult([Segment(outcome, "lines 1-1")])


class Escaper(_Base):
    """Starts a helper that leaves the worker's process group with setsid() (data: a
    marker number), then hangs."""

    def extract(self, data: bytes, path: str, limits: Limits) -> ExtractResult:
        marker = int(data.decode().strip())
        subprocess.Popen([sys.executable, "-c", "import os, time; os.setsid(); "
                          f"time.sleep({marker})"])  # fmt: skip
        time.sleep(3600)
        return ExtractResult()


class Leaky(_Base):
    """A parser taken over by its file: it switches its own worker's scanner off and
    returns the file's text, secrets included."""

    def extract(self, data: bytes, path: str, limits: Limits) -> ExtractResult:
        from lanternkb.scan import engine

        engine.Scanner.scan = lambda self, text, where="", _depth=0, **k: []  # type: ignore[method-assign]
        return ExtractResult([Segment(data.decode(), "lines 1-1")])


TEST_EXTRACTORS = {"test.sleep": Sleeper, "test.crash": Crasher, "test.hog": Hog,
                   "test.write": Writer, "test.net": Networker, "test.garbage": Garbage,
                   "test.partial": Partial, "test.wrongid": WrongId, "test.spawn": Spawner,
                   "test.childhog": ChildHog, "test.prefs": Prefs,
                   "test.signal": Signaller, "test.mach": MachLookup,
                   "test.iokit": IOKitOpen, "test.read": Reader,
                   "test.careless": Careless, "test.procargs": ProcArgs,
                   "test.escape": Escaper, "test.leaky": Leaky}  # fmt: skip


class Notes(_Base):
    """A parser taken over by its file, with its scanner off: `note:<text>` returns no
    segments and the text as its note; `split:<a>|<b>` returns two segments."""

    def extract(self, data: bytes, path: str, limits: Limits) -> ExtractResult:
        from lanternkb.scan import engine

        engine.Scanner.scan = lambda self, text, where="", _depth=0, **k: []  # type: ignore[method-assign]
        mode, _, rest = data.decode().partition(":")
        if mode == "note":
            return ExtractResult([], note=rest)
        a, _, b = rest.partition("|")
        return ExtractResult([Segment(a, "page 1"), Segment(b, "page 2")])


TEST_EXTRACTORS["test.notes"] = Notes


class CacheProbe(_Base):
    """Writes or reads a file in the GPU caches (data: `write <path> <text>` or `read
    <path>`), as two workers taken over by hostile files would to pass text between tiers
   ."""

    def extract(self, data: bytes, path: str, limits: Limits) -> ExtractResult:
        mode, target, *rest = data.decode().split(" ", 2)
        try:
            if mode == "write":
                with open(target, "w") as f:
                    f.write(rest[0] if rest else "")
                return ExtractResult([Segment("wrote", "lines 1-1")])
            with open(target) as f:
                return ExtractResult([Segment("read: " + f.read(), "lines 1-1")])
        except OSError as e:
            return ExtractResult([Segment(f"denied: {type(e).__name__}", "lines 1-1")])


class Helpers(_Base):
    """Leaves a helper behind (data: `<mode> <n>`; n is how long helpers sleep, a marker):
    `setsid` and `doublefork` return normally; `chain` leaves a helper with a
    child of its own; `group` leaves one in the worker's process group."""

    def extract(self, data: bytes, path: str, limits: Limits) -> ExtractResult:
        mode, marker = data.decode().split()
        sleeper = [sys.executable, "-c", f"import os, time; os.setsid(); time.sleep({marker})"]
        if mode == "setsid":
            subprocess.Popen(sleeper)
        elif mode == "group":  # stays in the worker's group
            subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({marker})"])
        elif mode == "chain":  # forks twice; the survivor keeps a child of its own
            code = ("import os, subprocess, sys, time; os.setsid(); subprocess.Popen("
                    f"[sys.executable, '-c', 'import time; time.sleep({marker})']); "
                    f"time.sleep({marker})")  # fmt: skip
            pid = os.fork()
            if pid == 0:
                if os.fork() == 0:
                    os.execv(sys.executable, [sys.executable, "-c", code])
                os._exit(0)
            os.waitpid(pid, 0)
        else:
            pid = os.fork()
            if pid == 0:
                if os.fork() == 0:
                    os.execv(sys.executable, sleeper)
                os._exit(0)
            os.waitpid(pid, 0)
        time.sleep(0.5)
        return ExtractResult([Segment("ok", "lines 1-1")])


TEST_EXTRACTORS["test.cache"] = CacheProbe
TEST_EXTRACTORS["test.helpers"] = Helpers
