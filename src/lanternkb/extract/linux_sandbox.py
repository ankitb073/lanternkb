"""The Linux sandbox for extraction workers: Landlock for files, seccomp for the
rest. Neither needs root or a helper program.

A worker applies it to itself right after it starts, before it reads anything from a
file, and tells the parent whether it worked. Then it may:
- read the system's folders, Python's code folders and the models folder (media); write
  only its private temp folder (and /dev/null); run only its Python, or a media tool;
- not open a network or any other socket, connect to one, trace or read other processes,
  use the kernel's key rings, io_uring or BPF, mount, change namespaces, or signal any
  process but itself.
Descriptors it was handed (the file to read, its pipes) keep working.

Landlock needs Linux 5.13 or later with Landlock enabled; on 6.7 or later it also denies
TCP itself, and on 6.12 or later it keeps signals and abstract sockets inside the sandbox.
"""

from __future__ import annotations

import ctypes
import glob
import os
import platform
import struct
import sys
import threading
from typing import Any

# Syscall numbers: Landlock's are the same on every architecture.
_LANDLOCK_CREATE, _LANDLOCK_ADD_RULE, _LANDLOCK_RESTRICT = 444, 445, 446
_SECCOMP = {"x86_64": 317, "aarch64": 277}
_ARCH = {"x86_64": 0xC000003E, "aarch64": 0xC00000B7}  # AUDIT_ARCH_*

# Denied outright (each: x86_64, aarch64).
_DENIED = {
    "socket": (41, 198), "socketpair": (53, 199), "connect": (42, 203), "bind": (49, 200),
    "listen": (50, 201), "accept": (43, 202), "accept4": (288, 242),
    "ptrace": (101, 117), "process_vm_readv": (310, 270), "process_vm_writev": (311, 271),
    "kcmp": (312, 272), "pidfd_open": (434, 434), "pidfd_getfd": (438, 438),
    "pidfd_send_signal": (424, 424), "keyctl": (250, 219), "add_key": (248, 217),
    "request_key": (249, 218), "io_uring_setup": (425, 425), "io_uring_enter": (426, 426),
    "io_uring_register": (427, 427), "bpf": (321, 280), "perf_event_open": (298, 241),
    "userfaultfd": (323, 282), "mount": (165, 40), "umount2": (166, 39),
    "unshare": (272, 97), "setns": (308, 268), "tkill": (200, 130),
}  # fmt: skip
_KILL, _TGKILL = (62, 129), (234, 131)  # allowed only towards itself
_CLONE, _CLONE3 = (56, 220), (435, 435)
# clone flags for new namespaces (mount, cgroup, UTS, IPC, user, PID, network): refused, as
# unshare and setns are. clone3 passes its flags in memory, where seccomp cannot
# look, so it reports "not implemented" and the C library falls back to clone.
_NEW_NAMESPACES = 0x7E020000

# Landlock file-system rights (ABI 1), then those added by later ABIs.
_EXECUTE, _WRITE_FILE, _READ_FILE, _READ_DIR = 1 << 0, 1 << 1, 1 << 2, 1 << 3
_ABI1_ALL = (1 << 13) - 1
_REFER, _TRUNCATE, _IOCTL_DEV = 1 << 13, 1 << 14, 1 << 15
_READ = _READ_FILE | _READ_DIR
_NET_TCP = 0b11  # bind and connect (ABI 4)
_SCOPED = 0b11  # abstract unix sockets and signals (ABI 6)

_libc: Any = None


def _syscall() -> Any:
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(None, use_errno=True)
        _libc.syscall.restype = ctypes.c_long
        _libc.prctl.restype = ctypes.c_int
    return _libc


def abi() -> int:
    """Landlock's ABI version on this kernel, or 0 if Landlock is missing or off."""
    if not sys.platform.startswith("linux"):
        return 0
    v = _syscall().syscall(_LANDLOCK_CREATE, None, ctypes.c_size_t(0), ctypes.c_uint32(1))
    return max(0, int(v))


def available() -> bool:
    return abi() >= 1 and platform.machine() in _ARCH


def _fs_rights(version: int) -> int:
    rights = _ABI1_ALL
    if version >= 2:
        rights |= _REFER
    if version >= 3:
        rights |= _TRUNCATE
    if version >= 5:
        rights |= _IOCTL_DEV
    return rights


def _add_path(ruleset: int, path: str, rights: int, handled: int) -> None:
    """Allow `rights` beneath `path` (or on the file `path`). Missing paths are skipped."""
    try:
        fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
    except OSError:
        return
    try:
        if not os.path.isdir(path):  # files take file rights only
            rights &= _EXECUTE | _WRITE_FILE | _READ_FILE | _TRUNCATE | _IOCTL_DEV
        attr = struct.pack("=Qi", rights & handled, fd)  # landlock_path_beneath_attr, packed
        buf = ctypes.create_string_buffer(attr, len(attr))
        rc = _syscall().syscall(_LANDLOCK_ADD_RULE, ruleset, 1, buf, 0)
        if rc != 0:
            raise OSError(ctypes.get_errno(), f"landlock_add_rule failed for {path}")
    finally:
        os.close(fd)


def _landlock(reads: list[str], writes: list[str], execs: list[str]) -> int:
    version = abi()
    if version < 1:
        raise OSError("Landlock is not available")
    handled = _fs_rights(version)
    attr = struct.pack("=QQQ", handled, _NET_TCP if version >= 4 else 0,
                       _SCOPED if version >= 6 else 0)  # fmt: skip
    buf = ctypes.create_string_buffer(attr, len(attr))
    ruleset = _syscall().syscall(_LANDLOCK_CREATE, buf, ctypes.c_size_t(len(attr)), 0)
    if ruleset < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset failed")
    try:
        for p in reads:
            _add_path(ruleset, p, _READ, handled)
        for p in execs:
            _add_path(ruleset, p, _READ_FILE | _EXECUTE, handled)
        for p in writes:
            _add_path(ruleset, p, handled, handled)
        # No TCP port rules: with the network handled (ABI 4), every bind and connect fails.
        if _syscall().syscall(_LANDLOCK_RESTRICT, ruleset, 0) != 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")
    finally:
        os.close(ruleset)
    return version


# -- seccomp -----------------------------------------------------------------------------

_LD_W_ABS, _JEQ, _JGE, _JSET, _RET = 0x20, 0x15, 0x35, 0x45, 0x06
_RET_ALLOW, _RET_KILL, _RET_EPERM = 0x7FFF0000, 0x80000000, 0x00050000 | 1
_RET_ENOSYS = 0x00050000 | 38


def _filter(arch: str, pid: int) -> bytes:
    """A BPF program: right architecture or die; denied calls fail with EPERM; kill and
    tgkill only towards this process (or its group, which it leads)."""
    i = 0 if arch == "x86_64" else 1
    prog: list[tuple[int, int, int, int]] = []

    def op(code: int, k: int, jt: int = 0, jf: int = 0) -> None:
        prog.append((code, jt, jf, k))

    op(_LD_W_ABS, 4)  # seccomp_data.arch
    op(_JEQ, _ARCH[arch], 1, 0)
    op(_RET, _RET_KILL)
    op(_LD_W_ABS, 0)  # seccomp_data.nr
    if arch == "x86_64":
        op(_JGE, 0x40000000, 0, 1)  # the x32 numbering would get around the numbers below
        op(_RET, _RET_EPERM)
    for nums in _DENIED.values():
        op(_JEQ, nums[i], 0, 1)
        op(_RET, _RET_EPERM)
    op(_JEQ, _CLONE3[i], 0, 1)
    op(_RET, _RET_ENOSYS)
    op(_JEQ, _CLONE[i], 0, 4)
    op(_LD_W_ABS, 16)  # clone's flags, low half
    op(_JSET, _NEW_NAMESPACES, 0, 1)
    op(_RET, _RET_EPERM)
    op(_RET, _RET_ALLOW)
    me, group = pid & 0xFFFFFFFF, (-pid) & 0xFFFFFFFF
    for nums, targets in ((_KILL, (me, 0, group)), (_TGKILL, (me,))):
        op(_JEQ, nums[i], 0, 2 + 2 * len(targets))
        op(_LD_W_ABS, 16)  # args[0], low half (little-endian)
        for t in targets:
            op(_JEQ, t, 0, 1)
            op(_RET, _RET_ALLOW)
        op(_RET, _RET_EPERM)
    op(_RET, _RET_ALLOW)
    return b"".join(struct.pack("=HBBI", *x) for x in prog)


class _Fprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]


def _seccomp(pid: int) -> None:
    arch = platform.machine()
    if arch not in _ARCH:
        raise OSError(f"no seccomp filter for {arch}")
    code = _filter(arch, pid)
    buf = ctypes.create_string_buffer(code, len(code))
    prog = _Fprog(len(code) // 8, ctypes.cast(buf, ctypes.c_void_p).value)
    # SECCOMP_SET_MODE_FILTER with TSYNC: every thread, or none.
    rc = _syscall().syscall(_SECCOMP[arch], 1, 1, ctypes.byref(prog))
    if rc != 0:
        raise OSError(ctypes.get_errno(), "seccomp failed")


# What a worker needs from /etc: the loader's cache, time zone, user and group
# names, certificates, MIME types and fonts; not the rest (host names, other settings).
_ETC = ["/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/ld.so.conf.d", "/etc/localtime",
        "/etc/timezone", "/etc/nsswitch.conf", "/etc/passwd", "/etc/group", "/etc/os-release",
        "/etc/ssl", "/etc/ca-certificates", "/etc/mime.types", "/etc/fonts"]  # fmt: skip


def restrict(tmp: str, reads: list[str], execs: list[str]) -> str:
    """Sandbox this process (and everything it starts) for good. Returns a short summary;
    raises OSError if any part could not be applied, and then the caller must not read
    files."""
    if threading.active_count() != 1:
        raise OSError("sandbox must be applied before any thread starts")
    if _syscall().prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), "no_new_privs failed")
    system = ["/usr", "/lib", "/lib64", "/lib32", "/bin", "/sbin", *_ETC,
              "/dev/urandom", "/dev/random", "/dev/zero", "/sys/devices/system/cpu",
              f"/proc/{os.getpid()}"]  # fmt: skip
    loaders = [os.path.realpath(p) for pattern in ("/lib*/ld-linux*.so*", "/usr/lib*/ld-linux*",
                                                    "/lib/*/ld-linux*.so*")
               for p in glob.glob(pattern)]  # fmt: skip
    version = _landlock(system + reads, [tmp, "/dev/null"], execs + loaders)
    _seccomp(os.getpid())
    return f"landlock {version}, seccomp"
