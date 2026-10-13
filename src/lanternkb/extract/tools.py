"""Running a media tool (ffmpeg, tesseract, whisper-cli) from a worker, with a time limit
that also holds in the Linux sandbox.

There a worker may signal only itself, so it cannot kill a tool that overruns. Each tool
therefore gets a CPU-time limit from the kernel, and on a timeout the worker stops waiting
for it and skips the file (`extract-timeout`). A tool stuck without using CPU lives on in
the worker's process group until the worker is closed, which kills the group.
"""

from __future__ import annotations

import contextlib
import resource
import subprocess

from lanternkb.extract.base import ExtractError


def run_tool(cmd: list[str], timeout: float, input: bytes | None = None
             ) -> subprocess.CompletedProcess[bytes]:  # fmt: skip
    """Run `cmd` (stdin `input`, output captured); ExtractError("extract-timeout") after
    `timeout` seconds of wall-clock time, or of the tool's CPU time."""
    cpu = int(timeout) + 1

    def limit() -> None:  # in the child, before it runs the tool
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))

    p = subprocess.Popen(cmd, stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         preexec_fn=limit)  # noqa: PLW1509 - workers are single-threaded  # fmt: skip
    try:
        out, err = p.communicate(input, timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            p.kill()
            p.wait(timeout=5)
        except (PermissionError, ProcessLookupError, subprocess.TimeoutExpired):
            pass  # the Linux sandbox: the CPU limit, or closing the worker, ends it
        for f in (p.stdin, p.stdout, p.stderr):
            if f is not None:
                with contextlib.suppress(OSError):
                    f.close()
        raise ExtractError("extract-timeout") from None
    return subprocess.CompletedProcess(cmd, p.returncode, out, err)
