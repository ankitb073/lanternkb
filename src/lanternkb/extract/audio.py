"""Audio: transcribe speech with MLX Whisper, else whisper.cpp, else skip.

Everything runs on this machine. Models must already be present (fetched by the user with
`lanternkb models pull`); extraction sets the Hugging Face offline switches and never downloads.
Decoding uses ffmpeg (found on PATH) on a private temporary copy of the file.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from lanternkb import models
from lanternkb.extract.base import (
    API_VERSION,
    CPU_ONLY_ENV,
    ExtractError,
    ExtractResult,
    Limits,
    Segment,
)
from lanternkb.extract.tools import run_tool

SAMPLE_RATE = 16000
TARGET_SECONDS = 30.0  # close a segment once it holds at least this much speech
MAX_SECONDS = 60.0  # never let one segment exceed this (unless a single utterance does)
SILENCE_PEAK = 0.005  # below this peak amplitude the clip is treated as silence
CPP_LINE = re.compile(r"\[(\d+):(\d+):(\d+)\.(\d+)\s*-->\s*(\d+):(\d+):(\d+)\.(\d+)\]\s*(.*)")


@dataclass(frozen=True)
class Timed:
    start: float
    end: float
    text: str


class _Unavailable(Exception):
    pass


def fmt_time(sec: float) -> str:
    s = int(sec)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def group_segments(
    items: list[Timed], target: float = TARGET_SECONDS, maximum: float = MAX_SECONDS
) -> list[Segment]:
    """Merge transcription pieces into segments of about 30 to 60 seconds. Deterministic."""
    out: list[Segment] = []
    cur: list[Timed] = []

    def flush() -> None:
        if not cur:
            return
        text = " ".join(t.text.strip() for t in cur if t.text.strip())
        if text:
            loc = f"{fmt_time(cur[0].start)}-{fmt_time(math.ceil(cur[-1].end))}"
            out.append(Segment(text, loc, 1, 1))
        cur.clear()

    for t in sorted(items, key=lambda t: (t.start, t.end)):
        if cur and t.end - cur[0].start > maximum:
            flush()
        cur.append(t)
        if cur[-1].end - cur[0].start >= target:
            flush()
    flush()
    return out


def _tmpdir() -> str:
    return tempfile.gettempdir()


def _decode(data: bytes, timeout: int, fmt: str):
    """Decode to 16 kHz mono with ffmpeg. fmt 'f32le' returns bytes of raw floats; 'wav' a path."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise _Unavailable
    with tempfile.TemporaryDirectory(dir=_tmpdir()) as d:
        src = Path(d) / "in.bin"
        src.write_bytes(data)
        if fmt == "wav":
            dst = Path(d) / "out.wav"
            cmd = [ffmpeg, "-nostdin", "-v", "error", "-i", str(src), "-ac", "1",
                   "-ar", str(SAMPLE_RATE), "-y", str(dst)]  # fmt: skip
        else:
            cmd = [ffmpeg, "-nostdin", "-v", "error", "-i", str(src), "-f", "f32le", "-ac", "1",
                   "-ar", str(SAMPLE_RATE), "pipe:1"]  # fmt: skip
        proc = run_tool(cmd, timeout)
        if proc.returncode != 0:
            raise ExtractError("audio-unreadable")
        if fmt == "wav":
            return dst.read_bytes()
        return proc.stdout


def _mlx(data: bytes, timeout: int) -> list[Timed] | None:
    """Returns pieces, [] for silence; raises _Unavailable if MLX or the model is missing."""
    try:
        import mlx_whisper
    except ImportError as e:
        raise _Unavailable from e
    name = models.audio_model_name()
    if name.startswith("ggml-") or not models.is_present(name):
        raise _Unavailable
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np

    raw = _decode(data, timeout, "f32le")
    audio = np.frombuffer(raw, dtype=np.float32)
    if audio.size < SAMPLE_RATE // 10:
        raise ExtractError("audio-unreadable")
    if float(np.max(np.abs(audio))) < SILENCE_PEAK:
        return []
    if os.environ.get(CPU_ONLY_ENV) == "1":  # no shared GPU caches in this worker
        import mlx.core as mx

        mx.set_default_device(mx.cpu)
    res = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=str(models.model_path(name)),
        verbose=None,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
    )
    out = []
    for s in res.get("segments", []):
        if s.get("no_speech_prob", 0.0) > 0.6 and s.get("avg_logprob", 0.0) < -1.0:
            continue
        out.append(Timed(float(s["start"]), float(s["end"]), str(s["text"])))
    return out


def parse_cpp(stdout: str) -> list[Timed]:
    out = []
    for line in stdout.splitlines():
        m = CPP_LINE.match(line.strip())
        if not m:
            continue
        g = m.groups()
        a = int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2]) + int(g[3]) / 10 ** len(g[3])
        b = int(g[4]) * 3600 + int(g[5]) * 60 + int(g[6]) + int(g[7]) / 10 ** len(g[7])
        if g[8].strip():
            out.append(Timed(a, b, g[8].strip()))
    return out


def _cpp_model() -> Path | None:
    name = os.environ.get(models.MODEL_ENV, "")
    cands = [name] if name.startswith("ggml-") else ["ggml-small.en.bin", "ggml-base.en.bin"]
    for c in cands:
        p = models.model_path(c)
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _whisper_cpp(data: bytes, timeout: int) -> list[Timed]:
    exe = shutil.which("whisper-cli")
    model = _cpp_model()
    if not exe or model is None or not shutil.which("ffmpeg"):
        raise _Unavailable
    wav = _decode(data, timeout, "wav")
    no_gpu = ["-ng"] if os.environ.get(CPU_ONLY_ENV) == "1" else []
    with tempfile.TemporaryDirectory(dir=_tmpdir()) as d:
        f = Path(d) / "a.wav"
        f.write_bytes(wav)
        proc = run_tool([exe, "-m", str(model), "-f", str(f), *no_gpu], timeout)
    if proc.returncode != 0:
        raise ExtractError("audio-unreadable")
    return parse_cpp(proc.stdout.decode("utf-8", errors="replace"))


def transcribe(data: bytes, timeout: int) -> list[Timed]:
    try:
        return _mlx(data, timeout) or []
    except _Unavailable:
        pass
    try:
        return _whisper_cpp(data, timeout)
    except _Unavailable:
        raise ExtractError("transcriber-unavailable") from None


class AudioExtractor:
    api_version = API_VERSION
    id = "extractor.audio"

    def extract(self, data: bytes, path: str, limits: Limits) -> ExtractResult:
        try:
            pieces = transcribe(data, limits.cpu_seconds)
        except ExtractError as e:
            if e.reason == "transcriber-unavailable":
                return ExtractResult(skip="transcriber-unavailable")
            raise
        segs = group_segments(pieces)
        if not segs:
            return ExtractResult(skip="no-speech-found")
        cap = limits.output_mb * 1024 * 1024
        kept: list[Segment] = []
        used = 0
        for s in segs:
            used += len(s.text.encode("utf-8"))
            if used > cap:
                return (
                    ExtractResult(kept, note="output truncated")
                    if kept
                    else ExtractResult(skip="too-large-text")
                )
            kept.append(s)
        return ExtractResult(kept)
