"""Reranking: a cross-encoder scores the top files of a search, and its order is
blended with the search's own by reciprocal rank fusion.

Chosen on development data only (`bench/explore-rerank.py`, `bench/explore-rerank2.py`):
Alibaba's gte-reranker-modernbert-base (Apache-2.0), half precision, 512 tokens, the top
20 files, each scored on its best chunk with its path in front (as the embedder sees it),
then fused with the search's order at RRF constant 5. Pinned to a revision, loaded from the
local models folder only, with no code from the model repository. The model is fetched
once with `lanternkb models pull rerank:<key>`. A reranker that fails or hangs leaves the
search's own order.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from typing import Any, Protocol

TOP = 20  # files reranked
RRF_K = 5
TIMEOUT_S = 10.0
CPU_TIMEOUT_S = 30.0  # the ONNX export on the CPU: about 1 s warm, several cold


@dataclass(frozen=True)
class RerankerSpec:
    key: str
    repo: str
    revision: str
    licence: str
    max_tokens: int = 512
    half: bool = True  # float16 on the GPU (on dev, the same order as float32, twice as fast)
    onnx_graph: str | None = "onnx/model.onnx"  # fp32, in the same repository


RERANKERS: dict[str, RerankerSpec] = {
    s.key: s
    for s in (
        RerankerSpec("gte-modernbert", "Alibaba-NLP/gte-reranker-modernbert-base",
                     "f7481e6055501a30fb19d090657df9ec1f79ab2c", "Apache-2.0"),
    )
}  # fmt: skip


class Reranker(Protocol):
    api_version: int
    key: str

    def score(self, pairs: list[list[str]]) -> Any: ...  # one float per (query, text) pair


class RerankerUnavailable(RuntimeError):
    """The extra or the model is missing: searches keep their own order."""


class TorchReranker:
    """A built-in reranker run by sentence-transformers (the macOS `embed` extra)."""

    api_version = 1

    def __init__(self, spec: RerankerSpec):
        from lanternkb.models import model_path

        path = model_path(spec.repo)
        if not (path / "config.json").exists():
            raise RerankerUnavailable(f"model {spec.repo} not found in {path}; run "
                                      f"`lanternkb models pull rerank:{spec.key}`")  # fmt: skip
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RerankerUnavailable("install the `embed` extra for reranking") from e
        try:
            from transformers.utils import logging as hf_logging

            hf_logging.disable_progress_bar()  # no "Loading weights" bar on stderr
        except ImportError:
            pass
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = CrossEncoder(str(path), device=device, max_length=spec.max_tokens,
                                  trust_remote_code=False, local_files_only=True)  # fmt: skip
        if spec.half and device != "cpu":
            self.model.model.half()
        self.key = spec.key
        self.spec = spec

    def score(self, pairs: list[list[str]]) -> Any:
        return self.model.predict(pairs, batch_size=max(1, len(pairs)), show_progress_bar=False)


class OnnxReranker:
    """The same model's ONNX export (fp32), run by ONNX Runtime on the CPU: on Linux, where
    torch is not shipped, and for one-off terminal searches on macOS, where it loads
    in a fraction of torch's time. It reads the graph and `tokenizer.json` only."""

    api_version = 1

    def __init__(self, spec: RerankerSpec):
        from lanternkb.models import model_path

        path = model_path(spec.repo)
        graph = path / (spec.onnx_graph or "")
        if (
            spec.onnx_graph is None
            or not graph.is_file()
            or not (path / "tokenizer.json").is_file()
        ):
            raise RerankerUnavailable(f"model {spec.repo} (ONNX) not found in {path}; run "
                                      f"`lanternkb models pull rerank:{spec.key}`")  # fmt: skip
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            raise RerankerUnavailable("install the `embed` extra for reranking") from e
        import contextlib

        with contextlib.suppress(AttributeError):
            ort.disable_telemetry_events()  # Windows-only events; off on principle (rule 3)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # errors only
        self.session = ort.InferenceSession(str(graph), opts, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(spec.max_tokens, strategy="longest_first")
        pad = self.tokenizer.token_to_id("[PAD]")
        self.pad_id = pad if pad is not None else 0
        self.key = spec.key
        self.spec = spec

    def score(self, pairs: list[list[str]]) -> Any:
        import numpy as np

        if not pairs:
            return np.zeros(0, dtype=np.float32)
        enc = self.tokenizer.encode_batch([(q, t) for q, t in pairs])
        width = max((len(e.ids) for e in enc), default=1)
        ids = np.full((len(enc), width), self.pad_id, dtype=np.int64)
        mask = np.zeros((len(enc), width), dtype=np.int64)
        for i, e in enumerate(enc):
            ids[i, : len(e.ids)] = e.ids
            mask[i, : len(e.ids)] = 1
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        logits = self.session.run(None, feed)[0].reshape(len(enc), -1)[:, 0]
        return 0.5 * (1.0 + np.tanh(logits / 2.0))  # sigmoid, as sentence-transformers reports


def load_reranker(key: str, using: str | None = None) -> Reranker:
    """A built-in reranker, run by the installed backend (`embed.backend`) or `using`."""
    from lanternkb.embed import backend

    if key not in RERANKERS:
        raise RerankerUnavailable(f"unknown reranker {key!r}; known: {sorted(RERANKERS)}")
    if (using or backend()) == "onnx":
        return OnnxReranker(RERANKERS[key])
    return TorchReranker(RERANKERS[key])


def enabled_rerankers(registry: Any) -> list[str]:
    return [m.id.split(".", 1)[1] for m in registry.manifests.values()
            if m.kind == "reranker" and registry.state(m.id).enabled]  # fmt: skip


def availability(registry: Any) -> tuple[str | None, str | None]:
    """The reranker search would use, and why it cannot (None if it can), cheaply."""
    from lanternkb.embed import backend

    keys = [k for k in enabled_rerankers(registry) if k in RERANKERS]
    if not keys:
        return None, "off: no reranker feature is enabled"
    key = keys[0]
    using = backend()
    if using is None:
        return key, "unavailable: install the `embed` extra"
    explicit = registry.settings.get(f"reranker.{key}", {}).get("enabled") is True
    if using == "onnx" and not explicit:
        # On the CPU it takes 0.4 to 1 s a search on a fast machine, over the 150 ms budget
        #: on only when asked for.
        return key, (f"off on the CPU unless switched on (0.4 to 1 s a search): lanternkb "
                     f"features enable reranker.{key}")  # fmt: skip
    if not has_form(RERANKERS[key], using):
        return key, f"unavailable: model missing (lanternkb models pull rerank:{key})"
    return key, None


def has_form(spec: RerankerSpec, using: str) -> bool:
    """Whether the model is installed in the form a backend runs."""
    from lanternkb.models import is_present, model_path

    if using == "onnx":
        path = model_path(spec.repo)
        return bool(spec.onnx_graph) and (path / str(spec.onnx_graph)).is_file() and (
            path / "tokenizer.json").is_file()  # fmt: skip
    return is_present(spec.repo) and any(model_path(spec.repo).glob("*.safetensors"))


def quick(key: str) -> bool:
    """Whether a one-off search may rerank with the ONNX export where torch is the backend
   : ONNX Runtime and the export are installed."""
    from lanternkb.embed import backend

    if backend() != "torch" or key not in RERANKERS:
        return False
    if any(importlib.util.find_spec(m) is None for m in ("onnxruntime", "tokenizers")):
        return False
    return has_form(RERANKERS[key], "onnx")


class Lazy:
    """A reranker loaded on its first use, so a search that has nothing to rerank (or a
    command that never searches) does not pay for loading it. A model that
    cannot load (damaged, say) makes `score` raise, and the search keeps its own order."""

    api_version = 1

    def __init__(self, key: str, using: str | None = None):
        from lanternkb.embed import backend

        self.key = key
        self.using = using
        self.timeout_s = CPU_TIMEOUT_S if (using or backend()) == "onnx" else TIMEOUT_S
        self._model: Any = None
        self._failed: str | None = None

    def score(self, pairs: list[list[str]]) -> Any:
        if self._failed is not None:
            raise RerankerUnavailable(self._failed)
        if self._model is None:
            try:
                self._model = load_reranker(self.key, self.using)
            except Exception as e:  # noqa: BLE001 - any load failure: no reranking
                self._failed = f"{self.key} did not load ({type(e).__name__})"
                raise RerankerUnavailable(self._failed) from e
        return self._model.score(pairs)


def active_reranker(registry: Any, one_off: bool = False) -> tuple[Reranker | None, str]:
    """The reranker to use (loaded on first use), or None and why. For a `one_off` search
    (the terminal), the ONNX export when it is installed: it loads faster."""
    try:
        key, problem = availability(registry)
        if one_off and key is not None and quick(key):
            return Lazy(key, "onnx"), ""
    except Exception as e:  # noqa: BLE001 - searches keep their own order
        return None, f"reranking is unavailable: {type(e).__name__}"
    if key is None or problem:
        return None, problem or ""
    return Lazy(key), ""


def fused(n: int, scores: list[float], rrf_k: int = RRF_K) -> list[tuple[int, float | None]]:
    """The new order of `n` items (indexes into the search's order) with their fused scores:
    the search's ranks and the reranker's, by reciprocal rank fusion. Items beyond the
    scored ones keep their place after them, with no new score."""
    scored = len(scores)
    by_score = sorted(range(scored), key=lambda i: -scores[i])
    total = {i: 1.0 / (rrf_k + i + 1) for i in range(scored)}
    for rank, i in enumerate(by_score):
        total[i] += 1.0 / (rrf_k + rank + 1)
    order = sorted(range(scored), key=lambda i: -total[i])
    return [(i, total[i]) for i in order] + [(i, None) for i in range(scored, n)]


def blend(n: int, scores: list[float], rrf_k: int = RRF_K) -> list[int]:
    """`fused`, order only."""
    return [i for i, _ in fused(n, scores, rrf_k)]
