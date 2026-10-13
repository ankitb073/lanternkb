"""Embedders: turn redacted chunk text into vectors, locally.

Candidates are pinned to a revision, load only from the local models folder with the hub
offline, and never run code shipped inside a model repository (no `trust_remote_code`).
`lanternkb models pull embed:<key>` is the only way a model arrives (`lanternkb.models`).
The optional extra `embed` brings a backend: sentence-transformers and torch on macOS (they
use the GPU), ONNX Runtime on Linux (torch there pulls NVIDIA's non-permissive
CUDA libraries). Without it lanternkb works keyword-only.
"""

from __future__ import annotations

import contextlib
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Protocol

API_VERSION = 1


@dataclass(frozen=True)
class OnnxSource:
    """An ONNX export of a built-in model, run by ONNX Runtime where torch is not shipped
   . Only its graph, weights, configs and tokenizer file are fetched. The
    prompts are literal prefixes, the ones the model's sentence-transformers config names."""

    repo: str
    revision: str
    query_prefix: str
    doc_prefix: str
    graph: str = "onnx/model.onnx"  # its weights sit next to it, in model.onnx_data
    output: str = "sentence_embedding"  # pooled and projected, as sentence-transformers does
    # Inputs up to this many tokens (prompt included) give the torch vectors (cosine above
    # 0.9999, #12), so a query that fits may be compared with either backend's index.
    window: int = 257
    # the weights sit beside the graph (`model.onnx_data`); a pull cut short leaves the
    # graph without them
    external_data: bool = True


@dataclass(frozen=True)
class EmbedderSpec:
    key: str
    repo: str
    revision: str
    dim: int
    licence: str
    query_prompt: str | None = None  # a literal prefix for queries
    query_prompt_name: str | None = None  # a named prompt in the model's own config
    doc_prompt_name: str | None = None
    max_seq: int = 512
    onnx: OnnxSource | None = None

    @property
    def model_id(self) -> str:
        return f"{self.repo}@{self.revision}"


EMBEDDERS: dict[str, EmbedderSpec] = {
    s.key: s
    for s in (
        EmbedderSpec("bge-small", "BAAI/bge-small-en-v1.5",
                     "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a", 384, "MIT",
                     query_prompt="Represent this sentence for searching relevant passages: "),
        EmbedderSpec("qwen3-0.6b", "Qwen/Qwen3-Embedding-0.6B",
                     "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", 1024, "Apache-2.0",
                     query_prompt_name="query"),
        # Google's repository is gated; this is Unsloth's ungated copy under the same terms.
        EmbedderSpec("embeddinggemma-300m", "unsloth/embeddinggemma-300m",
                     "bfa3c846ac738e62aa61806ef9112d34acb1dc5a", 768, "Gemma terms",
                     query_prompt_name="query", doc_prompt_name="document",
                     # fp32: on 256 corpus chunks its vectors match the torch ones (mean
                     # cosine 0.9999); the int8 export drops to 0.992 and is no faster.
                     onnx=OnnxSource("onnx-community/embeddinggemma-300m-ONNX",
                                     "5090578d9565bb06545b4552f76e6bc2c93e4a66",
                                     query_prefix="task: search result | query: ",
                                     doc_prefix="title: none | text: ")),
        # BGE-M3 was a candidate, but at this revision its weights exist only as a pickle
        # (pytorch_model.bin), and loading a pickle can run code. Excluded on that ground.
    )
}  # fmt: skip


# What precedes a chunk's text when it is embedded: the file's "name" or its relative
# "path" inside the root. The name keeps copies of a file in different folders down to
# one embedding; the path gives the folders' meaning ("car/", "httpx/docs/") to the
# vector. Compared on dev.
INPUT = "path"  # chosen on dev in phase 5
INPUT_FORMAT = 1  # bump when embed_input's format changes: stored vectors are then redone


def embed_input(rel: str, text: str) -> str:
    """What is embedded for a chunk: its file's name or relative path, then its text."""
    head = rel if INPUT == "path" else PurePosixPath(rel).name
    return f"{head}\n{text}"


class Embedder(Protocol):
    api_version: int
    key: str
    model_id: str
    dim: int

    def embed_documents(self, texts: list[str]) -> Any: ...  # float32 array, rows normalised

    def embed_query(self, text: str) -> Any: ...


class EmbedderUnavailable(RuntimeError):
    """The optional extra or the model is missing; lanternkb runs keyword-only."""


BACKEND_ENV = "LANTERNKB_EMBED_BACKEND"
BACKENDS = ("torch", "onnx")


def backend() -> str | None:
    """What runs the built-in embedders: "torch" (sentence-transformers) if installed, else
    "onnx" (ONNX Runtime), else None. LANTERNKB_EMBED_BACKEND picks one when both are."""
    import importlib.util

    def has(*mods: str) -> bool:
        return all(importlib.util.find_spec(m) is not None for m in mods)

    have = {"torch": has("sentence_transformers"), "onnx": has("onnxruntime", "tokenizers")}
    want = os.environ.get(BACKEND_ENV)
    if want in have:
        return want if have[want] else None
    return next((b for b in BACKENDS if have[b]), None)


def source(spec: EmbedderSpec, using: str | None = None) -> tuple[str, str] | None:
    """The model repository and revision a backend runs for `spec` (default: the installed
    one), or None if that backend has no copy of this model."""
    using = using or backend() or "torch"
    if using == "onnx":
        return (spec.onnx.repo, spec.onnx.revision) if spec.onnx else None
    return spec.repo, spec.revision


def query_ids(spec: EmbedderSpec) -> frozenset[str]:
    """The index model ids a built-in embedder's query vectors may be compared with, for a
    query that fits `OnnxSource.window`: both backends' when the model has an ONNX export.
    Documents never mix (`index_model_id`)."""
    ids = {index_model_id(spec, "torch")}
    if spec.onnx is not None:
        ids.add(index_model_id(spec, "onnx"))
    return frozenset(ids)


def index_model_id(spec: EmbedderSpec, using: str | None = None) -> str:
    """What the index records for an embedder's vectors: the model (as the backend runs it)
    and how its inputs were made. Any change means a full re-embed, never a mix of two
    kinds of vector, so moving an index between torch and ONNX re-embeds it."""
    repo, revision = source(spec, using) or (spec.repo, spec.revision)
    return f"{repo}@{revision};input={INPUT}.{INPUT_FORMAT};max_seq={spec.max_seq}"


class _BuiltIn:
    """Shared by the built-in backends: normalised float32 rows, and a small query cache."""

    api_version = API_VERSION

    def __init__(self, spec: EmbedderSpec, using: str, cache_size: int) -> None:
        self.spec = spec
        self.key = spec.key
        self.model_id = index_model_id(spec, using)
        self.query_ids = query_ids(spec)
        self.dim = spec.dim
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._cache_size = cache_size

    def _encode(self, texts: list[str], query: bool) -> Any:
        raise NotImplementedError

    def _query_tokens(self, text: str) -> int:
        raise NotImplementedError

    def fits_window(self, query: str) -> bool:
        """Whether this query's vector matches the other backend's (`OnnxSource.window`)."""
        src = self.spec.onnx
        return src is not None and self._query_tokens(query) <= src.window

    def embed_documents(self, texts: list[str]) -> Any:
        return self._encode(texts, query=False)

    def embed_query(self, text: str) -> Any:
        if text in self._cache:
            self._cache.move_to_end(text)
            return self._cache[text]
        vec = self._encode([text], query=True)[0]
        self._cache[text] = vec
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return vec


class SentenceTransformerEmbedder(_BuiltIn):
    def __init__(self, spec: EmbedderSpec, batch_size: int = 32, cache_size: int = 512):
        from lanternkb.models import model_path

        path = model_path(spec.repo)
        if not (path / "config.json").exists():
            raise EmbedderUnavailable(
                f"model {spec.repo} not found in {path}; "
                f"run `lanternkb models pull embed:{spec.key}`"
            )
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # no "Loading weights" bars on stderr
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbedderUnavailable("install the `embed` extra for semantic search") from e
        try:
            from transformers.utils import logging as hf_logging

            hf_logging.disable_progress_bar()
        except ImportError:
            pass
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = SentenceTransformer(str(path), device=device, local_files_only=True,
                                         trust_remote_code=False)  # fmt: skip
        self.model.max_seq_length = spec.max_seq
        super().__init__(spec, "torch", cache_size)
        self.batch_size = batch_size

    def _encode(self, texts: list[str], query: bool) -> Any:
        import numpy as np

        kwargs: dict[str, Any] = {"normalize_embeddings": True, "batch_size": self.batch_size,
                                  "show_progress_bar": False, "convert_to_numpy": True}  # fmt: skip
        name = self.spec.query_prompt_name if query else self.spec.doc_prompt_name
        if name:
            kwargs["prompt_name"] = name
        elif query and self.spec.query_prompt:
            kwargs["prompt"] = self.spec.query_prompt
        return np.asarray(self.model.encode(texts, **kwargs), dtype=np.float32)

    def _query_tokens(self, text: str) -> int:
        name = self.spec.query_prompt_name
        prompt = (self.model.prompts.get(name, "") if name else self.spec.query_prompt) or ""
        return len(self.model.tokenizer(prompt + text)["input_ids"])


class OnnxEmbedder(_BuiltIn):
    """A built-in model's ONNX export, run by ONNX Runtime on the CPU. It reads
    the graph, its weights and `tokenizer.json`; nothing in the repository runs as code."""

    # Batches of 8: on the corpus, +0.4 GB over the model at about 50 chunks/s on an M5 Max
    # CPU, against +0.8 GB for batches of 16 at the same speed.
    def __init__(self, spec: EmbedderSpec, batch_size: int = 8, cache_size: int = 512):
        from lanternkb.models import model_path

        if spec.onnx is None:
            raise EmbedderUnavailable(f"{spec.key} has no ONNX export; on this system use "
                                      f"embeddinggemma-300m")  # fmt: skip
        path = model_path(spec.onnx.repo)
        graph = path / spec.onnx.graph
        weights = path / (spec.onnx.graph + "_data")
        if (not graph.is_file() or not (path / "tokenizer.json").is_file()
                or (spec.onnx.external_data and not weights.is_file())):  # fmt: skip
            raise EmbedderUnavailable(
                f"model {spec.onnx.repo} not found in {path}; "
                f"run `lanternkb models pull embed:{spec.key}`"
            )
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            raise EmbedderUnavailable("install the `embed` extra for semantic search") from e
        with contextlib.suppress(AttributeError):
            ort.disable_telemetry_events()  # Windows-only events; off on principle (rule 3)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # errors only
        self.session = ort.InferenceSession(str(graph), opts,
                                            providers=["CPUExecutionProvider"])  # fmt: skip
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(spec.max_seq)  # special tokens included, as in torch
        pad = self.tokenizer.token_to_id("<pad>")
        self.pad_id = pad if pad is not None else 0
        super().__init__(spec, "onnx", cache_size)
        self.batch_size = batch_size

    def _encode(self, texts: list[str], query: bool) -> Any:
        import numpy as np

        src = self.spec.onnx
        assert src is not None
        prefix = src.query_prefix if query else src.doc_prefix
        ids = [e.ids for e in self.tokenizer.encode_batch([prefix + t for t in texts])]
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))  # less padding per batch
        for start in range(0, len(order), self.batch_size):
            rows = order[start:][: self.batch_size]
            width = max(len(ids[i]) for i in rows)
            batch = np.full((len(rows), width), self.pad_id, dtype=np.int64)
            mask = np.zeros((len(rows), width), dtype=np.int64)
            for j, i in enumerate(rows):
                n = len(ids[i])
                batch[j, :n] = ids[i]
                mask[j, :n] = 1
            vecs = self.session.run([src.output], {"input_ids": batch, "attention_mask": mask})
            out[rows] = vecs[0]
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)

    def _query_tokens(self, text: str) -> int:
        assert self.spec.onnx is not None
        return len(self.tokenizer.encode(self.spec.onnx.query_prefix + text).ids)


def enabled_embedders(registry: Any) -> list[str]:
    """Keys of the enabled embedder features; the first is used. Plugins come first: they
    are off until the user turns one on, so an enabled plugin is a deliberate choice, while
    the built-in embedder is on by default."""
    on = {m.id.split(".", 1)[1] for m in registry.manifests.values()
          if m.kind == "embedder" and registry.state(m.id).enabled}  # fmt: skip
    return sorted(on - set(EMBEDDERS)) + [k for k in EMBEDDERS if k in on]


QUERY_TIMEOUT_S = 30.0  # one query embedding
BATCH_TIMEOUT_S = 900.0  # one batch of documents, generous for a CPU


def with_deadline(embedder: Any, method: str, arg: Any, timeout_s: float,
                  as_array: bool = False) -> Any:  # fmt: skip
    """Call an embedder method on a worker thread with a deadline. Embedders
    run in lanternkb's own process (plugins too), and a hung call could not be stopped: it
    is abandoned, the embedder is marked as hung, and every later call fails at once, so a
    search falls back to keywords and a sync's embedding pass stops (keyword search keeps
    working) until the embedder is loaded again."""
    if getattr(embedder, "_lanternkb_hung", False):
        raise TimeoutError("the embedder stopped answering earlier")
    box: dict[str, Any] = {}

    def call() -> None:
        try:
            value = getattr(embedder, method)(arg)
            if as_array:  # converted here, inside the deadline
                import numpy as np

                value = np.asarray(value, dtype=np.float32)
            box["value"] = value
        except BaseException as e:  # noqa: BLE001 - re-raised in the caller's thread
            box["error"] = e

    t = threading.Thread(target=call, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        with contextlib.suppress(AttributeError):
            embedder._lanternkb_hung = True
        raise TimeoutError(f"the embedder did not answer within {timeout_s:.0f} s")
    if "error" in box:
        e = box["error"]
        if not isinstance(e, Exception):  # SystemExit and the like: callers catch Exception,
            raise RuntimeError(f"the embedder raised {type(e).__name__}") from e
        raise e
    return box["value"]


def availability(registry: Any) -> tuple[str | None, str | None]:
    """The embedder search would use, and why it cannot (None if it can), without loading
    any model: cheap enough for status commands."""
    keys = enabled_embedders(registry)
    if not keys:
        return None, "off: no embedder feature is enabled"
    key = keys[0]
    if key in EMBEDDERS:
        problem = builtin_problem(EMBEDDERS[key])
        if problem:
            return key, f"unavailable: {problem}"
    return key, None


def builtin_problem(spec: EmbedderSpec) -> str | None:
    """Why a built-in embedder cannot load here (None if it can), without loading it."""
    from lanternkb.models import is_present

    using = backend()
    if using is None:
        return "install the `embed` extra"
    src = source(spec, using)
    if src is None:
        return f"{spec.key} has no ONNX export; use embeddinggemma-300m"
    if not is_present(src[0]):
        return f"model missing (lanternkb models pull embed:{spec.key})"
    return None


def active_embedder(registry: Any, query: str | None = None) -> tuple[Embedder | None, str]:
    """The embedder to use and, if there is none, why (search is then keyword-only). With a
    `query` (a one-off search), a built-in model's ONNX export when it is installed and the
    query fits its window: it loads in a fraction of torch's time."""
    keys = enabled_embedders(registry)
    if not keys:
        return None, "semantic search is off (no embedder feature is enabled)"
    note = f"several embedders are enabled; using {keys[0]}" if len(keys) > 1 else ""
    if query is not None and (fast := quick_query_embedder(keys[0], query)) is not None:
        return fast, note
    try:
        return load_embedder(keys[0], registry), note
    except EmbedderUnavailable as e:
        return None, f"semantic search is unavailable: {e}"


def quick_embedder(key: str) -> Embedder | None:
    """The ONNX export of a built-in model, for queries, where torch is the backend: None
    unless ONNX Runtime and the export are installed (`lanternkb models pull embed:<key>
    --backend onnx`). It loads in a fraction of torch's time, gives the same query vectors
    within the window (`OnnxSource.window`, #33), and takes less memory. It is marked
    `query_only`: documents are never embedded with it, so an index never mixes backends."""
    import importlib.util

    spec = EMBEDDERS.get(key)
    if spec is None or spec.onnx is None or backend() != "torch":
        return None
    if any(importlib.util.find_spec(m) is None for m in ("onnxruntime", "tokenizers")):
        return None
    try:
        emb = OnnxEmbedder(spec)
    except Exception:  # noqa: BLE001 - a damaged or half-downloaded export: torch instead
        return None
    emb.query_only = True
    return emb


def quick_query_embedder(key: str, query: str) -> Embedder | None:
    """`quick_embedder`, if this query fits the window where both backends agree."""
    emb = quick_embedder(key)
    return emb if emb is not None and emb.fits_window(query) else None


def embedder_for_index(key: str, model_id: str) -> Embedder | None:
    """The built-in embedder whose vectors an index holds (`model_id`), run by that backend
    if it is installed, else None: a long query cannot cross backends."""
    spec = EMBEDDERS.get(key)
    if spec is None:
        return None
    for using in BACKENDS:
        if index_model_id(spec, using) != model_id:
            continue
        try:
            return OnnxEmbedder(spec) if using == "onnx" else SentenceTransformerEmbedder(spec)
        except EmbedderUnavailable:
            return None
    return None


def load_embedder(key: str, registry: Any = None) -> Embedder:
    """A built-in embedder, or a plugin's (in-process, given only its own options)."""
    if key in EMBEDDERS:
        if backend() == "onnx":
            return OnnxEmbedder(EMBEDDERS[key])
        return SentenceTransformerEmbedder(EMBEDDERS[key])
    fid = f"embedder.{key}"
    if registry is not None and fid in registry.manifests:
        from lanternkb import plugins

        try:
            emb = plugins.create_with_deadline(fid, registry.options(fid))
        except Exception as e:  # noqa: BLE001 - keyword search still works (a hang too)
            raise EmbedderUnavailable(f"plugin {fid} did not load: {type(e).__name__}") from e
        if getattr(emb, "key", None) != key or not isinstance(getattr(emb, "dim", None), int):
            raise EmbedderUnavailable(f"plugin {fid} must report key {key!r} and an int dim")
        return emb  # type: ignore[no-any-return]
    raise EmbedderUnavailable(f"unknown embedder {key!r}; known: {sorted(EMBEDDERS)}")
