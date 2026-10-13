"""Model storage and the one explicit download path.

`pull` is the ONLY code in lanternkb allowed to use the network (hard rule 3). It runs only
when the user asks for it; extraction never calls it and runs with the Hugging Face offline
switches set. Models live under ~/.local/share/lanternkb/models unless LANTERNKB_MODELS_DIR
says otherwise (tests point it at a temp folder).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from lanternkb.fsutil import private_dirs

DEFAULT_AUDIO_MODEL = "mlx-community/whisper-small.en-mlx"
MODEL_ENV = "LANTERNKB_AUDIO_MODEL"
DIR_ENV = "LANTERNKB_MODELS_DIR"
GGML_REPO = "ggerganov/whisper.cpp"
# Audio models are pinned like embedders: a repository change cannot bring new
# files into the models folder. Revisions checked on 2026-10-01.
AUDIO_REVISIONS = {
    DEFAULT_AUDIO_MODEL: "52a88bf6e98b114a210c21bb83e22d6e1505cb73",
    GGML_REPO: "5359861c739e955e79d9a303bcbc70fb988958b1",
}
# Weights, configs and tokenizers only, in any folder (the patterns match across "/"):
# never code or pickles.
ALLOW = ["*.json", "*.safetensors", "*.npz", "*.txt", "*.model", "*.tiktoken"]
# An ONNX export: its configs, tokenizer, one graph and that graph's weights.
ONNX_GRAPH = "onnx/model.onnx"
ALLOW_ONNX = ["*.json", ONNX_GRAPH, ONNX_GRAPH + "_data"]


def models_dir() -> Path:
    env = os.environ.get(DIR_ENV)
    return Path(env) if env else Path.home() / ".local/share/lanternkb/models"


def audio_model_name() -> str:
    return os.environ.get(MODEL_ENV) or DEFAULT_AUDIO_MODEL


def model_path(name: str) -> Path:
    """Where `name` lives locally: 'org/repo' becomes 'org--repo'; a 'ggml-*.bin' stays a file."""
    return models_dir() / name.replace("/", "--")


def _onnx_complete(p: Path) -> bool:
    """An ONNX export with its weights: beside the graph (`model.onnx_data`), or inside a
    graph big enough to hold them. A pull cut short leaves a small graph alone."""
    graph = p / ONNX_GRAPH
    if not graph.is_file():
        return False
    return (p / (ONNX_GRAPH + "_data")).is_file() or graph.stat().st_size > 50_000_000


def is_present(name: str) -> bool:
    p = model_path(name)
    if p.is_file():
        return p.stat().st_size > 0
    has_weights = p.is_dir() and ((p / "weights.npz").is_file() or any(p.glob("*.safetensors"))
                                  or _onnx_complete(p))  # fmt: skip
    return p.is_dir() and (p / "config.json").is_file() and has_weights


def status() -> dict:
    """What is installed: the models folder, the embedders, the audio model, and every
    model found."""
    from lanternkb.embed import EMBEDDERS, backend, source
    from lanternkb.rerank import RERANKERS, has_form

    d = models_dir()
    found = sorted(p.name for p in d.iterdir()) if d.is_dir() else []
    name = audio_model_name()
    return {
        "dir": str(d),
        "embedders": {
            k: {
                "model": src[0],
                "present": is_present(src[0]),
                "pull": f"lanternkb models pull embed:{k}",
            }
            for k, e in EMBEDDERS.items()
            if (src := source(e))
        },  # fmt: skip
        "rerankers": {
            k: {
                "model": r.repo,
                "present": has_form(r, backend() or "torch"),
                "forms": {f: has_form(r, f) for f in ("torch", "onnx")},
                "pull": f"lanternkb models pull rerank:{k}",
            }
            for k, r in RERANKERS.items()
        },  # fmt: skip
        "audio_model": name,
        "audio_model_present": is_present(name),
        "installed": found,
    }


def pull(name: str = DEFAULT_AUDIO_MODEL, force: bool = False, using: str | None = None) -> Path:
    """Download a model. The only network use in the project; says what and from where.

    `embed:<key>` fetches an embedder at its pinned revision (`lanternkb.embed.EMBEDDERS`),
    `rerank:<key>` a reranker at its pinned revision (`lanternkb.rerank.RERANKERS`),
    in the form the installed backend runs (torch, or an ONNX export; `using` picks one: on
    macOS the ONNX export as well makes one-off searches faster, #33); the default audio
    models are pinned too (`AUDIO_REVISIONS`). Only `ALLOW` (or `ALLOW_ONNX`) files are
    fetched. A model already present is not fetched again unless `force`.
    """
    revision = None
    allow = ALLOW
    if using is not None and not name.startswith(("embed:", "rerank:")):
        raise ValueError("--backend applies to embed: and rerank: models only")
    if name.startswith("embed:"):
        from lanternkb.embed import EMBEDDERS, backend, source

        key = name.split(":", 1)[1]
        if key not in EMBEDDERS:
            raise ValueError(f"unknown embedder {key!r}; known: {', '.join(sorted(EMBEDDERS))}")
        using = using or backend() or ("onnx" if sys.platform.startswith("linux") else "torch")
        src = source(EMBEDDERS[key], using)
        if src is None:
            raise ValueError(f"{key} has no ONNX export; use embed:embeddinggemma-300m")
        name, revision = src
        if using == "onnx":
            allow = ALLOW_ONNX
    elif name.startswith("rerank:"):
        from lanternkb.rerank import RERANKERS

        key = name.split(":", 1)[1]
        if key not in RERANKERS:
            raise ValueError(f"unknown reranker {key!r}; known: {', '.join(sorted(RERANKERS))}")
        from lanternkb.embed import backend
        from lanternkb.rerank import has_form

        spec = RERANKERS[key]
        name, revision = spec.repo, spec.revision
        form = using or backend() or ("onnx" if sys.platform.startswith("linux") else "torch")
        if form == "onnx":
            allow = ["*.json", str(spec.onnx_graph)]  # its graph holds its weights
        if has_form(spec, form) and not force:
            print(f"{name} ({form}) is already in {model_path(name)}; nothing downloaded")
            return model_path(name)
        force = True  # the other form may be there already: fetch this one beside it
    else:
        revision = AUDIO_REVISIONS.get(GGML_REPO if name.startswith("ggml-") else name)
    dest = model_path(name)
    if is_present(name) and not force:
        print(f"{name} is already in {dest}; nothing downloaded (--force fetches it again)")
        return dest
    private_dirs(dest.parent)
    from huggingface_hub import hf_hub_download, snapshot_download

    # The offline switches are for extraction; a pull must be allowed to reach the hub.
    saved = {k: os.environ.pop(k, None) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    try:
        at = f" at revision {revision}" if revision else " (not pinned: its current version)"
        if name.startswith("ggml-") and name.endswith(".bin"):
            print(f"Downloading {name}{at} from https://huggingface.co/{GGML_REPO} to {dest}")
            hf_hub_download(GGML_REPO, name, revision=revision, local_dir=dest.parent)
        else:
            print(f"Downloading {name}{at} from https://huggingface.co/{name} to {dest}")
            snapshot_download(name, revision=revision, local_dir=dest, allow_patterns=allow)
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    print(f"Done: {dest}")
    return dest
