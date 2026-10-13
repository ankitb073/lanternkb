"""Contract checks for plugin authors. Call them from your own tests.

    from lanternkb.testing import check_manifest, check_extractor, run_in_sandbox

    def test_contract():
        check_manifest("src/my_plugin/lanternkb_feature.toml")
        check_extractor(create({}), {"book.epub": SAMPLE_BYTES})
        results = run_in_sandbox("extractor.mything", {"a.thing": SAMPLE_BYTES})
        assert results["a.thing"].skip is None

Each check raises `ContractError` with a plain explanation when the plugin breaks a rule.
"""

from __future__ import annotations

import os
import random
import tempfile
import tomllib
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from lanternkb.features import Manifest, ManifestError, parse_manifest


class ContractError(AssertionError):
    """The plugin breaks a rule of the extension API."""


def check_manifest(path: str | Path) -> Manifest:
    """The manifest parses under the strict schema and fits the plugin rules."""
    from lanternkb.plugins import SUPPORTED_KINDS

    try:
        m = parse_manifest(tomllib.loads(Path(path).read_text()), str(path))
    except (ManifestError, tomllib.TOMLDecodeError) as e:
        raise ContractError(f"manifest: {e}") from e
    if m.kind not in SUPPORTED_KINDS:
        raise ContractError(f"manifest: kind {m.kind!r} cannot be a plugin; use one of "
                            f"{', '.join(SUPPORTED_KINDS)}")  # fmt: skip
    if not m.id.startswith(m.kind + "."):
        raise ContractError(f"manifest: the id must start with '{m.kind}.' (got {m.id!r})")
    if m.required:
        raise ContractError("manifest: a plugin cannot be required")
    return m


def check_extractor(extractor: Any, samples: Mapping[str, bytes],
                    output_mb: int = 20) -> None:  # fmt: skip
    """The extractor's output has the right shape, is deterministic, stays inside the
    output limit, and refuses empty, random, truncated and damaged input with an
    `ExtractError` (a reason the user can read), never another exception. (In a real sync
    another exception costs the file as `extract-failed`, a reason that helps nobody.)"""
    from lanternkb.extract.base import API_VERSION, ExtractError, ExtractResult, Limits, Segment

    if getattr(extractor, "api_version", None) != API_VERSION:
        raise ContractError(f"api_version must be {API_VERSION}")
    if not str(getattr(extractor, "id", "")).startswith("extractor."):
        raise ContractError("id must be the feature id, starting with 'extractor.'")
    limits = Limits(output_mb=output_mb)
    for name, data in samples.items():
        first = extractor.extract(data, name, limits)
        if not isinstance(first, ExtractResult):
            raise ContractError(f"{name}: extract() must return an ExtractResult")
        if first.skip is None and not first.segments:
            raise ContractError(f"{name}: no segments and no skip reason; return skip='...'")
        size = 0
        for s in first.segments:
            if not isinstance(s, Segment) or not isinstance(s.text, str):
                raise ContractError(f"{name}: segments must be Segment objects with text")
            if not s.location or len(s.location) > 120:
                raise ContractError(f"{name}: every segment needs a short location")
            size += len(s.text.encode())
        if size > output_mb * 1024 * 1024:
            raise ContractError(f"{name}: output of {size} bytes is over {output_mb} MB")
        if extractor.extract(data, name, limits) != first:
            raise ContractError(f"{name}: two runs on the same bytes gave different results")
    rng = random.Random(7)
    hostile = {"empty": b"", "random": bytes(rng.randrange(256) for _ in range(4096))}
    for name, data in samples.items():
        hostile[f"truncated {name}"] = data[: max(1, len(data) // 2)]
        middle = len(data) // 2  # damaged inside, valid at both ends (a bad checksum, say)
        hostile[f"damaged {name}"] = (data[:middle] + bytes(b ^ 0x5A for b in data[middle:
                                      middle + 64]) + data[middle + 64:])  # fmt: skip
    for label, data in hostile.items():
        try:
            r = extractor.extract(data, "input.bin", limits)
        except ExtractError:
            continue
        except Exception as e:  # noqa: BLE001 - the contract is exactly this
            raise ContractError(f"{label} input raised {type(e).__name__}; raise ExtractError("
                                "reason) for files you cannot read") from e  # fmt: skip
        if not isinstance(r, ExtractResult):
            raise ContractError(f"{label} input: extract() must return an ExtractResult")


def check_embedder(embedder: Any, key: str) -> None:
    """Vectors have the declared dimension, unit length, and are deterministic."""
    import numpy as np

    from lanternkb.embed import API_VERSION

    if getattr(embedder, "api_version", None) != API_VERSION:
        raise ContractError(f"api_version must be {API_VERSION}")
    if embedder.key != key or not isinstance(embedder.dim, int) or not embedder.model_id:
        raise ContractError("key must match the feature id's suffix; dim an int; model_id set")
    texts = ["a short note about plums", "def parse_config():\n    return 1", "東京の予算"]
    docs = np.asarray(embedder.embed_documents(texts), dtype=np.float32)
    if docs.shape != (len(texts), embedder.dim) or not np.isfinite(docs).all():
        raise ContractError(f"embed_documents must return {len(texts)} x {embedder.dim} floats")
    if not np.allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-3):
        raise ContractError("vectors must be normalised to unit length (cosine search)")
    q = np.asarray(embedder.embed_query("plums"), dtype=np.float32)
    if q.shape != (embedder.dim,) or not np.allclose(q, embedder.embed_query("plums"), atol=1e-5):
        raise ContractError("embed_query must return one deterministic vector of dim floats")


def check_tier_rule(rule: Any) -> None:
    """The rule only ever restricts: it returns LOCAL_ONLY or None, never OPEN."""
    from lanternkb.tiers import Tier

    if getattr(rule, "api_version", None) != 1 or not getattr(rule, "name", ""):
        raise ContractError("api_version must be 1, and the rule needs a name")
    if not isinstance(getattr(rule, "needs_text", None), bool):
        raise ContractError("needs_text must be True or False")
    for rel, text, n in (("a.md", "", 0), ("notes/b.txt", "hello", 0), ("x/y.md", "z", 3)):
        out = rule.tier_for(PurePosixPath(rel), text, n)
        if out is not None and not isinstance(out, Tier) or out == Tier.OPEN:
            raise ContractError(f"tier_for may only return Tier.LOCAL_ONLY or None, not {out!r} "
                                "(other values count as local-only)")  # fmt: skip


def run_in_sandbox(feature_id: str, files: Mapping[str, bytes],
                   options: Mapping[str, Any] | None = None) -> dict[str, Any]:  # fmt: skip
    """Run an installed extractor plugin through lanternkb's real extraction worker: the
    sandbox (no network, no writes), the limits from its manifest, secret scanning and
    redaction. Returns the worker's result per file name."""
    from lanternkb.config import Config
    from lanternkb.extract.pool import WorkerPool
    from lanternkb.extract.worker import Job
    from lanternkb.plugins import discover

    manifest = discover(refresh=True).manifests.get(feature_id)
    if manifest is None:
        raise ContractError(f"{feature_id} is not an installed plugin (see lanternkb doctor)")
    limits = {"cpu_seconds": manifest.limits.cpu_seconds, "memory_mb": manifest.limits.memory_mb,
              "output_mb": manifest.limits.output_mb}  # fmt: skip
    host = Config(roots=()).limits  # the caps a default sync applies, so results match it
    max_chars = min(host.max_text_chars, manifest.limits.output_mb * 1_000_000)
    with tempfile.TemporaryDirectory() as tmp:
        jobs, names = [], {}
        for i, (name, data) in enumerate(files.items()):
            p = Path(os.path.realpath(tmp)) / f"{i}-{Path(name).name}"  # resolved, as the policy
            p.write_bytes(data)
            st = os.stat(p)
            jobs.append(Job(i, str(p), name, feature_id, st.st_dev, st.st_ino,
                            host.max_file_bytes, limits, max_chars,
                            options=dict(options or {})))  # fmt: skip
            names[i] = name
        return {names[job.id]: result for job, result in WorkerPool(1).run(jobs)}
