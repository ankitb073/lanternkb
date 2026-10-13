"""Extractors: file bytes to text segments. Each is the implementation of one feature."""

from __future__ import annotations

import os
from collections.abc import Callable

from lanternkb.extract.base import Extractor

# feature id -> factory. Imported lazily inside the worker so a missing optional engine
# (OCR, audio) only affects its own feature.
FACTORIES: dict[str, Callable[[], Extractor]] = {}


def _text(fid: str) -> Callable[[], Extractor]:
    def make() -> Extractor:
        from lanternkb.extract.text import TextExtractor

        return TextExtractor(fid)

    return make


FACTORIES["extractor.text"] = _text("extractor.text")
FACTORIES["extractor.code"] = _text("extractor.code")


def _images() -> Extractor:
    from lanternkb.extract.ocr import ImageExtractor

    return ImageExtractor()


def _audio() -> Extractor:
    from lanternkb.extract.audio import AudioExtractor

    return AudioExtractor()


FACTORIES["extractor.images"] = _images
FACTORIES["extractor.audio"] = _audio


def _pdf() -> Extractor:
    from lanternkb.extract.pdf import PdfExtractor

    return PdfExtractor()


def _office() -> Extractor:
    from lanternkb.extract.office import OfficeExtractor

    return OfficeExtractor()


def _odf() -> Extractor:
    from lanternkb.extract.odf import OpenDocumentExtractor

    return OpenDocumentExtractor()


FACTORIES["extractor.pdf"] = _pdf
FACTORIES["extractor.office"] = _office
FACTORIES["extractor.opendocument"] = _odf


if os.environ.get("LANTERNKB_TEST_EXTRACTORS") == "1":  # tests only: misbehaving extractors
    from lanternkb.extract.testing import TEST_EXTRACTORS

    for _fid, _cls in TEST_EXTRACTORS.items():
        FACTORIES[_fid] = lambda _c=_cls, _f=_fid: _c(_f)


def get(feature_id: str, options: dict[str, object] | None = None) -> Extractor:
    """The extractor for a feature: built in, or a plugin's (imported here, inside the
    sandboxed worker, and given only its own options)."""
    if feature_id in FACTORIES:
        return FACTORIES[feature_id]()
    from lanternkb import plugins

    return plugins.create(feature_id, options)  # type: ignore[no-any-return]
