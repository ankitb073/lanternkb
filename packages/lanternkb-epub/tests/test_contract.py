"""The example plugin checked against lanternkb's contract (docs/plugins.md, step 4)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import lanternkb_epub
from lanternkb_epub import create

from lanternkb.extract.base import Limits
from lanternkb.testing import check_extractor, check_manifest, run_in_sandbox

MANIFEST = Path(lanternkb_epub.__file__).parent / "lanternkb_feature.toml"


def book(chapters: list[str]) -> bytes:
    """A minimal EPUB: container, package document, one XHTML file per chapter."""
    items = "".join(f'<item id="c{i}" href="c{i}.xhtml" media-type="application/xhtml+xml"/>'
                    for i in range(len(chapters)))  # fmt: skip
    spine = "".join(f'<itemref idref="c{i}"/>' for i in range(len(chapters)))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml",
                   '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                   '<rootfiles><rootfile full-path="b.opf"/></rootfiles></container>')  # fmt: skip
        opf = (f'<package xmlns="http://www.idpf.org/2007/opf"><manifest>{items}</manifest>'
               f"<spine>{spine}</spine></package>")  # fmt: skip
        z.writestr("b.opf", opf)
        for i, text in enumerate(chapters):
            z.writestr(f"c{i}.xhtml", f"<html><body><p>{text}</p></body></html>")
    return out.getvalue()


SAMPLE = book(["The keeper counted the quince trees.", "Then the medlars."])


def test_manifest() -> None:
    check_manifest(MANIFEST)


def test_extractor_contract() -> None:
    check_extractor(create({}), {"sample.epub": SAMPLE})


def test_chapters_and_locations() -> None:
    r = create({}).extract(SAMPLE, "sample.epub", Limits())
    assert [s.location for s in r.segments] == ["chapter 1", "chapter 2"]


def test_text_limit_keeps_what_fits() -> None:
    r = create({}).extract(SAMPLE, "sample.epub", Limits(max_chars=40))
    assert len(r.segments) == 1 and r.note


def test_in_the_real_sandbox() -> None:
    results = run_in_sandbox("extractor.epub", {"sample.epub": SAMPLE})
    assert results["sample.epub"].skip is None
    assert "quince" in results["sample.epub"].segments[0]["text"]
