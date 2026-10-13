# Write a lanternkb plugin

A plugin adds a **feature** to lanternkb without changing lanternkb: a new file type (an *extractor*), a different embedding model (an *embedder*), or a rule that keeps certain files local-only (a *tier rule*). It is an ordinary Python package. Installing it makes the feature appear, switched **off**, in `lanternkb features list`, `lanternkb doctor` and the settings page. lanternkb runs none of its code until the user turns it on.

The worked example is [`packages/lanternkb-epub`](../packages/lanternkb-epub), which indexes EPUB books. Copy it to start.

**Two words you will meet.** Every indexed file has a *tier*:
- `open`: assistants on the open-tier MCP server may see it. That can mean a cloud model.
- `local-only`: only local tools and the all-tier server see it.

Folders have a tier. Rules and secrets can only move a file towards `local-only`, never back.

## 1. The package layout

```
my-plugin/
  pyproject.toml
  src/lanternkb_mything/
    __init__.py               # create(options) and your implementation
    lanternkb_feature.toml    # the feature manifest (this exact name, next to __init__.py)
  tests/test_contract.py
```

`pyproject.toml`:

```toml
[project]
name = "lanternkb-mything"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []        # see "Dependencies" below

# This line is the whole registration: lanternkb finds the plugin through it.
[project.entry-points."lanternkb.features"]
mything = "lanternkb_mything:create"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/lanternkb_mything"]   # ships lanternkb_feature.toml with the code
```

Rules for the entry point:
- **The group** is always `lanternkb.features`.
- **The value** names your factory in a **top-level** package: `lanternkb_mything:create`, never `company.lanternkb_mything:create`. A dotted value is refused: finding it would import its parent package.
- **The manifest** sits next to that package's `__init__.py`, and must be inside the wheel. Hatchling includes every file in the package folder. With setuptools, add it to `package-data`. lanternkb reads it as a file, without importing your code. An extractor's code is then imported only inside the sandboxed extraction workers.

**Dependencies.** Install your plugin into lanternkb's own environment (step 5), where lanternkb is already present. You may list `lanternkb>=0.1` in `dependencies` (it is on PyPI since 0.1.0), but you need not. List only what your code needs beyond it, preferably nothing.

## 2. The manifest

```toml
[feature]
id = "extractor.mything"        # the kind, a dot, then 1-48 lower-case letters, digits or dashes
title = "My things"             # shown to the user
category = "documents"          # documents, code, images, audio, search or safety
kind = "extractor"              # extractor, embedder or tier_rule
api_version = 1
default_enabled = false         # plugins always start off, whatever this says
default_tier = "open"           # "local-only" makes everything this feature reads local-only
description = "What it reads, how, and anything slow or surprising. Shown before enabling."

[capabilities]                  # shown to the user before they enable it
reads = ["mything files inside approved roots"]
network = false                 # must be false: features never use the network
os_permissions = []

[limits]                        # enforced on extractors by the host, not trusted to you
cpu_seconds = 30                # at most 120
memory_mb = 512                 # at most 2048
output_mb = 20                  # at most 50

[handles]                       # extractors only: the file types you read
suffixes = [".thing"]
names = []                      # or exact file names
```

Limits above the maximums are lowered to them. `lanternkb doctor` shows the limits that apply.

lanternkb refuses a plugin, and `lanternkb doctor` says why, if any of these is true:
- the manifest has an unknown key, misses a required one, or is over 64 KB;
- the kind is not one of the three;
- the id breaks the form above, or a built-in feature has it;
- it handles a file type a built-in feature handles;
- it sets `required = true`.

**Two plugins that clash with each other**, by the same id or the same file type, are **both** refused. Which one wins must never depend on package names or install order.

## 3. The factory and the implementation

`create(options)` gets **only your feature's options** from the user's config, never anyone else's, and returns your implementation.

```toml
# the user's config.toml
[features."extractor.mything"]
enabled = true
options = { keep_headers = false }
```

Options can be text, numbers, booleans, lists and tables (no dates). lanternkb builds your implementation once per worker process, not once per file, so a factory may load a model.

### An extractor

It turns a file's **bytes** into text segments. Each segment carries a short location the user will see ("page 3", "chapter 2"), plus the line numbers its text covers.

```python
from collections.abc import Mapping
from typing import Any

from lanternkb.extract.base import API_VERSION, ExtractError, ExtractResult, Limits, Segment


def create(options: Mapping[str, Any]) -> "ThingExtractor":
    return ThingExtractor(keep_headers=bool(options.get("keep_headers", True)))


class ThingExtractor:
    api_version = API_VERSION
    id = "extractor.mything"            # the same id as the manifest

    def __init__(self, keep_headers: bool = True):
        self.keep_headers = keep_headers

    def extract(self, data: bytes, path: str, limits: Limits) -> ExtractResult:
        if not data.startswith(b"THING1"):
            raise ExtractError("thing-unreadable")      # a short code the user will see
        text = data[6:].decode("utf-8", "replace")
        if not text.strip():
            return ExtractResult(skip="thing-empty")
        if len(text) > limits.max_chars:
            raise ExtractError("too-large-text")
        return ExtractResult(segments=[Segment(text, "whole file", 1, text.count("\n") + 1)])
```

The types:
- **`Segment(text, location, start_line=1, end_line=1)`.** `start_line` and `end_line` are 1-based, counted within this segment's own text: a page's segment starts at line 1 of that page.
- **A `!` in a location names a sheet** (`Accounts!A1:D200`), as the built-in spreadsheet extractor writes it. Where the part before the `!` changes from one segment to the next, the secret scan ends any table it is reading, so the next sheet's columns are read from its own header. Use `!` only for that. Rows of one table spread over several segments keep the same sheet name, and the table goes on across them.
- **`Limits`.** lanternkb passes it to you. It holds `cpu_seconds`, `memory_mb`, `output_mb` and `max_chars`. `max_chars` is the most text this file may yield. If you can stop early, return what fits, with a note. Otherwise the file is refused as `too-large-text`. In your own tests, build one with `Limits()` or `Limits(max_chars=1000)`.

Rules for an extractor:
- **You get bytes, never a path to open.** `path` is the file's name, for messages only.
- **The sandbox:** your code runs in a separate worker process, with workers of its own. On macOS it is sandboxed by default-deny:
  - no network;
  - no reading other files (lanternkb opens the file and hands it over);
  - no writes;
  - no other processes, no system services.
  
  Its time and memory are limited by your manifest's `[limits]`. Going over a limit, crashing or hanging costs only that file, which is reported as skipped.
- **Files you cannot read:** raise `ExtractError("short-code")`, or return `ExtractResult(skip="short-code")`.
  - Codes are 1 to 64 characters: lower-case letters, digits, `.`, `_` and `-`. Anything else, or anything starting with `secret:`, is replaced by `extract-refused`.
  - Never put file content in a code.
  - Any other exception costs the file as `extract-failed`, a reason that helps nobody.
- **Partly read files:** return a `note` ("12 of 40 pages had no text"), so nothing is silently half-read. Notes are scanned for secrets like content and cut to 300 characters.
- **Archives:** use `lanternkb.extract.archive.SafeZip`.
  - It refuses zip bombs and XML with a DOCTYPE, and turns damaged members and malformed XML into `ExtractError`.
  - Read HTML with `SafeZip.read(name, allow_doctype=True)`, `decode_text` and `html.parser`.
  - Never parse untrusted XML that has a DOCTYPE.
- **What you don't need to do:** scan for secrets or decide tiers. lanternkb scans your text, redacts secrets, decides tiers and chunks the text after you return. It also scans your output again outside your process and refuses a file whose output still holds a secret.

### An embedder

It runs **in lanternkb's own process, with the user's privileges**. `features list`, `features enable`, the settings page and `doctor` tell the user so. It must provide:

```python
api_version = 1                 # lanternkb.embed.API_VERSION
key = "mymodel"                 # the feature id is "embedder.mymodel"
model_id = "my-org/my-model@rev3"   # change it whenever the vectors would change
dim = 768

def embed_documents(self, texts: list[str]) -> "numpy array, len(texts) x dim, unit length": ...
def embed_query(self, text: str) -> "numpy array of dim floats, unit length": ...
```

Models must already be on disk: an embedder must not download anything. If the user turns on your embedder, lanternkb uses it rather than the built-in one. A query vector of the wrong size makes search fall back to keywords.

### A tier rule

It also runs in-process, with the user's privileges. It can only make files stricter:

```python
from pathlib import PurePosixPath
from lanternkb.tiers import Tier

class NoPayslips:
    api_version = 1
    name = "no-payslips"
    needs_text = True           # False if you decide from the path and secret count alone

    def tier_for(self, rel_path: PurePosixPath, text: str, n_secrets: int) -> Tier | None:
        return Tier.LOCAL_ONLY if "payslip" in text.lower() else None
```

Return `Tier.LOCAL_ONLY` or `None`. Any other true value also counts as local-only: lanternkb fails closed.

Costs and timing of `needs_text = True`:
- lanternkb re-reads every file at every sync.
- When the user turns such a rule on or changes its options, every file is held local-only until the next sync has read it.
- Prefer path-based rules.

If your rule cannot be built (its factory raises), every file is local-only until it can, and `doctor` says why.

## 4. Test it against the contract

```python
# tests/test_contract.py
from pathlib import Path

from lanternkb.extract.base import Limits
from lanternkb.testing import check_extractor, check_manifest, run_in_sandbox
from lanternkb_mything import create

import lanternkb_mything

SAMPLE = b"THING1hello from a thing file\n"
MANIFEST = Path(lanternkb_mything.__file__).parent / "lanternkb_feature.toml"

def test_manifest():
    check_manifest(MANIFEST)

def test_extractor_contract():
    check_extractor(create({}), {"a.thing": SAMPLE})

def test_small_text_limit():
    assert create({}).extract(SAMPLE, "a.thing", Limits(max_chars=1000)).skip is None

def test_in_the_real_sandbox():          # needs the plugin installed or added (step 5)
    results = run_in_sandbox("extractor.mything", {"a.thing": SAMPLE})
    assert results["a.thing"].skip is None
    assert "hello" in results["a.thing"].segments[0]["text"]
```

What the checks cover:
- **`check_extractor`:**
  - the output's shape, determinism and the output limit;
  - empty, random, truncated (cut in half) and damaged (bytes changed in the middle) input, each of which must be refused with `ExtractError` or a skip.
- **`check_embedder(embedder, key)` and `check_tier_rule(rule)`** do the same for the other kinds.
- **`run_in_sandbox`:**
  - runs your installed plugin through lanternkb's real worker: the sandbox, your limits, secret scanning and redaction, with a default sync's size caps;
  - returns the worker's results, in which segments are **dicts** (`{"text", "location", "start_line", "end_line"}`), not `Segment` objects.

The example plugin's own tests are in [`packages/lanternkb-epub/tests`](../packages/lanternkb-epub/tests).

## 5. Install, enable, check

Work with a separate configuration and index while you develop, never your real ones. From a checkout of lanternkb, `uv run --with-editable` adds your plugin for one command without changing lanternkb's environment:

```sh
cd /path/to/lanternkb
export PLUGIN=/path/to/my-plugin DEV=/tmp/mything-dev
mkdir -p $DEV/files && cp some.thing $DEV/files/
cat > $DEV/config.toml <<END
index = "$DEV/index.db"                        # never your real index
[features."embedder.embeddinggemma-300m"]
enabled = false                                # keyword search only: no model needed
END
lk() { uv run --with-editable $PLUGIN lanternkb --config $DEV/config.toml "$@"; }
lk roots add $DEV/files --tier open            # nothing is indexed without a folder
lk doctor                     # lists your plugin: loaded (off), or refused with the reason
lk features enable extractor.mything
lk sync
lk search "hello"
uv run --with-editable $PLUGIN pytest $PLUGIN/tests    # the contract tests, step 4
```

`--config` goes before the command. In a checkout of lanternkb, the example plugin (`lanternkb-epub`) is already installed with the development tools, so it shows in `doctor` and `features list`; `--with-editable` matters for your own plugin. To install the plugin for good, run `uv pip install -e $PLUGIN` (or `pip install -e`) in lanternkb's environment; without `--config`, lanternkb uses `~/.config/lanternkb/config.toml`.

Feature controls:
- **Off** (`lanternkb features disable extractor.mything`) hides your feature's data at once. Nothing is deleted.
- **Purge** (`lanternkb features purge extractor.mything`) deletes it securely.
- **Uninstalling** a plugin: purge its feature first, then remove its table from config.toml.
  - Until the table is removed, search keeps working without that feature's data, but `lanternkb sync` stops and says why, so nothing is deleted by accident.
  - Data left in the index by an uninstalled plugin can still be purged by its id; `doctor` lists it.

## 6. Versioning

Interfaces carry an `api_version`. If a future lanternkb changes an interface, it raises the number and refuses implementations built for another one with a clear message (`plugin-api-mismatch` for an extractor). Your plugin is never half-loaded.

## 7. What lanternkb does and does not protect against

- **Installing any Python package trusts its author.** A package can run code when Python starts, through a `.pth` file, before lanternkb has any say. lanternkb's guarantee covers plugin code run through lanternkb: none until the user enables the feature.
- **An extractor plugin** runs sandboxed, on workers of its own, so it cannot read other files, the index or the config. It cannot see or alter the scanning of other features' files, and it cannot send anything anywhere. Its **own** files are only as safe as the plugin: it controls their text, so it could encode what it reads in a way no scanner recognises.
- **Embedder and tier-rule plugins** run with the user's privileges, and their declarations (`reads`, `network`) are their author's word, not enforced.
