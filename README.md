# lanternkb

Private, local search over your own files, for you and the AI tools you choose.

lanternkb indexes the folders you pick and searches them by meaning and by exact words. It runs as a local MCP
server for Claude Code and other assistants, with no daemon, no open port and no telemetry. Secrets it detects
are redacted before anything is stored, and you decide which folders an assistant that sends results to the
cloud may see.

## Install

macOS 14 or later on Apple silicon, or Linux, with [uv](https://docs.astral.sh/uv/):

```sh
uv tool install --managed-python --python 3.12 "lanternkb[embed,ui]"
lanternkb setup        # asks for folders and whether an assistant may see them, offers the models, indexes
```

`--python 3.12` matters: without it uv may pick an older Python already on the machine, and the install fails.
The install takes about 0.9 GB on macOS, plus about 1.2 GB for the embedding model. `uv tool upgrade lanternkb`
updates it.

## Search

```sh
lanternkb search "what did the plumber say about the boiler"
lanternkb search "boiler quote" --type pdf,docx --since 30d
lanternkb recent       # files changed lately
lanternkb ui           # the settings page, with a search box
lanternkb doctor       # checks the setup and says what to fix
```

## Use it from Claude Code

1. `lanternkb mcp-setup` prints a `claude mcp add` command. Run it.
2. The server it registers returns files from folders you marked open, and nothing else.
3. Claude Code can also read files with its own tools, so `mcp-setup` prints deny rules too. Add them to
   `~/.claude/settings.json` to keep its tools away from your local-only folders and the index.

## What it reads

Notes, code, PDFs, Word and OpenDocument files, rich text, saved web pages and subtitles; scanned pages and audio
optionally. Plugins can add file types, embedding models and rules for which folder is open ([plugins](docs/plugins.md)).

## Privacy and safety

- Every folder is `open` or `local-only`, and new folders are local-only.
- Credential folders and secret-shaped files are skipped, and a secret scanner redacts what it finds before
  anything is split up or embedded. A file in which it finds a secret becomes local-only, even in an open folder.
- Files are read in sandboxed worker processes that deny by default: `sandbox-exec` on macOS, Landlock and
  seccomp on Linux.
- Nothing goes over the network except the model downloads you agree to.

The [threat model](docs/threat-model.md) says what is protected and what is not.

## How well it works

Tested on a synthetic set of files (three open-source code bases and 654 generated notes) with 330 held-out
questions: nDCG@10 0.831, against 0.777 for grooveseek with its reranker and 0.760 for QMD, at about 30 ms a
search (about 120 ms with reranking, on an M5 Max). On the hardest held-out test of planted fake secrets, the
scanner kept 71.9% out of the index, so treat it as a second line of defence: keep credential folders out of
indexed folders, or local-only.

## Limits

- The secret scanner misses some secrets, and values that read as examples (`example`, `changeme`, a
  `<placeholder>`, a `${VARIABLE}`) are left alone.
- On Linux, reranking runs on the CPU at 0.4 to 1 s a search, so it is off until you enable it.
- Email, browser history, calendars and contact cards are not indexed.
- Scanned PDFs and images need OCR, which is off by default.

## More

- [User guide](docs/guide.md)
- [Plugins](docs/plugins.md)
- [Threat model](docs/threat-model.md)
- [Security policy](SECURITY.md)

## Licence

Apache-2.0: see [LICENSE](LICENSE), [NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
