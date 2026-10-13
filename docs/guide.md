# Using lanternkb

A practical guide: set it up, connect an assistant, search well and keep it safe. Design documents are linked at the end. From a checkout, put `uv run` before every `lanternkb` command.

## Set up in five minutes

```sh
uv tool install --managed-python --python 3.12 "lanternkb[embed,ui]"   # embed: semantic search; ui: the settings page
lanternkb setup           # asks for folders and tiers, offers the models, runs the first sync
```

From a checkout, `uv sync --extra embed` instead of the install.

`setup` asks for each folder, then whether a cloud assistant may see it.

| Folder | Example | Tier |
|---|---|---|
| Notes, docs and code you would paste into a chat | `~/Notes`, `~/code/myproject` | `open` |
| Anything you would not paste | finances, health, `~/Documents` | `local-only` (the default) |

It refuses your whole home folder and credential folders (`.ssh`, `.aws`, `Library`) unless you confirm. Nothing is downloaded without a yes. For scripts: `lanternkb setup --folder ~/Notes --tier open --yes --models`.

## Tiers: who sees what

- **`open`:** the MCP server started with `--tier open` returns these files. Use that server for cloud assistants such as Claude Code.
- **`local-only`:** only a server started with `--tier all` returns these. Use that only with a model running on this machine.
- **Secrets:** a file in which the scanner finds a secret becomes local-only by default, even inside an open folder. The secret itself is replaced by `[REDACTED:<rule>]` before anything is stored.
- **What tiers cover:** they limit what the MCP server returns. An assistant that can also run commands or read files could go around them. `lanternkb mcp-setup` prints Claude Code deny rules that close those doors ([threat model](threat-model.md)).

Change a folder's tier with `lanternkb roots tier <folder> open|local-only`. A folder made open stays local-only until the next sync reads it again.

## Connect Claude Code

```sh
lanternkb mcp-setup       # prints the registration command and suggested deny rules
claude mcp add --scope user lanternkb -- <path>/lanternkb --config <config> serve --tier open
```

The server starts when Claude Code needs it and stops with it: there is no daemon and no open port. Its tools are:
- `search`;
- `get_chunk`;
- `get_file_excerpt`;
- `status`;
- `sync`, which refreshes the index at most every 10 minutes.

Every result says it is untrusted data, so instructions inside your files are not followed as commands. In a test, Claude followed none of 129 planted instructions.

## Other MCP clients

lanternkb is a standard MCP server over stdio, so Codex, Cursor, VS Code, Gemini CLI, Windsurf, Claude Desktop and
local-model apps such as LM Studio can use it too. `lanternkb mcp-setup` prints the full path of the `lanternkb`
command; use it in place of `lanternkb` below, since an app started from the Dock may not see your shell's PATH.

Codex, in `~/.codex/config.toml`:

```toml
[mcp_servers.lanternkb]
command = "lanternkb"
args = ["serve", "--tier", "open"]
```

Cursor, Gemini CLI, Windsurf and Claude Desktop take the same entry as JSON in their MCP settings (VS Code's
`.vscode/mcp.json` names the key `servers`):

```json
{ "mcpServers": { "lanternkb": { "command": "lanternkb", "args": ["serve", "--tier", "open"] } } }
```

Give an assistant that sends results to the cloud the `open` server. The `all` server also returns local-only
files, so it refuses clients that name themselves as cloud assistants (Codex, Cursor, Copilot, Gemini and others):
use it only with a model that runs on this machine. The deny rules `mcp-setup` prints are for Claude Code; with
other tools, use their own permission settings to keep them out of your local-only folders.

## Search

```sh
lanternkb search "what did the plumber say about the boiler"
lanternkb search "parse_config_v2"                    # names and identifiers work too
lanternkb search "boiler quote" --type pdf,docx --since 30d
lanternkb search "budget" --path ~/Documents/2026 --before 2026-07-01
lanternkb recent --since 7d                           # what changed this week, newest first
```

- **Matching:** by meaning and by exact words together, then the top 20 files are reranked. Each result shows the passage where your words matched, with them in bold (`--json` for scripts).
- **Recent files:** `lanternkb recent` (or a search for `'*'`) lists files newest first, with the same filters. Assistants get it with the query `*`, and the settings page too.
- **Filters:** `--type` takes file extensions; `--since` and `--before` take a date (`YYYY-MM-DD`) or a span back (`30d`, `2w`, `6m`, `1y`). Assistants get the same filters (`types`, `modified_after`, `modified_before`).
- **Speed:** a terminal search loads the models each time, about 3 s. With the optional ONNX copies it takes about 0.5 s (1 to 2 s with reranking):

  ```sh
  lanternkb models pull embed:embeddinggemma-300m --backend onnx
  lanternkb models pull rerank:gte-modernbert --backend onnx
  ```

  The MCP server keeps its models loaded between searches, about 0.1 s each, and with the ONNX copy installed it holds about 1 GB less memory.

### In a browser

`lanternkb ui` opens the settings page, with a search box at the top. It searches every tier, as the terminal does, on this machine only (the page listens on 127.0.0.1 and closes with its tab). Each result shows the file, its folder and tier, the passage with your words marked, and a **Show in Finder** button (on Linux, **Show in folder**). Queries stay out of the page address, the browser's history and its autocomplete.

## What gets indexed

- **Text:** Markdown, plain text, reStructuredText, AsciiDoc, Org, LaTeX, BibTeX, logs, CSV/TSV, saved web pages (their readable text), rich text (`.rtf`, TextEdit's default) and subtitles (`.srt`, `.vtt`).
- **Documents:** PDF (with a text layer), Word, PowerPoint and Excel (`.docx`, `.pptx`, `.xlsx`), and OpenDocument (`.odt`, `.ods`, `.odp`).
- **Code and configuration:** about 70 languages and formats, Jupyter notebooks included.
- **Optional, off by default:** text in images (OCR) and audio transcription. Email and browser history are not included.

Every kind of file is a feature you can switch off: `lanternkb features list`, or the settings page (`lanternkb ui`).

## Keep it fresh

- `lanternkb sync`: unchanged files cost a `stat` each, and edited files re-embed only the changed passages.
- `lanternkb install-schedule`: prints a launchd (macOS) or systemd (Linux) schedule. It skips when the battery is low or the machine is hot.

## When something looks wrong

| Command | What it tells you |
|---|---|
| `lanternkb doctor` | checks the config, index, models, plugins and sandbox, and says what to fix |
| `lanternkb status` | counts by tier, skip reasons, last sync, whether semantic search is ready |
| `lanternkb skipped` | every file not indexed, and why |
| `lanternkb explain <path>` | whether a path would be indexed, with which tier, and why |
| `lanternkb --redact doctor` | the same, with your home folder and file names hidden, for a bug report |

## Limits worth knowing

- **Secret detection:** the scanner misses some secrets. On held-out tests through the full pipeline it kept 91 to 96% of planted secrets out of the index (95.7% on the latest). On the hardest split, raw text, it kept 71.9%. Keep credential folders out of indexed roots, or local-only.
- **Values that read as examples are not treated as secrets:** a value holding `example`, `changeme` or `placeholder`, a `<...>` placeholder or a `${VARIABLE}` reference is left alone, so documentation stays searchable. Do not keep a real password that looks like one.
- **Reranking on Linux:** it runs on the CPU there, 0.4 to 1 s a search, so it is off until you run `lanternkb features enable reranker.gte-modernbert`.
- **Scanned PDFs and images:** these need OCR (`extractor.images`, off by default).

## More

- [Plugins](plugins.md)
- [Threat model](threat-model.md)
