"""HTML for the settings page, generated from feature manifests (no per-feature UI code).

Every value that comes from the index, the config or a manifest is escaped: file names,
reasons and third-party manifest text are untrusted. The only script is inline, carries
the response's CSP nonce, and does five things: a heartbeat, a goodbye when the tab
closes, sync progress, a search's progress, and showing a result's file without leaving
the page.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from html import escape
from typing import Any
from urllib.parse import quote

from lanternkb.plugins import trust_note

CSS = """
:root { color-scheme: light dark; --bg:#fbfaf7; --card:#fff; --ink:#1d1d1f; --mute:#6e6e73;
  --line:#e5e3dd; --accent:#b4651e; --ok:#2e7d32; --err:#b3261e; --chip:#f1efe9; }
@media (prefers-color-scheme: dark) { :root { --bg:#151514; --card:#1f1f1d; --ink:#ecebe7;
  --mute:#a1a09b; --line:#33322f; --accent:#e39a50; --ok:#7bc47f; --err:#f28b82;
  --chip:#2a2927; } }
* { box-sizing: border-box; }
body { margin:0; font:15px/1.5 -apple-system, system-ui, sans-serif; background:var(--bg);
  color:var(--ink); }
main { max-width: 980px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 17px; margin: 28px 0 10px; }
.sub { color: var(--mute); margin: 0 0 16px; font-size: 13px; word-break: break-all; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
  padding: 14px 16px; margin: 10px 0; }
.row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
.grow { flex: 1 1 260px; min-width: 0; }
.chip { display: inline-block; background: var(--chip); border-radius: 999px;
  padding: 1px 9px; font-size: 12px; color: var(--mute); margin: 2px 4px 2px 0; }
.on { color: var(--ok); } .off { color: var(--mute); }
.chip.warn, small.warn { color: var(--err); }
.stat { display: inline-block; margin-right: 18px; } .stat b { font-size: 18px; }
.flash { padding: 10px 14px; border-radius: 10px; margin: 8px 0; }
.flash.ok { background: color-mix(in srgb, var(--ok) 14%, transparent); }
.flash.error { background: color-mix(in srgb, var(--err) 16%, transparent); }
button, select, input[type=text] { font: inherit; border-radius: 8px; border: 1px solid
  var(--line); background: var(--card); color: var(--ink); padding: 5px 11px; }
button { cursor: pointer; } button.primary { background: var(--accent); color: #fff;
  border-color: var(--accent); } button.danger { color: var(--err); }
form { display: inline; margin: 0; } input[type=text] { width: 100%; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
td, th { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line);
  vertical-align: top; word-break: break-all; }
th { color: var(--mute); font-weight: 600; } small, .mute { color: var(--mute); }
details summary { cursor: pointer; color: var(--mute); }
form.search { display: block; } form.search .row { flex-wrap: nowrap; gap: 8px; }
input[type=search] { font: inherit; border-radius: 8px; border: 1px solid var(--line);
  background: var(--card); color: var(--ink); padding: 5px 11px; flex: 1 1 auto;
  min-width: 0; }
form.search button { flex: none; white-space: nowrap; }
.filters { display: grid; grid-template-columns: repeat(auto-fill, minmax(118px, 1fr));
  gap: 8px; margin-top: 10px; max-width: 560px; }
.filters label { display: flex; flex-direction: column; gap: 2px; font-size: 12px;
  color: var(--mute); } .filters input[type=text] { font-size: 14px; padding: 4px 9px; }
p.snippet { margin: 6px 0 0; white-space: pre-wrap; overflow-wrap: anywhere; }
.sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden;
  clip: rect(0 0 0 0); white-space: nowrap; }
mark { background: color-mix(in srgb, var(--accent) 30%, transparent); color: inherit;
  border-radius: 3px; padding: 0 1px; } form.reveal button { font-size: 13px;
  padding: 2px 9px; }
a { color: var(--accent); } .controls { display: flex; gap: 6px; flex-wrap: wrap;
  align-items: center; } .card, .grow, small { overflow-wrap: anywhere; }
"""

SCRIPT = """
const pageId = crypto.randomUUID();
const body = () => new URLSearchParams({page: pageId});
const closed = () => {
  const b = document.getElementById('closed');
  if (b) b.hidden = false;
};
const beat = () => fetch('beat', {method: 'POST', body: body()})
  .then(r => { if (!r.ok) closed(); }).catch(closed);
beat(); setInterval(beat, 10000);
addEventListener('pagehide', () => navigator.sendBeacon('bye', body()));
const say = (b, text) => {
  b.textContent = text;
  const a = document.getElementById('announce');
  if (a) a.textContent = (b.getAttribute('aria-label') || '') + ': ' + text;
};
document.querySelectorAll('form.reveal').forEach(f => f.addEventListener('submit', e => {
  e.preventDefault();
  const b = f.querySelector('button');
  fetch('reveal', {method: 'POST', body: new URLSearchParams(new FormData(f))})
    .then(r => { say(b, r.ok ? 'Shown' : 'Not available'); })
    .catch(() => { say(b, 'Not available'); });
}));
const sf = document.querySelector('form.search');
if (sf) sf.addEventListener('submit', () => {
  sf.querySelector('button').textContent = 'Searching...';
});
const syncing = document.getElementById('syncing');
if (syncing) {
  const poll = async () => {
    try {
      const s = await (await fetch('sync/status')).json();
      if (!s.running) { location.reload(); return; }
      const stages = {checking: 'checking files', reading: 'reading changed files',
                      embedding: 'embedding passages'};
      const at = s.stage ? ': ' + stages[s.stage] + ' ' + s.done.toLocaleString() +
        (s.total ? ' of ' + s.total.toLocaleString() : '') : '';
      syncing.textContent = 'Sync running for ' +
        Math.round(Date.now() / 1000 - s.started) + ' s' + (at || '...');
    } catch (e) {}
    setTimeout(poll, 1000);
  };
  poll();
}
"""


def _e(value: Any) -> str:
    return escape(str(value), quote=True)


CATEGORY_ORDER = ("documents", "code", "images", "audio", "search", "safety")


def _n(count: int, word: str) -> str:
    return f"{count:,} {word}{'' if count == 1 else 's'}"


def _size(n: int) -> str:
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "bytes" else f"{n:,.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return str(n)


def _shell(title: str, body: str, nonce: str = "") -> str:
    style = f'<style nonce="{_e(nonce)}">{CSS}</style>' if nonce else ""
    script = f'<script nonce="{_e(nonce)}">{SCRIPT}</script>' if nonce else ""
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{_e(title)}</title>{style}</head><body><main>{body}</main>"
            f"{script}</body></html>")  # fmt: skip


def _form(action: str, csrf: str, fields: str, label: str, cls: str = "") -> str:
    return (f'<form method="post" action="{_e(action.lstrip("/"))}">'
            f'<input type="hidden" name="csrf" value="{_e(csrf)}">{fields}'
            f'<button class="{_e(cls)}" type="submit">{_e(label)}</button></form>')  # fmt: skip


def _hidden(name: str, value: str) -> str:
    return f'<input type="hidden" name="{_e(name)}" value="{_e(value)}">'


def _tier_select(current: str, what: str = "") -> str:
    """The tier choice; `what` names the folder or feature for screen readers, which
    otherwise hear a row of identical "tier" lists."""
    opts = "".join(f'<option value="{t}"{" selected" if t == current else ""}>{t}</option>'
                   for t in ("open", "local-only"))  # fmt: skip
    label = f"tier for {what}" if what else "tier"
    return f'<select name="tier" aria-label="{_e(label)}">{opts}</select>'


def _flash(flash: list[tuple[str, str]]) -> str:
    closed = ('<div id="closed" class="flash error" role="alert" hidden>This settings page has '
              "closed. Run <code>lanternkb ui</code> again to reopen it.</div>")  # fmt: skip
    return closed + "".join(f'<div class="flash {_e(k)}" role="status">{_e(m)}</div>'
                            for k, m in flash)  # fmt: skip


def _search_form(csrf: str, q: str = "", types: str = "", since: str = "",
                 before: str = "") -> str:  # fmt: skip
    """The search box. It is sent by POST and answered with a redirect to the results under
    a random handle (Post/Redirect/Get), so a query never enters the page's address, title,
    history or saved session (which may sync to a cloud account); autocomplete is off, which
    also keeps typed text out of the browser's saved form state."""

    def box(name: str, value: str, label: str, hint: str) -> str:
        return (f'<label>{_e(label)}<input type="text" name="{name}" value="{_e(value)}" '
                f'placeholder="{_e(hint)}" autocomplete="off" spellcheck="false">'
                "</label>")  # fmt: skip

    return ('<form method="post" action="search" class="card search" role="search">'
            f'<input type="hidden" name="csrf" value="{_e(csrf)}"><div class="row">'
            f'<input type="search" name="q" value="{_e(q)}" required maxlength="2000" '
            'aria-label="search your files" placeholder="Search your files, or * for the latest" '
            'autocomplete="off" spellcheck="false">'
            '<button class="primary" type="submit">Search</button></div>'
            '<div class="filters">' + box("types", types, "File types", "pdf, md")
            + box("since", since, "Changed since", "30d")
            + box("before", before, "Changed before", "2026-06-30") + "</div></form>")  # fmt: skip


def _home(path: str) -> str:
    home = os.path.expanduser("~").rstrip("/")
    return "~" + path[len(home) :] if home and path.startswith(home + "/") else path


def _marked(snippet: str, query: str) -> str:
    """A snippet, escaped, with the query's words marked where a word starts with them."""
    from lanternkb.search import match_spans

    text = re.sub(r"\n\s*\n", "\n", snippet.strip())
    out, k = [], 0
    for a, b in match_spans(text, query):
        out += [_e(text[k:a]), f"<mark>{_e(text[a:b])}</mark>"]
        k = b
    return "".join(out) + _e(text[k:])


def search(nonce: str, csrf: str, flash: list[tuple[str, str]], form: dict[str, str],
           results: list[dict[str, Any]], error: str, note: str, took_ms: int,
           reveal: str = "") -> str:  # fmt: skip
    """Search results. The title never carries the query (browsers keep titles in their
    history too). `reveal` labels a button that shows a result's file in the file manager."""
    box = _search_form(csrf, *(form.get(k, "") for k in ("q", "types", "since", "before")))
    parts = ['<h1>Search</h1><p class="sub"><a href="./">Back to settings</a></p>',
             _flash(flash), box]  # fmt: skip
    if error:
        parts.append(f'<div class="flash error" role="alert">{_e(error)}</div>')
    else:
        found = _n(len(results), "result") if results else "No results"
        parts.append(f'<p class="mute" role="status">{found} in {took_ms} ms'
                     f"{' · ' + _e(note) if note else ''}</p>")  # fmt: skip
    for r in results:
        folder, _, name = r["path"].rpartition("/")
        label = reveal.replace(" in ", f" {name} in ", 1)  # "Show notes.md in Finder"
        show = (f'<form method="post" action="reveal" class="reveal">{_hidden("csrf", csrf)}'
                f'{_hidden("id", r["id"])}{_hidden("path", r["path"])}'
                f'<button type="submit" aria-label="{_e(label)}">{_e(reveal)}</button></form>'
                if reveal else "")  # fmt: skip
        text = _marked(r["snippet"], form.get("q", ""))
        parts.append(
            f'<article class="card hit"><div class="row"><b class="grow">{_e(name)}</b>'
            f'<span class="chip">{_e(r["tier"])}</span>{show}</div>'
            f'<small>{_e(_home(folder))}/ · {_e(r["location"])}'
            f'{" · " + _e(r["modified"][:16].replace("T", " ")) if r.get("modified") else ""}'
            f'</small><p class="snippet">{text}</p></article>')  # fmt: skip
    parts.append('<p id="announce" class="sr-only" aria-live="polite"></p>')
    return _shell("lanternkb search", "".join(parts), nonce)


def main(nonce: str, csrf: str, flash: list[tuple[str, str]], data: dict[str, Any],
         sync: dict[str, Any]) -> str:  # fmt: skip
    st = data["status"]
    parts = [f"<h1>lanternkb settings</h1><p class=\"sub\">{_e(data['config'])}</p>",
             _flash(flash), _search_form(csrf)]  # fmt: skip
    if data.get("error"):
        parts.append(f'<div class="flash error" role="alert">The index or config could not be '
                     f"read: {_e(data['error'])}. Run <code>lanternkb doctor</code>."
                     "</div>")  # fmt: skip
    # status and sync
    by_tier = st.get("by_tier", {}) if isinstance(st, dict) else {}
    stats = "".join(f'<span class="stat"><b>{_e(v)}</b> {_e(k)}</span>'
                    for k, v in (("files indexed", st.get("indexed", 0)),
                                 ("open", by_tier.get("open", 0)),
                                 ("local-only", by_tier.get("local-only", 0)),
                                 ("chunks", st.get("chunks", 0)),
                                 ("skipped", sum(st.get("skipped", {}).values())))
                    )  # fmt: skip
    emb = st.get("embeddings", {})
    emb_parts = []
    for key, info in emb.items() if isinstance(emb, dict) else []:
        if isinstance(info, dict):
            done = "complete" if info.get("complete") else f"{info.get('pending', 0):,} pending"
            emb_parts.append(f"{_e(key)}: {_e(done)}")
    emb_line = "; ".join(emb_parts)
    if sync.get("running"):
        sync_ui = '<span id="syncing" class="mute">Sync running...</span>'
    else:
        sync_ui = _form("/sync", csrf, "", "Sync now", "primary")
        result = sync.get("result")
        if result and result.get("ok"):
            sync_ui += (f' <span class="mute">Last: {result["indexed"]} indexed, '
                        f'{result["unchanged"]} unchanged, {result["removed"]} removed, '
                        f'{result["errors"]} errors; {result["pending"]} chunks waiting for '
                        f"embeddings.</span>")  # fmt: skip
        elif result:
            sync_ui += f' <span class="flash error">Sync failed: {_e(result.get("error"))}</span>'
    parts.append(
        f'<div class="card"><div class="row"><div class="grow">{stats}</div>'
        f"<div>{sync_ui}</div></div>"
        f'<small>Last sync: {_e(st.get("last_sync") or "never")}'
        f"{' · semantic search: ' + emb_line if emb_line else ''}"
        f"{' · old pages wait in the WAL until the next sync' if st.get('wal_pending') else ''}"
        f"</small></div>"
    )  # fmt: skip
    # features, grouped by category (generated from manifests)
    groups: dict[str, list[Any]] = defaultdict(list)
    for f in data["features"]:
        groups[f.manifest.category].append(f)
    parts.append("<h2>Features</h2>")
    order = sorted(groups, key=lambda c: (CATEGORY_ORDER.index(c) if c in CATEGORY_ORDER
                                          else len(CATEGORY_ORDER), c))  # fmt: skip
    for category in order:
        parts.append(f'<h3 class="mute">{_e(category.title())}</h3>')
        for f in groups[category]:
            parts.append(_feature_card(f, csrf))
    # roots
    parts.append("<h2>Folders</h2>")
    for r in data["roots"]:
        path = str(r.path)
        parts.append(
            f'<div class="card"><div class="row"><div class="grow"><b>{_e(path)}</b></div>'
            + _form("/root/tier", csrf, _hidden("path", path) + _tier_select(r.tier.value, path),
                    "Set tier")
            + "</div><details><summary>Remove this folder</summary>"
            + _form("/root/remove", csrf, _hidden("path", path)
                    + '<label><input type="checkbox" name="confirm" value="yes" required> '
                      "Delete everything indexed from this folder</label> "
                    + '<label><input type="checkbox" name="force" value="yes"> even if files '
                      "become more visible (it sits inside an open folder)</label> ",
                    "Remove", "danger")
            + "</details></div>"
        )  # fmt: skip
    parts.append(
        '<div class="card"><div class="row">'
        + _form("/root/add", csrf,
                '<span class="grow"><input type="text" name="path" required autocomplete="off" '
                'placeholder="/full/path/to/a/folder or ~/folder" '
                'aria-label="folder path"></span> '
                + _tier_select("local-only", "the new folder") + " ", "Add folder")
        + '</div><small>New folders are local-only unless you choose open. Open files can '
          "reach cloud assistants.</small></div>"
    )  # fmt: skip
    # skipped files and notes
    skipped = data["skipped"]
    rows = "".join(f"<tr><td>{_e(p)}</td><td>{_e(r)}</td></tr>" for p, r in skipped[:300])
    more = f'<p class="mute">and {len(skipped) - 300} more</p>' if len(skipped) > 300 else ""
    table = (f'<div class="card"><table><tr><th>File</th><th>Why</th></tr>{rows}</table>'
             f"{more}</div>")  # fmt: skip
    parts.append(f"<h2>Skipped files ({len(skipped)})</h2>"
                 + (table if skipped else '<p class="mute">None.</p>'))  # fmt: skip
    if data["notes"]:
        rows = "".join(f"<tr><td>{_e(p)}</td><td>{_e(n)}</td></tr>" for p, n in data["notes"][:300])
        parts.append(f'<h2>Partly read files</h2><div class="card"><table>{rows}</table></div>')
    return _shell("lanternkb settings", "".join(parts), nonce)


def _feature_card(f: Any, csrf: str) -> str:
    m = f.manifest
    caps = "".join(f'<span class="chip">reads {_e(r)}</span>' for r in m.reads)
    caps += "".join(f'<span class="chip">needs {_e(p)}</span>' for p in m.os_permissions)
    note = trust_note(m.id)
    in_process = note is not None and m.kind != "extractor"
    if in_process:  # its declarations are not enforced: do not show them as if they were
        caps = '<span class="chip warn">runs with your privileges</span>'
    else:
        caps += '<span class="chip">no network</span>'
    if note:
        caps += f'<br><small class="{"warn" if in_process else "mute"}">This is {_e(note)}</small>'
    state = '<b class="on">On</b>' if f.enabled else '<b class="off">Off</b>'
    pending = "" if f.applied else ' <span class="chip">applies at the next sync</span>'
    if m.kind == "embedder":  # its data is vectors; their tier is each file's own
        counts = f"{_n(f.chunks, 'vector')}, {_size(f.bytes)}"
    elif m.kind == "extractor":
        counts = (f"{_n(f.files, 'file')}, {_n(f.chunks, 'chunk')}, {_size(f.bytes)}"
                  + (f", {f.skipped:,} skipped" if f.skipped else ""))  # fmt: skip
    else:
        counts = ""
    if m.required:
        controls = '<span class="chip">required</span>'
    elif m.kind == "embedder":
        action = "/feature/disable" if f.enabled else "/feature/enable"
        label = "Turn off" if f.enabled else "Turn on"
        toggle = _form(action, csrf, _hidden("feature", m.id), label)
        controls = f'{toggle} <a href="purge?feature={quote(m.id, safe="")}">Delete vectors...</a>'
    else:
        action = "/feature/disable" if f.enabled else "/feature/enable"
        label = "Turn off" if f.enabled else "Turn on"
        toggle = _form(action, csrf, _hidden("feature", m.id), label)
        tier = _form("/feature/tier", csrf,
                     _hidden("feature", m.id) + _tier_select(f.tier.value, m.title),
                     "Set tier")  # fmt: skip
        purge = f'<a href="purge?feature={quote(m.id, safe="")}">Purge data...</a>'
        controls = f"{toggle} {tier} {purge}"
    return (f'<div class="card"><div class="row"><div class="grow"><b>{_e(m.title)}</b> '
            f"{state}{pending}<br><small>{_e(m.description)}</small><br>{caps}<br>"
            f'<small>{_e(counts)}</small></div><div class="controls">{controls}</div></div>'
            "</div>")  # fmt: skip


def purge_preview(nonce: str, csrf: str, flash: list[tuple[str, str]], plan: Any) -> str:
    if plan.feature.startswith("embedder."):
        what = (f"<b>{_n(plan.chunks, 'vector')}</b> ({_e(_size(plan.bytes))}) made by this "
                "embedder. Search uses keywords until a sync embeds again "
                "(if it is on).")  # fmt: skip
        body = (f"<h1>Delete the vectors of {_e(plan.feature)}?</h1>{_flash(flash)}"
                f'<div class="card"><p>This deletes, securely and for good, {what}</p>'
                + _form("/feature/purge", csrf, _hidden("feature", plan.feature)
                        + '<label><input type="checkbox" name="confirm" value="yes" required> '
                          "Yes, delete them</label> ", "Delete", "danger")
                + ' <a href=".">Cancel</a></div>')  # fmt: skip
        return _shell(f"Delete vectors of {plan.feature}", body, nonce)
    body = (f"<h1>Purge {_e(plan.feature)}?</h1>{_flash(flash)}"
            f'<div class="card"><p>This deletes, securely and for good, everything indexed by '
            f"this feature: <b>{plan.files:,}</b> files, <b>{plan.skipped:,}</b> skipped-file "
            f"records, <b>{plan.chunks:,}</b> chunks ({_e(_size(plan.bytes))} of text), and "
            f"their vectors. The files on disk are not touched; a later sync with the feature "
            f"on would index them again.</p>"
            + _form("/feature/purge", csrf, _hidden("feature", plan.feature)
                    + '<label><input type="checkbox" name="confirm" value="yes" required> '
                      "Yes, delete it</label> ", "Purge", "danger")
            + ' <a href=".">Cancel</a></div>')  # fmt: skip
    return _shell(f"Purge {plan.feature}", body, nonce)


def forbidden() -> str:
    return _shell("Not allowed", "<h1>Not allowed</h1><p>Open the settings page from the link "
                  "that <code>lanternkb ui</code> printed. Each link works once.</p>")  # fmt: skip


def not_found() -> str:
    return _shell("Not found", "<h1>Not found</h1>")
