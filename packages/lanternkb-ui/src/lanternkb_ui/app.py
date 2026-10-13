"""The settings page server: on demand, loopback only, gone when the tab closes.

Threat model T10:
- **Loopback only, random port, random path.** The server binds 127.0.0.1 and serves
  under a random path prefix; the session cookie is scoped to that path, so other servers
  on 127.0.0.1 (cookies ignore ports) never receive it.
- **A one-time token** in the opening link is exchanged for the session cookie (HttpOnly,
  SameSite=Strict), then refused if used again. The link reaches the browser through a
  private file (mode 600 in a folder only the user can enter), never through a command line
  other users could read.
- **Host, Origin and Sec-Fetch-Site checks** on every request (against DNS rebinding and
  cross-site requests), and a per-session form token on every change. The referrer policy
  is `same-origin`, so browsers send the page's real Origin with its forms.
- **Every change goes through `lanternkb.control`,** which validates it, writes the config
  atomically and records it in the audit log.
- **Strict headers:** a Content-Security-Policy with per-response nonces, no framing, no
  caching, no cross-origin access. Nothing is logged, and errors never reach the terminal.
- **It exits** when its last open page says goodbye or stops its heartbeat, or after 10
  minutes without a user action.
"""

from __future__ import annotations

import hmac
import http.server
import json
import logging
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass, field
from html import escape
from http import HTTPStatus
from pathlib import Path
from typing import Any

from lanternkb.control import Control, ControlError
from lanternkb_ui import page

IDLE_S = 600  # exit after this long without a user action
BEAT_GRACE_S = 150  # a page whose heartbeat stopped this long is gone (hidden tabs beat slowly)
BYE_GRACE_S = 3  # after the last page says goodbye, wait this long for a reload
LOGIN_WINDOW_S = 300  # the opening link works this long
MAX_BODY = 64_000


# A looser tier needs a fresh read on an open worker first, so say so.
OPEN_LATER = " Its files become open after the next sync reads them again."


SEARCH_K = 20  # results on the search page
RESULTS_KEPT = 20  # searches whose results the page can show again (back, reload)
RESULTS_KEPT_S = 1800.0
SEARCH_TIMEOUT_S = 120.0  # the first search may load the models


@dataclass
class State:
    control: Control
    port: int = 0
    prefix: str = field(default_factory=lambda: "/" + secrets.token_urlsafe(12) + "/")
    token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    token_used: bool = False
    started: float = field(default_factory=time.monotonic)
    session: str | None = None
    csrf: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    searcher: Searcher | None = None  # made on the first search
    results: dict[str, tuple[float, dict[str, Any]]] = field(default_factory=dict)
    last_action: float = field(default_factory=time.monotonic)
    pages: dict[str, float] = field(default_factory=dict)  # page id -> last heartbeat
    seen_page: bool = False  # a page has beaten at least once
    bye_at: float | None = None
    flash: list[tuple[str, str]] = field(default_factory=list)  # (kind, message)
    sync: dict[str, Any] = field(default_factory=lambda: {"running": False})
    lock: threading.Lock = field(default_factory=threading.Lock)

    def keep(self, shown: dict[str, Any]) -> str:
        """Keep a search's results for the page that shows them; returns their handle."""
        handle = secrets.token_urlsafe(16)
        now = time.monotonic()
        with self.lock:
            for h, (at, _) in list(self.results.items()):
                if now - at > RESULTS_KEPT_S:
                    del self.results[h]
            while len(self.results) >= RESULTS_KEPT:
                del self.results[next(iter(self.results))]  # the oldest
            self.results[handle] = (now, shown)
        return handle

    def result(self, handle: str) -> dict[str, Any] | None:
        with self.lock:
            kept = self.results.get(handle)
        if kept is None or time.monotonic() - kept[0] > RESULTS_KEPT_S:
            return None
        return kept[1]

    def hosts(self) -> set[str]:
        return {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}

    def origins(self) -> set[str]:
        return {f"http://{h}" for h in self.hosts()}

    @property
    def cookie(self) -> str:
        return f"lk_session_{self.port}"


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "lanternkb-ui"
    sys_version = ""
    timeout = 10  # an idle connection cannot hold a thread
    state: State  # set on the subclass made by Server

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        pass  # never log: request lines carry the token and paths

    # -- checks -------------------------------------------------------------------------------

    def _host_ok(self) -> bool:
        return self.headers.get("Host", "") in self.state.hosts()

    def _session_ok(self) -> bool:
        session = self.state.session
        if session is None:
            return False
        for part in self.headers.get("Cookie", "").split(";"):
            name, _, value = part.strip().partition("=")
            if name == self.state.cookie and hmac.compare_digest(
                value.encode("utf-8", "replace"), session.encode()
            ):
                return True
        return False

    def _site_ok(self) -> bool:
        """Browsers say where a request comes from; only this page itself, or the user
        typing or following the link (`none`), may reach us."""
        site = self.headers.get("Sec-Fetch-Site")
        return site is None or site in ("same-origin", "none")

    def _guard(self, write: bool) -> bool:
        """Host and site for everything; session after login; Origin for changes."""
        ok = self._host_ok() and self._site_ok() and self._session_ok()
        if ok and write:
            ok = self.headers.get("Origin", "") in self.state.origins()
        if not ok:
            self._send(HTTPStatus.FORBIDDEN, page.forbidden(), "text/html")
        return ok

    def _route(self) -> tuple[str, dict[str, list[str]]] | None:
        """The path inside the app's random prefix, or None (not found)."""
        url = urllib.parse.urlsplit(self.path)
        if not url.path.startswith(self.state.prefix):
            return None
        return url.path[len(self.state.prefix) :], urllib.parse.parse_qs(url.query)

    # -- responses ----------------------------------------------------------------------------

    def _send(self, status: HTTPStatus, body: str, ctype: str, nonce: str = "",
              extra: dict[str, str] | None = None) -> None:  # fmt: skip
        data = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        script = f"'nonce-{nonce}'" if nonce else "'none'"
        self.send_header("Content-Security-Policy",
                         f"default-src 'none'; script-src {script}; style-src {script}; "
                         "connect-src 'self'; form-action 'self'; base-uri 'none'; "
                         "frame-ancestors 'none'")  # fmt: skip
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("X-Content-Type-Options", "nosniff")
        # same-origin, not no-referrer: with no-referrer browsers send `Origin: null` on
        # the page's own form posts, which the Origin check must refuse.
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _redirect(self, location: str, extra: dict[str, str] | None = None) -> None:
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "same-origin")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()

    def _render(self, html_fn: Any, **kw: Any) -> None:
        nonce = secrets.token_urlsafe(16)
        with self.state.lock:
            flash, self.state.flash = self.state.flash, []
        body = html_fn(nonce=nonce, csrf=self.state.csrf, flash=flash, **kw)
        self._send(HTTPStatus.OK, body, "text/html", nonce)

    # -- GET ----------------------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - stdlib name
        route = self._route()
        if route is None:
            return self._send(HTTPStatus.NOT_FOUND, page.not_found(), "text/html")
        path, query = route
        if path == "start":
            return self._login(query.get("t", [""])[0])
        if not self._guard(write=False):
            return None
        if path != "sync/status":
            self.state.last_action = time.monotonic()
        c = self.state.control
        if path == "":
            return self._render(page.main, data=_gather(c), sync=dict(self.state.sync))
        if path == "purge":
            try:
                plan = c.purge(query.get("feature", [""])[0], confirm=False)
            except (ControlError, ValueError, OSError) as e:
                self._flash("error", _message(e))
                return self._redirect(self.state.prefix)
            return self._render(page.purge_preview, plan=plan)
        if path == "sync/status":
            return self._send(HTTPStatus.OK, json.dumps(self.state.sync), "application/json")
        if path == "search":  # the results of a search just sent (Post/Redirect/Get)
            kept = self.state.result(query.get("r", [""])[0][:64])
            searcher = self.state.searcher
            if kept is None or searcher is None:
                self._flash("error", "That search has expired; search again.")
                return self._redirect(self.state.prefix)
            # what today's config still shows (a hand edit of the config since the search)
            still = searcher.still([r["path"] for r in kept["results"]])
            kept = dict(kept, results=[r for r in kept["results"] if r["path"] in still])
            return self._render(page.search, reveal=REVEAL_LABEL if _revealer() else "",
                                **kept)  # fmt: skip
        return self._send(HTTPStatus.NOT_FOUND, page.not_found(), "text/html")

    def _login(self, token: str) -> None:
        s = self.state
        fresh = time.monotonic() - s.started < LOGIN_WINDOW_S
        good = hmac.compare_digest(token.encode("utf-8", "replace"), s.token.encode())
        if not (self._host_ok() and self._site_ok()) or s.token_used or not fresh or not good:
            return self._send(HTTPStatus.FORBIDDEN, page.forbidden(), "text/html")
        s.token_used = True
        s.session = secrets.token_urlsafe(32)
        s.last_action = time.monotonic()
        cookie = f"{s.cookie}={s.session}; HttpOnly; SameSite=Strict; Path={s.prefix}"
        return self._redirect(s.prefix, {"Set-Cookie": cookie})

    # -- POST ---------------------------------------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802 - stdlib name
        route = self._route()
        if route is None:
            return self._send(HTTPStatus.NOT_FOUND, page.not_found(), "text/html")
        path, _query = route
        if not self._guard(write=True):
            return None
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if not 0 <= length <= MAX_BODY:
            return self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, page.forbidden(), "text/html")
        raw = self.rfile.read(length).decode("utf-8", "replace")
        form = {k: v[0] for k, v in urllib.parse.parse_qs(raw, keep_blank_values=True).items()}
        if path in ("beat", "bye"):  # page liveness, not user actions
            self._liveness(path, form.get("page", "")[:64])
            return self._send(HTTPStatus.NO_CONTENT, "", "text/plain")
        if not hmac.compare_digest(form.get("csrf", "").encode(), self.state.csrf.encode()):
            return self._send(HTTPStatus.FORBIDDEN, page.forbidden(), "text/html")
        self.state.last_action = time.monotonic()
        if path == "search":
            return self._search(form)
        if path == "reveal":
            return self._reveal(form)
        with self.state.lock:  # a change can hide or purge what kept results show
            self.state.results.clear()
        try:
            self._act(path, form)
        except (ControlError, ValueError, OSError) as e:
            self._flash("error", _message(e))
        return self._redirect(self.state.prefix)

    def _search(self, form: dict[str, str]) -> None:
        s = self.state
        with s.lock:
            if s.searcher is None:
                s.searcher = Searcher(s.control.config_path)
            searcher = s.searcher
        t0 = time.monotonic()
        results, error, note = searcher.search(form)
        took = int((time.monotonic() - t0) * 1000)
        s.last_action = time.monotonic()  # a slow first search is not idle time
        # Post/Redirect/Get: the results stay here under a random handle and the
        # browser is sent to fetch them, so its history and session hold no form body (the
        # query) and going back to the results is an ordinary page load.
        form = {k: form.get(k, "") for k in ("q", "types", "since", "before")}
        handle = s.keep({"form": form, "results": results, "error": error, "note": note,
                         "took_ms": took})  # fmt: skip
        return self._redirect(f"{s.prefix}search?r={handle}")

    def _reveal(self, form: dict[str, str]) -> None:
        """Show a search result's file in the file manager (never open or run it). Only a
        result id is accepted, and only a file today's config still shows is revealed."""
        searcher = self.state.searcher
        path = (searcher.locate(form.get("id", ""), form.get("path", ""))
                if searcher is not None else None)  # fmt: skip
        if path is None or not _show(path):
            return self._send(HTTPStatus.NOT_FOUND, "not available", "text/plain")
        return self._send(HTTPStatus.NO_CONTENT, "", "text/plain")

    def _liveness(self, what: str, page_id: str) -> None:
        s = self.state
        with s.lock:
            if what == "beat" and page_id:
                s.pages[page_id] = time.monotonic()
                s.seen_page = True
                s.bye_at = None
            elif what == "bye":
                s.pages.pop(page_id, None)
                if not s.pages:
                    s.bye_at = time.monotonic()

    def _act(self, path: str, form: dict[str, str]) -> None:
        c = self.state.control
        feature, tier = form.get("feature", ""), form.get("tier", "")
        if path == "feature/enable":
            c.enable(feature)
            self._flash("ok", f"{feature} is on. Its files are searchable again; a sync reads "
                              "any that changed.")  # fmt: skip
        elif path == "feature/disable":
            c.disable(feature)
            self._flash("ok", f"{feature} is off. Its data is hidden now; nothing was deleted.")
        elif path == "feature/tier":
            c.set_feature_tier(feature, tier)
            self._flash("ok", f"{feature} is now {tier}." + (OPEN_LATER if tier == "open" else ""))
        elif path == "feature/purge":
            if form.get("confirm") != "yes":
                raise ControlError("purge needs the confirmation box ticked")
            plan = c.purge(feature, confirm=True)
            self._flash("ok", f"Purged {feature}: {plan.files} files and {plan.chunks} chunks "
                              "deleted securely.")  # fmt: skip
        elif path == "root/add":
            raw = form.get("path", "").strip()
            if not (raw.startswith("/") or raw.startswith("~")):
                raise ControlError("type the folder's full path (starting with / or ~)")
            c.add_root(raw, tier or "local-only")
            added = Path(os.path.expanduser(raw)).resolve()
            self._flash("ok", f"Added {added} as {tier or 'local-only'}. Run a sync to index "
                              "it.")  # fmt: skip
        elif path == "root/tier":
            c.set_root_tier(form.get("path", ""), tier)
            self._flash("ok", f"Folder is now {tier}." + (OPEN_LATER if tier == "open" else ""))
        elif path == "root/remove":
            if form.get("confirm") != "yes":
                raise ControlError("removing a folder needs the confirmation box ticked")
            try:
                n = c.remove_root(form.get("path", ""), force=form.get("force") == "yes")
            except ControlError as e:
                if "--force" in str(e):
                    hint = ", or tick 'even if files become more visible'"
                    raise ControlError(str(e).replace(", or use --force", hint)) from e
                raise
            self._flash("ok", f"Folder removed; {n} indexed file records deleted.")
        elif path == "sync":
            self._start_sync()
        else:
            raise ControlError("unknown action")

    def _flash(self, kind: str, message: str) -> None:
        with self.state.lock:
            self.state.flash.append((kind, message))

    def _start_sync(self) -> None:
        s = self.state
        with s.lock:
            if s.sync.get("running"):
                return
            s.sync = {"running": True, "started": time.time()}

        def progress(stage: str, done: int, total: int) -> None:
            with s.lock:
                if s.sync.get("running"):
                    s.sync.update(stage=stage, done=done, total=total)

        def run() -> None:
            try:
                r = s.control.sync(progress)
                note = "; ".join(x for x in [*r.warnings, r.embed_note] if x)
                result = {"ok": True, "seen": r.seen, "indexed": r.indexed,
                          "unchanged": r.unchanged, "removed": r.removed,
                          "skipped": sum(r.skipped.values()), "errors": r.errors,
                          "embedded": r.embedded, "pending": r.pending,
                          "note": note}  # fmt: skip
            except Exception as e:  # noqa: BLE001 - shown on the page
                result = {"ok": False, "error": _message(e)}
            with s.lock:
                s.sync = {"running": False, "finished": time.time(), "result": result}
                s.results.clear()  # a sync can remove or change what kept results show

        threading.Thread(target=run, daemon=True).start()


def _message(e: BaseException) -> str:
    """What the page may say about an error: control messages in full, others by type."""
    return str(e) if isinstance(e, ControlError) else f"that did not work ({type(e).__name__})"


REVEAL_LABEL = "Show in Finder" if sys.platform == "darwin" else "Show in folder"


def _revealer() -> list[str] | None:
    """The command that shows a file in the file manager here, if there is one."""
    if sys.platform == "darwin":
        return ["/usr/bin/open", "-R"]
    opener = shutil.which("xdg-open")
    return [opener] if opener else None


def _show(path: str) -> bool:
    """Show the file (its folder, on Linux) in the file manager; True if that started."""
    cmd = _revealer()
    target = path if sys.platform == "darwin" else os.path.dirname(path)
    if cmd is None or not os.path.isabs(path) or not os.path.isfile(path):
        return False
    if sys.platform != "darwin" and not os.path.isdir(target):
        return False
    try:
        subprocess.run([*cmd, target], timeout=10, check=True, stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return False
    return True


class Searcher:
    """The search page's searches, run by the MCP server's own search on one thread: the
    index connection stays on that thread, the models load on the first search and unload
    when idle, and today's config decides what may be shown, as `lanternkb serve` does.
    Every tier, except inside Claude Code, where it shows open files only, as the terminal
    does (docs/threat-model.md: a speed bump, not a wall)."""

    def __init__(self, config_path: Path) -> None:
        from concurrent.futures import ThreadPoolExecutor

        from lanternkb.server import Server as Core
        from lanternkb.tiers import ServerTier

        self.open_only = bool(os.environ.get("CLAUDECODE"))
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lanternkb-search")
        self._lock = threading.Lock()
        self._pending: set[Any] = set()
        self._core = Core(config_path, ServerTier.OPEN if self.open_only else ServerTier.ALL)

    def search(self, form: dict[str, str]) -> tuple[list[dict[str, Any]], str, str]:
        """(results, error, note) for the form's query and filters."""
        from lanternkb.server import ToolError

        args = {"query": form.get("q", ""), "k": SEARCH_K, "types": form.get("types", ""),
                "modified_after": form.get("since", "").strip(),
                "modified_before": form.get("before", "").strip()}  # fmt: skip
        if not args["query"].strip():
            return [], "Type something to search for.", ""
        fut = None
        try:
            fut = self._submit(self._core.tool_search, args)
            out = fut.result(SEARCH_TIMEOUT_S)
        except ToolError as e:
            return [], str(e), ""
        except TimeoutError:
            if fut is not None:
                fut.cancel()  # not run later if it has not started
            return [], "The search took too long; try again in a moment.", ""
        except Exception as e:  # noqa: BLE001 - shown on the page, never a traceback
            return [], f"the search did not work ({type(e).__name__})", ""
        notes = [out["note"]] if out.get("note") else []
        if self.open_only:
            notes.append("inside Claude Code: open files only")
        return out["results"], "", "; ".join(notes)

    def locate(self, result_id: str, path: str) -> str | None:
        """The file of a search result, if today's config still shows it. The path is
        checked too: the same text at the same place in two folders has one result id
       ."""
        fut = None
        try:
            fut = self._submit(self._core.find_chunk, result_id[:40], path[:4096])
            hit = fut.result(SEARCH_TIMEOUT_S)
        except Exception:  # noqa: BLE001 - no such result, a timeout, a closed index
            if fut is not None:
                fut.cancel()
            return None
        return str(hit.path) if hit is not None else None

    def still(self, paths: list[str]) -> set[str]:
        """Those of `paths` today's config still shows (none if that cannot be checked)."""
        try:
            fut = self._submit(lambda: {p for p in paths if self._core._still_visible(p)})
            return set(fut.result(SEARCH_TIMEOUT_S))
        except Exception:  # noqa: BLE001 - shown as no results, never as an error page
            return set()

    def _submit(self, fn: Any, *args: Any) -> Any:
        with self._lock:
            self._pending = {f for f in self._pending if not f.done()}
            fut = self._pool.submit(fn, *args)
            self._pending.add(fut)
        return fut

    def close(self) -> None:
        with self._lock:
            for fut in self._pending:
                fut.cancel()  # queued searches never start
        self._pool.submit(self._core.close)
        self._pool.shutdown(wait=False)


def _gather(c: Control) -> dict[str, Any]:
    """Everything the page shows, read through the control API; a read error is shown, not
    raised."""
    out: dict[str, Any] = {"config": str(c.config_path), "status": {}, "features": [],
                           "roots": (), "skipped": [], "notes": [], "error": None}  # fmt: skip
    try:
        out["status"] = c.status()
        out["features"] = c.features()
        out["roots"] = c.roots()
        out["skipped"] = c.skipped()
        out["notes"] = c.notes()
    except (ControlError, ValueError, OSError) as e:
        out["error"] = _message(e)
    return out


class _HTTPServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 64

    def handle_error(self, request: Any, client_address: Any) -> None:
        pass  # never print tracebacks (they could carry paths or config text)


class Server:
    """A running settings page. `url` opens it once; `wait()` returns when it has exited."""

    def __init__(self, control: Control, idle_s: float = IDLE_S):
        self.state = State(control)
        handler = type("BoundHandler", (Handler,), {"state": self.state})
        self.httpd = _HTTPServer(("127.0.0.1", 0), handler)
        self.state.port = self.httpd.server_address[1]
        self.idle_s = idle_s
        self._stopped = threading.Event()
        self._link_dir: str | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.state.port}{self.state.prefix}start?t={self.state.token}"

    def link_file(self) -> Path:
        """A private file that forwards the browser to the one-time link, so the link never
        appears on a command line (`ps`) when the browser is started."""
        self._link_dir = tempfile.mkdtemp(prefix="lanternkb-ui-")  # mode 700
        path = Path(self._link_dir) / "open.html"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write('<!doctype html><meta charset="utf-8"><meta name="referrer" '
                    f'content="no-referrer"><meta http-equiv="refresh" content="0; url='
                    f'{escape(self.url, quote=True)}"><title>lanternkb</title>')  # fmt: skip
        return path

    def _watch(self) -> None:
        s = self.state
        while not self._stopped.wait(0.5):
            now = time.monotonic()
            with s.lock:
                for pid, beat in list(s.pages.items()):
                    if now - beat > BEAT_GRACE_S:
                        del s.pages[pid]
                lapsed = s.seen_page and not s.pages and s.bye_at is None  # beats stopped
                said_bye = s.bye_at is not None and now - s.bye_at > BYE_GRACE_S
            idle = now - s.last_action > self.idle_s
            if s.token_used and self._link_dir:
                shutil.rmtree(self._link_dir, ignore_errors=True)
                self._link_dir = None
            if (idle or said_bye or lapsed) and not s.sync.get("running"):
                self.stop()

    def start(self) -> None:
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        threading.Thread(target=self._watch, daemon=True).start()

    def stop(self) -> None:
        if not self._stopped.is_set():
            self._stopped.set()
            threading.Thread(target=self.httpd.shutdown, daemon=True).start()

    def wait(self) -> None:
        self._stopped.wait()
        self.httpd.server_close()
        if self.state.searcher is not None:
            self.state.searcher.close()
        if self._link_dir:
            shutil.rmtree(self._link_dir, ignore_errors=True)


def run(config_path: Path, open_browser: bool = True, idle_s: float = IDLE_S) -> int:
    """`lanternkb ui`: serve the page until it is closed, then exit."""
    # lanternkb's warnings (exception names only) stay off the terminal: no handler there
    logging.getLogger("lanternkb").addHandler(logging.NullHandler())
    server = Server(Control(config_path), idle_s=idle_s)
    server.start()
    print(f"Settings page (this link works once): {server.url}", flush=True)
    print("It closes with the tab, or after 10 idle minutes.", flush=True)
    if open_browser:
        webbrowser.open(server.link_file().as_uri())
    try:
        server.wait()
    except KeyboardInterrupt:
        server.stop()
    return 0
