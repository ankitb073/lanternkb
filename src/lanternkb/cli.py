"""`lanternkb`: the command line. A thin client of `lanternkb.control`.

Output never includes a matched secret, and document text only where the user asks for it
(`search` prints redacted snippets). `serve` runs the MCP server without loading the
control API at all.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from lanternkb import __version__
from lanternkb.config import DEFAULT_CONFIG, load
from lanternkb.hints import shown, use_config


def _control(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    from lanternkb.control import Control

    return Control(args.config)


def cmd_scan(args: argparse.Namespace) -> int:
    from lanternkb.config import load
    from lanternkb.scan.engine import Scanner, ScanTimeout
    from lanternkb.scan.rules import load_rules

    path = args.config.expanduser()
    sc = load(path).scanner if path.is_file() else None  # the config's rules and actions
    scanner = Scanner(load_rules(disabled=sc.disabled_rules, actions=sc.actions,
                                 enabled=sc.enabled_rules) if sc else None)  # fmt: skip
    total = 0
    for target in args.paths:
        files = (
            [target] if target.is_file() else sorted(p for p in target.rglob("*") if p.is_file())
        )
        for f in files:
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                print(f"{f}: unreadable ({e.strerror})", file=sys.stderr)
                continue
            rel = f.relative_to(target).as_posix() if target.is_dir() else f.name
            try:
                found = scanner.scan(text, rel)
            except ScanTimeout:
                print(f"{f}: scan timed out; sync would skip this file", file=sys.stderr)
                total += 1
                continue
            for finding in found:
                total += 1
                enc = " (base64)" if finding.encoded else ""
                print(f"{f}:{finding.line}: {finding.rule} -> {finding.action}{enc}")
    print(f"{total} finding(s); rule set {scanner.hash}", file=sys.stderr)
    return 1 if total else 0


def cmd_explain(args: argparse.Namespace) -> int:
    from lanternkb.policy import PathPolicy
    from lanternkb.store import Store
    from lanternkb.tiers import Tier

    cfg = load(args.config.expanduser())
    policy = PathPolicy(cfg)
    store = Store(cfg.index_path, readonly=True) if cfg.index_path.exists() else None
    try:
        for p in args.paths:
            d = policy.check(p)
            row = store.get_file(str(d.path)) if store is not None and d.allowed else None
            if not d.allowed:
                status = f"skip ({d.reason})"
            elif row is not None and row.status == "indexed":
                status = f"indexed as {row.tier} ({d.feature})"
                if d.tier == Tier.OPEN and row.read_tier != Tier.OPEN.value:
                    status += ("; read again at the next sync, on an open worker" if row.tier
                               == Tier.OPEN.value else "; open after the next sync reads it "
                               "again on an open worker")  # fmt: skip
                elif row.tier != (d.tier.value if d.tier else None):
                    status += f"; its path alone would give {d.tier}"
            else:
                status = (f"would index as {d.tier} ({d.feature}), by its path; a secret or "
                          "a tier rule found when it is read can make it local-only")  # fmt: skip
            print(f"{_plain(str(d.path))}: {status}")
    finally:
        if store is not None:
            store.close()
    return 0


def _mb(n: int) -> str:
    return f"{n / 1e6:.1f} MB"


LOOSER_NOTE = (
    "ok. Files that become open stay local-only until the next sync reads them "
    "again (`lanternkb sync`)."
)


def cmd_features(args: argparse.Namespace) -> int:
    from lanternkb.plugins import trust_note

    c = _control(args)
    if args.action == "list":
        for f in c.features():
            m = f.manifest
            state = "required" if m.required else ("on" if f.enabled else "off")
            if not f.applied:
                state += "*"
            print(f"{m.id:18s} {state:8s} tier={f.tier.value:10s} {m.category:9s} "
                  f"files={f.files} chunks={f.chunks} text={_mb(f.bytes)}  {m.title}")  # fmt: skip
            print(f"{'':18s} reads: {'; '.join(m.reads) or '-'}; network: no; "
                  f"OS permissions: {', '.join(m.os_permissions) or 'none'}")  # fmt: skip
            if note := trust_note(m.id):
                print(f"{'':18s} {note}")
        if not all(f.applied for f in c.features()):
            print("* set in the config but not yet applied to the index: run `lanternkb sync`")
        return 0
    if args.action == "enable":
        if note := trust_note(args.feature):
            print(f"note: {args.feature} is {note}")
        c.enable(args.feature)
        if args.feature.startswith("embedder."):
            from lanternkb.embed import enabled_embedders

            keys = enabled_embedders(c.registry())
            if len(keys) > 1:
                print(f"note: only one embedder is used: {keys[0]}. The others stay on but "
                      "unused; turn them off to save memory and disk.")  # fmt: skip
    elif args.action == "disable":
        c.disable(args.feature)
        print(f"{args.feature} is off; its data is hidden now (not deleted). "
              f"`lanternkb features purge {args.feature}` deletes it.")  # fmt: skip
        return 0
    elif args.action == "tier":
        c.set_feature_tier(args.feature, args.tier)
        print(LOOSER_NOTE if args.tier == "open" else "ok")
        return 0
    elif args.action == "purge":
        plan = c.purge(args.feature, confirm=False)
        print(f"{args.feature}: {plan.files} indexed files, {plan.skipped} skipped records, "
              f"{plan.chunks} chunks, {_mb(plan.bytes)} of text would be deleted.")  # fmt: skip
        if not args.yes:
            answer = input("Delete them? Type the feature id to confirm: ").strip()
            if answer != args.feature:
                print("Nothing deleted.")
                return 1
        done = c.purge(args.feature, confirm=True)
        print("Deleted. If the feature is on, the next sync rebuilds its data.")
        if not done.wal_clean:
            print("Note: a running reader kept old pages in the index's WAL file; they are "
                  "cleared at the next sync or when the reader exits.")  # fmt: skip
        return 0
    print("ok")
    return 0


def cmd_roots(args: argparse.Namespace) -> int:
    c = _control(args)
    if args.action == "list":
        for r in c.roots():
            print(f"{r.path}  tier={r.tier.value}")
        return 0
    if args.action == "add":
        c.add_root(args.path, args.tier)
    elif args.action == "remove":
        n = c.remove_root(args.path, force=args.force)
        print(f"Removed; {n} indexed file record(s) deleted.")
        return 0
    elif args.action == "tier":
        c.set_root_tier(args.path, args.tier)
        print(LOOSER_NOTE if args.tier == "open" else "ok")
        return 0
    print("ok")
    return 0


_STAGES = {"checking": "Checking files", "reading": "Reading changed files",
           "embedding": "Embedding passages"}  # fmt: skip


@contextlib.contextmanager
def _progress(wanted: bool) -> Iterator[Any]:
    """A one-line counter on the terminal while a sync runs (only on a terminal, never in a
    log); the line is cleared when it ends."""
    if not (wanted and sys.stderr.isatty()):
        yield None
        return
    last = [0.0]

    def show(stage: str, done: int, total: int) -> None:
        now = time.monotonic()
        if now - last[0] < 0.1 and done != total:
            return
        last[0] = now
        count = f"{done:,} of {total:,}" if total else f"{done:,}"
        sys.stderr.write(f"\r\033[K{_STAGES.get(stage, stage)}: {count}")
        sys.stderr.flush()

    class Clearing(logging.StreamHandler):  # type: ignore[type-arg]
        def emit(self, record: logging.LogRecord) -> None:
            sys.stderr.write("\r\033[K")  # a warning starts on a clean line
            super().emit(record)

    handler = Clearing(sys.stderr)
    handler.setLevel(logging.WARNING)
    logger = logging.getLogger("lanternkb")
    logger.addHandler(handler)
    try:
        yield show
    finally:
        logger.removeHandler(handler)
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()


def cmd_sync(args: argparse.Namespace) -> int:
    if args.scheduled:
        from lanternkb import power
        from lanternkb.schedule import rotate_log

        rotate_log()  # launchd appends to one file; keep it small
        reason = power.reason_to_wait()
        if reason:
            print(f"scheduled sync skipped: {reason}")
            return 0
        power.lower_priority()
    with _progress(not args.scheduled) as progress:
        r = _control(args).sync(progress)
    print(f"seen {r.seen}, indexed {r.indexed}, unchanged {r.unchanged}, hidden {r.hidden}, "
          f"re-tiered {r.retiered}, removed {r.removed}, errors {r.errors}, "
          f"skipped {sum(r.skipped.values())} {dict(r.skipped)}, "
          f"redacted files {r.redacted_files}, extraction {r.isolation}")  # fmt: skip
    if r.isolation == "process":
        print("note: no sandbox on this platform; extractors ran in plain worker processes")
    if r.embedder:
        print(f"embeddings ({r.embedder}): embedded {r.embedded}, reused {r.reused}, "
              f"pending {r.pending}, rejected {r.failed}")  # fmt: skip
    if r.embed_note:
        print(shown(f"note: {r.embed_note}"))
    for w in r.warnings:
        print(shown(f"warning: {w}"))
    if not r.wal_clean:
        print("note: a reader held the index open, so old pages may stay in its -wal file "
              "until the next sync")  # fmt: skip
    return 0


def _usable(emb: Any, index: Path, query: str) -> tuple[Any, str]:
    """An embedder whose query vector this index can use: `emb`, or the same built-in model
    run by the backend that made the index (a long query cannot cross); else
    None and why, so the search says it is by keywords."""
    from lanternkb.embed import embedder_for_index
    from lanternkb.search import comparable
    from lanternkb.store import Store

    if emb is None:
        return None, ""
    store = Store(index, readonly=True)
    try:
        model = store.vector_models().get(emb.key)
    finally:
        store.close()
    if model is None or comparable(emb, model, query):
        return emb, ""
    other = embedder_for_index(emb.key, model[0])
    if other is not None and comparable(other, model, query):
        return other, ""
    return None, "the index's vectors were made by another model or backend; run `lanternkb sync`"


def cmd_recent(args: argparse.Namespace) -> int:
    args.query = "*"
    return cmd_search(args)


def cmd_search(args: argparse.Namespace) -> int:
    if isinstance(args.query, list):  # search plumber boiler, without quotes, works too
        args.query = " ".join(args.query)
    from lanternkb.config import load
    from lanternkb.embed import active_embedder
    from lanternkb.features import Registry
    from lanternkb.search import needs_vectors, search
    from lanternkb.store import Store
    from lanternkb.tiers import ServerTier

    if args.k < 1:
        print("lanternkb: -k must be at least 1", file=sys.stderr)
        return 2
    from lanternkb.search import Filters, when

    try:
        types = tuple(args.type)  # "pdf, md" and "pdf,md" both work (Filters splits them)
        filters = Filters(types, when(args.since) if args.since else None,
                          when(args.before) if args.before else None)  # fmt: skip
    except ValueError as e:
        print(f"lanternkb: {e}", file=sys.stderr)
        return 2
    cfg = load(args.config.expanduser()) if args.config.expanduser().exists() else None
    if cfg is None or not cfg.index_path.exists():
        print("lanternkb: no index yet; add a root and run `lanternkb sync`", file=sys.stderr)
        return 2
    from lanternkb.search import RECENT, recent

    listing = args.query.strip() in RECENT  # `*`: the files changed most recently
    emb, note = None, ""
    if not listing and needs_vectors(args.query):  # a one-off search: the quickest model
        emb, note = active_embedder(Registry(settings=cfg.features), query=args.query)
        emb, why = _usable(emb, cfg.index_path, args.query)
        note = why or note
    from lanternkb.rerank import active_reranker

    reranker = (None if args.no_rerank or listing
                else active_reranker(Registry(settings=cfg.features), one_off=True)[0])  # fmt: skip
    from lanternkb.policy import PathPolicy
    from lanternkb.tiers import visible

    if args.tier is None and os.environ.get("CLAUDECODE"):
        # An assistant running this through its shell gets what the open MCP server would
        # give it (a speed bump, not a wall: docs/threat-model.md, "Not claimed").
        args.tier = "open"
        print("note: inside Claude Code, search shows open files only; --tier all shows "
              "local-only files too", file=sys.stderr)  # fmt: skip
    store = Store(cfg.index_path, readonly=True)
    try:
        tier = ServerTier(args.tier or "all")
        # As the MCP server does: only what today's config would still index and show (a
        # feature switched off or uninstalled since the last sync, a folder removed), checked
        # before reranking.
        policy = PathPolicy(cfg, Registry(settings=cfg.features))

        def still(path: str) -> bool:
            d = policy.check(Path(path), canonical=True)
            return bool(d.allowed and d.tier is not None and visible(d.tier, tier))

        prefix = os.path.expanduser(args.path) if args.path else None
        if listing:
            capped: list[int] = []
            hits = recent(store, tier, k=args.k, path_prefix=prefix, visible=still,
                          filters=filters, capped=capped)  # fmt: skip
            if capped:
                print(shown("note: the newest files are hidden by changes since the last "
                            "sync; run `lanternkb sync`, then list again"),
                      file=sys.stderr)  # fmt: skip
        else:
            hits = search(store, args.query, tier, k=args.k, embedder=emb, path_prefix=prefix,
                          reranker=reranker, visible=still, filters=filters)  # fmt: skip
    finally:
        store.close()
    if note and emb is None:
        print(shown(f"note: {note}; keyword search only"), file=sys.stderr)
    if args.json:
        from lanternkb.server import modified

        rows = [{"path": h.path, "location": h.location or f"lines {h.start_line}-{h.end_line}",
                 "tier": h.tier, **({"modified": modified(h.score)} if listing
                                    else {"score": h.score}),
                 "snippet": h.snippet.replace("\u2063", "")} for h in hits]  # fmt: skip
        print(json.dumps(rows, ensure_ascii=False, indent=1))
        return 0
    bold = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    for h in hits:
        where = h.location or f"lines {h.start_line}-{h.end_line}"
        lead = (time.strftime("%Y-%m-%d %H:%M", time.localtime(h.score)) if listing
                else f"{h.score:.4f}")  # fmt: skip
        print(f"{lead}  {_plain(h.path)}  ({_plain(where)}, {h.tier})")
        print("    " + _shown(" ".join(h.snippet.replace("\u2063", "").split())[:200],
                              args.query, bold))  # fmt: skip
    return 0


def _plain(text: str) -> str:
    """Control characters (escape sequences included) shown as spaces."""
    return "".join(" " if ord(c) < 32 or 0x7F <= ord(c) < 0xA0 else c for c in text)


def _shown(text: str, query: str, bold: bool) -> str:
    """A snippet for the terminal: the query's words in bold on a terminal (not when piped,
    or with NO_COLOR set); control characters from the file shown as spaces, so a file
    cannot move the cursor or change the terminal's colours."""
    from lanternkb.search import match_spans

    text = _plain(text)
    if not bold:
        return text
    out, k = [], 0
    for a, b in match_spans(text, query):
        out += [text[k:a], "\033[1m", text[a:b], "\033[0m"]
        k = b
    return "".join(out) + text[k:]


def cmd_install_schedule(args: argparse.Namespace) -> int:
    from lanternkb import schedule

    text, how = schedule.render(args.config.expanduser(), args.every)
    print(text)
    print(how, file=sys.stderr)
    return 0


def _ask(question: str, default: bool) -> bool:
    """A yes/no question on the terminal; Enter takes the default."""
    hint = "[Y/n]" if default else "[y/N]"
    try:
        answer = input(f"{question} {hint} ").strip().lower()
    except EOFError:
        return False  # input ended (Ctrl-D): no, never a download nobody agreed to
    return default if not answer else answer in ("y", "yes")


def _broad(folder: Path) -> str | None:
    """Why a folder is a risky first choice (None if it is not): indexing all of it would
    take in system files, other apps' data and every credential lying around."""
    home = Path.home().resolve()
    p = folder.resolve()
    # the home folder or any folder above it, whatever the path says (a symlink, or macOS's
    # /System/Volumes/Data alias for /Users: compared by inode)
    for above in (home, *home.parents):
        try:
            if p == above or os.path.samefile(p, above):
                return "it is your whole home folder or more"
        except OSError:
            continue
    if p.name in ("Library", "AppData", ".ssh", ".gnupg", ".aws", ".config"):
        return "it holds app data or credentials"
    return None


def _typed_path(raw: str) -> Path:
    """A folder as typed or dropped on a terminal: quotes and backslash-space escapes
    removed."""
    raw = raw.strip()
    if len(raw) > 1 and raw[0] == raw[-1] and raw[0] in "'\"":
        raw = raw[1:-1]
    return Path(os.path.expanduser(re.sub(r"\\(.)", r"\1", raw)))  # `\ `, `\(`, `\&`...


def cmd_setup(args: argparse.Namespace) -> int:
    """`lanternkb setup`: the first five minutes, step by step. Nothing is read until a folder
    is added; nothing is downloaded without a yes (or --models); the first sync runs here."""
    try:
        return _setup(args)
    except KeyboardInterrupt:
        print("\nsetup stopped; what was added stays (lanternkb roots list)", file=sys.stderr)
        return 130


def _setup(args: argparse.Namespace) -> int:
    import shlex

    from lanternkb.control import Control, ControlError
    from lanternkb.fsutil import canonical_case

    interactive = sys.stdin.isatty() and not args.yes
    c = Control(args.config)
    print(
        "lanternkb setup: a search index of folders you choose, on this machine. It reads\n"
        "nothing else, and the only network use is the model downloads it asks about.\n"
    )
    have = list(c.roots()) if c.config_path.exists() else []
    for r in have:
        print(f"  already indexed: {r.path} ({r.tier.value})")
    folders = [Path(os.path.expanduser(f)) for f in args.folder]
    if not folders and interactive:
        print(
            "Which folders should it index? Notes, documents and code work best; you can add "
            "more later\n(`lanternkb roots add`). Press Enter on an empty line to finish."
        )
        while True:
            try:
                raw = input("  folder: ").strip()
            except EOFError:
                break
            if not raw:
                break
            folders.append(_typed_path(raw))
    if not folders and not have:
        print(
            "lanternkb: no folder to index; pass --folder PATH (or run this in a terminal)",
            file=sys.stderr,
        )
        return 2
    added = 0
    for folder in folders:
        if not folder.is_dir():
            print(f"  skipped {folder}: not a folder")
            continue
        if (why := _broad(folder)) and not (interactive and _ask(
                f"  {folder}: {why}. Index it anyway?", False)):  # fmt: skip
            print(f"  skipped {folder}: {why} (pick a narrower folder)")
            continue
        tier = args.tier
        if tier is None:
            tier = "open" if interactive and _ask(
                f"  May a cloud assistant such as Claude Code see the files in {folder}?",
                False) else "local-only"  # fmt: skip
        inside = _inside_local_only(c, folder)
        if tier == "open" and inside and not (interactive and _ask(
                f"  {folder} is inside {inside}, which is local-only: its files would become "
                "visible to cloud assistants. Go ahead?", False)):  # fmt: skip
            print(f"  skipped {folder}: inside the local-only folder {inside}")
            continue
        try:
            c.add_root(str(folder), tier)
            added += 1
            print(f"  added {folder} ({tier})")
        except ControlError as e:
            now = next((r.tier.value for r in c.roots()
                        if r.path == canonical_case(folder.resolve())), None)  # fmt: skip
            hint = (f"; to make it {tier}: lanternkb roots tier {shlex.quote(str(folder))} "
                    f"{tier}" if "already a root" in str(e) and now and now != tier
                    else "")  # fmt: skip
            print(f"  skipped {folder}: {e}{hint}")
    if not added and not have:
        print("lanternkb: no folder was added; nothing to index", file=sys.stderr)
        return 2
    _setup_models(c, args, interactive)
    if not args.no_sync:
        print("\nIndexing (the first run reads every file; later runs only what changed)...")
        try:
            with _progress(True) as progress:
                r = c.sync(progress)
            print(f"  {r.indexed} files indexed, {sum(r.skipped.values())} skipped, "
                  f"{r.errors} errors" + (f"; embedded {r.embedded} chunks" if r.embedder
                                          else ""))  # fmt: skip
            if r.embed_note:
                print(shown(f"  note: {r.embed_note}"))
        except Exception as e:  # noqa: BLE001 - the rest of setup still stands
            print(
                shown(f"  the sync stopped ({type(e).__name__}: {e}); run `lanternkb sync` later")
            )
    from lanternkb import schedule

    print(
        shown(
            '\nDone. Try it:\n  lanternkb search "what did the plumber say about the boiler"\n'
            "\nTo search from Claude Code (open files only), register the server:"
        )
    )
    print("  " + schedule.mcp_commands(args.config.expanduser()).split("\n")[1])
    if not any(r.tier.value == "open" for r in c.roots()):
        print(shown("  Every folder here is local-only, so Claude Code will see none of them. To "
                    "let it see one:\n  lanternkb roots tier <folder> open"))  # fmt: skip
    print(shown("\nMore: `lanternkb mcp-setup` (deny rules for assistants that run commands), "
                "`lanternkb install-schedule` (keep it fresh), `lanternkb ui` (settings), "
                "`lanternkb doctor` (checks)."))  # fmt: skip
    return 0


def _inside_local_only(c: Any, folder: Path) -> Path | None:
    """The local-only root a folder sits inside, if any: the deepest root decides a file's
    tier, so an open folder there would show files that were local-only."""
    from lanternkb.fsutil import canonical_case

    p = canonical_case(folder.resolve())  # spelt as the config stores roots
    for r in c.roots():
        if r.tier.value == "local-only" and r.path != p and r.path in p.parents:
            return r.path
    return None


def _setup_models(c: Any, args: argparse.Namespace, interactive: bool) -> None:
    """Offer the search models that are missing: the embedder, the reranker, and where torch
    runs them, the ONNX exports that make terminal searches quick."""
    from lanternkb import models, rerank
    from lanternkb.embed import EMBEDDERS, availability, backend

    registry = c.registry()
    if backend() is None:
        print(
            "\nSemantic search needs the `embed` extra (`uv sync --extra embed`, or pip "
            "install 'lanternkb[embed]'); keyword search works without it."
        )
        return
    offers = []
    key, why = availability(registry)
    if key in EMBEDDERS and why and "model missing" in why:
        offers.append((f"embed:{key}", None, "the search model (about 1.2 GB)", True))
    rkey, rwhy = rerank.availability(registry)
    if rkey and rwhy and "model missing" in rwhy:
        offers.append((f"rerank:{rkey}", None, "the reranker, which orders results better "
                       "(about 0.6 GB)", True))  # fmt: skip
    if backend() == "torch" and key in EMBEDDERS and EMBEDDERS[key].onnx is not None:
        onnx = EMBEDDERS[key].onnx
        if not (models.model_path(onnx.repo) / onnx.graph).is_file():
            offers.append((f"embed:{key}", "onnx", "quick-start copies for terminal searches "
                           "(about 1.2 GB more)", False))  # fmt: skip
    for name, using, what, default in offers:
        if args.models is not None:  # --models fetches what search needs, not the extras
            want = args.models and default
        else:
            want = interactive and _ask(f"\nDownload {what} from Hugging Face?", default)
        if not want:
            print(shown(f"  not downloaded: {what} (later: lanternkb models pull {name}"
                        + (f" --backend {using}" if using else "") + ")"))  # fmt: skip
            continue
        try:
            models.pull(name, using=using)
        except Exception as e:  # noqa: BLE001 - setup goes on; search works by keywords
            print(
                shown(
                    f"  download failed ({type(e).__name__}); later: lanternkb models pull {name}"
                )
            )


def cmd_mcp_setup(args: argparse.Namespace) -> int:
    from lanternkb import schedule

    print(schedule.mcp_commands(args.config.expanduser()))
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    try:
        from lanternkb_ui.app import run
    except ModuleNotFoundError as e:
        if e.name not in ("lanternkb_ui", "lanternkb_ui.app"):
            raise  # lanternkb-ui is installed but broken: show the real error
        print("lanternkb: the settings page is a separate package: install lanternkb-ui",
              file=sys.stderr)  # fmt: skip
        return 2
    return run(args.config.expanduser(), open_browser=not args.no_browser)


def cmd_doctor(args: argparse.Namespace) -> int:
    from lanternkb import doctor

    checks = doctor.run(args.config.expanduser())
    marks = {doctor.OK: "ok  ", doctor.WARN: "warn", doctor.FAIL: "FAIL"}
    for status, what, advice in checks:
        print(f"[{marks[status]}] {what}" + (f"\n       -> {shown(advice)}" if advice else ""))
    return 1 if any(c[0] == doctor.FAIL for c in checks) else 0


def cmd_status(args: argparse.Namespace) -> int:
    print(json.dumps(_control(args).status(), indent=1))
    return 0


def cmd_skipped(args: argparse.Namespace) -> int:
    c = _control(args)
    skipped = c.skipped()
    if not skipped:
        print("No files skipped.")
    for path, reason in skipped:
        print(f"{reason:28s} {_plain(path)}")  # a file's name may hold escape sequences
    notes = c.notes()
    if notes:
        print("\nIndexed, with a note:")
        for path, note in notes:
            print(f"  {_plain(path)}: {_plain(note)}")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from lanternkb import models

    if args.action == "pull":
        print("Fetching a model: the only network use in lanternkb; indexing and search never "
              "download anything.")  # fmt: skip
        try:
            path = models.pull(args.name or models.audio_model_name(), force=args.force,
                               using=args.backend)  # fmt: skip
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(f"Saved to {path}")
        return 0
    print(json.dumps(models.status(), indent=1, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    import resource

    # No core dumps: a crashed sync or server would write its memory (file text) to disk.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--redact" in raw:
        return _redacted([a for a in raw if a != "--redact"])
    return _run(raw)


def _redacted(argv: list[str]) -> int:
    """Run a command with all its output through a `Redactor`: usage errors and tracebacks
    too, so streams are swapped before the arguments are even read."""
    import traceback

    from lanternkb.redaction import RedactingStream, Redactor, known_names

    config = DEFAULT_CONFIG
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            config = Path(argv[i + 1])
        elif a.startswith("--config="):
            config = Path(a.split("=", 1)[1])
    known = known_names(config.expanduser()) + [a for a in argv if "/" in a]
    try:
        from lanternkb.features import Registry

        features = list(Registry().manifests)  # feature ids stay readable
    except Exception:  # noqa: BLE001 - redaction works without them
        features = []
    r = Redactor(known, keep=features)
    out, err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = RedactingStream(out, r), RedactingStream(err, r)  # type: ignore[assignment]
    try:
        return _run(argv, redacting=True)
    except SystemExit as e:  # argparse's own errors, already printed through the redactor
        return e.code if isinstance(e.code, int) else 1
    except KeyboardInterrupt:
        return 130
    except BaseException:  # noqa: BLE001 - shown redacted rather than raw
        sys.stderr.write("lanternkb failed; the traceback, redacted:\n" + traceback.format_exc())
        return 1
    finally:
        sys.stdout, sys.stderr = out, err


def _run(argv: list[str], redacting: bool = False) -> int:
    # No abbreviations: `--red` must not run as `--redact` unredacted.
    parser = argparse.ArgumentParser(prog="lanternkb", allow_abbrev=False)
    parser.add_argument("--version", action="version", version=f"lanternkb {__version__}")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--redact", action="store_true",
                        help="hide your home folder and file names in the output (for bug "
                        "reports)")  # fmt: skip
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="scan files for secrets; prints rule and line, never values")
    s.add_argument("paths", nargs="+", type=Path)
    e = sub.add_parser("explain", help="say whether each path would be indexed, and why")
    e.add_argument("paths", nargs="+", type=Path)
    f = sub.add_parser("features", help="list, enable, disable or purge features")
    fs = f.add_subparsers(dest="action", required=True)
    fs.add_parser("list")
    for name in ("enable", "disable"):
        fs.add_parser(name).add_argument("feature")
    ft = fs.add_parser("tier")
    ft.add_argument("feature")
    ft.add_argument("tier", choices=["open", "local-only"])
    fp = fs.add_parser("purge", help="delete a feature's data (asks first)")
    fp.add_argument("feature")
    fp.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    r = sub.add_parser("roots", help="list, add or remove roots, or set a root's tier")
    rs = r.add_subparsers(dest="action", required=True)
    rs.add_parser("list")
    ra = rs.add_parser("add")
    ra.add_argument("path")
    ra.add_argument("--tier", choices=["open", "local-only"], default="local-only")
    rr = rs.add_parser("remove")
    rr.add_argument("path")
    rr.add_argument("--force", action="store_true", help="even if it makes files more visible")
    rt = rs.add_parser("tier")
    rt.add_argument("path")
    rt.add_argument("tier", choices=["open", "local-only"])
    mo = sub.add_parser("models", help="fetch or list local models (semantic search, audio)")
    ms = mo.add_subparsers(dest="action", required=True)
    mp = ms.add_parser("pull", help="download a model (needs network; asks nothing else): "
                       "embed:embeddinggemma-300m, or an audio model (default)")  # fmt: skip
    mp.add_argument("name", nargs="?", default=None)
    mp.add_argument("--force", action="store_true", help="fetch it again even if present")
    mp.add_argument("--backend", choices=("torch", "onnx"), default=None,
                    help="for embed:, the form to fetch (default: the installed backend's); on "
                    "macOS, --backend onnx as well makes terminal searches faster")  # fmt: skip
    ms.add_parser("status")
    st = sub.add_parser("setup", help="set up step by step: folders, tiers, models, first sync")
    st.add_argument(
        "--folder",
        action="append",
        default=[],
        metavar="PATH",
        help="a folder to index (repeat for more); asked for in a terminal if none",
    )
    st.add_argument(
        "--tier",
        choices=["open", "local-only"],
        default=None,
        help="the tier for the folders given (default: asked, or local-only)",
    )
    mg = st.add_mutually_exclusive_group()
    mg.add_argument(
        "--models",
        dest="models",
        action="store_true",
        default=None,
        help="download the missing search models without asking",
    )
    mg.add_argument("--no-models", dest="models", action="store_false", help="download nothing")
    st.add_argument("--no-sync", action="store_true", help="add folders but do not index yet")
    st.add_argument(
        "--yes",
        action="store_true",
        help="ask nothing: take the safe defaults (local-only, no downloads)",
    )
    sy = sub.add_parser("sync", help="bring the index up to date")
    sy.add_argument("--scheduled", action="store_true",
                    help="as a schedule runs it: skip on low battery or heat")  # fmt: skip
    se = sub.add_parser("search", help="search the index from the terminal")
    se.add_argument(
        "query",
        nargs="+",
        help="words, a question or a name; '*' (quoted) lists files newest first",
    )
    rc = sub.add_parser("recent", help="the files changed most recently (search's filters apply)")
    for p_ in (se, rc):
        p_.add_argument("-k", type=int, default=8)
        p_.add_argument(
            "--no-rerank",
            action="store_true",
            help="keep the search's own order (faster: no reranking model to load)",
        )
        p_.add_argument("--tier", choices=["open", "all"], default=None,
                        help="all, or open (the default inside Claude Code)")  # fmt: skip
        p_.add_argument("--path", default=None, help="only files under this path prefix")
        p_.add_argument(
            "--type",
            action="append",
            default=[],
            metavar="EXT",
            help="only files with this extension (repeat, or comma-separate: pdf,md)",
        )
        p_.add_argument(
            "--since",
            default=None,
            metavar="WHEN",
            help="only files modified since: YYYY-MM-DD, or 30d, 2w, 6m, 1y",
        )
        p_.add_argument(
            "--json",
            action="store_true",
            help="results as JSON (path, location, tier, score, snippet)",
        )
        p_.add_argument("--before", default=None, metavar="WHEN",
                        help="only files modified before: YYYY-MM-DD")  # fmt: skip
    sv = sub.add_parser("serve", help="run the MCP server over stdio (started by a client)")
    sv.add_argument("--tier", choices=["open", "all"], required=True,
                    help="open: for cloud assistants; all: only for a local client")  # fmt: skip
    sv.add_argument("--allow-cloud-client", action="store_true",
                    help="let --tier all answer a self-declared cloud assistant")  # fmt: skip
    sc = sub.add_parser("install-schedule", help="print a launchd or systemd schedule to install")
    sc.add_argument("--every", type=int, default=3600, help="seconds between syncs")
    sub.add_parser("mcp-setup", help="print the commands that register the MCP server")
    sub.add_parser("doctor", help="check config, index, plugins, models and sandbox")
    ui = sub.add_parser("ui", help="open the settings page (optional package lanternkb-ui)")
    ui.add_argument("--no-browser", action="store_true", help="print the link only")
    sub.add_parser("status", help="index counts by tier, skip reason and last sync")
    sub.add_parser("skipped", help="every file not indexed, with the reason")
    args = parser.parse_args(argv)
    use_config(None if args.config == DEFAULT_CONFIG else args.config)
    if args.cmd == "serve" and redacting:
        print("lanternkb: --redact is for commands people paste into reports, not serve",
              file=sys.stderr)  # fmt: skip
        return 2
    if args.cmd == "serve":  # the MCP server never loads the control API (threat model T10)
        from lanternkb.server import main as serve

        return serve(args.config, args.tier, args.allow_cloud_client)
    from lanternkb.config import ConfigError
    from lanternkb.control import ControlError
    from lanternkb.features import ManifestError

    handlers = {"scan": cmd_scan, "explain": cmd_explain, "features": cmd_features,
                "roots": cmd_roots, "sync": cmd_sync, "status": cmd_status,
                "skipped": cmd_skipped, "models": cmd_models, "search": cmd_search,
                "recent": cmd_recent,
                "setup": cmd_setup,
                "install-schedule": cmd_install_schedule, "mcp-setup": cmd_mcp_setup,
                "ui": cmd_ui, "doctor": cmd_doctor}  # fmt: skip
    try:
        return handlers[args.cmd](args)
    except (ControlError, ConfigError, ManifestError) as err:
        print(f"lanternkb: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
