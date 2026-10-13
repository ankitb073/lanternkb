"""Text for the user to review and install: a sync schedule and MCP registrations.

Nothing here installs, registers or starts anything. The user
saves the schedule and runs the printed commands themselves.
"""

from __future__ import annotations

import json
import os
import plistlib
import shlex
import sys
from pathlib import Path

LABEL = "com.lanternkb.sync"
MIN_EVERY = 300


def command() -> list[str]:
    """How to run lanternkb from a schedule or an MCP client: an absolute path, so neither
    depends on the PATH of a login shell."""
    script = Path(sys.argv[0])
    if script.name == "lanternkb" and script.exists():
        return [str(script.resolve())]
    # -I: never import modules from the working folder (a client's project) or PYTHON*.
    return [sys.executable, "-I", "-m", "lanternkb.cli"]


def _systemd_word(arg: str, exec_line: bool = True) -> str:
    """One value for a systemd unit, quoted with double quotes (systemd does not join
    shell-style quoted pieces), with specifiers (%) escaped everywhere and variables ($)
    escaped on exec lines, the only place systemd expands them."""
    escaped = arg.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    return '"' + (escaped.replace("$", "$$") if exec_line else escaped) + '"'


def _env() -> dict[str, str]:
    keep = ("LANTERNKB_MODELS_DIR", "LANTERNKB_AUDIO_MODEL")
    return {k: os.environ[k] for k in keep if os.environ.get(k)}


def render(config: Path, every: int, platform: str = sys.platform) -> tuple[str, str]:
    """(file text, how to install it) for this platform's scheduler."""
    every = max(MIN_EVERY, every)
    cfg = str(config.absolute())
    args = [*command(), "--config", cfg, "sync", "--scheduled"]
    printer = shlex.join([*command(), "--config", cfg, "install-schedule", "--every", str(every)])
    if platform == "darwin":
        logs = Path.home() / "Library" / "Logs" / "lanternkb"
        plist = {
            "Label": LABEL,
            "ProgramArguments": args,
            "StartInterval": every,
            "RunAtLoad": False,
            "ProcessType": "Background",  # the OS schedules it as background work
            "LowPriorityIO": True,
            "Nice": 10,
            "StandardOutPath": str(logs / "sync.log"),  # paths and counts only, never text
            "StandardErrorPath": str(logs / "sync.log"),
        }
        if _env():
            plist["EnvironmentVariables"] = _env()
        target = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
        t = shlex.quote(str(target))
        how = "\n".join([
            "Review the text above, then to install it:",
            f"  mkdir -p {shlex.quote(str(logs))}",
            f"  {printer} > {t}",
            f"  launchctl bootstrap gui/$(id -u) {t}",
            f"To remove it: launchctl bootout gui/$(id -u)/{LABEL} && rm {t}",
        ])  # fmt: skip
        return plistlib.dumps(plist).decode(), how
    env_lines = "".join(f"Environment={_systemd_word(f'{k}={v}', exec_line=False)}\n"
                        for k, v in _env().items())  # fmt: skip
    exec_line = " ".join(_systemd_word(a) for a in args)
    service = ("[Unit]\nDescription=lanternkb sync\n\n[Service]\nType=oneshot\n"
               f"ExecStart={exec_line}\n{env_lines}"
               "Nice=10\nIOSchedulingClass=idle\n")  # fmt: skip
    timer = ("[Unit]\nDescription=lanternkb sync schedule\n\n[Timer]\n"
             f"OnBootSec=10min\nOnUnitActiveSec={every}s\n\n"
             "[Install]\nWantedBy=timers.target\n")  # fmt: skip
    text = f"# lanternkb-sync.service\n{service}\n# lanternkb-sync.timer\n{timer}"
    how = ("Review the text above (made by: " + printer + "), then save the two units in "
           "~/.config/systemd/user/ as lanternkb-sync.service and lanternkb-sync.timer, and run:\n"
           "  systemctl --user daemon-reload\n"
           "  systemctl --user enable --now lanternkb-sync.timer")  # fmt: skip
    return text, how


MAX_LOG_BYTES = 1_000_000


def rotate_log(path: Path | None = None) -> None:
    """Keep a sync log to about MAX_LOG_BYTES: the current file and one older one. Logs
    hold paths and counts, never text, but paths are private too."""
    path = path or Path.home() / "Library" / "Logs" / "lanternkb" / "sync.log"
    try:
        if path.stat().st_size > MAX_LOG_BYTES:
            os.replace(path, path.with_name(path.name + ".1"))
    except OSError:
        pass


def mcp_commands(config: Path) -> str:
    base = [*command(), "--config", str(config.absolute()), "serve", "--tier"]
    private = {"mcpServers": {"lanternkb-private": {"command": base[0],
                                                    "args": [*base[1:], "all"]}}}  # fmt: skip
    return "\n".join([
        "# Claude Code, or any assistant that sends results to the cloud: open files only.",
        f"claude mcp add --scope user lanternkb -- {shlex.join([*base, 'open'])}",
        "",
        "# A client whose model runs on this machine only (for example LM Studio): every tier.",
        "# Never register this entry with an assistant that sends results to the cloud.",
        json.dumps(private, indent=2),
        "",
        *_deny_rules(config),
    ])  # fmt: skip


def _claude_path(p: Path) -> str:
    """A path as Claude Code's permission rules write it: ~/ under home, else // + path."""
    home = Path.home()
    if p == home or home in p.parents:
        return "~/" + p.relative_to(home).as_posix()
    return "/" + p.as_posix()


def _deny_rules(config: Path) -> list[str]:
    """Tiers bound what the MCP server returns. An assistant that can also run commands and
    read files (Claude Code can) could read local-only files directly, run lanternkb's CLI,
    or edit the config: Claude Code's own deny rules close those doors for its tools."""
    from lanternkb.config import load

    rules = []
    try:
        cfg = load(config) if config.is_file() else None
    except Exception:  # noqa: BLE001 - suggestions only; mcp-setup must still print
        cfg = None
    if cfg is not None:
        rules += [f"Read({_claude_path(r.path)}/**)" for r in cfg.roots
                  if r.tier.value == "local-only"]  # fmt: skip
        rules.append(f"Read({_claude_path(cfg.index_path.parent)}/**)")
    rules += [f"Edit({_claude_path(config.absolute())})", "Bash(lanternkb *)",
              "Bash(uv run lanternkb *)"]  # fmt: skip
    return [
        "# Optional, for Claude Code: keep its own tools away from local-only files, the index",
        "# and this config (tiers limit only what the MCP server returns). Claude Code then",
        "# cannot run lanternkb itself; you still can, in your own terminal. Add these to",
        '# "permissions": {"deny": [...]} in ~/.claude/settings.json. They stop Claude Code\'s',
        "# Read, Edit, Grep and Glob tools and these commands; a script it writes could still",
        "# read files, so keep confidential folders local-only and out of its working folder.",
        json.dumps({"permissions": {"deny": rules}}, indent=2),
    ]
