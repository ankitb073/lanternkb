"""Energy checks for scheduled syncs.

A scheduled sync skips itself on low battery, in Low Power Mode or under thermal pressure,
and runs at background priority. macOS is read through `pmset`; on Linux the battery is
read from /sys. Reading power state needs no permission. A check that cannot be read
never blocks a sync.
"""

from __future__ import annotations

import contextlib
import os
import re
import subprocess
import sys
from pathlib import Path

BATTERY_MIN = 30  # percent; below this, on battery power, a scheduled sync waits


def reason_to_wait() -> str | None:
    """Why a scheduled sync should not run now, or None."""
    if sys.platform == "darwin":
        return _mac(_run(["pmset", "-g"]), _run(["pmset", "-g", "batt"]),
                    _run(["pmset", "-g", "therm"]))  # fmt: skip
    if sys.platform.startswith("linux"):
        return _linux(Path("/sys/class/power_supply"))
    return None


def lower_priority() -> None:
    with contextlib.suppress(OSError):
        os.nice(10)


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _mac(general: str, battery: str, thermal: str) -> str | None:
    if re.search(r"^\s*(?:lowpowermode|powermode)\s+1\b", general, re.M):  # 1 = Low Power
        return "Low Power Mode is on"
    if "Battery Power" in battery:
        level = re.search(r"(\d+)%", battery)
        if level and int(level.group(1)) < BATTERY_MIN:
            return f"on battery at {level.group(1)}%"
    limit = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", thermal)
    if limit and int(limit.group(1)) < 100:
        return f"the CPU is slowed for heat ({limit.group(1)}%)"
    warning = re.search(r"thermal warning level[^0-9\n]*(\d+)", thermal, re.I)
    if warning and int(warning.group(1)) > 0:
        return "the Mac reports thermal pressure"
    return None


def _linux(supplies: Path) -> str | None:
    for supply in sorted(supplies.glob("*")):
        try:
            if (supply / "type").read_text().strip() != "Battery":
                continue
            status = (supply / "status").read_text().strip()
            capacity = int((supply / "capacity").read_text().strip())
        except (OSError, ValueError):
            continue
        if status == "Discharging" and capacity < BATTERY_MIN:
            return f"on battery at {capacity}%"
    return None
