"""Visibility tiers: which files a server process may return.

A file is `open` (may be shown to a cloud assistant) or `local-only` (only a model on this
machine may read it). A server runs with `--tier open` or `--tier all`, fixed at start-up
. New sources default to `local-only`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Protocol

from lanternkb import globs


class Tier(StrEnum):
    OPEN = "open"
    LOCAL_ONLY = "local-only"


class ServerTier(StrEnum):
    OPEN = "open"  # sees only `open` files
    ALL = "all"  # sees everything; for local clients only


DEFAULT_TIER = Tier.LOCAL_ONLY


def visible(file_tier: Tier | str, server_tier: ServerTier | str) -> bool:
    """Whether a server started with server_tier may return a file of file_tier.

    Unknown values are treated as the most restrictive case.
    """
    if server_tier == ServerTier.ALL:
        return True
    return server_tier == ServerTier.OPEN and file_tier == Tier.OPEN


def stricter(a: Tier, b: Tier) -> Tier:
    return Tier.LOCAL_ONLY if Tier.LOCAL_ONLY in (a, b) else Tier.OPEN


@dataclass(frozen=True)
class PathTierRule:
    """A glob over the path relative to its root, e.g. `finance/` -> local-only.

    Rules that make files local-only ignore case (so `finance/` also covers `Finance/` on a
    case-insensitive volume); rules that make files open match case exactly. Either way a
    mismatch fails closed.
    """

    pattern: str
    tier: Tier

    def matches(self, rel_path: PurePosixPath) -> bool:
        return globs.match(self.pattern, rel_path.as_posix(), self.tier == Tier.LOCAL_ONLY)


def path_tier(root_tier: Tier, rel_path: PurePosixPath, rules: tuple[PathTierRule, ...]) -> Tier:
    """Start from the root's tier; path rules apply in order and the last match wins."""
    tier = root_tier
    for rule in rules:
        if rule.matches(rel_path):
            tier = rule.tier
    return tier


class ContentTierRule(Protocol):
    """A rule that looks at a file after scanning and may only make it stricter."""

    api_version: int
    name: str
    needs_text: bool  # False if it decides from n_secrets and the path alone

    def tier_for(self, rel_path: PurePosixPath, text: str, n_secrets: int) -> Tier | None:
        """Return LOCAL_ONLY to restrict the file, or None to leave it alone."""
        ...


class SecretsMakeLocalOnly:
    """Built-in: a file in which the scanner found a secret becomes local-only."""

    api_version = 1
    name = "secrets-make-local-only"
    needs_text = False

    def tier_for(self, rel_path: PurePosixPath, text: str, n_secrets: int) -> Tier | None:
        return Tier.LOCAL_ONLY if n_secrets > 0 else None


def content_tier(
    tier: Tier,
    rel_path: PurePosixPath,
    text: str,
    n_secrets: int,
    rules: tuple[ContentTierRule, ...],
) -> Tier:
    """Apply content rules; they can only move a file towards local-only. A rule's answer
    other than None, a falsy value or Tier.OPEN (which cannot loosen anything) counts as
    local-only: a rule that says "restrict" in the wrong type still restricts."""
    for rule in rules:
        verdict = rule.tier_for(rel_path, text, n_secrets)
        if verdict and verdict != Tier.OPEN:
            tier = Tier.LOCAL_ONLY
    return tier
