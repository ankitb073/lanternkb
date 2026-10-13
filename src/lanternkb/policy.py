"""Path policy: decide what may be read, from the path and `stat` alone, before reading it.

Nothing is read unless it is inside an allowed root, passes the deny lists (checked against
the path and its resolved real path, ignoring case), is a regular file with one link, is
handled by an *enabled* extractor feature, and is under the size cap. A file's tier is the
stricter of its path tier and its feature's tier. Every refusal carries a reason, so nothing
silently vanishes (`lanternkb skipped` lists them).
"""

from __future__ import annotations

import os
import stat as stat_
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from lanternkb import globs
from lanternkb.config import Config, Root
from lanternkb.features import Registry
from lanternkb.fsutil import canonical_case
from lanternkb.tiers import Tier, path_tier, stricter

# Credentials, keys, history, mail, browsers, version-control internals: denied wherever
# they appear in the absolute path, whatever the root, ignoring case. A root cannot override.
SENSITIVE_DENY: tuple[str, ...] = (
    ".env*", ".ssh/", ".gnupg/", ".aws/", ".azure/", ".kube/", ".docker/", ".password-store/",
    ".config/gh/", ".config/gcloud/", ".config/rclone/", ".config/github-copilot/",
    ".config/configstore/", ".terraform.d/", ".pulumi/", ".mc/", ".cargo/credentials*",
    ".m2/settings.xml", ".dbt/profiles.yml", ".terraformrc", "Library/Keychains/",
    "*.pem", "*.key", "*.p12", "*.pfx", "*.p8", "*.kdbx", "*.keystore", "*.jks", "*.ppk",
    "*.ovpn", "*.tfstate", "*.tfstate.backup", "id_rsa*", "id_dsa*", "id_ecdsa*", "id_ed25519*",
    ".netrc", ".pgpass", ".npmrc", ".pypirc", ".git-credentials", "credentials.json",
    "wp-config.php", "secring.*", ".zshrc", ".zshenv", ".zprofile", ".bashrc",
    ".bash_profile", ".profile",
    ".bash_history", ".zsh_history", ".zhistory", ".python_history", ".node_repl_history",
    ".psql_history", ".mysql_history", ".sqlite_history", ".lesshst", ".viminfo",
    "fish_history",
    "Library/Mail/", "Library/Messages/", "Library/Cookies/", "Library/Safari/",
    "Library/Application Support/Google/Chrome/", "Library/Application Support/Chromium/",
    "Library/Application Support/BraveSoftware/", "Library/Application Support/Microsoft Edge/",
    "Library/Application Support/Arc/", "Library/Application Support/Firefox/", ".mozilla/",
    ".config/google-chrome/", ".config/chromium/",
    ".git/", ".hg/", ".svn/",
)  # fmt: skip
# Dependencies and build output: noise, not secrets. Denied only below the root, so a root
# that happens to live inside a folder called `build` still works.
NOISE_DENY: tuple[str, ...] = (
    "node_modules/", ".venv/", "venv/", "__pycache__/", "site-packages/", "dist/", "build/",
    ".tox/", ".mypy_cache/", ".pytest_cache/", ".next/", ".gradle/", "Pods/", "DerivedData/",
    ".Trash/",
)  # fmt: skip
DENY = SENSITIVE_DENY + NOISE_DENY
SF_DATALESS = 0x40000000  # macOS: file contents not on this machine (iCloud and similar)


@dataclass(frozen=True)
class Decision:
    path: Path  # absolute, as found (not resolved)
    allowed: bool
    reason: str | None = None  # why it was refused, e.g. "denied:*.pem"
    root: Root | None = None
    rel: PurePosixPath | None = None  # relative to its root
    tier: Tier | None = None
    size: int | None = None
    mtime_ns: int | None = None
    real_path: Path | None = None
    dev: int | None = None
    ino: int | None = None
    feature: str | None = None  # the extractor feature that handles it


def _stat(path: Path) -> os.stat_result:
    return os.stat(path)


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


class PathPolicy:
    def __init__(self, config: Config, registry: Registry | None = None):
        resolved = tuple(replace(r, path=canonical_case(Path(os.path.realpath(r.path))))
                         for r in config.roots)  # fmt: skip
        if resolved != config.roots:  # roots built in code may be unresolved
            config = replace(config, roots=resolved)
        self.config = config
        self.registry = registry or Registry(settings=config.features)
        self.user_rel = tuple(d for d in config.deny if not d.startswith("abs:"))
        self.user_abs = tuple(d[4:] for d in config.deny if d.startswith("abs:"))
        self._sensitive = globs.PatternSet(SENSITIVE_DENY, ignore_case=True, suffix=True)
        self._below_root = globs.PatternSet(NOISE_DENY + self.user_rel, ignore_case=True,
                                            suffix=False)  # fmt: skip
        self._real_roots = [(r, Path(os.path.realpath(r.path))) for r in config.roots]
        # lanternkb's own files are never indexed, wherever they sit
        own = [config.index_path] + ([config.source] if config.source else [])
        self._own = {canonical_case(Path(os.path.realpath(f))) for f in own}
        self._own |= {Path(str(f) + s) for f in list(self._own) for s in ("-wal", "-shm", ".lock")}

    def denied_by(self, path: Path, is_dir: bool = False) -> str | None:
        """The first deny pattern matching this absolute path, or None. Ignores case.

        A folder is tested as if it had contents, so `node_modules/` matches the folder itself.
        """
        posix = path.as_posix() + ("/-" if is_dir else "")
        if pat := self._sensitive.first(posix):
            return pat
        for pat in self.user_abs:
            if globs.match(pat, posix.lstrip("/"), ignore_case=True):
                return pat
        # Relative deny patterns apply below every root containing the path, so adding a
        # nested root never lifts a pattern written for the outer one.
        for root in self._roots_containing(path):
            if path != root.path:
                rel = path.relative_to(root.path).as_posix() + ("/-" if is_dir else "")
                if pat := self._below_root.first(rel):
                    return pat
        return None

    def _roots_containing(self, path: Path) -> list[Root]:
        """Every root containing the path, deepest first."""
        found = [r for r in self.config.roots if _within(path, r.path)]
        return sorted(found, key=lambda r: len(r.path.parts), reverse=True)

    def _root_of(self, path: Path) -> Root | None:
        """The deepest root containing the path (roots are unique after config merging)."""
        best = None
        for root in self.config.roots:
            if _within(path, root.path) and (
                best is None or len(root.path.parts) > len(best.path.parts)
            ):
                best = root
        return best

    def _tier_at(self, path: Path) -> Tier | None:
        """The tier a path gets from its own root and the tier rules, or None if outside."""
        root = self._root_of(path)
        if root is None:
            for r, real_root in self._real_roots:
                if _within(path, real_root):
                    rel = PurePosixPath(path.relative_to(real_root).as_posix())
                    return path_tier(r.tier, rel, self.config.tier_rules)
            return None
        return self._path_tier(path, root)

    def _path_tier(self, path: Path, root: Root) -> Tier:
        """The deepest root sets the base tier, but a local-only rule written relative to an
        enclosing root still applies: nesting a root never loosens an outer restriction."""
        rel = PurePosixPath(path.relative_to(root.path).as_posix())
        tier = path_tier(root.tier, rel, self.config.tier_rules)
        for outer in self._roots_containing(path)[1:]:
            rel_outer = PurePosixPath(path.relative_to(outer.path).as_posix())
            if any(r.tier == Tier.LOCAL_ONLY and r.matches(rel_outer)
                   for r in self.config.tier_rules):  # fmt: skip
                tier = Tier.LOCAL_ONLY
        return tier

    def check(self, path: Path, *, canonical: bool = False, not_link: bool = False) -> Decision:
        """Decide one path. Reads metadata only, never content.

        Paths from outside the walk (for example a client asking for an excerpt) are first
        spelt as on disk, so `.SSH/x` is judged as `.ssh/x`. The walk passes canonical=True,
        and not_link=True for entries it knows are not symlinks (it never enters symlinked
        folders and roots are resolved, so such a path is its own real path).
        """
        p = Path(os.path.abspath(path))
        if not canonical:
            # Spell it as on disk: case, and the folders above it resolved (/var is
            # /private/var on macOS), as roots are. The last part stays as given, so a link
            # is still judged as a link.
            p = canonical_case(Path(os.path.realpath(p.parent)) / p.name)
        root = self._root_of(p)
        if root is None:
            return Decision(p, False, "outside-roots")
        rel = PurePosixPath(p.relative_to(root.path).as_posix())
        if pat := self.denied_by(p):
            return Decision(p, False, f"denied:{pat}", root, rel)
        real = p if (canonical and not_link) else canonical_case(Path(os.path.realpath(p)))
        if real in self._own or p in self._own:
            return Decision(p, False, "lanternkb-own-file", root, rel)
        tier = self._path_tier(p, root)
        if real != p:
            target_tier = self._tier_at(real)
            if target_tier is None:
                return Decision(p, False, "symlink-escape", root, rel)
            if pat := self.denied_by(real):
                return Decision(p, False, f"denied:{pat}", root, rel)
            tier = stricter(tier, target_tier)  # a link never loosens its target's tier
        try:
            st = _stat(p)  # follows a symlinked file, whose target was checked above
        except OSError:
            return Decision(p, False, "unreadable", root, rel)
        if not stat_.S_ISREG(st.st_mode):
            return Decision(p, False, "not-a-file", root, rel)
        if getattr(st, "st_flags", 0) & SF_DATALESS:
            # An iCloud placeholder: reading it would make macOS download the file, outside
            # the sandbox and without the user asking. Skip it.
            return Decision(p, False, "cloud-placeholder", root, rel)
        if st.st_nlink > 1:
            return Decision(p, False, "hard-link", root, rel)
        # The content's type decides the feature; a symlink must also pass the feature its
        # own name belongs to, so a link named like a note cannot expose disabled code.
        feature = self.registry.extractor_for(real)
        if feature is None:
            return Decision(p, False, "no-extractor", root, rel, size=st.st_size)
        state = self.registry.state(feature.id)
        alias = self.registry.extractor_for(p) if real != p else None
        if alias is not None and alias.id != feature.id:
            alias_state = self.registry.state(alias.id)
            if not alias_state.enabled:
                feature, state = alias, alias_state
            else:
                tier = stricter(tier, alias_state.tier)
        if not state.enabled:  # disabled means free: nothing is read
            reason = f"feature-disabled:{feature.id}"
            return Decision(p, False, reason, root, rel, size=st.st_size,
                            mtime_ns=st.st_mtime_ns, feature=feature.id)  # fmt: skip
        if st.st_size > self.config.limits.max_file_bytes:
            return Decision(p, False, "too-large", root, rel, size=st.st_size, feature=feature.id)
        tier = stricter(tier, state.tier)
        return Decision(p, True, None, root, rel, tier, st.st_size, st.st_mtime_ns, real,
                        st.st_dev, st.st_ino, feature.id)  # fmt: skip

    def walk(self) -> Iterator[Decision]:
        """Every file under every root, in a stable order.

        Denied folders are reported once and not entered. Symlinked folders are reported
        and not followed. A file reached twice (two roots, or a symlink to a file that is
        also walked directly) is reported once; later sightings are `duplicate`.
        """
        seen: set[Path] = set()
        seen_files: set[tuple[int, int]] = set()
        for root in self.config.roots:
            stack = [root.path]
            while stack:
                folder = stack.pop()
                if folder != root.path and canonical_case(Path(os.path.realpath(folder))) != folder:
                    # It was a folder when its parent was listed; now it (or a parent) is a
                    # link. Do not trust the walk's no-symlink shortcut inside it.
                    yield Decision(folder, False, "changed-during-sync", root)
                    continue
                try:
                    with os.scandir(folder) as it:
                        entries = sorted(it, key=lambda e: e.name)
                except OSError:
                    yield Decision(folder, False, "unreadable", root)
                    continue
                subdirs = []
                for e in entries:
                    p = Path(e.path)
                    if p in seen:
                        continue
                    seen.add(p)
                    try:
                        is_dir = e.is_dir(follow_symlinks=False)
                        is_link = e.is_symlink()
                    except OSError:
                        yield Decision(p, False, "unreadable", root)
                        continue
                    if is_dir:
                        if pat := self.denied_by(p, is_dir=True):
                            yield Decision(p, False, f"denied:{pat}", root)
                        else:
                            subdirs.append(p)
                    elif is_link and p.is_dir():
                        yield Decision(p, False, "symlink-folder-not-followed", root)
                    else:
                        d = self.check(p, canonical=True, not_link=not is_link)
                        if d.allowed and d.dev is not None:
                            key = (d.dev, d.ino or 0)
                            if key in seen_files:
                                d = Decision(p, False, "duplicate", d.root, d.rel)
                            seen_files.add(key)
                        yield d
                stack.extend(reversed(subdirs))
