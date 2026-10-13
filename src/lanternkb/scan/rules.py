"""Load secret-detection rules from gitleaks-format TOML files.

The default set is gitleaks v8.30.1 (vendored, MIT) plus our own rules in the same format
. Go (RE2) regex syntax that Python's `re` does not accept is converted
when loading; rules that still fail to compile are reported, never silently dropped.

One deliberate difference from gitleaks: its *global path allowlist* (images, PDFs, Office
files, lock files, vendored folders) is ignored. gitleaks skips those files; we scan the
text extracted from PDFs and Office files, so skipping them would be a hole. Per-rule
path allowlists are kept.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

ENGINE_VERSION = 19  # bump when matching semantics change: forces a full re-scan
# Passes of the engine that are not regex rules (2026-10-03); disabled by id like rules.
STRUCTURAL = ("csv-secret-column", "quoted-printable", "base64-block", "hexdump",
              "csv-multiline-cell", "markup-view", "unfold-view", "flowed-view", "wrap-view",
              "json-escape-view", "rtf-view", "nul-view", "registry-hex", "yaml-fold-view",
              "punct-view", "json-backslash-view", "allowlist-resume",
              "kubernetes-secret-data", "separator-join-view", "latex-view",
              "prefix-wrap-view", "emphasis-view", "kubernetes-secret-json",
              "stopword-words")  # fmt: skip
# Off unless enabled (`[scanner] enabled_rules`): round 2 of 2026-10-03 and the
# label-and-value rules. On the third held-out split they kept no more secrets
# out than the default rules, so the plan fixed before that run switched them off
#.
ROUND2 = ("netrc-password", "pgpass-line", "plist-secret", "xml-named-secret", "base64-block",
          "hexdump", "csv-multiline-cell", "otpauth-secret", "roboform-password-field",
          "export-password-line", "keepass-xml-password", "json-password-field")  # fmt: skip
# Round 3: decoding views and two rules, developed on the spent third split. On by
# default since the sixth held-out test: +43 of 2,448 secrets kept out through the pipeline.
# Listed for the record.
ROUND3 = ("markup-view", "unfold-view", "flowed-view", "wrap-view", "json-escape-view",
          "rtf-view", "nul-view", "registry-hex", "registry-login-password",
          "service-password-directive")  # fmt: skip
# Round 4: views, a pass moved from round 2 and rules for indexed formats the fourth
# held-out split showed missing through the pipeline. On by default since the fifth held-out
# test: +31 of 2,450
# secrets kept out through the pipeline, no new false alarms. Listed for the record.
ROUND4 = ("yaml-fold-view", "punct-view", "json-backslash-view", "csv-multiline-cell",
          "allowlist-resume", "python-auth-tuple", "terraform-secret-variable",
          "ansible-inventory-password")  # fmt: skip
ROUND2 = tuple(r for r in ROUND2 if r not in ROUND4)
# Round 5 (2026-10-07): three rules for passwords the spent splits and new file types
# showed missing (Windows command lines and batch files, a password after a few words);
# a candidate until the sixth held-out split measures it.
ROUND5 = ("windows-credential-cli", "setx-secret", "password-after-words")
# Round 6 (2026-10-08): rules, views and passes for indexed formats the spent splits showed
# missing through the pipeline. On by default since the seventh held-out test
#: +36 of 2,455 secrets
# kept out through the pipeline, no new false alarms on clean files. Listed for the record.
ROUND6 = ("aws-secret-key-labelled", "connection-string-password", "libpq-password",
          "password-assignment-exported", "wifi-passphrase", "cloud-init-chpasswd",
          "xml-name-value-secret", "xml-named-secret-text", "latex-secret-macro",
          "helm-literal-b64enc", "secret-named-assignment",
          "secret-named-assignment-unquoted", "filezilla-password", "latex-view",
          "prefix-wrap-view", "emphasis-view", "kubernetes-secret-json",
          "stopword-words")  # fmt: skip
OPT_IN = ROUND2 + ROUND5  # rounds 3 and 6: on by default since the sixth and seventh tests
ACTIONS = ("redact", "skip-file")
SKIP_FILE_RULES = re.compile(r"private-key|service-account-key-file|pkcs12")


@dataclass(frozen=True)
class Allowlist:
    regexes: tuple[re.Pattern[str], ...] = ()
    target: str = "secret"  # "secret" | "match" | "line"
    stopwords: tuple[str, ...] = ()  # lower-case; allowlisted if the secret contains one
    paths: tuple[re.Pattern[str], ...] = ()
    condition: str = "OR"  # "OR": any criterion; "AND": every criterion given


@dataclass(frozen=True)
class Rule:
    id: str
    description: str
    regex: re.Pattern[str] | None
    keywords: tuple[str, ...] = ()  # lower-case; the rule runs only if one occurs
    entropy: float | None = None
    secret_group: int = 0
    path: re.Pattern[str] | None = None
    allowlists: tuple[Allowlist, ...] = ()
    action: str = "redact"
    source: str = "gitleaks"

    def fingerprint(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "regex": self.regex.pattern if self.regex else None,
            "keywords": self.keywords,
            "entropy": self.entropy,
            "group": self.secret_group,
            "path": self.path.pattern if self.path else None,
            "action": self.action,
            "allow": [
                ([r.pattern for r in a.regexes], a.target, a.stopwords,
                 [p.pattern for p in a.paths], a.condition)
                for a in self.allowlists
            ],
        }  # fmt: skip


@dataclass(frozen=True)
class RuleSet:
    rules: tuple[Rule, ...]
    global_allow: Allowlist
    failed: tuple[tuple[str, str], ...]  # (rule id, error) for rules that did not compile
    passes: tuple[str, ...] = STRUCTURAL  # the engine's structural passes that are on
    pass_actions: tuple[tuple[str, str], ...] = ()  # action overrides for passes

    @property
    def hash(self) -> str:
        """Changes whenever matching could change; stored in the index to force re-scans."""
        blob = json.dumps(
            {
                "engine": ENGINE_VERSION,
                "passes": list(self.passes),
                **({"pass_actions": list(self.pass_actions)} if self.pass_actions else {}),
                "rules": [r.fingerprint() for r in self.rules],
                "global": [
                    [p.pattern for p in self.global_allow.regexes],
                    self.global_allow.stopwords,
                ],
            },  # fmt: skip
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


# -- Go (RE2) to Python regex ---------------------------------------------------------------

_BARE_FLAGS = re.compile(r"\(\?([imsU]*)(?:-([imsU]+))?\)")
_POSIX = {
    "alnum": "a-zA-Z0-9", "alpha": "a-zA-Z", "digit": "0-9", "lower": "a-z", "upper": "A-Z",
    "xdigit": "0-9A-Fa-f", "space": r"\s", "word": r"\w", "blank": r" \t",
    "punct": r"!-/:-@\[-`{-~",
}  # fmt: skip
_POSIX_RX = re.compile(r"\[:(" + "|".join(_POSIX) + r"):\]")


def _group_end(p: str, start: int) -> int:
    """Index of the `)` closing the group that contains position start (or len(p))."""
    depth, i, in_class = 0, start, False
    while i < len(p):
        c = p[i]
        if c == "\\":
            i += 2
            continue
        if in_class:
            if c == "]":
                in_class = False
        elif c == "[":
            in_class = True
            if p.startswith("[]", i) or p.startswith("[^]", i):
                i += 2 if p[i + 1] == "]" else 3
                continue
        elif c == "(":
            depth += 1
        elif c == ")":
            if depth == 0:
                return i
            depth -= 1
        i += 1
    return len(p)


def _split_alternatives(p: str) -> list[str]:
    """Split a group body at its top-level `|`."""
    parts, depth, i, start, in_class = [], 0, 0, 0, False
    while i < len(p):
        c = p[i]
        if c == "\\":
            i += 2
            continue
        if in_class:
            if c == "]":
                in_class = False
        elif c == "[":
            in_class = True
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "|" and depth == 0:
            parts.append(p[start:i])
            start = i + 1
        i += 1
    parts.append(p[start:])
    return parts


def _find_bare_flag(p: str) -> re.Match[str] | None:
    """The first `(?i)`-style flag group not inside a character class or after `\\`."""
    i, in_class = 0, False
    while i < len(p):
        c = p[i]
        if c == "\\":
            i += 2
            continue
        if in_class:
            if c == "]":
                in_class = False
        elif c == "[":
            in_class = True
        elif c == "(" and (m := _BARE_FLAGS.match(p, i)):
            return m
        i += 1
    return None


def go_to_python(pattern: str) -> str:
    r"""Convert the RE2 constructs gitleaks uses into Python `re` syntax.

    - `\z` becomes `\Z`; POSIX classes such as `[[:alnum:]]` become `[a-zA-Z0-9]` (Python
      would otherwise read them as a set of literal characters, silently).
    - A bare flag group such as `(?i)` anywhere but the very start applies, in RE2, to the
      rest of the enclosing group. Python only allows that at the start, so the rest of the
      group becomes a scoped group, one per remaining alternative: `a(?i)b|c` inside a group
      becomes `a(?i:b)|(?i:c)`.
    """
    p = _POSIX_RX.sub(lambda m: _POSIX[m.group(1)], pattern.replace(r"\z", r"\Z"))
    for _ in range(50):  # each pass rewrites one flag group
        m = _find_bare_flag(p)
        if m is None or (m.start() == 0 and not m.group(2) and "U" not in m.group(0)):
            if m is not None and m.start() == 0:
                # leading global flag is fine; look for later ones after it
                rest = go_to_python(p[m.end() :]) if _find_bare_flag(p[m.end() :]) else p[m.end() :]
                return p[: m.end()] + rest
            return p
        flags = m.group(1).replace("U", "") + (
            ("-" + m.group(2).replace("U", "")) if m.group(2) else ""
        )
        flags = flags.rstrip("-")
        end = _group_end(p, m.end())
        body = p[m.end() : end]
        if not flags:
            p = p[: m.start()] + body + p[end:]
            continue
        wrapped = "|".join(f"(?{flags}:{alt})" for alt in _split_alternatives(body))
        p = p[: m.start()] + wrapped + p[end:]
    return p


# gitleaks ends most provider tokens with this group: a quote, whitespace, `;`, an escaped
# newline, or the end of the text. Real files also end tokens with `,` (CSV), `)` `>` `]` `}`
# (prose and markup), `|` `&` (URLs, tables), `.` (end of a sentence) and an escaped quote
# (JSON inside JSON, notebooks). Widened at load time; the vendored file is left unchanged.
GITLEAKS_END = r"""(?:[\x60'"\s;]|\\[nr]|$)"""
WIDER_END = r"""(?:[\x60'"\s;,)>\]}|&.]|\\[nr"]|$)"""


# Exact-text patches to the vendored rules, applied at load time (the file stays unchanged).
PATCHES = {
    # Operator precedence bug upstream: `^true|false|null$` allowlists any secret that starts
    # with "true", contains "false" or ends with "null". Intended: the whole value is one word.
    r"(?i)^true|false|null$": r"(?i)^(?:true|false|null)$",
    # Unbounded `.*` before `-H`/`-u` is linear under RE2 but quadratic under Python's `re`
    # on long lines with many "curl"s. Bound the distance from `curl` to the flag.
    r"\bcurl\b(?:.*?|.*?(?:[\r\n]{1,2}.*?){1,5})": r"\bcurl\b(?:.{0,500}?|.{0,500}?(?:[\r\n]{1,2}.{0,500}?){1,5})",  # noqa: E501
    r"\bcurl\b(?:.*|.*(?:[\r\n]{1,2}.*){1,5})": r"\bcurl\b(?:.{0,500}|.{0,500}(?:[\r\n]{1,2}.{0,500}){1,5})",  # noqa: E501
}  # fmt: skip


def _compile(pattern: str) -> re.Pattern[str]:
    """Compile a gitleaks-format pattern the way RE2 would read it.

    `re.ASCII` because RE2's `\\b`, `\\w`, `\\d` and `\\s` are ASCII-only; with Unicode
    semantics a token directly next to a Japanese or accented letter would not match.
    """
    for old, new in PATCHES.items():
        pattern = pattern.replace(old, new)
    return re.compile(go_to_python(pattern.replace(GITLEAKS_END, WIDER_END)), re.ASCII)


# -- loading ----------------------------------------------------------------------------------


def _allowlist(d: dict[str, Any]) -> Allowlist:
    return Allowlist(
        regexes=tuple(_compile(r) for r in d.get("regexes", [])),
        target=d.get("regexTarget", "secret"),
        stopwords=tuple(s.lower() for s in d.get("stopwords", [])),
        paths=tuple(_compile(p) for p in d.get("paths", [])),
        condition=d.get("condition", "OR").upper(),
    )


def load_toml(
    data: dict[str, Any], source: str
) -> tuple[list[Rule], Allowlist, list[tuple[str, str]]]:
    rules, failed = [], []
    for r in data.get("rules", []):
        rid = r["id"]
        try:
            allow = [r["allowlist"]] if "allowlist" in r else []
            allow += r.get("allowlists", [])
            action = r.get("action") or ("skip-file" if SKIP_FILE_RULES.search(rid) else "redact")
            if action not in ACTIONS:
                raise ValueError(f"unknown action {action!r}")
            rules.append(
                Rule(
                    id=rid,
                    description=r.get("description", ""),
                    regex=_compile(r["regex"]) if r.get("regex") else None,
                    keywords=tuple(k.lower() for k in r.get("keywords", [])),
                    entropy=r.get("entropy"),
                    secret_group=int(r.get("secretGroup", 0)),
                    path=_compile(r["path"]) if r.get("path") else None,
                    allowlists=tuple(_allowlist(a) for a in allow),
                    action=action,
                    source=source,
                )
            )
        except (re.error, ValueError, KeyError) as e:
            failed.append((rid, str(e)))
    g = data.get("allowlist", {})
    try:
        # Global *path* allowlist deliberately ignored (see module docstring).
        global_allow = _allowlist(
            {"regexes": g.get("regexes", []), "stopwords": g.get("stopwords", [])}
        )
    except re.error as e:
        failed.append(("<global allowlist>", str(e)))
        global_allow = Allowlist()
    return rules, global_allow, failed


def targeted_allowlists(data: dict[str, Any]) -> list[tuple[tuple[str, ...], Allowlist]]:
    """Top-level `[[allowlists]]` with `targetRules` (gitleaks 8.25's form): extra allowlists
    for rules defined in another file, such as the vendored `generic-api-key`."""
    out = []
    for a in data.get("allowlists", []):
        targets = tuple(a.get("targetRules", []))
        if targets:
            out.append((targets, _allowlist(a)))
    return out


def _merge(a: Allowlist, b: Allowlist) -> Allowlist:
    return Allowlist(a.regexes + b.regexes, "secret", a.stopwords + b.stopwords)


def default_rule_files() -> list[tuple[str, bytes]]:
    base = resources.files("lanternkb.scan") / "rules"
    return [("gitleaks", (base / "gitleaks.toml").read_bytes()),
            ("lanternkb", (base / "lanternkb.toml").read_bytes())]  # fmt: skip


def load_rules(
    extra: list[Path] | None = None,
    disabled: tuple[str, ...] = (),
    actions: dict[str, str] | None = None,
    enabled: tuple[str, ...] = (),
) -> RuleSet:
    """The default rules, plus extra rule files, minus disabled ids and the `OPT_IN` ids not
    `enabled`, with action overrides.

    A later file's rule replaces an earlier one with the same id.
    """
    disabled = tuple(disabled) + tuple(r for r in OPT_IN if r not in enabled)
    sources = default_rule_files() + [(p.stem, p.read_bytes()) for p in (extra or [])]
    by_id: dict[str, Rule] = {}
    global_allow = Allowlist()
    failed: list[tuple[str, str]] = []
    targeted: list[tuple[tuple[str, ...], Allowlist]] = []
    for name, raw in sources:
        data = tomllib.loads(raw.decode())
        rules, allow, bad = load_toml(data, name)
        for rule in rules:
            by_id[rule.id] = rule
        global_allow = _merge(global_allow, allow)
        failed += bad
        try:
            targeted += targeted_allowlists(data)
        except re.error as e:
            failed.append(("<targeted allowlist>", str(e)))
    for targets, al in targeted:
        for rid in targets:
            if rid in by_id:
                rule = by_id[rid]
                by_id[rid] = Rule(**{**rule.__dict__, "allowlists": (*rule.allowlists, al)})
    actions = actions or {}
    out = []
    for rid, rule in by_id.items():
        if rid in disabled:
            continue
        if rid in actions:
            rule = Rule(**{**rule.__dict__, "action": actions[rid]})
        out.append(rule)
    named = set(disabled) | set(actions) | set(enabled)
    known = set(by_id) | set(STRUCTURAL)
    failed += [(rid, "unknown rule id in [scanner] (a typo?)")
               for rid in sorted(named - known)]  # fmt: skip
    passes = tuple(p for p in STRUCTURAL if p not in disabled)
    # only the table pass reports findings under its own id; the others report the rule
    # that matched, with that rule's action
    pass_actions = tuple(sorted((p, a) for p, a in actions.items()
                                if p == "csv-secret-column" and p in passes))  # fmt: skip
    return RuleSet(tuple(out), global_allow, tuple(failed), passes, pass_actions)
