# Security policy


## Reporting a vulnerability

Report privately, never in a public issue:
- **GitHub:** the repository's **Security** tab, **Report a vulnerability** (private vulnerability reporting).

Please include:
- the version or commit, your OS, and how lanternkb was installed;
- a reproduction built from **synthetic files only** (never your real documents, paths or credentials; fake secrets clearly labelled);
- what an attacker gains: for example local-only text reaching an open-tier result, a secret reaching the index unredacted, or a sandboxed extractor reading a file it was not handed.

lanternkb is maintained by one person. Expect an acknowledgement within a week; fixes for issues that expose local-only text or secrets come first.

## What counts

The [threat model] says what lanternkb protects (T1 to T11) and what it does not claim. A way to break a stated mitigation is a vulnerability. Something in "Not claimed" is not, unless the documentation says otherwise somewhere.

## Principles

- No telemetry and no network calls in the core. The only network use is `lanternkb models pull`, run by hand.
- Never log document text, secrets or queries.
- Test fixtures are synthetic, and fake secrets are labelled.
- Every fix comes with a test that fails without it.
