# CIRCLES-era archive (MIG-03, 2026-09-24)

Superseded code from the damping-to-moving-wall escalation phase (OE-series), the
paused PPO pilots and earlier wall variants, relocated unchanged from the active tree.

- `traffic_shepherd/`: 63 modules (still importable as `traffic_shepherd.<name>` for
  historical replay; active modules must not import them).
- `tests/`: the 75 tests that exercised them. They are kept as evidence and are not
  collected by the default `pytest` run; some assume their original location.
- `docs/`: legacy check/how-to documents for those studies.
- `tools/`: three one-off builders/reviewers for those studies.

[`PATH_MAP.json`](PATH_MAP.json) lists every old path, new path, size and SHA-256, so
frozen ledger records keep resolving. Record: [MIG-03 report](../../output/repository_cleanup_v1/REPORT.md).
