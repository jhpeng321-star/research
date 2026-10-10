# Recovered legacy metastable-feasibility probe

Provenance: `_probe_metastable_feasibility.py` recovered VERBATIM from git
history (commit `8d993e9`, SHA-256 of file
`0d9c4635b98517f0644f2f4ab04535e2242ed786e925bbca4aaa2f3a514930e4` — the
file is identical at its last pre-deletion commit `6f8aad7`). The file was
deleted from the repository root by the 2026-08-27 cleanup
(commit `170ab28`). It is placed here so it is importable by
`run_stage10_wall_probe.py` and by nothing else; no maintained module
imports from this directory.

Dependencies: `evaluate_policy.run_policy_evaluation` (current tree,
unchanged from 8d993e9 except a 4-line diff), `policies.baselines.NoOpPolicy`,
`policies.metastable_rule.HealthRecordingWrapper` (current tree;
records (h, rho, v_mean) per step via the env's `metastable_health_detail()`
accessor — the accessor path the probe relies on).

Recovery note / recorded discrepancy: the driver as recovered contains the
variant tables for stages 1-7 only. The stage-10 table described in
`docs/RESEARCH_OVERVIEW.md` section 7 (which lists `no_op`, `w10_t30_h025`
and the other wall variants) was never committed to this repository — a
full-history grep for `w10_t30`/`h025` finds only naming comments in
`env/wall_supervisor.py` and `policies/collapse_gate.py`, never a variant
row. The stage-10 table in `run_stage10_wall_probe.py` is therefore a
documented reconstruction:

- `no_op`: ("no_op", {}) — identical to the driver's own stage-1 row.
- `w10_t30_h025`: a wall variant with the RUNGS[1] operating point named
  "w10_t30" in `env/wall_supervisor.py` (band 200-400 m, 10 m/s, tau 3.0)
  plus the `_h025` health trigger 0.25 and the legacy default
  release_h = trigger_h + 0.2 = 0.45, a_max_dec 0.3, instant tau. The
  alternative reading (build_inflow_control default band 200-1500 m) is
  recorded but not used: the RUNGS definition is the only in-repo
  authoritative "w10 t30" definition, and it is the one our HRULE arm also
  uses, so the A3 comparison is band-identical.
