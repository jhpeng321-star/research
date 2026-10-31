# S-COOPMEASURE — artifact-HOLD handoff

## Current v2 recovery (2026-11-01; supersedes HOLD status below)
Upstream ARTIFACT_CHECK now PASS for full-prefix reruns only; bound in61e26353.
Corrected source4f6b7332, registered release239a4726, old exact stopped worker
retired without resume. Latest final trained checkpointu15 is paired with MAPPO
and wall. CPU26/26PASS. Corrected full-prefix canary16/16 PASS, zero safety
events, wall C_G600+1658.239758, MAPPO/SUMOCF C_G0; no panel inference yet.
Full fixed panel207900sim-s is admitted under the243780combined charged maximum.
Do not dispatch a second canary or main: inspect `v2-canary_PID.json`,
`v2-main_PID.json`, the one-shot launch markers and `v2_status.py` first.
`finalize_v2.py --start` is the one-shot LOCAL completion observer, not a SUMO
dispatcher. Once its V2_FINISHER_ONCE/PID receipts exist, never start another.
It waits for the existing main, verifies process exit/lock/budget, exports exact
receipts/traces, commits step4, computes conditional/world CIs, independently
checks algebra/bootstrap/coverage, publishes <=60line report+registry and commits
step5. It refuses changed HEAD, other work or pre-staged files. Inspect
V2_FINISHER_PROGRESS.json / V2_FINISHER_FAILURE.json before manual intervention.
No simulator job is retried, migrated or launched by this observer. It can wait
up to24h on the unchanged spare-core gate; networking retries are read-only.
Raw namespace `/root/autodl-tmp/coopaudit20261101_v2` with links under old jobs;
ledger remains `/root/coopaudit20261031/accounting.jsonl`. Original data preserved.

## Historical HOLD handoff

The audit is **paused**, not complete. `HOLD_VERIFIED.json` is the latest proof.
No poller/dispatch was restarted in the recovery session. Do not SIGCONT the old
worker: its immutable code still contains the superseded five-arm/grid6 loop.

## Preserved work and budget
- Branch `c4/coopmeasure`; original commits through22e1cdfb retained.
- Canary969699 passed native live/load/repeat and wall/MAPPO repeated branches.
- All28 census prefixes complete; first old-design world969650 complete.
- Actual21180; precharged21480 of500000sim-s; setup-only300 remains charged.
- Own continuation39620 stopped after command/start-time verification.
- Own gpuR worker516243/startticks4206436595 is stopped (`T`) at an idle boundary;
  `YIELD` and `ARTIFACT_HOLD.json` remain set. No partial world is awaiting retry.
- No peer process was signalled; CPU-only tests consumed no SUMO budget.

## Release work (blocked on S-SUMOCF)
1. Read E:/agent-work/c4-sumocf/docs/paper_results/ARTIFACT_CHECK.md and its
   evidence. Located at final recovery check: explicit HOLD, no resume authorised.
   The initially checked root-level file was absent. Require a passed check; a suspected
   artifact or a replacement selection alone is insufficient to release SUMO.
2. Bind the passed artifact result and current authoritative raw-owner-rule SNR
   receipt, not the old CV-ranked `SNR_PROBE.json`. Current historical correction
   selects grid26/all-zone/60s/rho1, but it too is under investigation. `selection`
   now recomputes raw abs(mean DG)/sd rather than trusting a supplied CV score.
3. Migrate to an explicit new design/job namespace while retaining the original
   ledger and every old result/trace. Derive new G from saved census observations
   without any SUMO census rerun. The current runner intentionally refuses stale
   census groups and pre-existing historic main job names. This migration is
   pending the corrected design; no stale result is eligible for the new panel.
4. Implement/hash-bind the delivered trained-policy checkpoint adapter when it
   exists; no trained checkpoint was bound in this recovery. Compare its deployed
   mean on the same worlds/groups/native-release estimand, not probe exploration.
   Primary arms are MAPPO + wall + that policy. PI/CIRCLES/CTRL0 are optional only.
5. Reconcile complete-panel admission with the retained21480sim-s and all planned
   new canaries/prefixes/branches. Reserve3*(|G|+1) branches/world plus B/Brepeat,
   and a second300s prefix/world for the paired trained arm arriving later. No
   world/group truncation, retries or budget refunds. Cap remains500000sim-s.
6. Register/commit the corrected design, release and reviewed migration BEFORE
   dispatch. `ARTIFACT_RELEASE.json` requires PASS, registered provenance, amended
   scope, paired budget reservation, artifact document SHA and selected-design SHA.
   The artifact document needs explicit PASS in its status/title, without HOLD/FAIL. Local release
   reads the source document; remote release reads its exact hash-bound copy.
7. Retire only the identity-verified owned stale idle worker before a new worker
   acquires RUN.lock. Preserve old PID receipts. Issue no release by timer; do not
   remove YIELD/HOLD until both sides bind the reviewed evidence/design. Run CPU
   and applicable corrected-branch canaries; then admit and run the complete panel.

## Outputs and verification
- CPU21/21 PASS; `evidence/test_deploy.json` binds the latest tested source version.
- `evidence/HOLD_VERIFIED.json`: owned worker stopped, no new starts/charges,
  zero additional sim-s since HOLD, no release issued.
- `docs/paper_results/COOPAUDIT.md` is a pending report (<60 lines), mirrored in
  E:/research-dense/docs/paper_results/COOPAUDIT.md. No panel CI is claimed.
- Existing behaviour traces and native-release semantics: `TRACE_HANDOFF.md`.
- The one-world wall canary detects the coordinated mechanism but C_G180=-154.834
  veh-s; it is not evidence of beneficial synergy or a panel-level null.
