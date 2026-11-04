# HANDOFF: c4/harden-int (worktree E:/agent-work/c4-harden-int), 2026-11-03 (INTEGRATION COMPLETE)

RESUME RECEIPT 2026-11-04 (fresh session, W8): STILL NOTHING OUTSTANDING on this branch. Tree clean at
6682295f (docs-only since READY d51051a9). Re-verified on disk: HARDEN_INT_READY.json (ready=true,
commit d51051a9), HARDEN_INT_REPORT.md, rp4 results for all 7 scenarios, h3int2 roots with exactly
8 BANK_QUALITY.json + 16 LESSON_CHECK files - all matching the 11-04 morning receipt counts.
MAIN addenda re-checked through 2026-11-04 16:25 (newer than the previous 02:45 check): 03:46
launch/monitoring, 04:12-04:17 mover/budget, 06:32 V2V_STARTED, 07:12 disk, 07:20 checkpoint_003
symlink fix, 09:00-09:10 monitor release + old-arm stop, 10:12 slim-tapes migration, 12:02 u6
lesson snapshot restore, 12:40 effective-command imitation, 16:03 B u12 recovery, 16:25 cpuQ->cpuS
migration - ALL assigned to the live operator / Claude launch delegates / other branches
(c4/coopnative, c4/effective-imitation, c4/launch-coop, c4/slim-tapes); several carry explicit
"Do NOT ... monitor/report only" scoping. Nothing new for c4/harden-int. No tests or rehearsals
rerun this session: host at 100% CPU with 24 foreign python processes, including a live
rehearsal_cpu (PID 6928, G-V2V-REH dispatch, E:/Temp workdir; not signalled, not duplicated). The
documented 443/62/1 counts stand; cpuQ (Linux) remains the deployment gate and is run by the live
operator. Remaining work stays LIVE ONLY (operator gates at the bottom of this handoff).

RESUME RECEIPT 2026-11-04 (fresh session): STILL NOTHING OUTSTANDING on this branch. Tree clean at
e421afaa (docs-only since READY d51051a9). Re-verified on disk: HARDEN_INT_READY.json (ready=true,
commit d51051a9), HARDEN_INT_REPORT.md, rp4 results for all 7 scenarios
(C:/Users/jpeng/AppData/Local/Temp/h3res/rp4; h3int2 run roots with 8 BANK_QUALITY.json - B+R for
a_clean/b_slow_search/c2/e_malformed - and 16 LESSON_CHECK files), rp/rp2/rp3/rp5/rp6 kept.
MAIN addenda re-checked through 2026-11-04 02:45: everything after 19:14 (19:45 B/R resume,
gate-fix/intent-merged/launch-final/coop-only merges, per_seed_v1, all8_both_seeds backfill,
cooperative_only gate, launch + disk/archival work) is assigned to the c4/coopnative launch
operators and Claude delegates on cpuQ; nothing new for c4/harden-int. No tests or rehearsals
rerun this session: host load 81% with 27 foreign python processes (other agents' live jobs,
not signalled) - the documented 443/62/1 counts stand and cpuQ (Linux) remains the deployment
gate. Remaining work stays LIVE ONLY (operator gates at the bottom of this handoff).

RESUME RECEIPT 2026-11-03 19:2x (+08), fresh session: still NOTHING OUTSTANDING on this branch.
Tree clean at 855484df (no code change since READY d51051a9 - diff is docs only); HARDEN_INT_READY.json
(ready=true), HARDEN_INT_REPORT.md and the rp4/rp5/rp6 result roots re-verified present on disk.
MAIN addenda re-checked through 19:14: second-seed solo replication, bank rule v2 (c4/bank-boot),
imit fixes (c4/imit-fix2) and intent/launcher work are all assigned to the operator and other
branches; nothing new for c4/harden-int. This session deliberately did NOT rerun tests or
rehearsals: the Windows host is running other agents' live jobs (two a_clean rehearsal_cpu runs,
rehearsal_v2v_cpu, three pytest suites and their sumo_verify branch workers) and this host is
documented to produce false stalls/flaky failures under such load; nothing was signalled.
Remaining work stays LIVE ONLY (operator gates below).

RESUME RECEIPT 2026-11-03 18:1x (+08), fresh session: nothing outstanding on this branch. Verified
intact, no code changes: tree clean at f3ad7c78; HARDEN_INT_READY.json (ready=true, commit
d51051a9) and HARDEN_INT_REPORT.md present; rp4 results (7 scenarios) and run-root receipts
verified on disk (h3int2 a_clean/b_slow_search/c2/e_malformed each with 2 BANK_QUALITY + 4
LESSON_CHECK; d_transport_3/f_branch_crash rp4 vintage under h3int4; rp3 kept for the record).
Targeted fresh test receipt on HEAD: replay_bank confirm+lesson+noop_report CPU tests 15 passed
(72.6 s). Full targeted suite NOT rerun this session: host load 87% with ~30 foreign python
processes (other agents; not signalled, per rules) - the documented 443/62/1 counts stand, and
cpuQ (Linux) remains the deployment gate anyway. MAIN addenda after 04:26 (17:27/17:31/17:40/
17:52/17:55/18:00) assign the statistical bank filter, synergy replication, coopfix/intent merges
and V2V work to OTHER branches (c4/bank-stat-filter, c4/coopfix-int, c4/intent-int); nothing new
is assigned to c4/harden-int. Remaining work for this branch stays LIVE ONLY (operator gates below).

READY: HARDEN_INT_READY.json written at 8dd0607c (commit field d51051a9); HARDEN_INT_REPORT.md has
the full scenario table, receipt evidence and test accounting. All 7 rehearsal scenarios completed
on the current code (rp4 batch, roots h3int2, results h3res/rp4): a_clean, b_slow_search,
e_malformed, c2_worker_dies_on_centre on HEAD; d_transport_3 and f_branch_crash also rp4;
c_worker_dies latest at rp6 (pre-amendment, fault handling unaffected). No engineering STOPs.
Test counts at 1d476704: 443 pass / 62 fail / 1 error (breakdown in READY.json; the only non-env
delta is 3 stale budget-expectation failures caused by the registered 22:32 cap amendment
febba5bc - not hand-edited, cpuQ suite is the gate). Remaining work is LIVE ONLY (operator):
cpuQ suite rerun, arrival_fit phase-end wiring, LS layout read-only check, offline bank builder
(MAIN 04:06). Earlier resume notes below are kept for the record.

RESUME STATE (2026-11-03): items (1)-(3) of the original brief were already done (f023f10c, 62b80987);
(D)/(E)/(F) cooperation+exposure (b58ca5b2) and (G) lesson check (4f090b76) are implemented. This
session added:
- c5dccc1b: toy car-following in the rehearsal fake plant (a follower never drives through its
  leader; faster branches can no longer score TTC<1.5 pass-through bursts, so genuinely gentle
  plans reach the admit/cooperation path). Unit-tested (FakeFollowingTests).
- 64b6946d: BANK SAFETY RULE AMENDMENT (MAIN 2026-11-03 04:26, applied identically to B and R):
  (i) collisions/teleports must NOT increase (hard constraints); (ii) the TOTAL of hard-brake +
  TTC<1.5 s events must NOT increase. The OLD verdict (no single metric may increase) is recorded
  per decision (`safety_old_rule_ok`) and disclosed in BANK_QUALITY (`rejections.safety_old_rule`).
  The synergy solo branches now also run as a DIAGNOSTIC on every decision that passes replication
  + later-harm even when the safety check rejects it; its twin row is published tagged with the
  admission verdict (`synergy_diagnostic` count in BANK_QUALITY). Unit tests: the 986016 pattern
  (hard_brake +3, TTC<1.5 s events -92) is admitted by the new rule and disclosed as an old-rule
  rejection; a collision increase is rejected by both rules; a safety-rejected cooperative plan
  still publishes its twin (never imitated).
- REHEARSAL TO COMPLETION (a_clean, rp4 batch, roots C:/Users/jpeng/AppData/Local/Temp/h3int2,
  results C:/Users/jpeng/AppData/Local/Temp/h3res/rp4): B and R complete, no STOP, no stall,
  wall 1258 s. B: 12 decisions / 3 positives / 3 confirmed / 1 admitted; rejections replication 1,
  later_harm 1, safety 2 (= safety_old_rule 2), 0 incomplete; a safety-rejected 2-CAV entry
  published a diagnostic twin. R: 9 decisions / 2 positives / 1 admitted / cooperative=True
  (group of 4: +60.3 first seed vs best solo +44.4, gain +15.9 veh-s). All confirmation branches
  charged in `bank_verification_ext` with kind-prefixed ids (confirm_replication/extended/solo);
  LESSON_LIST freezes on first admitted entry (R: 1 entry; B: empty until its first admission),
  LESSON_CHECK_u004/u005_<arm>.json written after every update (R u5 recapture 9.05, action error
  1.62 m/s^2); per-update receipts embed bank counts and the exposure record (oversampling
  unit-tested; in this run no cooperative entry was admitted before an update read the bank).
  NOTE: the rp3 batch (h3res/rp3) predates the febba5bc charge fix - its budget_blocked
  confirmations were the duplicate-charge bug, already fixed; rp3 is kept for the record only.

The branch is based on harden-2 final (5d71ff01; it contains the harden-3 stall fix 07dd3cd7 via af56d125). No SUMO, GPU or remote work was done. Every rehearsal ran on local CPU with fake SUMO/GPU/WM leaves.

## Done (commits on c4/harden-int)
- ec090d1d, 2a7fc137, de0aecaf: harden-1 cherry-picks (347d3d60, fcb4f1ea, a0f000d7; a0f000d7 already includes 42233748's tests). Duplicate fixes were reconciled:
  - per-query rule: harden-1 semantics, plus harden-2's 'cuda' fatal word;
  - broker: harden-2's slot-local quarantine, plus harden-1's single-STOP `record_stop`;
  - R centre decline: harden-2 format (`unavailable` + `candidate_scores=0`).
- 9bd77016, 7eddf1b8: harden-3 opt-ins (fan-out, lag-1 R) and the CPU rehearsal harness `tools/coopmwm/rehearsal_cpu.py`.
- 5a87a137: B-REPLAY mode (`binding b_replay`). Also:
  - `tools/coopmwm/replay_bank.py`: collector, bank reader, AWR via `exact_grade.imitation_pairs` + `exact_update`;
  - resume config: fan-out 2 and R lag-1 ON in `incident_recovery.install`;
  - `query_errors.plan_shape` decline in `capacity_broker.forecast_result`.
- aaa8ecc3: merge of harden-2 final.
- 65a67132: MAIN decisions (1)-(3):
  - label_list8 top-8, unfiltered, with available_k (`mppi_runtime.label_list8_search`);
  - R uses the same k as matched B decision i;
  - partial branch failure rule in `sumo_verify.choose`, and `verification_grades.reuse` tolerates recorded failed candidates;
  - isolated `receipt_from_search`;
  - per-candidate overlap truncation (mppi_runtime, VerifyController, verified_guide).
  - Also two rehearsal-found B-REPLAY fixes: the quota gate STOP and collector job-id reuse.
- 15542a51: collector searches are blocking admissions, so they are never capacity-declined.
- 5729a269: the rehearsal uses the full registered check panel at u4.
- 62b80987:
  - LS-label reader: `replay_sources` = dirs with `labels/<key>.json`, read as B entries tagged `u3` (live default `/home/ubuntu/coopmwm20261102_lscollect1`);
  - R fairness cap: R uses at most B's bank decisions at the matching (lag-1) B update; shortfall/excess are recorded and R is never padded from B's data;
  - demand-phase admission guard: t0 + 420 s must be <= `demand_phase_end_s`, and exclusions are recorded.
- 4be95198: `verify_per_round` collector knob.
- f023f10c: CONFIRMATION STAGE (MAIN, latest). Per first-seed positive decision, own and LS:
  - (a) replication on an independent CRN seed;
  - (b) extended window to t0+D+600;
  - (c) safety: collisions/teleports/hard_brake, plus TTC<1.5 s events computed with `traffic_shepherd.metrics.closing_ttc` on the branch observable trace.
  - An entry is admitted only if (a), (b) and (c) all hold. Its weight is the existing temperature rule applied to the mean of the first-seed and replication advantages.
  - Outputs: sealed `replay_bank/<arm>/confirmations/*.json` and `BANK_QUALITY.json` per arm (also embedded in UPDATE receipts). Confirmation charges go to phase `sumo_verification`.

## Remaining
1. Confirmation stage (f023f10c): implemented but NOT yet rehearsed to completion. A rehearsal batch started at 22:37 (a_clean, c_worker_dies, d_transport_3, f_branch_crash). Results go to `C:/Users/jpeng/AppData/Local/Temp/h3res/rp3/<scenario>.json`, roots under `C:/Users/jpeng/AppData/Local/Temp/h3int/`.
   - Summarise with `python C:/Users/jpeng/AppData/Local/Temp/h3table.py C:/Users/jpeng/AppData/Local/Temp/h3res/rp3`.
   - Check `replay_bank/*/BANK_QUALITY.json` and `confirmations/` in each root.
   - Known gap: the COOPMWM plant totals have no TTC<1.5 counter (env.py counts only TTC<3). TTC<1.5 is therefore computed from observable rows (gap = leader rear - follower front, no SUMO minGap). MAIN should accept this or add a plant counter. A plant counter changes fingerprints, so it would break prefix replays of existing snapshots.
2. Rehearsal scenario table, B-REPLAY. Evidence before the confirmation stage (rp2/rp runs, same harness):
   - clean, worker-dies, branch-crash and transport-3 all completed B u4 + R u4 with bank AWR imitation; clean also completed B u5.
   - worker death produced a slot quarantine + restart request, no STOP.
   - 3 transport errors were retried with no STOP.
   - No final table with u5 for all scenarios yet. Still to run: e_malformed, c2_worker_dies_on_centre, b_slow_search (`--set latency_s=10`).
   - Run 3-4 at a time. More than that overloads this Windows host (OSError 22 resource exhaustion and false stalls).
3. LS-label reader (62b80987): implemented, not rehearsed against a real LS layout (no LS fixture on CPU). Live check: point `replay_sources` at lscollect1, then call `replay_bank.bank_decisions(root,'B',sources=[...])` and `bank_quality` read-only before training.
4. Demand-phase guard (62b80987): implemented. The live binding must supply `demand_phase_end_s` from the fitted arrival artifact (`observed_phase_end_s`); `incident_recovery` requires it for b_replay. Max look-ahead used: 420 s (final_horizon(300)); screen H300 is smaller.
5. Not yet reported: HARDEN_INT_REPORT.md (scenario table, faults) and HARDEN_3_REPORT.md on c4/harden-3.

## Exact next commands (Windows, rl_driving env, from E:/agent-work/c4-harden-int)
```
set OMP_NUM_THREADS=1 & set MKL_NUM_THREADS=1
python -X utf8 -m tools.coopmwm.rehearsal_cpu <scenario> --workdir <tmp> --set episodes_per_update=2 --set verify_workers=9 --set latency_s=0.5 --set stall_s=900 --set timeout_s=6800 --out <tmp>/<scenario>.json
   scenarios: a_clean b_slow_search c_worker_dies c2_worker_dies_on_centre d_transport_3 e_malformed f_branch_crash (defaults = resume config: B-REPLAY, k=8, label_list8, beta=1, fan-out 2, R lag-1)
python -X utf8 -m pytest -q -p no:cacheprovider tools/coopmwm/ -k "cpu and not RehearsalSmoke"
python -X utf8 -m pytest -q -p no:cacheprovider tools/coopmwm/test_harden3_cpu.py   (includes the ~20-40 min rehearsal smoke)
```
Live collector entry: `python -m tools.coopmwm.replay_bank collect <root> --arm B|R`. It needs a binding with `b_replay`, `demand_phase_end_s`, `sumo_verify`, `slots`.

## Test status
- Full CPU suite on Windows (before f023f10c): 432 pass / 60 fail / 1 error.
  - Every failure except two also fails at the base aadee535 on this host. Causes: Windows-only (fcntl/signal/proc), or missing local packages (tools.leverprobe, tools.wmteach).
  - The two remaining failures are new harden-1 tests with the same env causes (wmteach import, Windows file lock).
- The suite MUST be rerun on cpuQ (Linux) before deployment. It has not been rerun since f023f10c.
- New targeted tests pass locally:
  - test_harden3_cpu: plan_shape, broker decline, fan-out bound, label_list8;
  - harden-2 crash-rule tests, updated to MAIN rule (2).
- `fcntl_compat.py` + `tools/coopmwm/conftest.py` are a Windows-only flock shim and a no-op on Linux.

## Not rehearsable on CPU (live checks needed)
Real SUMO/TraCI branch determinism (prefix/CRN on real plant); GPU search wall time, memory and the label_list8 parity of the final K8 calls; the SSH transport (the fake raises SSHConnectionError); /proc owner identities; multi-host spool hot-add; disk/RAM admission; real arrival_fit phase-end value; the cpuQ CPU suite.
