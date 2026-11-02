# HANDOFF: c4/harden-int (worktree E:/agent-work/c4-harden-int), 2026-11-02 22:40 +08

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
