# HARDEN-INTEGRATION REPORT (branch c4/harden-int, HEAD 1d476704 + report commit)

Date: 2026-11-03. Worktree `E:/agent-work/c4-harden-int`. CPU-only rehearsal on this Windows host
(rl_driving env); fake SUMO/GPU/WM leaves. No SUMO, GPU or remote writes were performed; remote
hosts were not touched by this integration work.

## 1. What is integrated (Paper A, verified-plan bank)

All on top of harden-2 final (5d71ff01, includes the harden-3 stall fix):

- B-REPLAY binding, collector/bank reader, AWR imitation through `exact_grade.imitation_pairs` +
  `exact_update` (65a67132, 15542a51, 62b80987).
- MAIN decisions (65a67132..62b80987): label_list8 top-8 unfiltered; equal k for R (lag-1 match);
  partial-branch-failure rule; per-candidate overlap truncation; blocking collector searches;
  `verify_per_round` knob; LS-COLLECT label reader (`replay_sources`, tag `policy_version=u3`);
  R fairness cap (never padded from B; shortfall/excess recorded); demand-phase admission guard
  (t0 + max look-ahead 420 s <= `demand_phase_end_s`, exclusions recorded).
- CONFIRMATION STAGE (f023f10c): every first-seed positive decision (own bank and LS labels) must
  pass (a) replication on an independent CRN seed (advantage > 0 again), (b) extended window to
  min(episode end, t0+D+600) (advantage > 0), (c) safety vs the policy branch. Weight = temperature
  rule on the mean replicated advantage. Per-decision logging; `BANK_QUALITY.json` per arm;
  charges to `sumo_verification` / `bank_verification_ext`.
- TTC<1.5 s count is TRACE-BASED (`traffic_shepherd.metrics.closing_ttc` on the observable branch
  rows), per MAIN decision; NO plant counter, NO fingerprint change.
- COOPERATION (b58ca5b2, MAIN 00:01/00:04): for every admitted entry whose plan moves >= 2 CAVs,
  SOLO branches on the first seed (member m executes its part, others 'policy continues');
  cooperation gain = group advantage - best solo advantage; `cooperative=True` iff gain > 0;
  single-CAV plans are `cooperative=False`; all solo advantages recorded; solo charges to
  `bank_verification_ext`. Synergy twins table (group vs best solo) in `BANK_QUALITY.json`.
- EXPOSURE (b58ca5b2): cooperative entries are oversampled to 50% of imitation samples when they
  exist (duplicated pairs; weights unchanged); per-update receipts record cooperative entries,
  their sample share, duplicated pairs and mean cooperation gain with CI
  (`exact_imitation.exposure`).
- LESSON CHECK (4f090b76, MAIN 01:22): after every B and R update, on a frozen list of up to 12
  admitted snapshots (first admitted, frozen once chosen, `LESSON_LIST_<arm>.json`), ONE exact CRN
  branch of the NEW policy per snapshot; `LESSON_CHECK_u<k>_<arm>.json` reports per-entry
  recapture = (J_policy_u3 - J_new_policy) / (J_policy_u3 - J_best_plan), the mean with CI, and
  the imitation action error (m/s^2); charges to `bank_verification_ext` (lesson-prefixed ids).
- BANK SAFETY RULE AMENDMENT (64b6946d, MAIN 2026-11-03 04:26, identical for B and R):
  (i) collisions and teleports must NOT increase (hard); (ii) TOTAL hard-brake + TTC<1.5 s events
  must NOT increase. The OLD verdict (no single metric may increase) is recorded per decision
  (`safety_old_rule_ok`) and disclosed in `BANK_QUALITY` (`rejections.safety_old_rule`). Synergy
  solo branches also run as a DIAGNOSTIC on every replication+later-harm passer that safety
  rejects; its twin row is published tagged with the admission verdict (`synergy_diagnostic`).
- Rehearsal fake-plant fixes (52084665, c5dccc1b): hard_brake counts only requested decel beyond
  the lower bound; toy car-following (a follower never drives through its leader) so genuinely
  gentle plans reach the admit/cooperation path. Unit-tested.
- febba5bc: confirmation charges are kind-prefixed and charged exactly once (duplicate-charge bug
  found by rehearsal, fixed).
- `fcntl_compat.py` + `tools/coopmwm/conftest.py`: Windows-only flock shim, no-op on Linux.

## 2. Rehearsal scenario table (B-REPLAY, all scenarios completed)

Common config: k=8, label_list8, beta=1, fan-out 2, R lag-1, episodes_per_update=2,
verify_workers=9, stall 900 s, timeout 6800 s, verify_per_round=2; B and R run concurrently;
the batch ends via the registered harness-end health pause (not an engineering STOP).
"adm" = admitted entries, "coop" = admitted cooperative entries. Latest batch per scenario is
listed; earlier vintages agree unless noted.

| scenario | fault injected | batch (code vintage) | wall s | B | R | eng STOPs | notes |
|---|---|---|---|---|---|---|---|
| a_clean | none | rp4 (HEAD) | 1257.8 | complete: dec 12, adm 1, coop 0 | complete: dec 9, adm 1, coop 1 | 0 | R twin: group 4, first-seed +60.3 vs best solo +44.4, gain **+15.9 veh-s**; safety-rejected 2-CAV entry published a diagnostic twin; LESSON_CHECK u4/u5 both arms (R u5 recapture 9.05, action error 1.62 m/s^2) |
| b_slow_search | search latency 10 s | rp4 (HEAD) | 1093.6 | complete: dec 8, adm 1, coop 0 | complete: dec 4, adm 1, coop 1 | 0 | slow searches absorbed by stall guards; charges within quote |
| c_worker_dies | GPU worker dies on gpuA_second | rp6 (pre-amendment) | 644.6 | complete: dec 5, adm 0 | complete: dec 4, adm 0 | 0 | `gpu_worker_died` -> slot quarantine + restart request, no STOP; earlier vintages (rp, rp2) agree |
| c2_worker_dies_on_centre | worker dies on the centre slot | rp4 (HEAD) | 772.4 | complete: dec 5, adm 0 (diagnostic twin) | complete: dec 2, adm 1, coop 1 | 0 | centre death quarantined without stopping the broker; rp5 vintage (pre-amendment) hit the 6800 s batch timeout on R |
| d_transport_3 | 3 transient transport errors | rp4 (HEAD) | 766.3 | complete: dec 5, adm 0 | complete: dec 3, adm 0 | 0 | `transport_error_injected`; all retries succeeded, no STOP; rp3 vintage also complete (2443.2 s) |
| e_malformed | malformed GPU response | rp4 (HEAD) | 1741.1 | complete: dec 8, adm 1, coop 0 | complete: dec 8, adm 2, coop 1 | 0 | `gpu_malformed_response` declined and retried; 1 R diagnostic twin; LESSON_CHECK u5_R: 2 frozen entries, recapture mean 43.24 [30.58, 55.91], action error mean 1.38 m/s^2 |
| f_branch_crash | SUMO branch crash | rp4 (HEAD) | 706.7 | complete: dec 6, adm 0 (diagnostic twin) | complete: dec 2, adm 0 | 0 | partial branch failure handled by the `sumo_verify.choose` rule; rp3 vintage complete (1057.1 s); rp6 vintage (pre-amendment) hit the batch timeout on B |

No scenario produced an engineering STOP, a stall, or an idle deadlock; every confirmation,
solo and lesson branch was charged to `bank_verification_ext` with kind-prefixed ids and passed
the within-quote check.

Host-overload caveat, kept for the record: an rp4 `b_slow_search` attempt started while four
other heavy jobs (rehearsals with 9 verify workers + a foreign agent's suite) were running on this
Windows host failed after 178.7 s with `BrokenProcessPool` + a guided STOP
(`b_slow_search_overload.*` in h3res/rp4). This is resource exhaustion, not a code failure; the
same scenario rerun alone completed as reported above. Do not run more than 3 heavy jobs at a
time on this host.

Earlier batches: rp3 predates the febba5bc charge fix (its `budget_blocked` confirmations were
the duplicate-charge bug; kept for the record only). rp5/rp6 predate the 04:26 safety amendment
(their BANK_QUALITY has no `safety_old_rule` field); fault-handling evidence from them remains
valid.

## 3. Confirmation-stage behaviour in rehearsal (rp4, current code)

- Positives are rare and the funnel is honest: e.g. a_clean B 12 decisions -> 3 first-seed
  positives -> 3 confirmed -> 1 admitted (rejections: replication 1, later_harm 1, safety 2 =
  `safety_old_rule` 2); e_malformed R 8 -> 4 -> 4 -> 2 admitted (1 cooperative, replication 1,
  safety 1); c2 R 2 -> 1 -> 1 -> 1 admitted cooperative.
- The 04:26 amendment is exercised: the unit tests admit the 986016 pattern (hard_brake +3,
  TTC<1.5 s events -92) under the new rule while disclosing the old-rule rejection; a collision
  increase is rejected by both rules; a safety-rejected cooperative plan still publishes its twin
  (never imitated).
- Synergy twins observed in rehearsal include cooperative=True admissions (e.g. R group of 4:
  gain +15.9 veh-s) and cooperative=False admissions (single-CAV plans), plus diagnostic twins for
  safety-rejected multi-CAV passers (e.g. R group of 6: first-seed +66.7 vs best solo +53.7, gain
  +13.0, `admitted=false`, `safety_old_rule_ok=false`).
- LESSON_LIST freezes on the first admitted entry per arm and is never re-chosen;
  `LESSON_CHECK_u<k>_<arm>.json` written after every update (empty-list receipts when a bank has
  no admission yet).

## 4. Exposure (cooperative oversampling)

Implemented and unit-tested (oversampled to 50% of imitation samples via duplicated pairs, weights
unchanged, per-update record in `exact_imitation.exposure`). In the short rehearsals the first
cooperative admission usually landed after the updates had already read the bank, so the receipts
record `rule="no cooperative admitted entries; no oversampling"` with zero counts; the firing path
is covered by the b58ca5b2 unit tests. Live updates imitate the whole bank each time, so the 50%
share applies whenever cooperative entries exist.

## 5. Test status (this host, Windows, rl_driving env)

`python -X utf8 -m pytest -q -p no:cacheprovider tools/coopmwm/ -k "cpu and not RehearsalSmoke"`
at HEAD 1d476704: **443 passed / 62 failed / 1 error / 3 deselected / 38 subtests passed** (4:19).

Failure breakdown (all documented, none behavioural):
- 59 failures + 1 error are pre-existing at the base aadee535 / 22:06 baseline on this host:
  Windows-only (fcntl/proc/signal), missing local packages (tools.leverprobe, tools.wmteach), the
  `git --cached` shim error, and similar env causes.
- 1 failure (`test_sumo_verify_cpu.py::test_parallel_determinism`) is load-flaky and passes in
  isolation (it failed while a foreign agent's suite was loading the host).
- 3 failures are NEW but are stale BUDGET-ARTIFACT expectations, not behavioural regressions:
  `test_budget_keeps_every_failed_charge_and_approved_cap`,
  `test_final_phase_caps_use_approved_config_not_older_training_binding`,
  `test_exact_owner_budget_and_protected_liabilities`. Commit febba5bc registered the MAIN
  2026-11-02 22:32 owner-approved cap amendment (`bank_verification_ext=2600000` added to
  `PHASES` in accel_protocol.py); the three tests still hardcode the pre-amendment 7.5M-era
  numbers and now see the amended allocation. `repair_budget`'s guard semantics are unchanged.
  Budget caps are MAIN's authority, so the expectations were NOT hand-edited here; the cpuQ
  (Linux) suite run before deployment is the gate, and MAIN may want to re-bless those three
  expectations against the amended quote.

New targeted tests all pass locally: fake-plant car-following (FakeFollowingTests), safety
amendment trio, hard_brake scoping, oversampling, lesson check, charge dedupe, plus the earlier
harden-1/2/3 targeted sets.

## 6. Not rehearsable on CPU (live gates for the operator)

Real SUMO/TraCI branch determinism (prefix/CRN on the real plant); GPU search wall time, memory,
label_list8 parity of final K8 calls; the SSH transport (fake raises SSHConnectionError); /proc
owner identities; multi-host spool hot-add; disk/RAM admission; the real `arrival_fit`
`observed_phase_end_s` for the demand-phase guard; the LS-label layout check (point
`replay_sources` at `/home/ubuntu/coopmwm20261102_lscollect1`, then call
`replay_bank.bank_decisions(root,'B',sources=[...])` and `bank_quality` read-only before
training); and the full CPU suite rerun on cpuQ.

## 7. Operator commands

```
set OMP_NUM_THREADS=1 & set MKL_NUM_THREADS=1
python -X utf8 -m tools.coopmwm.rehearsal_cpu <scenario> --workdir <tmp> --set episodes_per_update=2 --set verify_workers=9 --set latency_s=0.5 --set stall_s=900 --set timeout_s=6800 --out <tmp>/<scenario>.json
   scenarios: a_clean b_slow_search c_worker_dies c2_worker_dies_on_centre d_transport_3 e_malformed f_branch_crash
   (b_slow_search adds --set latency_s=10)
python -X utf8 -m pytest -q -p no:cacheprovider tools/coopmwm/ -k "cpu and not RehearsalSmoke"
python -X utf8 -m pytest -q -p no:cacheprovider tools/coopmwm/test_harden3_cpu.py   (includes the ~20-40 min rehearsal smoke)
```
Live collector entry: `python -m tools.coopmwm.replay_bank collect <root> --arm B|R`; the binding
needs `b_replay`, `demand_phase_end_s` (from the fitted arrival artifact), `sumo_verify`, `slots`.
Offline bank builder per MAIN 04:06: run replay_bank's confirmation/synergy code over the
LS-COLLECT labels (B, u3) on cpuQ, charges to `bank_verification_ext`, publish
`BANK_QUALITY_B.json` (admitted count, mean advantage + CI, cooperative count, mean gain + CI,
synergy twins, safety rejections incl. old-rule count).
