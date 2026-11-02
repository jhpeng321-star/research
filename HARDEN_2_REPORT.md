# HARDEN-2: static resilience review of the B/R resume path, with fault-injection CPU tests

Branch `c4/harden-2` (worktree `E:/agent-work/c4-harden-2`) is forked from `c4/coopnative@aadee535`. It merges `c4/harden-3@07dd3cd7` (merge commit `af56d125`). Where the two branches overlapped, harden-3's parallel-group and broker-ordering fix was kept.

Scope and limits:
- Only engineering resilience was changed. No scientific setting was touched: screen K4/H300, final K8/H_final, N, iterations, groups 2-6, trust/beta, PPO/AWR and the accounting semantics are all unchanged.
- The work used no SUMO, no GPU and no remote writes. Only read-only inspection was done.

## Ranked findings

Severity scale:
- **CRIT**: a local fault causes a global STOP or a silent idle.
- **HIGH**: a local fault kills a process or the episode, or something breaks at the switch.
- **MED**: degraded behaviour or a hazard.

| # | Sev | file:line (at aadee535) | Failure scenario | Test (test_harden2_*_cpu.py) | Fix |
|---|---|---|---|---|---|
| 1 | CRIT | capacity_broker.py:223-233 | One slot's RPC fails: submit to a just-died spool (`unqualified or dead GPU spool` traceback), `Remote.verify` ValueError after a reconnect, ssh retries exhausted (DNS / Connection closed / reset / timeout / exit 255), or `TimeoutError` from bounded recovery. Result: `WARM_GUIDED_STOP`, so all B/R stop. | resume: `test_one_slot_transport_or_dead_spool_is_slot_local_not_global_stop` [4 cases], `test_invariant_fault_on_one_slot_still_stops_globally` | 0fab3d4b: `slot_local()` quarantines the slot and writes a restart request. Invariant words (parity / immutable / accounting / physics / safety / budget) still STOP. |
| 2 | CRIT | capacity_broker.py:212-218 | A malformed or mismatched forecast (`validate_response` ValueError, a KeyError on `result`, or wrong `record_sha256`) causes a global STOP. If it is delivered anyway, the client's identity check kills the episode. | `test_malformed_forecast_is_a_query_decline_and_slot_keeps_serving` | 0fab3d4b: `forecast_result()` turns it into a per-query decline. |
| 3 | CRIT | capacity_broker.py:77-82 | A stale unparsable request (the writer died mid-write, e.g. full disk, through the non-atomic `storage.write` used by `centre_client`) causes a global STOP. | `test_capacity_broker_cpu::test_stale_malformed_request_is_skipped_with_receipt_not_stop` (replaces the old "still_stops" test) | 0fab3d4b: the request is skipped and a `malformed_requests/` receipt is written. |
| 4 | CRIT idle | capacity_broker.py:52-65, centre_client.py:12 | `settle_unavailable` skipped centre requests. A centre claim on a dead slot is never adopted, so the **R episode waits until the 11-05 deadline**. | `test_dead_slot_settles_centre_claim_so_R_never_waits_forever`, `test_R_window_with_unavailable_centre_declines_and_releases_its_ticket` | 0fab3d4b + b659a75b: the claim is settled as `unavailable`, the R window declines, and its matched ticket is released. |
| 5 | CRIT idle | parallel_groups.py:47-57 | The episode waited for ALL group queries after the first accepted group in inherited order. Accepted responses sat unconsumed and no verification launched. This is the 18:33 incident. | harden-3 tests | Superseded by **harden-3 07dd3cd7**, merged in af56d125. My b659a75b parallel hunk was dropped; its superseded/mtime broker parts were removed. |
| 6 | CRIT idle | capacity_broker.py:239 | With every slot quarantined, quota-blocking requests wait forever and nothing raises an alert. | `test_all_slots_lost_writes_capacity_alert` | 0fab3d4b: writes a `capacity_alerts/*.json` receipt. Requests keep waiting for a hot-add rather than a STOP. |
| 7 | CRIT (MAIN a) | exact_runtime.py:93-101 | ONE failed, stopped or non-CRN exact grade wrote GRADER_STOP and WARM_GUIDED_STOP. | main: `test_a_*` | eb08a25e: per-grade `failed_grades/` and `grade_alerts/` receipts, the grade is excluded from evidence, no global STOP. |
| 8 | CRIT (MAIN b) | accel_runner.py:74-75 | ONE episode exception made `pool.map` raise, which became a global STOP with a job that cannot be retried. | main: `test_b_*` (replacement, shortfall, 3-consecutive, >20%, STOP and pause propagate) | 386e5c17: the failed episode gets `EPISODE_FAILED.json` and is never replayed. A charged replacement runs on the next unused TRAIN world of the same cell. Escalation happens on 3 consecutive failures or >20% of the wave. Only TRAIN waves are covered. |
| 9 | HIGH | query_errors.py:9, plan_service.py:174-178 | A per-query ValueError (group-size, history ticks, whitelist, extra-seed shape) kills the GPU worker. This is the 17:26 class of fault. | `test_per_query_value_error_is_a_decline_not_worker_death` [4]; updated `test_incident_cpu` fatal test | ce4c8db8: a ValueError declines unless it names an invariant. Request validation now runs inside the isolation. **plan_service.py is NUMERICAL-hashed, so redeploying needs fresh startup gates.** |
| 10 | HIGH (k=8 / MAIN c) | mppi_runtime.py:54, mppi_search.py:83 | The learned hot-swap: `warm_start` exists only on c4/plan-proposer. Calling it here raises a TypeError, which `execute` declines, so EVERY B query is declined silently. A second `config=` also raises a TypeError. | `test_c_*`, `test_learned_search_kwargs_against_plain_search_are_a_declined_type_error` | 6db74622: when the binding has `learned_search`, a startup gate runs a dry learned search plus `resume_preflight` and fails BEFORE READY. The integration itself is not done; see the live checks. |
| 11 | HIGH (k=8) | sumo_verify.py:389, mppi_search.py:485 | With plain search `top_k=4` and only saving>0 finalists, B verifies at most 4 candidates (often fewer) while R gets 8. The B/R comparison is not like-for-like. | `test_B_with_plain_top4_under_k8_records_shortfall`, `test_resume_preflight_flags_k8_hazards` | 531e3c94: per-decision `verify_k{requested,available}` and `resume_preflight`. **Needs a live owner decision**: cap R's k at B's available k, or accept and disclose. |
| 12 | HIGH (MAIN d) | sumo_verify.py:211-214 | With the k=8 phase mis-set, every decision becomes `budget_blocked` silently and B=A. | `test_d_budget_blocked_verification_writes_alert`, preflight | 6db74622: writes `alerts/BUDGET_BLOCKED_<vid>.json`; preflight requires `sumo_verification_k8` when k != 4. |
| 13 | HIGH | sumo_verify.py:215-216 | `runner.prepare` OSError (disk full writing SNAPSHOT.pt) is an episode exception, and therefore a global STOP. | `test_full_disk_snapshot_prepare_fails_decision_not_episode`, `test_disk_guard_blocks_before_any_charge` | 531e3c94: the decision is marked `failed` (precharged, no physics). The registered 2 GiB guard is checked before any charge. |
| 14 | HIGH | warm_spool.py:28-31 | The fixed `.publishing` temporary left by a killed or full-disk writer makes that destination unwritable forever (FileExistsError, so the broker dies). | `test_stale_publishing_file_*`, `test_full_disk_during_publication_*` | 00342bb4: per-writer temp plus `os.link`; same-payload replay and conflict semantics kept. |
| 15 | HIGH | warm_broker.py:38 | Exit 0 with truncated stdout gives JSONDecodeError, a global STOP. | `test_truncated_rpc_stdout_is_a_retryable_transport_fault` | 0fab3d4b: treated as SSHConnectionError and retried. |
| 16 | MED | hotadd.py:58,64,86,95 | The OWNER / ADMISSION_`<host>`_SECOND / FAILED receipts are immutable and per host, so a dead hot-added worker can never be replaced on that host. | `test_hotadd_replacement_uses_its_own_receipt_label` | ce31cd5c: optional `receipt_label`. |
| 17 | MED | sumo_verify.py:325-330 | `random_candidates` loops forever if draws collide (fully clipped profiles). | `test_random_candidates_terminate_when_draws_collide` | 531e3c94: bounded at 64·k draws. |
| 18 | MED | sumo_verify.py:444 + verification_live_ops.py:180 | k=8 with the bound `workers=5` runs the 9 branches in 2 blocking waves, about 2x decision wall time. | preflight | Default is now k+1. The live binding must set `workers=9`. |
| 19 | MED | verification_grades.py:33 | `record_smoke` hard-codes k=4 and 5 branches, so a k=8 smoke can never be recorded. | `test_k8_smoke_gate_and_grade_reuse_accept_nine_branches`, `test_k8_receipt_schema_nested_best_of_4_vs_8` | 531e3c94: takes a `k` parameter. `reuse()` already works for 9 branches. |
| 20 | MED | sumo_verify.py:167-170 | One crashed SUMO branch fails the whole decision; the group follows policy. Local, but at k=8 there are 9 chances per decision. | `test_one_crashed_sumo_branch_fails_the_decision_never_the_episode` | Kept (registered: never imputed). **Needs a live owner decision** on whether the best graded candidate may be chosen when one candidate branch fails. |
| 21 | MED | mppi_runtime.py:57-61 | Logging-only `receipt_from_search` runs inside per-query isolation. If it raises, a valid search result is declined. `overlap.recheck` truncation is computed for the Thompson plan but applied to whichever verified candidate wins. | - | **Needs a live owner decision**. mppi_runtime is NUMERICAL-hashed (requalification needed). |
| 22 | MED | plan_client.py:94, parallel_groups.py:49, centre_client.py:12 | Client waits have no per-query timeout. A broker process that dies without writing a STOP (SIGKILL/OOM) leaves episodes blocked at 0 CPU. | - | Live check: monitor the broker owner PID. |
| 23 | MED | plan_service.py:168 / warm_spool.py:107 | GPU workers read `deadline_unix` straight from GPU_BINDING.json; they do not apply RESOURCE_EXTENSION_1651. | - | Live check: every GPU root's binding deadline must be 2026-11-05 23:59. |
| 24 | MED | accel_runner.py (new) | A Python episode worker OOM-killed raises BrokenProcessPool for the whole wave. This is still a STOP by design. | `test_b_*` (propagation) | Disclosed. |
| 25 | LOW | verification_beta_ops.py:562 | `assert b['sumo_verify']['k']==4` in the deploy op. | - | The ops script must change at the k=8 switch. |

### Superseded or changed contracts

These need the owner's acknowledgement:
- `test_stale_malformed_request_still_stops` was replaced: a stale malformed request is now skipped with a receipt.
- `test_physics_parity_accounting_validation_faults_remain_fatal`: a generic `ValueError('fatal')` now declines. ValueErrors that name an invariant are still fatal (owner 17:26 directive).
- b659a75b (parallel_groups) is superseded by harden-3 07dd3cd7.

## CPU tests

- 35 new test functions (41 cases): `test_harden2_resume_cpu.py` (23 functions, 29 cases) and `test_harden2_main_cpu.py` (12). All pass.
- Full `tools/coopmwm -k cpu` regression on Windows rl_driving, using a test-only fcntl shim outside the repo: **418 passed, 57 failed, 1 error, 21 subtests passed (11 min)**. Versus the aadee535 baseline under the same shim (374 passed, 58 failed + 1 error): no new failures. Two baseline-flaky concurrency tests in test_verify_continuation now pass. The remaining failures are the baseline set, i.e. failures that are Windows/env-only: missing `tools.leverprobe`, `/proc`, `SIGCONT`, frozen model files. There are no new failures.

## Live checks needed at resume

1. Deploy `capacity_broker.py`, `warm_broker.py` and `warm_spool.py` (broker), then restart the broker with its own PID only. Confirm that `slot_local_fault` events, `worker_restart_requests/` and `capacity_alerts/` appear when expected, and that no `WARM_GUIDED_STOP_V5` exists.
2. GPU roots (incident1, second1, concurrent3, plus any hot-add):
   - If `query_errors.py` or `plan_service.py` is deployed, the startup gates must be re-run, because adopted gates are NUMERICAL-hash bound.
   - Confirm that each GPU_BINDING `deadline_unix` is 1793894340 (2026-11-05 23:59 +08).
   - Confirm each worker is READY and that `SPOOL_FAILED.json` is absent on every slot.
3. Broker owner PID and start_ticks are alive. Clients have no timeout, so a silent broker death idles every episode.
4. `exact_runtime` and `accel_runner` deployed. After deploy, `failed_grades/` and `grade_alerts/` are monitored, and every `jobs/*/EPISODE_FAILED.json` is reviewed together with its replacement charge.
5. At the k=8 switch:
   - `job['sumo_verify']` = `{k: 8, workers: 9, phase: 'sumo_verification_k8', guide_beta: 1}`.
   - `sv.resume_preflight(cfg, search_config, learned=True)` returns [] on the deployed tree.
   - cpuQ has capacity for (episodes x 9) SUMO branches. Today: 10 x 9 = 90 processes.
   - Free disk is at least 2 GiB.
   - A one-decision k=8 smoke is recorded with `record_smoke(..., k=8)`.
6. Learned proposer:
   - merge c4/plan-proposer `mppi_search.warm_start`, add a `learned_search_kwargs` hook in `mppi_runtime.plan` that merges `config`, and set `GPU_BINDING.learned_search={checkpoint,k:8,mode}`;
   - the worker must pass the new startup gate before READY;
   - measure search wall time at top_k=8 (16 finalists at K8/H_final).
7. B/R matching: decide what to do about `verify_k.available < 8` for B (finding 11). Then check `K8_VS_K4_NESTED` using the per-decision 9-branch receipts.
8. Hot-add replacement: use a fresh `receipt_label` for any replacement on a host that already has HOTADD receipts.
9. R concurrent with B (new design): R reads B's latest COMPLETED `B_RANDOM_QUOTA_<u-1>`. Its `random_match/u<R update>` folder must not collide with an in-flight R update.
