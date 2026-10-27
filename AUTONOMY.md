# S-AUTONOMY — remote-only continuation handoff

## Main chain — armed remotely, verified05:58:41Z

- gpuA successor **100482/start846087574**, SID100482, PPID1 verified after SSH
  closed; currently `WAIT_MAIN_C2H60B`. Existing coordinator95283/start845712191
  and actor95284 remain owned by S-C2H60B and were not restarted or modified.
- gpuA release `/root/autodl-tmp/c4/restart20261026`; canonical source
  `autonomy/remote.py`, owned successor receipt `AUTONOMY_PID.json`, phase status
  **`AUTONOMY_STATUS.json`**, append-only `AUTONOMY_EVENTS.jsonl`, child receipts
  `AUTONOMY_CHILDREN.jsonl`, log `arm/driver_autonomy.log`.
- Both REMOTE arm registrations now enforce **280000charged sim-s**. Main retains
  176473charged/103527remaining; coop-off retains133418charged and is **paused by
  owner priority, resumable**, no waiter/relaunch. gpuR is reserved for long-run
  WM work. Remote historical snapshots/charges/checkpoints are retained.
- Main: wait exact predecessor exit -> acquire own/driver/cycle1 locks -> reuse
  completed native-gated C2 policy -> zero-charge preflight -> healthy-box serial
  startup canaries -> four-world HARMFUL-only guard -> remainder -> convert ->
  registered40step WM fine-tune -> unchanged acceptance -> C3 policy at60s,
  hot-starting selected best C2 checkpoint -> gate -> collection/WM/check.
  One operational retry/phase; scientific/native/horizon/acceptance failure or
  safety/HARMFUL stops. No scientific threshold changes. Next full36933collection
  must fit280000; with no failed charges, C2projects213406, C3projects250339,
  another collection287272 is refused and status becomes WAIT_OWNER.
- gpuA mirror **99141/SID99141**; status `AUTONOMY_MIRROR_STATUS.json`, owned PID
  `AUTONOMY_MIRROR_PID.json`. Fresh CPU reclaim watchers **cpuP817387/SID817387**
  and **cpuQ802419/SID802419** report Tencent404/no termination scheduled. Own
  receipts `.../main14/restart20261026/autonomy/{SPOT_PID,SPOT_STATUS}.json`.
  Numerical/artifact mirrors stay on gpuA. A lost/reclaiming box is excluded;
  retain charges and canary-first retry on the survivor, or pause without a box.
- Childless old waiter100196 was exact-incarnation verified, retired and archived
  solely to install the collector-factory cache fix. The selected collector
  preflight/failover adapter now persists through actual `loop.fast_module()`
  dispatch. Four focused dispatch/E5 tests PASS locally; canonical broader checks
  and source receipts remain with `tools/autonomy/`.

### Resume/inspect later

```powershell
python -B -m tools.autonomy.transport status --host gpuA
```

The chain needs no laptop and is already armed. Do not issue another launch while
the receipt-bound successor is live. On WAIT_OWNER/OPERATIONAL_PAUSED inspect the
status, canonical log, per-phase state `arm/loop/LOOP14_STATE.json`, retry ledger
`AUTONOMY_RETRIES.json` and current registration before an owner-authorized resume.
Preserve budget/safety stops and completed phases; never restart C2H60B's actor.

## E5 optional cycle1 comparison — PREPARED, NOT LAUNCHED

The matched `ours-coopoff` arm is staged on cpuP/cpuQ under
`/home/ubuntu/e3long/inputs/coopoff_cycle1/`; operational adapter and zero-charge
readiness receipt are under `/home/ubuntu/e3long/operations_e5/`.
Both boxes passed baseline/coopON completeness, exact source/input/native-gate
bindings, policy CPU load and CIRCLES ONNX load. No E5 SUMO, simulation charge,
canary, coordinator or worker has started. No E5 PID is claimed.

- Policy: coop-off cycle1 checkpoint180,
  `e1fa4881bbc5c89d05298e979d98b23284308331ad1bbb12f9608fa67e105675`;
  native-relative PASS16/16, mean deltaJ−0.3592796326, original10s frozen panel.
- Same original228worlds973000–973227,19/cell,300s native warmup+600s intervention.
  Reallocates205,200charged sim-s from unused410,400 policy-phase allocation;
  max102,600/box, no simulation retries, no gated arms. Historical charges retained.
- Scientific runtime/source manifest stays
  `b3c031adb6bc2b5c2bca160bea308f83ba5567b97c1cd0e853eddcd8367a8193`.
  The immutable runtime's internal arm selector remains `ours`; the new GO and
  output namespace explicitly alias that selector to **ours-coopoff**.
- Same S-E3OURS nice19/one-thread,6workers/box, box-local live loop collection
  priority, load/owned YIELD checks and stale mirror heartbeat pause.
- Expected outputs: CPU `/home/ubuntu/e3long/main/e3long_coopoff/shard{0,1}/`;
  GPU mirror `/root/autodl-tmp/c4/restart20261026/e5/{cpuP,cpuQ}/`;
  self-analysis `e5/RESULTS.json`, also written to both CPUs' `operations_e5/`.
  Contrast is **coopON(cycle1) minus coopOFF(cycle1)** at120/300/600s, fixed5000
  world-cluster bootstrap draws stratified by the original12cells; complete228
  worlds and matching observable initial-state hashes required.
- Canary973000 is charged once within205,200, CRN/clock checked before remainder;
  main runner resumes its verified result in shard0 rather than rerunning it.

### Exact commands for main later

Refresh preparation/bindings without simulation from this worktree:

```powershell
python -B -m tools.autonomy_e5.prepare
```

Once the E5 operational adapter/priority/detached self-analysis preflights are
reviewed and approved, this **explicit launch** revalidates both boxes' current
source/input/GO bindings before starting one detached gpuA coordinator:

```powershell
ssh -o BatchMode=yes gpuA "/root/miniconda3/bin/python -B /root/autodl-tmp/c4/restart20261026/e5/remote.py launch"
```

Read `e5/PID.json`, `e5/CHILDREN.json`, `e5/STATUS.json`, `e5/driver.log` on gpuA.
CPU child PID records are `operations_e5/{CANARY_PID,MAIN_PID}.json`;
each launched child and coordinator uses its own session, DEVNULL stdin and log
redirection. E5 was deliberately left unlaunched in this deadline window because
the complete live runner/priority and end-to-end detached self-analysis preflight
has not been exercised. Three offline analysis regressions pass (contrast sign,
world-cluster CI, missing-world/CRN fail-closed behavior).

## S-E5RUN — launched, first-batch failure; WAIT_OWNER (~14:48 UTC+8)

- Main GO/amendment appended to the authoritative REGISTRATION.md before SUMO;
  both543-file/input/gate preflights and8dispatcher pause checks passed.
- Canary973000 PASS/reused once. Transport JSON repair preserved exited gpuA
  102196; successor **103399/start846363018** exited WAIT_OWNER after cpuQ's
  first6jobs failed before SUMO: detached SSH PATH omitted installed SUMO.
- cpuP **844872/start9770169** remains detached and safely stale-mirror-paused at
  an empty batch boundary:19successful worlds,17,100charged=actualsim-s. cpuQ
  **825898/start9770420** failed/exited:6failed reservations,5,400charged/0actual.
  Total22,500charged/17,100actual;182,700unspent. No simulation retry.
- cpuP CRN/clocks/conservation and first7 mirrored result hashes PASS. Receipts:
  gpuA `e5/FAILURE_RECEIPT.json`; CPUs `operations_e5/FAILURE_RECEIPT.json`,
  cpuP `operations_e5/FIRST_BATCH_VERIFIED.json`. No complete `e5/RESULTS.json`;
  **ETA blocked pending new main amendment** for the six charged pre-SUMO failures.
- Seven focused regressions PASS. Explicit detached-PATH/SUMO-presence fix is
  local only; do not rerun prepare/launch/resume against these retained attempts.

## S-E5FIX — cpuQ repaired and verified; detached continuation (~15:06 UTC+8)

- Main15:00 amendment appended to authoritative REGISTRATION.md BEFORE repair
  SUMO, with copies/hash-bound `FIX_AUTHORITY.json` on both CPUs and gpuA.
  All six historical pre-SUMO failures and5,400charges retained, no refunds.
- cpuQ detached zero-charge preflight **834420/start9894233** PASS: explicit
  `SUMO_HOME=/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo` and its
  `bin` first on PATH, matching working cpuP; SUMO1.27.1 resolves, CIRCLES ONNX
  and policy CPU load,543source hashes, original GO/input/native gate PASS.
- ONE cpuQ canary973001 **834772/start9897148** PASS,900new charged sim-s.
  Reused as the first of six authorized one-time repairs; all six now verified
  CRN/clocks/conservation/zero incidents, six result SHA256s match gpuA mirror.
  Original failed `jobs/` retained; repair results in `shard1/retry_jobs/` and
  distinct `.e5fix1` ledger attempt IDs preserve every charge and original job.
- Detached gpuA coordinator **104810/start846490743**, SID104810/PPID1;
  adopted existing cpuP **844872/start9770169** and new cpuQ remainder
  **835541/start9901074**, SID835541/PPID1; stdinDEVNULL verified. No signals.
  cpuP resumed only AFTER cpuQ canary proof. Both retain six nice19/one-thread
  workers, original box-local collection/load/owned-YIELD/stale-mirror guards;
  live main cycle-2 (~16:15) collection keeps priority.
- At15:06:31 snapshot: cpuP25results/27,900charged, cpuQ6results/16,200charged
  (includes historical failures and next in-flight reservations), total44,100,
  161,100remaining. First repaired-world maximum wall52.02s. Estimated registered
  work boundary **15:25–15:40 UTC+8**, extended by any live-priority pause.
- **Budget arithmetic stop:**209missing worlds needed188,100new sim-s, but main
  authorized182,700. Hard205,200total/102,600per-box caps therefore allow only
  **222/228successful worlds**, leaving six; no top-up is authorized. The detached
  coordinator writes `FIX_BUDGET_RECEIPT.json`/`WAIT_OWNER_BUDGET` with exact
  missing worlds and5,400additional sim-s needed, without incomplete-panel
  inference. Original self-analysis remains armed for a genuinely complete228
  panel, writing `e5/RESULTS.json` only after full pairing/accounting checks.
- Inspect gpuA `e5/{FIX_PID,FIX_CHILDREN,FIX_STATUS,FIX_VERIFIED}.json` and
  `fix_driver.log`; cpuQ `operations_e5/{FIX_PREFLIGHT,FIX_CANARY_VERIFIED,
  FIX_REMAINDER_PID,FIX_RUNNER_STATUS,FIX_VERIFIED}.json`. Old PID/status/logs
  preserved. Canonical new code `tools/autonomy_e5/{fix_runner,remote_fix,
  fix_ops}.py`;10focused tests PASS, focused Ruff checks PASS. Do not rerun
  original prepare/launch/resume or repeat any S-E5FIX attempt.

### S-E5FIX fresh-session verification — WAIT_OWNER_BUDGET (15:41 UTC+8)

- The authoritative REGISTRATION.md already contains the15:00 amendment and
  15:06 execution receipt. Repair, preflight, one canary, six one-time repairs
  and detached continuation finished before this session's read-only checks.
- Final cpuP **114/114** successful worlds,102,600charged=actualsim-s; cpuQ
  **108/114**,102,600charged/97,200actualsim-s. Total **222/228**,205,200charged,
  199,800actual; exactly182,700new charges after the original22,500. All six
  original cpuQ failures/5,400charges remain; every repair has exactly one
  `.e5fix1` charge, and canary973001 is reused without another attempt.
- Rechecked every successful result against native/coopON observable initial
  hashes,120/300/600s clocks, conservation, zero dropped steps/collisions/
  teleports, unique ledger charges and its per-job result file: **PASS**.
  All **222result SHA256s** match gpuA's mirror, including all108cpuQ outputs.
- gpuA `e5/FIX_STATUS.json` and `FIX_BUDGET_RECEIPT.json`, plus both CPUs'
  `operations_e5/FIX_FINAL.json`, report **WAIT_OWNER_BUDGET**. Missing cpuQ
  seeds: **973217,973219,973221,973223,973225,973227**. Completion requires
  **5,400additional sim-s**, outside current authority. `e5/RESULTS.json` is
  absent; no incomplete-panel inference was produced.
- Receipt-bound cpuP844872, cpuQ835541 and gpuA104810 have exited. No simulation,
  launch, heartbeat refresh, control edit or signal was issued in this session;
  loop collections retain priority. Further dispatch requires new main authority.
- Local proof snapshots: `tools/autonomy_e5/receipts_fix/`
  `{cpuP,cpuQ,gpuA}_resume_verified.json`; historical `VERIFIED.json` retained.
  Ten focused E5/E5FIX tests PASS; focused fatal-error Ruff checks PASS.

## S-E5DONE — full 228-world cycle-1 analysis complete (2026-10-27)

- The16:00 MAIN completion amendment already existed. The prior session completed
  the six remaining cpuQ seeds973217/219/221/223/225/227 exactly once; final
  cpuQ108,000charged/102,600actualsim-s, cpuP102,600/102,600. This fresh session
  dispatched **zero simulations**. E5 total210,600charged/205,200actual; historical
  six pre-SUMO failures/5,400charges retained. Policy415,800/615,600charged;
  production826,200/1,026,000charged (820,800actual).
- All228matched native/ON/OFF/MAPPO worlds,19/cell, all120/300/600s clocks,
  543source hashes/host, bound inputs, unique charges, standalone outputs and
  source/road/group conservation PASS (maximum residual0.000132114veh-s).
  All924E5scientific JSON/ledger files match gpuA's mirror; owned simulation
  predecessors and completion coordinator exited.
- **ON−OFF total delay including entrance waiting:0[0,0]veh-s at every clock,
  in every cell and every world.** Reverse OFF−ON saved-delay is also0[0,0].
  All228observable trajectory NPZ pairs are byte-identical. Deployment summaries
  bind distinct correct checkpoint paths; CPU loading matches stored tensors;
  residual-head parameters differ, as do requested/issued commands in all228worlds
  (maximum acceleration difference0.004012585m/s²). Plant/executor cause of equal
  realised trajectories is unresolved; no general objective-equivalence claim.
- Both policies minus native at120/300/600s:−67.2[−100.5,−34.6] /
  −323.2[−505.5,−135.1] /−970.0[−1491.7,−446.3]veh-s/world. Minus MAPPO:
  −0.7[−23.0,+22.4] /+162.6[−8.0,+333.6] /+938.7[+454.9,+1430.4].
  Fixed5000paired whole-world bootstrap draws within12cells, seed20261027;
  pointwise CIs. Throughput2263.7veh/h both, native2247.5, MAPPO2283.7;
  zero collisions/teleports/overflow over912arm-worlds.
- Frozen COOPMEAS452d44a7/thresholds9119cd77, no refit. At600s, ON=OFF rates
  (events/CAV-hour): multi-CAV slowing0.122, side-by-side0.022, metering0.539,
  lane-vacate1.993 (ramp-present subset1.463); corresponding native0.456/0.087/
  1.492/1.997(1.461), MAPPO0.649/0.193/3.891/0.669(0.443).
  Counts/exposures and paired CIs for all arms/clocks/cells are in CSV.
  Tapes start300.5s: feature warm-in/left-censoring explicitly disclosed;
  whole-policy behaviour associations are not causal cooperation evidence.
- Paper outputs: `E:/research-dense/docs/paper_results/E5_COOPOFF_C1.md`,
  `E5_COOPOFF_C1.csv`, `E5_COOPOFF_C1_worlds.csv`; committed byte-identical
  copies,2,736world-clock rows,15,960estimates, source/input SHA256 manifests,
  scalar independent primary-bootstrap verification and provenance audits under
  `tools/autonomy_e5/receipts_done/`. Reproduce with
  `python -B -m tools.autonomy_e5.summarize` then
  `python -B -m tools.autonomy_e5.verify_report`.
