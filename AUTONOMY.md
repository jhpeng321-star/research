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
