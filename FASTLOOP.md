# S-FASTLOOP — prepared, NOT armed

Base **9cc39f15** (native-lane ON); merged **c3e4034b** S-SPEED2. All seven
`configs/c4_fastloop_options.json` switches default **false**:
`speed2_policy`, `two_box`, `overlap_policy`, `spot_resume`, `cpu_checks`,
`repoint`, `base_three`.

## Changes and evidence

- Policy-only exact SPEED2 bundle: `WMS_SPEED_PIVOT_DIAG_OFF=1`,
  `WMS_SPEED_GRAD_AUDIT_EVERY=10`. WM-FT explicitly disables inherited SPEED2
  acceleration. No numerical compaction/replay acceleration is enabled.
- Expand the original weighted-cycle case plan before deterministic alternating
  cpuP/cpuQ assignment. Seven concurrent one-world workers/box, configurable up
  to 14; one numerical thread each. CPU conversion/checks retain ordinary
  manifests, authoritative returns, calibration/credit and acceptance rules.
- Policy k+1 starts during collection/conversion k using the accepted WM **at
  launch**, previous policy and latest completed returns/credit. Hashes and frozen
  horizon/panel are recorded in `overlap_cNN.json`. GPU work joins before WM-FT.
  Acceptance/rejection k never changes that policy's WM: collection k+1 uses its
  recorded `policy_wm`; WM-FT uses the latest accepted WM. Stops prevent promotion.
- Completed worlds are hash-verified on gpuR before acknowledgment. Explicit
  `loop14.py retry --config CONFIG collect` retains successes and replays missing
  worlds on their peer; check retries alternate boxes. Charges precede dispatch,
  survive coordinator crashes, and include failed attempts in full. Caps still refuse.
- Read `c4-newcpu/MAIN14_REPOINT.md`: cpuX becomes cpuP; exact runtime deployment
  is required. All three qualifications must exist. Among {13E,14b,16}, reject
  candidates with any new reaction sign/magnitude FAIL versus 13E; minimize strict
  all-pair TTC ratio; ties choose newer. Scored/deployed checkpoint hashes must match.

**Measured:** gpuB native-lane/wm_law/head synthetic policy update **7.013 →
2.034 s (3.45×)**, identical updated weights, owned GPU peak **634 MiB**.
**404 CPU tests pass**; **178 flag-off tensors** match 9cc39f15 storage bytes.
One-world CPU dry run exercises real conversion/merge/return export, **0 SUMO s**.
Receipts: `tools/fastloop/{UNIT_TESTS,FLAG_OFF,DRY_RUN,POLICY_TIMING}.json`.
The **5–7 h** cycle target remains unmeasured; two-box/overlap throughput is not
claimed as a measured speedup. No smoke reservation was needed.

## Exact later re-arm procedure (owner, after verdicts)

1. Fill `CPU_CONTRACT.template.json` from completed S-NEWCPU setup: tested fixture
   directories, TRAIN roots/environment, actual interpreter, and completed round-16
   checkpoint/config copied into **owned gpuR scratch**. Release scoring slots.
   Gracefully disarm the old waiter and verify its exact incarnation exited;
   preserve receipts. An initialized old loop refuses fresh seed reuse. Resolve
   gpuR disk headroom: observed ~1.3 GiB is below the inherited 1.5 GiB floor.
2. From this clean committed worktree:
   `python tools/fastloop/stage_release.py --cpu-contract FILLED_CONTRACT.json`
3. On gpuR (these commands are **not executed now**):

```bash
R=/root/autodl-tmp/c4-fastloop-sfastloop
P=/root/miniconda3/bin/python
$P $R/tools/fastloop/prepare.py --main-arm /root/autodl-tmp/c4/v5/main14_arm/ARMED.json --cpu-contract $R/CPU_CONTRACT.runtime.json --out $R/arm --deadline 2026-10-26T18:00:00+08:00 --enable --retry-reserve
$P $R/scripts/c4/main14_autolaunch.py --arm $R/arm/ARM_REQUEST.json --check-only
```

4. Append the following amendment and generated code/config/registration hashes
   to the active register **before** starting the waiter:

> C4-LOOP14-COOPON-20261026 / S-FASTLOOP: enable the seven prepared switches;
> native-lane stays ON. cpuP/cpuQ, <=14 workers each. Base: complete 13E/14b/16
> verdicts, no new reaction FAIL vs 13E, best strict TTC ratio, newer ties.
> Expiry 2026-10-26 18:00 +08:00. Primary 13 worlds/29,133 sim-s per cycle;
> add <=29,133 retry sim-s/cycle. Combined caps 58,266/cycle, 233,064 total.
> Replays reuse worlds; all attempts charged. Seeds/exclusions, four-cycle limit,
> policy/WM budgets and every behavioral, safety and acceptance gate stay fixed.
> Overlapped policy WM/returns provenance is immutable and recorded. This amendment
> supersedes operational scheduling/base selection only; preparation consumed 0 SUMO s.

5. Then, on gpuR:

```bash
setsid nohup $P $R/scripts/c4/main14_autolaunch.py --arm $R/arm/ARM_REQUEST.json </dev/null >>$R/waiter.log 2>&1 &
```

Source checks, tests, real-WM CPU smoke, frozen-panel and GPU gates repeat before launch.
