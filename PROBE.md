# S-PROBE — preparation only

Default OFF. Main enables with S-LOOPLONG at the cycle-2→3 boundary.
No SUMO, driver/trainer launch, deployment, gate change, or push was performed.

## Enable (main only)

Install this commit on the driver and both collectors. For **each** arm, put its
accepted-WM-rescored TRAIN real-pair CQR artifact at `ARM/probe_cqr.json`.
From the installed source root:

```sh
python tools/probe/prepare.py --arm /root/autodl-tmp/c4/restart20261026/arm --enable
python tools/probe/prepare.py --arm "$E5_COOP_OFF_ARM" --enable
```

`E5_COOP_OFF_ARM` must be main's registered coop-off arm directory; no twin
location was supplied. Both commands apply identical `collect.probe` settings:
`enabled=true`, `from_cycle=3`, arm-local calibration. They start nothing.
Without `--enable`, they write `CONFIG.probe.json`. Enable refuses missing/stale
CQR, missing long reservations, or a state already containing cycle 3.

## Registered selection / budget

Exactly one of each collection world's existing SINGLE/GROUP 300s slots is
eligible: 50%, alternating `(world_index+cycle)%2`, invariant across retries and
coop settings. All first-four guard worlds remain probe-free. Missing eligible
actions/calibration retain the original arm; skipped reasons are exported.
No additional branch/native rollout: **zero extra sim-s**, existing 7,800/8,000
long reserve and append-only retry charges retained, within the owner's 280,000
caps. No budget/registration ceiling is raised.

Ten proposals: eight learned actor samples, two registered absolute-zero holds;
30/60s persistence, single units or correlated multi-CAV units. Existing learned
lane/driver/safety execution remains authoritative. K=8 shared lane/merge futures
produce native-minus-action road-cost distributions. Negative-mean, nonzero-spread
candidates rank by calibrated optimistic 300s benefit; no positive-bound cutoff.
One-sided 90% split-CQR fits uncertainty scale against full source-plus-road real
differences, using whole-world maxima and disjoint hashed fit/calibration worlds. Fewer than
nine calibration worlds yields no finite bound. Collisions immediately terminate
the probe and suppress later probes in that world; teleports are recorded.

## Merge points / verification

Collection edits: `PairExtras`, CLI flag plumbing, `loop14_fast._world`, artifact
export. S-LOOPLONG consumes separate `artifact/probes` paired trajectories and
`PROBE_POINTS.jsonl`; connect these to long-loss/paired-critic TRAIN ingestion,
and supply accepted-WM rescored calibration rows to `tools/probe/fit.py`.
Policy/gate/guard manifests exclude probes. `PROBE_REPORT_cNN.json` reports
realized-minus-predicted benefit by action type at 60/120/300s.

CPU saved-cycle-1 replay: seed968157/t360, 61 vehicles, ten SINGLE proposals,
eight futures, 346.8s, zero SUMO; GROUP ineligible. Missing real CQR correctly
retained the original arm. Evidence: `tools/probe/DRY_RUN_RECEIPT.json`.
Collection/unit suite passes; Ruff passes (existing collector E402 excluded).
