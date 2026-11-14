# C1 v2 state sets with t0 entrance counters (re-materialisation)

C1-WMS-C1FAST-20261007 scored its states with `--entries` OFF because the
materialised C1 state files predate the counter channel: every record carried
`"counter": null`, and the wms/pure_physics specs refuse to synthesise entries
from nothing. This branch already fixed the source: `materialize.build_state`
copies the pack's RSU-observable t0 entrance counter `[2, 30]` into each record
(`counter` read AT the snapshot tick, never a future arrival), so any
re-materialised state set carries it and `wms_eval run --entries` becomes
available for C1 v2 scoring.

## Rebuild command (exact, CPU only)

Run from the wms_eval package root with the rl_driving interpreter. The source
packs and identity sidecars are the ones referenced in
`E:/rd-wt/ws-c1eval/_out/C1_EVAL.md`; the clean-set selection
(U-same / U-new / SEEN, fail-closed separation) is `wms_eval/c1sets.py`.

```powershell
$py = C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe
cd E:/rd-wt/ws-evalh/output/wd01/wm_n1
& $py -m wms_eval materialize `
  --pack pack_v3=D:/rd-staging/wmn1_feat/pack_v3 `
  --pack pack_dspeed=E:/rd-wt/dspeed-data/_out/pack_dspeed `
  --pack pack_damping=E:/rd-wt/damping-snap/_out/pack_damping `
  --identities pack_v3=E:/tmp/wmn1_train_runs_20261002/sidecars3_20261003T0005CN/identities `
  --identities pack_dspeed=E:/rd-wt/a9/_out/identities_dspeed `
  --identities pack_damping=E:/rd-wt/damping-snap/_out/identities_damping `
  --pi-states E:/rd-wt/ws-p5/_out/lpo/pi_eval_states.json `
  --windows-val E:/rd-wt/ws-p5/_out/lpo/windows_val.json `
  --train-windows E:/rd-wt/ws-p5/_out/lpo/windows_train_noPI.json `
  --unseen-u-same E:/rd-wt/ws-evalh/_out/c1_states_v2c/unseen_u_same_world.json `
  --unseen-u-new  E:/rd-wt/ws-evalh/_out/c1_states_v2c/unseen_u_new_world.json `
  --seen-v2       E:/rd-wt/ws-evalh/_out/c1_states_v2c/seen.json
```

Outputs land under `_out/c1_states_v2c/` (gitignored output area): the three
state files plus `BUILD_STDOUT.json` (per-set counts and errors),
`MANIFEST.json` (sha256, world lists, separation report) and
`SEPARATION.json`. Packs stay memory-mapped and windows are materialised one
at a time; peak RSS stays a few GB, well under the 6 GB ceiling.

## What was executed and verified (2026-10-07)

* 140/140 U-same-world PI states, 21/21 U-new-world PI states, 157/189 SEEN
  states written (the 32 dropped SEEN windows are the known origin-0
  `base`/`native_base` arms with degenerate 1-2 vehicle snapshots; refused,
  not scored).
* All 318 records carry a non-null `counter` `[2, 30]` (streaming substring
  check: records == counter-bearing records == 140/21/157).
* Separation is clean (`overlap_ok: true, refused: false`): no U-same/U-new
  world overlap, no seen/U-same world overlap, no training-window key/run leak
  in any set; seen x U-new world overlap (21 worlds) is the documented paired
  tier, reported separately, not mixed into the strict count.
* State counts match the scored C1-WMS-C1FAST-20261007 run exactly
  (140 / 21 / 157), so re-scoring with `--entries` is directly comparable.

## Scoring with entries and the standard horizon report

```powershell
& $py -m wms_eval run --model wms --states <c1_states_v2c set json> `
  --ckpt <wm_s.pt> --physics <TRAIN_RECEIPT.json> --out <run dir> --entries
```

`--entries` turns observed-counter vehicle entries on for the wms spec and the
pure_physics/A11 baselines (fair comparison); `keep_speed` ignores entries.
The per-state result files (`states__<model>__<set>.json`) are then summarised
by the standard report every future model is scored with:

```powershell
& $py -m wms_eval horizon `
  --states wms=u_same=E:/rd-wt/ws-c1eval/runs/active/<RUN>/states__wms__u_same.json `
  ... `
  --reference keep_speed --bootstrap 1000 --seed 0 --out _out/HORIZON_REPORT_C1.md
```

It reports, per model x set x tested arm (PI arm for the unseen sets, native
for SEEN) and horizon 10/30/60/120 s (300 s when present): the HDV `|ds|`
median across states with IQR, p90, sd, the median within-state p90, the
blow-up rate (HDV median |ds| > 50 m at 60 s AND > 100 m at 120 s; truncated
states are excluded and counted, never scored as survived), and a paired
bootstrap over worlds vs the reference model (states paired by world + t0).
`scan_states` in `c1sets.py` counts records by the `initial_state` marker so
manifests stay correct for counter-bearing records whose `world_meta` repeats
the `"world"` string.
