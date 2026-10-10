# R3 scoring harness (`_out/r3_score/score_round.py`)

CPU only (no training, no SUMO, no sealed data). Scores one candidate run dir against the current best (old C1 struct,
`E:/research-dense/runs/active/WM-S-C1FAST-20261006/box/full_struct`). CAVs are teacher-forced from the recorded arm,
entries OFF (default), each checkpoint with its own `TRAIN_RECEIPT.json` physics. Metric: across-states median of the
per-state HDV median |pred - recorded| (arm `wall`; `native` for `seen_controls`). Negative paired difference favours the candidate.

```
PY=C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe
cd E:/rd-wt/ws-r3-score2
$PY -I _out/r3_score/score_round.py --candidate <run_dir> --quick            # fixed 20-state subset, wall_same + wall_pilot
$PY -I _out/r3_score/score_round.py --candidate <run_dir> --quick --sets wall_same   # primary only (fastest)
$PY -I _out/r3_score/score_round.py --candidate <run_dir> --full             # all states of each set
```
Options: `--no-stages` (skip per-stage checkpoints; large speed-up), `--reference RUN_OR_PT` (default C1), `--sets wall_same,wall_pilot,seen_controls`, `--workers N` (hard cap 2),
`--toggles projection,merge_cal` (extra candidate rows with inference-time projection / merge_cal on),
`--history-states DIR`, `--history-cut-s 4.5`, `--entries off|v1|v2`, `--out NAME` (dir under `_out/r3_score/`),
`--limit N`, `--fingerprint-limit N`, `--n-boot`, `--synthetic-history` (smoke only).
Run dir contents used: `wm_s.pt` (cand_last; falls back to `wm_s_best.pt` if absent), `wm_s_best.pt` (best-vs-last),
`wm_s_stage*.pt` (per-stage at 60 s, always on the quick wall_same subset), `TRAIN_RECEIPT.json` (physics flags; if absent the
reference's flags are used and a NOTE is printed in the report). Checkpoints are read-only staged copies in `_out/r3_score/_ckpt`.

`--quick` subset = per set, sort by (world, t0_s, key), first state per world, first 20 worlds (for wall_same this is the
registered bisect subset). `--history-states DIR`: each set uses `DIR/<set>.json` when present, for BOTH candidate and reference. Supported dirs:
`E:/rd-wt/ws-r3-evalstates3/_out/wallc1_states_h60` (v2-identical states/futures, history_s=60 declared, available_span 5 s,
run_start_limited=true, masked: the FAIR comparison set vs C1) and `..._h60_real` (real 60 s history; same flag). A
`history_s` checkpoint scores on masked/short history as long as the state declares history_s; it exits with a clear message
only when a state has no declared long history (default v2 states). `--history-cut-s S` adds a `cand_hcut` row: the candidate with pre-t0 history masked to the last S s
(difference cut-full on the primary set).

Outputs in `_out/r3_score/<out>/`: `ROUND_SCORE.md`, `ROUND_SCORE.json`, `fingerprints.json`, per-model `states__*.json`.
Report contents: HDV median/IQR/p90 |ds| and |dv| at 10/30/60/120 s; blow-up rate (|ds|>50 m@60 s, >100 m@120 s); paired world
bootstrap (cand-ref, best-ref, best-last, toggles-ref); per-stage 60 s table; fingerprints (mean/p95 |g*delta_a| using the
pre-bound residual, also valid for the soft-bound path; cap-bind rate; predicted vs recorded lane-change rate; wrong-lane
share of |ds| at 60/120 s; signed speed bias; jam vs free-flow slices, where jam = any recorded HDV < 5 m/s in the horizon;
|ds| on recorded-slow HDVs).

Caching: deterministic results (checkpoint sha + physics + state identities) are cached in `_out/r3_score/_cache`, so the
reference (and a repeated checkpoint) is scored once; failed arms are never cached. Delete nothing; stale entries are harmless.

## Fingerprint windows and reconciliation with BISECT_RESIDUAL (2026-10-10)
Step-level fingerprints (g*delta_a, cap-bind, lane-change rate, speed bias) are reported for the FIRST 60 s of the rollout
(primary; same window as the dedicated diagnostic) and, in a second block, the full 120 s rollout (later steps are dominated
by drifted states: C1 mean |g*da| is 1.8 in the first 60 s vs ~7.5 over 120 s). |ds| slices are still at 60/120 s. The
dedicated diagnostic (`ws-bisect-residual/_out/BISECT_RESIDUAL.md`) differs from the harness default in two ways: it uses a 60 s
rollout and 10 states, and it scored `sorted(recorded)[0]` per state (the NATIVE arm for 3 of its 10 states) instead of the
registered `wall` arm. `fingerprint_model(..., arm="first_sorted")` reproduces it exactly; regression test
`_out/r3_score/test_fingerprint_bisect.py` (about 2 min) pins the C1 values: mean |g*da| 1.765, p95 3.542, mean 0.612,
frac>0 0.441, 646 predicted / 950 recorded lane changes (0.84 % / 1.23 % per HDV-step), speed bias -0.236, wrong-lane share
0.334, 77 847 HDV-steps. Run: `python -I -m pytest -q _out/r3_score/test_fingerprint_bisect.py --basetemp _out/r3_score/_pytest_tmp -p no:cacheprovider`.

Reconciled old-C1 fingerprints (wall arm, 20-state quick subset, from the 120 s rollout): first 60 s window mean |g*da| 1.76,
p95 3.42, predicted lane-change rate 0.98 % vs recorded 1.18 %, speed bias -0.31; full 120 s window 7.46 / 34.2 / 8.49 % vs
1.55 % / -1.14. The earlier "7.46 / 0.0849" were full-120 s numbers. A 4.5 s `--history-cut-s` is a no-op on
`wallc1_states_h60` (its history is already only ~4.5-5 s, masked); it only bites on `..._h60_real`.
