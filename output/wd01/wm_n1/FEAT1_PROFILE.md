# FEAT1_PROFILE — featurize-all bottleneck profile + safe speed-up (2026-10-01)

Branch `zcode/feat1prof` (from `zcode/feat1run3` bcf1ddc); workspace
`E:/rd-wt/feat1prof`; cwd `output/wd01/wm_n1`; interpreter
`C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe`; `WD_FORBID_D=1`
(D:/rd-staging accessed READ-ONLY via the approved override root
`D:/rd-staging/wmn1_feat` only).  The live run (launcher 14980, featurizer
32340, 12 workers, logs `D:/rd-staging/wmn1_feat/logs_v3`, shards
`D:/rd-staging/wmn1_feat/shards_v2`) was never stopped, signalled or
written to.  Zero SUMO.  All offline re-featurization ran on E: tmp
(`_feat1prof_tmp/`), 1 process, BELOW_NORMAL, free-virtual >= 8 GB gate;
TRAIN runs only (chunk1 seeds 8439/8443/8447 — outside the sealed chunk-5
block 8707-8736); dispatcher_rows/insertions/evaluate_summary/suffix_summary
never read.

## 1. Live-run observation (read-only)

Process/CPU (Get-Process/Get-CimInstance, ParentProcessId=32340):

- 12 python worker children (respawned per 12-run batch).  During batch
  execution each worker sits at ~99-100 % of one core (4x10 s samples:
  ~120 s cumulative CPU per ~10.6 s wall across 12 workers).
- Parent featurizer 32340: ~0.1 % CPU over a 63 s window — the coordinator
  is idle; there is NO serial parent-side section.
- One 20 s sample that straddled a batch boundary showed 2.5 %/worker
  (turnover, not the steady state).

Memory (Win32_PerfFormattedData_PerfOS_Memory / Win32_OperatingSystem):
free RAM 12.3-13.8 GB of 31.6; commit 36.2/57.7 GB (63 %); free virtual
~21.6 GB.  `shards_v2/ALLOCATION.json`: `worker_gate_events` = 0,
`n_gate_drops` = 0 and no `worker-gate:` lines in `logs_v3/featurize.log`
— the memory gate NEVER dropped workers.

Disk (Get-Counter PhysicalDisk, 3x2 s): E: reads burst 63-69 MB/s at each
batch start, then ~0; D: writes <= ~5 MB/s peaks (mean far lower).  I/O is
idle while runs compute.

Per-run timing (DONE.json / IDEA_A_DONE.json mtimes, read-only): DONE
count 420 -> 492 -> 504 (T+23.5 min -> T+33.6 -> T+46); ~440 runs/h in the
last 30 min.  Batches of 12 complete in clusters every ~50-110 s.  Within
each shard, IDEA_A_DONE lags DONE by 40-54 s — the Idea A pass alone is
40-54 s of the ~50-110 s batch slot.

## 2. Offline cProfile (1 process, BELOW_NORMAL, E: tmp)

Run `chunk1/w8447m2700p10t480` ... `w8447m2700p10t240__native` (TRAIN,
43.1 MB physics, 1200 rows / 1201 ticks, 589 vehicles); pstats dumps in
`_feat1prof_tmp/pstats_*_v1.pstats` / `_idea.pstats`.

v1 pass (`features.featurizer.build_shard`, profiled 10.65 s):

- `parse_run` 3.62 s (json `raw_decode` 2.60 s; Python row loop 1.16 s);
- `parse_departures` on the base-history run 3.20 s (a second full
  parse of the base arm's physics file);
- one-time imports (pandas/torch) ~3.1 s per process (each batch spawns
  fresh workers, so this is paid per run in the live driver);
- per-tick wall/action bookkeeping + writes: remainder; `write_shard`
  0.18-0.25 s.

Idea A pass (`features.idea_a.idea_a_channels_from_shard`, profiled
37.94 s) — `scene_baseline_channels` is 37.52 s of it:

- `_lane_sorted()` 244,614 calls (≈204/tick) = 25.4 s cumulative
  (list.sort 14.0 s + roster listcomp 10.5 s + sort key lambda 5.1 s);
- `_idm1` 557,993 calls 2.48 s; `_engine()` 1,200 calls 1.55 s (per-tick
  `Path.resolve` import-machinery probe 1.38 s); `_seg_limit1` 1.10 s;
  `_own_leader_follower` 0.99 s.

Plain timings (no profiler): whole-file JSON decode of physics_steps.jsonl
0.64-1.35 s (66->32 MB/s under live-run load); raw file read 0.015-0.064 s;
Idea A 5-array write 0.05-0.09 s.

## 3. Diagnosis

Per-run time is CPU-bound inside ONE worker-local serial section: the Idea
A per-tick Python baseline (`scene_baseline_channels`), dominated by
needless re-sorting of the three per-lane rosters (~204 full sorts per
tick, re-sorted on every candidate decision even though the rosters can
only change when a lane change is ACCEPTED).  It is NOT I/O (disks idle
after the batch-start read burst), NOT the parent (0.1 % CPU), NOT a lock
or a parent-side serial backfill (Idea A runs inside each of the 12
workers in parallel; the 40-54 s DONE->IDEA_A_DONE lags overlap).  The
proposed "rollout2 batched baseline" does NOT exist on this branch
(`rollout2/step.py` is a P3 stub), so a vectorized rewrite is not
available here; the measured hotspot is fixed instead.

## 4. Fix (opt-in flag, byte-identical by construction)

`features.idea_a.scene_baseline_channels(..., fast=True)`:

- caches the per-lane sorted rosters for the tick and rebuilds them ONLY
  after an accepted merge/discretionary change mutates `(r_lane, r_s)`
  (the two acceptance sites mark the cache dirty); a reused cache is by
  construction the identical list a fresh `_lane_sorted()` returns;
- memoizes the frozen `partA_v2g.engine` module (skips 1,200 per-tick
  `sys.path`/resolve probes).

`fast=False` (default) is the untouched reference path.  Threading:
`idea_a_channels_from_shard(..., fast=)` -> `corpus.idea_a.
write_idea_a_channels(..., fast=)` -> `_run_one` job key `idea_fast` ->
`corpus.featurize_all(..., idea_fast=)` -> `python -m corpus featurize-all
--idea-fast` (recorded as `idea_fast: true` in ALLOCATION.json) and
`python -m corpus.idea_a backfill --fast`.  IDEA_A_DONE.json schema and
receipt content unchanged, so mixed trees (slow-written finished shards +
fast remaining runs) resume/verify identically.

## 5. Byte-identity test (>= 3 TRAIN runs, vs live shards_v2 READ-ONLY)

`feat1prof_tools/byte_identity_check.py` re-featurized into
`_feat1prof_tmp/ident` with the fast code and compared every file against
`D:/rd-staging/wmn1_feat/shards_v2` via the receipts' sha256 maps:

| run (TRAIN, chunk1)                       | comparisons | result |
|-------------------------------------------|------------|--------|
| w8443m2600p10t480__switching1 (branch arm)| 20         | ALL equal |
| w8447m2700p10t240__native                 | 20         | ALL equal |
| w8439m2700p10t300__base (t_origin=0)      | 21         | ALL equal |

`BYTE-IDENTITY: 3/3 runs ok, 61 sha comparisons` — 12 v1 arrays +
slot_vid.json + meta.json (+ outcomes.parquet) via DONE.json `outputs`,
5 Idea A arrays via IDEA_A_DONE.json `arrays`, plus input_physics sha
equality.  (A first attempt failed ONLY on meta.json `policy` — a harness
artifact; fixed by taking `policy` from the inventory row like the real
driver.)  In-process slow-vs-fast array equality is also asserted for
every fast timing run below and by 8 new unit tests (48 seeded random
scenes incl. accepted merges/MOBIL changes, synthetic shard).

## 6. Measured per-run speed-up (same process, BELOW_NORMAL, live run active)

| pass                              | slow      | fast      | speed-up |
|-----------------------------------|-----------|-----------|----------|
| Idea A build (switching1, same-proc) | 33.38 s  | 10.01 s   | 3.33x    |
| Idea A wrapper (native, build+write+sha) | 28.47 s | 9.68 s | 2.94x    |
| harness run total (native: v1 build+write + idea) | 48.83 s | 20.44 s | 2.39x |

Live per-run Idea A cost is 40-54 s of a ~50-110 s batch slot; applying
the flag to the REMAINING runs should land throughput around ~1.8-2.2x
(~335-440/h -> roughly 600-800/h).  The finished 500+ shards are never
recomputed (skip-by-sha).  Not applied to the live run — integrator
decision.

## 7. Safe procedure to apply to the remaining runs

1. Let the live run finish OR stop it with the owner's own documented
   procedure (this task did not touch it).  Finished shards need nothing.
2. Take this branch's commit (code + tests) into the run workspace.
3. Re-issue the EXACT resume command from FEAT1_STATUS.md with ONE added
   flag: `--idea-fast` (skip-by-sha leaves every finished shard
   untouched; only remaining runs are featurized, with the fast
   baseline).  ALLOCATION.json then records `idea_fast: true`.
   Alternatively `python -m corpus.idea_a backfill --data-root ... --fast`
   for an already-v1-complete tree.
4. Verify as usual (`corpus verify`, `corpus.idea_a verify`); receipts and
   schemas are unchanged, so mixed slow/fast provenance is invisible to
   verification.

## 8. Test evidence

- `pytest -q features corpus` (WD_FORBID_D=1): **140 passed** in 222.5 s.
- New tests: `features/tests/test_idea_a_fast.py` (fast==reference exact
  equality), `corpus/tests/test_idea_fast_flag.py` (flag plumbing).
- `ruff check` on all touched files: clean.
