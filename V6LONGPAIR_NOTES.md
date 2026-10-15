# V6 P6: opt-in long paired branches

`loss.paired_diff.horizon_mode = "stage"` enables horizon-aware paired data. The
default, `"fixed"`, retains the previous global shortest-arm cap, sampling sequence,
legacy flat config and training logs. Paired loss remains off by default.

## Data and training

- `PairedBranchSource` measures the common available horizon **per pair group**,
  checks identical t0 slots/kinematics, and records group counts by horizon in
  `TRAIN_RECEIPT.data.paired_batches.source.horizon_buckets` (keys are 0.5 s steps).
  One short group no longer shortens every other group in stage mode.
- Each curriculum stage selects all groups covering its horizon; if none cover it,
  only the longest available bucket is used. Each paired batch has a common actual
  horizon, without padding missing future targets. Loaders remain lazy/memory-mapped.
- Stage-mode paired batches bypass the ordinary short-horizon mixture, use the
  stage's configured BPTT truncation, and log `paired_horizon_steps` and
  `paired_horizon_s`. Worker loaders are rebuilt after stage changes.
- Existing target construction caps to the actual unroll before preparing pair and
  aggregate targets/normalizers. `paired_diff.py` needs no change; future return
  terms can consume the same actual-horizon targets.
- `configs/c4_wm_v6_longpair.toml` extends v5. Set `WMS_DATA_V6_LONGPAIR` to the
  converted source. To train beyond 120 s, configure the corresponding longer
  curriculum stages and collect matching branches (e.g. 180/240/300 s).

## Collection and measured smoke

`scripts/boxops/collect_v6_longpair.sh` requires explicit `SEEDS`; the integrator
must supply newly registered, disjoint blocks. No seed registration was added.
`--population` is forwarded by `genrand_driver`; omitting it preserves old commands.
Existing collection and conversion already support longer branch durations.

One real base episode ran with seed **4107**, demand **2700**, penetration **0.2**,
population **textbook** (CAVs `cav_typical`), warmup **60 s**, horizon **240 s**,
cadence **120 s**, branch horizon **120 s**, **2 random arms + native**, one worker,
repeat check disabled. Two earlier dependency-initialization failures simulated no
steps. No additional base episode was simulated.

- **660 simulated seconds**: 300 base/warmup + 3 × 120 branches.
- **26.795 s collection + 8.847 s conversion = 35.643 s wall time**.
- Zero collisions, teleports, or dropped CAV steps.
- Raw: **2,313,561 bytes**, four NPZ episodes.
- Converted: **14,182,350 bytes**, four episodes / 56 NPY arrays.
- One paired group; each branch provides a complete **120 s / 240-step** window.
- Commands and raw evidence: `_out/v6longpair/smoke_real/SMOKE_RECEIPT.json`.
- Converted source: `_out/v6longpair/smoke_real/paired/branch/c4_episodes`.

Outputs are worktree-local to honor workspace-only execution; the brief's
`E:/agent-work/tmp/v6longpair/` location is outside this worktree.

The launcher header projects default production cost at **4,920 sim-s / ~200
worker-wall-s per base**. Its recommendation is **512 bases per box**, two 48-core
boxes with 46 workers each, approximately **110 minutes** including a 2× simulation
slowdown allowance and conversion/I/O reserve. This is a smoke-based projection,
not measured box throughput. Reserve about 80 GB raw + converted per box.

## Verification (2026-10-15)

Python 3.13, Torch CPU, math/Torch threads limited to one; no GPU training.

- `py -3.13 -B _out/v6longpair/run_tests.py`: **105 passed, 5 skipped** in 118.80 s
  across `test_v6longpair`, `test_pairloss`, `test_c4_genrand`, `test_v5data`, and
  `test_infrav5`. Existing suites were not edited. Five legacy optional smoke-data
  tests were skipped because their sources are outside this workspace.
- **37 new longpair cases** cover fixed-mode byte equality, 60/120/240/300 s
  buckets, short-arm isolation, fallback, assembly, curriculum transitions,
  persistent-worker refresh, and the real converted smoke source.
- A real model paired batch unrolls **240 steps at stage 120 s**, with finite
  nonzero backward gradients through twelve 20-step chunks; aggregate/pair
  targets and normalizers cover the full actual horizon.
- Eight launcher checks passed, including Bash syntax, mandatory seeds, argument
  routing, and conversion routing; no SUMO was used by those checks.
- Ruff passed on all nine changed Python files; `git diff --check` passed.

For portable pytest runs, select those five unit files with `python -m pytest`,
use a workspace-local `--basetemp`, and set `OMP_NUM_THREADS=MKL_NUM_THREADS=
OPENBLAS_NUM_THREADS=1`. The local runner also skips the five external-data tests.
