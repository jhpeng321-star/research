# S-RB-RUNA handoff

Updated 2026-10-31 23:57 +08:00. Branch `c4/rb-runa`, base `f0db3c82`.

## Status
- Read `rb_common.md`, owner registration (5683/5708), all four WHY reports,
  EVAL design and published CLI. No READY FOR EVAL variant at first poll.
- gpuA verified 23:51:48: RTX4090D, 1/49140 MiB, 0%; load6.94.
  Initial SSH closed; nonmultiplexed SSH succeeded. Local NVIDIA unavailable.
- Durable ten-minute watcher implemented, deadline 2026-11-01 07:00 +08:00.
- RUNA claims first unclaimed variant in published order using exclusive-create
  `<variant>.claim`. Claims and failures are retained; never retry scientific jobs.

## INT / EVAL: dispatch contract requested
Please link a JSON launch specification in your `READY FOR EVAL` section, e.g.
`- WM-PHYS-A: E:/agent-work/rb-shared/evalclaims/WM-PHYS-A.runa.json`.
This lets the detached runner execute newly published variants without another
chat turn. It requires a staged gpuA runtime and the exact approved INT loader.
See `E:/agent-work/rb-shared/evalclaims/RUNA_PROTOCOL.md` for fields.
RUNA will stage a NEW immutable copy of the published EVAL harness, run three
5 GiB inference shards (aggregate own-process guard 18,500 MiB, under20GB),
then invoke its frozen scorer. No duplicate WM2 run.

## Results
- `py -3.13 -B -m unittest tools.rbruna.test_runner -v`: 5 PASS /0.55s,
  covering concurrent claims, READY ordering, plain-text peer claims, all160
  completion hashes/corruption, and refusal to signal a reused PID.
- `py -3.13 -B tools/rbruna/watch.py --once`: initial poll23:56:35, no READY specs.
- gpuA fixed-input audit23:56: existing wmcalib starts/expected hashes, policy190,
  MAPPO binding, baseline-controller runtime and python dependencies all present.
  RUNA output root is NEW; no copying/extraction of existing recordings needed.
- No candidate inference or efficacy result yet. No SUMO/training/peer signals.
