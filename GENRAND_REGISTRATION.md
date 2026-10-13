# GENRAND registration draft (for the integrator) - C4-GENRAND-20261014

Status: DRAFT, branch c4/genrand. The integrator must register the range and finish the
holdout check before collection starts.

## Purpose

This is undirected world-model TRAINING data. The owner rule (2026-10-14) is that the WM learns
action effects only from native driving, generic random direct-output commands and the
learned policy's own rollouts.

The generator is `collect.py --policy generic_random`, speed mode (the default). It uses no
designed controllers and makes no spatial or behavioural selection. Each episode is one SUMO
world. On top of that, a subset of worlds has paired branches every 60 s: M = 4 re-seeded
random arms plus an all-native reference, each 60 s long. The arms start from one saved state
and share common random numbers. A later paired-difference loss can group them by
`pair_key = <world>@t<branch time>`. That loss is not implemented here.

## Seeds

- **Range:** `961000-961999` (code: `traffic_shepherd.seeds.C4_GENRAND_SEEDS`).
- **Loadings:** 2400 and 2700 veh/h only (`C4_GENRAND_LOADINGS`).
- **Guards:** the C4 guard is `traffic_shepherd/c4/seeds.py`. It asserts that the range does
  not intersect the sealed, final-test, chunk-5, TEST_REPL, PL-WM1, CV28-holdout, WM28 or
  WM34 blocks.
- **Use:** TRAINING only. These seeds must never be used for evaluation, CAL, HOLDOUT or
  planner scoring.

Searched so far (a seed token = 960000-961999 not inside a longer hex or number):

- Code registry: `traffic_shepherd/seeds.py`. The highest registered range ends at 930031;
  nothing is registered in 94xxxx-96xxxx.
- Main repo, files with extensions json, jsonl, py, toml and md:
  - `protocols/`, including `experiment_history.json`
  - `runs/active/**`, including every `eval_seeds.json`, `COLLECT_SUMMARY.json`,
    `INDEX.json` and `MANIFEST.json`
  - `tools/`, `traffic_shepherd/`, `src/`, `configs/` and `docs/`
  - Result: no hits.
- Main repo `output/`, `data_ops/`, `scenarios/` and `baselines/` (also yaml and txt files):
  4 files matched, all incidental (byte counts, timestamps, delay totals). None is a seed
  field.
- `E:/agent-work/**` files named `eval_seeds.json`, `COLLECT_SUMMARY.json`, `INDEX.json`,
  `MANIFEST.json` or `*.toml`: no hits.
- Files whose names contain restricted tokens were excluded from every search.
- **Still open: the holdout check.** `tools/holdout_guard.py` was not opened (the hard rule
  forbids it). Its public functions work on world roots (`filter_final_holdout(worlds)`,
  `excluded_roots()` ...). They offer no seed-membership check. **The integrator must run the
  holdout-exclusion membership check for 961000-961999** (or for the planned world ids
  `w9610xxm{2400,2700}p{05,10,20,40}t300`).

The laptop smokes used the exposed development seeds 4107-4110. No GENRAND seed has been
simulated yet.

## Budget (proposal)

| part | seeds | episodes | 60 s windows |
|---|---|---|---|
| plain generic_random | 961000-961479 | 480 | 7,200 non-overlapping (about 40.8k at 10 s stride) |
| branched generic_random | 961480-961515 | 36 | 540 base + **2,520 paired-branch** (36 x 14 points x 5 arms) |
| total | 516 | 516 worlds | about 10,260 (about 43.9k base windows at 10 s stride + 2,520 branch) |

- **Cases:** `--case-plan weighted-cycle` gives one case per seed. Penetrations
  0.05/0.1/0.2/0.4 have weights 1/1/2/2, and demands are 2400 and 2700. Every block of 12
  consecutive seeds contains every combination.
- **Timing:** warmup 300 s, horizon 900 s; branch cadence 60 s (points at 360..1140 s),
  branch horizon 60 s, 60 s pre-history per branch.
- **Machine:** Tencent 48-core spot box, 46 workers, OMP/MKL/OPENBLAS threads = 1.
- **Estimated wall time:** about 1.0 h at laptop core speed, about 1.3-1.5 h with slower
  cores. Total CPU is about 45 core-h. Estimates come from `genrand_driver --estimate-only`.
- **Disk:** about 9 GB raw (about 2 GB with `--no-physics-log`) plus about 15 GB converted (f32).
- **Output directory:** `runs/active/C4-GENRAND-20261014/` containing `plain/`, `branch/`,
  `c4_episodes/`, `GENRAND_MANIFEST.json` and `COLLECT_SUMMARY.json`.
- **Command:**
  `scripts/genrand/collect_genrand.sh runs/active/C4-GENRAND-20261014 961000-961479 961480-961515 46`
- **Remaining seeds:** 961516-961999 stay unused, reserved for later GENRAND rounds such as
  policy-rollout pairing.

## Comparators / closest prior

- **Closest prior:** C4-LOOP-SMOKE-20261012 (`--policy random`, RandomAccelPolicy, exposed
  seeds). The actiondata audit (E:/agent-work/tmp/actiondata) found these problems in earlier
  data:
  - policy commands overridden 57-59 % of the time;
  - holds of about 100 s;
  - CAV-CAV correlation 3-6x the training data;
  - no lane commands;
  - p40 under-covered.
- **What this data adds:**
  - commands with SUMO override recorded (requested vs issued vs realized);
  - 2-60 s holds;
  - controlled shared-latent correlation (rho ~ U(0,1));
  - random lane intents;
  - weighting toward p0.2/p0.4;
  - paired branches for action-effect differences.
