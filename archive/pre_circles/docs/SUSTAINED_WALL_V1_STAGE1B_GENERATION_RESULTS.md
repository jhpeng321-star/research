# Stage 1b Part B results — p20 generation stratum (restricted to p20; p30 deferred)

Date: 2026-09-03/04. Status: **development stratum; report restricted to
p = 0.20 (seeds 7001-7240). The p30 stratum (7241-7360) is registered and
unrun by order, and is explicitly deferred here.**

## Determinism gate prerequisite (recorded)

The generation ran under the verified-identical launcher after the A1/A2 gate
(see the Stage 1b gate report): a SUMO remote-port bind race was root-caused
and fixed with an inter-process launch lock
(`experiments/idm/deterministic_sumo.py`); the four-run identity check
(3 x 8-worker + 1 overlapped pair, seeds 6001-6008, D + W150/250/350)
produced **160/160 identical branch digests** before this stratum ran.

## Inventory and integrity

- Seeds completed: **240/240 (7001-7240)**; missing: none.
- Branches: **960** (4 per seed: D, W150, W250, W350), plus 12 D repeats
  (every 20th seed: 7001, 7021, ..., 7221).
- Integrity failures: **0** — exact prefix identity through t_d, first
  divergence at t_d + 1, zero collisions/teleports, verified natural
  clearance, on all 720 wall branches. Repeat identity: **12/12 exact**.
- Per-seed path: `runs/probe/stage1b_generation_p20_s7001_7240/`; summary:
  `runs/probe/stage1b_generation/p20_summary.json`.

## Recording defect (reported before fitting)

`cav_count_band`, `cav_count_upstream` and `ramp_queue_length` are **constant
0.0 across all 240 seeds at every t_d**. Root cause: the environment exposes
no public `active_cavs` attribute (only `_active_cavs`); the feature capture
used `getattr(env, "active_cavs", set())` and silently got the default. (The
same missing attribute makes `corridor_state`'s inband obs component a
silently-0 dummy inside its swallowed per-vehicle try/except.) One
per-lane record (`lane_..._1_nearest_to_band_top`) is also constant at
t = 250/350. **The per-lane `lane_*_count_in_band` / `lane_*_count_400_500m`
features and every vehicle-level count/speed/density feature vary correctly.**
The gate fit drops zero-variance columns and uses the valid geometry features.

## Boundary curve (per 100 veh/h bin, p20; n per bin in parentheses)

| bin | n | frac beneficial W150 / W250 / W350 | median Δdelay W150 / W250 / W350 (s/veh) | timing-matters |
| --- | ---: | ---: | ---: | ---: |
| 2200 | 27 | 0.04 / 0.04 / 0.07 | +4.57 / +3.45 / +2.42 | 0.04 |
| 2300 | 50 | 0.16 / 0.22 / 0.28 | +4.80 / +3.58 / +2.93 | 0.14 |
| 2400 | 37 | 0.35 / 0.46 / 0.43 | +3.45 / +1.17 / +0.59 | 0.24 |
| 2500 | 39 | 0.54 / 0.64 / 0.69 | −1.61 / −2.88 / −3.04 | 0.31 |
| 2600 | 47 | 0.45 / 0.55 / 0.66 | +2.90 / −1.36 / −2.43 | 0.49 |
| 2700 | 40 | 0.70 / 0.82 / 0.72 | −6.82 / −6.32 / −5.06 | 0.17 |

The finer-resolution boundary curve puts the beneficial-fraction sign change
between 2400 and 2500 (W250: 0.46→0.64; median +1.17→−2.88), with a
**non-monotonic 2600 pocket (W150 median +2.90 while W250/W350 negative)** —
the gate must separate seed-level rather than bin-level behaviour. The
late-engagement W350 arm dominates at 2500-2700 on fraction (0.69/0.66/0.72)
while W250 wins on median at 2700.

## Timing matters

Timing-matters fraction (best-of-three beneficial AND worst not) is **0.246
overall**, and exceeds 0.20 in bins 2400 (0.24), 2500 (0.31) and 2600 (0.49)
— the timing decision matters on a meaningful share of seeds in the
transition band, which is what triggers the Part-3 timing evaluation.

## p30

Explicitly deferred by order: seeds 7241-7360 registered under
`stage1b_generation` (role + prospective exclusion) with the p = 0.30
penetration strata in the contract; no branches have been run. The runner is
ready (`--strate p30`); the stratum would add 480 branches (~6 h at 8
workers).

## Artifacts

- `runs/probe/stage1b_generation_p20_s7001_7240/{manifest.json, seeds/}`
  (per-seed: features at t_d 150/250/350 incl. 120 s diffs and per-lane
  geometry, hidden demand, arm outcomes, paired targets, integrity, repeats).
- `runs/probe/stage1b_generation/p20_summary.json` — the tables above
  (SHA-256 at commit time).
- Code: `stage1b_generation_contract.py`, `run_stage1b_generation.py` (with
  `--demand-override` for the gate timing runs and `--allow-any-seeds` for
  the gate), `analyze_stage1b_generation_p20.py`.
