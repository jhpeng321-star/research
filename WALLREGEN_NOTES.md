# WALLREGEN — unseen-wall WM evaluation sets in the new driver population

Branch `c4/wallregen` (from `c4/int` 1b249be). Purpose: the WM's key evaluation (prediction
on UNSEEN wall actions; walls never enter WM training) uses state sets built from the OLD
3-type population (`data_cache/wallc1_states_h60` wall_same 67 / wall_pilot 53,
`data_cache/wallc1_states_h60_real`, scored by
`runs/active/WM-S-C4-20261013/score_refresh1/score_c4.py`). This pipeline regenerates
equivalent sets in any `CavSumoEnv` population (`population="textbook"` from c4/popfix) on a
fresh, registered seed block.

## What the old sets are (traced, read-only)

* Source: WM-59 walls pilot on chunk-4 worlds (ledger `WM-59`), restored from each world's
  chunk-4 `__base` t0 saved state, 4 single-wall arms x 600 s, non-members native,
  reference = the chunk-4 true-native sibling. Packed into `pack_v3`, materialised by
  `src/wms/eval/states.py` (`build_state`).
* Wall law: **`wall-v2.1-closure`** (`traffic_shepherd/wall_group.py`, sha256 LF
  `6e0a0805...`, Amendment B pin) - NOT v2.2. v2.2 (`wall_group_v22.py`) was frozen on
  2026-10-01, after the pilot registration; the WM-59 ledger entry records v2.1. The C4 env's
  own `controller="wall"` (`baselines.WallLaw`, NumpyWallExecutor v2.2 shape, other CAVs
  damping, harness formation rule) is a different arm and is not used here.
* Arm rules (pilot as run; `output/campaign_walls_v1/tests/wall_v21_pre/`): arms
  `w-{lo,mid}-{early,late}`; speed 0.60/0.85 x members' mean reported speed (floor 2.5 m/s);
  request early = first tick >= t0 with a feasible pair, late = first feasible tick >= t0+60,
  not_feasible after 120 s; pair = 2 CAVs on entry edge 1191554636, different lanes, span
  <= 75 m, road_left(front) >= v_wall x 80 s, minimal span then most road left; hold
  H in {60,120,240} s from assembly; road-end release (as coded, road_left uses the lane
  position of whatever edge the member is on, so in practice members leave the network ->
  `member_left_network` abort; the old wall_same has 10 such aborts and 0 road-end releases -
  reproduced faithfully); safety aborts hand back to native.
* State rules: `h60` t0 = branch start + 4.5 s (shift 9), 11 history rows (5 s) declared
  history_s 60 / `run_start_limited` true, recorded wall future 240 ticks of the t0 roster,
  native future included only when the native t0 state is identical (`paired`);
  `h60_real` t0 = branch start + 60.5 s with 121 history rows (60 s). wall_pilot = wall_same
  minus aborted_in_assembly / not_feasible (51 of its 53 keys are in wall_same).
* Training exclusion: walls never in WM training (wall arms removed by run in
  `data_cache/lpwo_v3/excluded_runs_PI_wall_v3.json`; C4 data rule = native + random direct
  only). The old wall worlds' OTHER arms were in training (`world_seen_in_training: true`);
  the new worlds are entirely fresh (`world_seen_in_training: false`).

## Pipeline

1. `traffic_shepherd/c4/wallregen.py` (driver, SUMO). Per plan world: fresh `CavSumoEnv`
   (`controller="direct"`, `population=` passed through when the constructor supports it -
   refuses a non-legacy population otherwise), native warmup to `t_w`, entry times tracked
   (RSU counter), `saveState` (precision 17 + RNG) + env deep copy
   (`traffic_shepherd.c4.branching` save/restore, every load in a FRESH SUMO process = common
   random numbers). Branches of `branch_s` s: `native`, the 4 wall arms, `native_repeat` on
   every 10th world (determinism check). The wall arm = `WallGroupSession` v2.1 + the pilot
   request rule + dispatcher tick order, commanded through the env's direct path
   (`CommandOwnership`, speedMode 27, ceiling 1.5 m/s^2 = the adopted bound; the law clips to
   [-3, 1.5]), non-members native. Tapes: `base.npz` (120 s before t_w), `arm_<name>.npz`
   (0.5 s slot state s/lane/v/a/mask/vid + first-seen arrays), `world.json` (registry, wall
   timeline/events, member lanes, totals incl. collisions/teleports, wall-clock).
   Resumable (`world.json complete` skipped; partial dirs renamed `.incomplete-<n>`), parallel
   (spawn pool, one fresh process per world, never kills anything).
2. `traffic_shepherd/c4/wallregen_states.py` (converter, no SUMO) -> `wms_eval_states/1`
   files exactly as `score_c4.py` / `wms.eval.score` read them (same record keys as the old
   sets + `population`, `variant`, `sumo_seed`, `t0_origin_s`): variants
   * `h60`  t0 = t_w + 4.5 s, 5 s history (shape-identical to `wallc1_states_h60`);
   * `real` t0 = t_w + 60.5 s, 60 s history (like `wallc1_states_h60_real`);
   * `req`  NEW: t0 = each arm's wall-request tick, 60 s history, wall AND native futures from
     the bit-identical t0 state (`paired` true) - the clean wall-BENEFIT set.
   Each dir: `wall_same.json`, `wall_pilot.json` (walls that assembled), `wall_native.json`
   (native arm at the same (world, t0) for unpaired wall states), `BUILD_RECEIPT.json`,
   `REJECTIONS.jsonl`, `SHA256SUMS.json`.
3. `tools/wallregen_score.py` (scorer copy, CPU): same per-state scorer as score_c4.py
   (`score_states`, teacher-forced CAVs, |ds|/|dv| at 10/30/60/120 s, blow-ups), any number of
   `--ckpt TAG=path` (paired world-bootstrap differences vs the first), cached per checkpoint
   sha + set sha; `--benefit` = recorded vs predicted wall benefit (native - wall HDV delay of
   the t0 roster, by section; CAVs excluded because they are teacher-forced), paired vs
   unpaired reported separately. The old gates (step0 11.9576, C1 8.71/99.84) are bound to the
   old sets and are not applied.

## Seed block (proposed; integrator registers + runs the holdout guard)

`C4_WALLREGEN_EVAL_SEEDS = 962000-962199`, loadings 2400/2600/2700, EVALUATION only
(`traffic_shepherd/seeds.py`, admitted by `traffic_shepherd/c4/seeds.check_case`; asserts
disjoint from the blocked sets and GENRAND 961000-961999). Scan: no integer in 962000-962199 in
any `traffic_shepherd.seeds` table (unit test), `protocols/experiment_history.json` or
`runs/active/**/eval_seeds.json` (word-boundary regex; the only textual hit is GENRAND's
exclusive `range(..., 962000)` end). Guard files with restricted names were NOT opened.
Default plan uses 962000-962035 (36 worlds; penetration cycles 0.1/0.2/0.4, demand cycles over
the chosen levels, t_w drawn from 240..540 by plan seed 20261015); 962036-962199 stay in reserve.
Loadings admitted for the block: any multiple of 50 veh/h in [1000, 2800].
`tests/unit/test_c4_genrand.py` boundary case moved from 962000 to 962200 (962000 is now
admitted by the new block).

## Demand levels (OWNER DECISION PENDING)

The textbook population saturates at ~1930 veh/h mainline (coordinator 2026-10-14), so the old
sets' 2400/2600/2700 give unbounded source queues. Demands are ONE variable,
`DEMANDS_VPH` at the top of `traffic_shepherd/c4/wallregen.py` (default `None` = undecided:
`plan` refuses), or `--demands a,b,c` / `DEMANDS=a,b,c` for `run_wallregen.sh run` (required).
Levels cycle per penetration triple (36 worlds / k levels; k = 2, 3 or 4 keeps cells balanced).

## Size and cost

36 worlds x 4 wall arms -> wall_same <= 144 states per variant (old: 67 over 29 worlds),
wall_pilot ~ the assembled subset, req <= 144 minus not_feasible. Laptop measurements
(throttled, 1 process, legacy3): native warmup 240 s in 23 s; a 182 s branch ~17 s
(~10.7 sim-s per CPU-s); one 600 s branch ~55 s. Per world (t_w ~390 s, 5-6 branches x 600 s)
~5-6 CPU-min. 36 worlds on a 48-core box (46 workers): ONE wave, ~6-10 min wall,
~3.5 CPU-h. Disk: ~8 MB tapes per world (~0.3 GB); converted sets ~4 MB/state (60 s history
JSON), ~0.6 GB per wall_same file for real/req, ~0.15 GB for h60. Conversion ~1-3 min per variant.
Scoring (CPU): ~30 s per state per worker for one checkpoint (C4 wm_s_best, laptop); 144 states
x 8 workers ~ 9 min per set per checkpoint; `--benefit` adds 2 rollouts per wall state.
Textbook population may change congestion and hence SUMO speed; plan for 2x.

Per-set budget of the default plan (36 worlds, any demand levels): SUMO 124,680 sim-s
(warmups 14,280 s + 184 branches x 600 s), ~2-3.5 CPU-h, one wave on 46 workers (~5-10 min
wall); every variant (h60 / real / req) is converted from the SAME tapes: wall_same <= 144
states, wall_pilot = assembled subset, wall_native <= 36 (h60/real only), req <= 144 minus
not_feasible arms.

## Box commands (Linux, C4 bundle layout; code at $C4_HOME/rd incl. this branch + popfix)

```bash
cd $C4_HOME/rd/scripts/c4
bash run_wallregen.sh check                       # imports, population arg present, plan dry run
bash run_wallregen.sh smoke                       # 1 world (962199), 182 s branches, ~3 min, foreground
DEMANDS=<owner levels, e.g. 1500,1700,1900> POPULATION=textbook bash run_wallregen.sh run
                                                  # detached; WORKERS defaults to cores-2
cat $C4_HOME/runs/wallregen_*/STATE; tail -f $C4_HOME/runs/wallregen_*/launch.log
RESUME=1 RUN_ID=<id> bash run_wallregen.sh run    # after an interruption (complete worlds skipped)
```

Second box (split the block, disjoint seeds, merge the converted sets by concatenating world
dirs into one `sim/` before converting, or just run both halves and convert each):
`SEEDS=962000-962017 ...` on box 1, `SEEDS=962018-962035 ...` on box 2.

Without the c4 helpers (plain checkout, PYTHONPATH=src:.):

```bash
python -m traffic_shepherd.c4.wallregen plan --out W/plan.json --population textbook --demands <levels>
python -m traffic_shepherd.c4.wallregen run --plan W/plan.json --out W/sim --workers 46
for v in h60 real req; do python -m traffic_shepherd.c4.wallregen_states --run W/sim --variant $v --out W/states_$v; done
python -m traffic_shepherd.c4.wallregen status --out W/sim
```

Fetch `out/states_{h60,real,req}` to the laptop as `data_cache/wallc1t_states_h60`,
`data_cache/wallc1t_states_h60_real`, `data_cache/wallc1t_states_req` (keep the raw `sim/` tapes
with the run evidence). Score (laptop or box, CPU):

```bash
python tools/wallregen_score.py --set-dir data_cache/wallc1t_states_h60 \
  --ckpt V5=<v5 wm_s_best.pt> --ckpt C4=runs/active/WM-S-C4-20261013/fetched/gpuW/c4_wm_main/wm_s_best.pt \
  --ckpt C1=data_cache/ckpt_c1fast/wm_s.pt --out runs/active/<NEW_SCORE_DIR>/h60 --workers 8
python tools/wallregen_score.py --set-dir data_cache/wallc1t_states_req --sets wall_same \
  --ckpt V5=<...> --ckpt C4=<...> --out runs/active/<NEW_SCORE_DIR>/req --workers 8 --benefit
```

(physics.json must sit next to each checkpoint, as for score_c4.py; `--entries` for the
entries-ON variant; `--limit N` for a quick subset.)

## Verification done (laptop, legacy3 population until popfix lands)

* `tests/unit/test_wallregen.py` 9 passed (seed block disjointness + guard, law sha pin,
  pilot constants, pair rule, arm runner request/assembly/hold/release/not_feasible, plan
  determinism, converter history/counter/matching); `test_c4_genrand.py` 19 passed;
  ruff clean on the new files.
* Smoke 1: w962000m2700p40t240, 182 s branches, 6 arms, 132 s wall: outcomes
  aborted_in_hold x2 (member_left_network), in_hold_at_end, assembled_released; native vs
  native_repeat max |ds| = 0.0, identical vehicle ids (CRN determinism); 0 collisions/teleports.
  Converted h60/real/req: 4 wall states each; record keys identical to the old sets (+
  population/variant metadata); h60 history [11,192,7] 5.0 s limited; real/req [121,192,7];
  late arms at t_w+4.5 matched native (paired), early arms not; all req states matched.
* Scored with `runs/active/WM-S-C4-20261013/fetched/gpuW/c4_wm_main/wm_s_best.pt`: no errors;
  h60 HDV median |ds| 1.27/4.47/4.80/2.70 m (10/30/60/120 s, 4 states); req 0.93/3.03/4.54/10.06;
  benefit pass runs (paired HDV-delay benefit rec vs pred at 60 s -111.1 vs -112.5 veh-s).
  These are smoke numbers on 1 legacy world, not evidence.
* Smoke 2: w962197m2400p10t300 + w962198m2400p20t300, 600 s branches (default), 2 workers in
  parallel: 211 / 190 s wall per world (16.5 sim-s per CPU-s at demand 2400), outcomes
  assembled_released 5 / aborted_in_hold 3, 0 failures; tapes 6-8 MB per world; conversion
  5 / 6 / 11 s (h60 / real / req, 8 states each; 40 / 51 / 68 MB); C4 scored the real set
  (4 states) without errors (60 s 5.34 m) and the unpaired benefit pass ran.

* Textbook (after merging c4/int b538acd = popfix): w962199m1900p20t300, 182 s branches,
  88 s wall; population_manifest textbook_v1 / cav_typical; native vs native_repeat max |ds|
  0.0; outcomes aborted_in_hold x2, in_hold_at_end x2; 0 collisions/teleports; h60/real/req
  4 states each; C4 req-set scoring: no errors (60 s HDV |ds| 45.2 m, 1/4 blow-ups; smoke only).

## BLOCKER for demand levels below 1900

`traffic_shepherd/scenarios.py::request_schedule` refuses loadings outside [1900, 2800]
("loading outside declared development family"); 1800 failed in the smoke. Levels below the
textbook saturation (~1930) other than 1900 need that family guard widened (shared code, owner/
integrator decision; not changed here). The WALLREGEN seed guard already admits 1000-2800.

## Pending (integrator)

* Register the seed block (ledger/registration) and run the holdout guard on 962000-962199.
* Owner: choose `DEMANDS` (textbook saturation ~1930 veh/h), then launch the run.
* Bundle: `scripts/c4/bundle_manifest.json` now ships
  `output/controller_verification_v7/pi_owner_adapter.py` (CavSumoEnv.reset imports it through
  `controller_factory` -> `pi_session`); rebuild the bundle before the box run.
* Ledger entry for the box run (new simulation) per AGENTS.md close-out.
