# C4 deployment runner (`src/wms/control/deploy.py`, branch c4/deploy, 2026-10-12)

One SUMO episode (`traffic_shepherd.c4.env.CavSumoEnv`) with a frozen baseline, the learned policy, or the
learned policy under the RSU world-model check (design `docs/WM_C4_DESIGN.md` 2.6, Option B).

## CLI

```powershell
$env:PYTHONPATH='src'
python -m wms.control.deploy --controller policy_check --state oracle|belief `
  --policy <c4_dreamer/1 policy ckpt> --wm-ckpt <wm_s_best.pt> [<more members>] `
  --seed 4107 --demand 2700 --penetration 0.2 --warmup 300 --horizon 600 `
  [--decision-period 5 --imagine-horizon 20 --n-driver-samples 4 --margin 0 --confidence-z 1.96] `
  [--belief-lead 60 --belief-propagator wms|idm --no-compact --no-classic-imagination] `
  [--device cpu|cuda --threads N --k 192 --no-physics-log --no-ttc] --out <new dir>
```

- `--controller`: `native|damping|pi|wall` (frozen env baselines, same as `collect --policy <name>`),
  `policy` (`wms.control.policy.load_policy_fn(ckpt)` every 1 s, the `collect --policy <ckpt>` path),
  `policy_check`. Campaign aliases accepted: `c4_policy`, `c4_policy_rsu_check`, `--loading`, `--output`,
  `--policy-ckpt`, `--rsu-state`, `--role exposed`, `--belief-ckpt` (ignored).
- Seeds: the env gate (`traffic_shepherd.c4.seeds.check_case`) admits only 4107-4148.

## policy_check

Every `--decision-period` s (first at t = warmup): build the imagination state, compact it to present
slots (no entries are imagined, so free padding is inert), call `RsuChecker.decide` (one batched WM call
per ensemble member, paired driver samples, keep `policy` unless paired mean > margin and LCB > 0), then
execute the choice in SUMO until the next decision:

**Owner 2026-10-12 late: direct-output candidates only** (no designed controller is a check candidate or
executed in `policy_check`; native / damping / pi remain whole-episode baselines via `--controller`).
Every candidate is executed through the policy path: direct accelerations + lane commands
(`CavSumoEnv.step(accel, lanes=...)`, see POLICY2_NOTES.md); one dwell clock and one command memory are
shared by all executors of the episode.

| candidate | imagination (WM-S) | SUMO execution |
|---|---|---|
| `policy` | `bridge.make_cav_controller` (deterministic actor + lane head, RSU encoder with `build_rsu_field(state)`, `safe_accel`) | `load_policy_fn` executor: accel + lanes |
| `policy_group_0` | same, zero group signal | same actor, zero group signal |
| `policy_group_1` | same, perturbed group signal (`--group-perturbation`, default 0.1) | same actor, perturbed signal |
| `policy_sample_0/1` | stochastic actor, `rsu_check.NoiseStream` (same draws in every WM member / driver sample) | same stream replayed per tick after the decision |
| `hold_speed` | direct a = 0, stay in lane (forced ramp merge kept), safety-filtered | same (direct, lanes -1) |

The oracle state also carries the true per-CAV dwell clocks (`lane_dwell_s`) and the last commanded
accel / lane intent (`cmd_accel_mps2`, `lane_intent`) so imagination starts from the executed history.
The smoke table below predates this change (it used native/damping/pi candidates).

State sources: `oracle` = env true WM-frame state + its rolling history (short 10-tick `history` and the
60 s long `driver_history`, trainer featurisers) - privileged upper bound. `belief` =
`wms.control.belief.BeliefTracker` updated every control step (and over the last `--belief-lead` s of the
warmup via an env hook) with CAV self reports + their own front/rear SUMO-sensor detections
(`TableSensor`, 100 m) and the RSU-zone detections; propagator = one `WmSModel.step` per 0.5 s (CAVs hold
speed, drivers at the encoder's population default because track slots are re-packed every tick). Its
long driver history is featurised from `tracker.history()` with `long_history_features`.

## Outputs (`--out`, must be new/empty)

- `episode/` - the collector's episode format (`manifest.json`, `summary.json` = `c4_sumo_episode_summary/1`,
  `episode.npz`, `run/`).
- `decisions.jsonl` - per decision: `time_s`, `candidates`, `imagined_returns` (mean/std/paired_mean/
  paired_std/paired_lcb/eligible + raw `[ensemble,sample]` returns and paired diffs), `choice`, `executed`,
  `state_source`, `source` (present/CAV counts; belief: true counts, coverage overall/by lane/RSU mainline,
  tracks, mean HDV age, tracker stats), `slots_imagined`, `timing` (checker total/rollout ms),
  `wall_clock_s`, `state_build_s`.
- `segments.jsonl` - per decision segment: realised SUMO interval totals and the SUMO CAV action labels.
- `DEPLOY_SUMMARY.json` - config, episode summary, choice counts, executed control ticks, decision timing
  stats, belief update timing, collisions/teleports, notes (tracebacks on failure; exit code 1).

## Smoke (laptop CPU, 4 torch threads, R5b `wm_s_best.pt` single member, untrained `policy_latest.pt`)

seed 4107, demand 2700, penetration 0.2, warmup 300 s, horizon 60 s, decision period 10 s, imagination 10 s,
2 driver samples, 5 candidates (batch 10). Outputs (local, not committed): `runs/active/C4-DEPLOY-SMOKE-20261012/`.

| source | decisions | choices | wall s / decision | rollout ms | slots | coverage | collisions | episode wall |
|---|---|---|---|---|---|---|---|---|
| oracle | 6 | pi 3, native 3 | 2.4-5.8 (mean 3.9) | 2297-5452 | 108-118 | 1.0 | 0 | 31 s |
| belief | 6 | pi 4, native 1, policy_group_0 1 | 2.3-3.5 (mean 2.7) | 2084-3175 | 71-87 tracks (true 108-119) | 0.59-0.71 (RSU mainline 1.0) | 0 | 33 s |

Executed controllers verified from SUMO action labels per segment: pi segments `pi-external-u` (rest native
fallback without leader), native segments all `native`, `policy_group_0` segment all `direct`. Belief update:
mean 0.074 s per control tick (WM-S propagator). Default settings (4 samples, 20 s imagination, 5 s period):
oracle 11.8-13.4 s, belief 9.4-15.2 s per decision (4 threads); 1 thread ~17.8-24.8 s. A 600 s episode =
120 decisions, i.e. ~25-45 min of checker compute on CPU - slower than real time at the 5 s period.

Tests: `tests/unit/test_c4_deploy.py` (11; SUMO parts skip without `sumo`, R5b-dependent ones without the
checkpoint). `tests/unit/test_c4_*.py` + `test_import_graph.py`: 307 passed. `test_cav_drive_bridge.py`,
config tests touching `cav_drive`: pass. ruff clean.

## Changes outside the new module

- `traffic_shepherd/c4/env.py`: `step(..., fallback="native"|"damping"|"pi")` for direct episodes (per-tick
  controller of CAVs without a finite command) and `reset(..., warmup_hook=fn)`. Defaults unchanged.
- `src/wms/train/cav_drive.py`: bug fix - `TimelineControllers._invalidate` passed only the dropped vehicle
  as PI's live set, which erased every other vehicle's PI state ("vehicle must be reset" on the next
  proposal); it now passes all active PI vehicles (regression test included).
- `scripts/c4/eval_campaign.py` / `run_eval.sh` / `runs/active/WM-S-C4-20261013/OPS_NOTES.md` sections 7 and 10:
  C4 module default `wms.control.deploy`, belief jobs no longer pending, `--classic-harness evaluate|deploy`,
  `--deploy-args`; open seed and sizing blockers recorded.

## Limitations

- The policy is untrained: it was never preferred by the check except `policy_group_0` once by +0.002
  (margin 0 lets near-identical alternatives win on negligible paired differences; set `--margin` > 0
  in summed team-return units for real evaluations). Smoke numbers are mechanics only, not efficacy.
- Oracle is privileged. The belief misses ~30 % of vehicles outside RSU/CAV coverage, uses population-default
  drivers in its propagator, and assumes 5 m HDV length; imagination does not spawn unseen/future arrivals.
- Imagined `native` = hold speed, while SUMO native is the vType car-following model; imagined damping/PI use WM-predicted
  leader gaps. PI state is not carried across a switch (fresh session on each selection).
- Latency: synchronous checker; at default settings each decision exceeds the 5 s period on CPU.
- The campaign's `eval_seeds.json` (WM28 fresh-validation seeds) is refused by the C4 seed gate (OPS_NOTES 10).
