# LANEFIX (2026-10-14): lane-change fixes of the WM-S hazard lane model

Branch `c4/lanefix` (from `c4/int` a2d3716). Code, tests and a CPU smoke only: **no GPU run;
the owner decides the next world-model run.** Every switch defaults OFF and the OFF path is
bit-identical to a2d3716 (golden numbers captured on that tree, see Tests).

Source diagnosis: `E:/agent-work/tmp/lanediag/` (an_*.txt, rdiag.py, tdiag.py). In short:
rollout lane decisions were argmax, so discretionary changes (teacher-forced p << 0.5) never
happened in rollouts (wall_same 60 s: truth 8.27 changes/state, v3 2.21 = merges only); the
hazard was over-confident (p/obs 4-10x) and mis-ranked (feasible-row log-loss worse than a
constant; merge AUC < 0.5), most likely because the rollout lane CE (weight 1) and the
flip-rate hinge back-propagated into it and made it chase the recorded lane; merges came too
late (physics urgency ramp) and 28 % of wall jam merges were masked when they happened.

## Switches

| key (flat key) | values (default first) | what |
|---|---|---|
| `model.lane_decision` | `argmax`, `sample` | hazard mode: lane decision drawn from the masked categorical |
| `model.lane_logit_offsets` | `[0,0,0]` | hazard mode: logit offsets (overtake 1->0, return 0->1, merge 2->1) |
| `model.lane_hazard_features` | `v1`, `v2` | hazard input: v2 adds 14 ranking features (44 inputs) |
| `model.merge_hazard_zone` | `forced_late`, `full` | who owns the ramp merge 2->1 |
| `model.gap_stopped_relax` | `false`, `true` | `lane_gap_rule="speed"`: relaxed gaps for stopped ramp mergers |
| `loss.lane_hazard_decouple` | `false`, `true` | rollout lane CE / flip-rate do not train the hazard |

Plumbing as for `lane_model` / `lane_gap_rule`: `ModelConfig`/`LossConfig` (validated: the
hazard switches need `lane_model="hazard"`, the relax needs the speed rule), `legacy_flat`
(+ `OMIT_WHEN_DEFAULT`, so every existing flat config/receipt is unchanged), `TrainSettings`
(validated), `ModelSpec` (the four model switches; stored in the checkpoint config, the v2
features change the hazard input width), `PhysicsConfig.gap_stopped_relax` via
`build_physics` (both branches) and therefore `physics.json`; `from_params_file` never reads
it from a params file. `loss.lane_hazard_decouple` is applied by the trainer
(`_apply_model_overrides` sets the runtime attribute `model.lane_hazard_detach`; never part
of a checkpoint).

## Design choices

1. **Sampling** (`wms.model.lane_hazard.sample_lane_choice`, `WmSModel.step/rollout`).
   Learned (HDV) slots draw from `softmax(logits)` restricted to the physics-feasible lanes
   (`lane_feasible = lane_base > MASKED_LOGIT`, the current lane always; infeasible = -inf).
   The feasibility comes from the physics prior, not the final logit, so a feasible change
   whose hazard logit drifts below -8 stays possible. Draw = inverse CDF of ONE uniform per
   (step, slot) over lanes 0,1,2 - rows that share the uniform choose monotonically coupled
   lanes (good common random numbers). CAV / absent slots keep the argmax (CAVs are
   overridden by their teacher anyway). Randomness sources, in priority order (rollout
   kwargs): `lane_uniform [B or 1, T, K]` (caller draws, expanded to counterfactual rows like
   `accel_noise`), `lane_seed` (private CPU generator, global RNG untouched), else the global
   torch RNG (training: seeded by `runtime.seed`); `lane_crn=True` shares one draw per
   (step, slot) across all rows. Draws are made outside the per-step activation checkpoint, so
   recomputation never re-draws.
   * Training validation (`WmSTrainer.evaluate`): for sampling models every evaluation runs in
     a forked RNG seeded with `runtime.seed + 7919` (reproducible selection, training RNG
     untouched); argmax models take the unchanged path.
   * Imagination (`wms.control.dreamer.imagine`): `wms.control.futures.lane_uniform_draws`
     draws `[n_starts, T, K]` once per start from a generator seeded by the update count /
     `future_seed`, shared by every ensemble member and expanded to the counterfactual rows
     of the start (CRN). Returns `None` when no WM samples (dreamer unchanged).
   * Scoring (`wms.eval.predictor.RolloutFlags.lane_seed`, default 0): sampling models are
     scored with fixed draws; argmax models unchanged. An evaluator can compare argmax vs
     sample on the same checkpoint by setting `model.lane_decision`.
   * Candidate batches (`coop_imagine.py`: its seeded generator; `rsu_check.py`: a fixed
     seed 0): `futures.arm_major_lane_uniform` gives every arm / candidate the same draws per
     sample index (arm-major rows, CRN); nothing is drawn for argmax WMs.
   * Offsets are added (hazard mode, feasible non-forced targets of learned slots only) in
     `hazard_lane_logits`, i.e. in the rollout AND in the teacher-forced event logits, so the
     calibration tool measures the offset model. They are a post-training calibration knob:
     set them at evaluation (`model.lane_logit_offsets`); trained with nonzero offsets the event
     CE would simply absorb them.
2. **Decouple** (`loss.lane_hazard_decouple`). Chosen mechanism: the hazard's learned delta
   enters the ROLLOUT lane logits detached (`hazard_lane_logits(detach_delta=True)`; forward
   values identical). The rollout logits feed only `loss.lane`, the flip-rate / lane-rate
   guards and diagnostics (the aggregate loss does not read them), so those terms no longer
   reach the hazard parameters or, through the hazard features, the state / driver embedding.
   `loss.lane_event` evaluates the hazard separately (`lane_event_logits`, never detached) and
   is its only objective. Why not zero `lane_weight`: `loss.lane` still trains the physics-prior
   path (the encoder's merge parameters through the forced-merge logit) and stays a logged /
   comparable metric; with `merge_hazard_zone="full"` that remaining path is small (constant
   base outside the last 5 m). Test: hazard grad exactly 0 with lane_event weight 0, > 0 with
   it, > 0 without decouple; the other parameters still get gradients.
3. **v2 ranking features** (`hazard_features_v2_extra`, 14 extras appended to the 30 v1
   inputs). Per lane (left target, right target, current; zero for a non-existent lane):
   (mean speed of the cars within 100 m ahead, capped at v0, - v0)/10 (0 on a free lane), the
   nearest leader's acceleration/3 (speed trend: the state's acceleration channel, no history
   needed), cars within 100 m and 200 m ahead x 7.5 m / range (~1 at jam density); plus the
   time blocked behind a slower leader (s, capped 60 s, /10) and a blocked-now flag. 'Blocked'
   = same-lane vehicle leader within 50 m slower than the car's v0 - 2 m/s; the count of
   consecutive blocked steps is carried by the rollout (`state["lane_blocked"]`, across BPTT
   chunks), reset when not blocked or absent, and initialised at t0 from the observed pre-t0
   long driver history (trailing run of blocked samples before t0 x stride, leader speed from
   the history's leader-dv channel). The teacher-forced event loss computes the same counts on
   the recorded states (`recorded_lane_blocked`, from the same pre-t0 initial count). Only the
   current state and the car's own past are used (test: perturbing steps >= 7 leaves the
   teacher-forced logits of steps < 7 unchanged). Thresholds are round defaults, not tuned.
4. **Merges** (`merge_hazard_zone="full"`). Inside the merge zone the merge 2->1 logit is
   `MERGE_HAZARD_PRIOR_LOGIT (-4, ~ the TRAIN per-step merge rate 1.6 % of lane-2 zone rows)
   + delta (+ offset)` instead of the physics urgency ramp + delta, so the hazard owns the
   timing from the start of the acceleration lane; in the last 5 m a feasible merge gets logit
   `max(physics, 8)` (forced; p(stay) ~ 3e-4 per step, also under sampling). The physics mask
   still decides feasibility everywhere (never unmasked). The v1 'physics prior logit' input
   feature keeps the physics value (informative), only the base changes.
   **Stopped relax** (`gap_stopped_relax`, `wms.physics.mobil.stopped_merge_gap_safe`): a
   ramp merger (lane 2 -> 1) slower than 1 m/s is judged on the CURRENT configuration (no
   anticipation), leader gap >= 0.5 m, follower gap >= 0.5 m + max(0, v_f^2 - v^2)/(2*6)
   (the follower can still stop), and no overlap with margin 0.5 m; OR-ed with the speed rule
   (only adds admissions, only for merges). Example (test): ego stopped, 4 m to a stopped
   leader and 4 m to a follower creeping at 3 m/s: speed rule masks it (anticipated follower
   gap 2.5 m < 3.05 m), relaxed rule admits it; a car alongside or a 12 m/s follower still
   blocks.

## Config

`configs/c4_wm_v4_lane.toml` = `c4_wm_v3.toml` + lane_decision sample, features v2, merge zone
full, stopped relax, decouple, `loss.lane_event.weight = 3`, own `runtime.out_dir`
(`runs/active/c4_wm_v4_lane`). Test: the flattened configs differ only in exactly these keys.
It extends v3 (not the JUNCTIONFIX v4); combine with `data.junction_fix = "ffill"` if wanted.

## Calibration tool

`tools/lanehazard_calib.py CHECKPOINT [--config configs/c4_wm_v3.toml] [--batches N]
[--data-root E:/research-dense] [--out json]`: teacher-forced p/obs, masked-at-event share,
feasible AUC and feasible-row log-loss vs the best constant per category (overtake / return /
merge) and pooled, plus a reliability table, on TRAIN windows only (adapted from
lanediag/tdiag.py + an_t.py; v2-feature models supported). Refuses restricted path tokens.

Demonstration run (the tool works; NOT an evaluation of any fix - no LANEFIX model is trained):
V3s60 (`fetched/gpuB/c4_wm_v3/wm_s_stage60.pt`, its physics.json alongside), 16 TRAIN batches
x 4 windows, first 120 steps each, seed 7 (`docs/lanefix/calib_V3s60_train.json`, 866k rows,
1179 events, 8 min on the shared CPU):

| category | events | p/obs | masked@event | AUC feasible | log-loss feasible | constant |
|---|---|---|---|---|---|---|
| overtake 1->0 | 645 | 4.02 | 0.08 | 0.80 | 0.053 | 0.043 |
| return 0->1 | 319 | 6.71 | 0.04 | 0.78 | 0.029 | 0.019 |
| merge 2->1 | 214 | 5.96 | 0.04 | 0.41 | 1.804 | 0.307 |
| pooled feasible | 1103 | 5.33 | - | - | 0.059 | 0.035 |

This reproduces the diagnosis (over-prediction 4-7x, merge AUC < 0.5, log-loss worse than a
constant). 'Masked' / 'feasible' here = hazard logit <= / > -8 (as in lanediag/tdiag.py);
the sampler's feasibility is the physics prior's, which differs only for feasible changes
whose hazard logit fell below -8. Target after the next run: p/obs ~ 1 and log-loss below the
constant in every category.

## Tests / smoke


`OMP_NUM_THREADS=1`: `tests/unit/test_lanefix.py` (21 at that point) + `test_wmlane.py
test_gaprule.py test_wmloss.py test_lane_rule.py test_c4_core.py test_c4_dreamer.py
test_c4_futures.py test_import_graph.py` -> **342 passed**; after the candidate-batch wiring
`test_lanefix.py` (22) + `test_c4_deploy.py test_c4_futures.py test_c4_rsu.py
test_import_graph.py` -> **262 passed**; ruff clean on every touched file.
Full `tests/unit` (with 20 of the 21 LANEFIX tests): 1010 passed, 1 skipped, 2 failed - both
unrelated, see the end of this file.

`test_lanefix.py` covers: switch-off bit-identity vs golden numbers captured on a2d3716
with this file's helpers (one hazard + EMA + event-loss + speed-gap-rule training step: loss
terms exact, total and hazard |grad| to 1e-9; an eval rollout with a biased hazard: positions,
lanes, logits, speeds exact; physics masks fixed/speed on a slowed batch state and the
GAPRULE fuzz scene); sampled decisions deterministic per seed, different across seeds and vs
argmax, global RNG untouched; every sampled lane feasible over 200 draws with an extreme
hazard, float-rounding edge falls back to argmax; CRN (`lane_crn`, `[1,T,K]` / `[B,T,K]`
caller draws) gives identical rows, independent draws differ; offsets shift exactly their
category; decouple: hazard |grad| exactly 0 from lane CE + flip-rate, > 0 without decouple,
> 0 from lane_event, other parameters still trained, forward values unchanged; the trainer
applies the switch; v2 features finite, widen the hazard to 44 inputs, carried through the
rollout state, exact lookahead / density / leader-accel / blocked values on a hand scene,
pre-t0 blocked run from the long history; no future leakage in the teacher-forced features
(perturbing steps >= 7 leaves steps < 7 bit-identical); full merge zone: early merge when the
hazard says so, no early merge when it says no, forced in the last 5 m, legacy window keeps
the physics logit, never unmasks an overlapping merge; stopped relax: 4 m / 0 m/s jam merge
with a creeping follower admitted (masked without), side-by-side car or 12 m/s follower still
blocks, moving ego unchanged, only adds admissions and only for lane-2 cars, needs the speed
rule; plumbing (flat keys omitted while default, flat round trip, settings / spec, validation
errors), v4-lane vs v3 flat diff = exactly the LANEFIX keys + lane_event weight + out_dir,
`build_physics`, checkpoint + physics.json round trip (switches, 44-input hazard, weights),
the imagination draw helper and the calibration tool's row/summary/AUC functions.

CPU smoke (`tools/c4_smoke.py E:/agent-work/tmp/lanefix_smoke_v4lane configs/c4_wm_v4_lane.toml`,
data via `wmlane_smoke.point_data(E:/research-dense)`): **passed**, 2 optimizer steps, 10 s
unroll, batch 4, both finite, loss 14.51 -> 15.64 (2 different batches, untrained), lane_event
0.024, predicted change rate 0.0023 / step, 2041 TRAIN / 261 val windows, physics
lane_gap_rule = speed (the receipt's switches come from the v4-lane config). Step time on the
shared, heavily loaded dev box (3.2 s per simulated step) is not a timing reference; a CPU
micro-benchmark (B 4, K 200, 20-step rollout + backward) showed no visible v2 overhead
against v3 within the noise. Memory note: the teacher-forced event pass evaluates the
features on all B x T recorded states at once; v2 adds per-lane pairwise [B*T, K, K] passes
(shared distance tensor), so peak memory of that pass grows - if the GPU run OOMs there,
chunk `recorded_event_logits` over time.

## Not done / for the integrator

* No training beyond the 2-step smoke. After the GPU run: `tools/lanehazard_calib.py` on the
  new checkpoint (TRAIN), then the wall diagnostics with `RolloutFlags.lane_seed` fixed;
  compare argmax vs sample (`model.lane_decision`) on the same checkpoint.
* Entrants: a slot inserted by the entry process starts with blocked count 0 (the count resets
  while the slot is absent); unlike the blocked count, the existing `lane_age` keeps the slot's
  old value (pre-existing behaviour, unchanged).
* Thresholds of the v2 features (100/200 m, 50 m, 2 m/s, 60 s cap), the merge base logit (-4)
  and the forced window (5 m) are round / TRAIN-rate defaults, not tuned on any set.
* Full `tests/unit` has two failures unrelated to LANEFIX: `test_drive_registry::
  test_committed_registry_matches_the_generator_output` (fails identically on a2d3716: local
  data_ops tree) and `test_c4_mappo::test_worker_pool_spawn_survives_worker_death` (flaky under
  the loaded box; passes in isolation on this branch).
