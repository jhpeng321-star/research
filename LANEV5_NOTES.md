# LANEV5 (2026-10-14): per-transition lane-change network for WM-S

Branch `c4/lanev5` (from `c4/int` 1b249be). Owner-approved design (2026-10-14 ~20:40). Code,
tests and a CPU smoke only: **no GPU run** (the integrator launches it after the owner approves
the full v5 recipe). Switch: `model.lane_model = "hazard_v5"`; every default is unchanged and
the OFF path is bit-identical (the golden tests of test_wmlane / test_lanefix pass unchanged;
no module, key, RNG draw or flat key is added while off).

Evidence: `E:/agent-work/tmp/laneranking/` (boosted trees AUC .99/.98/.99, AP .44/.16/.92 vs
v4 hazard AUC .66/.79/.49 with the same inputs; causes: overtake and merge shared one 'left'
output, raw inputs, ~0.3-0.5 % of the gradient), `c4-lanefix/LANEFIX_NOTES.md`.

## Design (owner rules)

* **No driver types anywhere.** Categories are road manoeuvres fixed by the car's lane:
  lane 1 -> 0 *overtake*, lane 0 -> 1 *return*, ramp 2 -> 1 *merge* (1 -> 2 is never allowed
  by the physics). ONE shared network for all cars: trunk (2 x `lane_hazard_hidden`=64, GELU)
  over situation features + road context, three heads = one logit (vs stay 0) per transition;
  the car's lane picks the head. **No driver embedding and no encoder settings** enter it
  (test: logits identical under different driver params / embeddings). The 'v0' used inside
  the features is the car's own maximum observed speed (own-history statistic, carried); so the
  network is a pure function of the state and the car's own past and can be pre-fitted offline.
  Driver individuality elsewhere (encoder, residual) is unchanged.
* Heads: zero-initialised weights, biases at the TRAIN feasible-row base rates
  (logit -5.7 / -6.6 / -2.3), soft cap `lane_hazard_cap` (10).
* **Inputs** (`wms.model.lane_v5.FEATURE_NAMES`, 48): v, a, v0h, deficit, v/v0h, lane age,
  never-changed; current leader gap/dv/accel (lane end = obstacle for lane 2); target-lane
  leader gap/dv/accel, follower gap/dv, speed advantage; **speed-gap-rule margins** of the target
  leader/follower gaps and the overlap flag (same rule + anticipation as the mask), IDM
  incentive, deceleration imposed on the new follower, safe-speed gain; mean speed (<= 100 m)
  and densities (100/200 m) ahead in the target and current lanes; blocked time / flag / 30 s
  EMA, deficit EMAs (10/30 s), **LC2013-style speed-gain accumulator** (for the car's own
  direction) and **keep-right accumulator + EMA** (lane 0), **5 s / 15 s averaged safe-speed
  gains**; **neighbours' recent lane changes** (cars within +-100 m that changed in the last 5 s,
  target and current lane); road context: lane one-hot, ramp-merge flag, distance to the merge
  end, in-zone flag, time in the merge zone, position. Downstream cells / traits / embedding
  are NOT used (evidence: no help).
* **Scaling**: signed log `sign(x) log(1+|x|)` then `(x - mu) / sd` with buffers
  `feature_mu/feature_sd` (identity by default; the pre-fit stores TRAIN statistics; they are
  in every WM checkpoint).
* **Accumulators** (16 channels) are a recurrent state `state["lane_v5_acc"]` carried by the
  rollout across steps and truncated-BPTT chunks (like the v4 blocked counter), initialised at
  t0 from the observed pre-t0 long driver history (max speed, blocked run, blocked/deficit EMAs,
  time in the zone; speed-gain/keep-right start at 0, safe-speed EMAs at the t0 value); a slot
  that was absent restarts ('new'), the direction accumulators reset on a lane change. The
  teacher-forced pass recomputes them causally on the recorded states from the same initial
  value (test: rollout step updates == sequence recomputation; chunked rollout == one rollout).
  All feature computation is no-grad (inputs, not a path for gradients into the state).
* **Feasibility**: the physics mask is never lifted. In the rollout the target entry is the
  head logit only on learned (HDV) slots whose target the physics allows; CAVs keep physics.
  For the teacher-forced rows the own-transition feasibility is read from the feature pass
  (speed gap rule + stopped relax + ramp-zone rule; `fast_mask_applies`), identical to
  `mobil_merge_logits > -8` (tested on fuzz scenes for 3 lane rules x relax on/off, and the row
  sets of both paths are equal); other physics configs fall back to the physics module.
* **Merges**: inside the merge zone the merge head is a learned gap acceptance on feasible rows;
  in the last 5 m (`FORCE_WINDOW_M`) a feasible merge gets `max(physics, 8)` (forced); those
  forced-window rows are excluded from the loss (the head does not act there).

## Training (`loss.lane_event`)

`LaneEventLoss` dispatches to the v5 path: rows of **every recorded step of each batch window**
(not only the unroll; test: 12 steps scored for a 6-step unroll), plain **binary CE per head**
on feasible rows (the car's own transition vs stay), mean per head, **summed over heads**
(`gamma` / `max_weight` unused). Inputs come from `prepared["lane_event"]` = pre-t0 lane age +
accumulators (no driver inputs). Decoupled as v4: with `loss.lane_hazard_decouple` the rollout
lane logits are computed under no-grad, so `loss.lane` / flip-rate never reach the network
(test: zero gradient with event weight 0, > 0 with it). The network's parameters are disjoint
from every other term.

Weight: `loss.lane_event.weight = 10` in `configs/c4_wm_v5_lane.toml`. Measured on one smoke
batch (CPU, fresh model, 20-step unroll): lane-network grad norm 1.06 vs 2.37 for all other
parameters (17 % of the squared total norm); weight 3 on another batch gave 0.19 vs 9.23
(0.04 %). Because the parameters are disjoint and the optimiser is Adam (weight decay 0), the
weight mainly sets the term's visibility / clip-budget share, not the lane network's step size.

## Decisions

`model.lane_decision`: `argmax` (change iff logit > 0), `sample` (seeded masked categorical, as
v4), `gated` (new): a head samples only if its teacher-forced feasible-row AP on TRAIN-val
>= its gate (`model.lane_v5_gates`, default overtake 0.25 / return 0.10 / merge 0.5), else
argmax. AP source: `model.lane_v5_head_ap` where >= 0, else the network buffer `head_ap`
(written by the pre-fit tool or `lanehazard_calib.py --write-ap`; unknown = -1 = never samples).
`model.lane_logit_offsets` = per-head offsets (overtake, return, merge), applied in the rollout
and in the event logits. Seeded draws / CRN are inherited unchanged: everything keys on
`_samples_lanes` (true for v5 `sample`/`gated`): `futures.lane_uniform_draws`,
`arm_major_lane_uniform` (coop_imagine, rsu_check), the forked validation RNG in
`WmSTrainer.evaluate`, `RolloutFlags.lane_seed`. Under `gated`, closed-head slots take the argmax
(test: same seed -> same lanes, different seeds differ, all heads closed == argmax).

## Plumbing

`ModelConfig` / `TrainSettings` / `ModelSpec`: `lane_model` += `hazard_v5`, `lane_decision` +=
`gated` (v5 only), new `lane_v5_gates`, `lane_v5_head_ap` (spec + checkpoint config),
`lane_v5_pretrain` (settings only). Validation: `gated` / non-default `lane_v5_*` need
`hazard_v5`; v5 needs a structured hybrid/driver_params model and `lane_head='off'` like the
hazard; `loss.lane_hazard_decouple` accepts `hazard_v5`. `legacy_flat`: three new keys, all in
`OMIT_WHEN_DEFAULT`. No physics change (physics.json unchanged). The v4-hazard switches
`lane_hazard_features` / `merge_hazard_zone` are ignored by v5 (the v5 config resets them).
`model.lane_v5_pretrain` is loaded by `WmSTrainer._apply_model_overrides` (strict; the report is
`model.lane_v5_pretrain_report`); `model.init_from`, applied afterwards, overrides matching keys.

## Tools

* `tools/lanev5_pretrain.py OUT.pt [--config configs/c4_wm_v5_lane.toml] [--data-root R]
  [--train-batches 40] [--val-batches 10] [--batch-size 4] [--epochs 25] [--stay-keep 0.25]`:
  fits the same network teacher-forced on the TRAIN windows of the C4 data mix (rows from
  `lane_event.lane_v5_batch_rows`, i.e. exactly the loss rows), sets the scaling from TRAIN,
  evaluates on the config's TRAIN-scoped validation windows (held-apart TRAIN worlds; never
  wall / held-out / CAL sets) and stores the per-head TRAIN-val AP in `head_ap`. Writes OUT.pt
  (`lanev5_pretrain/1`) + OUT.json.
* `tools/lanehazard_calib.py CKPT --config configs/c4_wm_v5_lane.toml` on a v5 checkpoint:
  per head p/obs, AUC, AP (vs constant), log-loss vs constant + reliability on TRAIN-val
  (`--split train` for TRAIN), offsets applied; `--write-ap OUT.pt` writes a checkpoint copy
  storing the APs (and copies physics.json). v4 path unchanged.

## Results (CPU, this laptop, power-throttled)

**CPU smoke** (`tools/c4_smoke.py` with `configs/c4_wm_v5_lane.toml`, data via
`wmlane_smoke.point_data(E:/research-dense)` + the GENRAND smoke dirs, script
`E:/agent-work/tmp/lanev5/run_smoke.py`, output `E:/agent-work/tmp/lanev5/smoke_out2`):
**passed**, 2 optimizer steps (10 s unroll, batch 4), finite, loss 15.73 -> 14.98 (two
different batches, untrained), lane_event 0.334 (unweighted; weight 10), predicted change rate
0.0026 / step, 603 TRAIN / 99 val windows, physics lane_gap_rule = speed, stopped relax on.
The first smoke (before the fast feasibility path, `smoke_out`) gave the identical lane_event
0.33449533581733704 on the same first batch, i.e. both mask paths produce the same rows on real
data. Gradient share: see Training.

**Pre-fit** (`tools/lanev5_pretrain.py`, modest TRAIN sample, CPU, 20.4 min:
`E:/agent-work/tmp/lanev5/pretrain/lanev5_pre.pt` + `.json`, sha256 3137...fab): 60 x 4 TRAIN
windows (227 windows, 49,800 steps, 1.08 M feasible rows, 3,797 events; overtake/return stay rows
kept at 25 % with weight 4 -> 278 k rows), 25 epochs Adam 2e-3; evaluated on 15 x 4 = 60 of the
99 TRAIN-scoped validation windows (272 k rows, 706 events; never wall / held-out sets):

| head | TRAIN-val rows | events | AUC | AP (constant) | log-loss (constant) | p/obs | TRAIN-fit AP |
|---|---|---|---|---|---|---|---|
| overtake 1->0 | 139,645 | 358 | 0.952 | 0.113 (0.0026) | 0.0124 (0.0179) | 1.56 | 0.402 |
| return 0->1 | 130,911 | 227 | 0.892 | 0.035 (0.0017) | 0.0109 (0.0128) | 0.98 | 0.310 |
| merge 2->1 | 1,704 | 121 | 0.926 | 0.608 (0.071) | 0.153 (0.256) | 1.05 | 0.865 |

All three heads beat the constant on log-loss and rank well (v4 hazard: AUC .66/.79/.49), but
the AP is below the tmp/laneranking boosted trees (.44/.16/.92 on many more rows) and the fit
over-fits this small sample (TRAIN-fit AP .40/.31/.87; loss still falling at epoch 25). With
the default gates (0.25 / 0.10 / 0.5) only the merge head would sample with this pre-fit. A
larger pre-fit sample (all TRAIN windows, GPU) and/or early stopping on TRAIN-val are the obvious
next steps; nothing was tuned on any evaluation set.

**Calibration tool** on a checkpoint built from this pre-fit (3 x 2 TRAIN-val windows, seed 3,
`E:/agent-work/tmp/lanev5/ckpt/calib.json`): AP .082 / .025 / .496; `--write-ap` wrote
`ckpt/ap/wm_s_ap.pt`, which loads with `head_ap` = those values and `lane_decision='gated'`.

## Cost

The teacher-forced pass now runs over the whole window (240 steps x batch) once per batch. On
this CPU (2 threads, loaded) one batch of 4 x 240 steps x 192 slots took 16.7 s for the rows
(after removing the physics-module mask, 79 s before) and the 10 s-unroll smoke update rose from
~17 s (v4 smoke, other load) to 28.6 / 40.1 s (v5, with the pre-fit running in parallel). On the GPU the pairwise feature pass is a few hundred
large kernels (cheap); the causal accumulator scan is ~240 x ~70 small elementwise kernels per
batch (estimate 0.2-0.5 s on gpuW's launch-bound loop). If that is too slow, the scan can be
vectorised or the event pass subsampled in time.

## Tests

`tests/unit/test_lanev5.py` (18): switch off (no module/keys/flat keys/state), heads per
transition + offsets, mask never lifted (+ CAVs keep physics), sampling never takes a masked
change, no driver input, features finite + causal (perturbing steps >= 7 leaves < 7 identical),
pre-t0 accumulator initialisation, chunk carry + teacher-forced equality, gated decisions,
end-of-lane forcing, event loss over every recorded step + decoupling, pre-fit tool round trip
(rows -> scaling -> fit -> save -> trainer load), metric helpers, checkpoint round trip, plumbing
/ validation, v5 config diff vs v4, fast feasibility == physics mask, fast/physics row paths agree.

`OMP_NUM_THREADS=1`: test_lanev5 (18) + test_lanefix, test_wmlane, test_gaprule, test_wmloss, test_pairloss, test_sched, test_c4_core, test_c4_dreamer, test_c4_futures, test_execcap, test_import_graph -> **390 passed** (3.6 min); ruff clean on every touched file.

## Not done / for the integrator

* No GPU training. Suggested order: run `tools/lanev5_pretrain.py` on a larger TRAIN sample
  (GPU or more CPU time), set `model.lane_v5_pretrain`, train; after training rerun
  `tools/lanehazard_calib.py --write-ap` on the selected checkpoint so `gated` uses the trained
  network's TRAIN-val AP (the pre-fit AP describes the pre-fit network).
* Recorded slot continuity: a car absent at a junction step restarts its accumulators (as the v4
  blocked counter); slot reuse without an absent step is not detected.
* Feature constants (100/200 m, 50 m, 2 m/s, 5 s, EMA taus, IDM/Krauss constants of the SUMO
  vType, LC2013-like keep-right formula) are the tmp/laneranking choices, not tuned.
* Deploy / RSU paths that rebuild the state every step (instead of carrying `lane_v5_acc`)
  re-initialise the accumulators from the driver history each call (as v4's blocked counter).
