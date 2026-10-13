# WMLOSS_NOTES — C4 world-model v2 loss (branch c4/wmloss, 2026-10-13)

Why: the C4 policy's decisions and team reward read AGGREGATE traffic (RSU cell field
`wms/control/rsu_field.py`; reward delay / throughput / entry queue `wms/control/reward.py`),
but WM-S training only penalised per-vehicle position / speed / lane of the t0 cohort, and the
C4 WM stopped at 60 s with no arrivals. Two opt-in changes, both OFF by default (bit-identical):

## B. Aggregate-consistency loss — `loss.aggregate` (`src/wms/train/losses/aggregate.py`)

Three chunk terms appended LAST to the composer (pre-existing summation order untouched):

| term | what is compared (rollout vs recorded, same window) | normalisation |
|---|---|---|
| `agg_field` | RSU cells (50 m x lane, `rsu_field.CELLS` order): density (soft count / cell km), mean speed (membership-weighted, SPEED_LIMIT pseudo-count = empty-cell convention), queue (soft slow AND occupied) | per feature scale (50 veh/km, 10 m/s, 1), mean over cells, Huber(1) |
| `agg_reward` | on-road delay `dt*relu(1-v/31.29)` (whole road + RSU zone) and throughput = soft same-vehicle crossings of MERGE_S1 (1 m width, as the reward) | delay error per recorded vehicle-second; throughput / 2 veh |
| `agg_flow` | soft same-vehicle crossings of every cell boundary (all lanes of the cross-section): matching all boundary fluxes = matching each cell's inflow - outflow (macroscopic conservation) | / 2 veh per boundary, mean over boundaries |

* Soft cell membership: `sigmoid((s-lo)/tau) - sigmoid((s-hi)/tau)` (tau = `cell_temp_m`, 5 m), lane
  indicator hard (lanes are discrete decisions). The hard limit equals `build_rsu_field` (unit test).
* Fields are averaged over `bin_s` (5 s) bins by absolute step index, so chunked BPTT and the
  single-chunk value agree when bins align (unit test); `every_steps` subsamples field steps.
* Horizon ramp `horizon_start_s` -> `horizon_full_s` (config: 5 -> 30 s). Because each component
  is normalised by the unroll's total ramp weight, the ramp shapes WHERE in a window the aggregate
  counts (late steps, where per-car identity matching thins out), not the term's overall size.
* Each component is a weighted mean over all (row, bin) of the whole unroll (`prepare` computes the
  unroll-wide weight), so chunk contributions sum to one horizon-invariant mean.
* Masked where the recorded future is unavailable (`road_valid`: window padding past the run end).
* Recorded side: with the entry process running AND the window carrying the recorded on-road
  presence (`road_present` = pack `veh_mask`, `road_continue` = `target_mask`, `road_valid`), the
  target is the FULL road including post-t0 arrivals and the rollout side includes the entrants.
  Otherwise both sides use the t0 cohort (alive-since-t0 presence; entrant padding excluded).
* Logging: `agg_field` / `agg_reward` / `agg_flow` in every train/val record, `aggregate_value_share`
  (share of the logged loss value), and gradient shares in `loss_share` when
  `loss.loss_share_logging` is on (c4_wm has it on). Gates (`required_terms`) require the terms.
* Constants are restated in the train layer (the import graph forbids train -> control); a unit
  test asserts equality with `wms.control.contract`, `rsu_field.CELLS` and the reward crossing width.

## E. Long horizon + entries in training

* 120 s stage: c4_wm already carries the R5b stabilisers (grad_decay_per_step 0.90, per-step
  activation checkpointing, per-stage truncation and lr); the v2 config adds the 120 s stage with
  truncation 120 (0.90^120 ~ 3e-6, longer chunks add cost, not gradient) and lr 1e-4.
* Entries: `data.entries_v2 = true` turns on the existing entries-v2 path (observed pre-t0 entry
  history `entry_feat`, `TorchEntryProcessV2` in `wms/train/inputs.py`, the same process the C4
  imagination uses via `start_states` / `dreamer.with_entry_slots`). New switch
  `data.entry_pad_auto` (+ `entry_pad_max`, default 256): pad = 1.5 x observed rate x unroll
  horizon + 16 per batch (the legacy fixed `entry_pad = 8` cannot hold 120 s of arrivals) and an
  uncapped pending backlog (blocked arrivals wait, as in SUMO and the C4 imagination). The process
  object carries `train_pad`; the timeline padding uses it.
* The entries-v2 loaders (`_PackWindowLoader`, `C4WindowLoader`) now also emit `road_present`,
  `road_continue`, `road_valid` (TRAIN-window labels for the aggregate loss only; never model
  inputs); `collate` stacks them when every record has them.
* Entrants stay out of the per-car loss (no recorded identity), but the aggregate loss counts them.

## Config `configs/c4_wm_v2_loss.toml` (extends c4_wm.toml)

entries_v2 + entry_pad_auto; loss.aggregate weight 20 (field 1, reward 1, flow 0.5, 5 s bins,
ramp 5 -> 30 s); stages 10/30/60/120 s, stage_steps [1200, 900, 400, 140], truncation
[20, 60, 120, 120], lr [1e-3, 3e-4, 1e-4, 1e-4], max_hours 6.5; out_dir runs/active/c4_wm_v2_loss.
Data windows become 240 steps (the loader horizon = the longest stage).

Budget (gpuW, batch 16, measured c4_wm step times 1.7 / 5 / 12 s at 20 / 60 / 120-step unrolls =
0.085 / 0.083 / 0.100 s per simulated step). 240-step estimate WITHOUT the new terms:
~0.105 s x 240 = ~25 s per step. Overhead of the new terms: entrant slots grow with the horizon
(smoke: 15 entrants in 10 s over 4 rows ~ 0.375 veh/s per row -> pad ~22 / 33 / 50 / 84 on top of
192 recorded slots), the aggregate terms (~5 %) and the per-step entry process (host sync when an
arrival is due). Assumed factors 1.20 / 1.25 / 1.30 / 1.45 -> 2.0 / 6.25 / 15.6 / 36 s per step
-> 2450 + 5625 + 6240 + 5040 s = 5.4 h + validation ~0.4 h = ~5.8 h (<= max_hours 6.5).
UNMEASURED on a GPU: the CPU A/B on the shared dev box was CPU-starved (100 % load from other
agents; 4 windows took 298 s to load and a 20-step forward/backward 454 s, all inside existing
model / history code), so the factors are estimates. If the first 120 s steps on gpuW exceed
~40 s, cut stage_steps[3].

## Tests (`tests/unit/test_wmloss.py`)

switch-off bit-identical (flat forms of default / c4_wm / r4 / r5b carry no new keys; explicit
off values give identical records and gradients; aggregate ON leaves position/speed/lane logged
values unchanged), aggregate zero on identical trajectories (full-road and cohort), positive and
differentiable otherwise (chunked), chunk-sum = single chunk, masking of unavailable future,
horizon ramp, soft-field hard limit = `build_rsu_field`, contract constants, adaptive pad formula,
entries-in-training shapes + full-road / cohort targets + one forward/backward, road presence +
collate, the v2 config reaching settings / composer / required log terms.

## Smoke / results

* Real-data probe (configs/c4_wm_v2_loss.toml, pack_v3 + dspeed windows, CPU, batch 4):
  batch keys include `entry_feat`, `road_present [4,241,192]`, `road_continue [4,240,192]`,
  `road_valid [4,241]`; 4-step unroll: aggregate 0 (ramp starts at 5 s), entrants 1;
  20-step unroll: loss 13.32, position 11.52, agg_field 0.368, agg_reward 0.219, agg_flow 0.448,
  aggregate_value_share 0.078, entrants 15 (finite).
* `tools/c4_smoke.py runs/active/c4_wm_v2_loss_smoke configs/c4_wm_v2_loss.toml` (full C4 data mix
  incl. the switching direct-mix arm: 2041 train / 261 val windows; CPU, batch 4, 10 s):
  status passed, 2 optimizer steps, finite; loss 19.09 -> 18.00; position 15.29 -> 14.08;
  agg_field 0.278 / 0.274, agg_reward 1.97 / 2.22, agg_flow 0.622 / 0.569; aggregate value share
  0.150 / 0.170; entrants 18 / 15; pre-clip grad norm 10.1 / 4.4. CPU profiles (starved box):
  10 s 73.7 s, 30 s 292 s per forward/backward - not representative of gpuW.
* Weight 20 thus puts the aggregate at ~8-17 % of the loss value on an untrained model; read the
  gradient shares (`loss_share.jsonl`) in the first gpuW stage before trusting it at 60/120 s.

## Unit suite (tests/unit, full)

896 passed, 2 failed, 1 skipped (49 min on the loaded dev box). Failures not caused by this
branch: `data/test_drive_registry.py::test_committed_registry_matches_the_generator_output` also
fails on the base commit; `test_c4_mappo.py::test_worker_pool_spawn_survives_worker_death` passes
when re-run alone (15 s; timing-sensitive under the 100 % CPU load). New tests: 16 in
`tests/unit/test_wmloss.py`, all pass; ruff clean on the changed files.
