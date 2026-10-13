# SRCWAIT: entry-queue (source waiting) term for C4 policy + MAPPO (2026-10-13)

**Problem.** In the first SUMO head-to-head (24 paired 10-min episodes), both learned controllers (the C4 policy trained in imagination and MAPPO trained in SUMO) cut on-road delay by about 13 % and braking by about 23 %. But they nearly doubled source waiting, 5.4k -> 10.2k veh-s. Source waiting is the time requested vehicles spend waiting to enter the network. As a result, total wait was worse than native. The cause: `team_reward` only charged vehicles present on the road, so a vehicle that could not enter cost nothing.

## 1. Reward (`src/wms/control/reward.py`, shared by both trainers)

* `team_reward(state, next_state, accel, weights, entry_queue=None)`. The queue can also be passed as `state["entry_queue"]`.
  * `entry_queue` [B] is the number of vehicles waiting to enter during the transition `state -> next_state`. It may be a time average, so fractional values are allowed.
  * The value is detached (it is a count) and clamped at >= 0.
* `source_wait = PHYSICS_DT_S * entry_queue`, in vehicle-seconds per 0.5 s physics step. It is a new returned component and is zero when no queue is given.
* `reward = (-w_delay * (delay + source_wait) + w_thr * throughput - w_smooth * smooth - w_hb * hard_brake) / max(present + entry_queue, 1) - w_fair * CVaR`.
  * One vehicle-second of entry waiting costs exactly as much as one vehicle-second of on-road delay.
  * Queued vehicles count in the per-vehicle normalisation.
* **Fairness:** the soft 90 % CVaR now covers present vehicles plus the queued ones. Each queued vehicle has delay `dt`, which is the maximal per-step delay.
  * This is implemented as one extra CVaR column whose weight is the queue count.
  * `_soft_cvar(delay, present, weight)` is exactly equivalent to replicating vehicles (tested).
* **Backwards compatible.** With no queue, or a queue of 0, every component is bit-identical to before (tested). `source_wait` is 0 in that case.
* `episode_summary` also sums `source_wait`.
* **Consequence for consumers:** the returned key set grows by `source_wait`. `test_c4_core` / `test_c4_branching` expected the exact old key set and were updated.

## 2. SUMO side (`traffic_shepherd/c4/env.py`, `src/wms/control/mappo/rollout.py`)

* `CavSumoEnv` records the RequestLedger source-wait increment of every 0.5 s physics step and exposes it per `step()`:
  * `info["source_wait_steps"] = [w1, w2]` (veh-s);
  * `info["entry_queue_steps"] = [w1/0.5, w2/0.5]`: the time-averaged queue of requested-but-not-inserted vehicles during each physics step;
  * `env.last_entry_queue_steps`;
  * `env.entry_queue_now()` = `ledger.due(now) - inserted`, an integer.
* `state["entry_queue"]` [1] is **opt-in** (`CavSumoEnv(entry_queue_in_state=True)`). It is off by default because deploy / rsu_check / coop_imagine repeat or gather every state key, and their behaviour must not change. MAPPO reads the queue from `info` instead.
* `control_step_reward(env, state, next_state, weights, info=None)` passes `q_i` for physics transition i.
  * Envs without a history (the fake) fold both steps into one 1 s transition with `q1 + q2`.
  * Envs without these fields get the old reward, unchanged.
* Episode/iteration logs now carry `source_wait`, `total_wait` (= delay + source_wait), `mean_source_wait_per_step` and `mean_total_wait_per_step`.

**Parity** (`_out/srcwait/parity.json`; native CAVs, warmup 300 s, horizon 300 s, 0 collisions / 0 teleports):

| run | summary window source_wait | sum of per-step queue x dt | integer queue (due - inserted at step start) x dt |
|---|---|---|---|
| seed 4107, 2700 vph, p0.2 | 4141.23 | 4141.23 (diff 9e-13) | 4077.0 (-1.6 %) |
| seed 4120, 2400 vph, p0.4 | 2781.85 | 2781.85 (diff 0) | 2720.5 (-2.2 %) |

* The integer count is lower because the ledger also charges each request that arrives inside a step from its request time to the step end. Integer + that partial term = the ledger, exactly.
* Under native control the queue is about 9-14 vehicles on average (max 23).
* With only 60 s of warmup the gap is larger, 13 % (small queue, unit test).

## 3. Imagination side (our policy; `src/wms/control/dreamer.py`, `start_states.py`, `train_cli.py`, `src/wms/model/entries.py`)

* **Switch:** `trainer.imagine_entries = true` (default; `configs/c4_policy.toml`) and `trainer.entry_pad_max = 256`.
  * `train_cli` builds the TRAIN start windows with `entries_v2=True`. Each window then carries `entry_feat`, its observed pre-t0 entry history (counter + insertion history, t <= t0 only). This is the same feature that WM-S entries-v2 training uses (`wms.data.pack_history.entry_features`).
  * Without `entry_feat`, or with the switch off, imagination is exactly as before (tested).
* **Padding:** each sampled batch gets empty entrant slots appended (`with_entry_slots`).
  * Size: 1.5 x the expected arrivals (rate x horizon) + 16, using the batch maximum, capped at `entry_pad_max`.
  * History tensors are padded with masked zeros, the same as the WM trainer's `extend_for_entries`.
  * Entrants may only use these padding slots. This follows WM training and avoids inheriting the driver parameters of an exited vehicle.
* **Entry process:** each WM member gets a fresh `TorchEntryProcessV2` per imagination, with:
  * `cfg.max_pending = 1e9`, so the backlog is uncapped (nothing is dropped);
  * per-row RNGs (`row_seeds`, a new optional argument), so all counterfactual rows of a start and all ensemble members share one seed. Their arrival streams are therefore identical until their traffic (blocking) differs, giving common random numbers.
  * Seeds come from a per-update generator, so the global RNG stream (variant choice, policy noise) does not change.
* **Backlog:** `_EntryProcessBase.backlog()` = pending arrivals summed over both entrances, per row. An arrival that finds no safe gap at its lane start, or no free slot, stays pending.
* **Transition states:** the state of transition t is the WM input *after* that step's insertions, so an entrant is charged from its first physics step. It gets `entry_queue` = the backlog after the step-t insertion, so `team_reward` charges `dt * backlog`.
  * Entrant CAVs follow the policy on all-CAV rows and the native behaviour on counterfactual rows (`act_mask` extended).
  * Insertions are written in place under `no_grad`, as in WM-S.
  * Gradients still flow through every WM output, and `entry_queue` itself is a detached count. The test confirms that actor gradients flow and are identical with and without activation checkpointing.
* **Logs:** `updates.jsonl` gains `source_wait`, `total_wait` and `entries`. `validation.jsonl` gains `delay`, `source_wait`, `total_wait` and `total_wait_vs_native`: all-CAV minus all-native total wait, summed over the horizon and averaged over starts with base rows (negative = less waiting than native). The selection score is unchanged (D_all / return), but the return now includes the queue penalty.
* **Smoke** (`--smoke`, CPU, batch 2, H 4; `_out/srcwait/c4_smoke`): completed.
  * Updates: `source_wait` 1.69 / 0.63 veh-s per control step next to `delay` 27.8 / 21.0.
  * Validation: `total_wait_vs_native` +7.5 / +5.9 veh-s (untrained actor).
* **Memory/time with activation checkpointing** (`_out/srcwait/entries_cost_fair.json`; CPU, 4 threads, batch 4, H 10, same 52 rows):

| entries | slots | autograd-saved | imagine + backward |
|---|---|---|---|
| off | 192 | 385 MB | 88 s |
| on | 222 | 489 MB (+27 %) | 159 s (+80 %) |

* The checkpointing path still works: entries on with checkpointing off stores 38 GB (`_out/srcwait/measure_run1.log`).
* Most of the extra cost comes from more slots and more present vehicles in the WM; the insertion runs on the host. Expect a similar relative slowdown on gpuW, plus a host sync per step while a backlog exists. Not measured on GPU.

## 4. MAPPO `--init-from` (`src/wms/control/mappo/train.py`, `trainer.py`)

* `--init-from <ckpt_XXXXX.pt>` loads policy, group encoder and critic (strict) into a FRESH `--out`.
  * An exported `policy_*.pt` is also accepted; it carries the actor only, so the critic stays fresh.
  * Optimisers, iteration counter, budget, RNG and return scaler are fresh.
* `--init-lagrange carry|reset`, default `carry`: carry keeps the checkpoint's `lagrange_alpha`.
* It is refused together with `--resume`, or when `--out` already has checkpoints.
* The record goes into the log config row, `RUN_SUMMARY.json`, checkpoint meta and `status.json["init_from"]`: path, sha256, kind, alpha before/source/used.
* **Resuming an interrupted warm-started run:** rerun with the same `--out` and without `--init-from` (automatic resume).
* **Smoke** (real SUMO, 2 workers, 2 iterations, horizon 60 s, warmup 120 s; `_out/srcwait/mappo_smoke`):
  * Episode rows carry `source_wait` (30-33 veh-s) and `total_wait`.
  * Iteration rows: `mean_source_wait_per_step` 0.53 / 0.52, `mean_total_wait_per_step` 37.6 / 37.7.
  * `--init-from ckpt_00002.pt`, 1 iteration (`_out/srcwait/mappo_initfrom`): completed, `init_from` recorded.

## Tests

* New: `tests/unit/test_c4_srcwait_imagination.py` (8) and `tests/unit/test_c4_srcwait_mappo.py` (8; SUMO parity test is skipped without SUMO).
* `pytest tests/unit/test_c4_*.py tests/unit/test_import_graph.py tests/unit/test_lane_rule.py`: **403 passed**.
* Existing tests updated:
  * the new `source_wait` key (core, branching);
  * the CLI `_load_starts` stub signature (dreamer);
  * a stale stage_steps expectation (loop; the config has been 200/400/150 since cfc1bbc).

## Limitations

* **The imagined queue starts empty at t0.** Vehicles already queued in SUMO are not RSU-observable; the counter counts insertions only. Imagination therefore sees only the queue built up within the horizon (30 s): the marginal effect of blocking, not the absolute level. Queued SUMO vehicles are typically about 9-14 under native control.
* **Insertion rules differ.** Imagined insertion uses the WM entries-v2 safe-gap rule (min gap + 1 s headway + braking, sampled entry gap), not SUMO's insertion logic, so imagined and SUMO queue magnitudes are not calibrated against each other.
* **Mainline queue is out of reach.** The mainline entrance is 2.6 km upstream. Within 30 s, imagined spill-back can only reach the ramp entrance (378 m upstream), which is the likely main channel anyway.
* The coop_imagine / branching team metrics still have no entries (unchanged).
* The single-transition fallback (fake env) charges `q1 + q2` in one transition; its normalisation count therefore differs from the real two-transition path.
