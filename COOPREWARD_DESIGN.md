# COOPREWARD_DESIGN.md — cooperation (team) reward option for the V2V PPO runtime

Branch `c4/coop-reward` (base `c4/rerun-cpus`). READ-first documentation of the current
reward/advantage path in the V2V runtime, followed by the registered design of the
`coop_reward` option. CPU only; no SUMO/GPU/ssh; default OFF is byte-identical.

## 1. Where the training reward is computed today

**Plant (`tools/coopmwm/accel_plant.py`, one job = one 1200 s episode).** The SUMO-worker
loop (`episode`, line ~78–103) keeps `before = b.totals['team_delay_vs']`, runs exactly one
1 s control tick (`cf.tick`), then appends the step:

```python
steps.append(dict(controller.events[-1], reward=before - b.totals['team_delay_vs']))
```

* `b.totals['team_delay_vs']` is the **cumulative source-inclusive team delay** of the whole
  simulation: per 0.5 s substep the physics adds
  `0.5 * sum_v max(0, 1 - v_v/SPEED_LIMIT_MPS)` over **all on-road vehicles (CAV and HDV)**
  plus the entry-queue `source_wait_vehicle_seconds` increment
  (`traffic_shepherd/c4/env.py:617–644`, `SPEED_LIMIT_MPS = 31.29` at line 85; accumulated in
  `tools/coopfree/backend.py:61–64`).
* Therefore the per-step reward `r_t` is the **negative increment of the global team delay J**
  during second `t`: `r_t = -(J_t - J_{t-1})` in **vehicle-seconds per second** (road delay +
  source wait, summed over the whole world, *not* normalised by vehicle count). One scalar per
  step, shared by every CAV of that step. There is no per-CAV reward decomposition in the
  receipts: `r_t` is global, and the per-vehicle observable rows in each step
  (`step['rows']`, whitelist `{'vid','s_m','v_mps','lane','is_cav','length_m'}`,
  `tools/coopnative/policy.py:8`) are the **pre-step observable frame** of the same tick.
  Per-update receipts label this `reward_source='SUMO source-inclusive team J only'`
  (`warm_runner.py`, update receipt; the registered team-J horizon metric is
  `team_delay_vs` at 600/900 s).
* No other reward terms exist in the V2V PPO path: no shaping, no per-CAV delay, no
  convenience/comfort terms. Executed plans do **not** change the reward; they only add
  imitation targets (AWR / bank pairs).

## 2. PPO advantage path (identical for arms A and B/R)

**`tools/coopmwm/accel_learner.py`** (`UNIT = 10000.` veh-s return unit,
`GAMMA = exp(-1/600)`, `LAMBDA = exp(-1/1200)`):

```python
def advantages(steps):          # GAE(λ, γ) over the episode's 1 s steps
    delta = s['reward']/UNIT + GAMMA*next_value - v        # s['reward'] = global team-J delta
    carry = delta + GAMMA*LAMBDA*carry
    -> (advantage, advantage + value) per step
```

* Arm A (plain PPO, also V2V-A): `accel_learner.update` — own-action PPO ratios
  (executed plan steps are excluded from the ratio), all-step critic on clipped squared error
  toward `advantage + value`, advantage normalisation over own steps only
  (`accel_learner.py:92–133`).
* Arms B/R (V2V): `exact_grade._ppo_exact_update` (`exact_grade.py:915–970`) — **exactly the
  same `advantages()`** over the same steps, own-step PPO + all-step critic; bank pairs enter
  only the separate imitation phase (never the PPO loss).
* Scale: rewards are vehicle-seconds; the GAE divides by `UNIT=10000`, so advantages/returns
  are in units of 10⁴ veh-s. Both update entries consume `ep['steps']` directly — the reward
  only enters training through `advantages(ep['steps'])`.

## 3. Is MAPPO's critic already centralized/shared?

**Yes — one joint, observable-state critic.** `tools/coopnative/policy.py` (`Policy.forward`):
the value head is a single MLP over `[mean over all observed vehicle feature rows x; RSU
per-lane aggregates]` and returns **one scalar value per control step** (shared across all CAVs
of that step, `float(value)` stored in `sample['value']`). It pools the full observable frame
(all vehicles within the sensing contract, CAV and HDV alike) plus the RSU aggregates — a
CTDE-style centralized critic over observable state, not per-CAV value heads, and not
ground-truth/global hidden state (the same whitelist rows the actor sees). The V2V runtime
widens the same critic input with the V2V partner blocks (`v2v_runtime.Policy`,
`v2v_model.extend_for_v2v`); the value remains one scalar per step.

Consequence for a cooperation reward: because the runtime trains one joint policy with one
per-step scalar reward and one shared critic, a per-CAV reward `r_i` is represented by the
shared scalar; the cooperation option therefore adds the neighbourhood-outcome term to the
per-step reward and it reaches every CAV of that step through the same GAE (see §5).

## 4. What the receipts already expose (observability contract)

Per step (`RESULT.json` steps, reused verbatim at update time):
`time_s`, `sample` (context `x/rsu/members/anchors/v2v`, joint `value`, `logp`, behaviour),
**`rows`** — the observable frame at the decision instant (per-vehicle `vid, s_m, v_mps, lane,
is_cav, length_m`; the *next* step's `rows` is exactly the post-step observable frame of this
second), `commands`, `group`, `requests`, `reward` (global team-J delta). The V2V observable
neighbourhood contract is the 100 m radio/sensing range (`v2v_obs.V2V_RANGE_M ==
wms.control.contract.SENSOR_RANGE_M == 100.0`; `encode()` senses
`abs(s_j - s_i) <= 100` over **all** rows, CAV and HDV).

Not observable in receipts: per-vehicle source-wait (entry-queue delay is only a whole-world
counter), mid-substep speeds (the env's road-delay estimate samples t+0.5 and t+1), and any
per-vehicle decomposition of `r_t` itself.

## 5. Registered option `coop_reward`

Formula (per CAV `i` at charged second `t`):
`r_i' = r_i + w * mean_{j in N_i(t)} r_j,t`, joint step reward `r'_t = r_t + w * mean_i(...)
` (equal weighting of every present CAV's augmented reward — the shared-reward semantics of the
joint policy/critic).

* `r_i` = the runtime's own per-step reward `r_t` (unchanged; identical for all CAVs, as today).
* `N_i(t)` = **only observable vehicles** (CAV **and** HDV rows, no driver-type oracle) of the
  SAME observable frame, within the SAME 100 m range the CAV already senses/receives via V2V
  (`v2v_obs.V2V_RANGE_M` / `SENSOR_RANGE_M`); neighbours are ranked/deduplicated by vid; no
  lane filter (radio range, as in `v2v_obs._ranked_partners` and `encode`'s sensed set).
* Outcome frame: the **post-step observable frame** (next step's `rows`), because the env's own
  interval delay estimate samples post-substep speeds; the charged second's final step falls
  back to its own (pre-step) frame and the fallback is counted in the report.
* `r_j,t` = observable per-vehicle outcome of neighbour j =
  `max(0, 1 - v_j/SPEED_LIMIT_MPS)` in vehicle-seconds (the exact speed-deficit form of the
  runtime's road-delay term, `SPEED_LIMIT_MPS = 31.29`, mirrored constant + unit test against
  `wms.control.contract`).
* **Background-traffic outcome term: NOT added.** The clause admits it "only if already
  observable in training receipts": the receipts contain no per-vehicle decomposition of
  background outcomes (no per-vehicle source wait, no mid-substep frame), and the own reward
  `r_t` already carries the whole-world (background included) team outcome. Adding a
  receipts-derived global term would double-count; omitting it keeps the registered formula
  exactly as specified.
* A CAV with an empty neighbourhood contributes 0 to the CAV-mean (counted in the report).
* `w` (`coop_reward_weight`) is a **REGISTERED config value**: a positive finite float; the
  config **refuses a missing `w` when the option is enabled** (no default, no tuning default);
  `coop_reward_provenance` (non-empty string naming the registering authority/decision) is
  **required** when enabled. Keys:
  - `coop_reward` (truthy = enabled; absent/False = OFF, byte-identical),
  - `coop_reward_weight` (required when ON),
  - `coop_reward_provenance` (required when ON),
  - `coop_reward_arms` (optional; default `('B','R')`; must contain **B and R together, A only
    by explicit inclusion** — B/R are pinned identical, one key set serves all arms).
* **Off identity**: with the option OFF (or the arm not enabled) the update inputs, the GAE,
  the receipts and every other byte of the run are identical to the branch point; the
  `coop_reward` receipt key is absent OFF.
* **Where applied**: once per arm-update in `warm_runner.run_arm`, producing a *new* list of
  episode dicts (per-step shallow copies with `reward = r_t + w*team_t`) passed **only** to the
  PPO/imitation update call (`accel_learner.update` for A, `exact_grade.exact_update` for B/R).
  Everything else that consumes `eps` (viability/evidence critic diagnostics, health/safety,
  practice reports, `team_only_noise`) keeps the raw registered team-J rewards, so bank grading,
  practice receipts and diagnostic semantics are untouched.
* **Per-update logging**: the arm's `{arm}_UPDATE_{u:03}.json` gains a `coop_reward` block only
  when the option is ON for that arm: weight/provenance/arms, per-update `own` vs `team`
  components (sum and mean of `r_t`; sum and mean of the added `w*team_t` and of the raw
  neighbourhood mean), ego/neighbourhood counts (empty-neighbourhood egos, egos, steps,
  own-frame fallback steps) plus the pinned observable constants (range, speed limit).
* **Registration points** (all additive, all default-OFF identical):
  - `v2v_runner.prepare_intent(..., coop_reward=False, coop_reward_weight=None,
    coop_reward_provenance=None, coop_reward_arms=None)` — the registered producer writes the
    keys only when enabled and validates them (missing `w`/provenance refused);
  - `v2v_live.hooks` re-validates at hook time (B/R pinning) before the runner binds;
  - `v2v_launch3.bind_runtime` refuses a deployed runtime whose coop keys are malformed or
    B≠R (absent keys = unchanged behaviour);
  - `warm_runner.run_arm` refuses a malformed config at use time (fail-closed).
