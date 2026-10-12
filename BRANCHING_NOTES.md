# SUMO saved-state branching (C4 cooperation measurement)

`traffic_shepherd/c4/branching.py` measures cooperation in the real simulator, identically
for any per-CAV controller (our C4 policy, MAPPO, CIRCLES damping, PI, random, ...).
Design: docs/WM_C4_DESIGN.md section 6 ("cooperation as the main training signal").

## CLI

```
PYTHONPATH=src python -m traffic_shepherd.c4.branching --controller <spec> [--base-controller <spec>] --out runs/active/<NEW_DIR> \
    [--seeds 4107-4110 --demand 2700 --penetration 0.2 --warmup 300 --points 2 --spacing 60 \
     --first <s> --horizon 30 --extra-solo 0 --n-flip 2 --link-m 150 --k-slots 192 --rng-seed 0 \
     --wm-ckpt <wm_s_best.pt> --wm-samples 4]
```

`<spec>`: `native | damping | pi | random | zero | module:attr | <policy checkpoint>`
(`wall` is a group action and is refused). Decision times: `first + k*spacing`
(default `warmup + spacing`). Seeds go through the C4 seed guard (4107-4148 only); restricted
path tokens are refused; `--out` must be new/empty.

Outputs: `branches.jsonl` (one record per branch: arm, members, metrics, wall-clock),
`points.jsonl` (per decision point: group G, metrics per arm, D for every metric, `coop` =
D_all / D_i / D_G / D_{G\i} / sum_D_i / C_G / M_i / I_i, determinism and save/load fidelity,
optional imagined block), `BRANCHING_SUMMARY.json` (bootstrap 95 % CIs over decision points
and seeds, determinism, fidelity, wall-clock), `MANIFEST.json`, `states/*.xml`.

## What it does

1. One base episode runs the controller on every CAV (native warmup). At each `t_k` the SUMO
   state is saved (`saveState`, launched with `--save-state.precision 17 --save-state.rng
   true`, the planner/closed_loop.py convention) and the env's Python state (slot pool,
   history, RequestLedger, CommandOwnership, PI session) plus the controller object are
   deep-copied (the TraCI module and the stateless CIRCLES ONNX session are shared).
2. Arms from each saved state, `H` s each, identical RNG: `B` (all native), `B_repeat`,
   `ALL`, `ONLY:i` (every member of G + `--extra-solo` sampled CAVs), `GROUP` (G = largest
   chain of CAVs with consecutive |ds| <= 150 m along s, any lane, size 2-6, cut to the
   tightest 6-window; ties -> nearest the merge end), `FLIP:i` (G minus i, `--n-flip`
   sampled members). CAVs entering during a branch are native except in `ALL`.
3. Metrics per branch: `wms.control.reward.team_reward` (default RewardWeights) on every
   0.5 s step of the SUMO slot state (exits carried, reused slots split, so identity is exact):
   reward, delay, soft throughput, smooth, hard_brake, cost_ttc, per-step CVaR; plus
   RequestLedger source waiting (`team_delay_vs` = on-road delay + source waiting, the
   primary metric), system vehicle-seconds, exact CVaR90 of per-vehicle branch delay (incl.
   source waiting), merge-end crossings, arrivals, braking, true-gap TTC<3 s exposure,
   collisions, teleports.
4. Sign convention: `D_X[m]` = improvement of X over B (cost_B - cost_X for costs,
   value_X - value_B for benefits); POSITIVE = BETTER. `C_G = D_G - sum_{i in G} D_i`,
   `M_i = D_G - D_{G\i}`, `I_i = M_i - D_i`.

Controller plug-in (`BranchController`): any `state -> accel [1,K]` callable (state = the
contract dict of `CavSumoEnv.state()`); outputs may also be `(accel, lane)` or
`{"accel", "lane"}`: lane commands are passed as `lane_by_slot` (member slots only, others 0)
when `env.step` accepts that keyword (lane action on branch c4/policy2), otherwise counted in
`lane_commands_dropped`. Named `damping`/`pi` use the env's per-tick fallback path; subset
arms use the new `env.step(..., fallback_ids=...)` (other CAVs released to native). The base
episode uses the same `direct`+fallback path as the arms. A stateful callable is deep-copied
per arm (MAPPO recurrent state, RandomAccelPolicy RNG).

## MAPPO vs ours

Run the same CLI twice with the same seeds/points/horizon and the same
`--base-controller` (e.g. `native` or `damping`), once with `--controller <ours>` (checkpoint
via the `make_policy` loader or a `module:attr` factory) and once with `--controller
<mappo module:attr factory returning state -> accel>`: both are then branched from IDENTICAL
saved start states (same groups G), and C_G, sum M_i, D_all are directly paired across the
two runs (bootstrap CIs per run; pair by (seed, t_s)). Without `--base-controller` each
controller is measured on its own on-policy start states. With a different base controller
the arms' callable starts each point from its reset state and the live fidelity check is
skipped.

## Cost (this machine, CPU, demand 2700 / pen 0.2 / H 30 s)

- 11 branches + 30 s live check per decision point: 25-28 s wall (single process, ~2.3 s per
  30 s branch including a fresh SUMO launch + `loadState`).
- Two processes concurrently + `--wm-ckpt` (R5b, 4 samples, CPU): 170-290 s per point, of
  which imagination 80-150 s.

## Smoke (seed 4107, 2700 veh/h, 20 %, warmup 300, points t = 360/420, H 30)

`runs/active/C4-BRANCH-SMOKE2-20261013/{damping,random}` (with `--wm-ckpt` R5b best) and
`runs/active/C4-BRANCH-SMOKE3-20261013/damping` (no WM; identical SUMO numbers). G = 4 / 6
CAVs, 11 arms per point, 0 collisions/teleports in every arm.

| controller | t | D_all | D_G | sum D_i | C_G (team_delay_vs, veh-s; + = better) |
|---|---|---|---|---|---|
| damping | 360 | -47.6 | -8.5 | -8.3 | -0.25 |
| damping | 420 | -112.2 | -34.4 | -30.7 | -3.7 |
| random | 360 | -1099.9 | -135.2 | -211.4 | +76.1 |
| random | 420 | -1171.4 | -222.1 | -364.5 | +142.4 |

Damping hurts versus all-native over 30 s at this state (sub-additive group effect); random
acceleration hurts much more, and a group of random CAVs hurts LESS than the sum of solo
damage (C_G > 0 is "less-than-additive harm", not cooperation; the w_coop <= w_ind guard of
design 6 exists for exactly this). Imagined (R5b, oracle state) vs SUMO arm improvements
(18 arms): random reward r = 0.92, delay r = 0.98, sign agreement 0.89; damping reward
r = -0.62, delay r = -0.77 (imagined damping is wrong-signed: the imagination replays the
damping timeline only on member slots, released CAVs are modelled as encoder-identified
drivers; investigate before using it as a cooperation validator for named controllers).

## Checks

- Determinism: `B` vs `B_repeat` identical in all 4 points (max |ds| = 0, metrics equal).
- `loadState` into the running SUMO is NOT clean: pre-load simulator state leaks into the
  first steps (observed B != B_repeat, 2 spurious junction collisions). Every load therefore
  goes into a FRESH SUMO process (same launch command) - fixed and verified.
- speedMode is not stored in the state file (all CAVs revert to 31); re-asserted to 27 for
  every CAV the ownership had configured (counted in `speed_mode_restored`).
- Save/load fidelity (`ALL` after a reload vs the uninterrupted base continuation): exact for
  random (both points) and in the unit test (t = 70); damping run: max |ds| 4.7 m / 3.3 m
  over 30 s, starting at the first half-step on slow queued HDVs (v ~ 1-2 m/s) in the merge
  zone - a SUMO state-file limitation, not a controller-path issue. All arms of a point share
  the identical reloaded start (common random numbers hold exactly); the base episode is
  resumed from a reloaded state after each point.

## Limitations

- Reloaded world can differ slightly from the uninterrupted one in congested merge states
  (above); report `fidelity_ALL_vs_live` alongside every result.
- Group selection is geometric (150 m chain on s, lane-agnostic); one group per point.
- 2 decision points x 1 seed in the smoke: CIs are not meaningful yet.
- Imagined helper (`src/wms/control/coop_imagine.py`): no entries/source waiting, oracle
  state, lane changes only via the forced merge; callable controllers run per batch row
  (slow for large sample counts).
