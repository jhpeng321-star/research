# SRCGRAD: gradient for the imagined source wait (C4 Dreamer, 2026-10-13)

**Problem.** `team_reward` charges `source_wait = dt * entry_queue` (SRCWAIT_NOTES.md). In imagination,
`entry_queue` is the backlog of the discrete entry process `TorchEntryProcessV2`, so the term had a value
but **zero gradient**. The actor only felt on-road delay, and the cheapest way to cut that is to keep
vehicles out of the network. Slow CAVs near an entrance block insertion, and that blocking was free for
the actor. The phase-4 SUMO head-to-head showed exactly this: source wait 10,656 vs 5,434 veh·s (native),
on-road delay -8 %, total wait +12 %. The critic could not see the backlog either, so the lambda-return
bootstrap could not price queues beyond the 30 s horizon.

## 1. Straight-through insertion-blocking surrogate (`src/wms/control/srcgrad.py`)

The reward value stays the **exact** backlog. Only the gradient comes from a smooth model of the insertion
test:

```
entry_queue = backlog + q_sur - q_sur.detach()                 (value = backlog exactly)
q_sur[b]    = sum_e D[b,e] * p_block[b,e]                      (vehicles)
D[b,e]      = pending[b,e] before this step's insertion + rate[b,e] * dt
p_block     = p_lane(0) * p_lane(1)   mainline (an arrival is refused only if BOTH lanes fail)
              p_lane(2)               ramp
p_lane(l)   = 1 - sigmoid((g - req)/tau) * Phi((g - gap_mean[e]) / sqrt(gap_std[e]^2 + tau^2))
g           = soft-min over present vehicles on lane l of (s - length - entry_s[l])
              (weights softmax(-gap/2 m); the same weights give the soft rear speed v_lead)
req         = min_gap + headway * v_new + relu(v_new - v_lead)^2 / (2 * decel)       (EntryConfig)
v_new       = relu(speed_mean[e] * mean speed of the n_nearby (3) rearmost vehicles on l)
tau         = 1 m
```

**What it mirrors.** This is the soft version of `TorchEntryProcessV2._try_lane`, with the same entrance
positions and lanes:
* `ENTRY_S_M` / `process.entry_s`: mainline lanes 0/1 at the upstream boundary (-2645.67 / -2644.22 m),
  ramp lane 2 at its start (-378.66 m);
* the same rear vehicle (the rearmost on the lane, its length), the same local-speed rule (mean of the
  3 rearmost) and the same `safe_to_insert` formula from the process's own `EntryConfig`;
* the sampled per-arrival required gap `u_gap ~ N(gap_mean, gap_std)`. Its exact acceptance probability
  is the normal CDF; `tau` only widens it.

Entry history parameters (rate, speed ratio, gap mean/std) are read per row from the process
(`history_tensors`). The surrogate is evaluated on the **pre-insertion** state of each step: the recorded
post-insertion state with that step's new slots masked out. This is exact, because insertion only writes
free slots.

**Why this form (`d source_wait / d p_block[e] = dt * D[e]`).**
* Losing one insertion opportunity at an entrance delays every vehicle waiting there (the backlog plus
  this step's arrival) by one step. That is the first-order queueing cost of a lost service slot. It is
  a slight underestimate when the loss extends the busy period.
* The realised 0/1 "arrival due" indicator is replaced by its expectation `rate * dt`. Blocking does not
  depend on whether an arrival happens to be due, so this is a Rao-Blackwellised, dense gradient. It is
  nonzero even when the queue is empty but arrivals are expected.
* Simplification: the second mainline slot (two arrivals in one step, one per lane) is ignored.

**Where the gradient goes in the reward (`reward.py`).** A queue that requires grad passes its gradient
**only to `source_wait`**. The normalisation count (`present + queue`) and the fairness-CVaR weight of the
queued column stay detached. The reason: blocking moves a vehicle between road and queue; it does not
change how many vehicles there are. So `d reward / d q_sur = -w_delay * dt / count` (tested).

**Sign.** Slowing the rear vehicle at an entrance keeps the gap small, so `p_block` and the cost go up.
On a synthetic ramp scene (a CAV 8 m past the ramp start at 2 m/s, 3 queued arrivals, 8 steps through
`team_reward`), d(sum reward)/d(accel) > 0, and it is larger than without the surrogate (test).
* Instantaneous effect of the speed: a slower rear platoon lowers `req` (SUMO-like insertion accepts
  tighter gaps at low speed). That is the process's own rule and it is kept.
* The positional effect dominates after about 1 s (`d(gap - req)/da ~ t^2/2 - headway*t` for the
  safe-gap part; the sampled-gap part depends on the gap only).

**Switch.** `trainer.source_wait_grad` (default false). Off: the surrogate is never evaluated and the code
path is unchanged.

## 2. Critic sees the per-entrance backlog

* `trainer.critic_entry_backlog` (default false) needs `TeamCritic(entry_backlog=True)`. The trainer raises
  if the flag and the critic disagree, and `train_cli` builds that critic when the flag is on.
* The value head gets `state["entry_backlog"]` [B,2] (mainline, ramp; pending vehicles after the step's
  insertion) divided by 10, concatenated to the pooled set embedding. Zeros are used when absent (no
  entries).
* The imagination writes it into every transition state and into the terminal (bootstrap) state, so the
  lambda-return bootstrap can price the queue still waiting at the horizon.
* **Checkpoints:** the head has 2 more inputs, so loading an older critic (or full `--resume` of an older
  run) into it fails loudly (size mismatch, tested). `--init-policy` loads the actor only and is
  unaffected. The final run is from scratch.
* **Group signal / actor: not changed.** Adding the queue there would change the actor architecture and
  the deploy path, because SUMO would have to supply per-entrance queues to the actor. That is not clean,
  so it is out of scope.

**Observability assumption (for the owner).** The critic is training-only (privileged, like its
all-vehicle input), so nothing at deployment depends on it. Even so, the per-entrance queue is something
an RSU can measure with its own detectors at the entrances (a queue/occupancy count of requested but not
inserted vehicles). The imagined backlog is the WM entry process's estimate of that count; it starts
empty at t0, see SRCWAIT_NOTES.md limitations. If the queue were later given to the actor/RSU encoder,
the detector-based measurement must be what is deployed.

## 3. Config

* `configs/c4_policy.toml`: `source_wait_grad = false`, `critic_entry_backlog = false` (documented, inert).
* `configs/c4_policy_final.toml` extends `c4_policy_gpuw_p4.toml` and sets:
  * both switches true;
  * `output_dir = runs/active/C4-POLICY-FINAL`.
* It is meant for a **from-scratch** run: no `--init-policy` / `--resume`. Old critics cannot load anyway.

## 4. Tests

* `tests/unit/test_c4_srcgrad.py`: 8 tests:
  * switch off never evaluates the surrogate, and the update is reproducible bit for bit;
  * a constant queue gives an identical reward through the ST path, and `d reward/d q = -w dt/count`;
  * the ST queue value equals the exact backlog at every step, and all rewards, returns and components
    are bit-identical off vs on;
  * the source-wait sum reaches the actor;
  * checkpointing parity with the surrogate and the backlog critic;
  * the gradient sign on the synthetic ramp scene;
  * `p_lane` limits (blocked, far, empty lane);
  * critic input shape, zeros = absent, and an old critic fails to load;
  * the trainer refuses a mismatched critic and feeds `entry_backlog` [B,2] in every state, including
    the terminal one.
* **Cross-commit bit-identity, switch off** (`_out/srcgrad/golden.py`, local): two `update()`s with entries
  and counterfactual rows on the tiny WM, run against the original `7213d9c` src tree and against this
  branch. All metrics are identical (`==`) and the post-update actor / group encoder / critic weights are
  `torch.equal`.
* `OMP_NUM_THREADS=1 pytest`:
  * test_c4_srcgrad, test_c4_dreamer, test_c4_srcwait_imagination, test_c4_srcwait_mappo, test_c4_policy2,
    test_c4_cfmem, test_c4_futures and test_import_graph: **274 passed**;
  * test_c4_srcgrad, core, branching, loop, belief, compare, rsu, rsu_field, mappo and deploy:
    **137 passed**.
* ruff clean on the changed files.

## 5. Smoke and gradient measurement (CPU, real C4 WM `c4_wm_main/wm_s_best.pt`, TRAIN starts)

**CLI smoke** (`train_cli --config configs/c4_policy_final.toml --wm-ckpt ... --smoke`, so batch 2, H 4,
counterfactual rows and lane head on; `_out/srcgrad/smoke_final`): completed with 2 updates + 2
validations and finite metrics.

| | update 1 | update 2 |
|---|---|---|
| source_wait | 1.94 | 0.06 |
| delay | 27.5 | 20.6 |
| actor grad norm | 0.75 | 0.17 |
| critic grad norm | 1.68 | 1.50 |
| update time | 73 s | 72 s |

Validation `total_wait_vs_native` was +7.5 / +5.9 (untrained actor).

**Off vs on on the SAME start batch** (`_out/srcgrad/diag.py`; batch 4, all-CAV rows, no lane head;
SW = source wait per control step, D = delay per control step):

| H | seed | SW (off = on) | D (off = on) | grad norm sum SW, off -> on | grad norm sum delay | objective grad norm off -> on | objective grad rel. change / cos |
|---|---|---|---|---|---|---|---|
| 10 s | 0 | 4.05 | 25.18 | 0 -> 0.079 | 113.2 | 0.5656 -> 0.5634 | not computed |
| 10 s | 1 (policy noise) | 3.96 | 26.32 | 0 -> 0.277 | 137.6 | 0.5988 -> 0.6077 | 1.6 % / ~1.0 |
| 30 s | 0 | 5.85 | 36.17 | 0 -> 1.81 | 515.4 | 1.2027 -> 1.2024 | 0.10 % / ~1.0 |

Values are bit-identical off vs on (straight-through).
* "Objective grad rel. change" = `||g_on - g_off|| / ||g_off||` of the actor objective `-mean(returns)`.
* The cosine prints as 1.00001 to 1.00002, i.e. float32 rounding.

**Where the gradient comes from.** The surrogate gradient is **local**: it is nonzero only while a
vehicle sits within about 0-25 m of a lane start. In these starts:
* the imagined backlog is almost all at the **mainline** entrance: `p_block` mean 0.81-0.91, demand about
  4-6 vehicles/step. Recorded start states have near-stationary vehicles about 1 m past the mainline lane
  starts;
* the ramp entrance is mostly free: `p_block` 0.13-0.33, demand only `rate*dt` (about 0.05-0.07);
* about 25-44 % of (row, step) pairs have `p_block` in (0.02, 0.98), i.e. a usable slope.

So, summed over the whole batch, the surrogate's share is small:
* 0.07-0.35 % of the delay-sum gradient norm;
* 0.1-1.6 % change of the actor-objective gradient.

But per controlled vehicle at an entrance it dominates:
* surrogate: `dt * D * dp/dgap` ≈ 0.5 x 4 x 0.08 ≈ 0.16 reward units per metre of gap;
* the CAV's own delay sensitivity: `dt / v_max` ≈ 0.017 per m/s per step.

This is the intended structure: crawling entrant CAVs at an entrance now pay for the queue they hold
back, and other CAVs are unaffected.

## Limitations / suspicious

* **Within one 30 s window the actor can mostly reach the mainline queue only through entrant CAVs**
  (inserted at the lane start and controlled from their first step). A queue caused by spill-back from
  the merge, 2.6 km downstream, is out of reach of both the gradient and the horizon. The critic backlog
  input is the only channel for those.
* **The imagined mainline backlog is driven by recorded start states.** These have vehicles almost
  stopped right at the mainline lane start. Whether that matches SUMO's insertion is a WM entry-model
  question (SRCWAIT_NOTES.md: the insertion rules differ).
* **The aggregate gradient change is small on random TRAIN starts** (0.1-1.6 % of the objective
  gradient), because few CAVs sit at an entrance in a given window. Whether this, together with the
  backlog critic, is enough to remove the SUMO blocking behaviour must be checked in the next SUMO
  head-to-head.
  * If it is not enough, a gradient-only multiplier on `q_sur` would strengthen it without changing the
    reward value. Not added.
* The insertion itself stays non-differentiable: new vehicles are written under `no_grad`. The
  surrogate prices the blocking probability; it does not model the downstream effect of an extra
  inserted vehicle.
