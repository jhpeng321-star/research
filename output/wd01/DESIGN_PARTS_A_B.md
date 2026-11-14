# WD-01(d) — World-model design: Part A (short-term per-vehicle rollout) and Part B (long-term delay estimator)

Design document, 2026-09-25. It implements the owner-approved two-part
design against `PREDICT_CONTRACT_v1.md` (`wd01_predict/1`). It contains no
results. Every number marked *(to measure)* is set by a WD experiment, not
assumed.

## 0. Constraints this design must meet

- Ideal sensing: observed vehicles and CAV controller state are exact. Hidden
  vehicles come from the reconstructor's sampled sets.
- Actions are per CAV: native / damping (CIRCLES) / PI external-U. The wall
  is a group action with a lock from assembly through handback. Several walls
  are allowed, and safety aborts are allowed, including during assembly.
- Uses: the planner's safety gate, the measured CAV-to-CAV interaction check,
  wall-assembly feasibility, and delay versus native over 10 min.
- The output is judged in this priority order: (1) planner decision quality
  versus the perfect simulator, (2) plan ranking versus native, (3) safety
  recall, (4) calibration of the M×K spread, (5) predictions per second on
  GPU.
- Code must run on CPU and CUDA without changes: `device` is resolved once
  from the environment or config; all tensors are created with
  `device=`; there are no `.cuda()` calls; checkpoints are saved with
  `map_location`-safe `state_dict` only (so `weights_only=True` works, unlike
  `vehicle_dynamics.pt`).
- Every trained weight file is pinned by SHA-256 in its run manifest and
  copied to `D:/research-dense-backups/wd<NN>/` before it is used (MIG-04 rule).

## 1. What the existing models tell us

| artifact | status | lesson |
| --- | --- | --- |
| `worldmodel_v3_vehicle/vehicle_dynamics.pt` (MLP 12→128→128→1, 1 s dv) | survives, sha256 `ded83907…a8f5` | Per-vehicle dv is learnable (holdout MAE 0.137 vs 0.329). It was trained on 60–600 s windows, which are identical across arms, so it is **action-blind**. Its 60 s open-loop rollouts from hybrid samples reach **42 m/s** (WM-30). |
| `worldmodel_v3_postswitch` (arm-aware MLP, 0.1228 vs 0.1364) | weights and tracks deleted in MIG-04; the post-600 s raw data is gone | Controller identity adds information beyond local state. That motivates action conditioning. **Owner Q1 = (b), 2026-09-25: reproduction waived.** The result stands as historical evidence only. |
| `worldmodel_dynamics_v1/dynamics_ensemble.pt` (5 members) | survives, sha256 `d017d17e…a934` | Unconstrained ensemble rollouts blow up after 100 s (RMSE 1.2e3 at 200 s). Member disagreement grew with the error, so the spread is informative, but nothing bounded the state. |

Design consequences:
1. Rollouts need **hard physical constraints**, not only a learned step.
2. Action conditioning needs **post-switch branch data** from the WM-36
   campaign.
3. Training must use a **multi-step loss** so the model sees its own drift.

## 2. Part A — action-conditioned per-vehicle rollout

### 2.1 State and step

- The entities are every vehicle on the section: observed, hidden (from one
  reconstructor sample) and arriving (from the arrival draw of `StateSample`).
- Per-vehicle state: `lane_id` (WM-30 physical lanes 0/1/2), `s`, `v`,
  `a_prev`, `is_cav`, `vehicle_class`, and a latent driver embedding `z` (§2.4).
- Step Δt = **0.5 s** (the SUMO physics step). Output is sub-sampled to 1 s
  for the contract's trajectories. TTC and hard-braking metrics are computed
  at 0.5 s. The 1 s step of the v3 model is too coarse for TTC < 1.5 s events.
- Horizon = `part_a_horizon_s`, provisional 60 s, set by WD-01(c) (§6).

### 2.2 Per-vehicle inputs

For each vehicle, relative to its current lane and s:
- Ego: `v`, `a_prev`, distance to merge end (accel lane), lane one-hot, speed
  limit.
- Neighbour slots, 6 in total: the leader and follower in the own lane, the
  left lane and the right lane. Each slot holds gap, relative speed, `is_cav`
  and a presence flag, capped at 250 m.
- Hidden-field context: expected count and speed of the next 200 m ahead in
  each lane, from the reconstructor field layer. This is used only where no
  sampled vehicle occupies the bins.
- **CAV action conditioning.** For each CAV:
  - action one-hot (native / damping / pi-external-u / wall-member);
  - wall phase one-hot (assemble / hold / release);
  - wall speed, time in phase, own lane index within the wall, and gap and
    relative position to its wall partner(s);
  - controller internal state where the controller has one (the CIRCLES
    damping target-speed memory, the PI external-U running target). Ideal
    sensing makes these exact. **Request: add them to the controller-state
    block (§7).**
- Human vehicles get the same slots with `is_cav=0`. The action slots are
  zero, but the neighbours' `is_cav` and action flags let humans react to
  controlled CAVs nearby.

### 2.3 Architecture (GPU-batched)

- Model: a shared per-vehicle encoder (MLP, 256×3) over ego + neighbour slots
  + action features. It has two heads:
  - Longitudinal head: a **residual on IDM**,
    `a = a_IDM(v, gap, Δv; θ_z) + Δa_θ`. `θ_z` are IDM parameters decoded from
    the driver latent `z`, bounded to the ranges of the three SUMO vTypes
    (aggressive / normal / cautious: τ 0.95–1.5 s, accel 2.0–3.0 m/s², minGap
    1.5–3.5 m). The prior carries the car-following physics. The residual
    learns controller effects, lane-change interactions and merge behaviour.
  - Lateral head: a 3-way categorical (stay / left / right) per step, masked
    by lane topology. The accel lane may only change into main_right, and the
    mandatory merge zone is s ∈ [378.66, 454.10] m.
- Optional **CAV emulator path (ablation A2)**. The damping, PI and wall
  control laws are known code in `traffic_shepherd` (`damping.py`,
  `pi_session.py`, the wall session). A2 re-implements the commanded speed of
  each law as batched torch functions. They are tested for equivalence
  against the Python controllers on logged frames. The learned residual then
  models only how SUMO executes the command and how humans respond. A1 (fully
  learned action conditioning) and A2 are compared on the same data. **A2 is
  preferred if its emulators match the reference within tolerance.** It
  generalises to unseen action combinations (walls, multiple walls) with far
  less data.
- **Hard constraints inside the integrator.** These are always on, and they
  are what prevents the 42 m/s failure:
  - `a ∈ [−9, a_max(z)]`;
  - `v' = clip(v + aΔt, 0, v_lane_max × speedFactor_max)`;
  - `s' = s + (v + v')/2 · Δt`;
  - a lane change is accepted only if the target-lane gaps are ≥ minGap
    (otherwise the vehicle stays);
  - no-pass-through ordering: if `s'` would overtake the in-lane leader, the
    step is flagged `overlap` and counted as a collision event, not silently
    corrected. Collision = overlap after the integrator. This is what the
    contract's `collision` reports.
- Neighbour search: per-lane sort by `s` with `torch.sort`, then index shifts.
  That is O(N log N) per step and fully batched over the flattened
  `(plan, member, sample)` batch. N ≈ 100–300 vehicles.
- Boundary handling: vehicles leave at the downstream exit (their trajectory
  is NaN afterwards). Arrivals from the arrival draw are inserted when the
  entry gap allows; otherwise they queue, and **queued time is entry waiting
  delay**.
- Stochasticity: each member is deterministic given `(z, noise seed)`. The
  noise seed is common across plans and native for the same (m, k). This is
  common random numbers, which keeps the paired comparison low-variance.

### 2.4 Driver latent z

Humans are drawn from three IDM vTypes with lane-change personalities. The
vType is a hidden per-vehicle variable.
- Observed vehicles: `z` is inferred by a small encoder from the vehicle's
  observed history (ideal sensing gives the last 10–20 s).
- Hidden or arriving vehicles: `z` is sampled from the prior (10/70/20 mix),
  per sample k.

The v3 model's lack of this latent is one candidate cause of its rollout
error.

### 2.5 Losses

- **Multi-step rollout loss** (the main loss):
  - unroll the model from true branch states for H_train steps with
    gradients (truncated BPTT);
  - curriculum H_train = 10 → 40 → 120 steps (5 → 20 → 60 s);
  - loss = Huber(position) + Huber(speed) + CE(lane change) at every step,
    matched per vehicle id;
  - vehicles that the model sends off a different lane stay matched by id.
- One-step auxiliary loss (weight 0.1) for stability early in training.
- **Safety-event auxiliary heads** on the shared encoder: per pair and per
  step, P(TTC < 3 s within 2 s) and P(a < −3 within 2 s), trained with BCE
  and class weighting. They are used in two ways:
  1. to **up-weight** rollout loss on windows near events;
  2. as a conservative **OR-gate** in the safety output: an event counts if
     the rollout shows it OR the head fires above a threshold set on
     validation for recall ≥ 0.95 *(to measure)*. Priority (3) says missing
     unsafe outcomes is the worst error.
- Wall-assembly auxiliary head: P(assembled by t) and assembly time, trained
  from wall branches.

### 2.6 Ensembles

- M = 5 members *(test 3/5/8)*. Each member has a different init seed and a
  **bootstrap over root scenarios**, not over branches. Branches that share a
  root saved state are correlated, and resampling them would understate
  spread.
- The total spread per prediction comes from M × K: members times the
  reconstructor hidden-vehicle samples and arrival draws.
- Calibration: on held-out roots, check that the M×K quantiles cover the
  realised delay difference and safety metrics (PIT histogram, coverage at
  50/80/90%). If needed, fit one temperature per output on a separate
  calibration split. The spread is recalibrated, never the mean.

## 3. Part B — long-term delay estimator

### 3.1 Target

For a branch point at t0 with plan P and its native sibling, Part B
estimates the remaining delay over `[t0 + A, t0 + 600]`, where A = Part A
horizon:
- `R_native`: the native branch's remaining delay;
- `ΔR = R_plan − R_native`.

The two heads are trained jointly. ΔR is learned directly because the
difference is small relative to the totals, and a paired target cancels most
of the demand variance.

What the contract then assembles:
- `delay_vs_native = (delay_A,plan − delay_A,native) + ΔR`
- `plan_delay = delay_A,plan + R_native + ΔR`
- `native_delay = delay_A,native + R_native`

Plan semantics beyond the Part A window: v1 assumes the plan is **held**
constant to the horizon. Walls follow `hold_s` / `release_at_s`, then
dissolve. This is a documented approximation, because the planner actually
re-decides continuously. A later version may add a "planner-continuation"
target once planner-driven branches exist.

### 3.2 Inputs

1. End-of-window state at t0 + A: a field aggregate on the physical lanes (3
   lanes × 10 m bins with count, mean speed, speed std), plus CAV tokens
   (s, lane, v, action, wall id/phase/speed, time in phase), plus queue
   lengths at both entrances.
2. Plan descriptor: per-CAV actions as tokens (above) and wall tokens
   (members' s, speed, remaining hold).
3. Future demand: the arrival draw summarised as per-minute counts per
   entrance over the remaining window, plus CAV share.
4. Context: time since the last native branch point, the demand regime, and
   Part A summary stats (delay so far, TTC exposure).

### 3.3 Architecture

- A 1D conv encoder over each lane's bin sequence feeds a transformer, with
  CAV and wall tokens added as extra tokens.
- The output is pooled into two heteroscedastic heads (mean + log σ) for
  `R_native` and `ΔR`.
- Size about 1–3 M parameters. Batched over plans × M × K, device-agnostic.
- Ensemble: M_B = 5, bootstrapped over roots, paired with the Part A members
  (member m of A feeds member m of B).

### 3.4 Train/test shift

At inference, Part B sees **Part A's predicted** end state, not the true
one. Training runs in two stages:
- (i) train on true end states from simulation;
- (ii) fine-tune on Part A–rolled end states (Part A frozen) for the same
  branches, with targets from the true continuation.

Report B's error on both input types.

### 3.5 Losses

- Gaussian NLL on `R_native` and `ΔR`.
- A pairwise ranking loss across plans from the same branch point (hinge on
  the sign of ΔR_i − ΔR_j), because evaluation priority (2) is ranking.
- ΔR errors near 0 are weighted up, since decision flips happen there.

## 4. Training-data requirements for the WM-36 campaign

Every branch shares a root saved state, and **every branch point has a
native branch** with the same arrival seed (common random numbers).

| branch type | Part A use | Part B use | logging needed |
| --- | --- | --- | --- |
| native | human dynamics, native reference | R_native target | all below |
| random per-CAV fixed actions (native/damping/PI mix) | action conditioning, interactions between neighbouring CAVs | ΔR targets | per-CAV action per step |
| random per-CAV switching every 20–120 s | switch transients (controller warm-up, PI target memory) | ΔR under changing plans (for v2) | switch times, controller internal state |
| fleet PI | high-penetration controlled flow | ΔR tail | — |
| random walls (1–2 walls, random members/speed/hold), incl. safety aborts | wall assembly, hold and release dynamics; abort behaviour | wall ΔR | wall id, members, phase transitions with times, abort reason |
| planner-driven (later) | on-policy states | on-policy ΔR | planner plan per tick |

Per-branch logging at 0.5 s for every vehicle:
- id, physical lane and s (or edge/lane/pos plus the WM-30 mapping), v, a,
  lane-change events, vType;
- for CAVs: commanded speed, owner/action, controller internal state, wall
  membership and phase.

Per-branch outcome record: delay decomposition (time on section,
free-flow time, entry waiting at mainline and ramp), TTC < 3 s exposure,
hard-brake seconds, min TTC, collisions, and wall events.

Arrivals:
- log the realised arrival list per branch;
- record root id, branch id, branch point t0 and the seed.

The **saved-state id** must be recorded so any branch can be regenerated.

Size (initial estimate, final value from learning curves *(to measure)*):
- Part A: ≥ 1,500 branches × 120 s, spread across ≥ 150 roots and all
  demand levels (2000–2700 veh/h), with ≥ 20% wall branches.
- Part B: ≥ 3,000 (plan, native) pairs of 600 s, spread across ≥ 150 roots.

These estimates are for the owner's campaign budget discussion. WD will
report learning curves at 25/50/100% of the data so the budget can be
revised.

Split: by **root**, never by branch: train / calibration / holdout
= 70 / 10 / 20 of roots. Holdout roots are also used for the planner
comparison (§5).

## 5. Evaluation (maps to the owner's priority list)

1. **Decision regret.** On holdout roots, run the planner at the same ticks
   with the learned `predict` and with the perfect-simulator `predict`.
   Score the learned-planner's choice with the simulator. Report extra delay
   (veh·s per 10 min) and the disagreement rate.
2. **Ranking.** Spearman correlation and top-1 agreement of
   `delay_vs_native` across candidate plans per tick, plus the sign accuracy
   versus native.
3. **Safety recall.** Recall and precision of {collision, min TTC < 1.5 s,
   increase in TTC < 3 s exposure, increase in hard braking} versus the
   simulator, per plan type. The target is recall ≥ 0.95 at the gating
   thresholds *(to measure)*.
4. **Calibration.** M×K coverage at 50/80/90%, PIT, and the rate at which the
   ≥ 80% agreement rule holds but the true sign disagrees.
5. **Speed.** `predict_many` throughput in (plan × M × K) per second, on CPU
   and on the RTX 3060 when it is attached. Also report wall-clock time for a
   reference planner tick: 50 plans × M 5 × K 16, Part A 60 s at 0.5 s, about
   200 vehicles.

## 6. WD-01(c) rollout-drift measurement (feeds `part_a_horizon_s`)

This runs on the surviving v3 per-vehicle model (native, action-blind) and
its held-out tracks. It is zero simulation.
- Rollout from true states at start ticks {120, 180, …, 480} of the 21
  holdout runs, to horizons 30 / 60 / 90 / 120 s. Rollouts are cut where the
  600 s window ends.
- Criterion 1, safety-event recall: TTC < 3 s and hard-brake events in
  predicted versus true trajectories, matched by vehicle id and time
  tolerance.
- Criterion 2, no runaway: fraction of vehicles with v > 1.2 × the lane limit
  or v < 0, and overlap (pass-through) events.
- Criterion 3, speed: per-vehicle speed RMSE and position error versus
  horizon, against the constant-speed baseline.
- Each criterion is reported both **with and without** the hard-constraint
  integrator of §2.3. This shows how much of the failure comes from the
  integrator and how much from the learned step.
- Recommendation: `part_a_horizon_s` = the longest horizon that passes all
  three. Pass thresholds are registered before scoring.
- Caveat: this is the native baseline only. The horizon is re-measured on the
  first action-conditioned Part A (the WD-02+ campaign data).

## 7. Open requests

- Reconstructor (integrator): add a controller-state block to the hybrid
  state. Per CAV: action, wall id/phase/time in phase/speed, locked, and
  controller internal state (damping target-speed memory, PI running target).
- WM-36 campaign: the logging fields in §4 and a native sibling at every
  branch point with a common arrival seed.
- Owner Q1 decided 2026-09-25: postswitch reproduction waived, no SUMO
  re-runs. The action-conditioned Part A is built from WM-36 campaign
  branches.
