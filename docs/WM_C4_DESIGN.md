# WM-S C4 — learned multi-CAV control inside the world model (design draft, 2026-10-12)

Status: DRAFT for owner review. No training, SUMO or GPU is authorised by this document.
Supersedes the Phase-D sketch in [WM_S2_ROADMAP.md](WM_S2_ROADMAP.md) ("C4") and moves C4 forward; the R5 fixes remain valid.

## 1. Claim (owner + professor, 2026-10-12)
A controller learned *inside* the physics-informed world model (Dreamer-style) works across CAV penetration rates,
adapts to each CAV's situation, discovers coordinated multi-CAV moves (wall-like), and can **check** that an action
is beneficial before taking it. Model-free RL controllers, by contrast, are trained for one traffic mix, break when the
penetration changes, and judge actions only from noisy returns.

Why it can hold:
- **Penetration-agnostic by construction.** One shared per-CAV policy + per-car world model: 5 % and 40 % differ only in
  the number of CAVs.
- **Matched counterfactuals.** The WM replays the same state under action A vs B (paired, noise cancels).
- **Analytic per-CAV credit.** The rollout is differentiable w.r.t. CAV accelerations (direct `action_command` path,
  `src/wms/model/cav.py:17-73`; HDV IDM reads leader gap/speed), so the gradient of the team reward gives each CAV its
  own credit — the hard part of multi-agent RL.
- **Short imagination.** 10–30 s imagined rollouts (R5b: 1.7 / 4.5 m median) + a learned value for the long tail; the
  120 s error is no longer blocking.

## 2. Components
### 2.1 Policy (actor) — direct acceleration, owner choice
- Shared network, applied per CAV every 1 s; output acceleration in [-3, 1.5] m/s² (tanh-Gaussian).
- Inputs (CAV-observable only, [OBSERVATION_DECISION_CONTRACT.md](OBSERVATION_DECISION_CONTRACT.md)): own speed/accel/
  lane/position/distance-to-merge, same-lane front/rear gap and relative speed within 100 m (`SumoLocalSensor`), plus
  reports from other CAVs relayed by the RSU (with receipt age / missingness).
- **CAV-to-CAV attention** over those reports (same mechanism as the HDV residual transformer).
- **RSU group signal:** a short learned vector computed by the RSU from CAV reports **and its own sensor field**
  (50 m cells over the on-ramp, merge, 800 m upstream, 150 m downstream: density, speed, queue, CAV share; owner
  2026-10-12) and broadcast to every CAV in a zone. The same RSU field feeds MAPPO if it is trained later.
  Lets synchronised moves (a wall = upstream CAVs across lanes slow together) be expressed without hand-coding them.
- **Physics safety layer:** clamp to an IDM-style safe-deceleration / safe-gap bound before execution (also in SUMO).
- No true density, penetration or driver type as input.

### 2.2 Critic
Centralised, training-only (privileged full WM state is allowed for the critic, not the actor). Predicts the team
λ-return; discount ≈ 0.99 per 1 s control step (~100 s effective) to bridge the ~5 min benefit horizon.

### 2.3 Reward (team, computed in torch on WM states — new code; existing metrics are numpy only)
- **Delay / throughput:** vehicle progress and served flow in the zone, including ramp / source waiting
  (guard against lane-blocking reward hacking).
- **Smoothness:** HDV and CAV ∑a², soft hard-brake penalty relu(-a-3)².
- **Fairness (owner):** no HDV group sacrificed — penalty on the worst-off group (ramp vs mainline, cars behind CAV
  formations) via the 90th/95th-percentile per-vehicle delay (soft CVaR).
- **Safety as a constraint:** TTC < 3 s exposure and collisions handled by a Lagrange multiplier + the safety layer,
  not traded off.

### 2.4 Imagination training (Dreamer loop)
1. Start states: TRAIN windows only (no sealed/CAL/HOLDOUT/chunk-5/INC1/J4B), penetrations 5/10/20/40 %.
2. HDV driver settings come from the WM-S encoder (learned, never given types).
3. Roll out 30 s (60 × 0.5 s) with the policy driving CAVs through `action_command`; λ-returns with critic bootstrap.
4. Actor loss = −λ-return via analytic gradients through the WM (+ entropy); existing grad damping
   (`grad_decay_per_step`) and per-stage lr carry over.
5. **Anti-exploitation:** uncertainty penalty (small residual ensemble or driver-setting sampling disagreement),
   action-rate limit, the safety layer, and the SUMO loop below.
- **No imitation of existing controllers (owner 2026-10-12):** the policy starts from scratch and must invent its own
  control. Exploration help comes only from the world model (gradients, many cheap imagined rollouts, varied start
  states), not from controller demonstrations.

### 2.5 World model for C4
- **No designed controllers in WM training (owner 2026-10-12):** the C4 world model learns HDV responses from native
  driving plus *random* direct-output CAV actions (pack_switching_train direct-mix, pack_dspeed sampled target-speed
  trajectories). CIRCLES damping, PI and the wall law appear only as evaluation baselines. Open check: whether the
  switching arms (which mix in damping/PI segments) are excluded too — default: excluded.
- C1 already shows such models predict HDV response to unseen controllers (PI, wall) zero-shot.
- Refreshed each loop with SUMO data from the learned policy.

### 2.6 Deployment — Option B (owner choice): policy + world-model check
Every few seconds the RSU imagines 10–30 s for a few candidates and acts on the best:
the policy's plan, policy with alternative group signals, stochastic policy samples and a hold-speed direct
fallback (owner 2026-10-12 late: direct-output candidates only; native, CIRCLES, PI and the wall law are
whole-episode baselines, never check candidates) — each under several HDV driver-setting guesses. Keep the
policy's plan unless an alternative wins clearly (paired margin). The policy outputs per-CAV acceleration AND
a stay/left/right lane decision and is trained on cooperative difference rewards (POLICY2_NOTES.md).
- **State for the check (owner question 2026-10-12).** The RSU only has CAV reports; imagination needs a state.
  Three levels, all reported:
  1. **Oracle check** — true SUMO state (upper bound, labelled as such).
  RSU sensing (owner 2026-10-12): the RSU has its own sensors (on-ramp, merge area, ~800 m upstream, ~150 m
  downstream, as in the earlier reconstructor setup) in addition to CAV reports. The old learned reconstructor is not
  reused directly (checkpoints lost in MIG-04, cell-field output, failed fresh-world queue detection); its sensing
  zones and its metric (WM forecast gap from reconstructed vs true state) are reused.
  2. **Belief state (main)** — Dreamer-like state built from RSU sensors + CAV observations only: tracks of every vehicle a CAV has
     detected, associated by position/lane/speed (no HDV ids), propagated by WM-S between sightings with growing
     uncertainty, plus histories for the driver encoder (`wms.control.belief`). Vehicles never detected do not exist.
  3. **Sensed-only** — CAVs + currently detected HDVs (ablation).
  Expected: the check's value grows with penetration (more coverage) — reported per penetration.

### 2.6b What each method sees (policy vs MAPPO)
| | Actor at deployment | Critic / training signal | State needed |
|---|---|---|---|
| C4 policy alone | CAV own sensing + RSU-relayed CAV reports + group signal | privileged imagined state (training only) | recorded full states only as training starts |
| C4 policy + check | same actor | — | belief state built from CAV reports (2.6) |
| MAPPO (main) | identical actor class and inputs | centralised critic on privileged SUMO state (CTDE, strongest baseline) | none at deployment |
| MAPPO (previous style) | identical | centralised critic on joint CAV + RSU observations only | none |
At deployment both policies see exactly the same inputs; only the world-model check adds a belief state.

### 2.7 SUMO loop
- Learned policy runs in SUMO through `CommandOwnership` (`wall_commands={vid: accel}` → direct target,
  `traffic_shepherd/ownership.py:39-143`); template `output/campaign_stage1_v3/trajectory_arm.py`.
- 1 s control / 0.5 s physics / 100 m sensing / accel [-3, 1.5] (`traffic_shepherd/config.py:43-44`).
- Each loop: collect policy data (registered budget) → WM refresh → policy fine-tune. 2–3 loops.

### 2.8 Discovery (owner question: can the WM learn wall-like moves on its own?)
With 2.4 and 2.5 every run is a discovery run: neither the world model nor the policy ever sees a designed controller,
and there is no wall action or wall bonus. Any coordinated formation that appears was found by the policy.
- Prior evidence that coordination can pay at the merge (exposed development data): pooled same-contract executed walls
  4 beneficial / 19 (OPEN_QUESTIONS OQ-1); WL-04 envelope-gated wall-v2 vs no-wall −30.2 s/veh (−18.2 %) at 920002/m2700
  (harness world without damping); no wall qualified over frozen CIRCLES damping, and the old predictors could not
  select the beneficial cases (OE-08). So walls help sometimes; *when* is the open problem — which the world-model
  check (2.6) is designed to answer.
- Give coordination a fair chance: include high-demand merge states and 20–40 % penetration in training starts.

## 3. Gates and evaluation
- **Gate 1 (first, CPU, existing data): action fidelity.** On paired arms from the same saved state, does WM-S rank
  native / damping / PI / wall like SUMO does at 10/30/60 s (pairwise sign agreement, Kendall τ, regret of the WM's
  pick)? If the WM cannot rank actions, the method fails — check before building the policy.
- **Final comparison (fresh SUMO development seeds, demands 2400/2600/2700 veh/h):**
  - Trained penetrations 5/10/20/40 %; **unseen** 15/30/60 % (and 2 %).
  - Controllers (owner 2026-10-12 late: MAPPO training deferred): native, CIRCLES damping, PI (comparators); wall law
    optional; learned policy alone; learned policy + WM check (oracle and belief state). MAPPO code is kept ready
    (same actor/inputs/reward) but not trained for the Friday showcase.
  - Metrics: delay incl. ramp/source waiting, throughput, braking, worst-group delay (fairness), TTC exposure,
    collisions, teleports.
- **Coordination evidence:** CAV formation metrics (side-by-side across lanes, synchronised slowdowns before the merge,
  gap hand-offs) vs the hand-designed wall; ablation without CAV-to-CAV attention and group signal; expected shift from
  individual damping at low penetration to group moves at high penetration.
- **WM honesty:** WM-predicted vs SUMO-measured gain for every evaluated controller.

## 4. Plan and estimate
| Step | Work | Estimate |
|---|---|---|
| 0 | This design + registration | ½ day |
| 1 | Gate 1 action fidelity (CPU) | 1 day |
| 2 | Torch reward (delay, smoothness, fairness, safety) + differentiable CAV action input checks | 1–2 days |
| 3 | Policy (CAV attention + group signal), critic, safety layer, imagination loop | 4 days + GPU runs |
| 4 | C4 world model on native + random direct-output data (one GPU run) | 1 day setup + ~6–8 h GPU |
| 4b | MAPPO baseline in SUMO (same observations/actions/reward; port archived MAPPO/PPO code) | 3–4 days + CPU days on the rented box |
| 5 | SUMO runner for the learned policy + RSU world-model check (Option B) | 2–3 days |
| 6 | Collect → refresh → fine-tune loops (2–3) | ~1 day each |
| 7 | Final comparison + coordination metrics + ablations | 2 days |

Implementation ≈ 2½–3 weeks (MAPPO in parallel with steps 3–5); complete results ≈ 4–5 weeks, dominated by MAPPO's
SUMO compute. Archived PPO/MAPPO code: `archive/circles_era/traffic_shepherd/ppo_*.py`,
`archive/pre_circles/code/env/mappo.py`.

## 5. Known constraints / risks
- Lane choice is an argmax (no gradient through lane changes); CAV-acceleration gradients still reach HDVs via IDM.
- Clamps zero gradients at action bounds (tanh-squash before the clamp).
- `cav_drive="closed_loop"` (frozen CIRCLES/PI on predicted states) is `no_grad`; fine for baselines in imagination,
  not for the policy.
- Policy exploitation of WM errors on unusual CAV behaviour → uncertainty penalty + SUMO loop.
- The original switching/direct-mix generator code lived on the box, not in this repo.
- No training without owner discussion; box off when not training.

## 6. Owner updates 2026-10-12 late (supersede earlier text where they conflict)
- **HDV lanes:** `lane_rule = "learned_discretionary"` — physics proposes only the forced ramp merge; safe adjacent
  discretionary changes are open at a neutral prior for the learned lane correction (MOBIL removed).
- **Policy actions:** per CAV, direct acceleration + lane change (stay/left/right, safety-masked); no designed controller
  anywhere in learning; the RSU check uses only direct-output candidates (policy, group-signal variants, policy samples,
  hold-speed). CIRCLES/PI/native are whole-run baselines only.
- **Dreamer loop:** world model and policy learn together in short cycles (collect with the current policy -> WM refresh
  -> policy updates in imagination); the first WM training is the warm-up.
- **Cooperation as the main training signal:** per start state (common random numbers) B = all native, D_i (only CAV i
  acts), D_G (close group acts), D_{G\i} (flip-one), D_all. Cooperation gain C_G = D_G - sum D_i; marginal-in-team
  M_i = D_G - D_{G\i}. Actor objective w_all D_all + w_group D_G + w_ind sum D_i + w_coop C_G + w_team sum M_i with the
  anti-gaming guard w_coop <= w_ind (C_G alone would reward sabotaging solo behaviour); curriculum shifts from individual
  to group/cooperation weight; group-rich start-state sampling; reports carry other CAVs' last command + lane intent.
- **Paper focus (novelty check vs Liu et al. 2026 and Vinitsky et al.):** cooperation gain and emergent coordination,
  run-time verification, extrapolation beyond the trained penetration range (60 %), sample efficiency vs a
  penetration-randomised ("universal") MAPPO at equal SUMO budget (needed for the paper, not for Friday).
