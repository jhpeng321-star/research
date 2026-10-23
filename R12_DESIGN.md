# R12_DESIGN - round 12: calibrated car-following base + the round-11 courtesy head (fix B dropped)

S-R12, 2026-10-24. Branch `c4/v10r12` (from `c4/v10r11` c476cad4, S-BASEDIAG final 116824b7 merged). Evidence:
BASE_FOLLOWER_DIAG.md (S-BASEDIAG), E:/agent-work/c4-v10r10/R10_DIAG.md (S-FIXBDIAG), RANK_DIAG.md, R11_DESIGN.md.
Offline numbers: R12_REVIEW_PACKET.md and docs/r12_*.json. Nothing launched; gpuR/cpuN untouched (one read-only copy
of the round-9 final checkpoint, sha256 af9bc643..., to `tmp_r12/ck/`).

## 0. Problem
- The base over-predicts the lane-1 target follower before a merge (+1.34 m/s^2 in the last 3 s) and slow cars
  everywhere (+1.52 on slow-start rows). Root cause (BASE_FOLLOWER_DIAG section 0): the **IDM core**.
  - (A) The v10 line inherits the v5 legacy encoder. Its tanh-bounded deltas around the defaults floor every driver at
    a >= 1.78, b >= 2.87, and the encoder sits on that floor (p1 = p10 = 1.78). SUMO start-ups fit a = 1.1.
  - (B) The residual attention read-out is dead: tanh-saturated at -1.5 on every row, i.e. a constant offset with zero
    gradient (R10_PREP).
  - (C) SUMO's target follower HOLDS while a ramp car waits; the base has no term for it. The round-11 contrast
    courtesy head learns it, but only on a calibrated base: with an unbiased base, matched controls carry no generic
    error, so the contrast target equals the absolute one.
  - (D) The teacher-forced replay drops junction-transit vehicles from the replay STATE (20 % of last-3-s partner rows
    have a hidden lane-1 vehicle between follower and replay leader).
- Fix B (round 10) did not reach its aim. R10_DIAG: follower post-insertion min-accel gap closed by about 0-11 %,
  intervals include 0. The cut-in loss was 0.7 % of the total, and a gate-only hybrid reproduces its whole effect.

## 1. What changes (every new switch defaults to the round-11 behaviour; tests/test_r12_base.py)
| part | switch | effect |
|---|---|---|
| A | `model.driver_param_mode = "physical"` (V6ENC, existing) + `model.driver_param_init = "centre"` (new, init-only) | Full-range driver settings: a 0.3-4.0, b 0.8-6.0, T 0.4-3.0, s0 0.5-5.0, v0 0.6-1.4 x limit. Map: `lo + (hi-lo) sigmoid(centre + r raw)`, with a learned shared centre and shrinkage in logit space, so there is no observation-dependent floor. After loading round 9, `WmSTrainer._r12_reinit` zeroes the head rows emitting the five longitudinal means (`driver_encoder.head.2` rows 0-4, weight and bias). Every driver therefore starts exactly at the centre, initialised from the physics defaults (V6ENC rule: a 2.49, b 4.2, T 1.135, s0 2.6, v0 = default free speed), and per-driver deviations are re-learned from prediction error. A legacy head's saturated raw outputs would otherwise map near the physical bounds. Lane-parameter rows and uncertainty rows are kept. |
| B | `model.residual_accel_reinit = true` (new, init-only) | Zeroes the delta row of `residual_net.accel[-1]`: tanh(0) = 0, live gradient. The log-sigma row is kept. The round-9 residual is not reused as a constant offset (BASE_FOLLOWER_DIAG 3.3: it is co-adapted to the high a). |
| C | round-11 courtesy v2 (contrast target, version-2 inputs), zero-init head | Unchanged code and invariants from round 11: exact 0 without a partner and on CAVs; `rollout_grad = false`; unbounded read-out, Huber 1.0. Zero init follows the R11_REVIEW C2 launch choice; round 9 has no head anyway. |
| D | `loss.recorded_replay_present = "filled"` (new) | The recorded-state replay (`one_step_replay`, the courtesy contrast matching features, the fix-B pairing) uses the filled window presence (`inp.present`, `junction_fix = ffill`) as its STATE. A junction-transit vehicle stays a neighbour/leader and keeps its recurrent state, history and driver encoding through the transit, as in a rollout. Supervision rows still come from `loss_valid` only (`_recorded`, unchanged). Telemetry: `follower_courtesy_replay_filled_extra_rows`. |
| fix B | `loss.cutin_reaction.weight = 0` | The term is not built (R10_DIAG). It stays available as a diagnostic: the post-run one-step checks read post-insertion follower rows. |
| gate | `gate_` added to `optim.freeze_prefixes` | R10_DIAG: the gate g_max fell in 87 % of round-10 steps, under a position-dominated gradient that was clipped on every step, and never converged. The cheapest way to reduce position error was to shrink the whole residual. With the read-out re-initialised, the residual must set its own magnitude inside the round-9 envelope (g_max about 0.39, tau about 70 steps). |

Not done (owner rules / evidence): no virtual-leader or hand-set yield term (BASE_FOLLOWER_DIAG 4-5: it over-brakes);
no clamp, floor, threshold or tolerance in the model; no driver labels, types or truth.

## 2. Trainable set and why (R10_DIAG point 4)
Round 12 retrains the base on purpose. Only the car-following path trains, and only what the diagnosis implicates:
| trainable | lr scale (stage lr 5e-5 / 3e-5) | why |
|---|---|---|
| `driver_encoder.head.*` | x10 (docs/r12_lr_rule.json) | per-driver settings (A); rows 0-4 restart at the centre |
| `driver_encoder.centre_logits` | x1000 (rule) | the population centre of the settings (A) |
| `residual_net.*` except `residual_net.lane.*` | x0.5 (round-8 recipe) | the re-initialised read-out (B) and its trunk, co-adapted to the old core |
| `temporal.*` | x5 (round 10/11) | the temporal residual is co-adapted to the old core (BASE_FOLLOWER_DIAG 2.1, 3.3) |
| `follower_courtesy.*` | x1000 (R11 zero-init rule) | (C) |

Frozen bitwise (verdict and micro-smoke check this):
- `merge_pivot` (round 9's fit) and `merge_core`.
- The two gate scalars.
- The driver-encoder trunk (`history_proj`, `query`, `interaction_proj`, `history_gru`), so the driver EMBEDDING
  that feeds `lane_v5` and the residual is unchanged.
- `driver_updater`, `lane_v5`, the lane gates, `history_proj`, `k_embedding` and the residual's lane read-out.

So the lane model, the merge law and every lane-side input stay those of round 9. 99 tensors, 0.74 M parameters.

## 3. Objective (rules for losses and weights)
- **Motion objective = round 9's effective config, unchanged** (pinned by `test_round12_effective_motion_objective_equals_round9`):
  - position/speed Huber;
  - courtesy-boost 3;
  - aggregate 20 (flow 0.5);
  - the long-horizon delay loss the owner requires: agg_delay 0.5 + paired 0.5, from 60 s, horizon 300 s;
  - paired_diff 1 / 10;
  - brake persistence 0.5;
  - lane event 10, lane flow 1, flip rate 1;
  - residual magnitude 0.05;
  - driver prior 0.05 (textbook);
  - merge_core TF 0.6 (merge parameters frozen, so diagnostic);
  - grad clip 1.0.

  The round-8 recipe trained the same motion terms. Round 12 changes no motion weight: the base is retrained against
  the objective it was built on.
- **Driver prior (provenance; owner decision flagged to main 2026-10-24).**
  - What it is: `src/wms/train/losses/driver_prior.py`, in every production config since v5 (10-14), round 9
    included. It pulls only the BATCH MEAN of the encoded t0 settings of observed HDV slots:
    `0.05 * sum_p ((mean - target)/target)^2`. Per-driver deviations are not penalised.
  - Its target: a 1.2, T 1.2, b 2.0, s0 2.0, v0 = limit. These are literature IDM highway values (Treiber & Kesting
    2013, Table 11.2), not vType or hetdata files.
  - The caveat: the owner's 10-14 textbook SUMO population (`traffic_shepherd/c4/population.py`) is centred on the same
    literature numbers. So the target equals the generator's population medians.
  - Why it matters: BASE_FOLLOWER_DIAG section 7 (one-step fit) shows a collapses to the 0.3 bound without it (median
    0.35). With it the encoder lands at a median 0.87 (p10 0.68), b 2.02. One-step error alone rewards a sluggish IDM.
  - Default: keep the existing prior. It is the round-9 base recipe; removing it would be a second change.
  - Owner alternatives at the gate: `R12_PRIOR=off` (weight 0; identifiability from the rollout position/speed/delay
    terms, which penalise sluggish cars unlike one-step error) and `R12_PRIOR=weak` (weight 0.01).
  - The offline check reports the per-driver spread of encoded a/b for each variant (packet section 3), so the reviewer
    can see that per-driver variation survives.
- **Courtesy weight 9.067, lr x1000:** the R11 zero-init rule values (docs/r11_weight_rule.json).
  - Weight 9.067 gives the head a gradient-norm budget equal to fix B's weighted term (0.737). It is a fixed reference
    and does not require fix B to train.
  - `rollout_grad = false`, so only this term reaches the head, and Adam is per-parameter scale invariant. The weight
    therefore acts only through the global clip and the logged total.
  - The R11 lr rule was measured on the old base; the base-dependence is a stated risk (section 7).
- **Clip 1.0 kept.** In round 10 every step was clipped (norm 3-79). With Adam this amounts to Adam on the
  unit-normalised gradient: per-parameter step sizes stay about lr, and spikes are damped. Changing the clip would
  change the effective step relative to every earlier round. The lr rule (section 4) is evaluated with the same clip.
  The micro-smoke measures per-term gradient shares over the round-12 trainable set (packet section 5).

## 4. lr rule for the driver encoder (docs/r12_lr_rule.json; tools/r12_offline_base.py)
- The centre must move from the physics default (a 2.49, logit +0.37) to the data's region (BASE_FOLLOWER_DIAG:
  around 0.9-1.2, logit about -1.1 to -1.6) within the schedule. Adam moves about lr per step, so the round-8 scalar
  convention (x100 = 5e-3) gives at most about 0.5 logit in 110 steps.
- **Result: head x10, centre x1000** (held-out objective 0.732; the grid is monotone in both scales). The centre
  reaches a about 1.4 / b about 2.4 by step 30 and is stable afterwards (packet section 3).
- Rule: (head scale, centre scale) in {1, 10} x {10, 100, 300, 1000}, choosing the lowest held-out (episode-disjoint
  TRAIN) objective after the PRODUCTION schedule (40 x 4 windows @ 5e-5 s, 70 x 2 @ 3e-5 s, Adam, clip 1).
  - The objective is the one-step Huber of the IDM core (+ round-9 gate x temporal) plus the production prior, on
    S-BASEDIAG's TRAIN captures: 96 windows, 24 held out.
  - Limit: a one-step proxy of the rollout objective, used only to size the step.

## 5. Schedule (`curriculum.stage_steps = [40, 70]`, 120 s x 4 then 300 s x 2, lr 5e-5 / 3e-5)
- Round 9 timing on gpuR: 90 steps in 4 h 50 min, about 190 s per step including the stage-end validation. Full
  validation took 1 h 05 min.
- Round 12 has the same rollout and backward. merge_pivot_state_st already kept the full state adjoint in round 9;
  the extra parameter gradients are cheap.
- Budget: 110 steps x about 190 s = about 5.8 h, plus final validation (16 windows) about 1 h, plus gate smoke about
  1 h = about 7.8 h. That stays inside the about-8 h budget on the 4090.
  - `WMS_LATEST_EVERY=20` checkpoints every 20 steps.
  - Stage checkpoints are written at 120 s and 300 s.
- Why more than round 8's 90 steps, and why weighted to 300 s: the base change matters most for long-horizon delay
  (queue discharge, merge-zone delay). Those terms are active only at the 300 s stage (agg_delay from 60 s, paired
  300 s). The 120 s stage is where the centre and the read-out move fastest; 40 steps x 4 windows covers the
  centre's travel under the lr rule.
- If the smoke's measured step time projects beyond about 8 h, the coordinator can pass
  `--set curriculum.stage_steps=[40,60]`. The registered run is [40,70].

## 6. Interaction with the merge law (fitted on the old base's geometry)
- merge_pivot stays frozen (round-9 decision). Its passage law was fitted on round 8/9's one-step geometry. A slower,
  calibrated base changes the approach speeds and gaps the law sees. The law is not refitted in this round (one fix
  at a time); the post-run checks read merges, stuck ramp cars and ramp delay against round 9.
- **The counted exclusion is never reused.** `MergePivotTerm.prepare` recomputes it in every batch from the recorded
  state and the WM's own one-step geometry (`build_recorded_geometry`, `counted_exclusion`, EXCLUSION_VERSION
  r9-excl-1). Round 12 therefore trains on the cohort its own base implies.
- `tools/r12_support_recount.py` runs the identical `prepare` on the same TRAIN windows with round 9 and with the
  round-12 init and reports the count changes per class (packet section 4).
  - `event_vetoed_explained` (vacated leader ahead / junction occupant: documented geometry) must not change.
  - The WM-motion classes may change.
  - The merge-pivot TF term is diagnostic in round 12 (pivot frozen), so the cohort only changes what is logged and
    scored.

## 6b. Expected effects (proposed; tested by the registered checks)
- **Calibrated core (A + B):** slow cars no longer start about 2x too fast, so queues in and before the merge zone
  discharge more slowly. The merge-zone delay ratio should rise from 0.48 toward SUMO (BASE_FOLLOWER_DIAG).
- **Learned hold on a calibrated base (C):** target followers stop filling the ramp car's gap. The ramp delay ratio
  should fall from 2.1 toward SUMO, and stuck ramp cars (R9_STUCK bistability) should become rarer.
- **Ranking:**
  - REGION_R9.md (coordinator 2026-10-24): at 300 s, ramp + merge carry 0.96 [0.71, 1.76] of round 9's WM-vs-SUMO
    paired offset (+546 veh-s; ramp +255, merge +269). Fixing those two regions alone would correct 12/14 wrong pairs
    (sign 0.875).
  - SUMO's late gains come from merge-zone (-622) and ramp (-508) dynamics that the WM misses (-38).
  - Round 12 targets exactly these processes. The registered read is the ramp + merge offset falling vs round 9.
    up_near (+241) is re-measured; it is the largest region in 7/14 wrong pairs, and slower discharge can move it
    either way.

## 7. Risks
1. **Long-horizon stability with physical settings.**
   - The start point (centre at the physics defaults, every driver equal) over-accelerates MORE than round 9
     (offline r12_init). Early steps may raise the position loss before the centre moves.
   - Lower a and b lengthen the IDM response, so queues may discharge too slowly (BASE_FOLLOWER_DIAG expects
     merge-zone delay to rise toward SUMO, and it could overshoot).
   - Watch the stage-end validation and the post-run delay-by-region.
2. **Identifiability of a** (prior provenance, section 3).
   - Without the prior, one-step error drives a to the 0.3 bound. The rollout terms should resist this, but that is
     unverified offline.
   - With the prior, the population centre is anchored to the literature values that also centre the generator.
   - Report the encoded spread (BASE-3 read).
3. **Re-initialised read-out.**
   - The residual restarts from zero inside a frozen gate. If the gate envelope is too small for what the residual
     must carry, it cannot compensate (deliberate: the core should carry the motion).
   - The trunk is co-adapted to the old head. lr x0.5 limits drift.
4. **Merge law on a new base**: fitted parameters, old geometry (section 6). The counted exclusion is recounted;
   the law itself is not refitted.
5. **Courtesy head on a moving base.** The contrast target uses the live model's control residual, recomputed every
   step. As the base calibrates, `r_hat -> 0`, and the head target moves toward the absolute residual (the intended
   coincidence). The head lr rule was measured on the round-8 base.
6. **Fix D data caveat** (S-R12 checks worker, one window): in the pack frame a junction occupant's `s` is frozen
   for about 2 steps while its recorded speed continues.
   - With filled presence, such a car is a leader at its frozen position (gap error up to about 3-8 m, vs about 58 m
     when hidden). It also keeps its own state through the transit.
   - On round 9 this moves one-step predictions by > 0.5 m/s^2 on about 11 % of that window's rows, mainly by no
     longer resetting the transit car's history and temporal state.
   - It affects only teacher-forced terms. In round 12 that is the courtesy contrast alone; fix B is off and
     rollouts never hide vehicles.
   - Packet section 3 measures it on more windows.
7. **Seed noise** (R10_DIAG): 300 s rollouts are chaotic; a seed change alone moves log ds120 by 0.375. Acceptance
   uses seed-replicated metrics (two seed sets, pooled, bootstrap CIs; packet section 7).
