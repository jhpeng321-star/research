# C4 policy v2 — lane head + cooperation-centred training (branch c4/policy2, 2026-10-12/13)

Owner decisions 2026-10-12 late. No designed controller anywhere in the policy's learning: no imitation,
no classic controller as an action or check candidate. The IDM "native CAV" appears only as the
counterfactual reference B.

## A. Per-CAV acceleration AND lane decision
- `CavPolicy(lane_head=True)`: shared trunk -> tanh-Gaussian acceleration (analytic) + categorical
  stay/left/right head (logits zero-initialised = uniform over feasible choices). With `lane_mask`
  it returns `PolicyOutput(accel, lane_choice, log_prob_accel, log_prob_lane, entropy_accel,
  entropy_lane, lane_options)`; without it the legacy `(accel, log_prob)`. `noise=PolicyNoise` gives
  common random numbers.
- Mask (`wms.control.lanes.lane_feasibility`, same code in imagination and SUMO): stay always;
  mainline only 0<->1, never into ramp lane 2; ramp CAVs may "merge now" (left) only inside the merge
  zone; target-lane bumper gaps front >= 6 + 1.0 s * v_ego + 1.0 s * closing, rear >= 6 + 1.0 s *
  v_rear + 1.0 s * closing (alongside blocks); dwell >= 3 s; no mainline change within v*1 s + 10 m of
  a SUMO edge boundary (s = 0, 75.44: lane indices differ across edges). Simultaneous changes are
  re-checked against everyone's proposed lanes (`apply_lane_choices`). The bridge's forced merge of
  ramp CAVs still runs every tick and is never blocked.
- Imagination (`bridge.make_cav_controller(..., lane_rules=...)`): the chosen lane is applied at the
  decision tick and held; the WM state lane changes for that CAV.
- Trainer: accelerations keep analytic gradients through the WM; the lane head is trained by
  REINFORCE (DreamerV3 discrete style): `-sum_r c_r * mean_t[A_r(t) * sum_k log pi(lane_k)]`, A = (R -
  V)/S on all-CAV rows, (R - B)/S on counterfactual rows, plus `lane_entropy_weight` * masked entropy
  (S = the shared return scale).
- Reports (contract `REPORT_FEATURES`, 11 -> 13): each other CAV's last commanded (safe) acceleration
  and lane intent (-1/0/+1) from its previous decision (`obs.CommandMemory` in SUMO, bridge memory in
  imagination; zeros for new / not-yet-commanded slots and at t0 of a training start).
- Checkpoint `c4_dreamer/3` (lane head, lane rules, counterfactual config). `/1` (accel-only) loads
  as `CavPolicy(lane_head=False)` (always stays); `/1` and `/2` report encoders are zero-padded for the
  two new features (identical outputs).
- SUMO (`traffic_shepherd/c4/env.py`): `step(accel, lanes=[1,K])` (alias `lane=`; -1 = no change,
  0/1/2 = target WM lane). Policy CAVs (finite command + lanes passed) get `laneChangeMode` 513 =
  strategic 01 (route-required changes incl. the on-ramp merge stay on unless they conflict with
  TraCI) + TraCI-requests-respect-others' gaps 10 (bit 9); cooperative, speed-gain, keep-right and
  sublane bits 0. Previous mode restored on release; HDVs and non-policy modes untouched. A target is
  issued as `changeLane(vid, index_on_current_edge, 1.0)`; edge-end straddles deferred, junctions
  refused. Counts in `info["lane_commands"]`/summary, rows in `action_timeline.jsonl.gz`
  (`lane_command`) and `episode.npz` `lane_command_target`. `load_policy_fn` returns a
  `PolicyExecutor`: `exe(state)` -> accel, `exe(state, with_lanes=True)` / `exe.act_with_lane(state)`
  -> (accel, lanes); `run_policy_episode` and `deploy` pass lanes through.

## B. Cooperation as the main training signal (`wms.control.counterfactual`)
Per selected start, one batched WM rollout per member with common random numbers (noise per
start/step/slot shared by all rows of the start): B (all CAVs native IDM, no lane changes),
D_i = J(i) - B, D_G = J(G) - B (disjoint close groups: connected components within 150 m, size 2..6,
contiguous window), D_G\i (flip-one, 2 members per group; pairs reuse the other member's solo row),
D_all = J(all) - B. C_G = D_G - sum D_i, M_i = D_G - D_G\i, I_i = M_i - D_i.
- Objective (maximised, lambda-return differences, mean over steps/members/starts):
  w_all*D_all + w_group*D_G + w_ind*sum D_i + w_coop*C_G + w_team*sum M_i. Implemented as row
  coefficients (`row_coefficients`); the lane head uses the same coefficients.
- ANTI-GAMING GUARD: C_G rewards making solo behaviour worse. Config validation enforces
  w_coop <= w_ind at every curriculum knot (piecewise-linear schedule => everywhere); groups of one
  start are disjoint; for pairs G\i is a solo CAV, so the team term is applied to groups of >= 3 only
  (M_i/I_i still logged). `row_coefficients` re-checks at run time that every individual row keeps a
  coefficient >= 0. Residual: flip rows of >= 3 groups carry -w_team (incentive to make the TEAM G\i
  worse), bounded by w_team.
- Curriculum (config): update 0 -> 1500: w_ind 1.0 -> 0.5, w_group 0.5 -> 1.0, w_coop 0 -> 0.5,
  w_team 0 -> 0.5, w_all 0.25 -> 1.0 (held after).
- Group-rich sampling: `group_state_fraction` 0.6 of each batch must have >= 2 CAVs chained within the
  radius in s in [-800, 75.44] AND an RSU-field queue cell or occupied cell with mean speed < 15 m/s
  (`StartStateSampler.sample_where`, up to 20 draws per slot, cached per window).
- Logging per update: B, D_i, D_G, D_all, C_G, M_i, I_i means, fraction of groups with C_G > 0,
  tables by group size and penetration bucket (world-id p02..p40), lane change rate / entropy / PG
  loss, group-rich fraction. Validation: `coop_gain_val`, D_all (selection score); `policy_best.pt`
  by D_all, `policy_best_coop.pt` by coop_gain_val.

## RSU check (owner 2026-10-12 late)
Candidates are direct-output only: policy, policy_group_0 (zero signal), policy_group_1 (perturbed
signal, `--group-perturbation` 0.1), policy_sample_0/1 (stochastic policy, `rsu_check.NoiseStream`:
same draws in every WM member, driver sample and in SUMO replay), hold_speed (a = 0, stay,
safety-filtered). All are executed through the policy path (direct accel + lane commands);
native/damping/pi are only whole-episode baselines (`--controller`).

## Measured (CPU laptop, 1 torch thread, real R5b `wm_s_best.pt`, real TRAIN windows, --smoke = batch 2, H 4)
| run | rows / update | s / update | notes |
|---|---|---|---|
| counterfactual off | 2 | 1.72-3.26 | lane head + REINFORCE only |
| counterfactual on (defaults) | 26-29 (2 all + 24-27 cf) | 20.96-28.43 | 3 groups (size 3-6) per update, group-rich 100 % |
Smoke update 2: D_all -0.18, D_G -0.068, D_i -0.016, C_G +0.0028 (3/3 groups > 0), M_i -0.015,
I_i +0.0006; VAL coop_gain_val 0.0007, D_all -0.067. (Two untrained updates: mechanics only.)
Output: `runs/active/C4-POLICY2-SMOKE/` (local, not committed; the earlier pre-cooperation smoke is
kept as `runs/active/C4-POLICY2-SMOKE-v2-precoop/`).

SUMO episode (`deploy --controller policy`, seed 4107, demand 2700, p 0.2, warmup 300, horizon 60,
the smoke `policy_latest.pt`): lane commands issued 69, completed within 1 s 26, not completed 43
(SUMO refused: gap below its own vType secure gap; re-requested next tick), 0 collisions, 0
teleports; 69 lane rows in the action timeline / npz. Gap-rule check on the pre-cooperation smoke
policy (same seed, 60 s): time gap 0.5 s -> 182 issued / 45 completed; 1.0 s -> 93 / 45 (default now
1.0 s); laneChangeMode 257 (only avoid immediate collisions) -> 67 / 67 but window hard braking 178
vs 73 vehicle-s, so 513 is kept.

## GPU sizing (ESTIMATE, not measured)
Memory: see CFMEM_NOTES.md (`trainer.checkpoint_imagination`, default on: B16/H20/max_starts 6 =
6.3 GB CPU-measured peak vs ~126 GB without; same objective and gradients, ~+40 % time).
CPU cost is ~linear in rows x physics steps (~0.1 s per row-step incl. backward, 1 thread). On a
4090 the WM-S step is launch-latency bound (estimate 30-60 ms forward+backward per physics step,
nearly flat up to a few hundred rows; the bridge's per-slot merge loop adds ~K small ops per step).
Proposal to keep an update < ~5 s: horizon 15-20 control steps (30-40 physics steps), batch_size 16,
max_starts 6, n_group 2, n_flip 2 (~16 + 6 x 13 = ~95 rows). Measure with `--max-updates 5` on the
box before the run and lower horizon or max_starts if above 5 s.

## Open issues
- SUMO accepts ~40-50 % of the policy's lane requests at the 1.0 s gap rule (its secure gap depends on
  vType tau/decel); imagination executes every feasible change -> sim-to-real gap for lane moves.
- Imagined lane changes are instantaneous; WM HDV reactions to CAV cut-ins are learned only from
  data with few CAV lane changes.
- At t0 of each training start the command/intent report features are zero (no recorded commands).
- Ramp CAVs: the head's "merge now" is redundant with the forced merge (ASAP when safe) in
  imagination; in SUMO it issues changeLane while the strategic model also merges.
- With counterfactual rows the critic is still fit on the all-CAV rows only.
