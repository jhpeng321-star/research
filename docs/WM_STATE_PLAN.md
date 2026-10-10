# WM-S: physics-informed next-state world model (owner-approved plan, 2026-10-05)

Owner decision (2026-10-05 ~20:00 chat): the world model becomes a **pure next-state predictor**.
Supersedes the effect-head (delay-head) direction of WM-N1 A8-A13 as the primary approach; A8-A13
results, data, scorer and closed-loop system are kept as evidence and reusable infrastructure.

## Model
- Output, every 0.5 s step: next position, speed, lane of every vehicle on the modelled road.
- CAVs: moved by the REAL controller code (native / PI / damping / wall / trajectory) -> controller-agnostic,
  unseen controllers plug in without retraining.
- Human drivers: physics base (IDM car-following, MOBIL + mandatory-merge rule, SUMO-matched vType
  parameters, open road: exits + entries from observed entrance counters) + a learned correction
  (acceleration + lane-change adjustment).
- The correction knows the **number of steps since the last real observation** (k) and each car's
  short motion history; correction strength grows with k (near-term physics-dominated, far-term
  stronger learned correction). Trained on long multi-step rollouts vs recorded SUMO trajectories.
- Delay, queues, flow etc. are COMPUTED from the predicted states (no delay head).
- One model for now (ensemble deferred; owner: "just one model for now").

## Evaluation (identical for every model)
1. Car-level vs real state at 5/10/30/60/120/300 s: position error (median, p90), speed error, lane
   agreement, presence (entries/exits); HDV and CAV separately.
2. Traffic-level per road section (upstream / merge / downstream) per 30 s: mean speed, density,
   merge queue length, stopped vehicles, merge flow, total delay.
3. Paired controller difference: predicted (PI - native) traffic-level differences from the same
   start + same random draws vs the true SUMO difference.
4. Baselines: keep-current-speed extrapolation; pure physics (no correction); previous model.
Data: TRAIN worlds for development; CAL-tune for model comparison; the 30 sealed fresh worlds
(TEST_REPL) only for final checks. CAL-cal / HOLDOUT / chunk-5 / J4B / INC1 stay sealed.

## Phases and gates
- P0 evaluation harness (CPU) - report for today's rollout variants + baselines.
- P1 physics base (CPU): merge geometry fix (done, flag), merge gap acceptance / yielding calibrated
  to the recorded merge rate, SUMO vType IDM (done), open road (done), per-driver desired speed from
  observed history. Gate: pure physics beats keep-current-speed at all horizons; merge queue/flow close.
- P2 step-aware correction (GPU): k + history inputs, gate growing with k, multi-step rollout training.
  Gate: beats pure physics (car + traffic level).
- P3 ensemble: deferred.
- P4 decisions (states -> delay -> paired PI-native difference -> decision test on CAL-tune, fresh
  worlds, closed loop). Gate: beats always-native on fresh worlds.
- P5 unseen controllers: leave-PI-out; unseen wall law v2.2.
Every phase registered before running (runs/active/WM-S-*), results logged; owner approves GPU use.

## Speed-up (owner: shorten preparation, start training ASAP)
P0, P1 and P2-code run in parallel on OpenCode executors (DeepSeek / GLM / gpt-5.6-luna), each in its
own worktree; P2 training starts as soon as P2 code passes its CPU dry run and P1's physics base is
merged (later P1 refinements can be picked up in a second training run).

## Addition 2026-10-06 ~00:55 (owner-approved direction): driver-identification physics base (WM-S2)
Owner: per-step world model stays the MAIN model; choose actions on their full after-effect from one snapshot
(open-loop - no re-anchoring on future observations); keep the unseen-controller novelty (no controller-specific
training signal, e.g. no paired-difference loss). To slow error accumulation, replace the free per-step acceleration
correction by a learned estimate of HOW EACH HUMAN DRIVER DRIVES, used inside the physics:
- Driver encoder: from each HDV's accumulated observed trajectory up to t0 (RSU/CAV-observable only; history length
  grows with observation time) plus its interactions with neighbours, an attention encoder (owner's earlier research
  idea: attention weighted by how each driver drives) outputs per-driver physics parameters: IDM v0 (desired speed),
  T (time headway), s0, a_max, b (comfortable braking), MOBIL politeness / threshold, merge gap acceptance and urgency;
  plus an uncertainty per parameter. Shrinks to vType defaults when history is short.
- Rollout: per step, pure physics (IDM + MOBIL + merge, SUMO-fidelity frame, open road) with those per-driver
  parameters; CAVs moved by the real controller code (any controller, unseen ones included). Optional small
  step-aware residual (existing g(k) gate) kept only if evaluation shows it helps; default = parameters only.
- Training: end-to-end through the differentiable torch physics on multi-step rollouts vs recorded SUMO
  (curriculum 10 -> 120 s, later 300 s); per-car losses early, traffic-level losses (section speed, queue, delay,
  slowdown extent) at long horizons; stability aids: state-noise injection during training, penalty on error
  amplification. All losses on HDV/traffic, never on controller labels.
- Evaluation unchanged (car-level 5-300 s, traffic-level per section, paired PI - native vs SUMO truth, decision test),
  comparators: keep-speed, pure physics (vType defaults), WM-S run 1 (free residual), WM-S2. Headline: leave-PI-out
  and unseen wall law.
- Reuse: T3 per-driver desired-speed estimator + [K,H,F] history tensor (wms/driver), torch physics (wms/torchphys),
  trainer/curriculum/check400 (wms/p2), eval harness (wms/eval).
- Deferred: micro->macro (traffic-wave) hand-off for very long horizons (owner: "leave the second one for now").
Steps: CPU prototype + dry run first; register before any GPU training; owner approves GPU.

## Roadmap
Weekly plan, thesis and contributions C1-C3: docs/WM_S2_ROADMAP.md (owner-approved 2026-10-06).
