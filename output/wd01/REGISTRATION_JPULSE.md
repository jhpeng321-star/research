# REGISTRATION: WD-01-JPULSE: a PULSE campaign of short actions (60, 120 or 240 s), then native, from saved chunk-4 TRAIN t0 states

**Status:** FROZEN, 2026-10-01, world model session. Owner-approved WITH the 240 s option (relayed and verified by main). This supersedes the pre-approval text with LF-sha `24a7b71b…`.

- Owner decision relayed by main discussion. Registered BEFORE any run.
- A rental-box CPU job in the payload. The launcher is GATED: it refuses to
  start without an owner AUTHORIZATION.json bound to this file's LF-sha256.

## 1. Question

Actions run only briefly, but their effects can come much later. How do a
60, 120 or 240 s damping or PI pulse, followed by fleet native, change delay,
TTC < 3 s exposure and hard braking at τ 120, 300 and 570 s? And how well do
the models (Part B v0.4b direct-difference; WM-N1 Idea A) predict those
delayed effects?

## 2. Worlds

- **Source:** the chunk-4 TRAIN worlds only, as labelled in
  `output/campaign_stage1_v3/results_plan_chunk4/WORLD_PLAN.json`.
  CALIBRATION, HOLDOUT and chunk 5 are never used.
- **Available (metadata check, 2026-10-01):** 63 TRAIN worlds, all with a
  saved `__base` t0 state. By cell:
  - 2000 veh/h: 4 worlds;
  - 2400: 8;
  - 2600: 21;
  - 2700: 30.
- **Selection:** 44 worlds, chosen a priori.
  - Exclude the 4 worlds at 2000 veh/h. They are mostly uncongested, as in
    the chunk-5 choice.
  - From the remaining 59, draw 44 by stratified random sampling within
    (loading × penetration) cells, proportional to the cell counts, with
    plan seed 20261001.
  - The draw is made by the launcher's plan builder, and its output list is
    recorded in `results_plan_jpulse/WORLD_PLAN.json` before any run.
  - Outcomes are never consulted in selection.

## 3. Arms (per world; all restored from `__base` saved state at that world's registered t0)

| arm | CAV fleet schedule over the 600 s window (1 s control / 0.5 s physics) | duration |
| --- | --- | --- |
| `__pulse_damp60` | damping [t0, t0 + 60), then native | 600 s |
| `__pulse_damp120` | damping [t0, t0 + 120), then native | 600 s |
| `__pulse_pi60` | PI external-U [t0, t0 + 60), then native | 600 s |
| `__pulse_pi120` | PI external-U [t0, t0 + 120), then native | 600 s |
| `__pulse_damp240` | damping [t0, t0 + 240), then native | 600 s |
| `__pulse_pi240` | PI external-U [t0, t0 + 240), then native | 600 s |

- **Reference:** the EXISTING chunk-4 `__native` arm of the same world. It is
  restored from the same `__base` state at the same t0, with fleet native,
  so the pairs share the state and seed stream.
- **No new reference run** is made. If a world's `__native` receipt is not
  complete, the world is replaced by the next draw from the same cell and
  the replacement is logged.
- **Implementation:** the switch is a planned per-CAV schedule
  (`switch_schedule.json`, 30 s-aligned segments) through the chunk-4
  campaign code. `CommandOwnership` is the sole writer: native selection
  releases the override. Everything else is unchanged (speedMode as
  chunk 4).
- **No walls.** J4B, a separate registration, covers short v2.2 walls.

**Budget:** 44 × 6 × 600 = **158,400 sim-s** on the rental CPU box.

- There are no retries of completed runs.
- A crashed run may be restarted from scratch once, with disclosure.
- Disk: about 5–6 GB of raw logs, scaled from chunk 4.

## 4. Integrity (as chunk 4 / J1)

- A receipt per run: complete, machine_id, rows written equal to expected,
  and the registration sha.
- The scenario copy digest, and the saved-state digest equal to that of
  `__base`.
- The schedule semantics are checked against `switch_schedule.json`.
- 0 collisions and 0 teleports are expected. Any occurrence is recorded, not
  re-run.
- Post-run grading reads receipts and row counts only, never outcomes.

## 5. Use

1. **As TRAIN data.** For Part B v0.4b (direct-difference; the pulse arm vs
   `__native`, with the planned-schedule descriptor per
   `ACTION_DESCRIPTOR_REQUIREMENT_20261001.md`), and for WM-N1 Idea A
   (fine-tune / retrain data).
   - If the pairs arrive after the first training, they are used in a
     disclosed retrain or fine-tune.
   - They are never used for selection on the chunk-5 TEST.
2. **As a registered diagnostic, "short action → delayed effect".**
   - For each model: sign accuracy and ΔV CRPS against the zero-contrast and
     TRAIN-cell-mean baselines, at τ ∈ {120, 300, 570}.
   - Evaluated OUT-OF-FOLD over the 44 PULSE worlds, with world-grouped
     5-fold CV. A model trained with PULSE excludes the evaluated fold's
     worlds.
   - Stratified by pulse length (60 / 120 / 240 s) and controller (damping / PI).
   - Reported as diagnostic evidence. It is NOT a selection input, and it
     does not affect the WM-N1 §4 verdict.
3. **Disclosure.** Before PULSE, the training data is dominated by
   whole-window actions, as stated in the action-descriptor requirement.

## 6. Owner decisions (resolved 2026-10-01)

1. 44 worlds, excluding 2000 veh/h.
2. Pulses of {60, 120, 240} s for {damping, PI}, i.e. 6 arms.
3. The existing `__native` is the reference.
