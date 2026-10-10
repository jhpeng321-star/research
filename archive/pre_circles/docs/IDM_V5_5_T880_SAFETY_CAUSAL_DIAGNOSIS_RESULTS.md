# IDM v5.5 t=880 safety-causality diagnosis results

Status: completed development diagnosis, 2026-09-01. No validation/final seeds,
predictor training, RL training, or controller repair was performed.

## Frozen execution

The diagnosis ran 12 SUMO branches: three Gate-F-admitted development seeds
(4401, 4402, 4404), two ablations (`sham`, `assembly_only`), and one paired
analytical-IDM damping control replay for each ablation. Six workers were used.
All arms replayed the same 2700/400 veh/h IDM scenario, 30% CAV penetration,
absolute activation at 880 s, and the common 1080 s horizon.

## Integrity gates

All machine gates pass:

- sham physical identity: **3/3 exact**;
- assembly-only first override: **3/3 at decision+1**;
- assembly-only HOLD state: **0/3** (the wall is released after one command);
- common horizon: **6/6**;
- predecision branch identity: **6/6**;
- hard-brake onset attribution: present for every arm, including phase,
  vehicle/lane, leader, gap/TTC and wall-membership fields.

The sham uses the complete request/latch/lifecycle code path but a zero-effect
wall actuator. Its command and RSU metadata differ by design; physical identity
compares the SUMO vehicle state, active-CAV set, collisions and teleports.

## Per-seed outcome comparison

| seed | arm | hard-brake steps (control → arm) | treatment-only onsets | post-release congestion (control → arm) | delay s (control → arm) |
|---:|---|---:|---:|---|---:|
| 4401 | sham | 3 → 3 | 0 | none → none | 4351.1 → 4351.1 |
| 4401 | assembly-only | 3 → 3 | 0 | none → none | 4351.1 → 4369.3 |
| 4402 | sham | 4 → 4 | 0 | sustained → sustained (same) | 7094.3 → 7094.3 |
| 4402 | assembly-only | 4 → 3 | 0 | 91 s → 91 s | 7094.3 → 7037.3 |
| 4404 | sham | 4 → 4 | 0 | none → none | 4741.5 → 4741.5 |
| 4404 | assembly-only | 4 → 4 | 0 | none → none | 4741.5 → 4741.5 |

Assembly-only does not reproduce the full-pulse safety pattern. Its physical
divergence occurs at the first assembly command by construction, but no new
treatment-only hard-brake onset appears in any seed. The small 4402 change is
not a safety failure and does not establish benefit.

## Hard-brake onset attribution

The full frozen-wall pilot is the reference for the safety failures. Comparing
event keys `(vehicle_id, relative_step)` against its matched damping arm gives
these treatment-only onsets (shared onsets are retained in every case file):

- **4401:** mainline.639 at 939 s, mainline.564 at 950 s and mainline.650 at
  963 s (all after release, while damping resumed).
- **4402:** mainline.658 at 973 s and mainline.670 at 981, 984, 987, 989 and
  1004 s (all after release, while damping resumed).
- **4404:** ramp.99 at 912 s during COOLDOWN, mainline.674 at 994 s and
  mainline.671 at 996 s (the latter two after release during damping).

The extra HOLD/TAPER onsets in the treatment arms are important exposure
evidence even when their event keys are shared with control; their lifecycle
phase, wall membership and leader/follower context are preserved in the raw
`event_starts` records.

These events were not controlled wall CAVs and generally had no controlled
wall member as their immediate leader. They are downstream/follower responses,
not direct actuator collisions. The diagnosis artifacts retain the complete
vehicle-level fields (`event_starts`) for exact re-analysis, including lane,
leader gap, leader TTC, lane change, nearest controlled-CAV distance, lifecycle
phase, and elapsed time from the 900 s planned release boundary.

## Causal classification

1. Replay/software inconsistency: **ruled out** (all shams exact).
2. Initial member selection/assembly: **not supported as dominant cause**;
   assembly-only generated no treatment-only onset.
3. Continued HOLD plus taper/release/discharge: **dominant supported mechanism**.
   The full pulse adds hard-brake exposure in HOLD/TAPER/COOLDOWN, while the
   one-second arm does not. The three seeds also show downstream service and
   metastable responses after release.
4. Pure metastable amplification: remains a contributing explanation for the
   late post-release events, but cannot explain the phase-localized excess by
   itself because the sham is exact and assembly-only is comparatively benign.

The evidence supports one diagnosis: **the sustained hold and its taper/release
discharge, rather than replay or initial formation, is the limiting safety
mechanism**. This is diagnosis evidence, not proof that a repaired wall will be
safe or beneficial.

## Decision and next action

The next defensible step is one separately versioned hold/release repair review:
retain the same 20 s timing, compact geometry, full-lane membership, damping
control arm, and outcome definitions; modify only the hold/release command law
to limit accumulated follower compression and release-wave discharge. The
repair protocol is frozen for review in
`docs/IDM_V5_5_T880_HOLD_RELEASE_REPAIR_PROTOCOL.md`.

No implementation or four-seed repair gate has been run. Predictor training and
expanded seed generation remain unauthorized.

## Artifacts

- runner: `experiments/idm/run_idm_v5_5_t880_safety_causal_diagnosis.py`;
- protocol: `docs/IDM_V5_5_T880_SAFETY_CAUSAL_DIAGNOSIS_PROTOCOL.md`;
- manifest and cases: `runs/probe/idm_v5_5_t880_safety_causal_diagnosis/`;
- machine summary: `runs/probe/idm_v5_5_t880_safety_causal_diagnosis/summary.json`.
