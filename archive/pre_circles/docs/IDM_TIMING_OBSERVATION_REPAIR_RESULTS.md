# Observation-Repair Screen — Results (Sequential P30 Timing)

**Date:** 2026-08-30
**Status:** observation-repair screen **complete — every family increment
FAILS**. Execution stopped at the screen; no RL, no reward audit, no Path A,
no seed ≥4201, no new treatment branch (capture was native-IDM only).
**Protocol:** `docs/IDM_TIMING_OBSERVATION_REPAIR_PROTOCOL.md` (frozen before
the SUMO capture)
**Artifacts:** observations `runs/eval/idm_timing_observations_v1/` (69
records, 170 landmark identity entries, **zero mismatches**); screen
`runs/eval/idm_timing_observation_value_screen_v2.json`.
**Tests:** focused 19/19 (window/provenance/identity/feature-order/replay);
full suite **764 passed, 0 failed**.

## 1. Reconstruction integrity

69 states captured (33 at 5%, 36 at 10% penetration; 12 seeds; 1900/2000/
2100 veh/h) with 170 landmark identity entries (69 t1 + 58 t2 + 43 t3), each
verified against the frozen schedule record on all seven identity fields
(step, landmark, candidate, lane, observation fingerprint, physical-state
fingerprint, config hash). Mismatch count = 0; every observation is joined
to its exact existing treatment outcome. No schedule, treatment, outcome,
reward or threshold was modified.

## 2. Captured feature quality

The compact B–D observations are non-degenerate: 24 of 32 B/C/D features
show standard deviation above 0.2 across the states (ego speed ±2.4 m/s,
leader gap ±58 m, merge-zone lane speeds ±1.9–10.2 m/s, zone density ±26
veh/km, zone severe counts ±2.5, etc.). The negative screen result is
therefore **not** an artifact of a degenerate capture.

## 3. Cross-fitted screen results (leave-one-seed-out)

| family subset | deployable (per-fold selection) | value | ridge cross-fitted | value | acts | useful/harmful/neutral | dose/constraint fails |
|---|---|---|---|---|---|---|---|
| A | simple stage rule (abstain) | 0.0000 | ridge | −0.1176 | 8 | 0/5/64 | 2/4 |
| A+B | simple stage rule (abstain) | 0.0000 | ridge | −0.1501 | 31 | 4/12/53 | 3/5 |
| A+B+C | simple stage rule (abstain) | 0.0000 | ridge | −0.1471 | 25 | 4/9/56 | 5/5 |
| A+B+C+D | simple stage rule (abstain) | 0.0000 | ridge | −0.1471 | 37 | 6/10/53 | 4/5 |

Fixed baselines (unchanged): abstain 0.0000; now −0.1918; r2 −0.0225;
r3 −0.0501. Oracle reference +0.0478. Shuffled-label controls: negative for
the ridge (e.g., A+B −0.0104) — the ridge is **below** its shuffled control
in every increment.

## 4. Frozen gate verdict — FAIL (all increments)

All four increments fail every primary gate: value ≥ +0.02 over abstain and
over best fixed (deployable 0.0000); ≥40% oracle recovery (0.0%); positive
when each seed is left out (0.0000 complements); ≥3 meaningful seeds (0);
activation/wait diversity (0 activations); no-single-cell dominance (no
positive value at all); above shuffled control (passes only vacuously for an
abstaining policy); above majority-action (no); safety/service (vacuous).

## 5. Interpretation

The reconstructed B/C/D observations — real, causally captured, varied
traffic state — do **not** make the +0.0478 timing headroom deployably
predictable. As feature families grow, the cross-fitted ridge generalizes
*worse* (more activations, more harmful selections, below its own shuffled
control): the signal that an oracle exploits is not recoverable from local,
merge-zone RSU or corridor state at the decision instants, and additional
model capacity would only overfit the 69 development states. This
replicates and strengthens the earlier screen result with the previously
missing families. Consistent with the earlier diagnosis, the headroom
appears tied to delayed, metastable merge-zone realizations that are not
encoded in any predecision observation tested.

Per protocol §6: fixed-P30 timing prediction is **unsupported**. Do not add
model complexity or further development seeds automatically.

## 6. Distinctions and recommendation

- retrospective oracle: +0.0478 (unchanged, unattainable);
- cross-fitted deployable policy value: 0.0000 (abstention) for every family;
- training-set fit: not evidence (reported only for selection transparency);
- prospective evidence: not begun.

Recommendation: adopt the rule-based damping boundary plus coordinated
moving-wall escalation as the paper path — the fixed 5–10% rule arm
(frozen, prospective acceptance on 4201–4220 when separately approved),
the demonstrated authority boundary at 5–10% penetration, and the
high-authority wall contribution. P30 activation-timing RL is not
supported by any observation family tested; no training, reward redesign or
Path A was started.
