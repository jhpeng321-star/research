# Sequential P30 Timing Observation/Value-Screen Results

**Date:** 2026-08-30
**Status:** observation/value screen **complete — the frozen gate FAILS**.
Execution stopped at the screen per protocol; no RL, no reward redesign, no
Path A, no prospective seed, no new SUMO run.
**Protocol:** `docs/IDM_TIMING_OBSERVATION_VALUE_SCREEN_PROTOCOL.md`
(frozen before fitting)
**Artifacts:** `runs/eval/idm_timing_observation_value_screen_v1.json`
(existing schedule + pilot artifacts only)
**Tests:** focused 13/13 (recursive targets, provenance, grouped folds,
replay, penalty retention); full repository suite **761 passed, 0 failed**
at execution time.

## 1. What was evaluable (feature provenance)

Only family A (recurrence/phase + static local: stage index, landmark kind,
persistence, runway, predicted dose, current instant, lane one-hot) exists
in the timing artifacts. Families B (local temporal history), C (merge-zone
RSU aggregates) and D (corridor-wide RSU aggregates) are **not present** in
the existing artifacts (no observation arrays, RSU block or history
sequences were stored for these 69 states) and were declared not-evaluable
in the frozen protocol; they require a separately authorized
artifact-generation phase. The screen therefore answers the narrow question:
can the coarse causally-available schedule-state features recover timing
value?

## 2. Screen results (fully leave-one-seed-out)

| policy (cross-fitted) | mean value | activations | waits | helpful/harmful/neutral | dose fails | constraint fails |
|---|---|---|---|---|---|---|
| always abstain | 0.0000 | 0 | 69 | 0/0/69 | 0 | 0 |
| best fixed (abstain) | 0.0000 | — | — | — | — | — |
| activate_now / r2 / r3 (fixed, penalties) | −0.1918 / −0.0225 / −0.0501 | — | — | 13/17/39, 16/13/40, 10/4/55 | 6/1/1 | 7/1/2 |
| **deployable (selected = simple stage rule)** | **0.0000** | **0** | 69 | 0/0/69 | 0 | 0 |
| ridge advantage policy | −0.1176 | 8 | 61 | 0/5/64 | 2 | 4 |
| shuffled-label control | −0.0104 | 28 | 41 | 11/3/55 | 1 | 1 |
| **constrained oracle (reference)** | **+0.0478** | 50 | 19 | 28/41/0 | 0 | 0 |

The deployable candidate chosen strictly inside each training fold is the
simple per-stage rule: no stage's training-grouped OOF mean advantage was
positive, so it abstains everywhere (value 0.0000, i.e., exactly abstention).
The richer ridge candidate generalizes *worse* than abstention (−0.1176,
largely because its 8 cross-fitted activations hit 5 harmful realizations,
2 of them dose-failed), and it is below the shuffled-label control
(−0.0104). Per-seed: the ridge policy's losses concentrate in seeds
4122/4124/4126/4128 (−0.33…−0.43); six seeds receive abstention (0.0).
Per cell (ridge): 5% −0.186, 10% −0.055; 1900 0.000, 2000/2100 −0.173.

## 3. Frozen gate result — FAIL

Every primary gate fails: value ≥ +0.02 over abstain and over best fixed
(0.0000 − 0.0000 = 0.000); ≥40% oracle recovery (0.0%); positive when each
seed is left out (all complements 0.0000, not > 0); ≥3 meaningful seeds
(0); not from one penetration/demand/recurrence (no positive value at all);
selects both activation and wait (0 activations); above shuffled-label
control (TRUE only because the abstaining deployable is ≥ shuffled + 0.01 —
an artifact of abstention, not of learned signal). The only passable
mechanics checks are replay exactness (replay utility 0.0000) and the
no-safety-violation condition (vacuously true for an abstaining policy).
`screen_passes = false`.

## 4. Failure interpretation (protocol §7)

The failure is **observation insufficiency combined with sample scarcity**,
not a mechanics defect and not a threshold artifact:

- the coarse family-A features contain no recoverable timing signal
  (the cross-fitted ridge model is worse than abstain and under the
  shuffled control);
- only 69 states are available, and their activation realizations are
  mostly neutral/negative, so any learned selection risks harm — which is
  exactly what the cross-fitted ridge policy demonstrates (5 harmful of 8
  activations);
- families B–D that could plausibly carry the signal (local dynamics,
  RSU merge-zone state, corridor aggregates) were **not evaluable** from
  the existing artifacts — this is the single justified observation repair
  candidate;
- this is consistent with the earlier diagnosis: the timing headroom is
  real in hindsight (+0.0478) but appears tied to delayed merge-zone
  interactions that the recorded per-landmark geometry does not encode.

Do **not** add prospective seeds; do **not** expand model complexity (a
neural model is not justified by 69 states and would overfit the same
absence of signal).

## 5. Explicit value distinctions

- retrospective oracle value: +0.0478 (unattainable, unchanged);
- cross-fitted deployable policy value: **0.0000–(−0.1176)** (this screen);
- training-set fit: reported above for selection transparency only;
- prospective evidence: not begun (no untouched seed opened).

No claim that any trained controller works is made. The corrected +0.04784
oracle headroom stands as evidence of *potential* timing value, while this
screen establishes that the currently captured causal state does not make
it deployably predictable.

## 6. Recommendation (protocol §7)

1. Either authorise one justified observation repair — a separately frozen
   artifact-generation phase that re-captures the 69 development states'
   schedule trajectories with the full causal observation, RSU and history
   records (native-IDM discovery runs only, same seeds, no treatment
   branches) so families B–D can be screened; or
2. proceed on the rule-based damping-plus-wall escalation paper path with
   the fixed 5–10% arm and the honest boundary documentation.

Either way, next steps require review; no RL, reward audit or Path A was
started.
