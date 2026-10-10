# DIAGNOSIS — the WD-01-PA2G-A discretionary lane-change churn

- **Symptom (PA2G-A, pre-fix engine, commit 1f657ff):** v2g predicted
  68,614 discretionary LCs vs 243 true at H10 and 486,428 vs 949 at
  H120, with 2.2–13.9 predicted collisions per 1000 vehicle-steps.
- **Date:** 2026-09-30. Zero simulation; all evidence from logged TRAIN
  windows (chunk-2 truth) + deterministic engine rollouts.
- **Verdict:** an implementation bug, not a parameter problem. The
  textbook IDM/MOBIL values were never touched.

## 1. The counting was checked FIRST (ruled out)

The eval counts lane changes identically for truth and prediction:
`WindowSeries.stats` (metrics.py:211-256) counts one event per
(vehicle, step) whose model-lane index changes, classified
2→1 merge / 1→0 disc_left / 0→1 disc_right; the same code path runs on
the truth rows and the engine snapshots (`eval_a.window_rows.emit`).
Independent check: for the reproduction window below the engine's own
`lc_counts` {'disc_left': 139, 'disc_right': 112} exactly equals an
external per-step trace over the engine snapshots (251 events). The
numbers were real engine behaviour, not a counting bug.

## 2. Minimal reproduction from one logged window

World `w8538m2600p40t360` (chunk-2 TRAIN), window w = 870 s: 94 tracked
zone vehicles (40 CAVs), the worst PA2G-A H10 window (251 predicted
disc LCs, 51 predicted collisions in 10 s). Reproducer:
`partA_v2g/diag_churn.py` (default arguments are this window).

Per-vehicle LC trace of the worst offenders (pre-fix):

```
mainline.515: 5  0->1@870.0 1->0@872.0 0->1@874.0 1->0@876.0 0->1@878.0
mainline.516: 5  0->1@870.0 1->0@872.0 0->1@874.0 1->0@876.0 0->1@878.0
mainline.361: 5  1->0@871.0 0->1@873.0 1->0@875.0 0->1@877.0 1->0@879.0
... (dozens of vehicles; interval stats: 142 of 166 inter-change gaps
     are exactly 4 steps = 2.0 s = the nominal cooldown_s cap)
```

Every churner ping-pongs at the cooldown-limited maximum rate — the
guard was masking the rate, not the cause.

## 3. Root cause: `_neighbors()` swapped leader and follower

`engine.py` pre-fix (commit 1f657ff) lines 406-408:

```python
rows = rows[np.argsort(self.s[rows], kind="stable")]   # ASCENDING s
for j in range(len(rows) - 1):
    follower[int(rows[j])] = int(rows[j + 1])   # WRONG
    leader[int(rows[j + 1])] = int(rows[j])     # WRONG
```

`s` increases downstream, so in ascending-s order `rows[j+1]` is AHEAD
of `rows[j]`: `rows[j+1]` is the LEADER of `rows[j]`. The assignments
are swapped — every vehicle's "leader" was the vehicle BEHIND it.
Direct observation (pre-fix): for `mainline.515` (s = 2770.9, real
same-lane gap 76.3 m ahead) the engine returned
`leader = mainline.519` at s = 2722.8 — 53.1 m BEHIND it.

Every MOBIL decision helper uses `_neighbors()` via
`_own_leader_follower`, so all decision bumper gaps were computed
against the wrong vehicle:

- `_a_stay` (the ego "stay" acceleration) computed
  `gap = s[behind] - length - s[ego] <= -length < 0`;
- `_idm1` floors non-positive gaps at 1e-6 (engine.py line 79), so the
  IDM interaction term `(s*/gap)**2` exploded:
  `a_stay = -2.2e15` for mainline.515 at step 1 (instrumented dump,
  section 2 reproducer);
- the new-follower "before" term (`a_new_before`) likewise used the
  swapped neighbour, so `d_new = a_new_after - a_new_before` came out
  at +1e15 … +1e16;
- gain = Δa_ego + 0.5·(d_new + d_old) ~ +1e15 ≥ threshold in BOTH
  directions → every candidate with a same-lane vehicle behind it
  changed lanes at every opportunity, in alternating directions —
  manufacturing congestion and collisions (the pre-fix window had 51
  collisions in its first 10 s), which in turn produced more
  non-positive gaps and sustained the churn.

The propagation path (`_leaders_vector`, engine.py:247-262) independently
implemented the correct order — which is why the model still drove
plausibly and the bug hid inside the lane-change layer only.

### Textbook MOBIL conformance (Treiber & Kesting)

- *Incentive from accelerations AFTER vs BEFORE for ego, new follower,
  old follower, same state:* violated — the "new follower"/"old
  follower" roles pointed at the wrong vehicles, and the accelerations
  used in the decision were not the model's accelerations (see 4).
- *Safety on the new follower:* violated — `b_safe` was evaluated on
  the vehicle AHEAD of the insertion point (the swapped "follower"),
  not on the follower behind it.
- *At most one lane change per vehicle per step:* held (each vehicle
  appears once per direction pass; the second pass re-filters with the
  updated `_lc_last`).
- *Simultaneous changes must not create overlaps:* held — changes are
  applied front-most-first with the target lane re-sorted after each
  acceptance (`gap_floor` on the new leader, `b_safe` on the new
  follower); pinned by `test_simultaneous_lcs_leave_no_overlap`.

## 4. Secondary inconsistency fixed: decision accelerations were unclipped

The dynamics path clips IDM output to `dynamics.a_clip_mps2 = [-9, 3]`
before use, but the decision helpers used raw `_idm1` values. Any
overlap/collision state (which the ballistic integrator can produce and
the model never repairs) drives the raw IDM to ~-1e15 through the same
1e-6 gap floor and poisons the incentive even with correct neighbours.
Fix: all decision accelerations (ego stay/change, new/old follower
before/after) now pass through the nominal `a_clip`
(`engine.py::_aclip1`), i.e. the MOBIL terms are exactly the
accelerations the model would assign — the textbook "same state"
requirement. The `b_safe` predicate is unchanged (clipping preserves
the `< -b_safe` rejection set).

## 5. Design-deviation flag for the owner (no parameter values changed)

`params_nominal.json` declared three NON-textbook "stability guards":
`cooldown_s = 2.0`, `lc_min_speed_mps = 1.0`, `ego_gain_floor_mps2`.
Per the owner's design-deviation rule (non-textbook rules stay out of
the nominal model) they are retired to `0.0` (OFF) in the nominal set;
`ego_gain_floor` additionally became a no-op at 0.0 (it previously
rejected any change with Δa_ego < 0 even at floor 0.0, which is not
textbook MOBIL — politeness may legitimately compensate a small ego
loss within the threshold test). **No textbook value changed**
(IDM T 1.5, a 1.0, b 1.5, s0 2, δ 4; MOBIL p 0.5, b_safe 4, threshold
0.1, keep-right bias 0.3; merge zone/b_safe ramp unchanged). Note
`param_set_id` (sha256 of the file) changes accordingly. The guards
remain available as the labelled `v2g_guards_on` comparison in the A2
eval.

## 6. Post-fix evidence (same window, same parameters otherwise)

`w8538m2600p40t360`, w = 870:

| quantity | pre-fix | post-fix |
|---|---|---|
| disc LCs in the first 10 s | 251 (ping-pong at the 2 s cap) | 11 (all one-time per vehicle) |
| collisions in the first 10 s | 51 | 0 |
| disc LCs over the 120 s window | 2325 (+1 merge; pre-fix engine replayed from commit 1f657ff) | 35 (+1 merge) |
| collisions over the 120 s window | 53 | 0 |
| decision gain magnitudes | 1e15 … 1e16 | O(1) (e.g. gain 1.646 vs eff 0.4) |
| engine anomalies | accel-end clamps | none |

Regression scenes now pin the fix
(`partA_v2g/tests/test_engine_lc_regression.py`): uniform two-lane
traffic produces ZERO disc LCs over 120 s with guards OFF; no vehicle
returns after a change without any cooldown; simultaneous target-lane
entries never overlap; `_neighbors` agrees with the lane order;
decision accelerations stay within `a_clip` in an overlap state.

## 7. What this does and does not claim

- The churn bug is fixed; the residual over-production of discretionary
  LCs relative to SUMO truth (a model-error question of textbook MOBIL
  under 0.5 s ballistic integration, order ~5-10x on the smoke window)
  is reported honestly in REPORT_PA2G_EVAL_A2.md — it is not addressed
  by tuning (registration rule).
- The junction-row mapping convention (a junction vehicle's model s is
  the next segment's start) can stack vehicles in the initial state;
  it contributes few vehicles and is NOT the churn driver (the churner
  above had a healthy 76 m real gap; the poisoning came from the swap).
