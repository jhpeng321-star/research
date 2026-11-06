# PREP_A — merge context tracker (`src/wms/model/merge_context.py`)

Pure, parameter-free torch feature computation for ramp-car context and its lane-1
followers.  Nothing is wired into the model or training; the next round can call
`recorded_context` offline (agent B) and `MergeContextTracker.step` step-wise inside
rollouts.  Frame follows `UnrollTargets.states`: `s` = front-bumper position (m, merge
zone `[0, 75.44]`, upstream negative), `v` (m/s), `a` (m/s^2), `length` (m); ramp lane
2, target lane 1.  CAVs are ordinary neighbours everywhere.

## API

```python
CONTEXT_FEATURES: tuple[str, ...]
MergeContextTracker(batch, k, *, dt=0.5, device=None, dtype=None)
  .reset_slots(new_car_mask)        # [B,K] bool: identity change / slot reuse
  .step(s, lane, v, a, present, is_cav, length)   # all [B,K] -> {name: [B,K]}
  .state_dict() / .load_state_dict(d) / .detach_()
recorded_context(veh_kin, present, is_cav, *, dt=0.5)  # [B,N,K,7] -> {name: [B,N,K]}
```

`lane` is the integer lane index (`-1`/absent when unknown).  `state_dict` returns
detached clones; `detach_` cuts the autograd graph of the carried state at TBPTT
boundaries.  Device/dtype are taken from the first `step` tensors (or the
`recorded_context` kinematics) and state is moved automatically.

## Features (ordered; `_valid` bool masks where a value can be missing)

| feature | unit | definition |
|---|---|---|
| `n_ramp_ahead` | count | present lane-2 vehicles with `s_j > s_i` (strictly ahead, any distance) |
| `ramp_lead_gap_m` | m | nearest lane-2 leader within 150 m: `s_lead - s_i - length_lead`, clamped `>= 0`; valid if a leader exists |
| `ramp_lead_dv_mps` | m/s | `v_lead - v_i`; same valid mask as the lead |
| `dist_to_lane_end_m` | m | `75.44 - s_i` for ramp slots (raw, may be negative) |
| `wait_s` | s | time since this car entered the zone: 0 on the first in-zone step, accumulates `dt` afterwards; valid iff ramp and in zone |
| `t_since_ramp_insertion_s` | s | world zipper timer: 0 on any ramp→lane-1 transition, `+dt` otherwise; valid after the first insertion |
| `t_since_lane1_passage_s` | s | world zipper timer: 0 when a lane-1 car crosses `s = 75.44`, `+dt` otherwise; valid after the first passage |
| `last_service_was_ramp` | 0/1 | 1 on a ramp insertion, 0 on a lane-1 passage (insertion wins if both in one step); valid after either event |
| `foll_speed_mps` | m/s | speed of the nearest lane-1 vehicle behind (`s_j <= s_i`, max 150 m); valid iff a follower exists |
| `foll_accel_now` | m/s^2 | that follower's current acceleration |
| `foll_accel_ema1_s` | m/s^2 | exponential EMA of the follower acceleration, tau = 1 s |
| `foll_accel_ema3_s` | m/s^2 | exponential EMA of the follower acceleration, tau = 3 s |
| `foll_dv_mps` | m/s | `v_follower - v_i` |
| `n_ramp_same_gap` | count | other present ramp cars whose lane-1 leader **and** follower slot identities equal this car's; 0 when neither side exists |

EMA update: `ema <- ema + (a_f - ema) * (1 - exp(-dt/tau))`; the first observation of
a follower (and the first step after a slot reset) sets `ema = a_f`.  The EMA is keyed
by the follower's slot index, so a new follower restarts it.

Zipper timers are world-level (per batch row) and are broadcast to every present slot.
Insertion = present in both steps with lane `2 -> 1`; passage = present in both steps,
lane 1 in both, and `prev_s < 75.44 <= s`.  Lane changes and crossings are therefore
detected by geometry, not by a tuned rule.

`n_ramp_same_gap` uses the lane-1 leader/follower **pair**, both associated within
`MAX_NEIGHBOUR_GAP_M = 150 m`; a missing side is key `-1`.

Masks emitted: `ramp_lead_gap_m`, `ramp_lead_dv_mps`, `wait_s`,
`t_since_ramp_insertion_s`, `t_since_lane1_passage_s`, `last_service_was_ramp`,
`foll_speed_mps`, `foll_accel_now`, `foll_accel_ema1_s`, `foll_accel_ema3_s`,
`foll_dv_mps` (each `<name>_valid`).  No mask is needed for `n_ramp_ahead`,
`dist_to_lane_end_m`, `n_ramp_same_gap`: they are counts/distances defined for every
ramp slot.

## Edge cases

* Non-ramp and absent slots: all ramp/follower features are 0; valid masks false.
* `present=False` slots may carry garbage kinematics; values are sanitised with the
  presence mask before any pairwise comparison.
* A ramp car appearing directly inside the zone starts `wait_s = 0` (entry step).
* `s = 75.44` is inside the zone (`wait_s` accumulates); a crossing is detected only
  on the following step (`prev_s < 75.44 <= s`).
* Equal `s` is not "ahead" for `n_ramp_ahead`/leaders; it can be a follower
  (`back >= 0`), matching `merge_core`'s gap convention.
* Two ramp cars within the same lane-1 gap get `n_ramp_same_gap = 1` each (self
  excluded).
* `reset_slots` clears per-car state (`wait_s`, previous lane/position/presence,
  follower index, both EMAs) so a reused slot is treated as a new car; world zipper
  timers are intentionally kept.
* `batch == 0`, `k == 0` and `N == 0` return correctly shaped zero outputs.

## Tests / checks

`tests/test_merge_context.py` covers: feature names, a 3-car ramp queue
(`n_ramp_ahead` 0/1/2 and leader gaps/dv), follower EMA on a known acceleration series
(tau 1 s/3 s), `wait_s` across entry, zipper timers and `last_service_was_ramp` after a
ramp insertion and a lane-1 passage, `n_ramp_same_gap` for two ramp cars sharing a gap,
missing neighbours (`valid=False`), `recorded_context` versus stepping the tracker,
slot-reuse reset, batch and K padding vectorisation, state-dict round-trip, graph
detach, float32/float64, and the neighbour association range.

```
C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest tests/test_merge_context.py -q
C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m ruff check src/wms/model/merge_context.py tests/test_merge_context.py
```

15 tests pass; ruff clean.

## Non-goals

No learned parameters, no hand-set behavioural thresholds or rules, no model wiring,
no driver truth tables, and no use of sealed/holdout paths.
