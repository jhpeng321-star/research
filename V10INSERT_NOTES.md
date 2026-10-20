# V10INSERT: learned entrance insertion speed / acceptance (2026-10-20)

Branch `c4/v10insert` from `c4/v10int` (`072fafda`, containing `26378490`). Inference and
fitting only: laptop CPU, <= 3 workers, no GPU, no SUMO. TRAIN-val only; evaluation seeds
964000-964199 were refused; no `sealed/` or holdout-named file was opened.

## 0. What changed

The WM entry process inserted mainline entrants at ~1.1 m/s and ramp entrants ~5 m/s below
SUMO (`ENTRY_DIAG2.md` section 2), and its hand-set `sumo_like_v2` admission did not use the
observable entrance state. V10INSERT replaces that decision with two networks fitted to
**recorded SUMO entrants**:

* **speed** - `v0` from (bumper gap to the rearmost vehicle in the entry lane, its speed,
  backlog, demand rate, lane cap, entrance);
* **hazard** - per-step probability that a *waiting* entrant is admitted now, from the same
  state, with one additive logit offset per entrance calibrated to SUMO's pooled insertion
  rate.

A fresh arrival (zero backlog) still enters on its renewal event; a waiting entrant is
admitted by the hazard. One learned decision per entrance per step, FIFO otherwise. The
fitted entrant is placed on the entrance lane with the largest gap at the learned `v0`,
subject to physical non-overlap and SUMO's insertion-safety test.

Switch: `physics.entry_learned` (default `false` = the pre-V10INSERT `sumo_like_v2`
behaviour, bit-identical). It needs `entries` + `entries_v2` (queue mode is the intended
use). The fit is `src/wms/model/entry_learned.json`; refit with
`python tools/entry_learned_fit.py`.

## 1. Fit (TRAIN-val `merge_calib/data/{v5_native,genrand_plain}`)

192 episodes (96 + 96), 150,362 insertion events and 415,603 at-risk entrance-steps
(backlog > 0). Episode-grouped 80/20 fit/validation split by seed hash (no start leaks).

| metric | fit | validation |
|---|---|---|
| speed MAE (m/s) | 0.520 | **0.583** |
| mainline v0 p50: SUMO / model (m/s) | 0.184 / 0.168 | 0.188 / 0.175 |
| ramp v0 p50: SUMO / model (m/s) | 21.92 / 22.12 | 21.79 / 22.12 |
| mainline admission rate: SUMO / model | 0.2504 / 0.2504 | 0.2505 / 0.2511 |
| ramp admission rate: SUMO / model | 0.0909 / 0.0909 | 0.0864 / 0.0896 |
| hazard Brier | 0.0641 | 0.0662 |

The rates are per 0.5 s step over the at-risk rows; the per-entrance offsets are
`[+0.068, -0.072]` logit. `_out/entry_learned.metrics.json` has the full report.

## 2. Before/after on 8 TRAIN-val starts

Protocol of `entry_diag2.py`: 8 starts, `t0 = 300/600/900 s`, native + generic_random,
demand 1900-2700 vph, penetration 0.05-0.40. Indices (delay_states.json):
`1,5,9,13` (`v5_native`) and `25,29,33,37` (`genrand_plain`); seeds
963019, 963109, 963178, 963274, 963497, 963584, 963643, 963724. Horizons 60/120 s
(the 300 s horizon was not run: one 300 s rollout costs ~11 min on this laptop).
`OFF` = `entry_learned=false`, `ON` = `entry_learned=true`; both use entries v2 + queue.

### 2.1 Insertion rate per entrance and time bin (mean over 8 starts, veh/s)

| entrance | bin (s) | SUMO | OFF | ON |
|---|---|---|---|---|
| mainline | 0-30 | 0.537 | 0.604 | **0.521** |
| mainline | 30-60 | 0.550 | 0.604 | **0.513** |
| mainline | 60-120 | 0.538 | 0.569 | **0.435** |
| ramp | 0-30 | 0.150 | 0.083 | 0.079 |
| ramp | 30-60 | 0.079 | 0.088 | 0.100 |
| ramp | 60-120 | 0.115 | 0.121 | 0.100 |

The mainline over-service of the hand-set rule (+6-13 %) is removed in the first minute
and over-corrected by ~19 % in 60-120 s. Ramp demand is 0.10-0.15 veh/s, so the 30 s ramp
bins are noisy (2-4 entrants per start); pooled 0-120 s is SUMO 0.115, OFF 0.103, ON 0.095.

### 2.2 Entrant initial state (mean of per-start medians)

| stream | quantity | SUMO | OFF | ON |
|---|---|---|---|---|
| mainline | v0 (m/s) | 3.78 | 1.08 | **3.87** |
| mainline | a(0-10 s) (m/s^2) | 0.92 | 1.26 | **1.02** |
| ramp | v0 (m/s) | 21.76 | 20.85 | **22.87** |
| ramp | a(0-10 s) (m/s^2) | -0.18 | 0.02 | **-0.05** |

The mainline v0 deficit (1.08 vs 3.78) is closed (3.87). The WM still accelerates
mainline entrants ~11 % harder than SUMO, and the ramp entrant now departs at the lane cap
instead of ~5 m/s too slowly, so its first 10 s deceleration is closer to SUMO's.

### 2.3 ENTRY_DIAG2 exposure / deficit table

Exposure = 0.5 s x vehicle-steps in the region; deficit = delay / exposure;
delay = `sum_t sum_v present * relu(1 - v/31.29)`. Pooled over the 8 starts (veh-s).
Regions as in `ENTRY_DIAG2.md` (`up_near` mainline [-800,0), `merge` mainline [0,75.44),
`ramp` lane 2 [-378.66,75.44)).

| H | region | SUMO exp | OFF exp | ON exp | SUMO delay | OFF delay | ON delay | SUMO def | OFF def | ON def |
|---|---|---|---|---|---|---|---|---|---|---|
| 60 | up_near | 3957 | 3998 | 3998 | 1495 | 1574 | 1574 | 0.378 | 0.394 | 0.394 |
| 60 | ramp | 1718 | 1564 | 1530 | 2659 | 2518 | 2413 | 1.548 | 1.610 | **1.577** |
| 60 | merge | 1008 | 876 | 887 | 1301 | 1102 | 1122 | 1.290 | 1.259 | 1.265 |
| 120 | up_near | 8242 | 8796 | 8514 | 4009 | 4148 | 4066 | 0.486 | 0.472 | 0.478 |
| 120 | ramp | 3032 | 3202 | 2976 | 4506 | 5081 | 4636 | 1.486 | 1.586 | **1.558** |
| 120 | merge | 2329 | 1661 | 1642 | 3283 | 2066 | 2055 | 1.410 | 1.244 | 1.252 |

The entry fix moves the ramp exposure/deficit towards SUMO at both horizons (ramp delay
-4.2 % at 60 s and -8.8 % at 120 s versus OFF). Upstream/merge are not entry-limited and
stay dominated by the WM's missing queue/spillback (the ENTRY_DIAG2 section 5 diagnosis);
the learned entrance process cannot create a merge bottleneck by itself.

## 3. Tests

`tests/unit/test_v10insert.py` (6 tests): switch default/validation, shipped fit load and
gap response, feature finiteness, learned rejection at a blocked entrance, learned
free-flow `v0` / one-per-step, and the queue-mode plumbing. Full run
`pytest tests/unit/test_v10insert.py tests/unit/test_v9entry.py
tests/unit/test_v9entry_eval.py tests/unit/test_entryfix.py tests/unit/test_v6entry.py
tests/unit/test_c4_dreamer.py tests/unit/test_import_graph.py -q`:
**284 passed, 1 skipped** (recorded v5 policy episode absent on this host).
`ruff check` clean on every changed file.

## 4. Limits / next action

* The hazard was fitted on recorded SUMO states; in the WM's own road the gap distribution
  differs, so closed-loop mainline service drifts below SUMO after 60 s (0.435 vs 0.538).
  A closed-loop hazard calibration (per time bin) is the next step if the source wait
  remains off.
* The ramp bins are demand-noise dominated (0.10-0.15 veh/s); a larger start sample or a
  longer horizon is needed for a firm ramp rate statement.
* The upstream/merge deficit is not addressed here; it needs the queue/spillback and
  merge-capacity supervision listed in `ENTRY_DIAG2.md` section 6.
