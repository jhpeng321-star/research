# ENTRYFIX: why imagined entrance queues are too large, and a switch-guarded fix (2026-10-14)

Scope: code + tests only (no SUMO, no GPU, no training). Branch `c4/entryfix` from `c4/int` 0be5b72.
Default behaviour is unchanged (`insert_rule = "v2"`, verified bit-identical, see Verification).

## Diagnosis

SUMO inserts this network's vehicles with `departLane="best" departSpeed="max"` and default
`departPos="base"` (`traffic_shepherd/scenarios.py`). With `departSpeed="max"` SUMO starts from the
lane limit x speedFactor and LOWERS the speed to the safe insertion speed behind the lane's rearmost
vehicle; insertion fails only if no speed >= 0 is safe, i.e. the bumper gap is below minGap.
Recorded episodes confirm it: mainline entrants enter at median 0.46 m/s (95th pct 31.29 = the lane
limit) when the entrance is queued.

`TorchEntryProcessV2._try_lane` (rule "v2") departs from this in four ways:

1. **Sampled required gap (the main cause).** Each arrival also needs `gap >= u_gap`,
   `u_gap ~ N(gap_mean, gap_std)`, the *realised* entry-gap distribution. A realised gap is an
   outcome (>= the minimum SUMO accepted), not a threshold, so using it as one refuses most
   insertions SUMO actually makes. Pack features give gap_mean ~ 34-61 m, gap_std ~ 28-95 m.
2. **No speed reduction.** `v_new = speed_ratio x local speed` is fixed and the gap test runs at that
   speed (`min_gap + headway*v + closing brake`); SUMO lowers the speed until the gap fits.
3. **Entry position 5.1 m too far upstream.** Entrants are placed with the FRONT at the lane start.
   SUMO puts the front at lane start + length + 0.1 m (recorded first positions are exactly
   `ENTRY_S_M + 5.10`), so v2 sees every entrance gap 5.1 m too small.
4. (Feature side, not fixed here; affects v2 only.) `wms.data.pack_history.entry_features` counts
   every vehicle present at the FIRST history tick as an entrant (`prev_same` is all-false at
   j = 0). For example, n_obs = 141 mainline "entries" in a 60 s history whose counter rate is
   0.66 veh/s. That puts mid-road vehicles into gap_mean / speed_ratio (speed ratio 2.6 +/- 2.0).
   The rate comes from the counter and is not affected; sumo_like ignores gap_mean and
   speed_ratio, so this matters for v2 only. Fixing it would change data features, so it needs its
   own switch in the data owner's branch.

Ruled out as causes: multi-insert per step (v2 already keeps inserting until the first refusal,
which is also what SUMO does per departure edge); units; double-counted arrival rate (the rate is
the recency-weighted 10 s counter of SUMO *insertions*).

## Fix: `EntryConfig.insert_rule = "sumo_like"` (default "v2")

`src/wms/model/entries.py`: the speed is `min(lane limit x speedFactor, SUMO safe insertion speed)`.
The safe speed is the largest v with `tau*v + v^2/(2b) <= (gap - minGap) + v_lead^2/(2b)`, using
SUMO vType tau/minGap/decel per vehicle class. The arrival is refused only when `gap < minGap`. The
front is placed at lane start + length + 0.1. Mainline arrivals try both entrance lanes,
largest gap first (`sumo_lane_mode="any"`, the default). The `"best_free"` mode, which only tries
the lane with the smallest vehicle-length sum, matched SUMO worse (table below). FIFO arrivals keep
inserting within a step until the first refusal. The random draws are the same as in v2, so both
rules see the same arrival stream for a given seed. Arrival rates, CAV share and vehicle type are
unchanged.

One key per consumer, default unchanged:
- WM training: `data.entry_insert_rule = "sumo_like"` (flat key `entry_insert_rule`, omitted from
  legacy flat/receipts while it is "v2") -> `wms.train.inputs.make_entry_process`.
- Dreamer policy: `trainer.entry_insert_rule = "sumo_like"` -> `dreamer._entry_process`.
- Decision check / scripts: `EntryConfig(max_pending=10**9, insert_rule="sumo_like")`.

Caveat: `wms.control.srcgrad` (the `source_wait_grad` surrogate, default off) is still the soft
version of the v2 test. With `source_wait_grad=True` and sumo_like, the reward VALUE is exact
(straight-through on the real backlog) but the gradient shape follows v2. Update srcgrad before
using both together.

## Evidence (recorded TRAIN-side C4 episodes, no SUMO run)

Episodes: GENRAND plain s4107 (p0.2), C4-SUMO-SPEED s4107/s4108 x p0.2/p0.4 ("random" CAVs),
C4-SUMO-SMOKE, C4-MAPPO2-DEPLOY, c4_dev b1_native (native, 2600 veh/h). All m2700 except b1.
Scripts: `E:/agent-work/tmp/entryfix/{accept,replay}.py`; results in `a3_*.json` and `r_*.json` there.

### A. Insertion rule on SUMO's own recorded states (no road model, no confound)

*True accept*: share of SUMO's actual insertions that the rule admits on the state SUMO saw (the
recorded tick minus that tick's entrants). *False accept*: share of ticks where an eligible vehicle
was waiting and SUMO inserted none, but the rule would insert. v2 is Monte Carlo over its draws,
with the pack features at the time.

| episode | SUMO inserts / refusal ticks | v2 true / false | sumo_like (any) true / false | best_free true / false |
|---|---|---|---|---|
| b1_native (native) | 81 / 1 | 0.54 / 0.05 | **1.00 / 0.00** | 0.96 / 0.00 |
| s4108 p0.2 | 142 / 66 | 0.50 / 0.15 | **0.99 / 0.05** | 0.92 / 0.05 |
| SMOKE s4107 p0.2 | 252 / 193 | 0.20 / 0.03 | **0.99 / 0.02** | 0.71 / 0.01 |
| MAPPO2-DEPLOY | 224 / 143 | 0.19 / 0.02 | **0.99 / 0.02** | 0.72 / 0.01 |
| GENRAND s4107 p0.2 | 293 / 409 | 0.24 / 0.17 | **0.99 / 0.02** | 0.72 / 0.02 |
| s4107 p0.4 | 88 / 311 | 0.25 / 0.06 | **1.00 / 0.01** | 0.65 / 0.01 |

v2 refuses 46-81% of the insertions SUMO actually made. sumo_like admits 99-100% of them and wrongly
admits at only 0-5% of SUMO's refusal ticks.

### B. 120 s replay: process inserting into teacher-forced recorded roads

Roads: t0 vehicles follow their recorded trajectories. SUMO's post-t0 entrants are replaced by the
process's entrants, which drive a SUMO-parameter IDM. *feat* = renewal arrivals from the pack
entry_features (the WM/Dreamer pipeline; starts with an empty queue). *oracle* = SUMO's recorded
request times plus SUMO's queue at t0 (isolates the rule). Source wait is in veh-s. Delay p50 is
the median time from request to insertion (oracle mode). The replay has a structural 0.5 s offset:
it inserts in the request's own step, while SUMO inserts in the next step. That is worth about
rate x 0.5 s x 120 s, roughly 45 veh-s per window; it is all of b1's SUMO wait.

| episode | t0 | SUMO wait | v2 feat | sumo_like feat | v2 oracle | sumo_like oracle | delay p50 SUMO / v2 / sumo_like |
|---|---|---|---|---|---|---|---|
| b1_native | 30 | 47 | 22 | 0 | 76 | 0 | 0.7 / 0.7 / 0.2 |
| b1_native | 60 | 44 | 73 | 0 | 148 | 0 | 0.7 / 1.2 / 0.2 |
| s4108 p0.4 | 120 | 60 | 566 | 0 | 1582 | 2 | 0.8 / 13.9 / 0.3 |
| s4108 p0.2 | 120 | 102 | 566 | 0 | 1582 | 2 | 0.8 / 13.9 / 0.3 |
| s4108 p0.4 | 150 | 354 | 302 | 0 | 705 | 3 | 0.8 / 6.5 / 0.3 |
| s4108 p0.2 | 150 | 562 | 516 | 0 | 606 | 2 | 0.8 / 5.7 / 0.3 |
| s4108 p0.4 | 180 | 1178 | 1276 | 0 | 1618 | 307 | 0.9 / 10.5 / 0.3 |
| s4108 p0.2 | 180 | 1490 | 922 | 0 | 1633 | 208 | 0.9 / 12.1 / 0.3 |
| s4107 p0.2 | 120 | 1006 | 1752 | 0 | 3131 | 1 | 0.9 / 14.2 / 0.3 |
| s4107 p0.2 | 150 | 1896 | 1241 | 0 | 2704 | 1 | 4.3 / 27.1 / 0.3 |
| SMOKE | 180 | 670 | 2067 | 1 | 1258 | 28 | 6.3 / 11.9 / 0.3 |
| SMOKE | 240 | 1012 | 2804 | 0 | 3312 | 34 | 7.8 / 26.8 / 0.3 |
| SMOKE | 300 | 3260 | 2592 | 0 | 3088 | 72 | 11.9 / 25.7 / 0.3 |
| GENRAND | 300 | 2958 | 2518 | 0 | 2977 | 72 | 8.4 / 22.1 / 0.3 |
| s4107 p0.2 | 180 | 3345 | 3437 | 768 | 3344 | 1930 | 17.4 / 35.0 / 20.0 |
| s4107 p0.4 | 120 | 3762 | 1696 | 0 | 2952 | 1 | 1.0 / 15.1 / 0.3 |
| s4107 p0.4 | 150 | 6296 | 2960 | 1556 | 4609 | 3042 | 49.8 / 42.8 / 13.8 |
| GENRAND | 360 | 6764 | 2404 | 299 | 5217 | 3419 | 72.1 / 58.5 / 34.9 |
| s4107 p0.4 | 180 | 9110 | 122 | 52 | 6676 | 5523 | 80.7 / 64.5 / 49.9 |
| GENRAND | 420 | 11354 | 200 | 538 | 11060 | 10747 | 79.5 / 95.7 / 76.6 |
| GENRAND | 480 | 16163 | 197 | 190 | 15741 | 14784 | 163.9 / 176.3 / 167.0 |

What the replay shows:
- **Free-entrance windows** (SUMO delay p50 < 1 s; this is the regime of the decision-check point,
  SUMO 172 veh-s). v2 builds a queue of its own: 6-27 s median delay against SUMO's 0.8 s. For
  example, s4108 p0.4 at t0 = 120: SUMO 60 veh-s, v2 566 (feat, 9.4x) and 1582 (oracle, 26x). That
  matches the decision-check 9x. sumo_like inserts on arrival; its 0.3 s delay plus the 0.5 s step
  offset equals SUMO's 0.8 s.
- **Jammed-entrance windows** (SUMO delay p50 >= 17 s). In SUMO the queue here comes from road
  dynamics: the downstream jam and, in these "random"/generic_random episodes, CAVs driven by random
  actions near the entrance. Teacher-forced IDM roads cannot reproduce that. With oracle arrivals
  both rules land within 5-45% of SUMO. v2 gets closer only because its over-blocking happens to
  stand in for the blocking the replay roads leave out. In the WM imagination that blocking has to
  come from the WM's dynamics and the policy, not from the insertion rule.
- **Arrival side (not changed).** *feat* arrivals are the counter's insertion rate and start from an
  empty queue. The RSU cannot see SUMO's pre-t0 source queue or demand above insertion capacity, so
  in jammed windows every rule underestimates SUMO's absolute source wait. This is a known RSU
  observability limit.

### C. The decision-check point (s4125 p0.2 t360, V2best checkpoint, entries on, 120 s)

CPU re-run of `E:/agent-work/tmp/decision_check/scripts/imagine_points.py`. The only edit is
`EntryConfig(..., insert_rule=$ENTRY_INSERT_RULE)`, in the scratch copy
`E:/agent-work/tmp/entryfix/imagine_points_rule.py`, run against this worktree's src. Imagined
source wait in veh-s (the SUMO value is from the decision check):

| rule | h = 60 s (B / ALL / GROUP) | h = 120 s (B / ALL / GROUP) | entrants inserted (all rows) | SUMO 120 s |
|---|---|---|---|---|
| v2, original decision-check file | 356 / 278 / 342 | **1552** / 1182 / 1530 | 395 | 172 |
| v2, this worktree (reproduces exactly) | 356 / 278 / 342 | **1552** / 1182 / 1530 | 395 | 172 |
| sumo_like | 0.5 / 0.5 / 0.5 | **12.5** / 12.5 / 12.5 | 552 | 172 |

(Controller "random"; damping and mappo give the same B values. Under v2 their per-arm values range
from 904 to 2145, and those differences are blocking artifacts. Under sumo_like every arm gives
12.5.)

The error falls from +1380 veh-s (9x) to -160 veh-s, and the source-wait term no longer differs
across arms because of insertion artifacts. Roughly 40 veh-s of SUMO's 172 is step accounting:
SUMO's ledger charges the wait from the request time to the next 0.5 s insertion step, about
0.72 veh/s x 0.5 s x 120 s. The rest is SUMO's pre-t0 queue, demand above capacity and entrance
blocking from the road dynamics, none of which is RSU-visible to the arrival process (see B).

## Verification

- v2 parity against the pre-change module (`git show 0be5b72:src/wms/model/entries.py`): 60 random
  trials x 80 steps x 3 rows, capped and uncapped pending, shared rng and per-row seeds. Masks,
  states, pending and summaries are identical (4444 insertions).
- `tests/unit/test_entryfix.py` (12 tests): default == explicit v2, bit-identical; shared arrival
  stream; sumo_like enters at the reduced (maximal-safe) speed where v2 refuses, with the front at
  lane start + length; refuses inside minGap; a free lane enters at the speed cap; never unsafe or
  overlapping (randomised, both lane modes); backlog drains; trainer, training and data/legacy-flat
  plumbing; recorded b1_native episode (sumo_like admits >= 90% of SUMO's insertions and v2 fewer;
  skipped if the episode is absent).
- `OMP_NUM_THREADS=1 pytest tests/unit/test_entryfix.py test_wmloss.py test_c4_srcgrad.py
  test_c4_dreamer.py test_c4_core.py test_import_graph.py test_c4_srcwait_imagination.py config/
  test_r5_switches.py eval/test_eval_units.py test_c4_branching.py test_c4_loop.py
  data/test_mixdata.py test_lane_rule.py`: 547 passed. ruff is clean on the changed files.

## Recommendation for tonight's v4 WM run

Turn it on (`data.entry_insert_rule = "sumo_like"`) **if v4 trains with `data.entries_v2 = true`**.
The v2 rule refuses 46-81% of the insertions SUMO made. Entrant-dependent training signals (the
aggregate road/flow loss against SUMO's recorded full-road presence) therefore see roads emptied
behind the entrance and a growing imagined backlog. The switch is isolated (one key, default off,
bit-identical when off), but it has never run in a real training job. Under the one-fix-at-a-time
protocol it should be recorded as a separate change in the run's registration. If v4 does not use
entries_v2, the key has no effect.

For the Dreamer policy, `trainer.entry_insert_rule = "sumo_like"` removes the inflated source-wait
reward. Keep `source_wait_grad` off until srcgrad is updated (see the caveat above).
