# A40 variant choice (written 2026-11-10 evening, BEFORE any controller arm runs on any variant; zero SUMO so far)

Authority: REGISTRATION.md AMENDMENT 40 (feasibility + build only). No controller outcome on any variant exists or
was read. The inputs to this choice are code constraints, frozen plan parameters and the wz3to2 sources only.

## Base road (unchanged, never modified)
wz3to2 (assets/research/newmap_candidates/wz3to2, builder runs/active/C4-V7FT-20261017/newmap_probe/build_maps.py):
edge `up` 0..2700 m, 3 lanes, 31.29 m/s, SUMO lane 0 ends at x = 2696 (drop X_B); edge `wz` 2700..4000 m, 2 lanes,
31.29 m/s; connections up_1 -> wz_0, up_2 -> wz_1. The builder re-run with netconvert 1.27.1 on the laptop reproduces
the committed wz3to2.net.xml body exactly (only the generated-on header differs; E:/agent-work/a40/regen_check).

## What the frozen code fixes (facts used by the rule)
- Executor (oracle_plan; LEARNED_TRIGGER and DUO_HOLD run on it): positions come in as the descriptor's drop-relative
  `s` and are converted with the constant S_OFF = 2696 - 75.44 (to_sumo_row). Zones, field window and release are
  therefore relative to the drop for ANY approach length. Plan zone 'full' = 400..1800 m upstream of the drop.
- Frozen hold speeds: bank fd385a2b templates v_w in {18, 19, 20, 21, 22} m/s; duo12_v20 v_w = 20 m/s. Max = 22.0 m/s.
- U0 (MAPPO d9042cc5, wms.control.obs): own features include s_to_merge_end_m = 75.44 - s scaled by 400 m and
  speeds scaled by 31.29, NOT clipped. On wz3to2, s_to_merge_end at insertion is 2696 m (6.74 scaled).
- LEARNED_TRIGGER classifier 8e0ddd3e (coopnative.encode): holder own s/1000, v/30, lane one-hot, six nearest
  vehicles within 100 m (ds/100, v/30); the holder is always inside the plan zone when the gate is evaluated.
- map_descriptor.load accepts only: one straight 3-lane approach edge whose rightmost lane ends without connection,
  through lanes SUMO 1/2, and ONE approach->downstream junction (no second junction, no re-opening lane).
- Team delay J = dt*max(0, 1 - v/31.29) + source waiting (measurement convention; not a loading constraint).

## Rule (fixed here, uses no controller outcome)
Candidates = single-parameter changes of the wz3to2 builder inputs (one named change, so the transfer is
attributable). Each candidate is pushed to its admissible extreme; the variant is the candidate with the LARGEST
relative change |p' - p| / p. Admissible = all of
 (i)  the unchanged code loads it: map_descriptor.load (with a registry entry only), oracle_plan, LEARNED_TRIGGER,
      DUO_HOLD, U0 observation code and nm_finaleval runner, with every frozen observation input inside the range it
      already takes on wz3to2 (no extrapolation of an unclipped feature);
 (ii) plan zones (400..1800 m upstream of X_B) lie inside the approach with >= 400 m (= the zone's own near margin)
      between the insertion point and the far zone edge, and every frozen hold speed (<= 22.0 m/s) stays <= the
      posted limit everywhere a held vehicle can be, so a "hold" remains a slowdown below the limit;
 (iii) SUMO stable: 0 collisions and 0 teleports in the native-base smoke (checked after the build; if it fails,
      the next-largest candidate is taken by the same rule).
Values are taken on a natural grid (lengths in 100 m, speed limits in 10 km/h steps).

| candidate | admissible extreme | relative change | status |
|---|---|---|---|
| approach length (X_B) | longer: excluded by (i) (s_to_merge_end beyond 2696 m is outside the U0 input range seen on wz3to2; unclipped). shorter: 1800 + 400 = 2200 m approach (drop at 2196) | 500/2700 = 18.5% | admissible |
| work-zone (downstream) length | any length loads, but the downstream edge is an uninterrupted 2-lane sink: its length changes only free-flow travel time after the drop, not the bottleneck; a closure that ENDS (2 lanes -> 3 lanes again) needs a second junction, refused by map_descriptor.load | n/a | excluded by (i) for a real closure length; plain length = no geometric change at the bottleneck |
| taper (gradual closure) | SUMO lane end is abrupt; a taper needs a non-straight lane shape or an extra edge/junction, refused by map_descriptor.load (_straight_x, single junction) | n/a | excluded by (i) |
| work-zone speed limit (edge `wz`) | lowest 10 km/h step >= max frozen hold speed 22.0 m/s (79.2 km/h) -> 80 km/h = 22.22 m/s (raising above 31.29 excluded by (i): speeds above the 31.29 normaliser are outside the range U0 sees) | (31.29 - 22.22)/31.29 = 29.0% | admissible, LARGEST |

## Choice
**wz3to2v80**: identical to wz3to2 except the downstream work-zone edge `wz` (2 lanes, 2700..4000 m) is posted at
22.22 m/s (80 km/h) instead of 31.29 m/s. Approach (3 lanes, 31.29 m/s, drop at x = 2696), lane drop, connections,
work-zone length and route are byte-for-byte the same builder inputs. 80 km/h is a common posted work-zone limit.

Physical consequences (expected from construction, not measured): vehicles must decelerate from 31.29 to 22.22 m/s
at the drop while merging, so merge gaps are accepted at lower downstream speed and the 2-lane discharge capacity
falls (IDM-type flow ~ v/(s0 + vT + l) drops by roughly 5-10% at 22 vs 31 m/s). The bottleneck capacity is NOT
unchanged by construction -> a demand calibration step is proposed (PROGRESS.md / registration draft).

In-range checks for the frozen inputs (by construction):
- oracle_plan / LT / DUO zones, field window (454 m before the drop), release (s >= 75.44): identical numbers.
- U0 s_to_merge_end_m, s_m: identical range (approach unchanged). Speeds: within [0, 31.29], i.e. inside the
  normaliser range; downstream speeds become lower on average (a distribution shift, not extrapolation).
- LT classifier: holder is upstream in the zone; own/neighbour s, ds, v ranges unchanged; neighbour speeds near the
  zone can be lower when queues are longer (in the range seen under congestion on wz3to2).
- NOT observed by any frozen controller: the posted limit itself. U0 commands accelerations without knowing the
  wz limit; DUO/LT release into a 22.22 m/s zone. This is part of what the transfer test measures.
- J keeps the registered 31.29 reference speed, so every vehicle-second inside the work zone adds >= 0.29 s of
  "delay" even in free flow (same in every arm; paired contrasts are unaffected by the constant part, but % changes
  are relative to a larger base). Disclosed.

Weakness (stated up front): this is ONE variant of ONE road family, a regulatory/speed change of the merge-into
section rather than a new layout; it is exploratory transfer evidence, not a second road.
