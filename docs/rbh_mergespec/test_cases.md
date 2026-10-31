# Test cases — SUMO 1.27.1 merge / lane-end / insertion spec (rbh-mergespec)

All vectors are produced/verified by `mergespec_tests.py` (`python mergespec_tests.py`, stdlib only, 38/38 pass).
Purpose: these pin the *formulas* a rebuild implementation (e.g. `src/wms/physics/junction_merge.py`) must reproduce, and give
numeric acceptance bounds. They are unit-level reference checks, not SUMO runs.

Conventions: IDM defaults a=2.6, b=4.5, tau=1.0, minGap=2.5, delta=4, 4 substeps (MSCFModel_IDM.cpp:38-51);
Euler update, TS=1 s; lengthWithGap = 7.5 m; URGENCY=2.0 etc. (spec.md §1).

| # | Area | Scenario (inputs) | Expected (reference) | Pins source |
|---|------|-------------------|----------------------|-------------|
| T1 | CF | `brakeGap(10, 4.5, 0)` Euler | **6.5 m** (n=2 steps: 2·10 − 4.5·3) | MSCFModel.cpp:101-108 |
| T2 | CF | IDM secureGap (10 vs 10); (10 vs 5) | **10.0 m** (=tau·v); **17.309 m** (=10 + 50/(2√(2.6·4.5))) | MSCFModel_IDM.cpp:217-221 |
| T3 | insertion | leader 5 m/s, ego wants 10; net gap ≥ s ⇒ keep speed; gap=2 m | keep **10** at gap≥s; at 2 m: **v=1.264**, stable fixed point | MSCFModel_IDM.cpp:137-162 |
| T4 | insertion/stop | stopSpeed at gap 0.005 with v=5; clamp at gap 1, v=30 | **0**; result ≤ **gap/TS** | MSCFModel_IDM.cpp:178-200 |
| T5 | lane-end | laDist boundary v=13.89, defaults | urgent iff usableDist < **153.9 m** (right); left boundary **292.8 m**; 2 changes ⇒ usable/2 vs laDist | MSLCM_LC2013.cpp:1244,1330-1333, h:197-199 |
| T6 | lane-end | remainingSeconds(100 m, v_look=10, 1 lane) | **5.0 s** = 100/10/1/2.0; floor 1 step; 2 lanes ⇒ 2.5 | MSLCM_LC2013.cpp:1502-1507 |
| T7 | lane-end | lookAheadSpeed: grow 5→8; decay 8→5 | instant **8**; decay **7.7** = 0.9·8+0.1·5 | MSLCM_LC2013.cpp:1235-1243 |
| T8 | insertion | follower gap acceptance, equal speeds 10 | pass at gap = **10** (tau·v), fail at 9.5 | MSLane.cpp:1228-1254 + IDM secureGap |
| T9 | merge | unsafeMergeSpeeds (10 lead, 5 follow) vs (5 lead, 10 follow) | safe: **false**; unsafe: **true** (25/4.5 ≤ 100/4.5) | MSLink.h:730-733 |
| T10 | merge window | foe left 0.5 s ago, same target lane | blocked (1 s lookahead); cross-merge target lane → **not** blocked by trailing gap; foe arrives +8 s after ego leaves → open unless unsafe speeds; overlap → hard conflict; zipper stretches follower window to **16 s** | MSLink.cpp:1175-1219, 72-74 |
| T11 | zipper | ego 25 m/s, 30 m from link; foe 25 m/s, 20 m, arrives 1 s later | trim to **≈16.9 m/s**; foe behind (40 m) that can brake → ignored (27.78); dist 200 > max(visibility, brakeGap) → untouched; tie ⇒ right lane ego untouched, left ego trims | MSLink.cpp:2157-2304 |
| T12 | zipper | avoidArrivalAccel(100 m, 10 s, v=15); (10 m, 10 s, v=15) | **−1.0** = 2(100/10−15)/10; **−11.25** = −v²/2d | MSCFModel.cpp:553-567 |
| T13 | lane-end | dead-end stop trajectory v0=10, decel 4.5, 50 m to end | cruise to ~20 m, then stopSpeed ramp (10→7.80→5.05→…), halts before the end | MSVehicle.cpp:2755-2767 + IDM stopSpeed |
| T14 | insertion | getExtraReservation(±2, near) / (±1) / (2, far) | **40 / 20 / 0 / 0** | MSAbstractLaneChangeModel.cpp:1178-1188 |

Additional (documented in spec.md, not unit-tested here because they are policy/discrete logic — assert via scenario replay once the rebuild exists):
- E1: `departLane=best/free` lane choice rules (MSEdge.cpp:547-613, 677-751) — largest gap past departPos, else least occupancy; best lanes capped at 3000 m continuation.
- E2: `departSpeed=max` vs `desired` under a near leader (MSLane.cpp:818-847): max → insertion proceeds at lowered speed; desired → departure delayed (patchSpeed=false).
- E3: insertion near minor/zipper junction requires stop-ability (`insertionStopSpeed`) and yields to committed foes (`hasApproachingFoe`) (MSLane.cpp:1052-1090, 1207-1216).
- E4: courtesy blocker reservation (myLeadingBlockerLength) braking to `stopSpeed(leftSpace − blocker − eps)` (MSLCM_LC2013.cpp:247-281).
- E5: `opened()` priority short-circuit does NOT apply to zipper links (MSLink.cpp:935-943).

Acceptance for the rebuilt WM merge stack: reproduce T1-T14 within 1e-3 (deterministic formulas) and match E1-E5 decision-for-decision on the replay scenarios supplied by S-RB-EVAL (targets/ folder).
