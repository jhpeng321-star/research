# HANDOFF — rbh-mergespec (helper)

Branch `c4/rbh-mergespec` (base f0db3c82), worktree `E:/agent-work/c4-rbh-mergespec`.
Deliverable (consumable copy): `E:/agent-work/rb-shared/mergespec/` — spec.md, test_cases.md, mergespec_tests.py, DONE.md, `_sumo_src/` (35 SUMO v1_27_1 source files, flat names `src__<path>.cpp`, read-only, nothing executed).

## What exists
- `docs/rbh_mergespec/spec.md`: implementable spec of SUMO 1.27.1 insertion (departLane/departSpeed resolution, MSLane::isInsertionSuccess oracle, checkFailure patch-vs-delay), lane-end forcing (LC2013 strategic trigger laDist = 10·v_look+15 m right / 20·v_look+15 left, remainingSeconds = leftSpace/v_look/lanes/2.0, patchSpeed courtesy braking, dead-end stopSpeed), junction/zipper/priority (opened() windows 1 s / 16 s zipper, blockedByFoe follower/leader/hard-conflict rule, unsafeMergeSpeeds, getZipperSpeed full algorithm, minor-link stop offsets, point-of-no-return), CF pieces (IDM follow/stop/free/secureGap/insertionFollowSpeed/insertionStopSpeed, Krauss secureGap/brakeGap), all defaults tables, and a per-step algorithm box (§6).
- `mergespec_tests.py`: dependency-free python port of the exact formulas + 38 numeric checks (all pass). Deterministic; no SUMO.

## Switches / integration notes for S-RB-INT
- No repo code was touched; everything is new files under `docs/rbh_mergespec/`. Nothing to switch off.
- For `junction_merge.py`: the merge decision surface = (a) insertion oracle §2.5, (b) strategic urgency §3.1-3.2, (c) opened()/blockedByFoe window §4.2-4.3, (d) getZipperSpeed §4.4. All constants are in spec.md §1 (SUMO documented/derived params — allowed per owner rules).
- IDM full numerics: see sibling `rb-shared/idmspec/` (ref_idm.py, S-RB-CF); LC2013 lane-change internals: `rb-shared/lcspec/` when it lands (S-RB-LANE). This spec treats bestLanes/currentDist/bestLaneOffset as inputs (ownership noted in spec.md §7).

## Tests
- `python docs/rbh_mergespec/mergespec_tests.py` → 38/38 PASS (portable, stdlib).

## Known gaps
- bestLanes (LaneQ) computation and MSLaneChanger execution are out of scope (lane workstream); zipper `foeVisibilityDistance` is a per-link net attribute — record actual net values when replaying.
- `MSLCHelper::updateBlockerLength` summarized, not fully specified (source in `_sumo_src/src__microsim__lcmodels__MSLCHelper.cpp`).
- Junction approach arrival times rely on `getMinimalArrivalTime` (cited, not ported numerically).
