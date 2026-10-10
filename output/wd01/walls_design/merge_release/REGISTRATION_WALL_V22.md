# REGISTRATION_WALL_V22.md — wall law v2.2 'merge-release'

Campaign surface: WM-59 wall pilot (J4B re-run) + chunk-5 fixed-policy
wall arms. Branch zcode/wallv22 (from zcode/wallmr fc322e7). Registered
2026-10-01. Supersedes wall v2.1-closure (J4) and the wallrt eee755f
road-end repair (subsumed; see section 4).

## 1. Owner-confirmed 2026-10-01 (BINDING, verbatim)

The following is quoted VERBATIM from
`output/wd01/walls_design/merge_release/OWNER_SPEC_V22_20261001.md`
("OWNER-CONFIRMED 2026-10-01 (relayed by main discussion). BINDING:
overrides any X / hold choice in the design draft"); it is the law this
registration implements. No derived value below overrides it.

1. RELEASE POINT = the START OF THE MERGE (X = 0 m before the merge edge
   start). The wall releases (native handback) when the lead member reaches
   the merge start. Not earlier, not through the merge zone. Log it as
   merge_release.
2. HOLD = PLANNER DECIDES, CAPPED. It can keep or release at each decision
   (60 s cadence), but it ALWAYS releases at the merge start at the latest.
   For the J4b/chunk-5 fixed-policy wall arms (no planner), use hold draws
   capped at the merge start (as registered), with the cap logged.
3. SHAPE = one CAV per mainline lane, side by side (2 abreast across both
   mainline lanes, at the wall speed). Unchanged.
4. WHEN = whenever the planner predicts a benefit (and it's safe). No
   congestion-only trigger.

Proposer: D_min = the assembly distance + a minimal meaningful hold before
the merge start.

## 2. Law constants (implemented; never tuned on outcomes)

| constant | value | where |
| --- | --- | --- |
| `WALL_LAW_VERSION` | `wall-v2.2-merge-release` | wall_group.py:70 |
| entry edge `1191554636` | 2644.2 m (`s_road` = `position_m`) | wall_group.py:80 |
| merge edge `1487606179` | 75.4 m (`s_road` = 2644.2 + `position_m`) | wall_group.py:81 |
| exit edge `1487606178` | 477.9 m (`s_road` = 2719.6 + `position_m`) | wall_group.py:82 |
| `MERGE_START_S` | 2644.2 m | wall_group.py:83 |
| `MERGE_END_S` / `NETWORK_END_S` | 2719.6 / 3197.5 m | wall_group.py:84-85 |
| `MERGE_RELEASE_X_M` | 0.0 m (owner binding) | wall_group.py:87 |
| `MERGE_RELEASE_POINT_S` | 2644.2 m | wall_group.py:88 |
| `MIN_MEANINGFUL_HOLD_S` | 60 s (one decision interval = smallest fixed-arm draw) | common.py:126 |
| `ASSEMBLY_TIME_ASSUMED_S` | 10 s (J4 assembly-time MEDIAN, mechanics only) | common.py:127 |
| hold draws (fixed-policy arms) | {60, 120, 240} s, recorded AS DRAWN | common.py:71 |
| shape | 2 abreast, one CAV per mainline lane, at the wall speed (unchanged) | wall_group.py pacing |

## 3. Mechanics (acceptance-mapped)

- **Release point (item 1).** `s_road` is the edge-aware cumulative road
  coordinate (`frame_s_road`, wall_group.py:108). In HOLD only, the wall
  session fires `merge_release` exactly once when the LEAD member's
  `s_road >= MERGE_RELEASE_POINT_S` (wall_group.py:239-241, 246-259):
  the event carries `release_kind="merge_release"`, `release_point_s`,
  `merge_start_s`, `handback="native"` — a RELEASE kind, never an abort.
- **Hold cap (item 2).** The registered draw is recorded as drawn and the
  dispatcher auto-releases at its expiry; the session's `merge_release` is
  the hard cap ("ALWAYS at the merge start at the latest"). The driver
  logs the realized cap once (`hold_cap` row + `branch_record`
  `requested_hold_s` / `achieved_hold_s` / `hold_capped_at_merge` /
  `merge_release_s` / `lead_s_road_at_release_m`, wall_driver.py:354-382,
  787-806) with the a-priori effective hold `hold_cap_s`
  (wall_request.py:174). Planner branches: keep-or-release each 60 s
  decision is the dispatcher/planner contract, unchanged; the cap binds
  identically.
- **Shape (item 3).** Unchanged v2.1-closure assembly/hold pacing (one
  member per lane, 2 abreast, bounded absolute acceleration).
- **When (item 4).** No congestion-only trigger anywhere in the pilot:
  fixed arms run the registered draw; planner branches propose on
  predicted benefit (PL-EXP1d hooks in REGISTRATION_J4B_DRAFT.md).
- **Assembly defence.** Assembly that would carry the lead member to the
  release point aborts `assembly_past_release` (wall_group.py:214-219);
  the proposer refuses such requests a priori (D_min, below).
- **Outcomes.** `assembled_released` (draw expiry) | `merge_released`
  (cap fired) | `aborted_in_assembly` | `aborted_in_hold` |
  `not_feasible`. The v2.1 `released_road_end` outcome is retired
  (outside `OUTCOMES`; wall_driver.py:99-110).

## 4. Proposer D_min (a priori; owner section-1 sentence implemented)

    D_min = X + v_wall * t_assemble + v_wall * H_min
          = 0 + v_wall * (10 s + 60 s) = 70 * v_wall   [m]

- `X = MERGE_RELEASE_X_M = 0.0 m` (owner binding, section 1).
- `t_assemble = 10 s` — the J4 assembly-time MEDIAN (n=131 assembled;
  median 10 s, max 55 s; J4_MERGE_ANALYSIS.md section 3; mechanics
  records only, no outcome fields).
- `H_min = 60 s` — the minimal meaningful hold: one 60 s planner
  decision interval = the smallest registered fixed-arm draw.
- Gate: the LEAD member's distance to the merge start
  (`MERGE_START_S - s_road(lead)`) must be `>= D_min` at the request
  (wall_request.py:153-171, 229-299). A span-feasible pair failing only
  this gate is refused AT THE PROPOSAL with reason `dmin_unreachable`
  (logged with the lead distance, D_min and wall speed;
  wall_request.py:327, wall_driver.py:344, 431, 443).
- **J4-geometry feasibility under D_min** (a-priori scoring of the 131
  assembled J4 requests with their MEASURED assembly time, no outcomes;
  J4_MERGE_ANALYSIS.md section 6): **109 / 131 feasible (83.2 %)**;
  D_min distribution median 1076.6 m (min 152.5, max 2653.8); margin
  median +980.8 m. Under the a-priori median-time formula the accepted
  set is the 109-request envelope minus requests with
  `t_assemble > 10 s` that the measured-time scorer admitted; the
  proposer always uses the a-priori formula (no per-request fitting).

## 5. Reconciliation with zcode/wallrt eee755f (edge-aware road_left_m)

eee755f repaired `road_left_m` per edge because the v2.1 planned release
fired at the ROAD END (`road_left < v_wall x 10 s`). v2.2 supersedes
both: the release point is the MERGE START on the cumulative `s_road`
coordinate, the road-end release and its outcome are retired
(`ROAD_END_RELEASE_S` / `OUTCOME_RELEASED_ROAD_END` kept only so
archived J4 records replay), span and downstream coordinates run on
`s_road` across entry/merge/exit (wall_group.py:286-302), and
`road_left_m` survives only as edge-aware telemetry
(`NETWORK_END_S - s_road`, wall_request.py:191).

## 6. Evidence & verification (zero SUMO)

- Patched copies + diffs vs the live files: `copies/{pre,post}/`,
  `patches/*.diff`, `PATCH_SHAS.json` (LF sha256 pre/post per file;
  `sensing.py` unchanged: frames already carry the lane/edge identity
  and lane-local position `s_road` needs).
- Tests (zero SUMO, synthetic report frames only):
  `python -m pytest output/wd01/walls_design/merge_release/tests -q`
  with `WD_FORBID_D=1` — 25 passed (s_road per edge; merge_release in
  HOLD only, exactly once at 2644.2; planner-held wall released at the
  merge start at the latest; fixed-arm hold capped + cap logged; D_min
  proposal refusal with reason; span on s_road across edges; synthetic
  hold crossing the merge without `member_left_network`; road-end
  superseded).
- Declared mechanics smoke fixture: `SMOKE_V22.json` / `SMOKE_V22.md`
  (3 x 300 sim-s, run by the reconstructor on the owner-approved
  laptop; never run by the proposer).
- J4B campaign draft: `REGISTRATION_J4B_DRAFT.md`.
