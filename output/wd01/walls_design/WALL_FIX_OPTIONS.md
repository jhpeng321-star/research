# J4 wall pilot: minimal fix options (for the owner, morning of 2026-09-30)

World model session, 2026-09-30. Docs plus one zero-SUMO, TRAIN-only
descriptive scan. Nothing is implemented yet.

## The two defects (J4 smoke w8693m2600p40t420__w-mid-early; confirmed in code and network)

1. **Assembly never converges for span > 6 m.**
   - `traffic_shepherd/wall_group.py` `WallGroupSession.tick` paces EVERY
     member toward the same wall speed (`_pace`).
   - Assembly completes only when every member is within 1 m/s of the wall
     speed AND the span is ≤ `ASSEMBLY_SPAN_TOL_M` = 6 m.
   - The feasibility rule, however, admits spans up to `MAX_WALL_SPAN_M` =
     75 m. A 55 m pair therefore stays 55 m apart.
   - The Part A emulator (`partA_ctrl/emulators.py::wall_pace`) mirrors the
     same law, so any closure rule is a LAW change in both places.
2. **Pairs are chosen on the terminal edge.**
   - `choose_pair` picks the feasible pair "nearest the merge".
   - The allowed edges (`MAINLINE_2LANE_EDGES`, taken from PLAN_DRAFT's
     feasibility rule) include 1487606178. That is the 477.9 m EXIT edge
     downstream of the merge, and it has no outgoing edges.
   - Network (sumolib, preserved net):

     | edge | role | length | lanes |
     | --- | --- | --- | --- |
     | 1191554636 | mainline ENTRY, no incoming edges | 2644.2 m | 2 |
     | 1487606179 | merge edge | 75.4 m | 3 |
     | 1487606178 | EXIT | 477.9 m | 2 |

   - At most about 3197 m of road remains ahead of any member.

## Scan (descriptive; TRAIN only; chunk-4 `<world>__native`, the same saved state the wall branches restore)

- **Sample:** 39 TRAIN worlds at p10/p20/p40 × {early, late} = 78 requests.
  Each looks for a pair within the registered 120 s request window.
- **Scripts:** walls_design/scan_pair_feasibility.py and scan_d1_tight.py.
- **Rows:** scan_rows.jsonl.
- **Caveat:** CAV physics positions stand in for self-reports.

| rule | pair found | on EXIT edge | span ≤ 6 m at request | road left (median) |
| --- | --- | --- | --- | --- |
| CURRENT (either edge, span ≤ 75, nearest merge) | 78/78 | **40/78** | **6/78** | 449 m |
| D1 (upstream edge only, minimal span) | 77/78 | 0 | 31/78 | 1622 m |
| D1-tight: upstream edge, span ≤ 6 m, road left ≥ 1200 m, first such tick within 120 s | **62/78** | 0 | 62/62 | — |
| D1-tight with road left ≥ 2400 m | 58/78 | 0 | 58/58 | — |

**D1-tight by penetration (road left ≥ 1200 m):**

| penetration | requests with a tight pair |
| --- | --- |
| p10 | 8/24 |
| p20 | 26/26 |
| p40 | 28/28 |

**Expected outcome, BEFORE the fix (current rule, current law):**

- About 72/78 (92 %) of requests get a pair wider than 6 m. With no closure
  law, those never assemble. They abort on member_frame_missing,
  member_span_exceeded or lane_coverage, or they stall until a member leaves
  the network.
- 40/78 are on the exit edge, with about 17–25 s of road left.
- Only about 6/78 could assemble at all.

**AFTER option A (below):**

- 62/78 (79 %) get a pair that is already within tolerance, so they assemble
  within a few seconds of speed alignment.
- 16/78 are `not_feasible` (logged; the branch continues native). 16 of those
  are p10 (p10 has 2–4 CAVs on the 2.6 km edge).

## Road needed ahead of the members

The requirement is X = v_wall × (A + H) + a release margin.

- **v_wall** = factor × the members' mean reported speed. The median mean
  speed at request is about 25.4 m/s, which gives:
  - W-mid (factor 0.85): about 21.6 m/s;
  - W-lo (factor 0.60): about 15.3 m/s.
- **A** (speed alignment, −3 m/s² clip from about 25 m/s) is about 4 s. I take
  A = 10 s.
- **Release margin:** 10 s × v_wall.

| hold H | W-lo X | W-mid X | fits in ≤ 3197 m? |
| --- | --- | --- | --- |
| 60 s | 15.3 × 80 ≈ 1224 m | 21.6 × 80 ≈ 1728 m | yes |
| 120 s | ≈ 2142 m | ≈ 3024 m | only when starting near the network entry |
| 240 s | ≈ 3978 m | ≈ 5616 m | **never** |

**Conclusion:**

- A 240 s hold can never complete on this network. A 120 s hold often
  cannot.
- The minimal guarantee is X_min = v_wall × 80 s (H = 60 plus margin), i.e.
  about 1.2–1.7 km.
- Longer draws are truncated by the planned road-end release (D2) and
  recorded as achieved vs requested hold.

## Options

### Option A: member choice and placement only. NO law change; stays inside the wall-v2 law. Recommended now.

- **A1.** Allowed member edge = 1191554636 only. Drop 1487606178 from
  MAINLINE_2LANE_EDGES for member choice.
- **A2.** Feasibility span cap for the request = the assembly tolerance,
  6 m (replacing 75 m). Among the feasible pairs, pick the minimal span; break
  ties by the most road left. The planner still waits up to the registered
  120 s request window, otherwise `not_feasible`.
- **A3.** Remaining-road check: road_left(front member) ≥ v_wall × 80 s
  (X_min above).
- **A4 (D2).** A planned road-end release in HOLD when road_left < v_wall ×
  10 s. It is logged as outcome `released_road_end` with achieved vs
  requested hold. It is NOT a safety abort. In ASSEMBLY the abort reason
  becomes `member_left_network` when the member is in arrived_ids.
- **Registered-design impact:**
  - A1 and A2 tighten PLAN_DRAFT's feasibility rule (the edge list and the
    75 m cap). A3 and A4 are new driver behaviour on top of the unchanged
    wall law.
  - Main discussion allowed "a pure member-choice/placement fix that stays
    inside the registered design". A1–A3 are member choice. A4 is a planner
    release, which the lifecycle already permits after assembly.
  - I recommend implementing A1–A4 now and recording them as a disclosed
    plan amendment. The owner can veto in the morning.
- **Cost:**
  - p10 walls are mostly not_feasible (8/24 feasible).
  - H = 240 is always truncated, and H = 120 usually is.
  - W-mid at the typical 25 m/s is 21.6 m/s, a mild wall.

### Option B: assembly-closure law. A LAW CHANGE, so it needs the owner. Not implemented.

- **Rule:**
  - During ASSEMBLY the front member targets v_wall.
  - Each rear member targets v_wall + Δ, with Δ = min(2.0 m/s,
    0.1 s⁻¹ × gap to the front member). This stays inside the existing
    [−3, +1.5] m/s² clip and speedMode 27 safe following.
  - Assembly completes at span ≤ 6 m and all speeds within 1 m/s.
  - A new safety abort `assembly_timeout` fires at A_max = 60 s.
- **Where:**
  - traffic_shepherd/wall_group.py (`tick`, assembly branch);
  - the mirrored partA_ctrl `wall_pace` (Part A emulation);
  - Part B wall tokens need no change.
- **Validation:** unit tests (closure from 50 m; timeout; no rear-end: the
  existing REAR_GAP/REAR_SPEED_EXCESS surveillance stays), the WM-43 fixture
  re-run, and a smoke with a pair ≥ 40 m wide.
- **Benefit:**
  - D1 finds pairs for 77/78 requests (median span 11.7 m, p90 59.8 m).
  - At Δ ≤ 2 m/s a 60 m span closes in about 30–40 s, within the 60 s
    timeout. So p10 would also be served.
  - It keeps the 75 m feasibility.
- **Cost:** a change to the validated wall controller. AGENTS.md requires
  feasibility AND incremental-benefit approval for wall assembly changes.

### Option C: hold draws (owner)

- Keep H ∈ {60, 120, 240} with the A4 truncation recorded (recommended; the
  achieved hold becomes data).
- Or re-register H ∈ {30, 60, 90} to fit the road.

## Recommendation

1. Implement Option A now in campaign_walls_v1, jointly with the
   reconstructor. R1 continues in parallel. Then re-smoke. The smoke world must
   yield a tight upstream pair; if the registered smoke world does not, the
   smoke may record not_feasible, which is valid.
2. The owner decides Option B (closure law) and Option C in the morning.
   - If B is approved, it lands as a separate, versioned wall-law change with
     its own tests. J4 then runs A + B (75 m cap restored).
   - If not, J4 runs A as is.
