# DESIGN_V22_MERGE_RELEASE.md — wall law v2.2 'merge-release' design note

2026-10-01. Implements `OWNER_SPEC_V22_20261001.md` (binding); the
verbatim owner text and the registered constants live in
`REGISTRATION_WALL_V22.md` (section 1 and 2). This note records the
formulas, the a-priori D_min derivation and the J4-geometry feasibility
count. Zero SUMO was spent here.

## 1. Road coordinate `s_road` (edge-aware, cumulative)

Let `e(f)` be the edge of a report frame (SUMO lane id `<edge>_<idx>`;
internal lanes start with `:`), `p(f)` the lane-local position
(`position_m`), `d(f)` the registered merge-distance coordinate:

    s_road(f) = p(f)                    if e(f) = 1191554636 (entry, 2644.2 m)
              = 2644.2 + p(f)           if e(f) = 1487606179 (merge,  75.4 m)
              = 2719.6 + p(f)           if e(f) = 1487606178 (exit,  477.9 m)
              = 2644.2 - d(f)           on internal/unknown edges (fallback)

Merge START = 2644.2 m; merge END = 2719.6 m; network END = 3197.5 m.
Every position comparison (span, rear surveillance, release point) runs
on `s_road` (`frame_s_road`, wall_group.py:108). This replaces the
edge-blind v2.1 assumption "`position_m` is an entry-edge coordinate"
(J4_MERGE_ANALYSIS.md section 1) and subsumes the wallrt eee755f
per-edge `road_left_m` repair.

## 2. Release (owner item 1) and hold cap (owner item 2)

Release point `X = 0 m` before the merge edge start (owner binding):

    MERGE_RELEASE_POINT_S = MERGE_START_S - X = 2644.2 m

In HOLD only, the wall session fires `merge_release` exactly once when

    max_members s_road >= MERGE_RELEASE_POINT_S - 1e-9

(tick-quantized: the fire tick is the first control tick whose REPORT
frames place the lead member at/after the point; members hand back to
native that tick and are never commanded through the merge zone).
Planner branches may release earlier at any 60 s decision; the draw of
a fixed-policy arm releases earlier at its expiry; the merge start caps
all of them. The a-priori effective hold of a drawn `H` at lead
coordinate `s_L` and wall speed `v_w`:

    hold_cap_s(H, s_L, v_w) = min( H , max(0, 2644.2 - s_L) / v_w )

Recorded telemetry keeps drawn vs effective distinct: `requested_hold_s`
(draw), `achieved_hold_s` (realized), `hold_capped_at_merge`,
`merge_release_s`, `lead_s_road_at_release_m`, plus the `hold_cap`
event row and the `merge_release` dispatcher event.

Defensive rule: an ASSEMBLY whose lead reaches the release point aborts
`assembly_past_release` (never a merge release) — the proposer's D_min
gate makes it unreachable in accepted requests.

## 3. Proposer D_min (owner sentence, a-priori derivation)

    D_min = X + v_wall * t_assemble + v_wall * H_min
          = 0 + v_wall * (10 + 60) = 70 * v_wall   [m]

Derivation (a priori, from the J4 analysis geometry only — request /
assembly / hold records, no outcome fields):

- **Assembly distance** = `v_wall * t_assemble`. J4 assembly-time
  distribution (n=131 assembled): min 1 s, p25 2 s, **median 10 s**,
  p75 16 s, max 55 s. The a-priori assumption registers the MEDIAN
  `t_assemble = 10 s` (`ASSEMBLY_TIME_ASSUMED_S`); no per-request
  fitting (the measured-time scoring below is diagnostics only).
- **Minimal meaningful hold** = one 60 s planner decision interval =
  the smallest registered fixed-arm draw {60, 120, 240}:
  `H_min = 60 s` (`MIN_MEANINGFUL_HOLD_S`).
- `X = 0` (owner binding).

**D_min value: `70 * v_wall` metres** (e.g. 1071 m at the J4 median
wall speed 15.3 m/s of assembled requests; 1190 m at v = 17 m/s).

Gate: `MERGE_START_S - s_road(lead) >= D_min` at the request tick;
span/lane rules unchanged (pair on the entry edge, two lanes,
span <= 75 m). A pair failing only D_min is refused at the proposal
(`dmin_unreachable`, with the lead distance / D_min / wall speed
logged). The old A4 rule `road_left >= v_wall x 80 s` is superseded
(D_min = 70 v binds earlier; the road end is no longer a control
surface).

## 4. J4-geometry feasibility under D_min (count; no outcomes)

Scoring each of the 131 assembled J4 requests with its MEASURED
assembly time (diagnostics; J4_MERGE_ANALYSIS.md section 6):
**feasible 109 / 131 = 83.2 %**; D_min median 1076.6 m (min 152.5,
p75 1470.4, max 2653.8); margin `dist_to_merge(req) - D_min` median
+980.8 m (p25 +283.8, min -981.2). The 22 infeasible requests cluster
at late requests (lead already close to the merge) — exactly the class
v2.2 refuses at the proposal instead of assembling into a forced
release. Under the registered a-priori median-time formula (70 v) the
accepted set is a subset of these 109 (requests whose measured
`t_assemble > 10 s` can fall below the a-priori envelope); the
registered J4B proposer uses the a-priori formula only.

## 5. Why J4's failures disappear under v2.2 (design expectation, not evidence)

- 26 `member_left_network` aborts finished at s_road ~3195 m (the road
  end): impossible in v2.2 — the hold ends at 2644.2 m at the latest.
- 17 `member_span_exceeded` (~2786 m, exit edge) and the 40 holds into
  the exit edge: the wall never enters the merge zone locked.
- The 48/147 holds past the merge start: now legal ONLY as the instant
  `merge_release` handback at the merge start, logged as a release.

These are mechanism expectations; efficacy evidence comes only from the
registered J4B run (REGISTRATION_J4B_DRAFT.md).
