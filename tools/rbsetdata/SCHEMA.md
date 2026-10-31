# `setdata_hdv_histories/1` — per-driver observable 60 s histories

Produced by `tools/rbsetdata/extract_hdv_histories.py` from the S-WMDATA FIT
block **969400–969431, native arm** (`episode.npz`, schema `c4_sumo_episode/1`;
SUMO_NOTES.md section 3).  One flat, compressed npz + a JSON side manifest.

## Frame

* lane `0` = main left, `1` = main right, `2` = ramp/acceleration lane;
  `s` = merge-relative coordinate (merge zone 0..75.44 m, upstream negative)
  (`traffic_shepherd/c4/geometry.py:9-12`).
* Junction internal lanes: `lane = -1`, `s` carried, `junction = 1`,
  `lane_carry` = last modelled lane.  The only junction is the s≈0 ramp merge.
* dt = 0.5 s; window = `[t0-60, t0]` with `t0 = warmup_s = 300` → 121 ticks,
  window-relative `t` ∈ [0, 60].

## Arrays

Segment (one driver's contiguous presence in the window; `S` rows; a present
gap splits segments — dropped-tick or exit):

| array | dtype | meaning |
|---|---|---|
| `seg_id` | i32 | global segment id (tick rows reference it) |
| `world` | i32 | index into the `world_*` arrays |
| `veh_id` | U12 | SUMO vehicle id |
| `is_cav` | u8 | CAV flag (default extraction: HDV only) |
| `t0`,`t1` | f32 | window-relative first/last tick |
| `nticks` | i32 | ticks in window (≥ `--min-ticks`, default 20 = 10 s) |
| `enters_window` | u8 | 1 if inserted inside the window |
| `first_lane` | i8 | effective lane at first tick |
| `n_lc` | i32 | recorded modelled-lane lane changes |
| `n_merge` | u8 | junction-exit events from lane 2 |

Per tick (flat, `N` rows, f32; NaN = no such neighbour):

`seg_id, t, s_m, v_mps, a_mps2, lane, eff_lane, junction, length_m,
lead_gap_m, lead_dv_mps, follow_gap_m, follow_dv_mps, left_lead_gap_m,
left_lead_dv_mps, left_follow_gap_m, right_lead_gap_m, right_lead_dv_mps,
right_follow_gap_m`

* `lead_gap_m` = net bumper gap `s_leader - s_ego - length_leader` (same lane);
  `follow_gap_m` = `s_ego - s_follower - length_ego`.
* `*_dv_mps` = neighbour speed − ego speed.
* `left_*` / `right_*` = adjacent lane (`lane-1` / `lane+1`); NaN when the lane
  does not exist (lane 0 has no left, lane 2 no right) or no neighbour within
  200 m (`NEIGHBOR_MAX_GAP_M`).  Adjacent follower speed is not recorded.
* Neighbour search uses `eff_lane` (junction vehicles keep carried `s`), so a
  leader inside the junction is visible.

Lane-change events (`lc_*`, `E` rows): `lc_seg_id, lc_t, lc_from_lane,
lc_to_lane, lc_s_m, lc_v_mps, lc_gap_lead_pre_m, lc_dv_lead_pre_mps,
lc_gap_lead_post_m, lc_gap_follow_pre_m, lc_is_ramp_to_main` — modelled-lane
moves only (no junction transits); `*_pre` = decision-time (tick t−1) state;
`lc_t` = first tick on the new lane.

Junction-exit/merge events (`mg_*`, `M` rows): `mg_seg_id, mg_t, mg_s_m,
mg_v_mps, mg_from_lane, mg_to_lane, mg_is_ramp_merge, mg_gap_lead_m,
mg_dv_lead_mps` — first tick back on a modelled lane after a junction transit;
`mg_is_ramp_merge=1` when the carried lane was 2 (the s≈0 ramp merge;
`mg_gap_lead_m` = leader gap in the arrival lane).

Worlds (`world_*`): `world_id` (`w<seed>m<demand>p<pct>t300`),
`seed, demand_vph, penetration, source (tape path), episode_sha256`.
Scalars: `schema`, `window_s`, `speed_limit_mps` (31.29 =
`wms/data/c4_episodes.py:69`).

Manifest JSON: counts + per-world tape provenance (path, sha256).

## The 32-FIT-world product (2026-10-31)

`fit969400-431_hdv_hist_60s.npz` (7.2 MB): 32 native tapes, 2 831 segments
(all HDV; no CAVs in window after default filter), 256 527 ticks, 652 LC
events (313 `1→0`, 175 `0→1`, 164 accel-lane `2→1`, none into lane 2),
1 094 junction-exit events: 104 ramp merges from lane 2 (97 `2→2` ramp edge →
acceleration lane at s≈0-6 m, 7 `2→1` direct), plus 990 carry-through
transits of the downstream edge junction at s≈75-88 m (`lane_carry ==
to_lane`; the merge-zone exit where the upstream edge ends).  `fit969400-431_per_driver_params.csv`:
population + 2 517 fitted drivers (see README).
