# A38 hand-written hold for road 2 (wz2to1): mechanical port of duo12_v20

Fixed 2026-11-10, before any SUMO, WM or outcome on wz2to1 (zero episodes run on wz2to1 when this file was written).
No behavioural value was chosen here. Every number is copied from duo12_v20. Only the lane set follows the geometry.

## Source (wz3to2): duo12_v20
UNION_UNC `GUIDANCE_C4SIT_priv_UNION_UNC.json` sha256 39f384a2..., row family duo12_v20 (regime D2800_p20), strategy_key
750a55c9439a2efc, template sha256 7b94b5ae...:
- trigger `wall`: one CAV per SUMO `up` lane (0, 1, 2) inside zone x in [896, 2296] (400-1800 m upstream of
  X_B = 2696), spread <= 200 m, key (spread, -max x).
- roles: SUMO lane 1 and SUMO lane 2, so both lanes that continue past the drop. Each role has rank 0, onset 0,
  v_w 20.0 m/s, dmax 1.0, amax 1.0, hold 120 s, kp 0.15, kv 0.6, lc_mode 513, no lane target.
- releases at hold end, or when the member reaches the drop (s 75.44 = x 2696). Cooldown 120 s, slots 1, no gate,
  no start gate, no phases. Decisions are made every whole second from t >= 300 s ("whenever feasible").

## Road 2 geometry (wz2to1, net sha256 7c7e9aa3...)
`up` has 2 lanes from 0 to 2696 m. SUMO lane 0 ends at x = 2696 (no connection). SUMO lane 1 continues onto `wz` (1 lane).
X_B, the zone and the release point are therefore numerically identical to wz3to2.
Lanes that continue past the drop: {1}, so there is exactly ONE.

## Port (PRIMARY, fixed): HOLD_R2 = "duo12_v20 port, continuing-lane rule"
The mapping is mechanical: one role per continuing lane, and the trigger takes one CAV per approach lane.
```json
{"name": "duo12_v20_r2port", "map": "wz2to1", "trigger": "wall", "wall_lanes_sumo": [0, 1], "zone": "full",
 "zone_x": [896.0, 2296.0], "spread_m": 200.0, "cooldown_s": 120, "slots": 1, "gate": null, "start_gate": null,
 "phases": null,
 "roles": [{"lane": 1, "rank": 0, "v_w": 20.0, "dmax": 1.0, "amax": 1.0, "hold_s": 120, "onset_s": 0,
            "lc_target": null, "kp": 0.15, "kv": 0.6, "lc_mode": 513, "release_s": 75.44}]}
```

**DEGENERATE: say this plainly in any report.** A 2-to-1 drop has only one continuing lane. The "two CAVs abreast in
the continuing lanes" rule therefore collapses to a SINGLE-CAV hold in SUMO lane 1. HOLD_R2 is not a multi-CAV
cooperation strategy. It is the road-2 analogue of a solo hold (structurally wall_g_solo2 with v_w 20 instead of 22).
On this road it cannot test "multi-CAV cooperation". The wall trigger still needs one CAV in each approach lane (0 and 1).
This is the port of "one CAV per approach lane", and lane 0's member receives no command, as on wz3to2.

## Documented two-CAV alternative (NOT primary; becomes an arm only if MAIN registers it before any outcome)
HOLD_R2_DUO01 = the same values with roles in SUMO lanes 0 and 1 (all approach lanes abreast). This keeps "two CAVs
abreast" but changes the mechanism. On wz3to2 the closing lane was never held. Here the closing-lane member holds until
it must merge, so the whole cross-section is held at 20 m/s (a full moving wall).
```json
{"name": "duo12_v20_r2duo01", "map": "wz2to1", "trigger": "wall", "wall_lanes_sumo": [0, 1], "zone": "full",
 "zone_x": [896.0, 2296.0], "spread_m": 200.0, "cooldown_s": 120, "slots": 1, "gate": null, "start_gate": null,
 "phases": null,
 "roles": [{"lane": 1, "rank": 0, "v_w": 20.0, "dmax": 1.0, "amax": 1.0, "hold_s": 120, "onset_s": 0,
            "lc_target": null, "kp": 0.15, "kv": 0.6, "lc_mode": 513, "release_s": 75.44},
           {"lane": 0, "rank": 0, "v_w": 20.0, "dmax": 1.0, "amax": 1.0, "hold_s": 120, "onset_s": 0,
            "lc_target": null, "kp": 0.15, "kv": 0.6, "lc_mode": 513, "release_s": 75.44}]}
```
Holder rule (oracle_plan.holder_of: the SUMO-lane-2 member, else the first) gives the first role, i.e. the lane-1 member.

## Executor note
oracle_plan._pick_wall hard-codes lanes (0, 1, 2), so on wz2to1 it would never fire. The port needs the wall lane set
(0, 1) per map (branch c4/a38-road2, `tools/coopmwm/road2.py`). The wz3to2 code path and oracle digest 7409a7b4 are
unchanged.
