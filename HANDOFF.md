# S-RB-LANE handoff

Branch `c4/rb-lane`, base `f0db3c82`. Owner authorization: REGISTRATION.md:5682,
WM-PHYS REBUILD PROGRAM (2026-10-31 23:12). No training.

## In progress
- Source-backed LC2013 (SUMO **v1_27_1**) implementation, default-off
  `ModelSpec(lane_model="lc2013")` planned. WM lane IDs are left=0, right=1,
  acceleration lane=2 (opposite to SUMO's native indexing).
- Public source downloaded for **reading only** into newly created
  `E:/Temp/opencode/sumo-1.27.1-rb-lane`; no downloaded code executed.
- Exact IDM secure gap is NOT the WM's existing braking-distance gap:
  `max(0, v*tau + v*(v-v_leader)/(2*sqrt(a*b)))` (IDM.cpp:218-220).
- cpuQ observable views reachable; only FIT worlds 969400–969431 permitted.
  Read only `episode.json`, `time_s.npy`, `veh_kin.npy`, `veh_mask.npy`,
  `is_cav.npy`, `vid.npy`. Local CPU busy (91%); use cpuQ, one thread per worker.

## Integration contract (planned)
- Pure tensor module in `src/wms/physics/lc2013.py`, no learned weights.
- Persistent state in `lc2013_*` tensor keys, reset on absence/identity changes
  and successful lane changes. Caller must carry these across rollout steps.
- Existing MOBIL/learned lanes remain default. Integrator must disable learned
  merge decision families when selecting LC2013, or they could overwrite it.
- Only directly comparable encoder IDM settings map; MOBIL politeness/threshold
  do **not** identify LC2013 cooperative/speed-gain settings. LC settings use
  SUMO population defaults unless explicit comparable inferred keys are supplied.

## Tests/results/gaps
Implementation and tape validation pending. Source scope: highway, sublane OFF;
no roundabouts, pedestrians, opposite-direction driving or TraCI wishes.
