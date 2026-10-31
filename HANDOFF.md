# S-RB-LANE handoff

Branch `c4/rb-lane`, base `f0db3c82`. Owner authorization: REGISTRATION.md:5682,
WM-PHYS REBUILD PROGRAM (2026-10-31 23:12). No training.

## Exists
- Source-backed LC2013 highway subset (SUMO **v1_27_1**) implemented;
  default-off `ModelSpec(lane_model="lc2013")` integrated. WM lane IDs are left=0, right=1,
  acceleration lane=2 (opposite to SUMO's native indexing).
- Public source downloaded for **reading only** into newly created
  `E:/Temp/opencode/sumo-1.27.1-rb-lane`; no downloaded code executed.
- Exact IDM secure gap is NOT the WM's existing braking-distance gap:
  `max(0, v*tau + v*(v-v_leader)/(2*sqrt(a*b)))` (IDM.cpp:218-220).
- cpuQ observable views reachable; only FIT worlds 969400–969431 permitted.
  Read only `episode.json`, `time_s.npy`, `veh_kin.npy`, `veh_mask.npy`,
  `is_cav.npy`, `vid.npy`. Local CPU busy (91%); use cpuQ, one thread per worker.

## Integration contract
- Pure tensor `lane_change` in `src/wms/physics/lc2013.py`, no learned weights.
- Persistent state in `lc2013_*` tensor keys, reset on absence/identity changes
  and observed lane changes. `rollout` carries across steps/chunks and resets
  newborns. Direct step callers carry `physics.lc2013.STATE_KEYS` themselves.
  Numeric current identity uses `vehicle_id` or `vid` when available.
- Existing MOBIL/learned lanes remain default. Integrator must disable learned
  merge decision families when selecting LC2013, or they could overwrite it.
- Only directly comparable encoder IDM settings map; MOBIL politeness/threshold
  do **not** identify LC2013 cooperative/speed-gain settings. LC settings use
  SUMO population defaults unless explicit comparable inferred keys are supplied.
- New `model/lc2013_adapter.py`: forward diagnostics use current snapshot; step
  recomputes after longitudinal motion/CAV execution, preserves fixed CAV lanes,
  and geometrically arbitrates simultaneous moves front-to-back. Ramp end held
  if no merge. Previous LC follower-help messages affect next HDV acceleration.
- Build with `ModelSpec(lane_model="lc2013", lane_head="off")` and the desired
  physics/encoder mode. No new weights. `merge_decision` must remain `hazard`
  (no learned merge module); other learned acceleration corrections must be
  disabled by WM-PHYS integration. Do not mutate `.lane_model` on a live
  hazard_v5/first-passage model with its old modules still active.

## Tests/results/gaps
56 CPU tests passed locally (one CPU thread, 22.64s), including 15 new tests
plus existing lane_rule/wmlane/native_lane suites. Tape validation pending.
Numeric secure-gap, discrete braking inverse,
strategic ramp, overlap/two-sided safety, assertive, persistent speed-gain,
5s keep-right, cooperative clearing, identity/absence reset, invalid settings,
default-off/spec checks, lane-logit ownership, fixed CAV lanes and exact chunk parity.
`tools/rblane/` validator/owned-dispatch tools added: 160 FIT episodes, 60s
causal wish warmup then 60s teacher-forced scoring, paired MOBIL. Smoke finished
on cpuQ (own PID180907 exited): LC exact7/30 vs MOBIL3/30; full panel pending.
19 current LC/model/scorer tests pass; empty-onehot connectors are excluded.
Source scope: highway, sublane OFF. **Known parity gaps:** synchronous messages
instead of SUMO downstream-first serial changes; full leader-overtake/fall-behind
negotiation and right-overtaking prevention not ported; no roundabouts,
pedestrians, opposite-direction driving or TraCI wishes. Optional route
distances/offset/queued occupation accepted; defaults use scenario ramp end and
continuing mainline. No SUMO or training launched, no tape result yet.
