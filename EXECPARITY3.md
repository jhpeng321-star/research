# S-EXECPARITY3 — the WM frame identifies leaders by SUMO's lane/link rule

**Status: COMPLETE; the registered parity criterion now PASSES and the gate
receipt is written — `verify_parity` accepts mode-2 deployment.** Branch
`c4/execparity3` off `c4/execparity2` HEAD `6b7e0d24`, worktree
`E:/agent-work/c4-execparity3`. CPU only, **zero SUMO executions, zero sim-s,
zero new charges**; the offline replay re-uses the fetched 00:20 recordings
(32,617 rows, unchanged row mapping and criterion). Flag OFF (the default) is
bitwise the frozen legacy behaviour everywhere; flag ON is enabled by
`C4_SUMO_LEADER_RULE=1` only. No truth driver types, no vType files, no
hand-set behavioural thresholds — the rule uses only observable frame
information (lane one-hot, corridor `s`, junction flag, carried lane) and the
registered network's connector lengths.

## 1. The mismatch anatomy (measured on the executor's own ledger)

Re-playing the S-EXECPARITY2 classification against the executor's logged
`getLeader` (gap converted to bumper by +minGap): 1,675/32,617 rows (5.14 %)
had a frame leader different from SUMO's. The dominant mechanisms, in order of
squared-error contribution:

1. **ramp egos** (frame lane 2, `s < 0`): SUMO's leader gap = frame gap
   **+ 2.93 m** — the ramp connector `:1780739017_0_0` (verified against
   `osm.net.xml.gz`) that the corridor coordinate `s` hides;
2. **mainline egos** near the entry junction (lanes 0/1, `s ∈ [-5, 0]`), near
   the merge-edge end (`s ≈ 73-75.44`) and across the exit junction: corridor
   projection misses the **3.00 m** entry / **8.00 m** exit connectors
   (`:1780739017_1_*`, `:13638027693_0_*`) — the 0.1-2 m "phantom leaders" of
   the EXECPARITY2 decomposition are the extreme of these (gap clamped at
   0.1 m by `leader_gap_dv`);
3. **junction-transit vehicles** (internal connector lanes, empty lane
   one-hot) participating in the corridor search with a phantom argmax lane;
4. residual: rows where SUMO finds **no** leader because its own search budget
   stops within `brakeGap(speed)+minGap` (SUMO 1.27.1 `MSVehicle::getLeader`,
   `dist == 0` branch) while the frame sees a far leader, and
   junction-internal allowed-speed rows — both unobservable in a WM frame and
   benign for the executed law at 60-130 m gaps.

## 2. The rule (default OFF)

`src/wms/physics/lane_chain.py` (physics layer, pure geometry): the corridor
`s` plus the registered connector lengths gives a **lane-chain coordinate**
`c` that is monotone along each frame lane's physical chain (mainline: entry
edge → `:1780739017_1_*` → merge edge → `:13638027693_0_*` → exit edge; ramp:
→ `:1780739017_0_0` → accel lane, a dead end); junction-transit vehicles map to
the ENTRY of their connector (observable lower bound; never overstates a gap).

* `wms.control.sumo_leader.leader_gap_dv_sumo` — the CAV execution mirror's
  leader: nearest-ahead in `c` on the same physical lane; junction slots
  participate with their carried lane (`state["lane_carry"]`, present in the
  recorded tapes) at the connector-entry bound; bumper-gap floor 0.1 m and
  no-leader conventions unchanged. Default OFF → calls the frozen
  `native_cav.leader_gap_dv` bitwise. `alwayscmd._wm_geometry` routes through
  it, so `inferred_base` / `inferred_cap` / `executed` follow the same leader.
* `wms.physics.neighbours.lane_neighbour` / `lead_neighbour` — the WM physics
  lane search: under the flag it runs in the chain coordinate and junction
  slots (empty lane one-hot) are neither candidates nor egos; the lane-2 dead
  end competes at chain `MERGE_S1 + 2.93`. Flag OFF is the legacy body
  byte-for-byte.
* `wms.control.contract` re-exports the three connector lengths (single
  definition in `lane_chain`, asserted in lockstep with `MERGE_S1_M`).

**Leader agreement vs the executor's logged getLeader (3 worlds, 32,617
rows):** mismatches **1,675 (5.14 %) → 450 (1.38 %)**; 1,225 rows fixed, **0
previously-matched rows broken**. The residual is dominated by the
budget-bound far-leader rows and junction-internal rows (§1.4).

## 3. Offline re-score — criterion UNCHANGED, verdict FLIPS TO PASS

`tools/execparity/alwayscmd3_replay_results.json` (rule: pooled imagination
executed-accel RMSE ≤ pooled WM native-head RMSE 0.1603; canary identity
unchanged; flag ON):

| | world 977001 | 977002 | 977003 | pooled |
|---|---:|---:|---:|---:|
| n rows | 10,695 | 5,387 | 16,535 | 32,617 |
| imagination executed RMSE | **0.1218** | **0.1958** | **0.1423** | **0.1465** |
| native-head band RMSE | 0.1586 | 0.1898 | 0.1506 | 0.1603 |
| within band | yes | no (+0.006) | yes | **PASS** |

Pooled mean error −0.0002, MAE **0.0143** (was 0.096 with the learned mirror,
0.035 with the exact law + legacy leader), median 0. The registered criterion
is pooled; world 977002 sits marginally above its own head band (0.1958 vs
0.1898) — reported as measured. **Gate receipt
`tools/execparity/alwayscmd3_execution_parity.json` written;
`execution.verify_parity` ACCEPTS mode 2 with it (checked directly: schema,
backend contract, CTRL0 identity, driver sha, residual RMSE ≤ head RMSE).**
Candidate-A launch condition (1) is met by this receipt, subject to MAIN's
remaining conditions.

## 4. Unit tests

`tests/unit/test_execparity3.py` (new, 19), ruff-clean: flag-OFF **bitwise**
identity of `leader_gap_dv_sumo` vs `native_cav.leader_gap_dv`, of the
`neighbours` lane search vs a byte-copy of the legacy body, and of
`alwayscmd._wm_geometry`; connector lengths pinned to the network attributes
(2.93/3.00/8.00); chain coordinates (entry/exit/ramp shifts, junction entry
bound); fixtures measured from the anatomy — ramp ego gap = legacy + 2.93,
mainline-entry 1 m phantom pushed to 4 m, exit connector, accel-lane ego blind
to upstream ramp vehicles, junction-transit leader only for its upstream lane,
no-carry fallback; neighbours chain rule + chain dead end + flag-OFF dead end;
replay tool wiring (`lane_carry` pass-through with lane fallback; replayed law
value under the flag equals `scalar_executed` on the chain gap); committed
receipt satisfies the mode-2 predicates. 69 execparity tests PASS (50 prior +
19 new); import-graph failures are the 8 pre-existing ones (identical on the
untouched base worktree); nativeinit environmental failures unchanged (6, all
pre-existing).

## 5. Effect on ALL vehicles' WM dynamics — 3 FIT event worlds

`tools/execparity/dynamics3.py` → `dynamics3_results.json`. E3 split **fit**
worlds only (never validation/pairs300/calibration), the event worlds
w973002/g0, w973006/g3, w973018/g12 (real onsets 90/10/70 s from the S-CAPFIX
prepass). WM2 b8797242 (sha-verified), ERRSRC runtime fetched read-only from
cpuP into `_parity3_data` (untracked), exactly two physics files overlaid from
this branch (SHA-bound; pristine copies kept), the S-CAPFIX diagnose seed
recipe, one 600-step rollout per world per flag (~90 s each, CPU):

| world, flag | hard-brake rows (a ≤ −3) | of which near-merge | legacy ≤2 m phantom rows | WM discharge change @ real onset | one-step accel MAE (60 s) |
|---|---:|---:|---:|---:|---:|
| g0 OFF | 0 | 0 | 55 | −360 veh/h | 0.193 |
| g0 ON | 2 | 2 | 12 | −360 | 0.193 |
| g3 OFF | 0 | 0 | 1 | +720 | 0.184 |
| g3 ON | 0 | 0 | 0 | +720 | 0.184 |
| g12 OFF | 0 | 0 | 1 | −480 | 0.127 |
| g12 ON | 0 | 0 | 5 | −240 | 0.127 |

Real changes: +360 / +360 / −120 veh/h; WM absolute discharge stays ~2-3× the
real (crossings 142-177 vs 49-63 per window). Validation: the OFF rollouts
reproduce S-CAPFIX's independent diag on cpuP (g0/g3 exactly — onsets,
pre/post counts; g12 differs by one 10 s detector bin). Flag-OFF rollouts are
by construction bitwise the runtime's frozen physics.

**Verdict on the capacity-drop question: phantom leaders do NOT explain the
WM's invented capacity drop.** The chain rule removes almost all ≤2 m
corridor-leader rows in the WM2 rollout frames (55→12 in g0, ≈0 elsewhere),
yet hard-braking is essentially absent in the WM2 HDV rollout either way
(0-2 events; real near-merge shows 1-5), the post-onset merge-exit discharge
moves only within world-level noise (pooled 3-world WM change −40 → +40 veh/h
vs real +200; each world's error direction unchanged), and the per-tick
acceleration MAE vs the real tape is identical to 3 decimals. The invented
drop (errsrc −297 vs real +68 veh/h) lives in the WM2 rollout's queue/entrant
dynamics — S-CAPFIX's diagnosis domain — not in the corridor leader
identification. The phantom-leader mechanism was real but acts on the **CAV
execution mirror** (the parity gate, §3), where fixing it cut the executed
RMSE 0.2679 → 0.1465.

## 6. Notes

* The receipt's `shared_module_sha256` still pins `sumo_idm.py` (unchanged);
  the leader-rule modules are pinned by the overlay receipt inside
  `dynamics3_results.json` and by the git history of this branch.
* `_parity3_data/` (fetched runtime + 3 fit windows, ~8 MB + extraction) and
  `_execparity_data` stay untracked, like the execparity2 tapes.
* Deployment semantics: imagination (mode 2) with `C4_SUMO_LEADER_RULE=1` sees
  the same leader as the deployment executor on 98.6 % of rows; enabling the
  flag is a MAIN decision (it changes WM physics lane searches for ALL
  vehicles, measured §5 as near-neutral on fit rollouts).

## Reproduce (CPU torch env, `PYTHONUTF8=1`, `PYTHONPATH=src;.`)

```text
python tools/execparity/alwayscmd_replay.py --jobs _execparity_data/execparity/always0020/jobs --driver configs/execparity_cav_idm_v1.json --identity _execparity_data/execparity/always0020/jobs/s977000_always0/CTRL0_IDENTITY.json --out tools/execparity/alwayscmd3_replay_results.json --receipt tools/execparity/alwayscmd3_execution_parity.json
#   with C4_SUMO_LEADER_RULE=1 in the environment (recorded jobs dir fetched in S-EXECPARITY/2)
python tools/execparity/dynamics3.py --groups 0,3,12 --steps 600 --out tools/execparity/dynamics3_results.json
python -m pytest tests/unit/test_execparity3.py tests/unit/test_execparity.py tests/unit/test_execparity2.py tests/unit/test_execparity_alwayscmd.py tests/unit/test_execparity_alwayscmd_replay.py tests/unit/test_execparity_continuous.py -q
```
