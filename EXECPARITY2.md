# S-EXECPARITY2 — imagination executes the deployment executor's own law

**Status: law sharing COMPLETE and verified exact; registered parity criterion
still FAILS on the WM frame's leader identification.** Branch `c4/execparity2`
off `c4/execparity` HEAD `ee20ff42`, worktree `E:/agent-work/c4-execparity2`.
CPU only, **zero SUMO executions, zero sim-s, zero new charges**; the offline
replay re-uses the fetched 00:20 recordings (32,617 rows, unchanged row
mapping). Imagination executed-accel RMSE improves 0.2556 → **0.2679 pooled is
WRONG to read as improvement** — it is 0.2679 vs the native-head band 0.1603:
FAIL pooled and in every world, so **no gate receipt was written and
`verify_parity` keeps mode-2 deployment refused**. The dominant 00:20 error
source (the TRAIN-learned IDM base law, RMSE 0.154 vs the executor's base) is
now ELIMINATED (exact-function base error 1.6e-6 on leader-matched rows); the
remaining 95.6 % of the pooled squared error is the WM frame's leader
identification in the merge zone (state, not law).

## 1. The exact base-law function and parameters of the deployment executor

Read from the executor source only (`src/wms/control/nativeinit.py:72`
`sumo_native_accel`, called by `wms.control.alwayscmd.commands` and the frozen
`tools/execparity/alwayscmd_adapter.py`; CAV vType from OUR controller design
`traffic_shepherd/c4/population.py` — no vType assignment file, no
hetdata_truth were opened). For a commanded CAV `vid` at tick start:

1. `speed = getSpeed(vid)`; `leader = getLeader(vid)` with SUMO's gap
   convention `gap = bumper_gap − minGap` (MSVehicle::getLeader subtracts the
   ego minGap);
2. `target = getFollowSpeed(vid, speed, gap + minGap, leaderSpeed,
   leaderDecel, lead_id)` — the **frozen legacy double-add** retained by
   S-CTRLARM CTRL0 bit-identity; with no leader: gap `inf` and the EGO speed
   as leader speed;
3. `target = min(target, getAllowedSpeed(vid))`; base = `(max(0, target) −
   speed)/0.5`.

`getFollowSpeed` is evaluated by SUMO's own car-following model of the CAV's
vType — **our design**: `cav_typical`, `carFollowModel="IDM"` with
`TEXTBOOK_CAV_PARAMS` = accel 1.2, decel 2.0, tau 1.2, minGap 2.0, delta 4,
speedFactor 1.0 (copied in `wms/model/cav.py:TEXTBOOK_CAV_SETTINGS`; unit
tests assert the equality). The exact formula is therefore
`MSCFModel_IDM::_v` of eclipse-sumo v1_27_1 with those parameters:
`myIterations = max(1, int(0.5/0.25 + .5)) = 2` substeps,
`twoSqrtAccelDecel = 2·sqrt(a·b)`, `NUMERICAL_EPS = 1e-10`, `respectMinGap`
(IDM adds minGap internally once), `desSpeed = getLane()->getVehicleMaxSpeed`
= lane speed limit × speedFactor. Lane limits of the registered merge network
(`assets/research/highway_merge_shepherd_long/osm.net.xml.gz`): mainline and
merge-edge lane 2 (s ≥ 0) **31.29** m/s (`SPEED_LIMIT_MPS`), ramp edge
`11553048#1` (lane 2, s < 0) **22.22** m/s, ramp→mainline internal connector
`:1780739017_0_0` **26.75** m/s. Everything is a designed vType parameter, a
network lane attribute or a SUMO source literal — no hand-set behavioural
threshold, nothing learned.

## 2. The shared module; imagination calls the SAME function

`src/wms/control/sumo_idm.py` is the single shared implementation:

* `follow_speed` — exact `MSCFModel_IDM::_v` (torch, differentiable) with the
  `CAV_PARAMS` design table; scalar twins evaluate the same torch ops on 0-dim
  float64 tensors;
* `base_accel` — the exact executor base law (legacy double-minGap gap
  convention, `getAllowedSpeed` clamp, `(max(0,t)−v)/dt`);
* `safe_speed_cap` — the exact speedMode-27 executed-speed ceiling:
  `vNext = max(0, min(v + issued·dt, vSafe, maxNextSpeed))` where `vSafe` is
  the CAV's IDM follow speed with SUMO's internal SINGLE-minGap add
  (`gap2pred = bumper_gap − minGap`), plus the registered ramp lane-end stop
  (`MSCFModel_IDM::stopSpeed` toward the lane-2 dead end at `MERGE_S1_M`) and
  `maxNextSpeed = min(v + 1.2·dt, allowed)`;
* `executed_accel` — the composed always-commanded law
  `max(−v/dt, min(clip(base+r, min(lo,base), max(min(hi,1.2),base)), cap))`.

`wms.control.alwayscmd.inferred_base` / `inferred_cap` / `executed` are now
thin wrappers that feed the WM state of the CAV and its leader
(`native_cav.leader_gap_dv`: bumper gap + leader speed; lane/s allowed speed)
into those shared functions; `scalar_executed` is the same composed law for
fixtures/offline replay. The TRAIN-learned IDM driver mirror
(`configs/execparity_cav_idm_v1.json` values) is RETIRED from the imagination
path (the artifact stays registered and gate-bound via
`C4_EXECUTION_DRIVER`, unchanged). `state[EXEC_CAP_KEY]` is no longer
consulted in mode 2: imagination executes commanded CAVs through the
executor's own function, not the WM cap machinery. The deployment-side TraCI
composer (`alwayscmd.commands`, `nativeinit.sumo_native_accel`, the frozen
adapter) is UNTOUCHED — the canary bit-identity guarantee and the recorded
runs' provenance are preserved; the shared module is its verified exact twin.

**Identity evidence (offline, recorded executor outputs):**
`tools/execparity/execparity2_base_validation.json` replays the executor's own
32,887 ledger rows through the shared function: base law **99.985 % bitwise
exact** at 1e-9 (mean error 1.0e-8, RMSE 1.06e-6; the 0.015 % residual is
junction-internal-lane allowed speed, unobservable in a WM frame), and the
speedMode-27 composed realised speed is exact on 94.5 % of rows (accel-domain
RMSE 0.060).

## 3. Unit tests

`tests/unit/test_execparity2.py` (new, 9) + updated
`test_execparity_alwayscmd.py`: C++ transcription guard of `_v` against an
independent python-math re-derivation; **executor-side identity**
(`sumo_idm.base_accel_scalar` bitwise `== sumo_native_accel` on a mock TraCI
domain whose `getFollowSpeed` is the shared `_v`, incl. the legacy gap+minGap
convention and the no-leader branch); tensor/scalar identity of the shared law
(bitwise on ≥85-98 % of random rows, <5e-9 everywhere — vectorised-vs-0-dim
`pow` last-ULP only); imagination `executed` == shared composed law on WM
geometry; **residual-0 == CTRL0 executed law** `max(−v/dt, min(base, cap))`;
cap composition == `max(0, min(v+issued·dt, vSafe, maxNextSpeed))/dt`; lane
allowed-speed rule (22.22/31.29/26.75); legacy `scalar_executed` signature.
50 execparity tests PASS; the 8-suite nativeinit/execcap/deploy/branching
selection passes with the same 6 pre-existing environmental failures as the
untouched base worktree (missing `NATIVEINIT_BASELINE` etc.).

## 4. Offline re-score on the same 32,617 recorded ticks — criterion UNCHANGED

`tools/execparity/alwayscmd2_replay_results.json` (registered rule: pooled
imagination executed-accel RMSE ≤ pooled WM native-head RMSE, canary identity
from `CTRL0_IDENTITY.json` — both unchanged):

| | world 977001 | 977002 | 977003 | pooled |
|---|---:|---:|---:|---:|
| n rows | 10,695 | 5,387 | 16,535 | 32,617 |
| imagination executed RMSE | 0.3133 | 0.2214 | 0.2491 | **0.2679** |
| native-head band RMSE | 0.1586 | 0.1898 | 0.1506 | **0.1603** |
| within band | no | no | no | **FAIL** |

(nightly-00:20 learned mirror for comparison: pooled 0.2556. The typical row is
now exact — median error 0.0 and MAE 0.096 → 0.035 — but the pooled RMSE is
essentially unchanged, because the 5.1 % phantom-leader rows dominate it under
either law: the exact executor function there reproduces SUMO's own hard
braking for a 0.1 m gap instead of the learned mirror's smoother profile.) **No parity gate receipt
exists; `verify_parity` still refuses mode-2 deployment.**

### Remaining error decomposition (`tools/execparity/execparity2_decomposition.json`)

Rows split by whether the WM frame's leader is the leader the executor used:

| group | n (frac) | base-law RMSE | imagination executed RMSE | head RMSE |
|---|---:|---:|---:|---:|
| leader matched | 30,942 (94.86 %) | **1.59e-6** | **0.0576** | 0.1525 |
| leader mismatched | 1,675 (5.14 %) | 0.6122 | 1.1558 | 0.2665 |

* **Base law: SOLVED.** On leader-matched rows the shared function reproduces
  the executor's base law to float32 evaluation precision (1.6e-6), and the
  imagination executes within 0.38× of the native-head band there (0.0576 vs
  0.1525).
* **Cap: small residual.** Within the matched group the mirror-from-issued
  RMSE is 0.0576 — junction-internal foe yielding, mid-lane-change leader
  switching and the 26.75 m/s connector tick, none observable from a WM frame
  (pre-registered limitation of the corridor frame).
* **State: dominant.** The mismatched 5.14 % of rows contribute **95.6 % of
  the pooled squared error**. Anatomy: ramp vehicles mid-merge whose corridor
  projection places them 0-2 m ahead of a mainline-lane ego (569 of them
  closer than SUMO's true leader, gap clamped at 0.1 m by
  `leader_gap_dv`) — SUMO does not count them as the ego's leader yet, so the
  executor's base/cap see the real ~9 m leader while imagination hard-brakes
  for a phantom one (floor −v/dt = −10.9 m/s²). This is a property of the
  WM frame's leader semantics (`native_cav.leader_gap_dv`, shared with the
  native head and the legacy twin), not of the execution law; the head keeps
  its band there only because its MLP output is clamped to [−3, 1.2].

## 5. Conclusions for the owner

The 00:20 registered follow-up option "register a TRAIN-only base-law refit"
is superseded: no refit can beat the executor's own function, which is now
shared and verified bitwise. Candidate A's mode-2 enable remains refused by
the unchanged gate. To pass the registered criterion the remaining lever is
STATE, not law: the frame's leader identification for mid-merge vehicles
(e.g. a registered leader rule for merge-zone slots that treats
laterally-in-transition vehicles the way SUMO's `getLeader` does) would have
to be designed and registered separately — nothing was tuned here, no
threshold was added, and the decomposition is reported instead.

## Reproduce (CPU torch env, `PYTHONUTF8=1`, `PYTHONPATH=src;.`)

```text
python tools/execparity/execparity2_validate.py --jobs _execparity_data/execparity/always0020/jobs --out tools/execparity/execparity2_base_validation.json
python tools/execparity/alwayscmd_replay.py --jobs _execparity_data/execparity/always0020/jobs --driver configs/execparity_cav_idm_v1.json --identity _execparity_data/execparity/always0020/jobs/s977000_always0/CTRL0_IDENTITY.json --out tools/execparity/alwayscmd2_replay_results.json --receipt tools/execparity/alwayscmd2_execution_parity.json
python tools/execparity/execparity2_decomposition.py --jobs _execparity_data/execparity/always0020/jobs --out tools/execparity/execparity2_decomposition.json
python -m pytest tests/unit/test_execparity2.py tests/unit/test_execparity_alwayscmd.py tests/unit/test_execparity_alwayscmd_replay.py tests/unit/test_execparity.py tests/unit/test_execparity_continuous.py -q
```

The receipt file is NOT written (criterion failed). Tapes/checkpoints remain
untracked; job hashes are inside the three JSON artifacts.
