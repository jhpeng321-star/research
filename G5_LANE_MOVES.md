# G5: lane-change group moves (CPU build)

Enable with the common B/R backend configuration `group_lane_moves: true`.
The default is false. A uses its existing policy/controller. Existing five-family
trigger checkpoints retain their shape when off; enabling G5 creates the
seven-family trigger descriptor layout and requires a matching new G5 sidecar.
No existing launch registration is changed by this build.

## Move and execution API

- `moves.LANE_FAMILIES = ('yield_lane', 'open_gap_merge')`;
  `moves.ALL_FAMILIES` includes the original five families. The default
  `moves.FAMILIES` remains the original vocabulary for checkpoint compatibility.
  `group_lane.families(cfg)` resolves the enabled vocabulary.
- `moves.Move(family, (mainline_id, ramp_id), onset, hold, depth)` retains the
  normal serialization and registered variety parameter windows. Both members
  must be observable, acquirable CAVs. HDVs are sensing/gap obstacles only.
- `moves.candidates(obs_state, rng, n)` accepts `group_lane_moves=True` and
  explicit `observed_lanes` in the local packet. G5 uses the existing F2
  balanced feasible-family slate and conditional ownership masks. None and
  singleton options remain available. B and R use identical code and sampling.
- `yield_lane`: mainline CAV attempts WM lane 1 -> 0 at onset, freeing the
  right lane for the ramp partner; acceleration offsets are zero.
- `open_gap_merge`: mainline CAV behind the ramp partner receives `-depth`
  throughout `[onset, onset+hold)`; ramp CAV attempts WM lane 2 -> 1 one
  registered control tick after onset. The ramp's acceleration is unchanged.
  This is sequencing, not an additional safety threshold. There is one lane
  attempt per macro. A blocked attempt is recorded and is not forced or retried.
- `group_lane.targets(move, elapsed)` returns absolute WM targets at the attempt
  tick. Parameters and look-ahead are not shortened to the attempt time.
- `group_lane.plan_fields(move, duration)` returns `g5_move`, `lane_actions`
  (member x second, 0 keep / 1 left / 2 right), and `uses_lane_actions`. Add these
  to the normal full-duration verified profile plan. `moves.from_plan` checks
  exact role order, lane timing and policy-relative offsets; it never silently
  turns a lane-bearing plan into an acceleration-only label.

The execution adapter reuses `c4/lane-policy` / `c4/lane-plans` integration
`e5740395`: `controller.lane_request(env)` -> the existing
`tools.coopfree.backend.tick` -> **one** `CavSumoEnv.step(accel, lanes=...)`.
The new module contains the adapted tensor/outcome helpers and the observable
version of the existing `lane_feasibility(..., LaneRules())` envelope. It loads
no lane model or policy lane head. The environment remains the only TraCI
writer and retains laneChangeMode 513, edge/junction guards and SUMO safety.

Each commanded actor's own sensor packet is checked again at execution. Missing
target-lane coverage, HDV identity, changed source lane, unsafe front/rear gap,
or an edge guard blocks the request. All sensed vehicles, including CAVs outside
the partner acquisition slots, remain gap obstacles. Native all8 verification
and practice plans can execute bound `g5_move` profiles through the same V2V
adapter using locally clipped sensing. Recorded prefix samples replay their
lane requests with live safety rechecks. Counterfactual member release applies
to lane requests as well as acceleration. Unwrapped acceleration controllers
refuse G5 instructions instead of silently dropping their lane effect.

Telemetry:

- `sample.g5_lane_requests`: attempted absolute WM targets;
- `sample.g5_lane_gate`: safe targets and per-CAV refusal reasons;
- `event.g5_lane_outcome`: actual tick requested / issued / blocked / executed
  aggregate and raw environment counters. Issuance is not reported as execution.

## Verified mined-example input

With G5 enabled, F3 `bank_breadth.inputs/visit` additionally consumes
`<run_root>/MINED_MOVES.jsonl`, or the explicit `mined_moves` path. A missing
default file adds no inputs; a missing explicitly configured file is an error.
Both arms read the same file. Paths are relative to the JSONL file unless
absolute. Each line has this v1 pointer contract:

```json
{
  "schema": "coopmwm-mined-move-v1",
  "arm": "B",
  "run_root": "verified_lane_bank",
  "verification_id": "native-verification-id",
  "plan_sha256": "<64 lowercase hex characters: protocol.digest(verified_plan)>",
  "replay_sources": {"B": [], "R": []}
}
```

Provide R's own verified references as R rows in the same file. Several plans
may share a verification ID (per-seed winners); each digest is a distinct
binding. Duplicate rows are deduplicated. These are source pointers, not raw
labels or self-certifying `verified: true` records. Missing native examples or
plan-hash mismatches fail admission. Restricted paths are rejected before reads.

The existing native reader still checks receipt/confirmation admission,
cooperation, own-seed tape hashes, effective-command limits and observable
contexts. For G5, the selected tape must also have an immutable result hash,
the expected lane request, an issued-and-completed actuator receipt, and the
commanded CAV's actual next-frame lane transition. Acceleration-only wins and
blocked lane attempts cannot become G5 lessons. F3's matched B/R family support,
common family weighting and immutable restart binding remain in force.
Late contexts do not restart an already-underway lane macro.

## CPU verification

Use `C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest -q`
with the explicit suites and counts recorded in `G5_READY.json`. New tests are
in `tools/coopmwm/test_group_lane_cpu.py`; simulator-facing calls use CPU fakes.
No physics is launched by imports, proposal construction or bank input parsing.
