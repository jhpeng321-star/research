# S-COOPBEHAV trace handoff

## Corrected v2 (authoritative)
- gpuR `/root/autodl-tmp/coopaudit20261101_v2/v2_sSEED/traces/*.jsonl.gz`;
  same directories linked from `/root/coopaudit20261031/jobs/v2_sSEED/`.
- Corrected canary: `v2_canary_s969699/traces/`; production seeds969650-969677.
- Every trace is from a fresh full native300s prefix +600s continuation, no loadState.
  60s deployed-mean intervention, native release at360s; pair frozen at300s.
- `requested_accel` now records issued-safe requests for MAPPO and SUMOCF as
  well as wall. It is not uncapped actor output or realised acceleration.
- Actual 0.5s timestamps cover (270,390]s; `a` is realised acceleration.
  Nearby ramp identity is tracked from observable lane2 reports through its merge.
- Portable local exports: `evidence/v2_canary/jobs/` and
  `evidence/v2_complete/jobs/` (after completion). Colons in ONLY:i filenames map
  to underscores on Windows; bytes/metric keys are unchanged, mapped paths and
  exact remote originals are in COLLECTION.json. Never infer i from slot order:
  use `FROZEN.json.group[i]`. Final production TRACE_MANIFEST.csv binds every file.
- Wall canary C_G600+1658.239758; both solos harm, joint helps. This is one
  selected mechanics world, not a panel estimate. Old saved-state credits below
  are invalid for scientific cooperation inference, although raw traces survive.

## Historical preserved data (superseded)

Own host/root: gpuR `/root/coopaudit20261031`.
Canary: `jobs/canary_s969699_attempt2/traces/*.jsonl.gz`.
Historical main: `jobs/main_s969650/traces/*.jsonl.gz` (one completed world).
**17:20 artifact HOLD:** all traces above are preserved. This main world used the
superseded grid6/two-member/60s/five-arm design. It is descriptive historical
evidence, not a corrected-design audit estimate. No new main worlds are being
dispatched; use `evidence/HOLD_VERIFIED.json` and `RESUME.md` for current status.
Corrected-design traces will receive a separately registered versioned namespace.

One compressed JSON-lines file per controller/subset, native B included. Each
row is one RSU-observable vehicle at a 0.5s physics frame: absolute `t`, stable
`vid`, downstream-increasing `s` metres, `lane` (0/1 mainline,2 ramp,-1 junction),
`v` m/s, realised `a` m/s², frozen `member` boolean, `requested_accel` for wall
commands (null for other controllers). Realised `a` is NOT the policy's raw
request. Production additionally tags `ramp_observed` and retains identities
observed on lane2 after they merge into lane0/1, so merge-event trajectories are
not cut at the lane change. No hidden type or simulator assignment is exported.

Coverage: t0-30 through intervention end+30. Include every frozen member and
ramp vehicles inside the observed approach/merge zone or within300m of a live
member. Prefix from native300s warmup is shared. Nearby ramp membership is
geometric/time-varying; group identities are fixed in `FROZEN.json` before acting.
`branch_CONTROLLER_SUBSET.json` and `result.json` supply wall lifecycle events,
clocked safety/cost totals, local cohort identities and trajectory hashes.

`ONLY:i` refers to the i-th identity in `FROZEN.json.group`; GROUP includes that
entire frozen set. Wall uses `wall_roster` (one geometric cross-lane pair) and
keeps that context in solos while disabling the other commands. Non-roster
group members are inert for wall. Native/IDM release after the window applies
to every branch. Behaviour events are descriptive; use exact C_G for causality.
