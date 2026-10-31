# S-COOPBEHAV trace handoff

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
