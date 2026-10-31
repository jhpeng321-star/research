# S-WALLSW implementation plan (CPU only)

Owner task: 2026-11-01; branch `c4/wallsw`, based on `c4/c1switch`.

## Recovered contracts

- `traffic_shepherd/wall_group.py`: frozen `wall-v2.1-closure` (closure targets,
  bounded pacing, 60 s assembly timeout). The file is unchanged against 7a12e126.
- `traffic_shepherd/wall_group_v22.py`: frozen `wall-v2.2-merge-release`, unchanged
  against bda1a5dc. The wall meters upstream inflow and releases at the merge START,
  cumulative road coordinate 2644.2 m, not the merge end.
- ab555c24 records the synthetic golden parity harness. bf3f330b binds historical
  data eligibility to the frozen v2.1 law; it does not define another runtime law.
- `src/wms/physics/wall_exec.py` already mirrors both laws; its v2.2 executor is
  used by `traffic_shepherd/c4/baselines.py::WallLaw` in the SUMO baseline.
- The existing library interface is `Controller.command(env) -> (accel, kwargs)`;
  `Handover.select` releases `CommandOwnership` and clears PI state on a change.
- The baseline's EXISTING formation rule (WallLaw, including its constants),
  nonmember CIRCLES fallback, and cooldown will be reused without new behavioural
  thresholds. This is the C4 baseline formation policy, not the historical
  campaign proposer's alternative speed-factor/D_min policy.

## Steps / commits

1. Record this source/interface plan.
2. Add an episode-local switchable wall object, explicit lifecycle stop, opt-in
   library integration, and a plant-independent WM callback. Preserve the original
   five-arm experiment registration/budgets; consumers opt into the expanded library.
3. CPU-check arbitrary starts/stops, every peer handover, single-writer ownership,
   unchanged safety boundaries, slot identity, branching/deepcopy, and independent
   frozen-session parity. Fix any failures and commit the checks.
4. Deliver `HANDOFF.md` with deployment, interface, source pins and test evidence
   for S-C1SWITCH2 and S-SELMAPPO/S-SELMAPPO2.

## Execution constraints

CPU only: no SUMO allocation or remote launch is authorised here. New SUMO debit
is zero. A future approved canary is at most two worlds / 3600 total sim-s, charged
once to S-C1SWITCH2. Use the existing Miniconda CPU environment; no new venv.
Only explicit permitted code/test files are read; restricted data paths and truth
parameter/assignment files are excluded. No process signalling, push, reset,
clean, stash operations, or file deletions.
