# S-SELMAPPO fleet-picker integration handoff

Latest scope: **fleet PPO only**, `S-C1SWITCH2/v2`, decisions every20s at300..880,
library `native, ctrl0, mappo, pi, circles, wall`. Per-CAV belongs to S-SELMAPPO2.
Design: `docs/paper_results/SELMAPPO_DESIGN.md`, commit `7e4def77`, SHA256
`992c05601f03d1d9caad0421b041dbeaa05fa8621ae7e3020be42e3c9f5b41d0`.
Exact fleet allocation **7,012,800sim-s**, owner advance approval applies after registration.

## Shared code for S-SELMAPPO2
- Import/copy from `E:/agent-work/c4-selmappo/tools/selmappo/` at the committed build revision.
- `protocol.py`:6-way order,20s cadence,30 decisions; fleet TRAIN18000000..18007999 ONLY.
  The per-CAV task must register and guard its own block; the old prospective18100000
  block is not an allocation made by this fleet task.
- `observations.py`:pure `build(payload, variant='fleet'|'per_cav', incumbents=...)`.
  `Observation(keys, actors, shared)`;46shared features,60actor features
  (46summary+6incumbent+8ego). IDs route per-CAV actions only; no numeric ID features.
- `environment.py`:pure backend-driven30-window wrapper and separate full-team reward.
  Backend: `now`, `observable()`, `select_fleet(name)` / `select_per_cav(names)`,
  `tick()` (one physical second with both half-second contributions), optional `current_names()`.
- `handover.py`:fleet ownership release/reset/incumbent-preservation and WALL lifecycle;
  tests cover WALL start, incumbent assembly preservation, stop and PI reset.
- Runtime's retained per-CAV prototype is UNVALIDATED for WALL; per-CAV task owns that integration.

## S-WALLSW integration needed
No `E:/agent-work/c4-wallsw/HANDOFF.md` was present at the resumed build checks.
The fleet bridge expects a hash-bound module `tools.wallsw.<module>` exporting
`Controller(name, inputs)` with `start(env)`, `stop(env)` and `command(env)->(command, kwargs)`.
The controller owns its assembly/release law; selectors add no behavioural thresholds.
If the published adapter uses another API, adapt the bridge after its handoff is available
and rerun integration CPU checks before any SUMO.

`tools/selmappo/integration.py` expects a `selmappo.integration/2` JSON with:
`status=CPU_INTEGRATION_PASS`, protocol, cadence_s=20, library, design_sha256,
wall_module, and `files` (relative safe path -> SHA256) under `--c1-root`.
Bind the WALL module and its HANDOFF.md along with the frozen controller/observer sources.
An approval string alone cannot start an incomplete five-controller runtime.

## S-C1SWITCH2 panel/checkpoint interface
Evaluation reserves16x900=14,400sim-s here; train first, evaluate only final169 checkpoint.
Charge/evaluate these fleet episodes ONCE and let S-C1SWITCH2 reuse the same receipts;
its per-decision replay branches are separate peer-budget work, not extra fleet episodes here.
Peer panel must be the registered hash-bound16world seed/cell manifest (not the old60s panel).
At the first read, prospective `tools/c1switch2/design.py` generated12worlds;
resolve its panel with MAIN before evaluation. This task does not allocate peer panel seeds.
Frozen selector bridge will expose `select(payload)->controller_name` for fleet_ppo;
same builder, incumbent memory, deterministic argmax, no scaler/gradient update.

## Resume safety
No selector training root or selector PID found on cpuQ/gpuR at resume; no SUMO dispatched.
The CPU-only stages are immutable receipts, not running jobs:
`/home/ubuntu/selmappo_cpucheck_9f8afc61e72189e6/CPU_RESULT.json` records28/28PASS,
including real CPU PyTorch optimizer, full600s return and RNG-isolation checks.
gpuR's matching CPU-only stage also passed28/28. Earlier local27tests:25PASS/2torch-onlySKIP.
Full home-root safe-name audits:cpuQ175,014/gpuR316,092;zero conflicts/errors.
Filename/registry audits are mechanical exposure checks,
not a claim that every historical dataset's file contents were searched.
