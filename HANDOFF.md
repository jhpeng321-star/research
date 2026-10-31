# S-SELMAPPO fleet-picker integration handoff

**CURRENT: MAIN01:48 continuation STOPPED before physics on HISTORY row reader; own606965 exited.**
New build3351504cece84e4b passed38/38CPU/real integration and registered before its
one-use dispatch. All5unused worlds were precharged then failed at budget.reserve
BEFORE simulator construction/reset; newactual0/newcharge4500 retained. Total43200charged,
actual historical30600..37800bounds unchanged,0PPOupdates/initial000only/no evaluation.
Original41worlds AND new5 reservations are consumed; all4locks/oldopenledgers retained.
CPU fix adds HISTORY-safe worker row lookup and a real worker-with-HISTORY regression.
MAIN schedule amendment required; no automatic launch/retry armed. See
`MAIN_CONTINUATION_REQUEST.md`, `SELMAPPO_STOP.md` and STOP_build_3351504cece84e4b.json.
CPU-only repair stage36c71820cd9e03d8 now PASS38/38 +real integration/registry/initial
parity, no SUMO. It is not dispatched; no continuation implemented or armed.

**Previous transport stop (preserved): own503239 exited, no descendants.**
Production began under MAIN01:08, but first wave exhausted PyTorch storage-FD transport
near32x30logp tensors against cpuQ softFDlimit1024 (`received0items of ancdata`).
41production worlds reserved,33complete result/actual receipts,0PPOupdates; initial000only.
Charged38,700sim-s; actual bounds30,600..37,800 (8killed workers lack terminal receipts).
Ledger retains9open reservations; original ledger/logs/results/locks remain immutable.
No final169 weights, no evaluation, no new retry authorization/automatic launch.
Byte transport/durable trajectory repair passed CPU35/35 +real frozen checkpoint/WALL/alias
integration with SUMO intercepted. CPU-ONLY stage`build_59a85b1215ade170`, fullbinding
`59a85b1215ade17045f65ea36f22f3f3c418aa3521e9d62094d9190ca5eae123`.
It is NOT DISPATCHED; the old production lock prevents relaunch. See`SELMAPPO_STOP.md`.
MAIN00:29 continuation was used once (commit `c082b233`); second seed reached900actual,
then the wrapper queried the frozen environment's already-closed clock. Total1,800charged /
900actual, production0/checkpoints0. Terminal repair commit `d219f5ca` is CPU31/31 plus
real frozen-controller/WALL/alias-gate PASS, staged ONLY at `build_68394a9568862929`.
MAIN acknowledgement is now present in REGISTRATION.md; both smoke slots are spent.
Fresh-session reconciliation confirmed own357398 exited and no fleet trainer at01:09+08:00.
A separate one-use production-only continuation passed CPU34/34 plus the real frozen
controller/WALL/seed-alias gate on cpuQ. Immutable stage
`build_b3d13ccfae77de71`, source binding
`b3d13ccfae77de71cd8ff70d2ee444de514104855652968aa70965a54aae4aba`.
Lowest-priority nice19 dispatch must yield above0.8xcores. See `SELMAPPO_STOP.md`.
Exclusive`FLEET_PRODUCTION_ONCE.json` claimed; receipt
`docs/paper_results/selmappo/DISPATCH_build_b3d13ccfae77de71.json`.
Both historical mechanics charges inherited unchanged; NO more mechanics run.
First production wave starts18000000; first status has10reservations and resource pause
atload15.265+pending10 on32cores (ceiling25.6). No successful update reported yet.
Status: `python -m tools.selmappo.launch status --build build_b3d13ccfae77de71`.
Do not launch another build or reuse old commands; MAIN must acknowledge this new stop.

Latest scope: **fleet PPO only**, `S-C1SWITCH2/v2`, decisions every20s at300..880,
library `native, ctrl0, mappo, pi, circles, wall`. Per-CAV belongs to S-SELMAPPO2.
Design: `docs/paper_results/SELMAPPO_DESIGN.md`, commit `7e4def77`, SHA256
`992c05601f03d1d9caad0421b041dbeaa05fa8621ae7e3020be42e3c9f5b41d0`.
Exact fleet allocation **7,012,800sim-s**, owner advance approval applies after registration.

## Shared code for S-SELMAPPO2
- Import/copy from `E:/agent-work/c4-selmappo/tools/selmappo/` at the committed build revision.
- `protocol.py`:6-way order,20s cadence,30 decisions; fleet TRAIN18000000..18007999.
  Shared per-CAV constant18100000..18107999 is retained for S-SELMAPPO2, which separately
  registers/authorizes its allocation. Fleet CLI accepts fleet only. See `SELMAPPO2_RESPONSE.md`.
- `observations.py`:pure `build(payload, variant='fleet'|'per_cav', incumbents=...)`.
  `Observation(keys, actors, shared)`;46shared features,60actor features
  (46summary+6incumbent+8ego). IDs route per-CAV actions only; no numeric ID features.
- `environment.py`:pure backend-driven30-window wrapper and separate full-team reward.
  Backend: `now`, `observable()`, `select_fleet(name)` / `select_per_cav(names)`,
  `tick()` (one physical second with both half-second contributions), optional `current_names()`.
- `handover.py`:fleet ownership release/reset/incumbent-preservation and WALL lifecycle;
  tests cover WALL start, incumbent assembly preservation, stop and PI reset.
- Runtime's retained per-CAV prototype is UNVALIDATED for WALL; per-CAV task owns that integration.

## S-WALLSW integration delivered
The handoff arrived during this session, at peer commit `07fb123e` (implementation
`1b0581d0`). Fleet runtime now directly reuses `tools.c1switch.library.make_handover(inputs)`;
the peer `Controller` lifecycle is `start(now_s=...)`, `stop(now_s=...)` and
`command(env)->(command,kwargs)`. It owns all assembly/release logic.
CPU gate loads BOTH real frozen controller checkpoints and exercises the real WALL
formation, incumbent preservation and outgoing reset, intercepting SUMO start/step.

`tools/selmappo/integration.py` expects a `selmappo.integration/2` JSON with:
`status=CPU_INTEGRATION_PASS`, protocol, cadence_s=20, library, design_sha256,
wall_module=`tools.c1switch.library`, and `files` (relative safe path -> SHA256) under `--c1-root`.
Bind the peer module, additive extensions and `WALLSW_HANDOFF.md` with controller/observer sources.
An approval string alone cannot start an incomplete five-controller runtime.

## S-C1SWITCH2 panel/checkpoint interface
Peer registered16worlds969136..969151 and explicitly owns ALL evaluation physics,
including fleetPPO and its300s held-last-action tail to t1200 for secondary J900.
Leave this task's prospective14,400sim-s reservation UNSTARTED and reuse peer receipts;
no duplicate main episodes or branches here. Await its exact hash-bound manifest/binding.
Supply ONLY final169 checkpoint/config/source SHA256 to the peer; no weights exist yet.
`selector.FrozenFleetSelector(checkpoint, expected_sha)` provides `reset()` per world,
then exactly30 `select(payload)->controller_name` calls at300..880. It verifies final169,
protocol/library/hash, uses the same builder/incumbent memory, frozen deterministic argmax,
and performs no scaler/gradient update. No selector calls during the peer's last-action tail.

## Preserved first dispatch
Original real-integration stage:
`/home/ubuntu/selmappo20261031/build_bc4c3aae72073204`, source binding
`bc4c3aae72073204cf046d800766d4fcc5fd9610a29ea099c3c40cfc195dce14`.
28/28CPU tests and real frozen-controller/WALL integration PASS; SUMO0 at staging.
Its first dispatch ownPID284174 exited on the frozen seed-alias refusal; never redispatch it
or reset the lock. Inspect using `python -m tools.selmappo.launch status --build build_bc4c3aae72073204`.
Repair stage `build_914fecba3ffdf4cc` remains CPU-only, never dispatched.
MAIN-authorized continuation used `build_999584a8cfacaf81`, ownPID357398 now exited.
Both original and continuation locks are retained; do not relaunch either build.
Current ledger hash `be3bcd873b401cce95b14a9659f33bcc8de38823eb8a6ca5a830119ca13347ce`;
terminal STOP receipt records1,800charged/900actual and incomplete mechanics result.
CPU-only terminal repair source binding
`68394a95688629292c307a20861c409d264a3c88104851b719caa5917d44e7a1` is NOT a launch.
The CPU-only stages are immutable receipts, not running jobs:
`/home/ubuntu/selmappo_cpucheck_9f8afc61e72189e6/CPU_RESULT.json` records28/28PASS,
including real CPU PyTorch optimizer, full600s return and RNG-isolation checks.
gpuR's matching CPU-only stage also passed28/28. Earlier local27tests:25PASS/2torch-onlySKIP.
Full home-root safe-name audits:cpuQ175,014/gpuR316,092;zero conflicts/errors.
Filename/registry audits are mechanical exposure checks,
not a claim that every historical dataset's file contents were searched.
