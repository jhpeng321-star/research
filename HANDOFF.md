# S-C1SWITCH2 integration handoff

Authoritative design: `docs/paper_results/C1SWITCH2_DESIGN.md` and `tools/c1switch2/design.py`.
Protocol `S-C1SWITCH2/v2`: **16 fresh main worlds 969136..969151**, canary969152,
six controllers in order `native,ctrl0,mappo,pi,circles,wall`, decisions300..880/20s.
The panel exports seed,penetration,demand and the common measurement index per world.
It replaces the unregistered preliminary12-world proposal seen by S-SELMAPPO.

## Evaluation ownership / selector peers
- **C1SWITCH2 owns all evaluation physics**, including both RL arms, once final weights exist.
- S-SELMAPPO and S-SELMAPPO2 should supply final checkpoint/config/source SHA256 receipts;
  keep their prospective evaluation episodes unstarted and reuse this task's paired receipts.
- This avoids duplicate comparator episodes and pays for the J900 tail here.
- Each main episode runs native300,30x20s choices, then last choice held300s to t1200.
  Primary J600 is t300..900; secondary J900 is t300..1200, with NO additional decisions.
- Selector ABI: frozen `select(anonymous_payload)->controller_name` for fleet PPO;
  per-CAV returns `{self_reported_cav_id:controller_name}` for exactly the current live CAVs.
- Per-CAV replay must rebuild its actual allocation/memory/entrant default rule causally;
  S-SELMAPPO2 supplies its validated grouped execution bridge. It is not simulated by fleet actions.
- Training observations/reward remain those registered by each selector task. No online updates here.

## WALL ABI / S-WALLSW
Use the frozen wall-v2.1/v2.2 law; do not add thresholds in either picker.
The v2 backend injects a wall facade with:
- `create_sumo()->controller`, whose `start(env)`, `stop(env)`, `command(env)->(command,kwargs)`
  obey the shared `Controller('wall',inputs)` lifecycle proposed in the selector handoff.
- `capture_memory(controller,env)->state`: only causal controller-owned group/command state,
  CAV self IDs only; no simulator HDV identities, hidden settings or future requests.
- `validate_memory(state,own_ids,now)->same_state`: exact field/self-ID/time validation;
  accepts `None` for inactive WALL. Predictor RPC preserves this memory in its anonymous key.
- `validate_lane_commands(counts)`: attest wall-specific commands; original five keep v7's zero-lane gate.
- Engine imaginations must use the same assembly/release state and retain it for incumbent WALL;
  stop/reset ownership when leaving WALL; fresh start when re-entering it.

## Runtime / WM-BB engine gates
`runner.py` loads hash-bound `runtime`, `predictors`, and `selectors` entrypoints from BINDING.json.
Runtime factory `(binding)->runtime`; `runtime.create_backend(path,world,ledger,stage,id,arm=... )`.
Fleet backends can reuse `tools.c1switch2.backend.Backend(...,wall=facade)` after frozen v7 setup.
The per-CAV runtime must supply the validated selector bridge with the same backend methods.
All construction happens AFTER current registration approval and integration admission.

`inference.Predictor` expects an engine attestation and
`imagine(payload,controller=...,K=8,horizon_s=60,seed=anonymous_seed,wall=facade)->8 team delays`.
The six candidates get identical anonymous CRN seeds, isolated controller memory and true feedback.
The v7 entries-OFF adapter predicts road delay only; it CANNOT satisfy this full-team contract.
Do not label road-only values as team delay or fill unknown source costs with zero.
Real full-team/source-admission engine bindings and GPU parity remain unresolved dependencies.
No WM-PHYS activation without its independently hashed S-RB-EVAL qualification receipt.

## Current execution state
Build and CPU preparation only; no task SUMO/GPU/physics processes.
The 23:48 owner approval is present, but WALL handoff and final trained selectors are pending.
`python -B -m tools.c1switch2.gate --polls 1` reports approval without authorizing an unbound runtime.
CPU preparation/exposure receipts are in `tools/c1switch2/evidence/`; prior versions are preserved.
The initial local exposure regex reported timestamp/negative-evidence false positives, fixed prospectively.
An exact-file Grep call expanded into the parent tree; incidental numerical matches were not used.
All subsequent registration reads/audits use exact checked paths; filename scans prune before traversal.
