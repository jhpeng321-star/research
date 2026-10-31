# S-C1SWITCH2 integration handoff

Authoritative design: `docs/paper_results/C1SWITCH2_DESIGN.md` and `tools/c1switch2/design.py`.
Protocol `S-C1SWITCH2/v2`: **16 fresh main worlds 969136..969151**, canary969152,
six controllers in order `native,ctrl0,mappo,pi,circles,wall`, decisions300..880/20s.
Use `docs/paper_results/c1switch2/PANEL_v2.json` (16 worlds, unchanged identities).
SHA256 `7c780c25237dde73a657937ab577b8e48f71766817f4363f3298982e90144827`.
It exports seed,penetration,demand, common measurement indices and registered design SHA.
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
Integrated S-WALLSW through `1b0581d0` (handoff `07fb123e`); frozen v2.1/v2.2 law.
The v2 backend injects a wall facade with:
- `create_sumo()->controller`, whose `start(now_s=...)`, `stop(now_s=...)`, `command(env)->(command,kwargs)`
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

`inference.Predictor` accepts attested `imagine_all(payload,K=8,horizon_s=60)` six-row batching,
or isolated `imagine(...)` per controller. `engine.Engine` implements real frozen WM2/BB feedback.
The six candidates get identical anonymous CRN seeds, isolated controller memory and true feedback.
MAIN00:29 permits a registered COMMON ROAD-ONLY prediction objective for both WM and BB.
Applied explicitly: entries OFF; mix configured but inactive; no source cost prediction.
Full-team J600/J900 and full-team branch oracle remain the evaluation objectives.
`selectors.py` bridges the fleet peer's FrozenFleetSelector and per-CAV save_checkpoint ABI;
`runtime.py` wires the per-CAV owner's make_mixed_fleet and shared frozen executor.
`SELECTOR_BINDINGS_PENDING.json` documents missing final169 weights: stubs raise, never pick.
Supply final checkpoint + immutable freeze/source/config receipts, not intermediate checkpoints.
Grouped execution/main admission and both-GPU parity remain gates.
Both real frozen checkpoints now PASS the CPU K8/60s six-way feedback probe on gpuR;
receipts `MODEL_CPU_{wm,bb}_steps120_c1switch2_run_7e448ccdafc0c1c3.json` preserve provenance.
Per S-SELMAPPO2's `SELMAPPO2_HANDOFF.md`, per-CAV features synchronize from live
backend contexts before EACH pick, including entrant/re-entry resets and retirement.
No WM-PHYS activation without its independently hashed S-RB-EVAL qualification receipt.

## Current execution state
Latest resume state: `tools/c1switch2/evidence/STATUS_CURRENT.json` plus task-only `operations status`.
Mechanics-only canary is RUNNING: owncpuQ PID467748, root
`/home/ubuntu/c1switch2_run_4ad4ece01d5a9ac1`; cap4680 sim-s, GPU0. Never redispatch it.
Initial PID446935 exited before setup/physics with zero charge; all receipts retained.
Repair1065f89d was explicitly registered and passed pristine-process CPU construction before dispatch.
The 23:48 owner approval is present; WALL integrated, final trained selectors pending.
`tools/c1switch2/wall.py` binds the real peer lifecycle and self-ID-only memory;
missing members remain absent and abort atomically on continuation. No simulator slots cross RPC.
Latest cpuQ AND gpuR CPU verification: 127 passed/1 host-only skip each.
Tests include real wall memory parity, six-row synthetic WM/BB mechanics and actual peer
feature/model interfaces with clearly synthetic final169 test weights (not trained/evaluated arms).
`python -B -m tools.c1switch2.gate --polls 1` reports approval without authorizing an unbound runtime.
CPU preparation/exposure receipts are in `tools/c1switch2/evidence/`; prior versions are preserved.
Old cpuQ `/home/ubuntu/c1switch2_run_2d2400c953ef2545` is a retained zero-physics STOP.
gpuR `/root/c1switch2_run_7e448ccdafc0c1c3` is CPU-only; no GPU services or main jobs armed.
New CPU-only snapshots `c1switch2_cpu_6b84f28df5d4` on both hosts pass127/1skip,
including the actual peer grouped-WALL retirement bridge and bound two-GPU RPC factories.
`operations.py status` inspects task-only dispatch receipts; canary is a write-once
cpuQ claim using the original4680 sim-s reservation, never an automatic retry.
The initial local exposure regex reported timestamp/negative-evidence false positives, fixed prospectively.
An exact-file Grep call expanded into the parent tree; incidental numerical matches were not used.
All subsequent registration reads/audits use exact checked paths; filename scans prune before traversal.
