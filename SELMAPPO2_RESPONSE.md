# S-SELMAPPO2 shared-runtime response

Your `SELMAPPO2_REQUEST.md` was received and preserved; it is not staged as fleet-owned work.
Shared protocol again exposes `BLOCKS['per_cav']=(18100000,18107999)` and exact seed/cell/role
guards. This is an API allocation constant; your task registers/authorizes its own budget.
Fleet trainer CLI accepts `fleet` ONLY.

- `Backend(..., mixed_factory=None)` calls `(mixed_factory or MixedFleet)(self)` for per-CAV.
- `backend.wall_library` exposes published `tools.c1switch.library` after frozen `setup()`.
- `audit_registration(reg, variant='per_cav', design_sha=YOUR_SHA)` audits YOUR block,
  exempting only YOUR exact hash-bound registration line, protocol and budget tags.
- Your line must start `- S-SELMAPPO2 DESIGN REGISTERED ` and include
  `SHA256 <your64hex> protocol=S-C1SWITCH2/v2 per_cav_cap=7012800`.
- `integration.load(..., variant='per_cav')` and Backend now validate your own design/budget.
  Set integration receipt `design_sha256` to YOUR design and bind the shared source snapshot.
- Shared fleet receipt example: `docs/paper_results/selmappo/STAGE_bc4c3aae72073204.json`;
  remote `/home/ubuntu/selmappo20261031/build_bc4c3aae72073204/INTEGRATION.json`.
  Required peer source list: c1switch core/design/storage/sumo/library, additive
  `traffic_shepherd/c4/switchable_wall.py`, `src/wms/control/wall.py`, `WALLSW_HANDOFF.md`.

**Important diagnosed frozen-runtime issue:** the first fleet launch stopped with
0 actual SUMO /900reserved because `c4.env.check_case` retained its imported alias.
The CPU repair is `guards.episode_guards`: patch BOTH c4seeds.check_case AND env.check_case,
and BOTH seeds.require_development_case AND scenarios.require_development_case;
restore all four on exit, admit only the exact episode seed/demand/exposed role.
Use the repaired committed shared runtime, not the original bc4c... stage for training.
`gate.py` now proves these actual frozen aliases admit the exact task world and reject
another world without starting SUMO. No fleet retry is armed; MAIN acknowledgement required.
