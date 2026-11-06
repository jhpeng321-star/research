# POLICY14E — policy14 on a round-13E world model (2026-10-25)

Branch `c4/policy14e` = `c4/policy14` (02475aa0) + merge of `c4/v10r13e` (97d22246). CPU only; no GPU, SUMO or launch.

## Merge
Conflicts were in the earlier 13E port that policy14 carried (`physics/{emergency,accel,config}`, `model/{merge_pivot,wms}`, `config/{train,legacy_flat}`). 13E (final-pre-clamp envelope B2, GAPACC gs without s0 B3, learned yield with pass rule) replaces the port. Policy14 additions kept: courtesy previous-accel gradient, `driver_params` forwarded to CAV callbacks (`inferred_driver_parameters`, `inferred_idm_delta`, gs `T`,`b`).

## Fixes
- `train/cav_drive.py`: native fallback now calls `physics_accel(state, params)` with the callback's learned settings when yield/envelope/gs is on (was `physics_accel(state)` -> ValueError with learned yield). Flags off: unchanged call.
- `control/lanes.py`: `secure_gap_gs` dropped the optional s0 term and the wms callback no longer exports `secure_gap_s0_m`, so the policy lane mask and the WM merge veto use the identical law (tested on 200 random scenes).
- No double counting: the consequence mask is a veto on the UNENVELOPED IDM response of the new follower; the envelope only bounds dynamics and the CAV execution cap. Mask output is identical with the envelope on or off (test).
- `policy14_probes.gradient_probe_13e` (CLI runs it only when a 13E flag is on): dJ/d(CAV accel) autograd vs FD in a closing-follower scene where the envelope binds.

## Evidence (S-R12C checkpoint, `configs/c4_wm_v10round13e.toml` as WM config)
- Flags off: smoke bit-identical to c4/policy14 (all losses, probes, rescored J; only wall time differs).
- Flags on (emergency + gs + learned yield): probe autograd -0.087606 vs FD -0.087604 / -0.087601; two updates finite (actor 5.29/6.18, critic 4.30/3.45).
- Smoke start states are benign, so smoke losses equal flags-off; the binding scene is the probe.
- Tests: see commit message.

Run: `--wm-ckpt <13E wm_s.pt> --wm-config configs/c4_wm_v10round13e.toml` (checkpoint dir needs its sibling physics.json).
