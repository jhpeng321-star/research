# S-RB-INT handoff

Updated 2026-10-31 23:22 +08:00; branch `c4/rb-int`, base `f0db3c82`.

## Status
- Read owner authorization at active C4 registration line 5682 and all four WHY reports.
- Integrator worktree initially clean. No physics branches merged yet.
- Initial poll: `c4-rb-{lane,merge,cf,eval}/HANDOFF.md` absent; SUMOCORE handoff absent.
- CPU: 22 logical processors, initial load 40%; local NVIDIA driver unavailable.
- No training, downloaded code, scientific rollout, or peer-process action performed.
- Resume preserved the previous uncommitted handoff and `tools/rbint/`.
- 23:20 poll: LANE and MERGE have source/interface handoffs, implementation pending;
  CF and EVAL handoffs absent. SUMOCORE says no dispatch yet. Shared README read.

## Integration contract for workers
- Please commit usable increments and state exact commit, switch names, test command, and known gaps here in your own handoff.
- Target A/B: SUMO IDM + LC2013 + SUMO merge/insertion; every learned motion/lane/merge correction OFF.
- A keeps the existing motion-history encoder settings; B uses population defaults, including new entrants.
- Loading the existing checkpoint must preserve encoder weights; switches must not silently substitute an untrained encoder.
- Audit findings: `rollout_mode=driver_params` alone is NOT sufficient: long-history residual lane corrections still run (`wms.py:1438`), learned merge/courtesy/yield/ramp terms run independently, and `accel_noise` can still use learned sigma. The rebuilt path must bypass these explicitly.
- Shared-file changes stay opt-in/default-OFF. Existing checkpoint behavior must remain unchanged without the switches.
- Evaluation must bind a fixed permitted FIT panel before scoring; never open prohibited paths or truth assignments.
- Need finite/conserved 60 s rollouts from two recorded FIT starts before the registered A/B distributional panel.

## Tests / results
- `tools/rbint/default_parity.py`: captured and independently compared two CPU
  synthetic legacy/default + WM2-architecture cases; weights, forward and rollout
  tensors bit-exact. Baseline: `docs/paper_results/WMPHYS_DEFAULT_OFF_BASE.json`.
- Command: `PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m tools.rbint.default_parity --compare docs/paper_results/WMPHYS_DEFAULT_OFF_BASE.json`
- Synthetic long-history fixture explicitly masks all history as missing; this is
  compatibility evidence only, NOT the recorded FIT smoke or an efficacy result.
- `tools/rbint/audit.py`: closed-cohort conservation/finite/zero-residual gate and
  temporary hooks rejecting execution of parameterized corrections (encoder allowed).
  `py -3.13 -m pytest -q tests/test_rbint_audit.py`: 9 passed (5.70 s).
- Existing local `py -3.13` has torch `2.15.0+cpu`; default Python 3.14 is not the selected runtime.
- Remote resource check: gpuR RTX4090 24,092 MiB free, 0% utilization; cpuQ load 0.94 / 32 logical CPUs, ~61 GB available. Recheck immediately before dispatch.
- Resume local load 68%; parity check used a single CPU thread. No running INT
  rollout/dispatch receipt exists in this worktree; no scientific job launched.

## Known gaps
- Worker APIs and registered evaluation panel are not yet available at first poll.
- No measured WM-PHYS efficacy; historical WM2 statistics are references, not matched-panel results.
- EVAL freeze now available: `c4-rb-eval/docs/paper_results/RBEVAL_DESIGN.md`,
  seeds 969420–969439, native/ctrl0/mappo/pi/circles, K8, t0=300, H=600.
- Worker request: expose checkpoint-preserving construction/runtime switches; CF
  must bypass parameterized residual/temporal/gate/lane branches, not merely zero
  final acceleration. INT will run `forbid_learned_corrections` on recorded smoke.
- LANE/MERGE ownership needs explicit reconciliation: LANE handoff proposes full
  LC2013 lane model; MERGE owns ramp 2->1. Mainline proposals must survive merge
  resolution and carry LC2013 persistent state across rollout chunks.
