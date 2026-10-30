# S-CAPREL — owner-approved CPU build, 2026-10-30 14:00
Shared-function SHA: `37bc2a57176ccb391c4c7755a284e32e6d834632` (S-DISTILL2 import ready).
Base: `9613f956`; branch `c4/caprel`; SUMO parity re-check pending registration.
Final code/test SHA: to be bound after the verified implementation commit.
Final documentation/delivery SHA: `git log -1 --format=%H -- docs/paper_results/CAPREL.md`.

## Law and API
`f=-v/dt; CTRL0=max(f,min(b,c)); L'=min(lo,CTRL0); U'=max(min(hi,1.2),CTRL0)`.
ON: `executed=max(f,min(clip(CTRL0+r,L',U'),c))`; OFF: original base-relative ops.
Negative r brakes immediately when above the lower/floor bounds; interior slope=1.
Zero is bitwise CTRL0; positive r above c is inert; emergency braking remains feasible.
API: `wms.control.residual.executed_from_residual(state,r,law)`; scalar or tensor.
State: WM frame or `{v_mps,base_mps2,cap_mps2}`; law: `"base"`, `"ctrl0"`, or
mapping with residual_reference/lo/hi/dt/accel_max. No new driver settings or thresholds.
Config: `configs/c4_policy14_candA_caprel.toml` extends `c4_policy14_candA_longgrad_skip.toml`;
only `[policy] residual_reference="ctrl0"` changes; requires execution mode 2.
An action-law change requires explicit `--new-phase`; checkpoint metadata retains the law.

## Files and deployment
Hash-bound changed source/config/test files: `tools/execparity/caprel_files.json` (Git blob SHA-1).
Control: residual/sumo_idm/alwayscmd/execution; policy/nativeinit/bridge/dreamer;
policy14_cli/train_cli/entrant/confident_gate; SUMO env/ownership; candA authority;
execparity adapter/replay/guard/worker; preset and `tests/unit/test_execparity_caprel.py`.
Live b: legacy `sumo_native_accel` -> getFollowSpeed(getLeader gap + minGap).
Live c: getFollowSpeed(getLeader gap), no-leader 1e6, getStopSpeed at ramp dead-end,
getAccel/getAllowedSpeed ceiling, nonnegative-speed floor; recomputed each physics tick.
ON calls the shared algebra; zero/inert actions retain the original CTRL0 write exactly.
Only setSpeed(-1) then setAcceleration writes. Ledgers retain actual b/CTRL0/c provenance.
Existing connector/leader/foe observability errors remain parity measurements, not law changes.

## CPU verification
239 tests PASS (66 new caprel cases); nanfix, keyerr/entrants, execparity and candA included.
Checks: both-side bitwise zero, monotone braking, autograd/FD unit slope, cap/floor saturation,
NaN/inf rejection and finite tiny/infinite-gap gradients, OFF forward/backward bitwise,
full imagine+replay empty-CAV/entrant ticks, typed ownership path, config and checkpoint roundtrip.
New files full Ruff PASS; touched files PASS ignoring pre-existing E402/E702/E731 styles.
CPU repro (bash, CPU torch):
```sh
PYTHONPATH=src:. python -m pytest tests/unit/test_execparity{,2,3,_alwayscmd,_alwayscmd_replay,_continuous,_caprel}.py tests/unit/test_keyerr.py tests/unit/test_nanfix.py tests/unit/test_canda_integration.py tests/unit/test_entrant.py -q -p no:cacheprovider
```

## Exact SUMO re-check — NOT RUN; registration/ACK required before each job
Stage this revision's adapter/worker and src/wms/control/{residual,alwayscmd,sumo_idm}.py
under the fresh root below on the registered CPU hosts; retain the frozen E3 runtime/inputs.
The worker requires interactive registration JSON ACKs; never pipe automatic authorizations.
```sh
export C4_CAPREL_ROOT=/home/ubuntu/execparity/caprel_20261030 C4_RESIDUAL_REFERENCE=ctrl0
export C4_POLICY_STOCHASTIC=0 C4_POLICY_ACCEL_NOISE=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nice -n 19 python "$C4_CAPREL_ROOT/tools/execparity/sumo_worker.py" canary cpuP
# Only after the fresh canary passes; register the fixed -0.01 m/s² engineering probe:
C4_CAPREL_PROBE_RESIDUAL=-0.01 nice -n 19 python "$C4_CAPREL_ROOT/tools/execparity/sumo_worker.py" production cpuP
C4_CAPREL_PROBE_RESIDUAL=-0.01 nice -n 19 python "$C4_CAPREL_ROOT/tools/execparity/sumo_worker.py" production cpuQ
# After owner-provided observable jobs from both hosts are staged locally at _caprel_jobs:
C4_SUMO_LEADER_RULE=1 PYTHONPATH=src:. python tools/execparity/alwayscmd_replay.py --residual-reference ctrl0 --jobs _caprel_jobs --driver configs/execparity_cav_idm_v1.json --identity _caprel_jobs/s977000_always0/CTRL0_IDENTITY.json --out tools/execparity/caprel_replay_results.json --receipt tools/execparity/caprel_execution_parity.json
```
Canary 977000 + probes 977001/2/3; 4×900 sim-s require NEW registration. No SUMO run here.
Criterion unchanged: bitwise zero identity AND pooled executed RMSE <= native-head RMSE.
Old receipts/logs cannot qualify ON; fresh source hashes must match deployment and replay.
