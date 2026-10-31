# S-C1CLOSED implementation checks (2026-10-31)

Recovered four historical 2 s CPU smoke outputs, unchanged and retained remotely.
Fresh code adds the prospectively registered aggregate fields and is checked on
the actual permitted WMDATA world 969400, t300, for 10 s / 20 half-second steps.
All four frozen model hashes match. WM2/BB-traj/historical execute five arms;
BB-id retains only native/PI/CIRCLES under its legacy ID contract.

`evidence/CPU_VERIFIED*.json` binds write-once actual outputs and verifier/worker
hashes: finite forecasts, endpoint-ID shapes, initial source-queue equality,
four-component exact J additivity and 1 x 15 fixed-grid shapes PASS. Perturbed
predicted-state speeds change native, PI, CIRCLES and MAPPO callbacks; native
production lane delegation is checked explicitly. Analytic fixed-grid density
and merge-queue fixture PASS. The first verifier failed because the manually
constructed callback state omitted the execution-cap key normally supplied by
rollout; supplying the registered inferred cap fixes the fixture, not the model.

`CANARY_DONE.json` records successful 10 s GPU inference with peak sampled owned
memory 1160 MiB and exact owned PID/start ticks. The canary process has exited.
gpuR probe: only 698 MiB free, so no task was launched there. gpuA canary admission
had 11211 MiB free. Two GPU lanes (3 GiB allocator each) must have >=5120 MiB
free before starting, monitor total-owned <=12 decimal GB and >=1024 MiB reserve,
and only terminate their exact Popen child on a guard violation. GPU launch load
ceiling 32; two one-thread nice19 CPU lanes wait for load <18. No other PIDs
are signalled. This admission policy does not alter registered K8 or metrics.

`python -B -m ruff check --select E9,F tools/c1closed` PASS. Full-panel dispatch
is prospective at this step; CPU tests and the GPU canary are implementation
checks, not a scientific C1 verdict. No recorded future was opened by inference.
