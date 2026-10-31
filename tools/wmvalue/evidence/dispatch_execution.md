# Actual inference dispatch (2026-10-31)

- Source226b0a8e committed before reserved export/dispatch; V/D/E SHAef3e7bce
  remains unchanged. Reserved200states/8worlds exported only after CPU gates;
  all120objective checks PASS, max6.265167e-6veh-s/source0. No fitting/selection.
- Private config inheritance independently checked against original WMCALIB
  canary identity: WM checkpoint/config, all resolved physics, courtesy switches,
  candidate and MAPPO export exactly equal. original_wm_binding.json PASS.
- Single durable local dispatcher PID28924, start2026-10-31 19:38:20, existing
  rl_driving Python. LAUNCH.json SHA8e3c9dccc1f728da09cb387ff6350b81a1a875a8db2ceedcbe4d7b93e24a7a12.
- First admitted jobs: gpuA CPU PIDs251940/251994/252054/252088, world969432,
  t300,futures0..3,fivearms each. At19:53 all alive/~100% of one CPU each,
 0completed/0errors; no peer GPU allocation. gpuR GPU100% with peer PIDs.
- Live logs/claims/results: _out/wmvalue_inference_v1. The dispatcher checks GPU
  compute ownership and CPU cgroup headroom before every new task. Exclusive
  task claims survive session loss; no claimed task is automatically redispatched.
- Complete320tasks trigger the unchanged frozen400pair evaluator, all10,000
  world-bootstrap draws, <=60line report, verdict registration and intended-file
  commit. Errors preserve outputs/claims and pause dispatch instead of repeating.
- Finalization-only refinement after dispatch: PASS plan uses60s only if B
  passes, otherwise90s if C passes. If inference was CPU-only, GPU-hour estimate
  cites the already audited WMCALIB GPU timing; no invented GPU benchmark.
  No evaluator, feature, value, runtime or threshold changed after reserved export.
- MAIN19:52 releases gpuR CPUs after SUMOCF STOP. Add a one-time disjoint
  seed969439/40task CPU partition, <=8single-thread nice19 lanes, initial admission
  requires10cores+40GiB free. Claims are reserved exclusively in the same root;
  original workers/dispatcher stay live and skip them.4new ownership/admission
  tests PASS (36total local tests); no repeated claims/SSH or peer GPU admission.
  If this partition finishes last, exclusive COMPLETE-file creation assigns the
  same finalizer exactly once. Frozen worker snapshot/source bindings unchanged.
- CPU assist admitted: local27688; eightgpuR workers588281/588282/588285/
  588311/588322/588341/588342/588354, all alive at20:07/~100% oneCPU each.
  44claims total (40reserved assist +4original),12active workers,0errors then.
  assist_dispatch.json binds actual launch/ACK; the original four are preserved.
- Additional actual frozen-WM CPU90s run on a small synthetic prefix PASS8.44s:
  180steps,all five arms,both60/90 terminal features,exact90endpoint/road+source
  cost composition. Runtime hash identical to the original CPU gate. No real
  outcome/seed loaded by that audit; cpu_long90.json links the full remote receipt.
- The efficacy test has NOT run. WMVALUE.md is an explicit running/NA receipt;
  no efficacy PASS/FAIL is asserted. No SUMO, policy training, refit, peer signals,
  prohibited paths or push. Monitor: tools.wmvalue.monitor (read-only).
