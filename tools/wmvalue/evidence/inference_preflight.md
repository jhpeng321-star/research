# Frozen inference / evaluation execution preflight

-32localCPUtests PASS21.456s; ruff PASS; no fitting APIs in the new evaluator.
-gpuA CPU audit PASS30.87s, exactly matching gpuR FIT-canary cost/birth output.
  Worker wrapper canaries on FIT969400@360 also PASS:16.20s gpuA,20.17s gpuR.
  cross_host_cpu.json binds identical134runtime/controller/model-source files.
-gpuA needs its EXISTING speedhead/pythondeps on sys.path for onnxruntime;
  no package installation or new environment. Original failed import preserved.
-Resource query19:16:gpuR recovery WMCALIB PID553396; gpuA GPU100% peers.
  No GPU inference admitted. gpuA CPU quota20/load12.6,~26GiB spare at probe.
  Dispatcher caps4nice19/single-thread CPU lanes and checks live cgroup headroom;
  gpuR GPU lanes<=3 only when no peer compute PID,5GiB allocator/worker.
-320world/time/future tasks, each five arms, closed-loop to900 with60/90 terminal
  checkpoints. Exclusive durable local claims, exact remote PID receipts, immutable
  outputs; restart only collects, never automatically repeats claims/failed tasks.
-A-E evaluator refuses incomplete200arm/320future coverage, wrong clocks,
  changed WM/source binding or changed frozen model SHAef3e7bce. Full test only,
 10000world bootstrap, original gate. Report<=60lines; automatic verdict receipt,
  shared registration append and intended-file commit after inference completion.
-At this source freeze, reserved outcomes remain unopened. Export checks all
  original120/300/600 real clocks before any test, then freezes the arrays. No
  outcome-driven choice, refit, SUMO, policy run, peer signal or push.
