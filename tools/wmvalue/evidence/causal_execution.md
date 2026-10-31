# Causal reconstruction receipt (resume of03dd03e0)

- 21 CPU tests PASS in18.654s; ruff PASS. Future suffix poison, physically
  truncated prefix, streamed-prefix equivalence, helper projection parity,
  unsensed kinematics/acceleration, source-feature rejection, entry deduplication,
  junction self counting, and birth/exit/post-step-source clock tests included.
- No hidden length/type/parameter arrays loaded: archive whitelist is explicit.
- The initial FIT exporter stopped on969400_pi's internal-junction lane=-1.
  Added explicit no-segment/known-self-only handling before fitting. No future
  lane backfill. Failed attempt created no arrays/result receipt.
- Read-only FIT exports v2 completed: cpuQ160 episodes/3360 states in29.19s;
  gpuR300 episodes/6300 states in79.06s. Both nice19, CUDA disabled, math threads1.
- All1380 objective clocks (120/300/600 across460 episodes) reconciled under
  preregistered .01veh-s float32 fleet tolerance and1e-6 source tolerance.
- Metadata demand/seed enters label-only request RNG, never observation features.
  No source backlog/request quantity appears among the111 learned inputs.
- Final exporter additionally filters command-log rows by contemporaneous known
  CAV identity and binds objective hashes. Re-export under a fresh v3 name before
  fitting; v2 packages/arrays/receipts remain preserved as audit evidence.
- Local ignored arrays _out/wmvalue_wmdata{,2}_causal_fit.npz are the v2 copies;
  committed JSON receipts wmdata_causal_fit.json/wmdata2_causal_fit.json bind them.
- Remote v2: /home/ubuntu/wmdata20261031/wmvalue_causal_fit03dd03e0_v2.{npz,json}
  and /root/wmdata2/wmvalue_causal_fit03dd03e0_v2.{npz,json}.
- Reserved969432..969439 real outcomes remain unopened. No fit, inference, SUMO,
  policy training, peer-process modification or push during this step.
