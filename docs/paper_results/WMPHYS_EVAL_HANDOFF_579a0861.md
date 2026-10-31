# Preserved S-RB-EVAL handoff at integration commit 579a0861

- Branch c4/rb-eval, base f0db3c82; fresh-session audit found clean tree and no
  previous RBEVAL implementation/dispatch receipts. Existing WMCALIB cost-only
  tasks lack trajectories and native predictions; they cannot fill this panel.
- Frozen design: docs/paper_results/RBEVAL_DESIGN.md. 20 worlds 969420–969439,
  all five arms, t0=300, K8, horizons 120/300/600. Small-cell precision prohibits
  qualification; retain separate diagnostic and gate verdicts.
- tools/rbeval/{core,run_wm,ops,prepare}.py implemented; shared model files untouched.
  AST snapshots reuse WMCALIB starts, SPEEDHEAD actions/real_J, registered BIC.
  Detector SHA 6de4df039e108508cf71f94abfe2e81136d5a9a502bb4803980411c0fa5b3f10.
  real_observables.json is the exact 200-episode S-WMBREAKDOWN receipt, SHA
  01d0d2b7027aad9910496c7bc5b9681c0d1ccf08f46b4e16a1113310a67362e9.
  No new raw-recording access for factual targets. Runtime-root/checkpoint/config
  CLI supports integration; --switches is an explicit model-attribute JSON file.
- CPU verification: python -B -m unittest tools.rbeval.test_core -v: 4 PASS;
  independently rechecks all 1,800 WMDATA BIC decisions, CRPS via CDF integral,
  births/exits/invalid-lane/endpoint binning and censoring/precision.
- gpuR first resource check: GPU 1 MiB/24564 MiB, utilization 0%; host load 18.84.
  No matching RBEVAL process seen. Canary dispatched once under
  /root/rbeval20261031/DISPATCH_1793460426125998926.json, PID706072 (now exited).
  Complete w969420_k0.json: 5 arms, 600s, 145.78s wall, peak4250MiB reserved.
  CANARY_VERIFIED.json binds saved output and historical four-arm sensitivity:
  source exact/road within0.00044veh-s at120s; five-vs-four-arm batching diverges
  up to686.924veh-s team delay at600s. This is a newly run baseline, not claimed
  byte/numeric parity with historical WMCALIB. BIC decisions rechecked, float
  coefficients permit1e-8 BLAS tolerance. No canary task will be rerun.
- Native uses the registered project_native/native_fn background law; all5 arms
  retain WM-native lanes and coupled entrants/driver/event streams.
- Reuse S-WMBREAKDOWN observable receipts and registered AST-extracted LRDIAG
  detector. Helpers rb-shared/targets and obsstate will be consumed if available.
