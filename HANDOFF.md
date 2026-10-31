# S-RB-EVAL handoff

## Exact candidate CLI (RUNA/RUNB/INT)

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 nice -n 19 /root/miniconda3/bin/python -B -u /root/rbeval20261031/phys_code_b0f66943ec30/tools/rbint/evaluate_A.py --out /root/rbeval20261031/NEW_phys_A_shard0 --runtime-root /root/rbeval20261031/phys_code_b0f66943ec30 --wm /root/autodl-tmp/c4/wmcombined_wm2/wm_s.pt --wm-config /root/rbeval20261031/phys_code_b0f66943ec30/configs/wm_phys.toml --model-name wm_phys_A --shard 0 --shards 3 --memory-gib 5
```

**Future-builder API now available in EVAL source**, matching INT's staged API:
`run_wm.main(model_loader=load_candidate, future_builder=candidate_future,
identity_sources=[__file__, rebuild_helper, recipe])`; callback returns
`([frozen_candidate], construction_receipt)`. Receipt must be deterministic across
shards and name selected variant/settings; scorer rejects differing receipts.
Stage current runner/core/references/
real_observables/PREPARED in a NEW candidate code directory; README documents
the callback. For a normal loadable config invoke `tools/rbeval/run_wm.py` directly
 with the same arguments. Use shards 0–2 with distinct NEW output directories;
check resources first and preserve existing candidate dispatches. K8/five arms/
t0=300/H600 are fixed. Score all four shard directories together with
`python -B -m tools.rbeval.score --inputs SHARD0 SHARD1 SHARD2 --out NEW_REPORT`.

The command above is the **existing frozen A recipe**, executable on gpuA.
RUNA owns A/C intake and RUNB already owns B; use their claim protocol, never
dispatch a second copy. Their immutable runtime already has the compatible hook;
do not replace any staged source. For a NEW staged candidate, use this EVAL runner:
builder receives `(model, batch, initial, future, seed, frozen)` as keyword args,
returns `(kwargs, instrumented_entries, JSON_receipt)`. It owns hidden-settings,
arrival and entrant draws; the runner bypasses WM2 futures and attaches the
entry observer's population audit. Builder source is automatically SHA-bound.
`future_sampling.population_audit` and the source ledger are retained per task;
the updated scorer independently rejects missing/false/unconserved audits.
Verification: `python -B -m unittest tools.rbeval.test_core tools.rbeval.test_score
tools.rbeval.test_finalize -v`: **14 PASS /14.24s**, including source-ledger
rejection, differing-builder pooling rejection and WM2 callback identity guards.

## Durable implementation and dispatch receipts

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
- Fresh-session resume: preserved uncommitted scorer/adapter/tests and ops fix.
  Full-panel dispatch ALREADY LIVE on gpuR: receipt
  /root/rbeval20261031/DISPATCH_1793460699582341507.json; PIDs706825–706828,
  shards0–3, each capped5GiB. Checked load6.53/224CPUs, GPU15029/24564MiB,
  utilization100%. No duplicate launch or process signalling. Fetched receipts
  into ignored tools/rbeval/evidence/. Initial canary remains reused.
- rb-shared/targets/DONE.md now available: development-only speed-bin and BIC
  targets independently match the existing receipt. No helper extraction repeated.
- Scorer and S-SUMOCORE observable adapter finished: tools/rbeval/README.md has
  exact invocation/schema and WM-PHYS runtime/config/switch interface. Scoring
  rejects mixed identities/duplicate copies, incomplete K8 ensembles, changed
  starts/prehistory/BIC/support/ledger. Real baselines exclude entire target world
  and reserved labels. All11 CPU tests PASS, including end-to-end synthetic
  import→gate, independent CRPS/detector and joint chunked max-stat equivalence.
  B10000 scoring is memory-bounded; sparse cells explicitly NOT_QUALIFIED.
- gpuA checked separately: load7.29, GPU1/49140MiB, utilization0%. Existing WM2
  panel remains on gpuR to preserve live tasks; gpuA is available for S-RB-INT.
- CPU-only auto-finalizer deployed separately under
  /root/rbeval20261031/analysis_scoring; PID708801, own boot/start-tick receipt
  FINALIZER_708801.json and ANALYSIS_DISPATCH_1793461451536912414.json. Poll120s,
  deadline~07:44 host time; hashes all160 completed tasks then scores once when
  CPUload<24. FINALIZED.json binds report hashes; stopped workers yield BLOCKED,
  never replacement inference. `ops fetch-reports` retrieves it. Completion safety
  test PASS (all160 hashes verified; changed task rejected; dead worker blocks).
  Latest check:6/160 tasks, all4 workers live, GPU15241MiB; no complete K8 world
  yet, so no factual distribution score claimed. The inference runtime is intact.
- INT recipe compatibility: `run_wm.main(model_loader=..., identity_sources=...)`
  accepts INT's strict-load→encoder-preserving rebuild without checkpoint edits;
  callback API matches load_members and must return ([frozen_model], receipt).
  Loader source is auto-hashed; add recipe/rebuild/entrypoint files explicitly.
  See README. Candidate --model-name must differ from wm2. Stage UPDATED runner
   in a NEW candidate code directory; never replace live WM2 `code/run_wm.py`.
- Fresh-session audit: all four original WM2 workers and CPU finalizer remain
  live, 10/160 tasks complete (4/2/2/2); GPU15323/24564MiB at100%, load36.47/224CPUs.
  Current task times267–461s imply roughly4–5h total at this throughput. No new
  inference dispatched; exact candidate CLI is now directly at top of this file.
- Preserved INT-constructor/module-import/scorer-identity updates verified in this
  resume: all13 CPU tests PASS (13.8s), including candidate identity guard and
  external import→gate. Existing auto-finalizer retains its original scorer;
  added constructor identity applies to new candidate scoring only, not WM2.
- `rb-shared/obsstate/DONE.md` is now available (helper commit541f109f,30 CPU
  tests): observable-state/route export is for S-SUMOCORE initialization. RBEVAL
  continues scoring its fixed observable contract through `import_external`;
  no duplicate exporter or source-state conversion was started.
- Fetched11 real inference tasks (55 arm futures) and finalizer receipts; the
  scorer independently validated every start hash, support/ledger and BIC event.
  Progress audit: `tools/rbeval/evidence/progress_resume_ff8de578/audit.json`.
  No complete K8 world yet: `gate.json` correctly reports incomplete evidence,
  not a measured efficacy failure. Remote finalizer remains responsible for the
  full160-task report; `python -B -m tools.rbeval.ops fetch-reports` retrieves it.
- Constructor receipts are now also compared across candidate shards (not just
  code/recipe hashes); end-to-end test PASS rejects pooling encoder/population
  receipts even when model name and all source hashes agree. WM2 live sources
  and original finalizer remain unchanged.
- 2026-11-01 fresh-session receipt audit: clean HEAD707698f6 preserved; read
  owner program/registration5695, synthesis/reviews, WMBREAKDOWN gate and RUNA
  request. Existing gpuR WM2 PIDs706825–706828 and finalizer708801 live;
  53/160 tasks (18/13/11/11), load13.63/224CPUs, GPU15658/24564MiB at100%.
  RUNB supervisor728893 also exists; no job launched or signalled. INT already
  publishes immutable b0f66943 runtime with the future-builder hook and A/C intake
  links; reconcile EVAL's source API next without replacing either live runtime.
