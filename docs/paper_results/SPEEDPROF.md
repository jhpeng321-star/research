# S-SPEEDPROF — exploratory B / B-L, owner 2026-10-30 22:35
Implementation `c10f73f7`; measurements/tooling `a61869b8`; branch `c4/speedprof`.
Evidence: `tools/candB/evidence/SPEEDPROF_{pilot_latest,GPU,kernel_probe,tests,regressions_retry,attempts}.json`.
## Attribution and evidence limits
- B gpuA chain `/root/autodl-tmp/c4/candB_pilot_final_run_EXPLORATORY_20261030`:
  23:18 +08 receipt still initializing: no update/rescore/memory trace. B-L waits for B's baseline.
  Thus no measured B/B-L s/update, checkpoint/IO split, or next-pilot timing exists in these receipts.
- Dry985766/start874675005: 7329 CPU-s/6967 wall-s = 1.05 cores despite OMP/MKL=2.
  Tiny GPU score: 430,729 Python calls/2.71s, 80 physics ticks; launch/Python-bound evidence.
- Define F=861–974s/K8 300s set: historical `ESLONG.md`861; same-WM/mix/K8 actual
  `rankcheck_saved_k8/CANDB_RANKCHECK.json` native970.39/CTRL0973.88s (one TRAIN start).
- One serial update: unperturbed K8=F; ES M2×2 signs×K8=4F=3444–3896s;
  dense60/90 replay+VJP D≈580/866s (ESLONG projection, includes next-input boundary).
- Native/CTRL0 are rows INSIDE those batches, not extra full rollouts. Dense zero-VJP skips stay ON.
- Independent rescore is ONE frozen start, dense-prefix+300s from one K8 rollout ≈F;
  native panel costs16F every10 absolute updates; phase10 separately costs another16F.
- Startup currently duplicates the 16-start native/gate panel (33F including the single start).
  IO/optimizer/transport residual is unmeasured; PROFILE adds wall+CPU phases without device sync.
## Evaluated speedups (gain estimates are not production benchmarks)
- (a) `RESCORE_EVERY=N`: default1; baseline, phase10/absolute signal gate and final score retained.
  OPTION N=5 saves≈0.8F=689–779s/ordinary update; policy/grad/Adam/RNG and scored gate values
  are bitwise in CPU tests. **Bookkeeping-inexact:** unused native entropy_weight may differ;
  intermediate diagnostics disappear. Stochastic actors with active entropy annealing refuse.
- Extra exact `REUSE_PANEL=1`: consume identical matched native scores for the same gate panel;
  binds update, parameter/buffer versions, WM/config/phase/validation identity; relabelling refuses.
  GPU16-start probe79.389→39.584s (2.01x), bitwise complete score receipts/gates/RNG;
  saves16F at startup (33F→17F) and phase10; native checkpoint selection cadence is unchanged.
- (b) CTRL0 row-trajectory cache: not admitted. Removing its row changes GEMM/tape geometry;
  theoretical work saving≈1/row_count, likely little launch-bound gain. Training vs rescore has
  different starts/CRNs; cross-use would be incorrect. Whole-panel reuse above is the exact alternative.
- (c) Stack4 perturbations: CPU physics microprobe0.2031→0.06237s (3.26x, noisy), all5 tested
  outputs bitwise. Full ES needs per-side heads, entrant/CRN tape mapping and separate joint-q10;
  NOT qualified/implemented. Potential ≤4x ES-forward component; no full-estimator gain claimed.
- (d) Pilot now honours opt-in real forward pool; default still serial. Ordered 3/4-worker CPU
  samples bitwise; GPU throughput/memory unqualified. Two-worker ideal: 5F→2.5F (not 2x update).
  Historical300s worker11.7GiB reserved; proposed two-worker .28 allows13.4GiB each, not a B fit proof.
  Three×.26/cap.90 need coordinator≤~2.75GiB; four×.26 exceed budget. Do not blindly use3–4.
- (e) WM-step fullgraph screen FAILS on Tensor→bool in gate.monotone_gate. No exact physics
  compile/graph flag admitted; fusion may reassociate floats and change long-horizon gates.
  Existing exact reward `WMS_SPEED_GRAPH_CVAR=1` is already ON. Further gain is unmeasured.
- (f) Both launchers already set WMS_WMSPEED_DEDUPE_MOBIL/MOBIL_LAZY_SAFE=1. Remaining
  master skips have no policy call sites: measured GPU score2.46085→2.45990s (noise; zero gain).
- (g) `SKIP_GRAD_DIAGNOSTICS=1` removes only actor/component and CF-term extra VJPs;
  CPU short M2/K8 fixture4.729→1.578s (3.00x); weights/grads/Adam/ES gates bitwise.
  Production gain depends on audits (currently every25 global updates); core module norms stay.
## MAIN-only clean-boundary diff (all library flags default OFF)
Stage committed LF source and regenerate source/config authority; never edit a live runtime.
```sh
export WMS_SPEED_REUSE_PANEL=1 WMS_SPEED_SKIP_GRAD_DIAGNOSTICS=1 WMS_SPEED_PROFILE=1
export WMS_SPEED_RESCORE_EVERY=1  # optional5 only with the bookkeeping difference above
# gpuA ALONE only; gpuR retains the serial pool defaults:
export WMS_SPEED_PILOT_FORWARD_POOL=1 WMS_SPEED_PARALLEL_FORWARD=1
export WMS_SPEED_FORWARD_WORKERS=2 WMS_SPEED_FORWARD_MEMORY_FRACTION=.28
export WMS_SPEED_FUTURE_BUDGET_CAP=.90
```
On gpuR ALONE, memory-only TOML diff `[run] gpu_memory_fraction=.90` avoids the inherited
.40≈9.6GiB cap vs historical10.1GiB forward allocation; bind a new authority. No trainer/reward diff,
so **no --resume-allow keys** for these changes. Retain segment2, K8, M2 and both LM300 flags.
Env is resume-safe; preserve Adam/RNG/sampler/update and original phase-zero/R4 baseline/clock.
Current pilot_cli is fresh-only: apply directly to the next fresh pilot; MAIN needs a same-phase
continuation adapter before restarting a trained pilot (disable zeroing; never fake a new phase).
## Expected seconds/update — ideal overlap projections, plus unmeasured pool/IO overhead
gpuA alone, two forward workers: exact N1 ordinary≈3600–4300; amortized native panels≈5000–5800.
Optional N5: ordinary≈2900–3500; amortized≈4300–5100. Phase10 still adds16F≈3.8–4.3h.
gpuR serial B-L: D ranges0 (guard all OFF) to866s; exact N1 ordinary≈5200–6700,
amortized≈6500–8300; optional N5 ordinary≈4500–5900, amortized≈5900–7500. Startup excluded.
Verification:21 new/pilot +83 regression PASS;5 heavyweight cases deselected. Earlier broad SSH
timeouts/import-mode failure retained, not counted. GPU probe idle-only, .06 allocator cap,
26MiB peak reserved; own probes exited. No pilot signal/relaunch, SUMO, push or tracked-file deletion.
