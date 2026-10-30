# S-WMFAST-B — owner 2026-10-31; MAIN-only clean-boundary handoff
Final tested code/evidence SHA: `9d7e7d8ea152855684c53045306e0a0497e0abcc`.
Delivery SHA: `git log -1 --format=%H -- docs/paper_results/WMFAST_B.md` (no self-hash).
Branch `c4/wmfastB`, base `43942f75`; each implementation/evidence step committed; no push.
Evidence: `tools/candB/evidence/WMFAST_B_{CPU,GPU,GPU_first,tests}.json`.
## Design / exactness boundary
- Default OFF: `WMS_SPEED_ES_STACK_SIDES=0`; explicit 2 or 4 stacks S perturbation sides.
- Repeat variant rows, NEVER starts: original K8 future/driver/lane tapes, entrant seeds,
  identity-keyed learned mix, merge keys and correlated/additive exploration stay paired.
- Functional per-side residual weight+bias; actor/group/native/safety/reward calls retain
  original side geometry (enlarged CPU vector kernels otherwise changed command bits).
- One stacked no-grad WM rollout/future; split ALL outputs BEFORE original gated300 sums,
  native/CTRL0 references, gain gate, symmetric pair gate, signed joint-q10 and ES estimate.
- Share actor-free native/CTRL0 physical rows within a stack; restore per-side unused
  actor/anchor/audit predictions too. R7/B2 references give S2=12, S4=22, not 14/28 rows.
  Learned unperturbed all-CAV rows are not identical to perturbed treatments and cannot alias.
- Original unperturbed forward still supplies calibration/tape/dense replay; no long backward,
  horizon, sigma, M/K, gate thresholds, dense objective, clipping or optimizer-law change.
## Parity evidence
- 12 new CPU cases PASS: S2/S4 × one/two starts, all K8/M2 full pre-prefix tensors bitwise;
  nonzero g_ES/pair receipts/RNG/parameters bitwise; real two-worker transport; exceptions.
- Additional actual 300-control/600-tick toy-WM fixture: all four sides' complete tensors
  bitwise for selected future7 (unperturbed capture still K8); no shortened production horizon.
- Real tiny hybrid WM, synthetic histories, actual births/learned mix, sampled drivers/lanes,
  exploration: 10 physical seconds, K8/M2; 4,896 full-output tensors compared per stack size.
- CPU: 39/41 nonbitwise tensors for S2/S4, max abs 1.91e-6; credit/actions/gate receipts bitwise.
- GPU: 138/174 nonbitwise tensors, max abs 1.53e-5; S4 one credit tensor differs by 1.53e-5.
  WM batch-kernel float reassociation is admitted at atol=2e-5, rtol=2e-6; discrete tensors exact.
  This is algebra-exact batching, NOT a universal bitwise or full real-WM300 qualification.
- Natural short-probe gates all OFF; additional synthetic gate stress (samples×10000+1000000)
  accepts10 pairs, nonzero ||g_ES||≈7.26e6; sequential vs S2/S4 g_ES/masks bitwise on CPU/GPU.
- 65 ESLONG/gated300/LONGGRAD regressions PASS; two unstaged config-preset cases deselected;
  Ruff and diff checks PASS. Probe receipts bind SHA256 of the three changed runtime modules.
## ES-component timing and memory (fixtures, NOT production measurements)
- CPU one thread, three samples: serial32.289s, S2 19.295s=1.67x, S4 12.359s=2.61x.
- GPU first/last one-sample ES: serial53.17/62.68s, S2 35.16/36.34s, S4 21.25/19.76s;
  ratios1.51–1.72x / 2.50–3.17x. Changing pilot load makes GPU ratios nonstationary.
- gpuR peak allocated/reserved MB: serial14.72/29.36, S2 15.25/29.36, S4 16.40/31.46.
  Own process reported488MiB; allocator capped.04 (<1GiB), admission free16.08GiB on final probe.
  Both cards checked; no gpuA probe; own CUDA probes exited; peers continued untouched.
- Production live row memory scales roughly [S*(R-B)+B]/R, plus fixed/model/coordinator costs.
  R7/B2 example ×1.71/3.14: historical10.1GiB allocation suggests ≈17.3/31.7GiB, UNMEASURED.
  Full300 shared-card fit is unqualified: S4 is especially large; even S2 can exceed24GB beside
  a pilot/coordinator. Retain worker allocator/budget guards; tiny probes do not justify increases.
## Projected seconds/update / exact adoption diff / resume
- Using SPEEDPROF F=861–974s, D=580–866s and CPU speedups v: ordinary N1 ≈2F+4F/v+D
  gives S2≈4360–5145s, S4≈3620–4305s, serial≈5746–6710s, plus unmeasured pool/IO overhead.
  Native panel16F/10 adds≈1378–1558s amortized; phase10 adds another16F. Startup excluded.
  These extrapolate tiny fixtures; neither full-update speed nor full-scale GPU fit is measured.
```sh
export WMS_SPEED_ES_STACK_SIDES=2  # optional4 only when the measured worker budget fits
# unset, or =0, restores sequential ES; no TOML/config diff
```
- MAIN stages immutable LF source and regenerates authority/source manifests; changes only this
  execution env at a clean boundary, sizing S against existing workers on gpuA47GB/gpuR24GB.
- No new checkpoint fields or --resume-allow keys: populated Adam, update count, qualification,
  actor bytes and next g_ES survive OFF→S4 restore bitwise in CPU fixtures. Preserve RNG/sampler,
  original phase/R4 baseline/clock; current fresh-only pilot_cli still needs a same-phase adapter.
