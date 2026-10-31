# S-WMFIX-ES fresh-session audit

- Resumed from clean `89d7e29b`; prior commits/artifacts retained. No push.
- Remote `ps` audit: no `tools.wmfixmerge.es` workers on gpuA or cpuQ.
- gpuA `es_run_combined` is empty; cpuQ combined v1/v2 registries exist.
- cpuQ `es_smoke` failed in inferred_timeline: CTRL0 direct/mixed actions
  require an explicit feedback controller. No combined ES generation completed.
- Prior registry used only every fifth wmdata world and CTRL0; MAIN 13:38
  instead requires all 40 worlds/all arms. Correct before dispatch.
- Prior CPU probe is single-future: Runtime uses FuturesConfig(n_samples=1),
  not K8. The measured 1766.6/1920.0 s and 1372/1775 MB remain valid only
  for that single-future two-arm 120+300 s workload. K8 claim withdrawn.
- Runtime source guard remains active. No SUMO, GPU launch or foreign PID
  signalling. cpuQ cap now 30 CPU workers; gpuA cap 20 CPU threads.
- Next: explicit frozen wmdata feedback adapters; all-world/all-arm registry;
  genuine K8 probe; budget the full-corpus calibration honestly.

## Corrected tooling / live probes
- Explicit CTRL0 uses the original 64f70af4 policy with its residual head zeroed;
  MAPPO binds d9042cc5. PI/CIRCLES use published TimelineControllers, including
  explicit algorithm continuity for entrants. Opt-in designed-arm loading is
  restricted in Runtime to verified wmdata_fit episodes; default loader unchanged.
- Registry now covers 52 worlds / 184 contrasts (both FIT arms too); fitness averages arms within
  world before averaging worlds. Original FIT objective scales remain fixed.
- cpuQ tests: 18 passed (ES + wmdata role); Windows has no torch (17 passed,
  the live-dataclass test requires torch; no Windows environment installed).
- K8 probe launched as eight sticky-future single-thread processes per host:
  cpuQ PIDs 3030648-3030655; gpuA PIDs 193235-193242. Each measures both arms
  at 120/300 s, entry path/mix ON. Dedicated `probe_k8/DISPATCH.json` + PID
  receipts prevent relaunch; gpuA code isolated under `es_cpu_v3`.
- CTRL0/MAPPO 60 s smoke complete, 31.2/31.3 s; frozen tensor checks pass.
- PI/CIRCLES first attempt lacked frozen controller sources; retained failures.
  Sources staged byte-for-byte from WMDATA frozen e3final runtime; new owned
  PIDs 3035044/3035045, `es_canary_deps_v3`; no other processes signalled.

## Full-corpus calibration dispatch step
- Missing adapter/model assets caused the second PI/CIRCLES attempt to fail;
  only adapter.json and model.onnx were copied from frozen e3final. Third
  attempt `es_canary_assets_v3` PASS: CIRCLES 31.9 s, PI 31.2 s, frozen checks.
- Both original FIT arms are bound, not just arm 0: 24 + 160 = 184 contrasts.
  Verified registry `TRAIN_PREPARED_COMBINED_FULL.json` copied cpuQ -> gpuA.
- Canonical physical export/reload smoke PASS on gpuA, under private code tree.
- Expanded ES/objective/role tests: 20 passed on cpuQ, including exact parity
  against the registered torch objective and equal-world aggregation.
- Protocol amendment before any combined fitness: full 52-world/184-contrast
  corpus at existing 120 s training horizon, population 2, sigma .10, lr .01,
  seed 140014, <=8 generations / 12600 s, stale patience 3. Six physics scalars;
  entry scalar frozen because canonical verdict entry config cannot carry it.
- Immutable generation exports bind physical config; all old WM2 tensors are
  bitwise checked on export and canonical reload. Failure receipts stop driver.
- Driver reserves live K8 probe slots before each generation (cpuQ total <=30,
  gpuA <=18 CPU workers), low-priority sweeps; no GPU code path.
- Full driver launched gpuA PID 195604, `es_cpu_v3/es_run_full`; cpuQ workers
  `es_run_full`. Read RUN/ENV/CELL/GEN/ES_FINAL/ES_ABORT before any resume.
- Unchanged-plan preflight on canonical smoke export found missing accept panel,
  C1 cohort/runner and candidate quartet completion. Required input/GPU-slot
  requests are recorded in WMFIX_ES.md; no evaluation result is claimed.
