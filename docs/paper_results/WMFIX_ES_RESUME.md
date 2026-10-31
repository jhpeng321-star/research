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
- Registry now covers 52 worlds / 172 contrasts; fitness averages arms within
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
