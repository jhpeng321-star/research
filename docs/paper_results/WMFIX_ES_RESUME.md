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
