# Unstarted C2 handover and publication coordination

Registered prospectively in the active REGISTRATION, after06:27 resume.

- C2 has neither a shared claim nor a RUNA remote root at preflight. RUNA's
  single-active-variant queue otherwise waits for A2 completion before C2.
- INT claims C2 via exclusive-create ONLY after the original published source,
  checkpoint, config, physical parameter JSON and smoke hashes match on gpuR.
- Same published4015b641 source/recipe/20worlds/fivearms/K8/H600; only host and
  ownership change. New root `/root/rbeval20261031/int/WM-PHYS-C2`.
- One successful five-arm canary is reused once. Any scientific failure is
  terminal. The C2-specific CLI cannot claim A/B or arbitrary C namespaces.
- All inference is nice19,one numerical thread,5GiB allocator per worker,free
  CPU/GPU admission. Exact-incarnation monitor owns11:30 bound; no peer signals.
- Monitor skips the OLD source's scorer. Corrected local noon collection owns
  C2 scoring and source-bound uncertainty; incomplete K8 worlds never score.
- `panel_ops ... --variant C --host gpuR --name WM-PHYS-C2` selects the unchanged
  `evaluate_C.py`, retaining original population point and all corrections OFF.
- Existing collector publication transaction is being preserved during fresh
  owner edits. It will stop on its foreign-work/HEAD guard rather than stage
  mixed edits. Retain all completed score outputs and recover only publication.
- New cooperative PAUSE/PAUSED/RESUME receipts let the collector acknowledge
  owner edits BETWEEN transactions. Wait for PAUSED before any later edits;
  commit, then RESUME. This does not bypass branch/staged/source/deadline guards.
- Verification: panel monitor/orchestration,exclusive handover,tamper detection,
  pause/deadline,and noon report suites **36 CPU PASS/2.46s**. No model change.
