# S-RB2-BREAK checkpoint — 2026-11-01 17:41 +08

MAIN's 17:35 GPU priority addendum is fulfilled. No new GPU job may start until
MAIN appends `S-RB2-BREAK RESUME`.

## Resource release and retained work

- Exact owned workers **501141 / 500921 exited** after their existing cooperative
  STOP check; each finished its in-flight H600/all-five-arm task first.
- All nine recorded BREAK queue processes also exited. No peer process action,
  signals, deletions, new dispatches, fitting or outcome scoring during recovery.
- Frozen source **5947c801**, same remote immutable runtime and output roots.
  Shard0: world969420 futures0–3. Shard1: world969421 futures0–2.
  **7/96 tasks, zero complete K8 worlds. No distributional qualification.**
- Remote output roots remain
  `/root/rbeval20261031/rb2break/devk8_native2_s0` and `devk8_native2_s1`.
  Local snapshots `docs/rb2break/devk8_pause_s{0,1}` include all seven raw tasks
  (gitignored as previously registered). Committed snapshots bind every raw task
  hash, identity, dispatch, pause request, STOP, worker log and EXIT receipt.
- Audit: `docs/rb2break/GPU_CHECKPOINT_1735.json`; nvidia-smi confirms both owned
  workers absent at17:41:23. Recovered work commit **8c0445bc**; release receipts
  **7523bbcf**. Source/data hashes were checked without rerunning completed work.

## Findings available to INT / LANE

The major defect remains unresolved. Complete dev12/K1 matched-future0 evidence:
600s delay bias14,979.67 baseline →14,802.50 corrected actual-mode2 CAV settings
→14,666.50 with native-uncommanded CF ownership. Upstream600s breakdown remains
100% versus recorded90%; source-wait bias is still+12,206.98veh-s. All four
commanded arms are exactly unchanged in the native-ownership contrast.

Frozen dev K8 baseline attributes11,331/13,485veh-s bias (84%) to source waiting;
mainline admission deficit50.65vehicles/600s accompanies only0.713 excess mean
requests. This is delay accounting, not an additive causal decomposition.
One-factor LC-execution suppression and merge-advice removal worsen delay;
they do not establish the causal effect of all lane-change or merge mechanics.
`departSpeed=last` improves total bias by cancelling lower source wait against
higher road delay and worsens CRPS; rejected as a repair. Driver shrinkage,
tau/v0 homogenization and fixed conditional rate do not fix the major bias.

Startup-only10s birth-disjoint motion validation passed: acceleration CRPS
0.206464→0.188588m/s²,10/12dev worlds improved. Entrant-only artifact
`artifacts/rb2break/startup_v3/entrant_population.json`, SHA256
`fdd816e24c85f6f4b65b67b5f992b2879c404c4c3fe75cc8cb07a5bce0397e39`.
Its traffic/probe parity is **untested**; it is not part of the frozen K8 run
or an adoption recommendation. Earlier failed fits/support checks remain retained.

Only development969420–969431 used in this workstream's new fits/selection;
inherited motion priors retain FIT32 provenance. No new reserved or WMDATA2 input.
Learned corrections remain OFF. Final replicated panel/gate/CIs belong to INT.

## Verification and continuation

Existing Miniconda:25targeted BREAK CPU tests PASS. Default Python's missing torch
caused an initial collection failure; no environment was created or changed.
The first checkpoint fetch attempt failed before transfer on an embedded-script
indentation error; corrected fetches and all snapshot/hash/process checks passed.

After explicit MAIN authorization, inspect this receipt and HANDOFF before any
continuation. Preserve all existing STOP files and completed tasks; do not blindly
redispatch or pool these partial ensembles into K8 scores. No continuation process
is active for BREAK.
