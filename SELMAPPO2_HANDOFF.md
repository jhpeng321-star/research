# S-SELMAPPO2 implementation/CPU handoff

Per-CAV implementation is `E:/agent-work/c4-selmappo2/tools/selmappo2/mixed.py`;
shared env/observations/learning/runtime are S-SELMAPPO's exact staged snapshot.
Design SHA09499f2af60311b3690a940dca8d94605d710465afb9c9eb05d5b0a79b23900c,
registered cap7012800; local and cpuQ56/56CPU PASS, real frozen runtime gate PASS.
Stage `/home/ubuntu/selmappo2_20261101/build_b42c257b09813c175fcb7261e341c7599d0bb5c4c971529e2856890d2fc8038c`.
No final checkpoint yet. Training dispatch is the next step; never evaluate an
intermediate checkpoint. Final169 plus FROZEN_FINAL.json carry source/binding SHA.

We leave the local14400evaluation reservation UNSTARTED: your task owns all
panel969136..969151 evaluation and J900 tail. Please reuse the exact mixed adapter
and shared snapshot in your PerCavBackend. Our checkpoint format is already
compatible with your selectors.PerCav record checks (variant/optimization_seed/
sources/bindings). Final source package is stage/pytools/tools/selmappo; mixed
package is stage/pytools/tools/selmappo2. Both require hash-bound source specs.

Entrant/re-entry fairness: previous-controller features must reflect the live
backend's current_names at each decision, including retirements/re-entries during
the20s window. Your PerCav selector currently carries previous action dict only;
please synchronize it from backend contexts before select, and retire missing IDs,
so re-entrants do not inherit old controller features. No truth features.

WALL adapter uses public self/RSU receipts. Standard receipts lack junction flag;
it uses carried lane unless an observable junction flag is supplied. Outside-
coverage traffic is absent from rear surveillance; no hidden telemetry fill.
