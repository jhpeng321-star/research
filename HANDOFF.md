# S-SELMAPPO2 live continuation / resume

Branch `c4/selmappo2`; report `docs/paper_results/SELMAPPO2.md`.
Original advance-approved exact cap7,012,800; designSHA09499f2af60311b3690a940dca8d94605d710465afb9c9eb05d5b0a79b23900c.

## Current state

**MAIN01:48 continuation dispatched once** on cpuQ, own supervisor606968,
UID1000/startticks48374316. Current root is
`/home/ubuntu/selmappo2_20261101/build_c141d5715f26af4e9b50d0866cdf971ef92d55dd1223c4312c4f25bdec5a8b29`.
Parent `PER_CAV_CONTINUATION_ONCE.json` is now consumed; NEVER dispatch again.
Original lock/root/failed seed/900charge remain untouched. First current job is
ONLY second smoke18107999; production follows its success under spare-CPU admission.
At dispatch:32cores/load7.08,60.13GBavailable RAM,2.601GBfree disk. Disk is tight;
admission waits at2GiB without any deletion/signals/new reservations.
Read-only collection command (>=60s SSH intervals):
`python -B -m tools.selmappo2.report --action collect --build build_c141d5715f26af4e9b50d0866cdf971ef92d55dd1223c4312c4f25bdec5a8b29 --pid 606968`.
Use current reporter: it accepts ONLY the exact hash-bound historical failure.

Collection Unix1793469726 verifies second smoke SUCCESS900actual and20production
episodes complete;19,200actual/26,100reserved incl history,7openproduction,0updates.
Owned supervisor606968 remains live. Separate read-only five-minute watcher
`scripts/selmappo2_watch.py` is **running locally as ownPID12432**, launched once
at Unix1793469838 (4CPU tests/Ruff pass); it never dispatches simulation.
ScriptSHA2119509d4b46020bc5a6b67e5e34898c93db0405560bedd99c225984b711983f.
Its WATCH_ONCE lock and process/dispatch receipts live in evidence; do not relaunch.
Let it collect rather than duplicating SSH polls. It publishes the
paper's live tail plus immutable snapshots and only hands COMPLETE final169 to
the peer. Peer evaluation manifest path/schema are documented in the report.
Read-only resource receipt Unix1793469890:32cores/load17.11,55.98GBavailable RAM,
2.860GBfree disk, admission passes. No task/peer artifacts deleted or signals sent.

Original immutable cpuQ root:
`/home/ubuntu/selmappo2_20261101/build_b42c257b09813c175fcb7261e341c7599d0bb5c4c971529e2856890d2fc8038c`.
Supervisor428649 exited;900reserved/300actual, no controlled windows, no production,
no gradients; checkpoint000only. First smoke failed missing inherited execution
module; failed seed18107998 must NEVER replay. Original unused smoke18107999 and
all7774production seeds remain untouched. Retain original failed ledger/source/
logs/locks. Main registry contains explicit same-cap acknowledgement REQUEST;
no relaunch authorized/armed by that request.

Local repair intercepts CTRL0 in mixed CavContext, preserving the original frozen
forced-base residual contract at both halfsteps through shared Backend.tick.
It also fixes handover clocks and atomic terminal group cleanup. Shared modules
are imported/staged, never copied into this branch. Runtime regression compiles
the exact SHA-checked frozen control-tick body; fake transport means0SUMO.
86local CPU tests pass with `C:/Users/jpeng/miniconda3/python.exe`.
Repaired CPU-only stage `build_b299a4b31582d7bfc57b9486dbdf89287044be79a66d045fbd3bd6a0e3fe2e0d`
also passes86/86 on cpuQ, zero skips, plus real frozen checkpoint/WALL/guards/PPO
integration;0newSUMO, no continuation dispatch authorization. Do not mistake the
INTEGRATION.json for acknowledgement of a new training attempt.

## Resume verification after MAIN01:48

Continuation implementation88078774 passes93/93 CPU locally and on cpuQ, zero
skips, plus real frozen checkpoints/WALL/exact guard aliases/shared PPO gate.
New immutable continuation-ready stage:
`build_c141d5715f26af4e9b50d0866cdf971ef92d55dd1223c4312c4f25bdec5a8b29`.
Registration snapshotSHA2c3fa57d03f956bac2626378885429d835f65fce205e05a0700d55a159bccf7e
contains exact unchanged cap and MAIN01:48 acknowledgement. Reviewed/shared
registration_history.py is now included explicitly in the source inventory;
its exemptions apply only to fleet disclosures, not per-CAV worlds.
cpuQ census199529safe filenames/zero conflicts/errors; original failed path only
is excluded after original PID/UID/ledger/lock validation. Pre-stage disk2.608GB,
load6.62/32cores; dispatch/each episode rechecks8GiB memory/2GiB disk and spare
nice19 CPUs. Stage is CPU-only; new simulation awaits explicit `--action continue`
with this exact build. Never use original `--action dispatch` or remove any lock.

MAIN's registry line `01:48 MAIN ACKNOWLEDGEMENTS / DECISIONS` and the fresh
session addendum explicitly acknowledge unused second smoke only, then production.
Read-only collection `COLLECTION_b42c257b09813c175fcb7261e341c7599d0bb5c4c971529e2856890d2fc8038c_1793468957194697000_9e2e08b6f97e49469002bdb211625332.json`
again verifies original428649 exited,900reserved/300actual,0updates. Parent metadata
contains only the original dispatch lock and four known stages, no continuation
lock/build/job. cpuQ has32cores/load2.105 at inspection. One-use continuation
implementation is the next step; the old lock remains intact.

## Before a continuation

Read MAIN's response in shared registration. Original no-retry protocol requires
an explicit amendment to accept one successful mechanics world after this failure.
Same-cap proposal: retain failed900, run only second900, unchanged production
6,996,600, leave14,400evaluation unstarted for peer-owned evaluation. No additional
seed/reserve/refund/retry. Bind exited original PID and exact ledger SHA
`eb53747c54b045b8a4c3447969d14ac4fc6f24e3719cdc5b18aff31d358417a9`.
Build an explicit one-use continuation entrypoint; current trainer/dispatch lock
intentionally refuses a restart. Never remove/overwrite that lock or old data.
Same inherited charge must appear in global actual/reserved accounting and report.
Do not reuse reporter's strict original-success-prefix checks for an amended run
without teaching it the exact hash-bound historical failure.

CPU repair stages do not authorize simulation. Observe MAIN01:08 priority:
cooperation/WM-PHYS, then C1SWITCH2, then spare nice19 RL training. No peer signals.

## Collection / reporting

Read-only collection (SSH intervals>=60s):
`python -B -m tools.selmappo2.report --action collect`
defaults to the original known build/PID. Offline status/ledger report:
`python -B -m tools.selmappo2.report --action report --input <COLLECTION.json>`.
Collections, CSV/SVG curve assets are append-only; no curve from0updates.
Current collection ends in `...1793467168386196000_5bc9500c19944cfea658dcdc2dcd958e.json`.

Only final169 eligible; hand checkpoint+FROZEN_FINAL+exact source/config SHA to
S-C1SWITCH2. Its panel969136..969151/PANEL_v2SHA7c780c25237dde73a657937ab577b8e48f71766817f4363f3298982e90144827
owns all evaluation physics incl J900 tail; no duplicate local evaluation.
