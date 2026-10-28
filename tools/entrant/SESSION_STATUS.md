# S-ENTRANTFIX resume-safe status

## Code and verification

- Registration: `a0630635`; recovery: `6c7d821b`; rollout implementation: `e18c0697`.
- Thirteen focused tests PASS on cpuP, using an isolated `~/entrant/verify/`
  snapshot. Never overwrite the live numeric runtime to update tests/analysis.
- Final scoring source ON g01/k0 repeated identically (receipt on cpuQ).
- OFF full-horizon g01/k0 is exactly identical to original LRDIAG in every
  arm's primary120/300 delay, timeline, breakdown and speed/class bins.
- Reused original OFF trajectories after that proof:576 E3 +768 pairs300
  results across the two boxes. Primary120/300 values unchanged; supplementary
  WM horizons sum retained10s HDV bins with the same reduction for all variants;
  recorded targets recomputed with original delay_proxy and120/300 exact-checked.

## Owned inference — do not duplicate

- cpuP E3 coordinator PID2463892/start_ticks17369438, parity0,20 single-thread
  workers, launched1793159418.6605482.
- cpuQ E3 coordinator PID2441043/start_ticks17367079, parity1,20 single-thread
  workers, launched1793159393.0632024.
- Both queues are resumable and skip existing `~/entrant/results_v3` files.
  OFF units are already imported, so their queued jobs skip without inference.
- Only `entrant_on` and `entrant_on_ramp_off` need new factorial trajectories.
- Earlier canaries in `results`/`results_v2` are preserved, NOT final-source ON
  evidence. Current raw evidence is `results_v3`. No SUMO/training/push/GPU use.
- `status.py HOST` checks the recorded PID incarnation and task receipts only.
- `gather.py` incrementally downloads owned JSON. Use detached `local.py`
  operations for transfers/statistics so short foreground calls do not abort work.
- Background local operations are receipted under `evidence_entrant/operations`;
  inspect those before dispatching another operation with the same name.

## Remaining work

Finish factorial, aggregate whole-world CIs with the original owner-amended rule,
then score the best combination on pairs300 and run its registered60–300 gate.
Evaluate accept.py if an adoption candidate qualifies. Report no-regression checks,
deliver ENTRANT.md <=500 words, commit each finished step, no push/training.

The central register added owner semantics/near-term-observable-arrival amendments
(FOLLOWUPS43). CAV shares already use observable insertions, never penetration or
type truth. The inspected converted E3/pairs300 records have neither queue arrays
nor upstream near-term sensor snapshots. Keep a near-term extension separately
OFF unless observable inputs and prospective improvement evidence are available.
