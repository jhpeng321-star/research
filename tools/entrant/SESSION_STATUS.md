# S-ENTRANTFIX resume-safe status

## Code and verification

- Registration: `a0630635`; recovery: `6c7d821b`; rollout implementation: `e18c0697`.
- Seventeen focused tests PASS on cpuP, using an isolated `~/entrant/verify/`
  snapshot. Never overwrite the live numeric runtime to update tests/analysis.
- Additional gate-path audit found the attribution loader AND dataset builder
  force entries OFF. ON now restores both, passes births into rollout, and keys
  single-row lane/arrival/merge CRNs; dedicated loader/rollout tests pass. The
  public gate CLI exposes K samples, entrant specs and post-load ramp overrides.
- V3 ON g01/k0 repeated identically, but V3 ON science is superseded: recorded
  neighbours' discarded feedback lane intents and half-second command reports
  were corrected to executed intent / the ordinary one-second communication clock.
- OFF full-horizon g01/k0 is exactly identical to original LRDIAG in every
  arm's primary120/300 delay, timeline, breakdown and speed/class bins.
- Reused original OFF trajectories after that proof:576 E3 +768 pairs300
  results across the two boxes. Primary120/300 values unchanged; supplementary
  WM horizons sum retained10s HDV bins with the same reduction for all variants;
  recorded targets recomputed with original delay_proxy and120/300 exact-checked.

## Owned inference — inspect receipts before dispatch

- Superseded V3 cpuP coordinator PID2463892/start_ticks17369438 and cpuQ
  PID2441043/start_ticks17367079 were stopped by their verified PID incarnations
  and verified descendants (21 own PIDs each, none left running). Receipts:
  `~/entrant/SUPERSEDED_V3_STOP.json`. Every old result remains in place.
- V4 source is staged on both nodes. Check `queue_LAUNCH.json` / `OWNER.json`
  for new dispatches; queues skip existing `~/entrant/results_v4` files.
  All1344 OFF files were copied byte-identically from V3, so those jobs skip.
- Only `entrant_on` and `entrant_on_ramp_off` need new factorial trajectories.
- Earlier evidence in `results`/`results_v2`/`results_v3` is preserved, NOT final
  ON evidence. Current raw evidence is `results_v4`. No SUMO/training/push/GPU use.
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
