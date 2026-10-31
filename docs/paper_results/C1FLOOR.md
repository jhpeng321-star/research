# S-C1FLOOR — BLOCKED at saved-state identity gate (2026-10-31)
**No RNG-only divergence floor or confidence interval was measured.**
Owner approval: active C4 registration S-C1FLOOR, 2026-10-31; branch `c4/c1floor`.
The original WMDATA native run, seed969400, t300→600, reproduces its recorded continuation bitwise.
All six tape arrays t/ID/position/speed/acceleration/lane match:45,631 observed rows through t600.
The original-seed saved-state/RNG continuation does NOT reproduce that live continuation.
Its tape contains45,227 rows; drift already appears at t300.5, before any alternative seed is tried.
Two initial HDVs (observable registry IDs101,104) differ at that first half-second; neither is a CAV.
For ID101, recorded/restored position29.3600769/29.1990948m, speed1.1483253/0.8263615m/s, acceleration0.6439276/0.
For ID104, recorded/restored position21.8051739/21.7588921m, speed0.5324166/0.4398550m/s, acceleration0.1851233/0.
Thus a saved-state-vs-recorded comparison would confound RNG divergence with restoration drift; the floor is unavailable.
No alternative-seed branch, production world, fitting, or GPU inference was launched. Durable task STOP is installed.
## Prospective protocol and safeguards
`FLOOR_DESIGN.md` retains the original24-world plan and every subsequent pre-outcome mechanics/budget amendment.
Final prospective subset was first18 C1CLOSED WMDATA worlds969400–969417, native arm, t300, two RNG initializers/world969050–969085.
Only canary world969400 was simulated; panel reductions followed retained mechanical reservations, never RNG-error outcomes.
Frozen E3 source543/543 SHA-verified, manifest `b3c031ad`; WMDATA observer `b22af462`; SUMO1.27.1.
COOPMEAS2 save/snapshot/fresh-process restore reused, precision17+complete RNG; original-state file used for the failed repeat.
Both frozen seed guards passed all54 final planned cases; exact registration/range and12,794 safe-filename checks found no alternative-seed exposure.
`tools/c1floor/evidence/EXPOSURE_NOTE.md` discloses an early content-search scope anomaly; its sibling-file numeric matches were unused.
Assignment/truth reads are blocked; the original builder may only create its own generated definition/truth outputs, which were not read.
## Accounting and preservation
Six attempts conservatively reserved1,500sim-s each:9,000charged; actual0+0+0+600+900+900=2,400sim-s including all prefixes.
First three setup failures used0; subsequent adapter/clock failures used600/900; final bitwise gate failure used900.
All failed bytes and reservations retained. Actual use is below30,000; no outcome retries or further dispatch are armed.
One nice19 single-thread gpuR CPU worker maximum, CUDA hidden, quota/load admission; SUMOCF/COOPMEASURE priority preserved.
All12 dispatched own coordinator/worker PIDs verified exited; no peer process changed, no files deleted, no push.
Raw canaries remain gpuR `/root/c1floor20261031/jobs`; final failure `s969400_native_attempt6`; task-level `BLOCKED.json` binds hashes.
Local immutable receipts/accounting/logs: `tools/c1floor/evidence/evidence_1793437482207139800/`.
## Overlay contract and next dependency
`C1FLOOR_error_horizon.csv` uses the exact14 C1CLOSED bootstrap columns (including units) at10/30/60/90/120/180/300s.
All estimates/intervals are blank, n_worlds=0, status=`blocked_saved_state_canary`: unavailable, never zero.
Scorers reuse C1CLOSED forecast/endpoint/errors/CSV functions; its bootstrap changes only panel dimension40→18. No alternate metric was invented.
Ten mechanics CPU tests, three scientific-contract tests and a zero-SUMO full setup probe pass; the real bitwise canary FAILS.
Required before resumption: a genuinely bitwise-compatible saved-state/reseeding mechanism on this recorded start, then a new prospective binding/precharge.
Scope remains native only; do not substitute this missing floor for PI/MAPPO curves or claim a proven irreducible prediction lower bound.
