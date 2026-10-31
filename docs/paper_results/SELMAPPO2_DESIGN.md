# S-SELMAPPO2 — per-CAV MAPPO controller selection

Prospectively registered 2026-11-01 under OWNER23:42 advance approval and
OWNER/MAIN23:46 six-controller, 20-second re-scope. Fleet PPO is S-SELMAPPO.

## Protocol and information boundary

- Reuse S-SELMAPPO's environment, observation builder, categorical PPO and frozen
  library backend by import from a hash-bound shared source tree; no duplicate
  environment or sensor features. Actor parameters are shared across CAVs.
- Decisions at t=300,320,...,880 (30 windows), native warmup300s and control600s.
  Six categorical actions: native, CTRL0, speed-MAPPO169, PI, CIRCLES, WALL.
- Only observable CAV self telemetry, detected neighbours, anonymous RSU receipts,
  coverage/age and genuinely sensed entrance counters; absent counters stay masked.
  IDs route commands, never numerical features. No truth or future-world inputs.
- Training-only central critic receives the shared observable summary. Each CAV
  receives the team advantage; one critic loss per window including empty rosters.
- Reward is negative road speed-deficit vehicle-seconds plus source-wait
  vehicle-seconds over each20s window. Gamma1/lambda1 gives complete remaining
  600s team-delay return, terminal value0; no shaping or learned early stopping.
- Entrants/re-entrants start native until the next grid decision; departures
  retire memory. Changed CAVs release their own writer and reset peer-controller
  memory; unchanged choices preserve it. No extra physics step at hand-over.

## WALL membership and frozen law

WALL means consent to join the single episode wall group. At each grid decision,
the eligible roster is exactly the currently observable CAVs choosing WALL.
One shared frozen C4 WallLaw object selects its closest eligible cross-lane pair
from that roster, using the unchanged v2.1 closure/v2.2 merge-release law in
S-WALLSW HANDOFF.md. No independent wall objects are created per CAV. Nonmember
consenting CAVs use frozen CIRCLES fallback; all other CAVs execute their own
chosen library action. A pair member withdrawing, departing or re-entering ends
the group atomically before another group can form. An unchanged roster preserves
assembly/HOLD/cooldown memory. Newly consenting CAVs join the candidate roster;
they never replace active members silently. The existing WallLaw eligibility,
pair selection, bounds, cooldown and safety constants are inherited verbatim;
no new behavioural cutoffs are introduced. Native lanes and original safety
ownership remain in force, with exactly one command writer per CAV per tick.

## Exact budget and fresh worlds

- TRAIN block18100000..18107999. Production18100000..18107773:169 updates x46
  fresh worlds x900s = **6,996,600 sim-s**, comprising4,664,400 controlled and
  2,332,200 warmup sim-s. Smoke18107998/18107999: **1,800 sim-s**.
- Seeds18107774..18107997 remain unallocated. Every world uses the shared seeded
  per-iteration permutation cycling all12 penetration(.05,.1,.2,.4) x
  demand(1900,2400,2700) cells. Optimization RNG1; no acceleration-policy warmstart.
- Frozen final169 evaluation:16 x900 = **14,400 sim-s**, on the independently
  registered S-C1SWITCH2 panel, pinned before access. Do not assume the older
  C1SWITCH panel is the new panel. No evaluation selection or normalization update.
- Exact cap **7,012,800 sim-s**, including every warmup, smoke and failure;
  **zero retry/top-up reserve**. No refunds, seed replacement, sweeps or restarts.
  Partial failures retain actual and reserved charges and stop dispatch.
- Registry interval audit and pruned filename/path exposure audit precede SUMO;
  task-scoped guards require exact seed/cell/role. Never964000..964199/962xxx/977xxx.

## Optimization, execution and reporting

Reuse shared PPO: fresh128x128 tanh actor/critic; actor3e-4/critic1e-3,
clip.2/value-clip.2, epochs4, minibatch256frames, entropy.01, value.5,
grad-norm.5, targetKL.03 and existing ReturnScaler. Initial and every-update
checkpoint are immutable. Fixed169 updates, no reward-based checkpoint choice.

cpuQ/gpuR spare CPUs only, nice19, CUDA hidden, single-thread spawned workers;
one mechanics canary before admission to <=12workers, updates<=4threads.
Yield on host load/memory/disk admission, cooperative own-worker stop only.
24h wall limit, nonfinite/clock/slot/ownership/hash/ledger/safety faults stop.
Parent precharges each900s reservation before dispatch; no worker owns budget.

Report raw per-update team J600/return learning curves, compute and checkpoint
hashes in SELMAPPO2.md. Evaluate deterministic argmax final169 (lowest-index tie).
Compare paired J600 versus WM/BB/six fixed, world-bootstrap2000/seed0/95%CI;
best fixed reselected within each resample. Report richer per-CAV action space,
road/source components, switching/entrant exposure and safety. Require complete
hash-bound comparator receipts; missing results stay pending. Disclose single
training replicate and inherited library/WM training separately, no equal-budget
claim. Commit design, build/tests, registration, dispatch and result steps.
