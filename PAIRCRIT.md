# S-PAIRCRIT-1 — offline paired credit / C3 hazard prototype

**Delivered, both flags OFF. Breakdown deployment is NOT qualified.** Branch
`c4/paircrit`, parent `c4/restart@6e458c8b`. Registration preceded implementation;
native-rule fallback/constants were appended before policy event labels. No SUMO,
new traffic seeds, GPU allocation, live-loop mutation or gpuA writes. Computation
and models remain on gpuR at `/root/autodl-tmp/c4/paircrit20261026`.

## Implementation

`[paircrit].enabled` directly fits cost advantage `A=J_policy−J_native`; the
actor bootstraps **−A**, with native-row tail identically zero. Separate60/120/300s
outputs prevent fitting60s WM differences to a300s real target. Real300s pairs
require matching seeds, horizon, requested population, saved-start lineage,
vehicle IDs/initial numeric arrays, both export hashes and collection-policy SHA.
Development worlds never fit the critic or supply imagination starts. A changed
cohort requires positive boundary warmup; checkpoint/optimizer state is supported.
Real labels become explicitly off-policy after actor updates.

WM assistance reuses production CRN drivers, arrivals, lane/action noise and
no-grad rollouts; only differences enter regression. Checkpoint-bound measured
reliability and per-start variance-of-mean shrinkage weight the loss. Existing
R16 t0-HDV proxy evidence supports60s weight0.386339, **120/300s zero**. No absolute baseline:
there is no measured variance-reduction justification for adding a level target.
Both conversion paths export pair proofs; overlap binds both current-cohort arms
and waits for conversion.

`[breakdown].enabled` supplies native/policy cumulative probabilities at60/120/300s
to actor group features and critic features. Only RSU sensor aggregates enter;
hidden drivers/backlog are excluded. Real labels and K-sample paired WM event
frequencies have a supervised fitting API. WM event weights require their own
rule/WM-bound Brier-skill evidence: **currently all zero**, not borrowed from delay
reliability. Heads are temperature-calibrated on separate worlds, then frozen.
Deployment additionally requires a qualified, external **candidate-SHA-bound**
gate head; recalibrating it cannot change the actor checkpoint/hash.

The Gaussian native-regime rule refused identification. The preregistered fallback
maximizes native between-class variance in log1p merge speed: boundary
**5.590301496934885m/s**,11,953 fitting observations, rule SHA`8d44ea58…`.
First lower-class entry is an operational congestion proxy, not proof of physical
bistability. Initially low/empty starts are excluded from incident fitting;
already-low states report inclusive risk1; missing coverage is censored.

## Measurements and limitations

Fixed200-update fits:20 starts/10 worlds; development6 starts/3 worlds.
Direct paired RMSE **5,517veh-s**, R²**0.416**, Spearman**0.771**, harm sign6/6.
The current controller-unconditioned critic subtracts to zero on identical starts:
paired RMSE**23,558** on these six. Absolute RMSEs on the same six are
**11,399 current /6,517 horizon-matched refit**; these predict a different target,
so their RMSEs are not a like-for-like percentage improvement.

CRITAUDIT's full26-state result is reproduced: absolute RMSE16,615, R²0.192,
rank0.859. Its12 deterministic headroom repeats identify no stochastic noise
ceiling. The old critic fitted12 native600s labels, not these policy300s labels.
Rejected `fa242777…` is an offline example only, never adopted here.

Risk fitting excludes15/26 initially-low starts. One calibration world and two
test worlds leave **two eligible test outcomes/output**. Calibrated60s Brier is
native0.245/policy0.246, worse than uncalibrated0.054/0.091. At120/300s Brier0
is degenerate—all eligible outcomes are events. Event differences have zero
observed variation: **probability-difference predictability is unidentified**.
Qualification fails; this head cannot authorize deployment.

Evidence: `tools/paircrit/{MEASURED,RELIABILITY,NATIVE_RULE,BIT_IDENTITY}.json`,
[reliability diagram](tools/paircrit/reliability.svg), `tests.log` and reproduction
scripts. **44 offline tests pass**; flag-OFF matches parent exactly on337 tensors,
two updates each/plain and real-WM cooperative trainers, all three RNGs and
non-timing metrics. Bimodal synthetic futures reverse the absolute ordering;
CRN advantage plus the implemented risk-feature tail recover ordering and the
actor gradient, including0.002m/s-separated observable states.

## Exact proposed cycle-boundary amendment — NOT installed

At the next unstarted policy phase with an accepted policy's completed300s cohort
`C`, export its proof with `python -m wms.control.paircrit_export`. Bind
`state.real_returns=C.returns`, `state.native_real_returns=C.native_returns`, and
`state.pairing_receipt=NEW_PAIRING_JSON`; source-policy SHA must equal
`state.policy.sha256`. Set loop `policy.config` to a boundary-owned clone of
`configs/c4_policy14_paircrit.toml`, `policy.paircrit=true`, `policy.longcritic=false`,
`policy.breakdown=false`. In that TOML set `paircrit.enabled=true`,
`paircrit.warmup_updates=100`, `longcritic.enabled=false`, `real_share.enabled=false`,
`breakdown.enabled=false`. Preserve native vetoes, budgets and imagination≤60s.
Only after independent calibration qualifies may `breakdown.enabled` and loop
`policy.breakdown` become true, with qualified `breakdown.checkpoint` and separate
candidate-bound `breakdown.gate_checkpoint`. Full binding/qualification text is
`tools/paircrit/AMENDMENT.md`. Neither amendment nor any flag was installed here.
