# S-NATIVEINIT-1 — prepared only

Registered before implementation in the shared `REGISTRATION.md`. Branch `c4/nativeinit`
starts at `c4/mainstart` **186a1b09**. No SUMO, traffic-seed allocation, loop restart,
deployment or push occurred. Existing WAIT_OWNER and accounting remain authoritative.

## Native at initialization

`--native-init` enables `r=min(-a_min,a_max)*tanh(W*f+b)`, with **W=b=0**.
Only this residual head receives actor gradients; native WM/head/IDM weights,
random feature networks and group encoder are frozen. No MAPPO weights or external
acceleration noise are used. The residual is deterministic even when stochastic
actions are requested. Lane overrides abstain: there is no calibrated lane confidence
artifact, so both ramp and mainline lanes remain native.

Imagination recomputes the configured counterfactual native head/IDM every **0.5s**,
adds the **1s-held residual**, and applies the identical native safety/execution envelope.
All CAV lanes use the reference WM lane law. Hybrid merge RNG keys are shared per
start/slot across variants, including padded entrant slots. Row-local cost reductions
remove CUDA alignment-dependent rounding without an equality tolerance. Native command
and intent reports are explicitly unavailable/zero in both executors.

SUMO receives a typed residual, never an absolute-acceleration reinterpretation.
Every physics tick, `sumo_native_accel` queries SUMO `getFollowSpeed` using observed
leader speed/gap plus own minGap; no leader uses the infinite-gap limit. It caps the
result by `getAllowedSpeed`, converts speed to acceleration at0.5s, adds the residual
within the native-preserving action envelope and uses `setAcceleration(...,0.5)`.
**Zero residual adds no command beyond ordinary native ownership**, making update0
identical by construction.

**Mismatch after learning:** the WM native twin is not SUMO's internal controller;
`getFollowSpeed` is a car-following command, not junction/stop/lane-change finalization.
SUMO speedMode27 remains the final executor. Nonzero sim-to-real parity is unverified.

## Acceptance and training length

Frozen panel SHA `3eaa69a170c17302660d12dd4d7fddb4657e43fa334eece17844dd0fa6c0411e`:
16 TRAIN starts, paired critic-free **road+source J**, fixed shared randomness.
Require every finite `J(policy)-J(native)<=0`; retain the existing update10/horizon
gates. Acceptance is checkpoint/WM/config SHA-bound and checked before collection.
Missing/stale/failing receipts reject deployment, including the old catastrophic policy.

Round16,10s pilot: mean paired deltaJ at updates**0/10/20/30** was
**0/−0.033669/−0.067456/−0.097231veh-s**. Update0 is exactly equal on16/16;
all later checkpoints improve16/16. Recommend **30 updates next cycle**, selecting
the earliest minimum eligible measured panel mean. This is the measured pilot's
right edge, not evidence of convergence or a300s benefit;150 is unsupported.
Delivered-source rescoring reproduces update30 exactly. Receipts/scripts:
`tools/nativeinit/`; GPU allocator cap0.40, below12GB, own jobs completed.

## Exact next-cycle amendment — not installed

After explicit continuation authorization, integrate this branch on GPU and both
collectors. Set loop JSON `policy.native_init=true`, `policy.updates_per_cycle=30`;
retain the existing native-lane
config, frozen `--gate-panel`, WM-specific `--horizon-gate`, registered seeds and budgets.
Start a fresh residual actor; never resume the old absolute-action checkpoint. Retain
all superseded checkpoints, reservations and seed/charge histories.
Honor S-RESUMEPREP's separate mechanics-smoke reservation968142–968143:
while reserved/unconsumed, append `[[968142,968143]]` to the successor registration's
`exclude` list. Its two-world mechanics smoke does not replace the four-world guard.

Two-box collection dispatches/charges only the first **four** deterministic worlds,
then joins both boxes. Sum both paired starts per world; bootstrap whole-world
policy-minus-native J,5000 draws,seed20261026,95% CI,margin**0veh-s**. Lower bound>0
stops as harmful; upper bound>0 stops as inconclusive. Only upper bound<=0 releases
the remaining worlds. Existing incident/budget stops also apply. This is registered
preparation, not authorization for that future smoke.

Verification: **352 offline tests passed**, including three-update flag-OFF tensor,
optimizer, metric and RNG identity; strict/hybrid IDM/head rollout equality;
residual-only gradients; mocked actual environment composition/zero-native equality;
stale-gate rejection and smoke dispatch/charge barriers.
