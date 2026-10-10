# Focused report-aware dissolution check — version 1

This is a new, separately budgeted development hypothesis following the completed
five exact observer replays. It is not an accepted controller or gate.

## Evidence and falsifiable question

At seed 4108 wall release (631 s), both members lack a detected front vehicle
within their own 100 m sensor range. Received self reports already locate slower
CAVs in the same lanes 165–178 m ahead. One travels about 3.24 m/s. Native handback
is followed by acceleration and then local detection/braking at about 635 s.
At seed 4107 release (624 s), the nearest reported CAVs ahead travel about
27 m/s at 208–226 m, and the slow reported queue in lane 0 is about 591 m away.

The narrow hypothesis is that a bounded wall-owned transition using the acquired
downstream reports can avoid the locally blind accelerate-then-brake response.
It does not assume that this transient explains all subsequent network harm.
Net benefit still requires the original full-horizon physical criteria.

## One fixed revision, no search

Keep the existing cohort, engagement, assembly and 20 s HOLD. Only when the
original controller would release because it requests native, test a common
speed envelope from already received, fresh downstream CAV self reports in the
same mapped lane. No global state, driver type, future inflow or online shadow.

For each downstream report, use the measured front-position separation d, a
conservative passenger-car length allowance 5 m, the existing 5 m minimum gap,
one control interval of ego travel, and the observed downstream speed v_d:

    available_distance = max(0, d - 5 - minimum_front_gap - v_ego * control_dt)
    speed_envelope = sqrt(v_d^2 + 2 * 1.5 * available_distance)

The 1.5 m/s² value is one fixed comfortable-braking design assumption, not a fitted
threshold. Its role is to price kinetic speed reduction over acquired distance;
the existing [-3, 1.5] command bounds and local front/rear guards remain in force.
This constant-speed downstream approximation is not a safety guarantee.
Five metres is the current passenger-car fixture's length; this assumption must
be checked on the preserved diagnostic rows and cannot transfer silently to trucks.

Activate the transition only if the envelope is below the frozen wall target.
Keep the common target at or below that target and preserve relative-position
coordination. Recompute from fresh reports each second. Release immediately if
all members regain numeric local damping references, acquired evidence disappears,
a guard fails, or the existing 10 s maximum dissolution budget ends. This is
explicit continued wall ownership; native selection occurs only at actual release.

## Separate physical budget and decision

Exactly two treatment trajectories: seeds 4107 and 4108, M2700, p20, engagement
600 s, complete 2400 s horizons. Maximum 4800 simulated seconds, zero retries,
no parameter alternatives or automatic follow-up. Reuse the already verified
immutable damping arms; do not repeat a known baseline. Before reuse, verify
source/model/scenario identity and exact predecision state/report/command prefixes.
Different proposal version metadata is bound in a derived comparison view, never
by rewriting old manifests or outcomes.

Require synthetic tests for the report envelope, stale/missing/future information,
native release, early numeric handback, timeout and existing local guards before
registration. Record every commanded and measured transition. Retain both cases
and all physical guardrails. A local transient improvement without full-horizon
benefit fails this check. Two exposed cases cannot validate population performance.
