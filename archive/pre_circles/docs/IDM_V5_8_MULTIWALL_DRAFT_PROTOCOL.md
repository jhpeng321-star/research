# IDM v5.8 two-wall coordinated moving-bottleneck draft protocol

Status: **draft, nonexecuted treatment. No feasibility or outcome claim is made.**

Date: 2026-09-02

## 1. What this document is

This is a *draft protocol proposal* for a future two-wall (two-cohort)
coordinated moving-bottleneck treatment. The software counterpart is an
isolated, pure-Python state-management and command-composition skeleton
(`policies/multiwall_coordinator.py`) plus a draft contract
(`experiments/idm/wall_v5_8_multiwall_contract.py`). It is **preparatory
engineering only**. Running it changes nothing scientific: no SUMO branch has
been opened, no outcome has been generated, and no wall treatment has been
executed.

The eventual control arm remains **frozen decentralized IDM damping**. Every
CAV that is not an explicitly active wall member must retain the identical
already-computed damping action. No released, aborted, or removed member may
be silently handed to uncontrolled SUMO behavior; removed members return
immediately to the identical supplied damping action.

## 2. Purpose

The two-wall hypothesis under investigation: distribute inflow regulation
across **two mild walls** instead of one stronger wall, so that the authority
required from each wall stays low while the treated region remains protected.
Why so far it is only a hypothesis — the previous audit
(`IDM_V5_6_MULTIWALL_FEASIBILITY_AUDIT.md`) established only that the four
development cases contain enough in-band CAVs numerically (2/4, 8/3, 6/2 and
12/6 per-lane counts). It did **not** establish:

- two spatially separated full-lane cohorts;
- safe front/rear gaps for either cohort;
- minimum inter-wall spacing;
- storage capacity between the walls;
- independent release safety;
- sufficient formation runway for two sequential assemblies;
- noninteracting treatment effects.

None of these may be inferred from stored outcomes or guessed by the
skeleton. They are explicit unresolved configuration fields
(`min_inter_wall_spacing_m`, `storage_threshold_m`, per-wall authority,
combined authority budget, release order, minimum release separation,
candidate-selection rule, CAVs per lane, treatment duration, assembly/HOLD
zones, activation offset).

## 3. Fixed mechanical requirements (already implemented in the skeleton)

- At most two active cohorts (a third activation is rejected).
- A CAV may be a primary member of at most one cohort, a reserve of at most
  one pool, and never both; membership is globally disjoint.
- Reserve transfer is atomic (a reserve promoted to a member is removed from
  every reserve pool in the same state change).
- The action map starts as an exact copy of the supplied damping map; only
  explicitly active wall members may be replaced; nonmembers keep the
  identical supplied damping payload.
- Each cohort has an independent lifecycle (INACTIVE, ASSEMBLING, HOLDING,
  TAPERING, RELEASED, ABORTED) with its own timers, members/reserves by lane,
  requested duration, target speed, realized authority, lane coverage, center
  and span, speed spread, handoff count, abort/release reasons and events.
- Hard safety (externally computed; never invented here) has priority over
  HOLD duration, spacing objectives and taper preferences. A hard-safety
  failure aborts the affected cohort with an exact recorded reason and
  restores damping; the other cohort is unchanged unless a declared
  coordinated-abort interaction rule is configured.
- Diagnostics are deterministic and JSON-serializable.
- The draft contract refuses to construct an executable profile while the
  scientific fields above remain unresolved. An unresolved draft manifest is
  serializable but is never fingerprinted as a scientific treatment.

The skeleton expresses the future hypothesis that two mild walls share a
bounded authority budget (combined-request/realized accounting) but makes **no
claim** that this is safe or equivalent to one stronger wall.

## 4. Non-claims

Nothing in this document or in the skeleton asserts or implies:

- that a two-wall treatment is feasible in any stored case;
- that two walls are safe, beneficial, or noninteracting;
- that a split authority budget is equivalent to a stronger single wall;
- that any stored artifact contains the positions needed to form two cohorts.

## 5. Primary risks (to be audited, not assumed)

1. **Cohort scarcity:** four development cases may not contain two disjoint
   full-lane cohorts by the time the first wall must form.
2. **Inter-wall storage:** the space between wall spans may be below any
   capacity needed to avoid the second wall starving the first of traffic.
3. **Wall interaction:** the upstream wall's wave and the downstream wall's
   hold may reinforce or obscure each other's commanded patterns.
4. **Accumulated service loss:** two sequential holds can double the delay
   cost for the same treated queue.
5. **Unsafe overlapping releases:** release of one wall while the other is
   still holding or tapering can create compression at the merge.

## 6. Treatment-blind feasibility must precede any outcome execution

Before any two-wall outcome branch, a **treatment-blind audit** (see Gate M1
below) must establish, in at least **three approved development cases**, all
of the following at the declared activation time — and none may be inferred
from wall or outcome telemetry:

- two disjoint, simultaneous full-lane cohorts with trajectory-coherent
  members and sufficient formation runway for both assemblies;
- externally computed front- and rear-safe status for every candidate member;
- an inter-wall separation and storage headroom that the profile requires;
- independent release safety for both cohorts, with a recorded release order
  and separation;
- no candidate whose cohort membership overlaps another wall's ownership.

Fresh seeds, predictor-scale data generation, predictor fitting and RL
training remain **prohibited** in this phase and in the development screen.
The active v5.7.1 Gate A experiment, its contract, runner and output directory
are out of scope and must remain untouched.

## 7. Future gated sequence (not executed; no result is implied)

The only way to a two-wall outcome is through this treatment-blind gated
sequence. Every gate is binary and must be recorded machine-readably.

1. **Gate M0 — treatment-blind telemetry sufficiency.** The stored
   decision-time telemetry of the approved development cases must contain the
   per-CAV positions, lanes, speeds, gaps, rear/front leader ids and lifecycle
   runway needed to evaluate two cohorts. If the existing artifacts lack the
   positions needed to define feasibility, the multiwall program stops here.
2. **Gate M1 — two-cohort spatial and safety feasibility.** At least three
   approved development cases admit two disjoint, spatially separated,
   full-lane, front/rear-safe cohorts with runway, configured spacing and
   storage, and release interaction mechanics. One candidate profile may then
   be bound as a fully specified (fingerprinted) development profile.
3. **Gate M2 — synthetic/unit integrity.** The pure coordinator passes the
   focused unit suite (28 cases) and the bound profile is accepted by the
   v5.8 contract as executable-mechanics-only (never as a scientific claim).
4. **Gate M3 — damping replay and predecision identity.** In the real
   pipeline, the two-wall overlay must reproduce the damping arm exactly
   before the decision, override only declared active members, and return
   every removed/released/aborted member to the identical damping action.
5. **Gate M4 — four-case paired mechanism pilot.** Only after M0–M3: if M3
   passes, the unchanged profile may run matched damping versus two-wall
   branches on the four approved development cases with a dose, safety and
   service screen. No fresh seed is opened by M4 and no predictor data are
   generated.
6. **Only after Gate M4 passes:** comparison against the simpler single-wall
   treatment is authorized as the next step — and only then under an
   additional fresh-seed protocol. A failed M4 is a multiwall no-go, not a
   reason to retune or to open more seeds.

## 8. Integration discipline

The skeleton is deliberately isolated:

- It does not import TraCI, SUMO, Ray, RLlib, the active counterfactual policy
  or the active v5.7.1 contract.
- It does not compute damping, gaps, TTC or compactness.
- It cannot be executed in SUMO as-written; a policy adapter is required after
  Gate A finishes and after Gate M1–M3 pass.
- No scientific value (spacing, authority, duration, release order,
  candidate rule, CAVs per lane) is frozen anywhere in this skeleton or in its
  tests; the contract refuses an executable profile while any of them remain
  unresolved.
