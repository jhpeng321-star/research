# G2 — Observable coordination opportunities

Implemented against `tracks_spec.md` MAIN 23:15 item G2 and rank 2 of
`CLEAN2_GROUP_COOP_RESEARCH.md`. CPU engineering only; no physics dispatched.

## Opt-in and compatibility

The common B/R configuration can opt in with:

```json
{
  "coordination_opportunities": {
    "enabled": true,
    "rule": "g2-observable-disjoint-merge-pairs-v1",
    "provenance": "tracks_spec.md MAIN 23:15 owner item G2; CLEAN2_GROUP_COOP_RESEARCH rank 2"
  }
}
```

Absent or `false` preserves the existing proposal path, random draws, actions,
cadence, and receipt fields. At least two non-none candidate slots are required
when enabled. Settings have no arm-specific overrides. A keeps the original
policy/controller objects. Actor/head tensor shapes and optimizer parameters do
not change; existing sidecars load strictly.

This changes the proposal distribution and is an explicitly versioned successor
recipe, not an unchanged Clean2 continuation. No launch config or registration
was switched by this task. Both B and R use identical settings/code/sampling.

## Graph and role-aware ordering

`coordination_opportunities.graph(obs_states, observable_cavs, eligible,
occupied=(), command_reservations=None, now=...)` returns ordered `Opportunity`
edges. Each edge uses only the initiating mainline CAV's authorized local rows
and CAV partner slots. The controlled-member set and ownership ledgers restrict
actuator availability; they supply no traffic features. HDVs cannot be members.

An edge binds two roles: mainline yielder/gap-opener and ramp merger. Both must
be inside the existing registered merge zone, inside the registered association
link and sensing range, and free under the controller's original expiry and
the executor's commitments. The zone, lanes and distances come from the existing
move/variety registrations. There are no new distance or time thresholds.

Edges are ordered by the ramp's observable constant-speed time to the zone end,
then pair separation and stable IDs. Stopped ramps have infinite remaining time
unless already at the endpoint (zero). `disjoint(edges)` greedily chooses
nonoverlapping pairs. All selected mainline proposers decide before ordinary
ID-ordered egos, preventing an earlier singleton from taking their partners.
Pair arbitration itself consumes no random draws or outcomes.

Each prioritized proposer receives pair-specific two-CAV options: first
`ramp_assist`, then singleton `ease_off`, then `gap_open`; subsequent cycles are
randomly permuted and family counts differ by at most one. The trigger always
retains its separate `none` action. Joint onsets are capped to precede positive
constant-speed ramp remaining time; at the endpoint onset is zero. Registered
hold/depth windows, full global look-ahead, execution semantics, feasibility
clamps and final command ownership remain intact.

These are complementary *traffic roles* using the current action vocabulary.
`ramp_assist` still eases both actors in one shared window; `gap_open` still has
only one actuator and is not counted as physically joint. Per-role acceleration
profiles, teacher certification, and measured synergy credit are separate tasks.
Offered pairs are not certificates of positive synergy or physical effect.

Ownership is rechecked before every actual slate. Acquisition remains atomic.
An acquired follower gets none-only; if a proposer selects none or a singleton,
ordinary remaining egos can still act using their then-feasible slates.
Original context member order, recorded tables, masks, descriptors, actions,
old log probabilities and values are retained for PPO. Replay never rebuilds
the graph or regenerates a candidate slate.

## Approach event and churn bound

`EpisodePolicy` checks the local graph each control tick when enabled. A newly
available disjoint pair can create an extra decision when it enters the useful
merge geometry between ordinary periodic decisions. Each ordered role pair can
create at most one such event per episode; an actual periodic offer also marks
it seen. Disappearance/reappearance and commitment expiry do not repeatedly
retrigger the same pair. Pairs blocked before their first offer wait for free
ownership, including exact-expiry availability.

Events do not reset or shorten `decision_interval_s`. Only selected pair
participants can get non-none options on event ticks; unrelated rows are
none-only to keep full per-CAV reward/context alignment and avoid reselection
churn. New episodes create fresh event state. No simulation or credit horizon
is shortened and no extra episode is scheduled.

## Precision contract

G2 accepts the exact float64 registered zone endpoint and its exact float32
observable encoding as the same endpoint (notably 75.44000244140625 versus
75.44). It canonicalizes copies, never input rows. Even the next representable
float32 coordinate outside the zone is excluded. The opt-in path applies this
same contract to graph generation and ordinary feasible slates.

## Receipts and APIs

- `settings(cfg)` validates the common opt-in registration.
- `graph(...) -> tuple[Opportunity]`; `disjoint(edges) -> tuple[Opportunity]`.
- `Opportunity.to_dict()` records proposer, members, roles, separation and
  remaining time (JSON null for infinity).
- `ApproachEvents.pending(edges)` / `.record(pairs)` implement the event latch.
- `feasible_proposals.decide_and_start` adds optional `coordination`,
  `opportunity_pairs`, and `event` keywords; its return API is unchanged.
- Per-decision telemetry lives at
  `sample.track2_trigger.proposals.coordination_opportunities`, including rule,
  provenance, graph edge count, disjoint pairs, pair roles, actual acquisition
  order, event flag, decisions and offered-joint count.
- `feasible_proposals.aggregate` carries sum counts and the ratio-of-sums
  `offered_joint_rate` into existing
  `learning_monitor/<arm>/uNNN.json` at
  `metrics.move_proposals.coordination_opportunities`.
- `offered_joint` counts decision rows with at least one candidate having two
  or more actual library actors; it excludes one-actuator `gap_open` and
  includes existing walls offered on ordinary rows. The denominator is all
  trigger rows, including forced none. `event_decisions` counts extra decision
  calls, not CAV rows. Graph counts are sampled at decision calls, not summed
  over every control tick. Mixed enabled/legacy registration receipts refuse
  aggregation rather than fabricating missing observations.

Incremental simulation cost of this build: **0 sim-s**. Added graph/candidate
decisions incur CPU overhead inside existing collection windows. Positive
synergy and actual offered-rate changes need later authorized qualification;
no such outcome is inferred from synthetic tests.
