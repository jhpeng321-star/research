# F2: feasible, family-balanced move proposals

Registration: **`f2-feasible-balanced-v1`**, from the owner’s MAIN 15:15
2026-11-05 item F2. This is a code-bound continuation rule, applied identically
to B and R; it is not fitted to outcome receipts.

- At a decision tick, expire old commitments, then process acting CAVs in
  stable vehicle-id order. Acquire each selected move before constructing the
  next CAV's slate. Preserve the original context order in the recorded batch.
- A member must be in the ego's observable sensing/V2V CAV list, within the
  registered radio range, presently controlled, and unowned. The ego must also
  be unowned. Reserve **all** members, including gap_open's non-actuating
  partner. An occupied ego receives only `none` until commitment expiry.
- The training/practice episode hook and both diagnostic observation hooks
  (`launch_production.run_branch` for lessons/probes/replicated tests, including
  `MixedPolicy`, and `launch_evaluation.episode` for paired evaluation) refresh
  controller ownership on every tick
  from `Controller.active_plans`, including the real `VerifyController`'s
  restored practice/snapshot plans and candidate plans. Only each ego's visible
  CAV IDs and original `start_s + duration_s` expiry enter its reservation map;
  plan commands and nonlocal members do not enter the trigger inputs. A plan
  remains reserved for `now < expiry` and is available at exact expiry. This
  does not mutate the controller ledger or extend commitments. Executor
  ownership is unioned with this map before **each** serial slate.
- Discover feasible families before assigning slots. Reuse the registered
  library geometry, onset/hold/depth windows, and wall arrival/release rule.
  Enumerate the bounded observable wall member sets so nearby same-lane CAVs
  cannot hide an available cross-lane partner.
- Allocate one slot per feasible non-none family per cycle, randomly permuting
  family order each cycle. Counts differ by at most one. Unavailable families
  receive no slots; no joint option is manufactured without feasible geometry.
  `none` remains the separate baseline slot. Any uniform random teacher uses
  this same slate through `track2_backend.MoveAPI.candidates`; there is no
  arm-dependent distribution. The current launch has dense teachers disabled.
- PPO evaluates the recorded conditional slate, mask and old log probabilities.
  It never redraws proposals during replay. The A episode hook returns the
  original policy and controller.

## Receipts and API

`feasible_proposals.candidates(obs_state, rng, n)` returns library `Move`s.
`obs_state` contains local `self` and `rows`, plus optional locally filtered
`acquirable_cav_ids` and a `proposal_stats` output dictionary. Offline bank
slates without commitments default to their supplied observable CAV rows.

`decide_and_start(...)` returns `(TriggerBatch, accepted_by_ego, stats)`.
Its optional `command_reservations` input maps ego to local CAV ID to original
controller expiry. `EpisodePolicy.observe(..., command_commitments=...)`
projects the authoritative controller ledger into that local map. The default
empty input supports offline/test policies without a controller ledger.
Every live decision stores `stats` at `sample.track2_trigger.proposals`.
`aggregate(receipts)` computes update-level counters/rates, stored by
`launch_production.update_metrics` at `metrics.move_proposals` in the arm's
per-update learning-monitor JSON:

- `infeasible_rate`: rejected non-none selections / all non-none selections.
  Expected to be zero under serial acquisition; `null` if no move was selected.
- `family_infeasible_rate`: unavailable family checks / all family checks.
  This measures pre-selection geometry/ownership exclusions, **not** rejected
  actions. Four checks per acting CAV with a nonzero proposal budget.
- Raw denominators, decision count and `proposed_by_family` accompany the rates.
  Update rates are ratios of summed counts, not averages of decision rates.
  Legacy episodes without F2 receipts are unmeasured (`null`), not fake zeros.
- `owned_cav_checks` counts occupied locally observable, currently controlled
  CAV slots across egos, including earlier same-tick executor acquisitions.
  `controller_owned_cav_checks` counts the controller-owned subset. These are
  slot counts, not unique-vehicle counts. Their update sums are `null` if any
  contributing receipt predates the ownership bridge, rather than fabricated
  zeros. Thus a zero post-selection rejection rate still exposes exclusions.

CPU tests use synthetic observable rows and real CPU model/controller paths.
No simulation or remote access is needed.

The fix-round regression uses production `OPTIMIZATION`, the actual V2V-wrapped
practice `VerifyController`, snapshot/CRN restoration and the episode hook for
both B and R. It checks owned-ego `none` slates, owned-partner exclusion from a
free ego, local-only reservation projection, original commands and expiry,
exact-expiry acquisition and exclusion receipts. The compatibility suite loads
historical V2V producer/pin dependencies alongside the historical rerun module;
its absent-component check covers opt-in-only modules. `practice_weighting` is
a current shared legacy practice-config validator even when weighting is off,
and is therefore not an absent-module requirement. Absent/false Track 2 config
still must match the historical runtime config exactly.

The second fix-round regression calls the actual lesson, isolated-all
`MixedPolicy`, and paired-evaluation observation hooks and production policy
loader for A/B/R, replacing only outer physics/grading. It checks t0 cached
replay and CRN restoration, a free ego's owned-partner exclusion, owned-ego
`none` slates, newly added live commitments, local-only projection, unchanged
authoritative commands/expiry, exact-expiry availability, serial same-tick
acquisition, truthful exclusion counters, and full diagnostic window contracts.
All six B/R cases failed on the missing reservation map before the hook fix;
the three A cases passed before and after it.
