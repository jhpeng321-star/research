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

CPU tests use synthetic observable rows and real CPU model/controller paths.
No simulation or remote access is needed.
