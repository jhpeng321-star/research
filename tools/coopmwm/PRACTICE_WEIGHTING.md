# T2H practice weighting

The existing `practice_starts` path accepts two additive configuration keys:

- `practice_weighting`: boolean, absent/false uses the current path unchanged.
- `practice_weighted_fraction`: float, registered alternative `0.50`. Used only
  when weighting is enabled; the existing `practice_fraction=0.25` remains intact.

Enable through `v2v_runner.prepare_intent(..., practice_starts=True,
practice_weighting=True, practice_weighted_fraction=0.50)`. Explicit keys are
preserved; absent keys are not emitted. `v2v_launch3.practice_options_for` validates
them, requires practice starts ON, and enforces the existing B/R equality rule
including `practice_starts_by_arm`. A has no practice starts. T2C can use
`practice_weighting.effective_fraction(cfg)` for its own budget quote and smoke
episode count; it must not continue to quote the legacy fraction when enabled.

Weighting requests 12 of 24 episodes instead of 6. Eligible own-bank and
augmented restart states are uniformly sampled, with the existing per-example
repeat cap. Cooperative capacity is consumed before R's documented other-admitted
fallback. R still waits for and matches B's **realised** practice count, so bank
shortages or failed episodes do not silently increase R's request. An empty or
capacity-limited pool yields normal starts for the remainder. This is a target
share, not a guarantee that a bank contains enough distinct eligible examples.

## T2G producer contract

This base does not contain `plan_augment.py`. Its producer (or T2C adapter) can
write `replay_bank/<ARM>/PRACTICE_AUGMENTED.json`, sealed with
`exact_grade._seal(record, 'practice_augmented_sha256')`. The object contains:

- `schema`: `coopmwm-practice-augmented-v2` (v1 WM-only manifests refuse)
- `arm`: `B` or `R`, matching the containing own-bank directory
- `rows`: objects containing `example_id` (unique augmentation ID),
  `parent_example_id` (an own admitted cooperative bank example),
  `verification_id` (an own decision link), `state_id` (that link's snapshot ID),
  `seed_index` (the parent's seed index), `wm_cooperative: true`, and
  `snapshot_sha256` (the snapshot **after** the existing seed-index CRN derivation),
  and `sumo_evidence` with `receipt_path` (relative to the own bank),
  `receipt_sha256`, `plan_sha256`, and `charge_ids`.
- `prelaunch`: the parent integration's registered `phase`, `allocation_sim_s`
  (positive integer, at most 200000), and nonempty registration `provenance`.
  P1 also requires this object to equal `cfg.augmentation_prelaunch`.

The producer supplies only SUMO-verified, admitted cooperative children and applies
the same augmentation procedure to B and R's respective banks. It stages the
decision link under `decisions/` and the restart snapshot under `snapshots/` in
the same arm's bank. Snapshot and CRN semantics are unchanged; practice does not
execute a new augmented plan. Perturbing a plan without changing its restart
state does not manufacture practice capacity: identical `(state_id, seed_index,
CRN salt)` entries are deduplicated, including against the bank pool. Missing
manifest means bank-only sampling; a present invalid manifest refuses before
sealing a new practice plan or dispatching episodes. No arbitrary snapshot paths
or foreign-bank parents are accepted. A child needs its own admitted cooperative
per-seed bank example and sealed accepted SUMO receipt. Its selected branch tape,
chosen plan, derived snapshot hash, world, seed index and arm must agree. Each
receipt branch must resolve to its existing `<verification_id>_<branch>` charge
in the parent's prelaunch ledger phase, at the receipt's full branch cost. The
entire phase must fit its registered allocation. WM flags and parent admission
alone are insufficient. T2H does not execute verification jobs.

## Budget, receipts, and resume

The horizon stays 1200 seconds. Each practice episode replaces one normal
episode and receives one matched A-policy safety reference through the existing
batch, ledger, failure replacement, and safety paths. At full capacity the
reference upper bound is `2 arms * 24 updates * 12 * 1200 = 691200 sim-s`,
versus the legacy 345600. Runtime allocation, smoke/train child quotes, fresh
watcher quote, restart credits, and per-arm resume quotes use the effective share.
Existing budget-transfer validation still applies.

Weighted `PRACTICE_PLAN_<ARM>_<UUU>.json` gains a `weighting` receipt with the
effective fraction, augmentation manifest hash, augmented capacity and selected
counts. Weighted episode `practice.weighting` records source/parent provenance;
the per-update `practice.weighting` aggregates realised weighted/cooperative/
augmented counts. Empty banks still produce a weighted frozen-plan receipt.
Safety references retain the same provenance. Frozen plans are always reused,
even if config or the bank/manifest grows on restart; enable changes at a new
update boundary. Augmented snapshot hashes, child verification/admission and
prelaunch charges are checked again at replay time, including frozen old plans.

When weighting is absent/false, no augmentation manifest is read, RNG inputs and
sampling calls are unchanged, and plan/episode/update fields remain unchanged.
CPU tests compare default config and sealed plan bytes against shared-base
`13f184b5` with the timestamp held constant.
