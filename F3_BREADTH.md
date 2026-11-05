# F3 — verified practice and trigger-supervision breadth

Owner amendment: `tracks_spec.md`, MAIN 15:15, F3 (2026-11-05).

## Production behavior

The Track 2 production backend passes its configuration into
`practice_starts.plan_for_update(..., breadth_cfg=cfg)`. B and R now use the
same `bank_breadth.visit` reader as trigger supervision. This includes the
current bank's **all usable admitted cooperative per-seed examples**, imported
verified LS contexts from `replay_sources`, and optional PLAN_FIX bank outputs.
Legacy admitted cooperative receipts are also supported. Existing confidence,
safety, tape-integrity, observable-command availability and effective-target
checks still determine usability. No new simulation is dispatched by this code.

Practice loads each example's original decision snapshot through the native
own-bank or LS reader. Seed 2 and extra seeds retain the native replication salt;
the frozen plan binds the resulting snapshot hash. Only ordinary TRAIN snapshots
from the corresponding arm are eligible. Distinct snapshots/CRNs are deduplicated
within a family, rather than counting duplicate bank links as breadth.

Selection assigns equal mass to supported families, subject to the smaller
B/R family's verified capacity and the existing repeat cap. Both arms get the
same family sequence and counts. Within each family, every available context is
used before repetition in an update, and less-used contexts from earlier frozen
plans are selected first. All eligible contexts participate in this rotation;
episode count, horizon, repeat cap and practice fraction remain registered inputs.
R follows B's realized family sequence after failed practice slots are replaced
by normal episodes; it does not recompute different quotas from a smaller count.

Trigger supervision scans the retained observed frames rather than dropping an
example after inspecting only its first pair. Each visible acting member is taught
at its earliest representable frame, with remaining onset/hold and causal local
intent reports. Unobserved partners are never inserted into the policy input.
Contexts with less than the library's minimum remaining hold are ineligible.
Every usable context contributes to CE. Each family has equal total CE weight,
then its contexts share that weight equally; legacy AWR weights cannot concentrate
trigger supervision on a few examples. Continuous bank AWR retains its existing
implementation. Native-plan-to-library projection is still an approximation;
loading a native verified plan does not certify that its projection has the same
benefit. Re-verified library plans can supply exact representable labels through
the same bank reader.

Both practice and trigger supervision explicitly reject unequal B/R family
support rather than silently teaching different family distributions or inventing
unverified labels. The producer must provide verified guidance in both arms for
each new supported family. An empty PLAN_FIX output list uses the current banks;
two empty practice pools produce normal starts.

An existing frozen practice plan is never overwritten. F3 refuses a pre-F3 plan
for the same update; apply the amendment at a new update boundary. New frozen
plans resume exactly, even if a later handoff arrives; later updates read the
new handoff. A does not call the bank reader or practice selection.

## PLAN_FIX_READY output contract

Default handoff: `<run_root>/PLAN_FIX_READY.json`. A common configuration may
override `plan_fix_ready` with an explicit path. Missing file or `outputs: []`
means current banks only. Example:

```json
{
  "schema": "coopmwm-plan-fix-banks-v1",
  "outputs": [
    {
      "run_root": "verified_library_and_variety",
      "replay_sources": {
        "B": ["variety_labels/B"],
        "R": ["variety_labels/R"]
      }
    }
  ]
}
```

Paths are absolute or relative to the handoff directory. `run_root` contains the
ordinary `replay_bank/B` and `replay_bank/R` structures: verification receipts,
confirmations (including per-seed/all8 evidence), branch results and observable
tapes, decision links and snapshots. Optional `replay_sources` use the existing
LS label/pending/snapshot/branch format; their confirmations belong to the named
bank root. S-VARIETY and PLAN_FIX can each publish an output entry. The handoff
itself is **not verification evidence** and cannot supply standalone move labels.
All referenced inputs must be staged with their native reader dependencies before
deployment. `bank_breadth.inputs(cfg)` enumerates the configured bank roots/sources.

## Receipts and CPU verification

- `PRACTICE_PLAN_<arm>_<update>.json`: eligible context identities, family quotas,
  distinct selected count, source-input digest, and each selected snapshot hash.
- UPDATE practice block: actual context identities/counts, distinct realized
  contexts and family histogram, including failures/replacements as realized.
- Trigger CE result: example identities, frame times, observable acting members,
  family counts and normalized family mass, plus CE/KL optimization receipts.

Tests use the registered Python interpreter and CPU fakes. The F3 suite exercises
native per-seed admission/tape integrity, imported LS restart reconstruction,
PLAN_FIX expansion, seed restoration, broad rotation, B/R family matching,
visibility, actual trigger learning and frozen-base/A isolation.
