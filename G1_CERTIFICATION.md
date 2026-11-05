# G1 — certify executable joint teaching labels

Owner scope: `tracks_spec.md`, MAIN 23:15, G1; research report rank 1.
Implementation: `tools/coopmwm/executable_labels.py`. All operations are CPU
arithmetic, JSON contracts and existing F3 restart validation. There is no job
runner or simulation dispatch in this module.

## Contract and APIs

1. `restart_binding(snapshot)` derives the snapshot, exact recorded prefix/sample,
   and base-policy hashes from an already verified F3 restart.
2. `bind_lesson(...)` binds an ordinary TRAIN B/R source receipt, example, native
   plan, restart binding, runtime/source hash, exact `moves.Move`, decision time,
   original receipt window, and proposer-local visibility/availability/slate.
   `local` contains `self`, `observed_at_s`, `rows` (observable library row keys),
   `acquirable_cav_ids`, and `offered_moves` (serialized exact moves).
   Observations must be t-1; every member must be an in-range observable free CAV.
   The exact family geometry, ordering and wall arrival/expiry must be feasible.
   Roles reflect the existing library: gap_open has ONE actuator, wall and
   ramp_assist act on all members. No new role-specific action is introduced.
3. `job_spec(lesson, seeds)` creates an immutable prospective verification
   contract. Supply exactly two distinct `{verification_seed, snapshot_sha256,
   rng_sha256}` restart bindings, including distinct actual RNG hashes. For EACH
   seed it requests group, common base, and EVERY executable member's solo.
   It includes non-actuating context members' solos; partial solo sets never
   certify synergy. All branches retain the same macro and vary only which
   actors receive its offsets. No member is removed from traffic.
4. `admit(job, results)` recomputes the statistics, safety and physical effect;
   returns an immutable admission or diagnostic-only receipt, including reasons.
5. `budget_quote(jobs=None)` returns the bounded prospective quote. A concrete
   batch must match B/R context counts, family/arity, verification seeds and
   full-prefix simulation lengths; at most eight distinct contexts per arm.
6. `prioritize(lessons)` prospectively prefers two actual actors, two-member
   native profiles and exact native/library fidelity before immutable ID order.
   Outcomes never determine this ordering. There is no automatic fill from
   diagnostic evaluation states.
7. `manifest(entries)` accepts `{job, results}` entries, retains ALL negatives,
   zero controls and missing results, and computes admissions plus batch quote.
   `read_manifest(path)` guards the input path, checks its hash, reconstructs
   jobs, and recomputes every admission. A saved `admitted: true` is insufficient.

The default projection is only a proposed label. Pass the exact remaining-window
target explicitly when certifying a later F3 decision. Offset-fidelity error is
computed over the entire native profile and zero padding through macro release;
small error or native admission does not certify executable synergy.

## Full-window execution/result requirements

The job's `engine` is a required execution contract for a later authorized
adapter. It is intentionally not a legacy native-plan SUMO specification:

- Preserve the captured prefix and t0 action; start the macro strictly after t0.
- All non-selected members/background vehicles follow the same base checkpoint.
  Residual corrections stay on the common base in every branch. Disable later
  macro reselection so the measured intervention is this exact fixed macro.
- Apply full library strength through the existing single-writer ownership,
  release, feasibility and final safe-command path; HDVs are never commanded.
- Use matched member-keyed/counter-based CRN, including after membership diverges.
  No mean-noise fallback or partial alignment is certifiable.
- Use the ORIGINAL group receipt's complete registered window in every branch.
  Do not intersect shorter old solo windows or select a favorable tail. Every
  branch is quoted/charged from simulation time zero through the window end.

Each result is sealed with `result_sha256` and contains:

```python
{
    'branch_sha256': spec['branch_sha256'],
    'status': 'graded',
    'window': spec['window'],
    'prefix_matched': True,
    'crn_aligned_through_s': spec['window']['end_s'],
    'actual_sim_s': spec['sim_s'],
    'delay_increments': [...],  # one finite source-inclusive team_delay_vs increment/second
    'safety': {'collisions': 0, 'teleports': 0, 'hard_brake': 0,
               'ttc_below_1p5_events': 0},
    'library_actuation_audit': [...]  # required on group branches
}
```

Group audit rows carry `time_s`, `vid`, the executor's `observable_state`,
`library_move_offset_mps2`, `requested_raw`, `without_library_raw`, `issued_safe`,
`without_library_issued_safe`, and `library_issued_delta_mps2`. For walls, include
the locally observed release positions (`observable_state.s_m`), as the existing
production executor does. The admission step cross-checks actual raw and safe
command deltas against library semantics and requires at least two actual actors
with simultaneous nonzero issued effects exceeding the production 1e-6 m/s²
threshold, separately on each seed. Membership, requested offsets, asynchronous
single-actor changes or completely clipped macros do not establish execution.

Safety uses the existing MAIN04:26 rule: group-minus-base collisions and
teleports must not increase, and the sum of hard-braking and TTC<1.5 event deltas
must not increase. Additionally every branch must have zero collisions/teleports.
All counters must be recorded, finite nonnegative integers; missing counters do
not mean zero.

## Statistics and admission

For lower-is-better source-inclusive delay J:

`total benefit = J_base - J_group`

`synergy = min(J_solo_i over ALL members) - J_group`

The implementation reports these separately. It uses the existing bank's
2,000 bootstrap draws, RNG seed 20261103 and 30-second circular time blocks,
paired across ALL branches. The best solo is reselected from FULL-window totals
inside every draw; it is not a per-tick minimum or a preselected optimistic solo.
Seeds are resampled independently within their recorded windows. The pooled
quantity is the equal-seed mean. Admission requires the pooled synergy CI95 low
AND each seed's synergy CI95 low strictly greater than zero, plus safety and
physical joint execution on both seeds. CI95 is explicitly two-sided percentile
(.025/.975), conservative relative to the historical one-sided bank bound.
This is within-context block uncertainty, not an across-world generalization CI.

## F3 integration (prospective opt-in boundary)

Set the SAME common configuration keys for B and R:

```python
executable_label_manifest = '/absolute/staged/path/G1_MANIFEST.json'
executable_label_runtime_sha256 = '<immutable hash of the certified execution source/config>'
```

`bank_breadth.visit` applies the gate after existing native admission, effective
targets and augmentation checks, and revalidates the original/own-seed restart.
Only certificates matching source receipt, example, plan, snapshot, prefix,
base-policy and configured runtime hash reach either the F3 practice or trigger
consumer. Imported PLAN_FIX banks go through the same gate. The manifest path
should be absolute so additional bank roots and resumed plans use the same file.

Trigger supervision uses only the certificate's exact proposer, decision time,
observable local rows and macro. The observation timestamp is explicitly
`decision_s - 1`: replay retains the real observer at that timestamp and compares
its actual local rows (including positions and speeds) with the certificate.
Teacher sensor features are rebuilt with the existing observable encoder from
those validated local rows; local-only partner features come from that same
observer view. Recorded current-time `pair.context.x` is not reused. Candidate
availability uses the certificate's acquirable ids, restricted to members present
as CAVs in both the decision frame and the acting pair; visible reserved CAVs
are not made available. The decision-time frame never replaces or relabels the
certified geometry. Missing/stale t-1 evidence, mismatched local rows, absent
actors and invalid availability fail closed. Yielded teacher contexts record
both `observed_at_s` and `decision_s`. It does not certify an onset-adjusted later
target or an alternative proposer by association. Practice uses the certified
macro's family; frozen lookups bind the lesson hash and re-read/recompute the
certificate at resume. B/R supported-family matching remains enforced by F3.
Negative/missing certificates are diagnostic-only. A bypasses all bank reads.
With the option absent, existing bank/practice/trigger behavior is retained.
Native continuous AWR is still governed by its existing reader; this G1 gate
certifies F3 executable trigger lessons and their eligible practice contexts.

## Budget, commands and handoff

`G1_BUDGET.json` records the full prospective ceiling: 8 contexts/arm × 2 arms ×
2 seeds × (group + base + 2 solos) × 1,200 = **153,600 sim-s** for preferred
two-actor lessons; up to four member solos costs **230,400 sim-s**. Full prefix
replay is included, no branch-reuse discount is assumed, and execution is zero.

```text
C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m tools.coopmwm.executable_labels quote
C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m tools.coopmwm.executable_labels certify --input <entries.json>
```

Both commands print JSON. The latter consumes already collected ordinary TRAIN
receipts; it does not run jobs. A later authorized collector must implement and
attest the execution contract (including full CRN alignment and command audit),
publish a balanced batch manifest, and stage it at the registered update boundary.
No genuine executable labels or SUMO measurements are fabricated by CPU tests.

CPU verification uses the registered interpreter, `CUDA_VISIBLE_DEVICES=-1`,
and worktree-local pytest temporary directories. The G1 tests cover positive,
zero, negative and uncertain synergy, all-solo reselection, missing/misaligned
evidence, safety, clipping/physical simultaneity, visibility/roles/geometry,
budget fairness, manifest integrity, F3 gate, exact-time trigger targets and
frozen restart revalidation.

## Independent review r0 fix

`REVIEW_G1_r0.md` blocker 1 identified that a second observation at the decision
time overwrote the t-1 rows before certificate matching. Ordinary moving contexts
were therefore discarded and could leave B/R with no trigger teaching records.
The opt-in replay path now retains and reconstructs the causal view as described
above. The default-off replay path retains its original observation/feature flow.
CPU regressions use real `EpisodePolicy` / `IntentPolicy` observers with changing
positions AND speeds for both B and R, plus stationary controls. They check exact
time/proposer/macro and sensor/partner features, missing/stale/mismatched evidence,
decision-time actor presence, and reserved-visible candidate exclusions. Enabled
manifest-to-F3-to-trigger-optimizer tests demonstrate successful teaching with a
frozen base, and reject mismatched/stale replay evidence before any head update.
The moving replay and enabled B/R teaching regressions failed before the fix.
