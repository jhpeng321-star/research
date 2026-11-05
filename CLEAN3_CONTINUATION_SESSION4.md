# S-LAUNCH recovery — 2026-11-06 01:25 +08

## New fault and preserved evidence

Recovered clean c4/rerun-launch `c79edc97`, read history/status, the full
tracks_spec and last300 registration lines, and adopted completed setup,
integration, qualification, smoke, gate, launch and diagnostics.

The existing full live audit correctly refused because STOP/FAULT now exist.
Read-only accepted-F6 inventory and targeted audit (commit **dbffaa78**) prove:

- Monitor301796 faulted at **01:06:45 +08**, reserving lesson-u18:
  `ValueError('whole-wave budget blocked')`. Chain296173 subsequently exited
  on the durable stop at01:07:12; all original A/B/R/P3 owners are exited.
- Actual sealed **A18/B18/R18**; all54 update seals match checkpoint hashes.
  Combined monitorsu1..u18 are seal-valid and config-bound, CONTINUE/no DIAGNOSE,
  with zero collisions/teleports. This is a phase-budget fault, not a safety-event
  finding. u24 test was not reached.
- Actual native B frozen lesson list:12 branches, **8760sim-s per wave**,
  same full original windows/prefixes. u6/u12 completed and charged17520.
  Legacy phase cap21840 leaves4320, insufficient foru18's8760. u18/u24 lessons
  are unperformed. Four waves require35040, an underquote of13200.
- STOP/FAULT SHA `3aedfd34d122a48c5149f5e92b11a301487dda00fe68630928707bede45e68c2`.
  Clean2 ledger1656rows/**1849200sim-s**, SHA
  `9c0a557a686b39c1095f736ed92db92c5aae4162bbf93ab791fb82bae936ff09`.
  Global actual clean1+clean2+parity **3098640sim-s**.

No stop cleared, process signalled, replacement launched, measured work repeated,
or deployed research source/configuration amended. Existing local follower9504's
u17/u18 and fault summaries are retained in CLEAN1_STATUS.md.

## Corrected prospective lesson quoting

Commit **48c54cd3** extracts the unchanged lesson-pair expansion used by actual
scheduled execution, adds its zero-physics quote, and requires those complete
native manifests in clean3 cumulative admission (schema v2). All scheduled
waves/full prefixes count, with original21840 minimum retained for smaller banks.
Missing waves, truncated windows, suffix-only cost, duplicate IDs, manipulated
sums and measured-as-dry receipts refuse. No phase transfer or guard bypass.

Verification: focused clean3/clean2 budget, native lesson, launch-clean2 and launch
integration suites **72passed/1existing Windows symlink skip** (37.85s).
The first test attempt's wrong synthetic proposer was corrected to the registered
`wm` fixture; no production assertion weakened. Final suites pass.

Commit **87df6d4e** adds a bounded read-only accepted-F6 quote runner and retains
`CLEAN3_QUOTE_STOPPED_SESSION4.json`. At **01:23:09 +08**, current operator
sources were compiled in memory in the qualified cpuV environment; the actual
bank readers produced all four8760 manifests. Current-policy loading, reservation,
branch dispatch and pool creation are forbidden in this qualifier. New source is
**operator-only, not deployed**. Original CONFIG/launch binding, exited owners,
STOP/FAULT and unchanged ledger were checked; no physics ran.

## Remaining gates

- G1/G2 accepted. G4_READY round1 production fix remains under the existing
  independent review (`autoreview_state.json` phase=review/round1), without an
  acceptance receipt. G5 is separately under review. No duplicate review/build.
- All required acceptances precede the new clean3 branch/merges and integrated
  Windows/Linux qualification. G1 native certification adapter, G smoke receipts
  and chain remain to be genuinely qualified before physics.
- Corrected bound: prior3098640 + protected unspent plan_fix566400 + prospective
  clean33254640 = **6919680**, **919680 above registered clean3-scope6M**.
- Separate mining ledger actually0charged; registered500000 remains protected.
  Overall prospective bound **7419680 vs6500000**, the same **919680 shortfall**.
  No reservation cancelled or shifted. Even omitting plan_fix's protected remainder
  would still leave353280 over budget; that omission is not authorized.
- Accepted final G adapters and smoke expansion can change costs; this is a
  stopped-ledger preparatory quote, not final admission or launch authorization.

MAIN notified via `_sol/CLEAN2_STOP_U018_LESSON_BUDGET.md` and updated budget
blocker/current pointer. Preserve stopped clean2 evidence and its exact stop;
do not mark the fault as an owner-stop release or claim successfulu24 completion.
MAIN/owner budget resolution and G4 acceptance are required for further launch
progress. No smoke/gate/chain dispatch is admitted by this handoff.
