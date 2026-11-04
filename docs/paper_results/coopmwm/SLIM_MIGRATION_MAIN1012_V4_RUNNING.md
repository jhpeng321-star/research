# MAIN10:12 rejected-tape migration — registered and running

**23 Linux tests PASS**, including the reviewed slim/partial-move tests and six
remote proof/eligibility/partial safeguards. Six local transport tests also pass.
Source was staged in ten small files (311,503bytes), not a large source upload.
No live training source was replaced.

The complete read-only real-bank dry-run admits **447 explicitly rejected
branches /894 marker+payload files /936,070,977bytes**, zero planning errors.
It excludes admitted examples, references, solos, first-seed chosen branches,
data-missing entries, observables and sealed records under the reviewed rule.

Registration SHA
`08a2f42e7c9060dd986d6d9669f79408bfbf0d6b084c2cf5bcba15b77fc6f714`
was appended to canonical REGISTRATION.md and remote AUTHORITY.md before dispatch.
Qualification/registration commit: `76088fbc`.

Actual migration was dispatched once at unix1793759355.57. Dispatch2600682;
control `/home/ubuntu/coopmwm20261104_slim_migration1/source4`.
Adopt `SLIM_MIGRATION_MAIN1012_V4_RUN_DISPATCH.json`; do not start a parallel move.
Transfers use the live mover's forwarded-agent transport to gpuA port11000.
No payload is staged locally. Each64MiB batch keeps a manifest and durable remote
file/payload/result-digest proof, followed by fresh seal/rejected-eligibility,
root FD and source identity/SHA checks before source unlink. SLIM stays locally;
full originals remain in gpuA `cpuq_archive_20261103/slim_migration1` at their
home-relative paths. Partial source removal retains every archive copy and stops.
The8GiB gpuA floor is checked before/after every batch.

At dry-run completion cpuQ free was7,198,322,688bytes (6.70GiB). Even recovering
the whole gross936,070,977-byte pool leaves at least2,603,024,575bytes short of
10GiB, before SLIM overhead or new writes. The scope is not broadened to reach
the target. Backfill's9GiB threshold and MAIN's collector STOP remain intact.
The previously registered legacy-tail move completed476,840,470bytes separately.

Superseded staging/test failures are retained as source1/2/3; source4 adds the
needed import/test glue and4078b574's report-only counters. Reviewed selection,
sealing and partial-move behavior from3a0e19c5 are unchanged.

No SUMO, inference, budget charge, training release or scientific change.
Observe: `python E:/agent-work/_sol/coopmwm_slim_migration_ops.py status`.
