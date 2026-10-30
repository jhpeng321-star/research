# S-INTERIM-A registration text — append before simulation

Owner approval: S-GATEMEAS, 2026-10-30 14:00; 14,400 sim-s already approved.
The exact eight fresh identities and fixed cell allocation are confined to
`tools/candA/interim_a.py`. No exposure audit or simulation has been run here.

After an identity-only exposure audit, generate the complete append block:

```sh
python tools/candA/interim_a.py CHECKPOINT.pt --exposure-audit EXPOSURE_AUDIT.json
```

Append its stdout verbatim to the run registration BEFORE executing:

```sh
python tools/candA/interim_a.py CHECKPOINT.pt --exposure-audit EXPOSURE_AUDIT.json --registration REGISTRATION.md --driver LEARNED_DRIVER.json --parity-receipt PARITY.json --output NEW_OUTPUT --run
```

The audit JSON contract is `schema: interim_exposure_audit/1`, `passed: true`,
`fresh: true`, `excluded_from_training_and_selection: true`, `seeds` exactly
the script's fixed block, `plan_sha256` equal to `interim_a.plan_sha()`, and
nonempty `surfaces` with `path`, `sha256`, `identity_only: true` for every audited
identity registry. This script validates the receipt; it cannot establish
freshness from an absence search. No outcome files are opened by preparation.

Fixed comparison: ours vs CTRL0, 300 s native warmup + 600 s intervention,
two worlds per registered focus cell, same executor/native lanes/entrants.
Primary result: full-fleet-plus-source raw paired J difference, equal-world mean,
all pairs and a paired-world bootstrap CI; safety/drop receipts accompany it.
No automatic retry, replacement seed, checkpoint selection, training or tuning.
